"""
stt_engine.py — Speech-to-Text engine usando Moonshine Voice v2.

ARQUITECTURA (v4 — Transcriber base + captura manual):
  MicTranscriber fue descartado porque hardcodea el device de audio
  internamente y no expone ningún parámetro para cambiarlo. En sistemas
  Linux con sof-hda-dsp (laptops Intel modernas), el stream que abre
  MicTranscriber no recibe samples válidos del VAD → ningún callback
  dispara nunca, cuelgue infinito.

  Solución: usar el Transcriber base de moonshine-voice (solo ASR, sin
  captura) y manejar el pipeline de audio manualmente:

    sounddevice.RawInputStream  →  buffer circular  →  VAD de energía
    →  segmento de audio  →  Transcriber.transcribe()  →  texto

  Esto nos da control total sobre:
    - Qué device de audio usar (pulse, hw, default)
    - Sample rate y canales (siempre 16000 Hz mono)
    - Parámetros del VAD (umbral de energía, silencio post-habla)

VAD de energía RMS:
  RMS = sqrt(mean(samples^2))
  Para int16: rango [0, 32767]. ENERGY_THRESHOLD=300 ≈ 0.9% del rango
  máximo — distingue voz de ruido de fondo en ambiente de oficina/habitación.
  Ajustar con --threshold si hay mucho ruido ambiental.
"""

import sys
import threading
import queue
import time
import numpy as np
from typing import Optional


# ── Configuración ─────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16000   # Hz — Moonshine requiere 16kHz
CHANNELS           = 1       # mono
CHUNK_SIZE         = 512     # samples por callback (~32ms a 16kHz)
DTYPE              = "int16"

ENERGY_THRESHOLD   = 8000    # calibrado: ruido de fondo ~3k-7k, voz ~10k+
SILENCE_SECONDS    = 0.8     # segundos de silencio para cortar la frase
MIN_SPEECH_SECONDS = 0.4     # duración mínima de habla (filtra clicks y ruidos breves)
LISTEN_TIMEOUT     = 30.0    # timeout máximo por turno

SILENCE_CHUNKS     = int(SILENCE_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
MIN_SPEECH_CHUNKS  = int(MIN_SPEECH_SECONDS * SAMPLE_RATE / CHUNK_SIZE)

# Device de audio:
#   "auto" → prueba 'pulse', después 'default'
#   None   → deja que sounddevice elija
#   int    → índice fijo
AUDIO_INPUT_DEVICE = "auto"


# ── Resolución de device ──────────────────────────────────────────────────────

def _resolve_input_device() -> Optional[int]:
    """
    Busca el device de entrada más adecuado.

    Prioriza 'pulse' porque los chips sof-hda-dsp en laptops Intel modernas
    solo aceptan 48000 Hz a nivel ALSA directo. PulseAudio acepta 16000 Hz
    y hace el resampling SRC internamente antes de mandar al hardware.
    """
    if AUDIO_INPUT_DEVICE != "auto":
        return AUDIO_INPUT_DEVICE if isinstance(AUDIO_INPUT_DEVICE, int) else None

    try:
        import sounddevice as sd
        devices = sd.query_devices()

        for name_target in ("pulse", "default"):
            for d in devices:
                if d["max_input_channels"] > 0 and d["name"].lower() == name_target:
                    print(f"  [STT] Device de entrada: [{d['index']}] {d['name']}")
                    return d["index"]

        print("  [STT] Usando device default del sistema.")
        return None
    except Exception as e:
        print(f"  [STT] Error resolviendo device: {e}")
        return None


# ── Engine ────────────────────────────────────────────────────────────────────

class STTEngine:
    """
    Motor STT: Transcriber base Moonshine + captura manual sounddevice.

    El Transcriber (solo ASR) se crea una vez y permanece en memoria.
    El RawInputStream de sounddevice se abre y cierra por turno para
    no mantener el mic tomado mientras el LLM genera la respuesta.
    """

    def __init__(self, language: str = "es", use_keyboard: bool = False):
        self.language             = language
        self.use_keyboard         = use_keyboard
        self._transcriber         = None
        self._model_path          = None
        self._model_arch          = None
        self._ready               = False
        self._moonshine_available = False
        self._device_idx          = None

        if not use_keyboard:
            self._init_moonshine()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_moonshine(self) -> None:
        try:
            from moonshine_voice import Transcriber, get_model_for_language
            import sounddevice
            self._moonshine_available = True
        except ImportError as e:
            print(f"  [STT] Dependencia faltante: {e}")
            self.use_keyboard = True
            return

        try:
            self._model_path, self._model_arch = get_model_for_language(self.language)
        except Exception as e:
            print(f"  [STT] Error cargando modelo '{self.language}': {e}")
            self.use_keyboard = True
            return

        try:
            from moonshine_voice import Transcriber
            self._transcriber = Transcriber(
                model_path=self._model_path,
                model_arch=self._model_arch,
            )
            self._device_idx = _resolve_input_device()
            self._ready = True
            print(f"  [STT] Moonshine inicializado (idioma={self.language})")
        except Exception as e:
            print(f"  [STT] Error iniciando Transcriber: {e}")
            self.use_keyboard = True

    # ── Cambio de idioma ──────────────────────────────────────────────────────

    def set_language(self, language: str) -> None:
        if language == self.language and self._transcriber is not None:
            return
        self.language = language
        if self.use_keyboard or not self._moonshine_available:
            return
        print(f"  [STT] Cambiando idioma a: {language}...")
        self._transcriber = None
        self._ready = False
        self._init_moonshine()

    # ── Listen ────────────────────────────────────────────────────────────────

    def listen(self, prompt: str = "[VOS]: ", timeout: float = None) -> str:
        if self.use_keyboard or self._transcriber is None:
            return self._listen_keyboard(prompt)
        return self._listen_moonshine(timeout or LISTEN_TIMEOUT)

    def _listen_keyboard(self, prompt: str) -> str:
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return "salir"

    def _listen_moonshine(self, timeout: float) -> str:
        """
        Captura audio con VAD de energía y transcribe con Moonshine.

        Pipeline por turno:
          1. Abrir RawInputStream (16kHz, mono, int16) sobre device resuelto
          2. Callback de audio → queue thread-safe
          3. Loop principal: leer chunks, calcular RMS, detectar habla/silencio
          4. Al detectar SILENCE_CHUNKS consecutivos después de habla → cortar
          5. Concatenar audio, convertir a float32 normalizado, transcribir
          6. Cerrar stream (libera el mic)

        RawInputStream entrega bytes int16 sin conversión — evita el round-trip
        float32→int16 que haría InputStream y es más eficiente en RAM.
        """
        import sounddevice as sd

        audio_queue: "queue.Queue[bytes]" = queue.Queue()

        def _audio_callback(indata: bytes, frames: int, time_info, status) -> None:
            audio_queue.put(bytes(indata))

        sys.stdout.write("  🎤 Escuchando...\r")
        sys.stdout.flush()

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_SIZE,
                device=self._device_idx,
                callback=_audio_callback,
            )
        except Exception as e:
            print(f"\n  [STT] No se pudo abrir el stream de audio: {e}")
            return self._listen_keyboard("[VOS]: ")

        speech_chunks      = []
        silence_count      = 0
        speech_chunk_count = 0
        speech_started     = False
        t_start            = time.monotonic()

        try:
            with stream:
                while True:
                    if time.monotonic() - t_start > timeout:
                        break

                    try:
                        raw = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    # VAD de energía RMS
                    chunk_np = np.frombuffer(raw, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(chunk_np.astype(np.float32) ** 2)))
                    is_speech = rms > ENERGY_THRESHOLD

                    if is_speech:
                        if not speech_started:
                            speech_started = True
                            sys.stdout.write("  🎤 Hablando...   \r")
                            sys.stdout.flush()
                        speech_chunks.append(raw)
                        speech_chunk_count += 1
                        silence_count = 0

                    elif speech_started:
                        speech_chunks.append(raw)
                        silence_count += 1
                        if silence_count >= SILENCE_CHUNKS:
                            break  # pausa suficiente → fin de frase

        except KeyboardInterrupt:
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
            return "salir"

        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

        if speech_chunk_count < MIN_SPEECH_CHUNKS:
            return ""  # demasiado corto, probablemente ruido

        try:
            audio_bytes = b"".join(speech_chunks)
            audio_np    = np.frombuffer(audio_bytes, dtype=np.int16)
            # Moonshine espera float32 normalizado en [-1.0, 1.0]
            audio_f32   = audio_np.astype(np.float32) / 32768.0

            result = self._transcriber.transcribe_without_streaming(audio_f32)

            text = " ".join(result).strip() if isinstance(result, list) else str(result).strip()
            if text:
                print(f"[VOS]: {text}")
            return text

        except Exception as e:
            print(f"\n  [STT] Error en transcripción: {e}")
            return ""

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._transcriber = None
        self._ready = False
        print("  [STT] Apagado.")

    @property
    def is_ready(self) -> bool:
        return self._ready or self.use_keyboard

    def get_info(self) -> dict:
        return {
            "engine":     "moonshine" if not self.use_keyboard else "keyboard",
            "language":   self.language,
            "ready":      self.is_ready,
            "model_path": str(self._model_path) if self._model_path else None,
            "device":     self._device_idx,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[STTEngine] = None


def get_stt(language: str = "es", use_keyboard: bool = False) -> STTEngine:
    global _engine
    if _engine is None:
        _engine = STTEngine(language=language, use_keyboard=use_keyboard)
    return _engine


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test STT Engine")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--lang", default="es", choices=["es", "en"])
    parser.add_argument("--threshold", type=int, default=ENERGY_THRESHOLD,
                        help=f"Umbral RMS del VAD (default: {ENERGY_THRESHOLD})")
    args = parser.parse_args()

    # Permitir override del threshold desde CLI
    if args.threshold != ENERGY_THRESHOLD:
        ENERGY_THRESHOLD = args.threshold
        SILENCE_CHUNKS   = int(SILENCE_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
        MIN_SPEECH_CHUNKS = int(MIN_SPEECH_SECONDS * SAMPLE_RATE / CHUNK_SIZE)

    engine = STTEngine(language=args.lang, use_keyboard=args.keyboard)
    print(f"\nSTT: {engine.get_info()}")
    print(f"VAD: threshold={ENERGY_THRESHOLD} RMS | silencio={SILENCE_SECONDS}s\n")
    print("Di algo (Ctrl+C para salir)...\n")

    try:
        while True:
            text = engine.listen()
            if not text:
                print("  (silencio/timeout)\n")
                continue
            if text.lower() in ("salir", "exit", "quit"):
                print("Chau!")
                break
            print(f"  → '{text}'\n")
    finally:
        engine.shutdown()