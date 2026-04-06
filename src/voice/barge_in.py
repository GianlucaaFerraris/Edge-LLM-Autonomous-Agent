"""
barge_in.py — Detector de interrupciones por voz (barge-in).

Permite al usuario interrumpir al asistente mientras habla diciendo
"pará", "bueno", "está bien", etc. El sistema detecta la voz,
aborta el TTS y el stream del LLM, y vuelve a escuchar.

PROBLEMA DE ECHO:
  Si el speaker y el mic están en el mismo dispositivo (laptop, SBC
  sin headset), el audio del TTS entra por el mic y el VAD lo detecta
  como voz del usuario → interrupción falsa constante.

SOLUCIÓN — "Gated VAD":
  El monitor de barge-in NO escucha continuamente. Tiene dos estados:

  1. GATE CERRADO (durante reproducción TTS):
     El mic captura audio pero el VAD está inhibido. No importa qué
     energía tenga el signal — no se reporta interrupción.

  2. GATE ABIERTO (entre oraciones, durante síntesis Piper):
     Hay un gap natural de ~20-50ms entre que termina sd.play() de
     la oración N y empieza sd.play() de la oración N+1. Durante
     este gap, Y durante la síntesis Piper (~20ms), el speaker está
     en silencio. El VAD se activa y si detecta voz, es del usuario.

  Además: cuando se abre el gate, hay un BLANKING_PERIOD de ~100ms
  donde se descarta audio para que el tail del playback anterior
  no genere un false positive.

PROTOCOLO DE USO (desde speak_stream):
  1. monitor.start()           → arranca thread de captura
  2. [entre oraciones]:
       monitor.open_gate()     → habilita detección
       if monitor.interrupted: → chequear flag
           break               → abortar stream
       monitor.close_gate()    → inhibe detección antes de sd.play()
  3. monitor.stop()            → cierra stream, limpia estado

CONSUMO DE RECURSOS:
  - Thread daemon: ~0 CPU cuando gate cerrado (solo append a buffer)
  - VAD: RMS sobre int16 array, ~0.1ms por chunk de 512 samples
  - RAM: buffer circular de ~1s de audio (32KB a 16kHz int16)
  - No carga ningún modelo ML — puramente energía RMS
"""

import threading
import queue
import time
import numpy as np
from typing import Optional


# ── Configuración ─────────────────────────────────────────────────────────────

SAMPLE_RATE       = 16000   # Hz — consistente con stt_engine
CHANNELS          = 1
CHUNK_SIZE        = 512     # samples por callback (~32ms)
DTYPE             = "int16"

# Umbral de energía para considerar "voz detectada".
# Más alto que el ENERGY_THRESHOLD del STT (8000) porque durante el gap
# entre oraciones puede haber reverberación residual del speaker.
# El usuario que interrumpe habla con intención → energía más alta.
BARGE_IN_ENERGY_THRESHOLD = 10000

# Cantidad de chunks consecutivos con voz para confirmar interrupción.
# A 32ms/chunk, 4 chunks = ~128ms de voz sostenida.
# Filtra transitorios (clicks, golpes) sin agregar latencia perceptible.
BARGE_IN_CONFIRM_CHUNKS = 4

# Blanking period después de abrir el gate (en chunks).
# Descarta audio residual del playback anterior.
# 3 chunks × 32ms = ~96ms de blanking.
BARGE_IN_BLANKING_CHUNKS = 3


# ── Monitor ───────────────────────────────────────────────────────────────────

class BargeInMonitor:
    """
    Monitor de interrupciones por voz durante la reproducción de TTS.
    
    Diseño thread-safe:
      - Un thread daemon captura audio continuamente del mic
      - El thread principal (speak_stream) controla el gate
      - La comunicación es via threading.Event (lock-free en CPython)
      - El flag `interrupted` es un Event que se chequea sin blocking
    
    Ciclo de vida:
      start() → [open_gate/close_gate por cada oración] → stop()
    """

    def __init__(self, device_idx: Optional[int] = None,
                 energy_threshold: int = BARGE_IN_ENERGY_THRESHOLD,
                 confirm_chunks: int = BARGE_IN_CONFIRM_CHUNKS):
        self._device_idx = device_idx
        self._energy_threshold = energy_threshold
        self._confirm_chunks = confirm_chunks

        # Estado — todos thread-safe (Event/atomic en CPython)
        self._interrupted = threading.Event()
        self._gate_open = threading.Event()       # gate cerrado por default
        self._running = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # Contadores internos (accedidos solo desde capture thread)
        self._consecutive_voice = 0
        self._blanking_remaining = 0

    @property
    def interrupted(self) -> bool:
        """True si se detectó voz del usuario (interrupción confirmada)."""
        return self._interrupted.is_set()

    def start(self) -> bool:
        """
        Inicia la captura de audio para monitoreo de barge-in.
        Retorna True si se pudo iniciar, False si no hay mic disponible.
        
        El gate arranca CERRADO — no se detectan interrupciones hasta
        que se llame open_gate().
        """
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return True  # ya corriendo

        self._interrupted.clear()
        self._gate_open.clear()
        self._running.set()
        self._consecutive_voice = 0
        self._blanking_remaining = 0

        try:
            import sounddevice as sd
            # Verificar que el device existe y tiene input
            sd.check_input_settings(
                device=self._device_idx,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
            )
        except Exception as e:
            print(f"  [BARGE-IN] No se puede abrir mic: {e}")
            return False

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="barge-in-vad",
            daemon=True,
        )
        self._capture_thread.start()
        return True

    def stop(self) -> None:
        """Detiene la captura. Llamar al final de speak_stream."""
        self._running.clear()
        self._gate_open.clear()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None

    def open_gate(self) -> None:
        """
        Abre el gate: habilita la detección de voz.
        Llamar DESPUÉS de que sd.wait() termine (speaker en silencio)
        y ANTES de la próxima síntesis/playback.
        """
        self._blanking_remaining = BARGE_IN_BLANKING_CHUNKS
        self._consecutive_voice = 0
        self._gate_open.set()

    def close_gate(self) -> None:
        """
        Cierra el gate: inhibe la detección de voz.
        Llamar ANTES de sd.play() para evitar detectar el echo del speaker.
        """
        self._gate_open.clear()
        self._consecutive_voice = 0

    def reset(self) -> None:
        """Limpia el flag de interrupción para reutilizar el monitor."""
        self._interrupted.clear()
        self._consecutive_voice = 0

    # ── Thread de captura ─────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Loop de captura de audio en thread daemon.
        
        Usa RawInputStream (no InputStream) para evitar conversión
        float32 innecesaria — solo necesitamos int16 para calcular RMS.
        
        El stream se mantiene abierto todo el tiempo que dura el
        speak_stream. Abrir/cerrar el stream por cada gap entre
        oraciones agregaría ~50-100ms de latencia (reconfiguración
        de ALSA/PulseAudio), lo cual consumiría todo el gap.
        """
        import sounddevice as sd

        audio_q: queue.Queue[bytes] = queue.Queue(maxsize=50)

        def _callback(indata: bytes, frames: int, time_info, status) -> None:
            if status:
                pass  # underrun/overflow — ignorar silenciosamente
            try:
                audio_q.put_nowait(bytes(indata))
            except queue.Full:
                pass  # descartar si el consumer no da abasto

        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_SIZE,
                device=self._device_idx,
                callback=_callback,
            )
        except Exception as e:
            print(f"  [BARGE-IN] Error abriendo stream: {e}")
            return

        with stream:
            while self._running.is_set():
                try:
                    raw = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Si el gate está cerrado, descartar el audio
                if not self._gate_open.is_set():
                    self._consecutive_voice = 0
                    continue

                # Si ya se confirmó interrupción, no procesar más
                if self._interrupted.is_set():
                    continue

                # Blanking period: descartar chunks iniciales post-gate
                if self._blanking_remaining > 0:
                    self._blanking_remaining -= 1
                    continue

                # VAD de energía RMS
                chunk_np = np.frombuffer(raw, dtype=np.int16)
                rms = float(np.sqrt(np.mean(chunk_np.astype(np.float32) ** 2)))

                if rms > self._energy_threshold:
                    self._consecutive_voice += 1
                    if self._consecutive_voice >= self._confirm_chunks:
                        self._interrupted.set()
                        print("\n  [BARGE-IN] 🛑 Interrupción detectada")
                else:
                    # Reset parcial: permitir 1 chunk de silencio en medio
                    # de la frase (pausa natural entre palabras).
                    # Solo resetear si hay ≥2 chunks de silencio consecutivos.
                    if self._consecutive_voice > 0:
                        self._consecutive_voice = max(0, self._consecutive_voice - 1)

        # Stream cerrado automáticamente por context manager


# ── Singleton ─────────────────────────────────────────────────────────────────

_monitor: Optional[BargeInMonitor] = None


def get_barge_in_monitor(device_idx: Optional[int] = None) -> BargeInMonitor:
    """Retorna la instancia global del monitor de barge-in."""
    global _monitor
    if _monitor is None:
        _monitor = BargeInMonitor(device_idx=device_idx)
    return _monitor