"""
tts_engine.py — Text-to-Speech engine usando Piper.

Piper es un sistema TTS neural que usa modelos VITS exportados a ONNX.
En ARM (aarch64), OnnxRuntime aprovecha NEON SIMD → ~20ms por oración
en Raspberry Pi 4, ~10ms en Rock 5B (RK3588 con mejores cores).

Características clave:
  - Modelos monolingües (~15-60MB cada uno)
  - Voces pre-entrenadas para es_AR, es_ES, es_MX, en_US
  - No requiere GPU — corre 100% en CPU
  - Latencia imperceptible para frases cortas

Mapeo de voces por modo:
  ┌────────────────┬────────────────────────────────────────────┐
  │ Modo           │ Voz (idioma / modelo)                      │
  ├────────────────┼────────────────────────────────────────────┤
  │ agent          │ es_MX-ald-medium (hombre amigable)         │
  │ engineering_es │ es_ES-davefx-medium (hombre maduro/grave)  │
  │ engineering_en │ en_US-hfc_male-medium (hombre, fallback)   │
  │ english_es     │ es_AR-daniela-high (mujer, intros español) │
  │ english_en     │ en_US-amy-medium (mujer, feedback inglés)  │
  └────────────────┴────────────────────────────────────────────┘

Uso:
    from tts_engine import TTSEngine

    tts = TTSEngine()
    tts.speak("Hola, ¿qué necesitás?", voice_key="agent")
    tts.speak("That's a great answer!", voice_key="english_en")

Requisitos:
    pip install piper-tts

    Los modelos se descargan automáticamente en la primera ejecución.
    Ubicación default: ~/.local/share/piper-voices/
"""

import io
import os
import subprocess
import sys
import time
import wave
import hashlib
import struct
from pathlib import Path
from typing import Optional

# ── Configuración de voces ────────────────────────────────────────────────────

# Directorio donde se almacenan los modelos descargados
VOICES_DIR = Path.home() / ".local" / "share" / "piper-voices"

# Base URL para descargar voces desde HuggingFace
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Definición de voces por key
# Cada entrada tiene: idioma de Piper, nombre de voz, calidad, y metadata
VOICE_MAP = {
    # Orquestador y Agente — hombre amigable, español latinoamericano
    "agent": {
        "lang_code": "es_MX",
        "lang_family": "es",
        "name": "ald",
        "quality": "medium",
        "description": "Hombre amigable (agente/orquestador)",
        # Piper speech rate: 1.0 = normal, <1 = más lento, >1 = más rápido
        "length_scale": 1.0,
    },

    # Tutor de ingeniería — hombre maduro/sabio, español peninsular
    # davefx tiene un tono más grave y "profesoral"
    "engineering_es": {
        "lang_code": "es_ES",
        "lang_family": "es",
        "name": "davefx",
        "quality": "medium",
        "description": "Hombre maduro (ingeniería, español)",
        "length_scale": 1.05,  # ligeramente más lento → más gravitas
    },

    # Tutor de ingeniería — fallback para respuestas en inglés
    "engineering_en": {
        "lang_code": "en_US",
        "lang_family": "en",
        "name": "hfc_male",
        "quality": "medium",
        "description": "Hombre (ingeniería, inglés)",
        "length_scale": 1.0,
    },

    # Tutor de inglés — mujer, intros en español
    "english_es": {
        "lang_code": "es_AR",
        "lang_family": "es",
        "name": "daniela",
        "quality": "high",
        "description": "Mujer (tutor inglés, partes en español)",
        "length_scale": 1.0,
    },

    # Tutor de inglés — mujer, feedback en inglés
    "english_en": {
        "lang_code": "en_US",
        "lang_family": "en",
        "name": "amy",
        "quality": "medium",
        "description": "Mujer (tutor inglés, feedback)",
        "length_scale": 1.0,
    },
}


def _model_filename(voice_cfg: dict) -> str:
    """Genera el nombre del archivo .onnx para una voz."""
    return f"{voice_cfg['lang_code']}-{voice_cfg['name']}-{voice_cfg['quality']}.onnx"


def _model_url(voice_cfg: dict) -> str:
    """URL de descarga del modelo .onnx desde HuggingFace."""
    lf = voice_cfg["lang_family"]
    lc = voice_cfg["lang_code"]
    name = voice_cfg["name"]
    quality = voice_cfg["quality"]
    filename = _model_filename(voice_cfg)
    return f"{HF_BASE}/{lf}/{lc}/{name}/{quality}/{filename}"


def _config_url(voice_cfg: dict) -> str:
    """URL de descarga del config .onnx.json."""
    return _model_url(voice_cfg) + ".json"


# ── Detección de idioma (heurística rápida) ───────────────────────────────────

# Palabras que aparecen frecuentemente en español y casi nunca en inglés
_ES_MARKERS = {
    "quiero", "puedo", "tengo", "necesito", "vamos", "dame",
    "hacé", "decime", "pasame", "cambiemos", "estoy", "puede",
    "sobre", "porque", "también", "ahora", "después", "pero",
    "como", "para", "donde", "cuando", "todos", "mejor",
    "bueno", "gracias", "hola", "chau", "dale", "algo",
    "creo", "sería", "podría", "debería", "muy", "más",
    "qué", "cómo", "cuál", "dónde", "cuándo", "todavía",
    "listo", "evento", "tarea", "calendario", "recordatorio",
    "momento", "déjame", "revisar", "libros", "retomando",
    "concepto", "explorar", "perfecto", "arrancamos",
}


def detect_language(text: str) -> str:
    """
    Heurística rápida para detectar idioma del texto.
    Retorna "es" o "en".
    
    No usa LLM — puramente estadístico para no agregar latencia.
    La misma lógica que _is_likely_english() en main.py pero invertida.
    """
    words = set(text.lower().split())
    es_count = len(words & _ES_MARKERS)

    if es_count >= 2:
        return "es"
    if es_count >= 1 and len(words) < 10:
        return "es"
    return "en"


# ── Engine ────────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Motor de Text-to-Speech basado en Piper.
    
    Soporta múltiples voces simultáneas (una por modo).
    Los modelos se descargan automáticamente la primera vez.
    En memoria se mantiene solo el modelo activo (Piper CLI los carga por archivo).
    
    Dos backends:
      1. piper-tts Python package (preferido, más control)
      2. Piper CLI binary (fallback si el package no funciona)
      3. print() fallback si nada funciona
    """

    def __init__(self, preload_voices: list[str] = None,
                 use_print: bool = False):
        """
        Args:
            preload_voices: Lista de voice_keys a pre-descargar.
                           Default: todas las voces definidas.
            use_print: Si True, solo imprime texto sin generar audio.
        """
        self.use_print = use_print
        self._piper_available = False
        self._piper_cli_path: Optional[str] = None
        self._voice_models: dict[str, Path] = {}  # voice_key → path al .onnx
        self._current_voice_key: Optional[str] = None

        if not use_print:
            self._detect_piper()

        if not use_print and self._piper_available:
            voices_to_load = preload_voices or list(VOICE_MAP.keys())
            self._ensure_voices(voices_to_load)

    def _detect_piper(self) -> None:
        """Detecta si Piper está disponible (Python package o CLI)."""
        # Opción 1: Python package
        try:
            import piper
            self._piper_available = True
            self._backend = "python"
            print("  [TTS] Piper (Python package) detectado.")
            return
        except ImportError:
            pass

        # Opción 2: CLI binary
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self._piper_available = True
                self._piper_cli_path = "piper"
                self._backend = "cli"
                print("  [TTS] Piper CLI detectado.")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Opción 3: Binary en paths comunes
        for path in [
            Path.home() / "piper" / "piper",
            Path("/usr/local/bin/piper"),
            Path("/opt/piper/piper"),
        ]:
            if path.exists():
                self._piper_available = True
                self._piper_cli_path = str(path)
                self._backend = "cli"
                print(f"  [TTS] Piper CLI encontrado en: {path}")
                return

        print("  [TTS] ⚠ Piper no encontrado.")
        print("        Opción 1: pip install piper-tts")
        print("        Opción 2: descargar binario desde https://github.com/rhasspy/piper/releases")
        print("  [TTS] Usando modo texto como fallback.")
        self.use_print = True

    def _ensure_voices(self, voice_keys: list[str]) -> None:
        """
        Verifica que los modelos de voz estén descargados.
        Si no, los descarga de HuggingFace.
        """
        VOICES_DIR.mkdir(parents=True, exist_ok=True)

        for key in voice_keys:
            cfg = VOICE_MAP.get(key)
            if not cfg:
                print(f"  [TTS] Voice key desconocida: {key}")
                continue

            model_file = VOICES_DIR / _model_filename(cfg)
            config_file = VOICES_DIR / (_model_filename(cfg) + ".json")

            if model_file.exists() and config_file.exists():
                self._voice_models[key] = model_file
                continue

            # Descargar
            print(f"  [TTS] Descargando voz '{key}' ({cfg['description']})...")
            try:
                self._download_file(_model_url(cfg), model_file)
                self._download_file(_config_url(cfg), config_file)
                self._voice_models[key] = model_file
                print(f"  [TTS]   ✅ {_model_filename(cfg)}")
            except Exception as e:
                print(f"  [TTS]   ❌ Error descargando '{key}': {e}")

    def _download_file(self, url: str, dest: Path) -> None:
        """Descarga un archivo con progress básico."""
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))

    def speak(self, text: str, voice_key: str = "agent",
              auto_detect_lang: bool = False) -> None:
        """
        Sintetiza y reproduce texto como audio.
        
        Args:
            text: Texto a hablar.
            voice_key: Clave de la voz a usar (ver VOICE_MAP).
            auto_detect_lang: Si True, detecta el idioma del texto
                             y elige la voz apropiada dentro del modo.
                             Ej: si voice_key="engineering_es" pero el texto
                             es inglés, usa "engineering_en".
        """
        if not text or not text.strip():
            return

        if self.use_print:
            self._speak_print(text, voice_key)
            return

        # Auto-detección de idioma
        if auto_detect_lang:
            voice_key = self._resolve_voice_for_text(text, voice_key)

        model_path = self._voice_models.get(voice_key)
        if not model_path or not model_path.exists():
            self._speak_print(text, voice_key)
            return

        # Obtener length_scale de la config
        cfg = VOICE_MAP.get(voice_key, {})
        length_scale = cfg.get("length_scale", 1.0)

        try:
            if self._backend == "python":
                self._speak_python(text, model_path, length_scale)
            else:
                self._speak_cli(text, model_path, length_scale)
        except Exception as e:
            print(f"  [TTS] Error sintetizando: {e}")
            self._speak_print(text, voice_key)

    def _resolve_voice_for_text(self, text: str,
                                 base_voice_key: str) -> str:
        """
        Dado un voice_key base y texto, determina la voz correcta
        según el idioma detectado.
        
        Mapeo de fallback:
          engineering_es ↔ engineering_en
          english_es ↔ english_en
          agent → siempre español
        """
        lang = detect_language(text)

        # Mapeos de fallback
        fallback_map = {
            ("engineering_es", "en"): "engineering_en",
            ("engineering_en", "es"): "engineering_es",
            ("english_es", "en"): "english_en",
            ("english_en", "es"): "english_es",
        }

        resolved = fallback_map.get((base_voice_key, lang), base_voice_key)

        # Verificar que la voz resuelta está disponible
        if resolved in self._voice_models:
            return resolved
        return base_voice_key

    def _speak_python(self, text: str, model_path: Path,
                      length_scale: float) -> None:
        """
        Síntesis usando el Python package de Piper.

        API confirmada (piper-tts instalado):
          synthesize(text, syn_config=None) -> Iterable[AudioChunk]

        synthesize() NO recibe wav_file. Retorna un iterable de AudioChunk;
        cada chunk tiene un atributo con los bytes PCM crudos. El caller
        es responsable de escribirlos al WAV. No iterar = 0 frames escritos
        = buffer de 44 bytes (solo header) = silencio sin error.

        AudioChunk tiene al menos uno de estos atributos según sub-versión:
          .audio        → bytes  (más común)
          .audio_bytes  → bytes
          .audio_array  → np.ndarray int16
        """
        import piper
        import inspect

        voice = piper.PiperVoice.load(str(model_path))

        # Sample rate desde el config JSON del modelo
        config_path = str(model_path) + ".json"
        sample_rate = 22050
        try:
            import json as _json
            with open(config_path) as f:
                cfg = _json.load(f)
            sample_rate = cfg.get("audio", {}).get("sample_rate", 22050)
        except Exception:
            pass

        # Construir SynthesisConfig
        syn_config = None
        try:
            from piper import SynthesisConfig
            try:
                syn_config = SynthesisConfig(length_scale=length_scale)
            except TypeError:
                # Esta sub-versión no acepta length_scale en el constructor
                syn_config = SynthesisConfig()
        except ImportError:
            pass  # syn_config=None usa defaults del modelo

        # Sintetizar y escribir frames chunk a chunk
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)   # 16-bit PCM
            wav_file.setframerate(sample_rate)

            for chunk in voice.synthesize(text, syn_config):
                # AudioChunk confirmed attributes (piper-tts installed version):
                # .audio_int16_bytes → bytes  (int16 LE PCM, primary)
                # .audio_int16_array → np.ndarray int16 (fallback)
                # .audio_float_array → np.ndarray float32 (last resort)
                if chunk.audio_int16_bytes:
                    wav_file.writeframes(chunk.audio_int16_bytes)
                elif chunk.audio_int16_array is not None:
                    wav_file.writeframes(chunk.audio_int16_array.tobytes())
                elif chunk.audio_float_array is not None:
                    import numpy as np
                    arr = (chunk.audio_float_array * 32767).astype(np.int16)
                    wav_file.writeframes(arr.tobytes())

        buf_size = wav_buffer.tell()
        if buf_size < 100:
            print(f"  [TTS] ⚠ Buffer WAV vacío ({buf_size}b). "
                  f"Texto: {text[:40]!r}")
            return

        wav_buffer.seek(0)
        self._play_wav(wav_buffer)

    def _speak_cli(self, text: str, model_path: Path,
                   length_scale: float) -> None:
        """Síntesis usando el CLI de Piper + aplay/paplay."""
        player = self._find_audio_player()

        # Pipeline: echo text | piper --model X --output-raw | player
        piper_cmd = [
            self._piper_cli_path,
            "--model", str(model_path),
            "--output-raw",
            "--length-scale", str(length_scale),
        ]

        # Leer config para sample rate
        config_path = str(model_path) + ".json"
        sample_rate = 22050  # default de Piper
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            sample_rate = config.get("audio", {}).get("sample_rate", 22050)
        except Exception:
            pass

        if player == "aplay":
            player_cmd = [
                "aplay", "-r", str(sample_rate), "-f", "S16_LE",
                "-t", "raw", "-c", "1", "-q",
            ]
        elif player == "paplay":
            player_cmd = [
                "paplay", "--raw",
                "--rate", str(sample_rate),
                "--channels", "1",
                "--format", "s16le",
            ]
        elif player == "sox":
            player_cmd = [
                "play", "-t", "raw", "-r", str(sample_rate),
                "-e", "signed", "-b", "16", "-c", "1", "-q", "-",
            ]
        else:
            # Fallback: guardar como WAV temporal y reproducir
            self._speak_cli_wav_fallback(text, model_path, length_scale)
            return

        # Ejecutar pipeline
        piper_proc = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        player_proc = subprocess.Popen(
            player_cmd,
            stdin=piper_proc.stdout,
            stderr=subprocess.DEVNULL,
        )

        piper_proc.stdin.write(text.encode("utf-8"))
        piper_proc.stdin.close()
        player_proc.wait()
        piper_proc.wait()

    def _speak_cli_wav_fallback(self, text: str, model_path: Path,
                                 length_scale: float) -> None:
        """Fallback: genera WAV temporal y reproduce."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            subprocess.run(
                [
                    self._piper_cli_path,
                    "--model", str(model_path),
                    "--output_file", tmp_path,
                    "--length-scale", str(length_scale),
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            self._play_wav_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _play_wav(self, wav_buffer: io.BytesIO) -> None:
        """
        Reproduce un WAV desde un buffer en memoria.

        Estrategia de selección de dispositivo (en orden):
          1. sounddevice con device='pulse' → usa PulseAudio/PipeWire,
             que ya sabe cuál es el output activo (auriculares/parlantes).
          2. sounddevice con device=AGENTY_AUDIO_DEVICE (int) si está definido
             como variable de entorno — override manual.
          3. sounddevice con default del sistema (puede ser HDMI en algunos setups).
          4. Fallback: aplay vía archivo temporal.

        Para ver los índices de dispositivo disponibles:
          python3 -c "import sounddevice as sd; print(sd.query_devices())"

        Para forzar un dispositivo específico sin tocar el código:
          export AGENTY_AUDIO_DEVICE=5   # índice numérico de sounddevice
        """
        try:
            import sounddevice as sd
            import numpy as np

            wav_buffer.seek(0)
            with wave.open(wav_buffer, "rb") as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

            # Determinar dispositivo de salida
            device = self._resolve_audio_device(sd)

            sd.play(audio_float, samplerate=sample_rate, device=device)
            sd.wait()
            return
        except ImportError:
            pass
        except Exception as e:
            # No silenciar — ayuda a diagnosticar problemas de dispositivo
            print(f"  [TTS] sounddevice error: {e} — usando fallback aplay")

        # Fallback: escribir a archivo temporal y usar aplay
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_buffer.getvalue())
            tmp_path = f.name
        try:
            self._play_wav_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _resolve_audio_device(self, sd) -> object:
        """
        Determina el dispositivo de salida para sounddevice.

        Prioridad:
          1. Variable de entorno AGENTY_AUDIO_DEVICE (índice int o nombre string)
          2. String 'pulse' — sounddevice acepta nombres parciales y PulseAudio/
             PipeWire ya sabe cuál es el sink activo (auriculares, parlantes, etc.)
          3. None → default del sistema

        sounddevice.play(device=X) acepta:
          - int  → índice del listado de sd.query_devices()
          - str  → substring del nombre del dispositivo (case-insensitive)
          - None → default del sistema (puede ser HDMI en laptops con dGPU)

        Para ver los índices disponibles en tu sistema:
          python3 -c "import sounddevice as sd; print(sd.query_devices())"
        Para forzar un dispositivo sin tocar el código:
          export AGENTY_AUDIO_DEVICE=11   # índice del dispositivo 'pulse'
        """
        # Override manual por env var (máxima prioridad)
        env_device = os.environ.get("AGENTY_AUDIO_DEVICE")
        if env_device is not None:
            try:
                device = int(env_device)
            except ValueError:
                device = env_device
            if not getattr(self, "_audio_device_logged", False):
                print(f"  [TTS] Audio device (env override): {device!r}")
                self._audio_device_logged = True
            return device

        # Intentar PulseAudio/PipeWire por nombre string.
        # sd.play(device='pulse') hace que PortAudio delegue el routing
        # a PulseAudio, que ya tiene configurado el sink activo correcto.
        # Verificamos que el nombre exista antes de usarlo.
        try:
            devs = sd.query_devices()
            # query_devices() sin args retorna un DeviceList; indexar con
            # un string hace búsqueda por substring — si no encuentra lanza
            # ValueError, que capturamos para caer al default.
            sd.query_devices('pulse', 'output')
            device = 'pulse'
        except (ValueError, Exception):
            device = None  # default del sistema

        if not getattr(self, "_audio_device_logged", False):
            print(f"  [TTS] Audio device: {device!r} "
                  f"(None = system default, 'pulse' = PulseAudio)")
            self._audio_device_logged = True

        return device

    def _play_wav_file(self, path: str) -> None:
        """Reproduce un archivo WAV con el player disponible."""
        player = self._find_audio_player()
        try:
            if player == "aplay":
                # Intentar primero con el default del sistema (PulseAudio/dmix)
                # Si falla, reintentar con plughw:1,0 (HDA Analog, típico en laptops)
                result = subprocess.run(
                    ["aplay", "-q", path],
                    capture_output=True, timeout=30
                )
                if result.returncode != 0:
                    subprocess.run(
                        ["aplay", "-q", "-D", "plughw:1,0", path],
                        capture_output=True, timeout=30
                    )
            elif player == "paplay":
                subprocess.run(["paplay", path],
                               capture_output=True, timeout=30)
            elif player == "sox":
                subprocess.run(["play", "-q", path],
                               capture_output=True, timeout=30)
            elif player == "ffplay":
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                    capture_output=True, timeout=30
                )
            else:
                print(f"  [TTS] No se encontró reproductor de audio.")
        except Exception as e:
            print(f"  [TTS] Error reproduciendo audio: {e}")

    def _find_audio_player(self) -> Optional[str]:
        """Detecta el reproductor de audio disponible."""
        for player in ["aplay", "paplay", "sox", "ffplay"]:
            try:
                cmd = [player, "--help"] if player != "ffplay" else [player, "-version"]
                subprocess.run(cmd, capture_output=True, timeout=3)
                return player
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    def _speak_print(self, text: str, voice_key: str) -> None:
        """Fallback: solo imprime el texto."""
        cfg = VOICE_MAP.get(voice_key, {})
        desc = cfg.get("description", voice_key)
        # Elegir prefijo según modo
        prefixes = {
            "agent": "[AGENTE]",
            "engineering_es": "[INGENIERO]",
            "engineering_en": "[INGENIERO]",
            "english_es": "[TUTOR]",
            "english_en": "[TUTOR]",
        }
        prefix = prefixes.get(voice_key, "[ASISTENTE]")
        print(f"\n{prefix}: {text}\n")

    @property
    def is_ready(self) -> bool:
        return self._piper_available or self.use_print

    def get_info(self) -> dict:
        """Info de diagnóstico."""
        return {
            "engine": "piper" if self._piper_available else "print",
            "backend": getattr(self, "_backend", "none"),
            "voices_loaded": list(self._voice_models.keys()),
            "voices_available": list(VOICE_MAP.keys()),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[TTSEngine] = None


def get_tts(use_print: bool = False) -> TTSEngine:
    """Retorna la instancia global del TTS engine."""
    global _engine
    if _engine is None:
        _engine = TTSEngine(use_print=use_print)
    return _engine


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-only", action="store_true",
                        help="Solo imprimir texto, sin audio")
    parser.add_argument("--voice", default="agent",
                        choices=list(VOICE_MAP.keys()),
                        help="Voice key a usar")
    parser.add_argument("text", nargs="*", default=["Hola, ¿qué necesitás?"])
    args = parser.parse_args()

    engine = TTSEngine(use_print=args.print_only)
    print(f"\nTTS Engine: {engine.get_info()}")

    text = " ".join(args.text)
    print(f"\nSintetizando con voz '{args.voice}': {text}\n")
    engine.speak(text, voice_key=args.voice)

    # Test de todas las voces
    if not args.print_only:
        print("\n--- Test de todas las voces ---\n")
        tests = [
            ("agent", "Hola, tenés tres tareas pendientes."),
            ("engineering_es", "La retropropagación calcula el gradiente de la función de pérdida."),
            ("engineering_en", "Backpropagation computes the gradient of the loss function."),
            ("english_es", "Ahora vamos a practicar con un tema interesante."),
            ("english_en", "That's a great answer! Let me ask you a follow-up question."),
        ]
        for voice_key, test_text in tests:
            if voice_key in engine._voice_models:
                print(f"  [{voice_key}] {test_text}")
                engine.speak(test_text, voice_key=voice_key)
                time.sleep(0.5)
            else:
                print(f"  [{voice_key}] ⚠ Modelo no disponible")