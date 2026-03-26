"""
stt_engine.py — Speech-to-Text engine usando Moonshine Voice v2.

Moonshine es un modelo ASR diseñado para edge devices con:
  - Encoder de longitud variable (no zero-padding como Whisper)
  - VAD integrado (Silero-based) para detección de fin de frase
  - Streaming con caching de estados del decoder
  - Modelos mono-idioma especializados (en, es)

Latencias medidas (CPU ARM):
  - Moonshine Tiny:   ~50ms  por frase (26MB modelo)
  - Moonshine Base:   ~120ms por frase (58MB modelo)

El STT opera en modo "phrase detection": escucha continuamente,
detecta inicio de voz (VAD), transcribe incrementalmente, y emite
el texto final cuando detecta una pausa >= SILENCE_THRESHOLD.

Uso:
    from stt_engine import STTEngine

    stt = STTEngine(language="es")  # o "en" para inglés
    text = stt.listen()             # bloquea hasta que el usuario termina de hablar
    stt.set_language("en")          # cambiar idioma en caliente

Requisitos:
    pip install moonshine-voice

Para testeo sin micrófono:
    stt = STTEngine(language="es", use_keyboard=True)
    text = stt.listen()  # fallback a input()
"""

import os
import sys
import time
import threading
from typing import Optional, Callable

# ── Configuración ─────────────────────────────────────────────────────────────

# Modelo por idioma — Moonshine recomienda modelos mono-idioma para mejor WER
MODEL_PREFERENCE = {
    "en": "base",   # inglés: base para mejor accuracy en el tutor
    "es": "base",   # español: base (los tiny son solo inglés por ahora)
}

# Timeout de silencio después del cual se considera que el usuario terminó
# de hablar (en segundos). Moonshine VAD usa ~300ms internamente;
# este es un timeout adicional de safety.
LISTEN_TIMEOUT = 30.0  # máximo tiempo de escucha antes de cortar

# ── Engine ────────────────────────────────────────────────────────────────────

class STTEngine:
    """
    Motor de Speech-to-Text basado en Moonshine Voice v2.
    
    Modos de operación:
      - Micrófono real: usa MicTranscriber con VAD para escucha continua
      - Keyboard fallback: usa input() para desarrollo sin mic
    
    Thread safety: listen() es blocking y debe llamarse desde un solo thread.
    El MicTranscriber interno maneja sus propios threads para captura de audio.
    """

    def __init__(self, language: str = "es", use_keyboard: bool = False):
        self.language = language
        self.use_keyboard = use_keyboard
        self._transcriber = None
        self._model_path = None
        self._model_arch = None
        self._ready = False
        self._moonshine_available = False

        if not use_keyboard:
            self._init_moonshine()

    def _init_moonshine(self) -> None:
        """Inicializa Moonshine Voice. Si falla, cae a keyboard mode."""
        try:
            from moonshine_voice import (
                MicTranscriber,
                TranscriptEventListener,
                get_model_for_language,
            )
            self._moonshine_available = True
        except ImportError:
            print("  [STT] moonshine-voice no instalado.")
            print("        pip install moonshine-voice")
            print("  [STT] Usando modo teclado como fallback.")
            self.use_keyboard = True
            return

        try:
            self._model_path, self._model_arch = get_model_for_language(
                self.language
            )
            self._ready = True
            print(f"  [STT] Moonshine inicializado (idioma={self.language})")
        except Exception as e:
            print(f"  [STT] Error inicializando Moonshine: {e}")
            print("  [STT] Usando modo teclado como fallback.")
            self.use_keyboard = True

    def set_language(self, language: str) -> None:
        """
        Cambia el idioma del STT en caliente.
        Moonshine usa modelos mono-idioma, así que esto recarga el modelo.
        Es una operación costosa (~500ms) — hacerlo solo al cambiar de modo.
        """
        if language == self.language:
            return

        self.language = language

        if self.use_keyboard or not self._moonshine_available:
            return

        try:
            from moonshine_voice import get_model_for_language
            self._model_path, self._model_arch = get_model_for_language(
                language
            )
            # Forzar recreación del transcriber en el próximo listen()
            self._transcriber = None
            self._ready = True
            print(f"  [STT] Idioma cambiado a: {language}")
        except Exception as e:
            print(f"  [STT] Error cambiando idioma a {language}: {e}")

    def listen(self, prompt: str = "[VOS]: ", timeout: float = None) -> str:
        """
        Escucha al usuario y retorna el texto transcripto.
        
        Bloquea hasta que:
          - El usuario termina de hablar (pausa detectada por VAD)
          - Se alcanza el timeout
          - El usuario presiona Ctrl+C (retorna "salir")
        
        Args:
            prompt: Prompt visual para modo teclado.
            timeout: Timeout en segundos (default: LISTEN_TIMEOUT).
        
        Returns:
            Texto transcripto, o "salir" si se interrumpe.
        """
        if self.use_keyboard:
            return self._listen_keyboard(prompt)
        return self._listen_moonshine(timeout or LISTEN_TIMEOUT)

    def _listen_keyboard(self, prompt: str) -> str:
        """Fallback a input() para desarrollo sin micrófono."""
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return "salir"

    def _listen_moonshine(self, timeout: float) -> str:
        """
        Escucha con Moonshine MicTranscriber.
        
        Usa un event listener para capturar el texto cuando el VAD
        detecta fin de frase (committed transcript).
        """
        from moonshine_voice import (
            MicTranscriber,
            TranscriptEventListener,
        )

        result_text = ""
        done_event = threading.Event()

        class PhraseListener(TranscriptEventListener):
            """Listener que captura la primera frase completa."""

            def on_transcription_committed(self, text: str) -> None:
                nonlocal result_text
                if text.strip():
                    result_text = text.strip()
                    done_event.set()

            def on_transcription_updated(self, text: str) -> None:
                # Feedback visual mientras el usuario habla
                sys.stdout.write(f"\r  🎤 {text}   ")
                sys.stdout.flush()

        listener = PhraseListener()

        try:
            # Crear transcriber fresco cada vez para evitar estado stale
            # Moonshine cachea el modelo en memoria, así que la creación
            # es rápida después de la primera vez
            transcriber = MicTranscriber(
                model_path=self._model_path,
                model_arch=self._model_arch,
            )
            transcriber.add_listener(listener)
            transcriber.start()

            # Indicador visual
            sys.stdout.write("  🎤 Escuchando...")
            sys.stdout.flush()

            # Esperar a que el VAD detecte fin de frase o timeout
            done_event.wait(timeout=timeout)

            transcriber.stop()

            # Limpiar línea de feedback
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

            if result_text:
                print(f"[VOS]: {result_text}")
                return result_text
            else:
                return ""

        except KeyboardInterrupt:
            if hasattr(transcriber, 'stop'):
                transcriber.stop()
            return "salir"
        except Exception as e:
            print(f"\n  [STT] Error: {e}")
            # Fallback a teclado para este turno
            return self._listen_keyboard("[VOS]: ")

    @property
    def is_ready(self) -> bool:
        return self._ready or self.use_keyboard

    def get_info(self) -> dict:
        """Info de diagnóstico."""
        return {
            "engine": "moonshine" if not self.use_keyboard else "keyboard",
            "language": self.language,
            "ready": self.is_ready,
            "model_path": str(self._model_path) if self._model_path else None,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[STTEngine] = None


def get_stt(language: str = "es", use_keyboard: bool = False) -> STTEngine:
    """Retorna la instancia global del STT engine."""
    global _engine
    if _engine is None:
        _engine = STTEngine(language=language, use_keyboard=use_keyboard)
    return _engine


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--keyboard", action="store_true",
                        help="Usar teclado en vez de micrófono")
    parser.add_argument("--lang", default="es", choices=["es", "en"],
                        help="Idioma del STT")
    args = parser.parse_args()

    engine = STTEngine(language=args.lang, use_keyboard=args.keyboard)
    print(f"\nSTT Engine: {engine.get_info()}")
    print("Di algo (o escribí si estás en modo teclado)...\n")

    while True:
        text = engine.listen()
        if not text:
            continue
        if text.lower() in ("salir", "exit", "quit"):
            print("Chau!")
            break
        print(f"  → Transcripción: {text}\n")