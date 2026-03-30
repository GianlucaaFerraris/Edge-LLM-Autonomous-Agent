"""
voice_io.py — Facade de voz que unifica STT (Moonshine) + TTS (Piper).

Esta es la interfaz que main.py y las sesiones usan directamente.
Reemplaza input() → listen() y print() → speak() con voz real,
manteniendo un fallback a texto si el hardware no está disponible.

Responsabilidades:
  - Routing automático de voz según el modo activo
  - Cambio de idioma del STT al entrar/salir del tutor de inglés
  - Detección de idioma del texto para elegir la voz TTS correcta
  - Indicadores visuales de estado (escuchando, procesando, hablando)

Uso:
    from voice_io import VoiceIO

    vio = VoiceIO()
    vio.set_mode("english")  # configura voces y STT para tutor de inglés

    text = vio.listen()       # escucha con Moonshine
    vio.speak("Great job!")   # habla con Piper (voz de mujer en inglés)

Modos soportados:
    "idle"        → STT español, TTS agente
    "agent"       → STT español, TTS agente
    "english"     → STT inglés, TTS mujer (auto-detect es/en)
    "engineering" → STT español, TTS hombre sabio (auto-detect es/en)
"""

import os
import sys
from typing import Optional

from stt_engine import STTEngine, get_stt
from tts_engine import TTSEngine, get_tts, detect_language

# ── Configuración de voz por modo ────────────────────────────────────────────

# Cada modo define:
#   stt_lang:        idioma para Moonshine (cambia el modelo de STT)
#   tts_voice_es:    voz Piper para texto en español
#   tts_voice_en:    voz Piper para texto en inglés
#   auto_detect_tts: si True, detecta idioma del texto para elegir voz
#   prefix:          prefijo visual en modo print

MODE_CONFIG = {
    "idle": {
        "stt_lang": "es",
        "tts_voice_es": "agent",
        "tts_voice_en": "agent",  # idle siempre habla español
        "auto_detect_tts": False,
        "prefix": "[ASISTENTE]",
    },
    "agent": {
        "stt_lang": "es",
        "tts_voice_es": "agent",
        "tts_voice_en": "agent",
        "auto_detect_tts": False,
        "prefix": "[AGENTE]",
    },
    "english": {
        "stt_lang": "en",  # el tutor espera respuestas en inglés
        "tts_voice_es": "english_es",
        "tts_voice_en": "english_en",
        "auto_detect_tts": True,  # intros en español, feedback en inglés
        "prefix": "[TUTOR]",
    },
    "engineering": {
        "stt_lang": "es",  # la mayoría de consultas son en español
        "tts_voice_es": "engineering_es",
        "tts_voice_en": "engineering_en",
        "auto_detect_tts": True,  # puede responder en inglés si le preguntan en inglés
        "prefix": "[INGENIERO]",
    },
}


class VoiceIO:
    """
    Interfaz unificada de voz para Agenty.
    
    Encapsula STT (Moonshine) y TTS (Piper) con routing automático
    de idioma y voz según el modo activo.
    
    Diseño:
      - set_mode() configura el STT language y las voces TTS
      - listen() siempre usa el idioma STT del modo activo
      - speak() auto-detecta el idioma del texto para elegir la voz correcta
        (solo en modos con auto_detect_tts=True)
      - Fallback graceful: si STT/TTS falla, cae a input()/print()
    """

    def __init__(self, use_keyboard: bool = False, use_print: bool = False):
        """
        Args:
            use_keyboard: Forzar teclado como STT (para desarrollo).
            use_print: Forzar texto como TTS (para desarrollo sin audio).
        """
        self.use_keyboard = use_keyboard
        self.use_print = use_print
        self._mode = "idle"
        self._mode_cfg = MODE_CONFIG["idle"]

        # Inicializar engines
        print("  Inicializando Voice I/O...")
        self.stt = STTEngine(
            language=self._mode_cfg["stt_lang"],
            use_keyboard=use_keyboard,
        )
        self.tts = TTSEngine(use_print=use_print)
        print(f"  [VoiceIO] STT: {self.stt.get_info()['engine']} | "
              f"TTS: {self.tts.get_info()['engine']}")

    def set_mode(self, mode: str) -> None:
        """
        Configura el modo activo. Cambia idioma del STT y voces del TTS.
        
        Esto se llama al entrar/salir de cada sesión:
          - Entrar a english → STT cambia a "en"
          - Salir de english → STT vuelve a "es"
        
        El cambio de modelo de Moonshine toma ~500ms, así que solo
        se hace cuando realmente cambia el idioma.
        """
        if mode not in MODE_CONFIG:
            mode = "idle"

        if mode == self._mode:
            return

        old_lang = self._mode_cfg["stt_lang"]
        self._mode = mode
        self._mode_cfg = MODE_CONFIG[mode]
        new_lang = self._mode_cfg["stt_lang"]

        # Solo recargar modelo de STT si cambia el idioma
        if new_lang != old_lang:
            self.stt.set_language(new_lang)
            print(f"  [VoiceIO] STT idioma: {old_lang} → {new_lang}")

        print(f"  [VoiceIO] Modo: {mode}")

    def listen(self, prompt: str = None) -> str:
        """
        Escucha al usuario y retorna texto transcripto.
        Usa el idioma STT configurado para el modo activo.
        
        Args:
            prompt: Prompt visual (solo para modo teclado).
        
        Returns:
            Texto transcripto, o "salir" si se interrumpe.
        """
        if prompt is None:
            prompt = "[VOS]: "
        return self.stt.listen(prompt=prompt)

    def speak_stream(self, token_iter, force_voice: str = None) -> str:
        """
        TTS en streaming: habla oración por oración mientras el LLM genera.
        Retorna el texto completo acumulado.

        Args:
            token_iter:  Iterable de tokens (deltas del stream del LLM).
            force_voice: Voice key explícita (overrides auto-detection).
        """
        cfg = self._mode_cfg
        voice_key = force_voice or cfg["tts_voice_es"]
        auto_detect = cfg["auto_detect_tts"] and not force_voice

        return self.tts.speak_stream(
            token_iter,
            voice_key=voice_key,
            auto_detect_lang=auto_detect,
        )

    def speak(self, text: str, force_voice: str = None) -> None:
        """
        Sintetiza y reproduce texto como audio.
        
        Selecciona la voz automáticamente según:
          1. force_voice si se especifica
          2. Auto-detección de idioma del texto (si el modo lo permite)
          3. Voz default del modo en español
        
        Args:
            text: Texto a hablar.
            force_voice: Voice key explícita (overrides auto-detection).
        """
        if not text or not text.strip():
            return

        if force_voice:
            self.tts.speak(text, voice_key=force_voice)
            return

        # Determinar voz según modo y contenido
        cfg = self._mode_cfg

        if cfg["auto_detect_tts"]:
            lang = detect_language(text)
            voice_key = cfg["tts_voice_en"] if lang == "en" else cfg["tts_voice_es"]
        else:
            voice_key = cfg["tts_voice_es"]

        self.tts.speak(text, voice_key=voice_key)

    def speak_and_print(self, text: str, force_voice: str = None) -> None:
        """
        Habla Y muestra texto en pantalla.
        Útil para desarrollo: ver lo que dice mientras se escucha.
        """
        prefix = self._mode_cfg.get("prefix", "[ASISTENTE]")
        print(f"\n{prefix}: {text}\n")
        if not self.use_print:
            self.speak(text, force_voice=force_voice)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_ready(self) -> bool:
        return self.stt.is_ready and self.tts.is_ready

    def get_info(self) -> dict:
        return {
            "mode": self._mode,
            "stt": self.stt.get_info(),
            "tts": self.tts.get_info(),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_vio: Optional[VoiceIO] = None


def get_voice_io(use_keyboard: bool = False,
                 use_print: bool = False) -> VoiceIO:
    """Retorna la instancia global de VoiceIO."""
    global _vio
    if _vio is None:
        _vio = VoiceIO(use_keyboard=use_keyboard, use_print=use_print)
    return _vio


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test de VoiceIO")
    parser.add_argument("--keyboard", action="store_true",
                        help="Usar teclado como STT")
    parser.add_argument("--print-only", action="store_true",
                        help="Solo imprimir, sin audio TTS")
    args = parser.parse_args()

    vio = VoiceIO(use_keyboard=args.keyboard, use_print=args.print_only)
    print(f"\nVoiceIO: {vio.get_info()}\n")

    # Test: ciclo de modos
    modes = ["idle", "agent", "english", "engineering"]
    test_phrases = {
        "idle": "Hola, ¿qué querés hacer?",
        "agent": "Listo, agregué la tarea a tu lista.",
        "english": "That's a great answer! You used the present perfect correctly.",
        "engineering": "La retropropagación calcula el gradiente usando la regla de la cadena.",
    }

    for mode in modes:
        print(f"\n{'='*40}")
        print(f"  Modo: {mode}")
        print(f"{'='*40}")
        vio.set_mode(mode)
        vio.speak(test_phrases[mode])

    # Test interactivo
    print(f"\n{'='*40}")
    print("  Test interactivo")
    print(f"{'='*40}")
    vio.set_mode("idle")
    vio.speak("Hola, decime algo y te respondo con la voz del modo que elijas.")

    while True:
        text = vio.listen()
        if not text:
            continue
        if text.lower() in ("salir", "exit"):
            vio.speak("¡Chau!")
            break

        # Cambiar modo según contenido
        if "inglés" in text.lower() or "english" in text.lower():
            vio.set_mode("english")
            vio.speak("Switching to English mode. How are you today?")
        elif "ingeniería" in text.lower() or "técnico" in text.lower():
            vio.set_mode("engineering")
            vio.speak("Dale, modo ingeniería activado.")
        else:
            vio.speak(f"Dijiste: {text}")