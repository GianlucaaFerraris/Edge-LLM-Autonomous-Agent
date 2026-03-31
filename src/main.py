"""
main.py — Punto de entrada del asistente edge (v3 — Voice).

Cambios respecto a v2:
  - VoiceIO reemplaza input()/print() con Moonshine STT + Piper TTS
  - set_mode() en VoiceIO al entrar/salir de cada sesión (cambia idioma STT + voz TTS)
  - TutorSession.listen/speak hooks inyectados con VoiceIO
  - engineering_session._chat_stream conectado a TTS
  - Flag --keyboard / --print-only para desarrollo sin hardware de audio

Flujo de voz por turno:
  Mic → VAD (Moonshine) → texto → classify_intent → LLM → texto → Piper → Speaker
       ~50ms                       ~1-3s                        ~20ms

Uso:
  python main.py                    # modo voz completo
  python main.py --keyboard         # STT por teclado, TTS por audio
  python main.py --print-only       # todo por texto (dev mode)
  python main.py --keyboard --print-only  # modo texto puro (como v2)
"""

import os
import sys
import datetime
import time
import argparse

import requests
from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_FILE = os.path.abspath(__file__)
_SRC_DIR   = os.path.dirname(_THIS_FILE)
_ROOT_DIR  = os.path.dirname(_SRC_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Agregar el directorio de voice modules al path
_VOICE_DIR = os.path.join(_SRC_DIR, "voice")
if _VOICE_DIR not in sys.path:
    sys.path.insert(0, _VOICE_DIR)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.orchestrator.intent_classifier import (
    classify_idle, classify_english, classify_engineering,
    confirm_intent, IntentResult,
)
from src.orchestrator.orchestrator import (
    generate_greeting, generate_clarification, generate_return_prompt,
    _resolve_model,
)
from src.orchestrator.context_manager import ContextManager
from src.agent.agent_session import AgentSession
from src.agent import reminder_manager as reminders
from src.english.language_tool_server import ensure_running, ensure_stopped

# Voice I/O (usar import absoluto de paquete para robustez)
from src.voice.voice_io import VoiceIO

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# Global VoiceIO instance — inicializada en main()
vio: VoiceIO = None


# ── I/O hooks (delegados a VoiceIO) ──────────────────────────────────────────

def listen() -> str:
    """Hook global de input — delegado a VoiceIO."""
    return vio.listen()


def speak(text: str) -> None:
    """Hook global de output — delegado a VoiceIO con print visual."""
    vio.speak_and_print(text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_ollama() -> bool:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _show_alerts() -> None:
    alerts = reminders.pop_alerts()
    if not alerts:
        return
    print("\n" + "⏰" * 20)
    for a in alerts:
        dt = datetime.datetime.fromisoformat(a["remind_at"])
        print(f"  ⏰  {a['title']} — {dt.strftime('%H:%M')}")
    print("⏰" * 20 + "\n")
    # También hablar la alerta
    for a in alerts:
        vio.speak(f"Recordatorio: {a['title']}", force_voice="agent")


def _handle_low_confidence(intent: IntentResult) -> bool:
    if intent.confidence != "low" or not intent.question:
        return True
    speak(intent.question)
    answer = listen()
    return confirm_intent(intent.question, answer)


def _handle_agent_interrupt(user_text: str, active_mode: str,
                            ctx: ContextManager) -> str:
    agent = AgentSession()
    # Inyectar VoiceIO en el agente
    agent.listen = listen
    agent.speak = lambda text: speak(text)

    result = agent.run_turn(user_text, return_mode=active_mode)
    speak(result["text"])

    return_q = generate_return_prompt(active_mode, client, _resolve_model())
    speak(return_q)
    answer = listen()

    ans = answer.lower().strip()
    if ans in ("sí", "si", "dale", "yes", "claro", ""):
        return "resume"
    return "exit"


# ── MODO ENGLISH ──────────────────────────────────────────────────────────────

def _ensure_lt_up() -> bool:
    print("  [LT] Verificando servidor LanguageTool...")
    return ensure_running()


def _ensure_lt_down() -> None:
    print("  [LT] Apagando servidor LanguageTool...")
    ensure_stopped()


def _run_english(ctx: ContextManager) -> str:
    from src.english.tutor_session import (
        TutorSession, check_errors, generate_feedback,
        generate_opening, extract_proposed_topic, load_topics,
    )

    # ── Configurar VoiceIO para modo inglés ──
    vio.set_mode("english")
    ctx.set_active("english")
    eng_ctx = ctx.get("english")

    lt_ok = _ensure_lt_up()

    if eng_ctx.session is None:
        eng_ctx.session = TutorSession()
        eng_ctx.session.lt_ok = lt_ok

        # ── Inyectar VoiceIO en la sesión ──
        # speak() del tutor necesita auto-detect para alternar es/en
        eng_ctx.session.speak = lambda text, total=None: _tutor_speak(text, total)
        eng_ctx.session.listen = listen

        eng_ctx.session._ask_topic_preference()
    else:
        speak("Retomando el tutor de inglés donde lo dejaste.")
        eng_ctx.session.lt_ok = lt_ok
        eng_ctx.session.speak = lambda text, total=None: _tutor_speak(text, total)
        eng_ctx.session.listen = listen

    session = eng_ctx.session
    session._start_topic()

    try:
        while True:
            _show_alerts()
            student_text = session.listen()

            if not student_text:
                continue

            intent = classify_english(student_text, current_topic=session.topic)
            print(f"  [intent: {intent.action} (conf={intent.confidence})]")

            if intent.confidence == "low" and intent.question:
                if not _handle_low_confidence(intent):
                    intent.action = "respond"

            if intent.action == "exit_app":
                session._save_log()
                _ensure_lt_down()
                vio.set_mode("idle")
                ctx.set_active("idle")
                return "exit_app"

            if intent.action == "exit_mode":
                speak("Cerrando el tutor de inglés.")
                session._save_log()
                _ensure_lt_down()
                vio.set_mode("idle")
                ctx.set_active("idle")
                return "idle"

            if intent.action == "switch_engineering":
                speak("Dale, pasamos al tutor de ingeniería.")
                session._save_log()
                _ensure_lt_down()
                vio.set_mode("idle")
                ctx.set_active("idle")
                return "engineering"

            if intent.action == "agent":
                # Cambiar STT a español temporalmente para el agente
                vio.set_mode("agent")
                result = _handle_agent_interrupt(
                    student_text, "english", ctx
                )
                vio.set_mode("english")  # volver a inglés

                if result == "exit":
                    session._save_log()
                    _ensure_lt_down()
                    vio.set_mode("idle")
                    ctx.set_active("idle")
                    return "idle"
                session._start_topic()
                continue

            if intent.action == "change_topic":
                session._pick_new_topic()
                session._start_topic()
                continue

            if intent.action == "propose_topic":
                proposed = intent.topic
                if not proposed or len(proposed) < 2:
                    proposed = extract_proposed_topic(student_text)
                session.topic = proposed
                session.used_topics.add(session.topic)
                session.history = []
                print(f"  [INFO] Tema propuesto: {session.topic}")
                session._start_topic()
                continue

            # respond: turno normal
            if session.lt_ok and _is_likely_english(student_text):
                errors = check_errors(student_text)
            else:
                errors = []
                if session.lt_ok and not _is_likely_english(student_text):
                    print("  [LT] Texto no parece inglés, salteo chequeo.")

            session._run_normal_turn_with_errors(student_text, errors)

    except Exception as e:
        print(f"  [ERROR] English session: {e}")
        _ensure_lt_down()
        vio.set_mode("idle")
        ctx.set_active("idle")
        return "idle"


def _tutor_speak(text: str, total: float = None) -> None:
    """
    Hook de speak para el TutorSession.
    Auto-detecta idioma del texto para elegir voz (mujer ES / mujer EN).
    """
    prefix = "[TUTOR]"
    print(f"\n{prefix}: {text}")
    if total is not None:
        print(f"  ⏱  Total={total}s")
    print()

    # VoiceIO en modo "english" auto-detecta es/en
    vio.speak(text)


def _is_likely_english(text: str) -> bool:
    """Heurística rápida para detectar si el texto es inglés."""
    es_markers = {
        "quiero", "puedo", "tengo", "necesito", "vamos", "dame",
        "hacé", "decime", "pasame", "cambiemos", "estoy", "puede",
        "sobre", "porque", "también", "ahora", "después", "pero",
        "como", "para", "donde", "cuando", "todos", "mejor",
        "bueno", "gracias", "hola", "chau", "dale", "algo",
        "creo", "sería", "podría", "debería", "muy", "más",
    }
    words = set(text.lower().split())
    es_count = len(words & es_markers)
    if es_count >= 2:
        return False
    if es_count >= 1 and len(words) < 8:
        return False
    return True


# ── MODO ENGINEERING ──────────────────────────────────────────────────────────

def _run_engineering(ctx: ContextManager) -> str:
    from src.engineering.engineering_session import (
        SYSTEM_ENGINEERING, SYSTEM_ENGINEERING_WITH_CONTEXT,
        _resolve_model as eng_resolve, _chat_stream, _chat_stream_iter,
        _load_rag, _build_rag_prompt,
    )

    # ── Configurar VoiceIO para modo ingeniería ──
    vio.set_mode("engineering")
    ctx.set_active("engineering")
    eng_ctx = ctx.get("engineering")

    model = eng_resolve()
    print(f"\n{'═'*60}")
    print(f"  TUTOR DE INGENIERÍA  |  Modelo: {model}")
    print(f"  'salir' → menú  |  'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")

    rag_engine = _load_rag()

    if not eng_ctx.history:
        speak("Hola, soy tu tutor de ingeniería. ¿Sobre qué concepto querés hablar?")
    else:
        speak("Retomando donde lo dejamos. ¿Qué más querés explorar?")

    while True:
        _show_alerts()
        user_text = listen()

        if not user_text:
            continue
        if user_text.lower() in ("limpiar", "clear"):
            eng_ctx.history = []
            print("  [INFO] Contexto limpiado.\n")
            continue

        intent = classify_engineering(user_text)
        print(f"  [intent: {intent.action} (conf={intent.confidence})]")

        if intent.confidence == "low" and intent.question:
            if not _handle_low_confidence(intent):
                intent.action = "respond"

        if intent.action == "exit_app":
            vio.set_mode("idle")
            ctx.set_active("idle")
            return "exit_app"

        if intent.action == "exit_mode":
            vio.set_mode("idle")
            ctx.set_active("idle")
            return "idle"

        if intent.action == "switch_english":
            speak("Dale, pasamos al tutor de inglés.")
            vio.set_mode("idle")
            ctx.set_active("idle")
            return "english"

        if intent.action == "agent":
            vio.set_mode("agent")
            result = _handle_agent_interrupt(user_text, "engineering", ctx)
            vio.set_mode("engineering")

            if result == "exit":
                vio.set_mode("idle")
                ctx.set_active("idle")
                return "idle"
            speak("Dale, ¿qué más querés explorar?")
            continue

        # respond: turno normal CON RAG
        eng_ctx.history.append({"role": "user", "content": user_text})

        used_rag = False
        rag_context = ""

        if rag_engine is not None:
            import time as _time
            t_rag = _time.perf_counter()
            result = rag_engine.query_with_domain_check(user_text)
            t_rag = round(_time.perf_counter() - t_rag, 2)

            if result.relevant:
                print(f"  [RAG] {len(result.chunks)} fragmento(s) "
                      f"(score={result.top_score:.2f}, "
                      f"domain={result.domain_score:.2f}, {t_rag}s)")
                rag_context = result.context
                used_rag = True
                speak("Un momento, déjame revisar en mis libros...")

        if used_rag:
            messages = _build_rag_prompt(
                user_text, rag_context, eng_ctx.history[:-1]
            )
        else:
            messages = (
                [{"role": "system", "content": SYSTEM_ENGINEERING}]
                + eng_ctx.history[-8:]
            )

        # Streaming del LLM con TTS por oración simultáneo
        print("\n[INGENIERO]: ", end="", flush=True)

        token_iter  = (tok for tok, _, _ in _chat_stream_iter(
                            messages, temperature=0.3, max_tokens=600))
        response = vio.speak_stream(token_iter)

        # Métricas aproximadas (speak_stream consume el iter)
        print(f"\n  ⏱  Total={round(0.0, 2)}s\n")

        eng_ctx.history.append({"role": "assistant", "content": response})
        if len(eng_ctx.history) > 12:
            eng_ctx.history = eng_ctx.history[-12:]


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    global vio

    # ── Argumentos CLI ──
    parser = argparse.ArgumentParser(description="Agenty — Asistente Edge v3")
    parser.add_argument("--keyboard", action="store_true",
                        help="Usar teclado como STT (sin micrófono)")
    parser.add_argument("--print-only", action="store_true",
                        help="Solo texto como TTS (sin audio)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  AGENTY — Asistente Edge v3 (Voice)")
    print(f"  {datetime.datetime.now().strftime('%A %d/%m/%Y %H:%M')}")
    print("═" * 60)

    if not _check_ollama():
        print("\n[ERROR] Ollama no está corriendo.")
        print("  Arrancalo con: ollama serve")
        sys.exit(1)

    model = _resolve_model()
    print(f"[OK] Ollama conectado | Modelo: {model}\n")

    # ── Inicializar Voice I/O ──
    vio = VoiceIO(
        use_keyboard=args.keyboard,
        use_print=args.print_only,
    )

    reminders.start_scheduler(interval_minutes=30)
    ctx = ContextManager()

    greeting = generate_greeting(client, model)
    speak(greeting)

    while True:
        _show_alerts()

        user_text = listen()
        if not user_text:
            continue

        intent = classify_idle(user_text)
        print(f"  [intent: {intent.action}]")

        if intent.action == "exit_app":
            speak("¡Hasta luego!")
            ensure_stopped()
            vio.stt.shutdown()
            break

        if intent.action == "unclear":
            clarification = generate_clarification(user_text, client, model)
            speak(clarification)
            user_text2 = listen()
            intent = classify_idle(user_text2)
            if intent.action == "unclear":
                speak("No pude entender qué querés hacer. "
                      "¿Practicar inglés, consultar algo técnico, "
                      "o una tarea del agente?")
                continue
            user_text = user_text2

        # ── Routing por modo ──
        next_mode = intent.action

        while next_mode and next_mode != "idle":

            if next_mode == "english":
                speak("¡Perfecto, arrancamos con el tutor de inglés!")
                next_mode = _run_english(ctx)

            elif next_mode == "engineering":
                speak("Dale, te conecto con el tutor de ingeniería.")
                next_mode = _run_engineering(ctx)

            elif next_mode == "agent":
                vio.set_mode("agent")
                agent = AgentSession()
                agent.listen = listen
                agent.speak = lambda text: speak(text)
                result = agent.run_turn(user_text, return_mode=None)
                speak(result["text"])
                vio.set_mode("idle")
                next_mode = None

            elif next_mode == "exit_app":
                speak("¡Hasta luego!")
                ensure_stopped()
                vio.stt.shutdown()
                return

            else:
                next_mode = None

        print()


if __name__ == "__main__":
    main()