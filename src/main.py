"""
main.py — Punto de entrada del asistente edge (v2).

Cambios respecto a v1:
  - Clasificación unificada: un solo LLM call por turno decide el intent.
  - LanguageTool lifecycle limpio: se levanta al entrar a english,
    se apaga al salir (sin importar a dónde vaya).
  - RAG integrado en el flujo de engineering desde main.
  - Detección de idioma antes de pasar texto a LanguageTool.
  - Las sesiones de tutor no se destruyen al hacer switch (se pausan).

Flujo:
  1. Inicia scheduler de recordatorios
  2. Saluda y detecta intent (classify_idle)
  3. Lanza el modo correspondiente
  4. Dentro de cada modo, classify_{mode} decide en un solo paso:
     - respond / change_topic / propose_topic (intents internos)
     - switch_X (cambio de modo)
     - agent (tool liviana)
     - exit_mode / exit_app
"""

import os
import sys
import datetime
import time

import requests
from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_FILE = os.path.abspath(__file__)
_SRC_DIR   = os.path.dirname(_THIS_FILE)
_ROOT_DIR  = os.path.dirname(_SRC_DIR)

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

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

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


# ── I/O hooks ─────────────────────────────────────────────────────────────────

def listen() -> str:
    try:
        return input("[VOS]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "salir"


def speak(text: str) -> None:
    print(f"\n[ASISTENTE]: {text}\n")


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


def _handle_low_confidence(intent: IntentResult) -> bool:
    """
    Si el clasificador tiene baja confianza, pregunta al usuario.
    Retorna True si el usuario confirma, False si niega.
    """
    if intent.confidence != "low" or not intent.question:
        return True  # alta confianza → proceder
    speak(intent.question)
    answer = listen()
    return confirm_intent(intent.question, answer)


def _handle_agent_interrupt(user_text: str, active_mode: str,
                            ctx: ContextManager) -> str:
    """
    Ejecuta un turno del agente desde un modo activo (inglés/ingeniería).
    search_web ya es inline — ya no genera interrupción de sesión.
 
    Retorna:
        "resume"  → volver al modo activo
        "exit"    → salir al menú principal
    """
    agent = AgentSession()
    result = agent.run_turn(user_text, return_mode=active_mode)
 
    # Mostrar la respuesta del agente (incluye síntesis de búsqueda si la hubo)
    speak(result["text"])
 
    # Preguntar si el usuario quiere retomar el modo anterior
    return_q = generate_return_prompt(active_mode, client, _resolve_model())
    speak(return_q)
    answer = listen()
 
    ans = answer.lower().strip()
    if ans in ("sí", "si", "dale", "yes", "claro", ""):
        return "resume"
    return "exit"


# ── MODO ENGLISH ──────────────────────────────────────────────────────────────

def _ensure_lt_up() -> bool:
    """Levanta LanguageTool si no está corriendo."""
    print("  [LT] Verificando servidor LanguageTool...")
    return ensure_running()


def _ensure_lt_down() -> None:
    """Apaga LanguageTool si estaba corriendo."""
    print("  [LT] Apagando servidor LanguageTool...")
    ensure_stopped()


def _run_english(ctx: ContextManager) -> str:
    """
    Ejecuta el tutor de inglés.
    Retorna: "idle", "engineering", "exit_app"
    """
    from src.english.tutor_session import (
        TutorSession, check_errors, generate_feedback,
        generate_opening, extract_proposed_topic, load_topics,
    )

    ctx.set_active("english")
    eng_ctx = ctx.get("english")

    # Levantar LanguageTool
    lt_ok = _ensure_lt_up()

    # Crear o retomar sesión
    if eng_ctx.session is None:
        eng_ctx.session = TutorSession()
        eng_ctx.session.lt_ok = lt_ok
        eng_ctx.session._ask_topic_preference()
    else:
        speak("Retomando el tutor de inglés donde lo dejaste.")
        eng_ctx.session.lt_ok = lt_ok

    session = eng_ctx.session
    session._start_topic()

    try:
        while True:
            _show_alerts()
            student_text = session.listen()

            if not student_text:
                continue

            # ── Clasificación UNIFICADA ──
            intent = classify_english(student_text, current_topic=session.topic)
            print(f"  [intent: {intent.action} (conf={intent.confidence})]")

            # Baja confianza → confirmar
            if intent.confidence == "low" and intent.question:
                if not _handle_low_confidence(intent):
                    # Usuario negó → tratar como respuesta normal
                    intent.action = "respond"

            # ── Dispatch por acción ──
            if intent.action == "exit_app":
                session._save_log()
                _ensure_lt_down()
                ctx.set_active("idle")
                return "exit_app"

            if intent.action == "exit_mode":
                speak("Cerrando el tutor de inglés.")
                session._save_log()
                _ensure_lt_down()
                ctx.set_active("idle")
                return "idle"

            if intent.action == "switch_engineering":
                speak("Dale, pasamos al tutor de ingeniería.")
                session._save_log()
                # No destruimos la sesión, solo la pausamos
                _ensure_lt_down()
                ctx.set_active("idle")
                return "engineering"

            if intent.action == "agent":
                result = _handle_agent_interrupt(
                    student_text, "english", ctx
                )
                if result == "web_search":
                    session._save_log()
                    _ensure_lt_down()
                    return "idle"
                if result == "exit":
                    session._save_log()
                    _ensure_lt_down()
                    ctx.set_active("idle")
                    return "idle"
                # resume → seguir con el tutor
                session._start_topic()
                continue

            if intent.action == "change_topic":
                session._pick_new_topic()
                session._start_topic()
                continue

            if intent.action == "propose_topic":
                # Extraer tema del clasificador o del texto
                proposed = intent.topic
                if not proposed or len(proposed) < 2:
                    proposed = extract_proposed_topic(student_text)
                session.topic = proposed
                session.used_topics.add(session.topic)
                session.history = []
                print(f"  [INFO] Tema propuesto: {session.topic}")
                session._start_topic()
                continue

            # ── respond: turno normal ──
            # Detectar idioma antes de mandar a LanguageTool
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
        ctx.set_active("idle")
        return "idle"


def _is_likely_english(text: str) -> bool:
    """
    Heurística rápida para detectar si el texto es inglés.
    Evita mandar español a LanguageTool (que lo "corrige" como inglés).
    Sin LLM call — puramente estadístico para no agregar latencia.
    """
    # Palabras comunes en español que rara vez aparecen en inglés
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
    # Si >= 2 palabras son marcadores de español, probablemente no es inglés
    if es_count >= 2:
        return False
    # Si >= 1 marcador y el texto es corto (< 8 palabras), probablemente español
    if es_count >= 1 and len(words) < 8:
        return False
    return True


# ── MODO ENGINEERING ──────────────────────────────────────────────────────────

def _run_engineering(ctx: ContextManager) -> str:
    """
    Ejecuta el tutor de ingeniería con RAG.
    Retorna: "idle", "english", "exit_app"
    """
    from src.engineering.engineering_session import (
        SYSTEM_ENGINEERING, SYSTEM_ENGINEERING_WITH_CONTEXT,
        _resolve_model as eng_resolve, _chat_stream, _load_rag,
        _build_rag_prompt,
    )

    ctx.set_active("engineering")
    eng_ctx = ctx.get("engineering")

    model = eng_resolve()
    print(f"\n{'═'*60}")
    print(f"  TUTOR DE INGENIERÍA  |  Modelo: {model}")
    print(f"  'salir' → menú  |  'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")

    # Cargar RAG una sola vez por sesión
    rag_engine = _load_rag()

    if not eng_ctx.history:
        speak("Hola, soy tu tutor de ingeniería. ¿Sobre qué concepto querés hablar?")
    else:
        speak("Retomando donde lo dejamos. ¿Qué más querés explorar?")

    while True:
        _show_alerts()

        try:
            user_text = input("[VOS]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            ctx.set_active("idle")
            return "idle"

        if not user_text:
            continue
        if user_text.lower() in ("limpiar", "clear"):
            eng_ctx.history = []
            print("  [INFO] Contexto limpiado.\n")
            continue

        # ── Clasificación UNIFICADA ──
        intent = classify_engineering(user_text)
        print(f"  [intent: {intent.action} (conf={intent.confidence})]")

        if intent.confidence == "low" and intent.question:
            if not _handle_low_confidence(intent):
                intent.action = "respond"

        # ── Dispatch ──
        if intent.action == "exit_app":
            ctx.set_active("idle")
            return "exit_app"

        if intent.action == "exit_mode":
            ctx.set_active("idle")
            return "idle"

        if intent.action == "switch_english":
            speak("Dale, pasamos al tutor de inglés.")
            ctx.set_active("idle")
            return "english"

        if intent.action == "agent":
            result = _handle_agent_interrupt(user_text, "engineering", ctx)
            if result == "web_search":
                return "idle"
            if result == "exit":
                ctx.set_active("idle")
                return "idle"
            speak("Dale, ¿qué más querés explorar?")
            continue

        # ── respond: turno normal CON RAG ──
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
                print("\n[INGENIERO]: ", end="", flush=True)
                print("Un momento, déjame revisar en mis libros... ",
                      end="", flush=True)

        if not used_rag:
            print("\n[INGENIERO]: ", end="", flush=True)

        if used_rag:
            messages = _build_rag_prompt(
                user_text, rag_context, eng_ctx.history[:-1]
            )
        else:
            messages = (
                [{"role": "system", "content": SYSTEM_ENGINEERING}]
                + eng_ctx.history[-8:]
            )

        response, ttft, total = _chat_stream(messages, temperature=0.3,
                                              max_tokens=600)

        rag_tag = (f" [RAG score={result.top_score:.2f}]"
                   if used_rag else "")
        print(f"\n  ⏱  TTFT={ttft}s | Total={total}s{rag_tag}\n")

        eng_ctx.history.append({"role": "assistant", "content": response})
        if len(eng_ctx.history) > 12:
            eng_ctx.history = eng_ctx.history[-12:]


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  AGENTY — Asistente Edge v2")
    print(f"  {datetime.datetime.now().strftime('%A %d/%m/%Y %H:%M')}")
    print("═" * 60)

    if not _check_ollama():
        print("\n[ERROR] Ollama no está corriendo.")
        print("  Arrancalo con: ollama serve")
        sys.exit(1)

    model = _resolve_model()
    print(f"[OK] Ollama conectado | Modelo: {model}\n")

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
        next_mode = intent.action  # "english", "engineering", "agent"

        while next_mode and next_mode != "idle":

            if next_mode == "english":
                speak("¡Perfecto, arrancamos con el tutor de inglés!")
                next_mode = _run_english(ctx)

            elif next_mode == "engineering":
                speak("Dale, te conecto con el tutor de ingeniería.")
                next_mode = _run_engineering(ctx)

            elif next_mode == "agent":
                agent = AgentSession()
                result = agent.run_turn(user_text, return_mode=None)
                # search_web es inline: result siempre tiene action="return_to_mode"
                # y text contiene la respuesta sintetizada con los resultados de búsqueda.
                speak(result["text"])
                next_mode = None

            elif next_mode == "exit_app":
                speak("¡Hasta luego!")
                ensure_stopped()
                return

            else:
                next_mode = None

        print()


if __name__ == "__main__":
    main()