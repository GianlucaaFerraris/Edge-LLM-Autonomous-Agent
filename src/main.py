"""
main.py — Punto de entrada del asistente edge.

Arranca siempre al prender la SBC.
Orquesta los tres modos y el agente transversal.

Flujo:
  1. Inicia scheduler de recordatorios
  2. Saluda y detecta intent
  3. Lanza el modo correspondiente
  4. Desde cualquier modo, detecta interrupciones de agente
  5. Tools livianas → ejecuta y vuelve al modo anterior
  6. search_web → confirma, entra en modo búsqueda (sin retorno)

Uso:
    python main.py              # desde src/
    python -m src.main          # desde raíz del proyecto
"""

import os
import sys
import datetime

import requests
from openai import OpenAI

# Ajustar path para imports relativos
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SRC_DIR))

from src.orchestrator.orchestrator import (
    detect_intent,
    detect_intent_from_active_mode,
    generate_greeting,
    generate_clarification,
    generate_return_prompt,
    _resolve_model,
)
from src.orchestrator.context_manager import ContextManager
from src.agent.agent_session import AgentSession
from src.agent import reminder_manager as reminders

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# ── Lazy imports de modos (para no cargar todo al inicio) ─────────────────────
_english_session_cls     = None
_engineering_session_fn  = None


def _get_english_session():
    global _english_session_cls
    if _english_session_cls is None:
        from src.english.tutor_session import TutorSession
        _english_session_cls = TutorSession
    return _english_session_cls


def _get_engineering_fn():
    global _engineering_session_fn
    if _engineering_session_fn is None:
        from src.engineering.engineering_session import run_engineering_session
        _engineering_session_fn = run_engineering_session
    return _engineering_session_fn


# ── I/O hooks (reemplazar por Whisper/TTS en producción) ─────────────────────

def listen() -> str:
    try:
        return input("[VOS]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "salir"


def speak(text: str) -> None:
    print(f"\n[ASISTENTE]: {text}\n")


# ── Verificaciones de inicio ──────────────────────────────────────────────────

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


# ── Modos ─────────────────────────────────────────────────────────────────────

def _run_english_with_interrupts(ctx: ContextManager) -> None:
    """
    Corre el tutor de inglés con soporte de interrupciones de agente.
    Las interrupciones livianas ejecutan y retornan; search_web no retorna.
    """
    TutorSession = _get_english_session()
    ctx.set_active("english")
    eng_ctx = ctx.get("english")

    # Reusar sesión existente o crear nueva
    if eng_ctx.session is None:
        eng_ctx.session = TutorSession()
        # Preguntar tema y arrancar
        eng_ctx.session._ask_topic_preference()
        eng_ctx.session._start_topic()
    else:
        speak("Retomando el tutor de inglés donde lo dejaste.")
        eng_ctx.session._start_topic()

    agent = AgentSession()

    while True:
        _show_alerts()
        student_text = eng_ctx.session.listen()

        if not student_text:
            continue
        if student_text.lower() in ("salir", "exit", "quit", "menu"):
            speak("Cerrando el tutor de inglés.")
            eng_ctx.session = None
            ctx.set_active("idle")
            break

        # Detectar si es una interrupción de agente
        interruption = detect_intent_from_active_mode(student_text, "english")

        if interruption == "agent":
            result = agent.run_turn(student_text, return_mode="english")

            if result["action"] == "web_search":
                # search_web: confirmar abandono de sesión
                speak(
                    f"Para buscar en internet tengo que pausar el tutor de inglés y no vas a poder retomarlo. "
                    f"{result['text']}\n¿Querés continuar con la búsqueda?"
                )
                confirm = listen()
                if confirm.lower() in ("sí", "si", "dale", "yes"):
                    eng_ctx.session = None
                    ctx.set_active("idle")
                    agent._enter_web_mode(result["search_data"])
                    break
                else:
                    speak("Entendido, seguimos con el tutor de inglés.")
                    continue

            # Tool liviana ejecutada → mostrar resultado
            speak(result["text"])

            # Preguntar si quiere retomar el tutor
            return_q = generate_return_prompt("english", client, _resolve_model())
            speak(return_q)
            answer = listen()

            intent = detect_intent(answer)
            if intent == "english" or answer.lower() in ("sí", "si", "dale", "yes", "claro"):
                # Retomar tutor — generar nueva apertura con el tema actual
                eng_ctx.session._start_topic()
                continue
            elif intent == "agent":
                # Otra acción de agente
                continue
            else:
                # Salir del tutor
                eng_ctx.session = None
                ctx.set_active("idle")
                break
        else:
            # Turno normal del tutor de inglés
            from src.english.tutor_session import classify_intent, extract_proposed_topic
            intent_tutor = classify_intent(student_text)

            if intent_tutor == "exit":
                eng_ctx.session.speak(
                    "It was great practicing English with you! Keep it up!"
                )
                eng_ctx.session._save_log()
                eng_ctx.session = None
                ctx.set_active("idle")
                break

            if intent_tutor == "change_topic":
                eng_ctx.session._pick_new_topic()
                eng_ctx.session._start_topic()
                continue

            if intent_tutor == "propose_topic":
                eng_ctx.session.topic = extract_proposed_topic(student_text)
                eng_ctx.session.used_topics.add(eng_ctx.session.topic)
                eng_ctx.session.history = []
                eng_ctx.session._start_topic()
                continue

            # Respuesta normal
            from src.english.tutor_session import check_errors, generate_feedback
            import time as _time

            eng_ctx.session.turn_count += 1
            errors = check_errors(student_text) if eng_ctx.session.lt_ok else []

            if errors:
                print(f"  [LT] {len(errors)} error(s): " +
                      " | ".join(f"'{e['wrong']}' → '{e['correct']}'" for e in errors))
                eng_ctx.session.error_log.append({
                    "turn": eng_ctx.session.turn_count,
                    "topic": eng_ctx.session.topic,
                    "errors": errors,
                })
            else:
                print("  [LT] Sin errores.")

            t0 = _time.perf_counter()
            feedback = generate_feedback(
                eng_ctx.session.topic,
                eng_ctx.session.history,
                student_text,
                errors
            )
            elapsed = round(_time.perf_counter() - t0, 2)
            eng_ctx.session._add_history("user", student_text)
            eng_ctx.session._add_history("assistant", feedback)
            eng_ctx.session.speak(feedback, total=elapsed)


def _run_engineering_with_interrupts(ctx: ContextManager) -> None:
    """
    Corre el tutor de ingeniería con soporte de interrupciones de agente.
    """
    from src.engineering.engineering_session import (
        SYSTEM_ENGINEERING, _resolve_model as eng_resolve,
        _chat_stream
    )
    ctx.set_active("engineering")
    eng_ctx = ctx.get("engineering")

    agent = AgentSession()

    print(f"\n{'═'*60}")
    print(f"  TUTOR DE INGENIERÍA  |  Modelo: {eng_resolve()}")
    print(f"  'salir' → menú | 'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")

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
            break

        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "quit", "menu"):
            ctx.set_active("idle")
            break
        if user_text.lower() in ("limpiar", "clear"):
            eng_ctx.history = []
            print("  [INFO] Contexto limpiado.\n")
            continue

        # Detectar interrupción de agente
        interruption = detect_intent_from_active_mode(user_text, "engineering")

        if interruption == "agent":
            result = agent.run_turn(user_text, return_mode="engineering")

            if result["action"] == "web_search":
                speak(
                    f"Para buscar en internet tengo que pausar el tutor de ingeniería y no vas a poder retomarlo. "
                    f"{result['text']}\n¿Querés continuar con la búsqueda?"
                )
                confirm = listen()
                if confirm.lower() in ("sí", "si", "dale", "yes"):
                    ctx.set_active("idle")
                    agent._enter_web_mode(result["search_data"])
                    break
                else:
                    speak("Entendido, seguimos con el tutor.")
                    continue

            # Tool liviana
            speak(result["text"])
            return_q = generate_return_prompt("engineering", client, _resolve_model())
            speak(return_q)
            answer = listen()

            intent = detect_intent(answer)
            if intent == "engineering" or answer.lower() in ("sí", "si", "dale", "yes", "claro"):
                speak("Dale, ¿qué más querés explorar?")
                continue
            elif intent == "agent":
                continue
            else:
                ctx.set_active("idle")
                break
        else:
            # Turno normal del tutor de ingeniería
            import time as _time
            eng_ctx.history.append({"role": "user", "content": user_text})
            messages = [{"role": "system", "content": SYSTEM_ENGINEERING}] + eng_ctx.history[-8:]

            print("\n[INGENIERO]: ", end="", flush=True)
            response, ttft, total = _chat_stream(messages, temperature=0.3, max_tokens=600)
            print(f"\n  ⏱  TTFT={ttft}s | Total={total}s\n")

            eng_ctx.history.append({"role": "assistant", "content": response})
            if len(eng_ctx.history) > 12:
                eng_ctx.history = eng_ctx.history[-12:]


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  AGENTY — Asistente Edge")
    print("  Iniciando...")
    print("═" * 60)

    # Verificar Ollama
    if not _check_ollama():
        print("[ERROR] Ollama no está corriendo. Arrancalo antes de iniciar el asistente.")
        sys.exit(1)

    model = _resolve_model()
    print(f"[OK] Ollama conectado | Modelo: {model}")

    # Iniciar scheduler de recordatorios
    reminders.start_scheduler(interval_minutes=30)

    ctx = ContextManager()

    # Saludo inicial
    greeting = generate_greeting(client, model)
    speak(greeting)

    while True:
        _show_alerts()

        user_text = listen()
        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "chau", "quit"):
            speak("¡Hasta luego!")
            break

        intent = detect_intent(user_text)

        if intent == "unclear":
            clarification = generate_clarification(user_text, client, model)
            speak(clarification)
            # Segunda oportunidad
            user_text2 = listen()
            intent = detect_intent(user_text2)
            if intent == "unclear":
                speak("No pude entender qué querés hacer. Volvemos a empezar.")
                continue
            user_text = user_text2

        if intent == "english":
            speak("¡Perfecto, arrancamos con el tutor de inglés!")
            _run_english_with_interrupts(ctx)

        elif intent == "engineering":
            speak("Dale, te conecto con el tutor de ingeniería.")
            _run_engineering_with_interrupts(ctx)

        elif intent == "agent":
            agent = AgentSession()
            # Turno directo de agente (sin modo previo)
            result = agent.run_turn(user_text, return_mode=None)

            if result["action"] == "web_search":
                speak(result["text"])
                confirm = listen()
                if confirm.lower() in ("sí", "si", "dale", "yes"):
                    agent._enter_web_mode(result["search_data"])
                else:
                    speak("Entendido, cancelé la búsqueda. ¿Necesitás algo más?")
            else:
                speak(result["text"])

        # Volver al greeting loop


if __name__ == "__main__":
    main()