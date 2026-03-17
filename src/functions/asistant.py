"""
assistant.py

Asistente edge con 3 modos, selector conversacional y herramientas.

Uso:
    python assistant.py

Requisitos:
    - Ollama corriendo con el modelo fine-tuneado registrado
    - LanguageTool en puerto 8081 (solo para modo inglés)
    - local_calendar.py en el mismo directorio (para modo agente)

Registro del modelo en Ollama (hacer una vez tras el fine-tuning):
    ollama create asistente -f Modelfile.assistant
"""

import re
import sys
import json
import random
import datetime
import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/v1"
LT_URL         = "http://localhost:8081/v2/check"
MODEL          = "asistente"          # nombre en Ollama tras el fine-tuning
FALLBACK_MODEL = "qwen2.5:7b"         # si el fine-tuneado no está aún
TOPICS_FILE    = "topics.json"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# ─────────────────────────────────────────────────────────────────────────────
# System prompts — EXACTAMENTE los mismos del dataset de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_ENGLISH = (
    "You are a warm, friendly spoken English tutor for Spanish-speaking B1/B2 students. "
    "You always speak naturally and conversationally, never using bullet points or markdown. "
    "You output a single spoken paragraph ready for Text-to-Speech. "
    "You NEVER mention grammar errors that are not explicitly listed in the prompt. "
    "English only (except 2-sentence Spanish intros). No Chinese characters."
)

SYSTEM_ENGINEERING = (
    "You are a brilliant, warm retired engineer and scientist with decades of experience "
    "across software, AI, electronics, physics, chemistry, and systems engineering. "
    "You explain concepts with clarity, depth, and genuine enthusiasm — like a mentor who "
    "loves sharing knowledge. You NEVER write code, never do numerical calculations, and "
    "never solve logic puzzles. Instead, you explain the intuition, the trade-offs, the "
    "history, the analogies, and the real-world implications of technical concepts. "
    "You adapt your depth to what is being asked: a definition gets a crisp explanation, "
    "a comparison gets a structured contrast, a 'why' gets philosophy and context. "
    "You are direct and concrete — no filler phrases like 'Great question!' or 'Sure!'. "
    "If asked in Spanish, answer in Spanish. If asked in English, answer in English. "
    "No bullet-point lists unless the question is explicitly a comparison or enumeration. "
    "No Chinese characters."
)

# El system prompt del agente se construye dinámicamente con las herramientas disponibles
SYSTEM_AGENT_BASE = (
    "Sos un asistente personal de IA que corre localmente. "
    "Respondés siempre en español con voseo rioplatense. "
    "Tenés acceso a herramientas para ayudar al usuario. "
    "Cuando necesitás usar una herramienta, respondés EXACTAMENTE así:\n"
    "TOOL_CALL: {{\"tool\": \"nombre\", \"args\": {{...}}}}\n"
    "Esperás el resultado con TOOL_RESULT: ... antes de continuar.\n"
    "Si no necesitás herramientas, respondés directamente en prosa natural.\n"
    "Herramientas disponibles:\n"
    "{tools}"
)

TOOLS_DESCRIPTION = """- search_web(query): busca información actual en internet
- task_add(title, priority?): agrega una tarea a tu lista (priority: alta/media/baja)
- task_list(): muestra todas tus tareas pendientes
- task_done(task_id): marca una tarea como completada
- reminder_set(title, datetime_str): configura un recordatorio
- wa_send(contact, message): envía un mensaje de WhatsApp
- wa_read(contact?): lee mensajes de WhatsApp recientes
- cal_add(title, start, end?, description?): agrega un evento al calendario
- cal_list(date?): muestra eventos del calendario (default: hoy)
- cal_delete(event_id): elimina un evento del calendario"""


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def _model_name() -> str:
    """Devuelve el modelo fine-tuneado si existe, sino el fallback."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if MODEL.split(":")[0] in names:
            return MODEL
    except Exception:
        pass
    print(f"[WARN] Modelo '{MODEL}' no encontrado — usando '{FALLBACK_MODEL}'")
    return FALLBACK_MODEL


def _lt_check(text: str) -> list[dict]:
    """Llama a LanguageTool y devuelve lista de errores."""
    try:
        resp = requests.post(LT_URL, data={"text": text, "language": "en-US"}, timeout=10)
        errors = []
        for m in resp.json().get("matches", []):
            if m.get("replacements"):
                errors.append({
                    "wrong":    text[m["offset"]: m["offset"] + m["length"]],
                    "correct":  m["replacements"][0]["value"],
                    "sentence": m.get("sentence", ""),
                    "rule_id":  m["rule"]["id"],
                    "reason":   m["message"],
                })
        return errors
    except Exception:
        return []


def _chat(messages: list[dict], temperature: float = 0.4, max_tokens: int = 600) -> str:
    """Llamada básica al modelo."""
    model = _model_name()
    try:
        res = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Selector de modo — conversacional (el modelo pregunta, vos respondés)
# ─────────────────────────────────────────────────────────────────────────────
SELECTOR_SYSTEM = (
    "Sos un asistente personal que corre localmente en una Radxa Rock 5B. "
    "Respondés siempre en español con voseo rioplatense. "
    "Al iniciar, saludás brevemente y preguntás qué quiere hacer el usuario "
    "mencionando las 3 opciones disponibles de forma natural y concisa:\n"
    "  1. Practicar inglés (tutor con corrección gramatical)\n"
    "  2. Consultar algo técnico (ingeniero experto en software, IA, robótica, electrónica)\n"
    "  3. Pedirte una tarea concreta (buscar info, calendario, WhatsApp, to-do list)\n"
    "Respondés en 2-3 oraciones máximo, sin listas con bullets, de forma conversacional."
)

MODE_DETECT_PROMPT = """El usuario dijo: "{text}"

¿Qué modo quiere usar?
- english_tutor: practicar inglés, hablar en inglés, mejorar gramática
- engineering: pregunta técnica de software, IA, física, química, robótica, electrónica
- agent: hacer algo concreto (buscar, agendar, recordatorio, WhatsApp, tareas)
- unclear: no queda claro

Respondé con exactamente una palabra: english_tutor, engineering, agent, o unclear."""


def detect_mode(text: str) -> str:
    """Detecta el modo deseado a partir del texto del usuario."""
    prompt = MODE_DETECT_PROMPT.format(text=text)
    try:
        res = client.chat.completions.create(
            model=_model_name(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        result = res.choices[0].message.content.strip().lower()
        if "english" in result:   return "english_tutor"
        if "engineer" in result:  return "engineering"
        if "agent" in result:     return "agent"
        return "unclear"
    except Exception:
        return "unclear"


# ─────────────────────────────────────────────────────────────────────────────
# MODO 1 — English Tutor
# ─────────────────────────────────────────────────────────────────────────────
def _load_topics() -> list:
    try:
        with open(TOPICS_FILE, encoding="utf-8") as f:
            topics = json.load(f)
        return topics if isinstance(topics, list) else list(topics)
    except Exception:
        return ["technology", "travel", "daily life", "food", "music"]


def english_tutor_session():
    """
    Sesión completa del tutor de inglés.
    Genera apertura, recibe respuesta del estudiante, corre LT, da feedback.
    Loop hasta que el usuario diga 'exit' o 'salir'.
    """
    # Verificar LT
    lt_ok = False
    try:
        requests.get("http://localhost:8081/v2/languages", timeout=3)
        lt_ok = True
    except Exception:
        print("[WARN] LanguageTool no está corriendo — el tutor funcionará sin detección de errores.")

    topics  = _load_topics()
    history = []
    model   = _model_name()

    print("\n[Tutor de Inglés activo — escribí 'salir' para volver al menú]\n")

    while True:
        # ── Generar apertura ──────────────────────────────────────────────────
        topic = random.choice(topics) if isinstance(topics[0], str) else random.choice(topics)
        if isinstance(topic, dict):
            topic = topic.get("topic", str(topic))

        open_prompt = (
            f"Start a new English conversation about: '{topic}'.\n"
            f"Open with 2 sentences in Spanish introducing the topic, "
            f"then ask ONE open question in English. "
            f"Natural, conversational, ready for TTS. No markdown."
        )
        opening = _chat(
            [{"role": "system", "content": SYSTEM_ENGLISH},
             {"role": "user",   "content": open_prompt}],
            temperature=0.7, max_tokens=150
        )
        history.append({"role": "assistant", "content": opening})
        print(f"\n[TUTOR]: {opening}\n")

        # ── Recibir respuesta del estudiante ──────────────────────────────────
        student_text = input("[VOS]: ").strip()
        if not student_text:
            continue
        if student_text.lower() in ("salir", "exit", "quit", "menu", "menú"):
            print("[INFO] Volviendo al menú principal...")
            break

        # ── LanguageTool ──────────────────────────────────────────────────────
        lt_errors = _lt_check(student_text) if lt_ok else []
        if lt_errors:
            errors_str = "\n".join(
                f"  - Error: '{e['wrong']}' → '{e['correct']}' | {e['reason']}"
                for e in lt_errors
            )
            feedback_instruction = (
                f"The student said: \"{student_text}\"\n\n"
                f"LanguageTool found these errors:\n{errors_str}\n\n"
                f"Give warm feedback:\n"
                f"1. Mention ONLY the errors listed above, cite the student's exact wrong phrase.\n"
                f"2. Briefly explain why it's wrong.\n"
                f"3. End with ONE follow-up question about the topic.\n"
                f"Single spoken paragraph, no markdown, TTS-ready."
            )
        else:
            feedback_instruction = (
                f"The student said: \"{student_text}\"\n\n"
                f"No grammar errors found. Give warm feedback:\n"
                f"1. Briefly acknowledge something good in their response.\n"
                f"2. End with ONE follow-up question about the topic.\n"
                f"Single spoken paragraph, no markdown, TTS-ready."
            )

        history.append({"role": "user", "content": student_text})
        messages = [{"role": "system", "content": SYSTEM_ENGLISH}] + history[:-1] + \
                   [{"role": "user", "content": feedback_instruction}]

        feedback = _chat(messages, temperature=0.3, max_tokens=300)
        history.append({"role": "assistant", "content": feedback})

        # Mantener ventana de contexto
        if len(history) > 8:
            history = history[-8:]

        print(f"\n[TUTOR]: {feedback}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODO 2 — Engineering Tutor
# ─────────────────────────────────────────────────────────────────────────────
def engineering_tutor_session():
    """
    Sesión del tutor de ingeniería.
    Q&A con historial. Loop hasta 'salir'.
    """
    history = []
    model   = _model_name()

    print("\n[Tutor de Ingeniería activo — escribí 'salir' para volver al menú]")
    print("[Podés preguntar sobre software, IA, física, química, robótica, electrónica]\n")

    greeting = (
        "Hola, soy tu tutor de ingeniería. Tengo décadas de experiencia en software, "
        "inteligencia artificial, electrónica, física y más. "
        "¿Sobre qué concepto o tema querés que hablemos hoy?"
    )
    print(f"[INGENIERO]: {greeting}\n")

    while True:
        user_text = input("[VOS]: ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "quit", "menu", "menú"):
            print("[INFO] Volviendo al menú principal...")
            break

        history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM_ENGINEERING}] + history[-8:]

        response = _chat(messages, temperature=0.3, max_tokens=800)

        history.append({"role": "assistant", "content": response})
        if len(history) > 12:
            history = history[-12:]

        print(f"\n[INGENIERO]: {response}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODO 3 — Agente con herramientas
# ─────────────────────────────────────────────────────────────────────────────

# Dispatcher de herramientas reales
def _dispatch_tool(tool_name: str, args: dict) -> str:
    """Ejecuta la herramienta y devuelve el resultado como string."""

    if tool_name == "search_web":
        return _tool_search_web(args.get("query", ""))

    elif tool_name == "task_add":
        return _tool_task_add(args.get("title", ""), args.get("priority", "media"))

    elif tool_name == "task_list":
        return _tool_task_list()

    elif tool_name == "task_done":
        return _tool_task_done(args.get("task_id", ""))

    elif tool_name == "reminder_set":
        return _tool_reminder_set(args.get("title", ""), args.get("datetime_str", ""))

    elif tool_name == "wa_send":
        return _tool_wa_send(args.get("contact", ""), args.get("message", ""))

    elif tool_name == "wa_read":
        return _tool_wa_read(args.get("contact", None))

    elif tool_name == "cal_add":
        return _tool_cal_add(
            args.get("title", ""), args.get("start", ""),
            args.get("end", None), args.get("description", None)
        )

    elif tool_name == "cal_list":
        return _tool_cal_list(args.get("date", None))

    elif tool_name == "cal_delete":
        return _tool_cal_delete(args.get("event_id", ""))

    else:
        return f"[ERROR] Herramienta desconocida: {tool_name}"


# ── Implementaciones de herramientas ─────────────────────────────────────────

_tasks: list[dict] = []   # lista simple en memoria (reemplazar por SQLite si se quiere persistencia)
_task_counter = 0

def _tool_task_add(title: str, priority: str = "media") -> str:
    global _task_counter
    _task_counter += 1
    _tasks.append({"id": _task_counter, "title": title, "priority": priority, "done": False})
    return f"Tarea #{_task_counter} agregada: '{title}' (prioridad: {priority})"

def _tool_task_list() -> str:
    pending = [t for t in _tasks if not t["done"]]
    if not pending:
        return "No tenés tareas pendientes."
    lines = [f"#{t['id']} [{t['priority']}] {t['title']}" for t in pending]
    return "Tareas pendientes:\n" + "\n".join(lines)

def _tool_task_done(task_id) -> str:
    tid = int(str(task_id))
    for t in _tasks:
        if t["id"] == tid:
            t["done"] = True
            return f"Tarea #{tid} '{t['title']}' marcada como completada."
    return f"No encontré la tarea #{tid}."

def _tool_reminder_set(title: str, datetime_str: str) -> str:
    # Placeholder — en producción conectar con cron o APScheduler
    return f"Recordatorio configurado: '{title}' para {datetime_str}. (⚠ El dispatcher de recordatorios debe estar corriendo)"

def _tool_wa_send(contact: str, message: str) -> str:
    # Placeholder — en producción conectar con whatsapp-web.js
    return f"[SIMULADO] WhatsApp enviado a '{contact}': '{message}'"

def _tool_wa_read(contact=None) -> str:
    # Placeholder
    if contact:
        return f"[SIMULADO] Últimos mensajes de {contact}: (conectar whatsapp-web.js)"
    return "[SIMULADO] Bandeja de WhatsApp: (conectar whatsapp-web.js)"

def _tool_search_web(query: str) -> str:
    # Placeholder — en producción usar DuckDuckGo API o SearXNG local
    return f"[SIMULADO] Resultados de búsqueda para '{query}': (conectar motor de búsqueda local)"

def _tool_cal_add(title: str, start: str, end=None, description=None) -> str:
    try:
        from local_calendar import LocalCalendar
        cal = LocalCalendar()
        event_id = cal.add_event(title=title, start=start, end=end or start, description=description or "")
        return f"Evento '{title}' agregado al calendario (ID: {event_id})."
    except ImportError:
        return f"[SIMULADO] Evento '{title}' el {start} agregado. (instalar local_calendar.py)"
    except Exception as e:
        return f"[ERROR] No se pudo agregar el evento: {e}"

def _tool_cal_list(date=None) -> str:
    try:
        from local_calendar import LocalCalendar
        cal = LocalCalendar()
        target = date or datetime.date.today().isoformat()
        events = cal.list_events(date=target)
        if not events:
            return f"No hay eventos para {target}."
        return "\n".join(f"- {e['start']}: {e['title']}" for e in events)
    except ImportError:
        return "[SIMULADO] Lista del calendario (instalar local_calendar.py)"
    except Exception as e:
        return f"[ERROR] {e}"

def _tool_cal_delete(event_id) -> str:
    try:
        from local_calendar import LocalCalendar
        cal = LocalCalendar()
        cal.delete_event(int(event_id))
        return f"Evento #{event_id} eliminado del calendario."
    except Exception as e:
        return f"[ERROR] {e}"


# ── Loop del agente ───────────────────────────────────────────────────────────
_TOOL_CALL_RE = re.compile(r'TOOL_CALL:\s*(\{.*?\})', re.DOTALL)

def agent_session():
    """
    Sesión del agente con herramientas.
    Parsea TOOL_CALL, ejecuta la herramienta, inyecta TOOL_RESULT y continúa.
    """
    history = []
    system  = SYSTEM_AGENT_BASE.format(tools=TOOLS_DESCRIPTION)

    print("\n[Agente activo — escribí 'salir' para volver al menú]")
    print("[Podés pedirme: buscar info, agendar, recordatorios, WhatsApp, tareas]\n")

    greeting = (
        "Hola, soy tu agente personal. Puedo buscar información, gestionar tu calendario, "
        "mandarte recordatorios, enviarte mensajes de WhatsApp y manejar tu lista de tareas. "
        "¿Qué querés hacer?"
    )
    print(f"[AGENTE]: {greeting}\n")

    while True:
        user_text = input("[VOS]: ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "quit", "menu", "menú"):
            print("[INFO] Volviendo al menú principal...")
            break

        history.append({"role": "user", "content": user_text})

        # ── Loop interno: el agente puede encadenar múltiples tool calls ──────
        max_tool_rounds = 4
        for _ in range(max_tool_rounds):
            messages = [{"role": "system", "content": system}] + history[-10:]
            response = _chat(messages, temperature=0.2, max_tokens=400)

            # ¿Hay un TOOL_CALL?
            match = _TOOL_CALL_RE.search(response)
            if not match:
                # Respuesta final sin herramienta
                history.append({"role": "assistant", "content": response})
                break

            # Extraer y ejecutar la herramienta
            try:
                tool_data  = json.loads(match.group(1))
                tool_name  = tool_data.get("tool", "")
                tool_args  = tool_data.get("args", {})
                print(f"  [TOOL] {tool_name}({tool_args})")
                tool_result = _dispatch_tool(tool_name, tool_args)
                print(f"  [RESULT] {tool_result}")
            except json.JSONDecodeError:
                tool_result = "[ERROR] No se pudo parsear el TOOL_CALL."

            # Agregar la llamada y el resultado al historial para que el modelo continúe
            history.append({"role": "assistant", "content": response})
            history.append({"role": "user",      "content": f"TOOL_RESULT: {tool_result}"})
        else:
            # Demasiados tool rounds — responder directamente
            history.append({"role": "assistant", "content": "Procesé las herramientas disponibles."})

        # Mostrar la última respuesta del asistente (no el TOOL_CALL, sino la final)
        last_assistant = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"
             and "TOOL_CALL" not in m["content"]),
            "Listo."
        )
        print(f"\n[AGENTE]: {last_assistant}\n")

        if len(history) > 20:
            history = history[-20:]


# ─────────────────────────────────────────────────────────────────────────────
# Menú principal — el modelo pregunta, vos respondés
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Verificar Ollama ──────────────────────────────────────────────────────
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        print("[OK] Ollama conectado")
    except Exception:
        print("[ERROR] Ollama no está corriendo.")
        sys.exit(1)

    model = _model_name()
    print(f"[OK] Modelo: {model}")
    print()

    while True:
        # ── El modelo genera el saludo y pregunta el modo ─────────────────────
        saludo = _chat(
            [{"role": "system", "content": SELECTOR_SYSTEM},
             {"role": "user",   "content": "Inicia la sesión"}],
            temperature=0.5, max_tokens=120
        )
        print(f"\n[ASISTENTE]: {saludo}\n")

        # ── El usuario elige ──────────────────────────────────────────────────
        eleccion = input("[VOS]: ").strip()
        if not eleccion:
            continue
        if eleccion.lower() in ("salir", "exit", "quit", "chau"):
            print("\n[ASISTENTE]: ¡Hasta luego!")
            break

        # ── Detectar modo ─────────────────────────────────────────────────────
        modo = detect_mode(eleccion)

        if modo == "unclear":
            # Pedir aclaración sin entrar en ningún modo
            aclaracion = _chat(
                [{"role": "system", "content": SELECTOR_SYSTEM},
                 {"role": "user",   "content": eleccion},
                 {"role": "assistant", "content": saludo},
                 {"role": "user",   "content": (
                     "No entendiste qué quiere el usuario. "
                     "Pedile que aclare cuál de los 3 modos quiere usar."
                 )}],
                temperature=0.4, max_tokens=80
            )
            print(f"\n[ASISTENTE]: {aclaracion}\n")
            eleccion2 = input("[VOS]: ").strip()
            modo = detect_mode(eleccion2)
            if modo == "unclear":
                print("[ASISTENTE]: No pude entender qué querés hacer. Volvemos a empezar.\n")
                continue

        # ── Lanzar el modo elegido ────────────────────────────────────────────
        if modo == "english_tutor":
            print("\n[ASISTENTE]: ¡Perfecto, arrancamos con el tutor de inglés!\n")
            english_tutor_session()

        elif modo == "engineering":
            print("\n[ASISTENTE]: Bien, te conecto con el tutor de ingeniería.\n")
            engineering_tutor_session()

        elif modo == "agent":
            print("\n[ASISTENTE]: Listo, activando el agente.\n")
            agent_session()

        # Al salir de cualquier modo, vuelve al selector
        print("\n" + "─"*50)
        print("[INFO] Sesión terminada. Volviendo al menú...\n")


if __name__ == "__main__":
    main()