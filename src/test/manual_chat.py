"""
manual_chat.py — Chat interactivo para validar el modelo manualmente.

Modo English: flujo completo de TutorSession (LanguageTool + classify_intent
              + topics.json + change topic / propose topic / exit).
Modo Engineering y Agente: loop directo con streaming y latencia visible.

Uso:
    python manual_chat.py
    python manual_chat.py --mode english
    python manual_chat.py --mode engineering
    python manual_chat.py --mode agent

Requisitos:
    - Ollama corriendo con el modelo registrado
    - LanguageTool en puerto 8081 (solo para modo english)
    - topics.json en src/finetuning/ (se busca automáticamente)
"""

import argparse
import json
import os
import random
import re
import sys
import time

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config global
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/v1"
LT_URL         = "http://localhost:8081/v2/check"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"
MAX_LT_ERRORS  = 2

# Busca topics.json: junto al script → src/finetuning → ../../src/finetuning
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPICS_PATH = next(
    (p for p in [
        os.path.join(_SCRIPT_DIR, "topics.json"),
        os.path.join(_SCRIPT_DIR, "..", "finetuning", "topics.json"),
        os.path.join(_SCRIPT_DIR, "..", "..", "src", "finetuning", "topics.json"),
    ] if os.path.exists(p)),
    None
)

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades compartidas
# ─────────────────────────────────────────────────────────────────────────────
def resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def chat_stream(messages, temperature, max_tokens) -> tuple[str, float, float]:
    """Llama al modelo en streaming. Retorna (texto, ttft_s, total_s)."""
    model = resolve_model()
    t_start = time.perf_counter()
    first_token_t = None
    full_text = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta and first_token_t is None:
                first_token_t = time.perf_counter()
            full_text += delta
            print(delta, end="", flush=True)
    except Exception as e:
        full_text = f"[ERROR] {e}"
        print(full_text)
    t_end = time.perf_counter()
    ttft = (first_token_t - t_start) if first_token_t else (t_end - t_start)
    return full_text, round(ttft, 2), round(t_end - t_start, 2)


def chat_silent(messages, temperature=0.0, max_tokens=20) -> str:
    """Llama al modelo sin imprimir. Para classify_intent y similares."""
    model = resolve_model()
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


def extract_tool_call(response: str) -> dict | None:
    """Extrae TOOL_CALL balanceando llaves — maneja JSON anidado correctamente."""
    idx = response.find("TOOL_CALL:")
    if idx == -1:
        return None
    start = response.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(response[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(response[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _clean(text: str) -> str:
    text = re.sub(r'[#*\\]', '', text)
    text = re.sub(r'(Assistant|User|Tutor|Student)\s*:', '', text, flags=re.I)
    return " ".join(text.split()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
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
    "No Chinese characters. "
    "When stating specific facts (dates, names, measurements, records), if you are not "
    "completely certain, acknowledge it explicitly: 'si no me falla la memoria', 'creo que', "
    "'el dato exacto no lo tengo presente, pero...' — never substitute an uncertain fact "
    "with a wrong one."
)

SYSTEM_AGENT = (
    "Sos un asistente personal de IA que corre localmente en una Radxa Rock 5B. "
    "Hablás siempre en español rioplatense natural. "
    "Usás voseo correctamente: 'podés', 'tenés', 'querés', 'hacé', 'decime', 'avisame'. "
    "NUNCA usás 'vos' como muletilla al inicio de las oraciones. "
    "Tu tono es directo, amigable y conciso — como un asistente eficiente, no un chatbot genérico.\n\n"
    "HERRAMIENTAS DISPONIBLES:\n"
    "- task_add(title, priority?): agregá una tarea. priority: 'alta', 'media' o 'baja'.\n"
    "- task_list(): mostrá todas las tareas pendientes.\n"
    "- task_done(task_id): marcá una tarea como completada.\n"
    "- reminder_set(title, datetime_str): configurá un recordatorio.\n"
    "- wa_send(contact, message): enviá un mensaje de WhatsApp.\n"
    "- wa_read(contact?): leé mensajes de WhatsApp recientes.\n"
    "- cal_add(title, start, end?, description?): agregá un evento al calendario.\n"
    "- cal_list(date?): mostrá los eventos del calendario.\n"
    "- cal_delete(event_id): eliminá un evento del calendario.\n"
    "- search_web(query): buscá información en internet.\n\n"
    "REGLAS DE USO DE HERRAMIENTAS:\n"
    "1. Si el usuario pide mandar un mensaje → usá wa_send, NO task_add.\n"
    "2. Si el usuario pide agregar una tarea → usá task_add.\n"
    "3. Si el usuario pide agendar algo → usá cal_add.\n"
    "4. Cuando uses una herramienta, respondés EXACTAMENTE en este formato:\n"
    "   TOOL_CALL: {\"tool\": \"nombre_herramienta\", \"args\": {\"arg1\": \"valor1\"}}\n"
    "5. Esperás el TOOL_RESULT antes de responder al usuario.\n"
    "6. Si no necesitás herramientas, respondés directamente en prosa.\n\n"
    "EJEMPLOS DE TONO CORRECTO:\n"
    "Usuario: '¿qué tengo pendiente?' → TOOL_CALL task_list, después: 'Estas son tus tareas pendientes: ...'\n"
    "Usuario: 'agregá comprar leche' → TOOL_CALL task_add, después: 'Listo, agregué \"Comprar leche\".'\n"
    "Usuario: 'mandá un whatsapp a mamá diciendo que llego tarde' → TOOL_CALL wa_send\n"
    "Usuario: '¿cómo andás?' → 'Todo bien, listo para ayudarte. ¿Qué necesitás?'\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# MODO ENGLISH — TutorSession completa
# ─────────────────────────────────────────────────────────────────────────────
RELEVANT_RULES = {
    "HE_VERB_AGR": "Subject-verb agreement: 3rd person singular",
    "NON3PRS_VERB": "Subject-verb agreement: missing 3rd person -s",
    "DOES_DOESN_T": "Subject-verb agreement: do/does",
    "HAVE_PART_AGREEMENT": "Subject-verb agreement: have/has",
    "SVA": "Subject-verb agreement",
    "MD_BASEFORM": "Modal verb error: infinitive after modal",
    "TO_AFTER_MODAL_VERBS": "Modal verb error: 'to' after modal verb",
    "EN_A_VS_AN": "Wrong article: 'a' instead of 'an'",
    "R_MISSING_ARTICLE": "Missing article",
    "MISSING_DETERMINER": "Missing article",
    "SINCE_FOR_AGO": "Wrong tense: tense marker mismatch",
    "PRESENT_PERFECT_FOR_PAST": "Wrong tense: Present Perfect with past marker",
    "INFORMATIONS": "Uncountable noun pluralized: 'informations'",
    "ADVICES": "Uncountable noun pluralized: 'advices'",
    "EQUIPMENTS": "Uncountable noun pluralized: 'equipments'",
    "FURNITURES": "Uncountable noun pluralized: 'furnitures'",
    "KNOWLEDGES": "Uncountable noun pluralized: 'knowledges'",
    "LOOSE_LOSE": "Confused words: 'loose' instead of 'lose'",
    "AFFECT_EFFECT": "Confused words: 'effect' instead of 'affect'",
    "BORROW_LEND": "Confused words: 'borrow' instead of 'lend'",
    "I_AM_AGREE": "Syntax error: 'I am agree'",
    "I_AM_VB": "Syntax error: 'I am' + verb base form",
    "DOUBLE_NEGATIVE": "Syntax error: double negative",
    "MORFOLOGIK_RULE_EN_US": "Spelling error",
    "HUNSPELL_RULE": "Spelling error",
}
BLACKLIST_RULES = {
    "UPPERCASE_SENTENCE_START", "PUNCTUATION_PARAGRAPH_END",
    "COMMA_PARENTHESIS_WHITESPACE", "EN_QUOTES", "WHITESPACE_RULE",
    "DOUBLE_PUNCTUATION", "ENGLISH_WORD_REPEAT_RULE", "SENTENCE_WHITESPACE",
    "UNIT_SPACE", "CURRENCY", "EN_UNPAIRED_BRACKETS",
}
RELEVANT_CATEGORIES = {"GRAMMAR", "CONFUSED_WORDS", "TYPOS"}


def check_errors(text: str) -> list[dict]:
    try:
        resp = requests.post(LT_URL, data={"language": "en-US", "text": text}, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [LT] No disponible: {e}")
        return []
    errors = []
    for match in resp.json().get("matches", []):
        rule_id     = match.get("rule", {}).get("id", "")
        category_id = match.get("rule", {}).get("category", {}).get("id", "")
        if rule_id in BLACKLIST_RULES:
            continue
        if rule_id not in RELEVANT_RULES and category_id not in RELEVANT_CATEGORIES:
            continue
        ctx    = match.get("context", {})
        offset = ctx.get("offset", 0)
        length = ctx.get("length", 0)
        wrong  = ctx.get("text", "")[offset: offset + length]
        reps   = match.get("replacements", [])
        correct = reps[0].get("value", "") if reps else ""
        if not wrong or not correct or wrong.lower() == correct.lower():
            continue
        errors.append({
            "wrong":    wrong,
            "correct":  correct,
            "sentence": match.get("sentence", "").strip(),
            "reason":   match.get("message", "Grammar error."),
            "rule_id":  rule_id,
        })
        if len(errors) >= MAX_LT_ERRORS:
            break
    return errors


def errors_to_block(errors: list[dict]) -> str:
    if not errors:
        return ""
    lines = ["CONFIRMED GRAMMAR ERRORS — mention ONLY these, do not invent or add others:"]
    for e in errors:
        ctx = f' (in the sentence: "{e["sentence"]}")' if e.get("sentence") else ""
        lines.append(
            f'  - Student said "{e["wrong"]}" → correct: "{e["correct"]}"'
            + ctx + f' — reason: {e["reason"]}'
        )
    return "\n".join(lines)


def classify_intent(text: str) -> str:
    prompt = (
        f"Classify this student message with exactly ONE word.\n"
        f"Message: \"{text}\"\n"
        f"Options:\n"
        f"- exit: student wants to stop, end, quit, or finish the session\n"
        f"- change_topic: student wants a random different topic\n"
        f"- propose_topic: student proposes or suggests a specific topic to practice\n"
        f"- respond: student is answering or continuing the conversation\n"
        f"Answer with exactly one word: exit, change_topic, propose_topic, or respond."
    )
    result = chat_silent([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=5).lower()
    if "exit"    in result: return "exit"
    if "propose" in result: return "propose_topic"
    if "change"  in result or "topic" in result: return "change_topic"
    return "respond"


def extract_proposed_topic(text: str) -> str:
    prompt = (
        f"Extract the specific topic the student wants to practice from this message.\n"
        f"Message: \"{text}\"\n"
        f"Answer with ONLY the topic name, 1-5 words, no punctuation.\n"
        f"Topic:"
    )
    topic = chat_silent([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=15)
    return topic.strip("'\".").title() or text[:50]


def generate_opening(topic: str) -> str:
    model = resolve_model()
    prompt = (
        f"Topic: '{topic}'. "
        f"Write 2 short sentences in SPANISH introducing this topic naturally. "
        f"Then write ONE open-ended question in ENGLISH for a B1/B2 student. "
        f"No labels, no markdown. Ready for Text-to-Speech."
    )
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_ENGLISH},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        max_tokens=150,
    )
    return _clean(res.choices[0].message.content)


def generate_feedback(topic: str, history: list[dict],
                      student_text: str, errors: list[dict]) -> str:
    error_block = errors_to_block(errors)
    if errors:
        instruction = (
            f"You are giving spoken feedback to a B1/B2 English student.\n\n"
            f"TOPIC: {topic}\nSTUDENT RESPONSE: {student_text}\n\n"
            f"{error_block}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Acknowledge the student's idea in ONE short positive sentence (max 8 words, not about grammar).\n"
            f"2. For each confirmed error say naturally: \"In your sentence [brief quote], "
            f"you said '[wrong]' but it should be '[correct]' because [short reason].\"\n"
            f"3. End with ONE follow-up question in English.\n\n"
            f"STRICT RULES: One single fluid spoken paragraph. No lists, no markdown. "
            f"Mention ONLY the confirmed errors above. NEVER invent corrections. English only. TTS-ready."
        )
    else:
        instruction = (
            f"You are giving spoken feedback to a B1/B2 English student.\n\n"
            f"TOPIC: {topic}\nSTUDENT RESPONSE: {student_text}\n\n"
            f"The student made no grammar errors this turn.\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Briefly acknowledge the quality of the answer (max 10 words).\n"
            f"2. Pick ONE phrase the student used well and say why it works in English.\n"
            f"3. End with ONE follow-up question about the topic.\n\n"
            f"STRICT RULES: One single fluid spoken paragraph. No lists, no markdown. "
            f"English only. TTS-ready."
        )
    model = resolve_model()
    messages = [{"role": "system", "content": SYSTEM_ENGLISH}]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": instruction})
    res = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, max_tokens=300,
    )
    return _clean(res.choices[0].message.content)


def load_topics() -> list[str]:
    if TOPICS_PATH:
        with open(TOPICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [t if isinstance(t, str) else t.get("topic", str(t)) for t in data]
    return [
        "Artificial Intelligence", "Climate Change", "Social Media",
        "Remote Work", "Space Exploration", "Healthy Habits",
        "Travel and Culture", "Music and Identity", "Food and Traditions",
    ]


class TutorSession:

    def __init__(self):
        self.topics      = load_topics()
        self.topic       = random.choice(self.topics)
        self.used_topics = {self.topic}
        self.history     = []
        self.turn_count  = 0
        self.error_log   = []
        self.lt_ok       = False

    def _check_lt(self) -> bool:
        try:
            requests.get("http://localhost:8081/v2/languages", timeout=3)
            return True
        except Exception:
            return False

    def speak(self, text: str, total=None) -> None:
        print(f"\n[TUTOR]: {text}")
        if total is not None:
            print(f"  ⏱  Total={total}s")
        print()

    def listen(self) -> str:
        try:
            return input("[VOS]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "salir"

    def _pick_new_topic(self) -> None:
        available = [t for t in self.topics if t not in self.used_topics]
        if not available:
            self.used_topics = set()
            available = self.topics
        self.topic = random.choice(available)
        self.used_topics.add(self.topic)
        self.history = []
        print(f"\n  [INFO] Nuevo tema: {self.topic}\n")

    def _add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 4:
            self.history = self.history[-4:]

    def _start_topic(self) -> None:
        print(f"  [Generando apertura para: {self.topic}]")
        t0 = time.perf_counter()
        opening = generate_opening(self.topic)
        elapsed = round(time.perf_counter() - t0, 2)
        self._add_history("assistant", opening)
        self.speak(opening, total=elapsed)

    def _ask_topic_preference(self) -> None:
        if TOPICS_PATH:
            print(f"  [topics] {len(self.topics)} temas cargados desde: {os.path.basename(TOPICS_PATH)}")
        else:
            print("  [topics] topics.json no encontrado — usando fallback")
        print()
        print("  ¿Querés practicar un tema específico?")
        print("  (o presioná Enter para un tema aleatorio)")
        custom = input("  → ").strip()
        if custom:
            self.topic = custom
            self.used_topics.add(self.topic)
            print(f"  [INFO] Tema: {self.topic}")
        else:
            print(f"  [INFO] Tema aleatorio: {self.topic}")

    def run(self) -> None:
        print(f"\n{'═'*60}")
        print(f"  ENGLISH TUTOR  |  Modelo: {resolve_model()}")
        print(f"  'otro tema'            → tema aleatorio nuevo")
        print(f"  'quiero practicar X'   → proponer tema")
        print(f"  'stop' / 'salir'       → terminar sesión")
        print(f"{'═'*60}")

        self.lt_ok = self._check_lt()
        status = "[OK] conectado" if self.lt_ok else "[WARN] no disponible — sin detección de errores"
        print(f"\n  LanguageTool: {status}\n")

        self._ask_topic_preference()
        self._start_topic()

        while True:
            student_text = self.listen()
            if not student_text:
                continue

            intent = classify_intent(student_text)
            print(f"  [intent: {intent}]")

            if intent == "exit":
                self.speak(
                    "It was great practicing English with you! "
                    "Keep it up and see you next time!"
                )
                break

            if intent == "change_topic":
                self._pick_new_topic()
                self._start_topic()
                continue

            if intent == "propose_topic":
                self.topic = extract_proposed_topic(student_text)
                self.used_topics.add(self.topic)
                self.history = []
                print(f"  [INFO] Tema propuesto: {self.topic}")
                self._start_topic()
                continue

            # Turno normal
            self.turn_count += 1
            errors = check_errors(student_text) if self.lt_ok else []

            if errors:
                print(f"  [LT] {len(errors)} error(s): " +
                      " | ".join(f"'{e['wrong']}' → '{e['correct']}'" for e in errors))
                self.error_log.append({
                    "turn": self.turn_count, "topic": self.topic, "errors": errors,
                })
            else:
                print("  [LT] Sin errores.")

            t0 = time.perf_counter()
            feedback = generate_feedback(self.topic, self.history, student_text, errors)
            elapsed  = round(time.perf_counter() - t0, 2)

            self._add_history("user", student_text)
            self._add_history("assistant", feedback)
            self.speak(feedback, total=elapsed)

        self._save_log()

    def _save_log(self) -> None:
        total_errors = sum(len(e["errors"]) for e in self.error_log)
        log = {
            "turns":          self.turn_count,
            "topics_covered": list(self.used_topics),
            "total_errors":   total_errors,
            "error_log":      self.error_log,
        }
        log_path = os.path.join(_SCRIPT_DIR, "session_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        print(f"  [LOG] Turnos: {self.turn_count} | Errores: {total_errors}")
        print(f"  [LOG] Guardado en: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODO AGENTE
# ─────────────────────────────────────────────────────────────────────────────
def simulate_tool(tool_name: str, args: dict) -> str:
    if tool_name == "task_list":
        return "[ ] Revisar fine-tuning\n[ ] Comprar yerba\n[ ] Llamar al médico"
    elif tool_name == "task_add":
        return f"Tarea '{args.get('title', '?')}' agregada."
    elif tool_name == "task_done":
        return f"Tarea #{args.get('task_id', '?')} completada."
    elif tool_name == "cal_list":
        return "14:00 - Reunión con equipo\n17:00 - Dentista"
    elif tool_name == "cal_add":
        return f"Evento '{args.get('title', '?')}' agregado para {args.get('start', '?')}."
    elif tool_name == "cal_delete":
        return f"Evento #{args.get('event_id', '?')} eliminado."
    elif tool_name == "wa_send":
        return f"WhatsApp enviado a '{args.get('contact', '?')}': \"{args.get('message', '?')}\""
    elif tool_name == "wa_read":
        return f"[SIMULADO] Mensajes de {args.get('contact', 'todos')}."
    elif tool_name == "search_web":
        return f"[SIMULADO] Resultados para: {args.get('query', '?')}"
    elif tool_name == "reminder_set":
        return f"Recordatorio '{args.get('title', '?')}' para {args.get('datetime_str', '?')}."
    else:
        return f"[SIMULADO] {tool_name}({args}) ejecutado."


def run_agent_turn(history: list, user_text: str) -> list:
    history.append({"role": "user", "content": user_text})
    for _ in range(4):
        messages = [{"role": "system", "content": SYSTEM_AGENT}] + history[-10:]
        print("\n[AGENTE]: ", end="", flush=True)
        response, ttft, total = chat_stream(messages, temperature=0.1, max_tokens=300)
        print(f"\n  ⏱  TTFT={ttft}s | Total={total}s\n")
        match = extract_tool_call(response)
        if not match:
            history.append({"role": "assistant", "content": response})
            break
        tool_name = match.get("tool", "")
        tool_args = match.get("args", {})
        print(f"  🔧 {tool_name}({tool_args})")
        result = simulate_tool(tool_name, tool_args)
        print(f"  📦 {result}\n")
        history.append({"role": "assistant", "content": response})
        history.append({"role": "user", "content": f"TOOL_RESULT: {result}"})
    if len(history) > 20:
        history = history[-20:]
    return history


def run_agent_session():
    history = []
    print(f"\n{'═'*60}")
    print(f"  AGENTE  |  Modelo: {resolve_model()}")
    print(f"  'salir' → menú | 'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")
    print("[AGENTE]: ¡Hola! ¿Qué necesitás?\n")
    while True:
        try:
            user_text = input("[VOS]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "quit", "menu", "menú"):
            break
        if user_text.lower() in ("limpiar", "clear", "reset"):
            history = []
            print("  [INFO] Contexto limpiado.\n")
            continue
        history = run_agent_turn(history, user_text)


# ─────────────────────────────────────────────────────────────────────────────
# MODO ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def run_engineering_session():
    history = []
    print(f"\n{'═'*60}")
    print(f"  TUTOR DE INGENIERÍA  |  Modelo: {resolve_model()}")
    print(f"  'salir' → menú | 'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")
    print("[INGENIERO]: Hola, soy tu tutor de ingeniería. ¿Sobre qué concepto querés hablar?\n")
    while True:
        try:
            user_text = input("[VOS]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "quit", "menu", "menú"):
            break
        if user_text.lower() in ("limpiar", "clear", "reset"):
            history = []
            print("  [INFO] Contexto limpiado.\n")
            continue
        history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM_ENGINEERING}] + history[-8:]
        print("\n[INGENIERO]: ", end="", flush=True)
        response, ttft, total = chat_stream(messages, temperature=0.3, max_tokens=600)
        print(f"\n  ⏱  TTFT={ttft}s | Total={total}s\n")
        history.append({"role": "assistant", "content": response})
        if len(history) > 12:
            history = history[-12:]


# ─────────────────────────────────────────────────────────────────────────────
# Menú y main
# ─────────────────────────────────────────────────────────────────────────────
def choose_mode() -> str:
    print("\n┌─────────────────────────────────────┐")
    print("│       SELECCIONÁ UN MODO            │")
    print("├─────────────────────────────────────┤")
    print("│  1. English Tutor                   │")
    print("│  2. Tutor de Ingeniería             │")
    print("│  3. Agente con herramientas         │")
    print("│  q. Salir                           │")
    print("└─────────────────────────────────────┘")
    choice = input("Modo: ").strip().lower()
    return {"1": "english", "2": "engineering", "3": "agent"}.get(choice, choice)


def main():
    parser = argparse.ArgumentParser(description="Chat manual para validar el modelo fine-tuneado")
    parser.add_argument("--mode", choices=["english", "engineering", "agent"])
    args = parser.parse_args()

    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"[OK] Ollama conectado | Modelo: {resolve_model()}")
    except Exception:
        print("[ERROR] Ollama no está corriendo en localhost:11434")
        sys.exit(1)

    dispatch = {
        "english":     lambda: TutorSession().run(),
        "engineering": run_engineering_session,
        "agent":       run_agent_session,
    }

    if args.mode:
        dispatch[args.mode]()
        return

    while True:
        mode = choose_mode()
        if mode in ("q", "salir", "exit"):
            print("\n¡Hasta luego!\n")
            break
        elif mode in dispatch:
            dispatch[mode]()
            print("\n" + "─" * 50 + "\n")
        else:
            print(f"[ERROR] Modo desconocido: '{mode}'")


if __name__ == "__main__":
    main()