"""
tutor_session.py — Sesión del tutor de inglés.

Módulo standalone. Puede correrse directamente o ser llamado desde main.py.

Flujo por turno:
  1. Tutor genera apertura (español + pregunta en inglés)
  2. Usuario responde
     - classify_intent → exit / change_topic / propose_topic / respond
  3. LanguageTool detecta errores (el LLM NO detecta errores)
  4. Tutor da feedback + follow-up question

Dependencias:
    pip install openai requests
    LanguageTool corriendo en puerto 8081
"""

import json
import os
import random
import re
import time

import requests
from openai import OpenAI

OLLAMA_URL  = "http://localhost:11434/v1"
LT_URL      = "http://localhost:8081/v2/check"
MODEL       = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"
MAX_ERRORS  = 2

# Busca topics.json subiendo por el árbol de directorios
_HERE = os.path.dirname(os.path.abspath(__file__))
TOPICS_PATH = next(
    (p for p in [
        os.path.join(_HERE, "topics.json"),
        os.path.join(_HERE, "..", "finetuning", "topics.json"),
        os.path.join(_HERE, "..", "..", "src", "finetuning", "topics.json"),
        os.path.join(_HERE, "..", "..", "finetuning", "topics.json"),
    ] if os.path.exists(p)),
    None
)

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

SYSTEM_PROMPT = (
    "You are a warm, friendly spoken English tutor for Spanish-speaking B1/B2 students. "
    "You always speak naturally and conversationally, never using bullet points or markdown. "
    "You output a single spoken paragraph ready for Text-to-Speech. "
    "You NEVER mention grammar errors that are not explicitly listed in the prompt. "
    "English only (except 2-sentence Spanish intros). No Chinese characters."
)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def _chat_silent(messages: list[dict], temperature=0.0, max_tokens=20) -> str:
    model = _resolve_model()
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


def _clean(text: str) -> str:
    text = re.sub(r'[#*\\]', '', text)
    text = re.sub(r'(Assistant|User|Tutor|Student)\s*:', '', text, flags=re.I)
    return " ".join(text.split()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# classify_intent y extract_proposed_topic
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent(text: str) -> str:
    """
    Clasifica la intención del mensaje del estudiante.
    Retorna: 'exit', 'change_topic', 'propose_topic', o 'respond'.
    """
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
    result = _chat_silent(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=5
    ).lower()
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
    topic = _chat_silent(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=15
    )
    return topic.strip("'\".").title() or text[:50]


# ─────────────────────────────────────────────────────────────────────────────
# LanguageTool
# ─────────────────────────────────────────────────────────────────────────────

RELEVANT_RULES = {
    "HE_VERB_AGR":              "Subject-verb agreement: 3rd person singular",
    "NON3PRS_VERB":             "Subject-verb agreement: missing 3rd person -s",
    "DOES_DOESN_T":             "Subject-verb agreement: do/does",
    "HAVE_PART_AGREEMENT":      "Subject-verb agreement: have/has",
    "SVA":                      "Subject-verb agreement",
    "MD_BASEFORM":              "Modal verb error: infinitive after modal",
    "TO_AFTER_MODAL_VERBS":     "Modal verb error: 'to' after modal verb",
    "EN_A_VS_AN":               "Wrong article: 'a' instead of 'an'",
    "R_MISSING_ARTICLE":        "Missing article",
    "MISSING_DETERMINER":       "Missing article",
    "SINCE_FOR_AGO":            "Wrong tense: tense marker mismatch",
    "PRESENT_PERFECT_FOR_PAST": "Wrong tense: Present Perfect with past marker",
    "INFORMATIONS":             "Uncountable noun pluralized: 'informations'",
    "ADVICES":                  "Uncountable noun pluralized: 'advices'",
    "EQUIPMENTS":               "Uncountable noun pluralized: 'equipments'",
    "FURNITURES":               "Uncountable noun pluralized: 'furnitures'",
    "KNOWLEDGES":               "Uncountable noun pluralized: 'knowledges'",
    "LOOSE_LOSE":               "Confused words: 'loose' instead of 'lose'",
    "AFFECT_EFFECT":            "Confused words: 'effect' instead of 'affect'",
    "BORROW_LEND":              "Confused words: 'borrow' instead of 'lend'",
    "I_AM_AGREE":               "Syntax error: 'I am agree'",
    "I_AM_VB":                  "Syntax error: 'I am' + verb base form",
    "DOUBLE_NEGATIVE":          "Syntax error: double negative",
    "MORFOLOGIK_RULE_EN_US":    "Spelling error",
    "HUNSPELL_RULE":            "Spelling error",
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
        resp = requests.post(
            LT_URL, data={"language": "en-US", "text": text}, timeout=5
        )
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
        if len(errors) >= MAX_ERRORS:
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


# ─────────────────────────────────────────────────────────────────────────────
# Generadores LLM
# ─────────────────────────────────────────────────────────────────────────────

def generate_opening(topic: str) -> str:
    model = _resolve_model()
    prompt = (
        f"Topic: '{topic}'. "
        f"Write 2 short sentences in SPANISH introducing this topic naturally. "
        f"Then write ONE open-ended question in ENGLISH for a B1/B2 student. "
        f"No labels, no markdown. Ready for Text-to-Speech."
    )
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
            f"1. Acknowledge the student's idea in ONE short positive sentence (max 8 words).\n"
            f"2. For each confirmed error say naturally: \"In your sentence [brief quote], "
            f"you said '[wrong]' but it should be '[correct]' because [short reason].\"\n"
            f"3. End with ONE follow-up question in English.\n\n"
            f"STRICT RULES: One single fluid spoken paragraph. No lists, no markdown. "
            f"Mention ONLY the confirmed errors above. NEVER invent corrections. "
            f"English only. TTS-ready."
        )
    else:
        instruction = (
            f"You are giving spoken feedback to a B1/B2 English student.\n\n"
            f"TOPIC: {topic}\nSTUDENT RESPONSE: {student_text}\n\n"
            f"The student made no grammar errors this turn.\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Briefly acknowledge the quality of the answer (max 10 words).\n"
            f"2. Pick ONE phrase the student used well and say why it works.\n"
            f"3. End with ONE follow-up question about the topic.\n\n"
            f"STRICT RULES: One single fluid spoken paragraph. No lists, no markdown. "
            f"English only. TTS-ready."
        )
    model = _resolve_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": instruction})
    res = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2, max_tokens=300,
    )
    return _clean(res.choices[0].message.content)


# ─────────────────────────────────────────────────────────────────────────────
# Topics
# ─────────────────────────────────────────────────────────────────────────────

def load_topics() -> list[str]:
    if TOPICS_PATH:
        with open(TOPICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [t if isinstance(t, str) else t.get("topic", str(t)) for t in data]
    return [
        "Artificial Intelligence", "Climate Change", "Social Media",
        "Remote Work", "Space Exploration", "Healthy Habits",
        "Travel and Culture", "Music and Identity", "Food and Traditions",
        "Sports and Teamwork", "The Future of Cities", "Education Technology",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TutorSession
# ─────────────────────────────────────────────────────────────────────────────

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
        """Hook para TTS en producción."""
        print(f"\n[TUTOR]: {text}")
        if total is not None:
            print(f"  ⏱  Total={total}s")
        print()

    def listen(self) -> str:
        """Hook para Whisper en producción."""
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
        print(f"  [Generando apertura: {self.topic}]")
        t0 = time.perf_counter()
        opening = generate_opening(self.topic)
        elapsed = round(time.perf_counter() - t0, 2)
        self._add_history("assistant", opening)
        self.speak(opening, total=elapsed)

    def _ask_topic_preference(self) -> None:
        if TOPICS_PATH:
            print(f"  [topics] {len(self.topics)} temas desde: {os.path.basename(TOPICS_PATH)}")
        else:
            print("  [topics] topics.json no encontrado — usando fallback")
        print()
        print("  ¿Querés practicar un tema específico?")
        print("  (o presioná Enter para uno aleatorio)")
        custom = input("  → ").strip()
        if custom:
            self.topic = custom
            self.used_topics.add(self.topic)
            print(f"  [INFO] Tema: {self.topic}")
        else:
            print(f"  [INFO] Tema aleatorio: {self.topic}")

    def run(self) -> None:
        """Sesión standalone completa."""
        print(f"\n{'═'*60}")
        print(f"  ENGLISH TUTOR  |  Modelo: {_resolve_model()}")
        print(f"  'otro tema'          → tema aleatorio")
        print(f"  'quiero practicar X' → proponer tema")
        print(f"  'stop'/'salir'       → terminar")
        print(f"{'═'*60}")

        self.lt_ok = self._check_lt()
        status = "[OK]" if self.lt_ok else "[WARN] sin detección de errores"
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
            self._run_normal_turn(student_text)

        self._save_log()

    def _run_normal_turn(self, student_text: str) -> None:
        """Ejecuta un turno normal de práctica. Llamable desde main.py."""
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

    def _save_log(self) -> None:
        total_errors = sum(len(e["errors"]) for e in self.error_log)
        log = {
            "turns":          self.turn_count,
            "topics_covered": list(self.used_topics),
            "total_errors":   total_errors,
            "error_log":      self.error_log,
        }
        log_path = os.path.join(_HERE, "..", "..", "session_log.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        print(f"  [LOG] Turnos: {self.turn_count} | Errores: {total_errors}")
        print(f"  [LOG] Guardado en session_log.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point standalone
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TutorSession().run()