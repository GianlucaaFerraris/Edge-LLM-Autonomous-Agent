"""
tutor_session.py — v2

Sesión infinita de conversación con el tutor de inglés.
Flujo por turno:

  1. Tutor genera apertura (español + pregunta en inglés)
  2. Usuario responde (texto o audio transcripto por Whisper)
     - Si dice "stop/exit/terminar/salir/fin" → termina la sesión
     - Si dice "otro tema/change topic/next topic" → nuevo tema sin feedback
  3. LanguageTool detecta errores — el LLM NO detecta errores, solo genera texto
  4. Tutor da feedback + follow-up question → volver a 2

Dependencias:
    pip install openai requests

LanguageTool debe estar corriendo en puerto 8081.
"""

import json
import os
import random
import re
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL  = "http://localhost:11434/v1"
LT_URL      = "http://localhost:8081/v2/check"
MODEL       = "qwen2.5:7b"
TOPICS_PATH = "topics.json"
MAX_ERRORS  = 2

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# ---------------------------------------------------------------------------
# Control de flujo — keywords
# ---------------------------------------------------------------------------
def _extract_proposed_topic(text: str) -> str:
    """
    Extrae el tema que el usuario quiere practicar de su mensaje.
    Ej: "I want to practice job interviews" → "job interviews"
    """
    prompt = (
        f"Extract the specific topic the student wants to practice from this message.\n"
        f"Message: \"{text}\"\n"
        f"Answer with ONLY the topic name, 1-5 words, no punctuation.\n"
        f"Examples: 'job interviews', 'cooking', 'travel vocabulary', 'medical English'\n"
        f"Topic:"
    )
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=15,
        )
        topic = res.choices[0].message.content.strip().strip("'\".").title()
        return topic if topic else text[:50]
    except Exception:
        return text[:50]


def classify_intent(text: str) -> str:
    """
    Clasifica la intención del mensaje del estudiante.
    Devuelve: 'exit', 'change_topic', 'propose_topic', o 'respond'.
    Usa el LLM con temperatura 0 — más robusto que keyword matching.
    """
    prompt = (
        f"Classify this student message with exactly ONE word.\n"
        f"Message: \"{text}\"\n"
        f"Options:\n"
        f"- exit: student wants to stop, end, quit, or finish the session\n"
        f"- change_topic: student wants a random different topic\n"
        f"- propose_topic: student proposes or suggests a specific topic to practice "
        f"(e.g. 'I want to practice job interviews', 'let\'s talk about cooking')\n"
        f"- respond: student is answering or continuing the conversation\n"
        f"Answer with exactly one word: exit, change_topic, propose_topic, or respond."
    )
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        result = res.choices[0].message.content.strip().lower()
    except Exception:
        return "respond"

    if "exit" in result:
        return "exit"
    if "propose" in result:
        return "propose_topic"
    if "change" in result or "topic" in result:
        return "change_topic"
    return "respond"

# ---------------------------------------------------------------------------
# Temas
# ---------------------------------------------------------------------------
def load_topics(path: str = TOPICS_PATH) -> list[str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return [
        "Artificial Intelligence", "Climate Change", "Social Media",
        "Remote Work", "Space Exploration", "Education Technology",
        "Healthy Habits", "Travel and Culture", "Music and Identity",
        "The Future of Cities", "Food and Traditions", "Sports and Teamwork",
    ]

# ---------------------------------------------------------------------------
# LanguageTool — ÚNICO detector de errores
# ---------------------------------------------------------------------------
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
    "UPPERCASE_SENTENCE_START",
    "PUNCTUATION_PARAGRAPH_END",
    "COMMA_PARENTHESIS_WHITESPACE",
    "EN_QUOTES",
    "WHITESPACE_RULE",
    "DOUBLE_PUNCTUATION",
    "ENGLISH_WORD_REPEAT_RULE",
    "SENTENCE_WHITESPACE",
    "UNIT_SPACE",
    "CURRENCY",
    "EN_UNPAIRED_BRACKETS",
}

RELEVANT_CATEGORIES = {"GRAMMAR", "CONFUSED_WORDS", "TYPOS"}


def check_errors(text: str) -> list[dict]:
    try:
        resp = requests.post(
            LT_URL,
            data={"language": "en-US", "text": text},
            timeout=5
        )
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
    except Exception as e:
        print(f"[LT] No disponible: {e}")
        return []

    errors = []
    for match in matches:
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

        replacements = match.get("replacements", [])
        correct = replacements[0].get("value", "") if replacements else ""

        if not wrong or not correct or wrong.lower() == correct.lower():
            continue

        errors.append({
            "wrong":    wrong,
            "correct":  correct,
            "sentence": match.get("sentence", "").strip(),
            "category": RELEVANT_RULES.get(rule_id,
                        match["rule"].get("description", "Grammar error")),
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
        sentence_ctx = f' (in the sentence: "{e["sentence"]}")' if e.get("sentence") else ""
        lines.append(
            f'  - Student said "{e["wrong"]}" → correct: "{e["correct"]}"'
            + sentence_ctx +
            f' — reason: {e["reason"]}'
        )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# LLM — genera texto, nunca detecta errores
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a warm, friendly spoken English tutor for Spanish-speaking B1/B2 students. "
    "You always speak naturally and conversationally, never using bullet points or markdown. "
    "You output a single spoken paragraph ready for Text-to-Speech. "
    "You NEVER mention grammar errors that are not explicitly listed in the prompt. "
    "English only (except 2-sentence Spanish intros). No Chinese characters."
)


def _clean(text: str) -> str:
    text = re.sub(r'[#*\\]', '', text)
    text = re.sub(r'(Assistant|User|Tutor|Student)\s*:', '', text, flags=re.I)
    return " ".join(text.split()).strip()


def generate_opening(topic: str) -> str:
    prompt = (
        f"Topic: '{topic}'. "
        f"Write 2 short sentences in SPANISH introducing this topic naturally. "
        f"Then write ONE open-ended question in ENGLISH for a B1/B2 student. "
        f"No labels, no markdown. Ready for Text-to-Speech."
    )
    res = client.chat.completions.create(
        model=MODEL,
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
        instruction = f"""You are giving spoken feedback to a B1/B2 English student.

TOPIC: {topic}
STUDENT RESPONSE: {student_text}

{error_block}

INSTRUCTIONS:
1. Acknowledge the student's idea in ONE short positive sentence (max 8 words, not about grammar).
2. For each confirmed error, say naturally:
   "In your sentence [quote the student sentence briefly], you said \'[wrong phrase]\' but it should be \'[correct form]\' because [short reason]."
3. End with ONE follow-up question in English to continue the conversation.

STRICT RULES:
- One single fluid spoken paragraph. No lists, no markdown, no newlines.
- Mention ONLY the confirmed errors listed above. NEVER invent corrections.
- English only. Ready for Text-to-Speech."""

    else:
        instruction = f"""You are giving spoken feedback to a B1/B2 English student.

TOPIC: {topic}
STUDENT RESPONSE: {student_text}

The student made no grammar errors this turn.

INSTRUCTIONS:
1. Briefly acknowledge the quality of the answer (max 10 words).
2. Pick ONE phrase the student used well and say why it works in English.
3. End with ONE follow-up question about the topic.

STRICT RULES:
- One single fluid spoken paragraph. No lists, no markdown, no newlines.
- English only. Ready for Text-to-Speech."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": instruction})

    res = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return _clean(res.choices[0].message.content)


# ---------------------------------------------------------------------------
# Sesión infinita
# ---------------------------------------------------------------------------
class TutorSession:

    def __init__(self):
        self.topics     = load_topics()
        self.topic      = random.choice(self.topics)
        self.used_topics = {self.topic}
        self.history    = []
        self.turn_count = 0
        self.error_log  = []

    def speak(self, text: str) -> None:
        """Reemplazar por TTS en producción."""
        print(f"\n[TUTOR]: {text}\n")

    def listen(self) -> str:
        """Reemplazar por Whisper en producción."""
        return input("[STUDENT]: ").strip()

    def _pick_new_topic(self) -> None:
        available = [t for t in self.topics if t not in self.used_topics]
        if not available:
            # Todos los temas usados — resetear y empezar de nuevo
            self.used_topics = set()
            available = self.topics
        self.topic = random.choice(available)
        self.used_topics.add(self.topic)
        self.history = []  # reset contexto al cambiar tema
        print(f"\n[INFO] Nuevo tema: {self.topic}\n")

    def _add_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 4:
            self.history = self.history[-4:]

    def _start_topic(self) -> None:
        print(f"[Tema: {self.topic}] Generando apertura...")
        opening = generate_opening(self.topic)
        self._add_history("assistant", opening)
        self.speak(opening)

    def _ask_topic_preference(self) -> None:
        """
        Al inicio pregunta si el usuario quiere proponer un tema o usar uno aleatorio.
        En producción reemplazar print/input por TTS/Whisper.
        """
        print("\n¿Querés practicar un tema específico?")
        print("Ejemplos: 'job interview', 'travel', 'technology', 'cooking'...")
        custom = input("Escribí el tema o presioná Enter para uno aleatorio → ").strip()
        if custom:
            self.topic = custom
            self.used_topics.add(self.topic)
            print(f"[INFO] Tema propuesto: {self.topic}")
        else:
            print(f"[INFO] Tema aleatorio: {self.topic}")

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Tutor de Inglés — Sesión iniciada")
        print(f"  'otro tema' → cambiar tema | 'stop' → terminar")
        print(f"{'='*60}")

        self._ask_topic_preference()
        self._start_topic()

        while True:
            student_text = self.listen()

            if not student_text:
                continue

            intent = classify_intent(student_text)

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
                # Extraer el tema propuesto del texto del estudiante
                self.topic = _extract_proposed_topic(student_text)
                self.used_topics.add(self.topic)
                self.history = []
                print(f"[INFO] Tema propuesto por el usuario: {self.topic}")
                self._start_topic()
                continue

            # Turno normal
            self.turn_count += 1
            errors = check_errors(student_text)

            if errors:
                print(f"[LT] {len(errors)} error(s): " +
                      " | ".join(f"'{e['wrong']}' → '{e['correct']}'" for e in errors))
                self.error_log.append({
                    "turn":   self.turn_count,
                    "topic":  self.topic,
                    "errors": errors,
                })
            else:
                print("[LT] Sin errores detectados.")

            feedback = generate_feedback(
                self.topic, self.history, student_text, errors
            )
            self._add_history("user", student_text)
            self._add_history("assistant", feedback)
            self.speak(feedback)

        self._save_log()

    def _save_log(self) -> None:
        total_errors = sum(len(e["errors"]) for e in self.error_log)
        log = {
            "turns":               self.turn_count,
            "topics_covered":      list(self.used_topics),
            "total_errors":        total_errors,
            "error_log":           self.error_log,
        }
        with open("session_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        print(f"\n[LOG] Turnos: {self.turn_count} | Errores: {total_errors}")
        print(f"[LOG] Guardado en session_log.jsonl")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    try:
        requests.get("http://localhost:8081/v2/languages", timeout=3)
    except Exception:
        print("[ERROR] LanguageTool no está corriendo en puerto 8081.")
        print("Arrancalo con:")
        print("  java -cp languagetool-server.jar org.languagetool.server.HTTPServer \\")
        print("       --port 8081 --allow-origin '*' --public")
        sys.exit(1)

    TutorSession().run()


if __name__ == "__main__":
    main()