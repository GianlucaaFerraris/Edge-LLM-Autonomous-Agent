"""
generate_dataset_english_tutor.py

Genera el dataset de fine-tuning para el modo English Tutor.
Usa exactamente la misma lógica que tutor_session.py:
  - Mismo SYSTEM_PROMPT
  - Mismo RELEVANT_RULES / BLACKLIST_RULES / RELEVANT_CATEGORIES
  - Mismo errors_to_block()
  - Mismo generate_opening() / generate_feedback()
  - Lee temas desde topics.json (igual que TutorSession.load_topics())

Estructura del ejemplo generado:
  messages[0]: system  — SYSTEM_PROMPT
  messages[1]: assistant — apertura (español + pregunta en inglés)
  messages[2]: user      — respuesta del estudiante (con errores inyectados)
  messages[3]: assistant — feedback (cita errores LT + follow-up question)

Uso:
    python generate_dataset_english_tutor.py
    python generate_dataset_english_tutor.py --target 20    # test rápido
    python generate_dataset_english_tutor.py --topics mi_topics.json

Salida: english_tutor_dataset.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from tqdm import tqdm
import requests
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/v1"
LT_URL       = "http://localhost:8081/v2/check"
MODEL        = "qwen2.5:7b"
TARGET       = 300
OUTPUT       = "english_tutor_dataset.jsonl"
CHECKPOINT   = 50
MAX_ERRORS   = 2   # igual que tutor_session.py

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# ── System prompt — IDÉNTICO al de tutor_session.py ─────────────────────────
SYSTEM_PROMPT = (
    "You are a warm, friendly spoken English tutor for Spanish-speaking B1/B2 students. "
    "You always speak naturally and conversationally, never using bullet points or markdown. "
    "You output a single spoken paragraph ready for Text-to-Speech. "
    "You NEVER mention grammar errors that are not explicitly listed in the prompt. "
    "English only (except 2-sentence Spanish intros). No Chinese characters."
)

# ── LanguageTool — IDÉNTICO al de tutor_session.py ───────────────────────────
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

# ── Instrucciones de error para inyectar en la respuesta del estudiante ───────
# El modelo genera una respuesta de estudiante con estos errores naturalmente
# embebidos — LT los detecta después y eso da el ground-truth para el feedback
ERROR_INSTRUCTIONS = [
    "Use 'should to' instead of 'should' before a verb (e.g. 'I should to go').",
    "Use 'can to' instead of 'can' before a verb (e.g. 'I can to help').",
    "Use 'will to' instead of 'will' before a verb.",
    "Use 'must to' instead of 'must' before a verb.",
    "Omit the article 'a' before a singular countable noun (e.g. 'She is doctor').",
    "Use 'an' instead of 'a' before a consonant sound (e.g. 'an university').",
    "Write 'informations' instead of 'information'.",
    "Write 'advices' instead of 'advice'.",
    "Write 'furnitures' instead of 'furniture'.",
    "Write 'equipments' instead of 'equipment'.",
    "Write 'loose' instead of 'lose' as a verb (e.g. 'I don't want to loose time').",
    "Use 'effect' as a verb instead of 'affect' (e.g. 'It effects my mood').",
    "Use 'don't' with a third-person singular subject (e.g. 'She don't know').",
    "Use 'have' instead of 'has' with a third-person singular subject.",
    "Use 'I am agree' instead of 'I agree'.",
    "Use Present Perfect with 'ago' (e.g. 'I have started this two years ago').",
    "Use Present Perfect with 'last year' (e.g. 'I have visited Paris last year').",
    "Use 'depend of' instead of 'depend on'.",
    "Use 'interested on' instead of 'interested in'.",
    "Use 'arrive to' instead of 'arrive at' or 'arrive in'.",
    "Misspell a common word slightly (e.g. 'recieve', 'definately', 'occured').",
    "Use 'it has' instead of 'there is' (e.g. 'It has a lot of people here').",
    "Use 'good in' instead of 'good at' when talking about skills.",
    "Use 'borrow' instead of 'lend'.",
    "Use 'make' instead of 'do' in a fixed expression (e.g. 'make homework').",
    "Use 'since' with a duration instead of 'for' (e.g. 'I have lived here since 3 years').",
]


# ── Cargar topics.json — igual que TutorSession.load_topics() ────────────────
def load_topics(path: str = "topics.json") -> list[str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            topics = json.load(f)
            print(f"[OK] topics.json cargado: {len(topics)} temas")
            return topics
    # Fallback si no existe el archivo
    print("[WARN] topics.json no encontrado — usando temas por defecto")
    return [
        "Artificial Intelligence", "Climate Change", "Social Media",
        "Remote Work", "Space Exploration", "Education Technology",
        "Healthy Habits", "Travel and Culture", "Music and Identity",
        "The Future of Cities", "Food and Traditions", "Sports and Teamwork",
        "Job Interviews", "Environmental Issues", "Digital Privacy",
        "Mental Health", "Urban Transportation", "Cinema and Series",
        "Entrepreneurship", "Volunteering", "Online Learning",
        "Cultural Differences", "Renewable Energy", "Automation and Jobs",
        "Globalization", "Tourism", "Video Games", "Fashion",
        "Cooking and Nutrition", "Public Transportation",
    ]


# ── LanguageTool — IDÉNTICO a tutor_session.py:check_errors() ────────────────
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


# ── errors_to_block — IDÉNTICO a tutor_session.py ────────────────────────────
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


# ── _clean — IDÉNTICO a tutor_session.py ─────────────────────────────────────
def _clean(text: str) -> str:
    text = re.sub(r'[#*\\]', '', text)
    text = re.sub(r'(Assistant|User|Tutor|Student)\s*:', '', text, flags=re.I)
    return " ".join(text.split()).strip()


# ── generate_opening — IDÉNTICO a tutor_session.py ───────────────────────────
def generate_opening(topic: str) -> str:
    prompt = (
        f"Topic: '{topic}'. "
        f"Write 2 short sentences in SPANISH introducing this topic naturally. "
        f"Then write ONE open-ended question in ENGLISH for a B1/B2 student. "
        f"No labels, no markdown. Ready for Text-to-Speech."
    )
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        text = res.choices[0].message.content or ""
        if re.search(r'[\u4e00-\u9fff]', text):
            return ""
        return _clean(text)
    except Exception as e:
        print(f"  [LLM ERROR] generate_opening: {e}")
        return ""


# ── generate_student_response — genera respuesta B1/B2 con errores inyectados
def generate_student_response(topic: str, opening: str) -> str:
    n      = random.randint(1, 2)
    errors = "\n".join(f"  - {e}" for e in random.sample(ERROR_INSTRUCTIONS, n))
    prompt = (
        f"You are a Spanish-speaking B1/B2 English student responding to your tutor.\n\n"
        f"TUTOR SAID: {opening}\n\n"
        f"Write a natural 4-6 sentence response about '{topic}'. "
        f"Each sentence should be at least 10 words. "
        f"Include EXACTLY these grammar mistakes embedded naturally "
        f"(do NOT label or highlight them):\n"
        f"{errors}\n\n"
        f"Rules:\n"
        f"- English ONLY. Every single word must be English. "
        f"Do NOT mix in Spanish words — not even proper nouns or place names.\n"
        f"- Do NOT invent words. Use only real English words, even if spelled wrong.\n"
        f"- No markdown. Sound like a real B1/B2 student sharing their opinion."
    )
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        text = res.choices[0].message.content or ""
        if re.search(r'[\u4e00-\u9fff]', text):
            return ""
        return _clean(text)
    except Exception as e:
        print(f"  [LLM ERROR] generate_student_response: {e}")
        return ""


# ── Verificador híbrido: regex instantáneo + LLM solo para lo ambiguo ────────
#
# Estrategia:
#   1. Regex detecta los errores que tienen forma léxica fija (strings exactos).
#      Es instantáneo, 100% confiable para esos patrones, 0 llamadas LLM.
#   2. LLM (checklist YES/NO, temperatura 0) solo para patrones que requieren
#      entender el contexto gramatical y el regex no puede resolver.
#
# Si cualquiera de los dos niveles encuentra algo → descartar el ejemplo.

# ── Nivel 1: regex — detección por string exacto ─────────────────────────────
# Cada entrada: (nombre, compiled_regex)
# Se aplica sobre el texto en minúsculas.
REGEX_CHECKS = [
    # Uncountables pluralizados — presencia del token es suficiente
    ("uncountable_informations", re.compile(r'\binformations\b')),
    ("uncountable_advices",      re.compile(r'\badvices\b')),
    ("uncountable_furnitures",   re.compile(r'\bfurnitures\b')),
    ("uncountable_equipments",   re.compile(r'\bequipments\b')),
    ("uncountable_knowledges",   re.compile(r'\bknowledges\b')),
    # Modales + to (can to, should to, must to, will to, would to, may to)
    ("modal_to", re.compile(
        r'\b(can|could|should|must|will|would|may|might|shall)\s+to\s+\w'
    )),
    # I am agree / I am disagree
    ("i_am_agree", re.compile(r'\bi\s+am\s+(dis)?agree\b')),
    # Present Perfect con marcador de pasado cerrado — dos sub-patrones:
    # a) "have/has ... last year/week/yesterday/in YYYY"
    # b) "have/has ... N years/months/days ago" (N = dígito o número escrito)
    ("present_perfect_past_a", re.compile(
        r"\b(have|has)\b.{0,80}(last\s+\w+|yesterday|\bin\s+\d{4}\b)",
        re.IGNORECASE
    )),
    ("present_perfect_past_b", re.compile(
        r"\b(have|has)\b.{0,80}"
        r"(two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(years?|months?|days?|hours?|minutes?)\s+ago",
        re.IGNORECASE
    )),
    # "good in" hablando de habilidades (good in + gerundio o sustantivo)
    ("good_in",  re.compile(r'\bgood\s+in\s+\w')),
    # "borrow" usado donde debería ser "lend" (borrow + OI personal)
    # heurística: "borrow you/him/her/them/us" o "borrow me"
    ("borrow_lend",   re.compile(r'\bborrow\s+(you|him|her|them|us|me)\b')),
    # Preposiciones incorrectas frecuentes que LT a veces pierde por contexto
    ("depend_of",     re.compile(r'\bdepend\s+of\b')),
    ("interested_on", re.compile(r'\binterested\s+on\b')),
    ("arrive_to",     re.compile(r'\barrive\s+to\b')),
]



def regex_check(text: str) -> list[str]:
    """Devuelve lista de check_names que matchearon. Vacía = texto limpio."""
    t = text.lower()
    return [name for name, pattern in REGEX_CHECKS if pattern.search(t)]


# ── Nivel 2: LLM checklist YES/NO — solo para lo que el regex no puede ver ───
# "loose" como verbo y "effect" como verbo requieren contexto semántico.
# "she don't / he have" requieren identificar el sujeto.
LLM_CHECKLIST = [
    (
        "loose_lose",
        'Does the text use "loose" as a verb meaning to lose, waste, or miss something? '
        'Examples: "loose time", "loose track", "loose weight", "loose the game". '
        'Answer YES or NO only.',
    ),
    (
        "affect_effect",
        'Does the text use "effect" as a verb? '
        'Examples: "it effects me", "this effects us", "effects my mood". '
        'Answer YES or NO only.',
    ),
    (
        "subject_verb_agreement",
        "Does the text have a subject-verb agreement error with a third-person singular subject? "
        'Examples: "she don\'t know", "he have a car", "it don\'t work". '
        "Answer YES or NO only.",
    ),
]


def llm_verify_errors(student_text: str) -> list[str]:
    """
    Nivel 2: checklist LLM binario (YES/NO) para patrones que requieren
    comprensión semántica. Solo se llama cuando LT devuelve 0 errores Y
    el regex_check también pasó (evitar llamadas innecesarias).
    Devuelve lista de check_names con YES.
    """
    found = []
    for check_name, question in LLM_CHECKLIST:
        prompt = f'Read this English text:\n"{student_text}"\n\n{question}'
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            answer = (res.choices[0].message.content or "").strip().upper()
            if answer.startswith("YES"):
                found.append(check_name)
        except Exception:
            pass
    return found



# ── _already_in_lt — evita falsos positivos en el regex post-LT ─────────────
# Si LT ya catcheó el error que regex detecta, no descartar el ejemplo.
# Mapea nombre de regex → rule_ids de LT que cubren ese patrón.
_REGEX_TO_LT_RULES = {
    "uncountable_informations": {"INFORMATIONS"},
    "uncountable_advices":      {"ADVICES"},
    "uncountable_furnitures":   {"FURNITURES"},
    "uncountable_equipments":   {"EQUIPMENTS"},
    "uncountable_knowledges":   {"KNOWLEDGES"},
    "modal_to":                 {"TO_AFTER_MODAL_VERBS"},
    "i_am_agree":               {"I_AM_AGREE", "I_AM_VB"},
    "present_perfect_past_a":   {"PRESENT_PERFECT_FOR_PAST", "SINCE_FOR_AGO"},
    "present_perfect_past_b":   {"PRESENT_PERFECT_FOR_PAST", "SINCE_FOR_AGO"},
    "good_in":                  set(),   # LT no cubre este — nunca se considera cubierto
    "borrow_lend":              {"BORROW_LEND"},
    "depend_of":               {"DEPEND_ON"},
    "interested_on":           set(),   # LT raramente lo cubre
    "arrive_to":               set(),
}


def _already_in_lt(regex_name: str, lt_errors: list) -> bool:
    """True si LT ya reportó un error que cubre lo que el regex detectó."""
    lt_rule_ids = {e.get("rule_id", "") for e in lt_errors}
    covered_by  = _REGEX_TO_LT_RULES.get(regex_name, set())
    return bool(lt_rule_ids & covered_by)


# ── generate_feedback — IDÉNTICO a tutor_session.py:generate_feedback() ──────
# (sin history — cada ejemplo es un turno único, igual que el primer turno real)
def generate_feedback(topic: str, student_text: str, errors: list[dict]) -> str:
    error_block = errors_to_block(errors)

    if errors:
        instruction = f"""You are giving spoken feedback to a B1/B2 English student.

TOPIC: {topic}
STUDENT RESPONSE: {student_text}

{error_block}

INSTRUCTIONS:
1. Acknowledge the student's idea in ONE short positive sentence (max 8 words, not about grammar).
2. For each confirmed error, say naturally:
   "In your sentence [quote the student sentence briefly], you said '[wrong phrase]' but it should be '[correct form]' because [short reason]."
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

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": instruction},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        text = res.choices[0].message.content or ""
        if re.search(r'[\u4e00-\u9fff]', text):
            return ""
        return _clean(text)
    except Exception as e:
        print(f"  [LLM ERROR] generate_feedback: {e}")
        return ""


# ── QC ────────────────────────────────────────────────────────────────────────
def quality_check(opening: str, student: str, feedback: str,
                  lt_errors: list) -> tuple[bool, str]:
    if not opening or not student or not feedback:
        return False, "Empty field"
    if len(feedback) < 80:
        return False, f"Feedback too short ({len(feedback)} chars)"
    if "?" not in feedback:
        return False, "No follow-up question in feedback"
    if len(student) < 120:
        return False, f"Student response too short ({len(student)} chars)"
    if re.search(r'[\u4e00-\u9fff]', opening + student + feedback):
        return False, "Chinese characters"
    if re.search(r'[*#`]', feedback):
        return False, "Markdown in feedback"

    # El feedback no debe reproducir >15 palabras consecutivas del estudiante
    fb_words = feedback.lower().split()
    st_lower  = student.lower()
    for i in range(len(fb_words) - 15):
        if " ".join(fb_words[i:i + 15]) in st_lower:
            return False, "Feedback reproduces student text verbatim"

    # Si LT encontró errores, el feedback debe referenciarlos
    if lt_errors:
        fb_lower = feedback.lower()
        if not any(k in fb_lower for k in ("said", "noticed", "used", "wrote", "sentence")):
            return False, "Feedback doesn't address confirmed errors"

    return True, "OK"


# ── Guardar ───────────────────────────────────────────────────────────────────
def _save(examples: list) -> None:
    with open(OUTPUT, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(target: int, topics_path: str, resume: bool = False) -> None:

    # Verificar servicios
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        print("[OK] Ollama conectado")
    except Exception:
        print("[ERROR] Ollama no está corriendo."); sys.exit(1)

    try:
        requests.get("http://localhost:8081/v2/languages", timeout=3)
        print("[OK] LanguageTool conectado en puerto 8081")
    except Exception:
        print("[ERROR] LanguageTool no está corriendo en puerto 8081.")
        print("Arrancalo con:")
        print("  java -cp languagetool-server.jar "
              "org.languagetool.server.HTTPServer \\")
        print("       --port 8081 --allow-origin '*' --public")
        sys.exit(1)

    topics = load_topics(topics_path)

    # ── Resume: contar ejemplos ya guardados ─────────────────────────────────
    already_done = 0
    if resume and os.path.exists(OUTPUT):
        with open(OUTPUT, "r", encoding="utf-8") as f:
            already_done = sum(1 for line in f if line.strip())
        print(f"[RESUME] {already_done} ejemplos ya guardados — "
              f"generando {max(0, target - already_done)} más hasta llegar a {target}")
        if already_done >= target:
            print(f"[INFO] Ya tenés {already_done} ejemplos, objetivo cumplido.")
            return
    elif os.path.exists(OUTPUT) and not resume:
        os.remove(OUTPUT)
        print(f"[INFO] Output anterior eliminado: {OUTPUT}")

    passed, failed, attempts = [], 0, 0
    # Contador de errores LT para el reporte final
    lt_dist = {0: 0, 1: 0, 2: 0}
    effective_target = max(0, target - already_done)

    with tqdm(total=effective_target, desc="english_tutor") as pbar:
        while len(passed) < effective_target:
            attempts += 1
            topic = random.choice(topics)

            # ── Etapa 1: apertura ────────────────────────────────────────────
            opening = generate_opening(topic)
            if not opening:
                failed += 1; continue

            # ── Etapa 2: respuesta del estudiante (con errores inyectados) ───
            student = generate_student_response(topic, opening)
            if not student:
                failed += 1; continue

            # ── Etapa 3: LT detecta errores reales (ground-truth) ────────────
            lt_errors = check_errors(student)

            # ── Etapa 3b: verificación secundaria ────────────────────────
            # El regex corre SIEMPRE (no solo cuando LT da 0) porque LT puede
            # detectar 1 error y perder otro en el mismo texto (ej: depend_of +
            # equipments). Si el regex encuentra algo que LT no catcheó → descartar.
            regex_found = [r for r in regex_check(student)
                           if not _already_in_lt(r, lt_errors)]
            if regex_found:
                failed += 1
                continue

            # LLM checklist solo cuando LT dio 0 (loose/lose, effect/affect, SVA)
            if not lt_errors:
                llm_found = llm_verify_errors(student)
                if llm_found:
                    failed += 1
                    continue

            # ── Etapa 4: feedback basado en errores LT ───────────────────────
            feedback = generate_feedback(topic, student, lt_errors)
            if not feedback:
                failed += 1; continue

            # ── QC ───────────────────────────────────────────────────────────
            ok, reason = quality_check(opening, student, feedback, lt_errors)
            if not ok:
                failed += 1; continue

            # ── Ejemplo válido ───────────────────────────────────────────────
            n_errors = len(lt_errors)
            lt_dist[min(n_errors, 2)] += 1

            ex = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "assistant", "content": opening},
                    {"role": "user",      "content": student},
                    {"role": "assistant", "content": feedback},
                ],
                "_debug": {
                    "mode":         "english_tutor",
                    "topic":        topic,
                    "lt_errors":    lt_errors,
                    "lt_verified":  True,   # LLM confirmó que no hay errores ocultos
                    "qc_reason":    reason,
                }
            }
            passed.append(ex)
            pbar.update(1)
            pbar.set_postfix(
                fail=failed,
                lt=f"{n_errors}err",
                rate=f"{len(passed)/attempts*100:.0f}%"
            )

            # Checkpoint
            if len(passed) % CHECKPOINT == 0:
                _save(passed[-CHECKPOINT:])
                print(f"\n  [CHECKPOINT] {len(passed)}/{target} guardados")

            # Safety — evitar loop infinito
            if attempts > effective_target * 7:
                print(f"\n[WARN] Demasiados intentos, parando con {len(passed)}")
                break

    # Guardar el resto
    remainder = len(passed) % CHECKPOINT
    if remainder:
        _save(passed[-remainder:])

    # ── Estadísticas finales ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Passed:       {len(passed)}")
    print(f"  Failed:       {failed}")
    print(f"  Attempts:     {attempts}")
    print(f"  Success rate: {len(passed)/max(attempts,1)*100:.1f}%")
    print(f"  LT 0 errors:  {lt_dist[0]} ejemplos ({lt_dist[0]/max(len(passed),1)*100:.0f}%)")
    print(f"  LT 1 error:   {lt_dist[1]} ejemplos ({lt_dist[1]/max(len(passed),1)*100:.0f}%)")
    print(f"  LT 2 errors:  {lt_dist[2]} ejemplos ({lt_dist[2]/max(len(passed),1)*100:.0f}%)")
    print(f"  Topics used:  {len(topics)} disponibles")
    print(f"  Output:       {OUTPUT}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset del English Tutor usando la misma lógica que tutor_session.py"
    )
    parser.add_argument("--target", type=int, default=TARGET,
                        help=f"Número de ejemplos total (default: {TARGET})")
    parser.add_argument("--topics", type=str, default="topics.json",
                        help="Ruta al archivo topics.json (default: topics.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Retomar desde ejemplos ya guardados sin borrar el output")
    args = parser.parse_args()
    main(target=args.target, topics_path=args.topics, resume=args.resume)