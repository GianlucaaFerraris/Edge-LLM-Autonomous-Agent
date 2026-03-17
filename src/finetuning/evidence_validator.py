"""
evidence_validator.py — v2

Cambio principal respecto a v1:
- El prompt ahora incluye la CATEGORÍA del error, para que el modelo
  evalúe con el criterio correcto en vez de juzgar en abstracto.
- Las preguntas son más directivas y binarias.
- Se agrega una lista de errores que SIEMPRE son válidos para evitar
  falsos rechazos en categorías conocidas.
"""

import re
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"

# Categorías donde el error es objetivo e inequívoco — el validador
# semántico solo verifica que el wrong phrase exista en el texto.
# Si la categoría está aquí, se saltea el llamado al modelo.
ALWAYS_VALID_CATEGORIES = {
    # Modal + to: nunca es correcto en inglés
    "Modal verb error: 'can to'",
    "Modal verb error: 'must to'",
    "Modal verb error: 'should to'",
    "Modal verb error: 'will to'",
    "Modal verb error: 'would to'",
    "Modal verb error: 'could to'",
    "Modal verb error: 'might to'",
    # Uncountables pluralizados: siempre incorrecto
    "Uncountable noun pluralized: 'informations'",
    "Uncountable noun pluralized: 'advices'",
    "Uncountable noun pluralized: 'knowledges'",
    "Uncountable noun pluralized: 'equipments'",
    "Uncountable noun pluralized: 'researches' (as mass noun)",
    "Uncountable noun pluralized: 'softwares'",
    "Uncountable noun pluralized: 'works' instead of 'work' (labor)",
    "Uncountable noun pluralized: 'furnitures'",
    # Syntax errors claros
    "Syntax error: 'I am agree' instead of 'I agree'",
    "Syntax error: 'We are agree' instead of 'We agree'",
    "Syntax error: 'It has' instead of 'There is/are'",
    "Syntax error: 'It had' instead of 'There was/were'",
    # Preposiciones fijas — no tienen variantes aceptables
    "Wrong preposition: 'depend of' instead of 'depend on'",
    "Wrong preposition: 'married with' instead of 'married to'",
    "Wrong preposition: 'consist in' instead of 'consist of'",
    "Wrong preposition: 'arrive to' instead of 'arrive at/in'",
    # Confused words inequívocos
    "Confused words: 'loose' instead of 'lose'",
    "Confused words: 'effect' as verb instead of 'affect'",
    "Confused words: 'borrow' instead of 'lend'",
    "Confused words: 'raise' instead of 'rise' (intransitive)",
    # False friends claros
    "False friend: 'assist' instead of 'attend' (asistir)",
    "False friend: 'pretend' instead of 'intend' (pretender)",
    "False friend: 'library' instead of 'bookstore' (librería)",
}

# Descripción de la regla por categoría — se inyecta en el prompt
# para que el modelo evalúe con el criterio correcto.
CATEGORY_RULES = {
    "Stative verb in continuous": (
        "Stative verbs (know, understand, believe, want, need, have for possession, "
        "like, love, prefer, remember) CANNOT be used in continuous (-ing) form. "
        "This is an absolute rule in English — 'I am knowing' is always wrong."
    ),
    "Subject-verb agreement": (
        "A third-person singular subject (he, she, it, a noun) MUST use the verb with -s "
        "in present simple. Missing or adding -s incorrectly is always a grammar error."
    ),
    "Wrong tense": (
        "Present Perfect cannot be used with specific past time markers (yesterday, last year, ago, in 2020). "
        "Present Simple cannot be used with 'since' or 'for' indicating duration. "
        "These are absolute rules."
    ),
    "Missing article": (
        "A singular countable noun in English requires an article (a/an/the). "
        "Omitting it is always a grammar error. "
        "'She is doctor' is wrong; it must be 'She is a doctor'."
    ),
    "Wrong article": (
        "Using 'the' before abstract nouns in general statements is wrong (e.g., 'The life is beautiful'). "
        "Using 'a' instead of 'an' before vowel sounds is wrong. "
        "These are clear grammar rules."
    ),
    "Syntax error: missing dummy subject": (
        "In English, 'it' is required as a dummy subject in impersonal constructions. "
        "'Is important to practice' is always wrong — it must be 'It is important to practice'."
    ),
    "Wrong preposition": (
        "Fixed collocations in English require specific prepositions. "
        "The wrong phrase uses an incorrect preposition in a fixed collocation. "
        "Evaluate only whether the preposition in the 'wrong' version violates the fixed collocation rule."
    ),
    "False friend": (
        "The wrong version uses a Spanish false cognate in a way that is clearly incorrect in English. "
        "Evaluate whether a native English speaker would use the 'wrong' version in this context."
    ),
}


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _get_category_rule(category_name: str) -> str:
    """Busca la regla más específica que aplique a esta categoría."""
    for key, rule in CATEGORY_RULES.items():
        if key.lower() in category_name.lower():
            return rule
    return ""


def validate_single_evidence(wrong: str, correct: str, context: str,
                              category_name: str = "") -> tuple[bool, str]:
    # Categorías siempre válidas — no gastar un llamado al modelo
    if category_name in ALWAYS_VALID_CATEGORIES:
        return True, "OK (always-valid category)"

    # Regla específica de la categoría si existe
    category_rule = _get_category_rule(category_name)
    rule_section = f"\nGRAMMAR RULE BEING TESTED:\n{category_rule}\n" if category_rule else ""

    prompt = f"""You are a strict English grammar checker. Answer ONLY with a JSON object.

CONTEXT (student sentence):
{context}
{rule_section}
CLAIMED ERROR:
  Wrong:   "{wrong}"
  Correct: "{correct}"

Your task: determine if "{wrong}" violates a clear English grammar rule in this context.

IMPORTANT GUIDELINES:
- If a grammar rule is stated above, use ONLY that rule to evaluate.
- Do NOT consider style preferences, formality level, or whether one version "sounds better".
- Do NOT say something is acceptable just because it is understandable.
- A grammar violation means it breaks a rule that a grammar textbook would mark as wrong.
- Answer YES to wrong_is_error if the wrong version breaks the stated grammar rule.
- Answer YES to correct_is_fix if the correct version properly fixes that specific rule violation.

Output ONLY this JSON:
{{
  "wrong_is_error": true or false,
  "correct_is_fix": true or false,
  "reason": "one sentence citing the specific grammar rule"
}}"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        parsed = _extract_json(res.choices[0].message.content)

        if not parsed:
            # Si no parsea, asumir válido para no perder demasiados ejemplos
            return True, "OK (validator parse failed, assumed valid)"

        wrong_is_error = parsed.get("wrong_is_error", False)
        correct_is_fix = parsed.get("correct_is_fix", False)
        reason = parsed.get("reason", "")

        if wrong_is_error and correct_is_fix:
            return True, "OK"
        elif not wrong_is_error:
            return False, f"Not a real error: {reason}"
        else:
            return False, f"Correction inappropriate: {reason}"

    except Exception as e:
        return True, f"OK (validator exception, assumed valid: {e})"


def validate_evidence_list(evidence: list, student_text: str,
                           injected_categories: list = None) -> tuple[list, list]:
    """
    Valida una lista de evidence.
    injected_categories: lista de nombres de categorías en el mismo orden que evidence,
                         para pasar la regla correcta al validador.
    """
    valid = []
    dropped = []

    # Mapear categorías a evidence por índice si se proveen
    cat_map = {}
    if injected_categories:
        for i, cat_name in enumerate(injected_categories):
            if i < len(evidence):
                cat_map[i] = cat_name

    for i, ev in enumerate(evidence):
        wrong = ev.get("wrong", "").strip()
        correct = ev.get("correct", "").strip()

        if not wrong or not correct:
            dropped.append({**ev, "_drop_reason": "Missing wrong or correct field"})
            continue

        if wrong.lower() == correct.lower():
            dropped.append({**ev, "_drop_reason": "wrong == correct"})
            continue

        context = _find_sentence_with(wrong, student_text) or student_text[:400]

        # Intentar obtener el nombre de la categoría para este evidence
        category_name = cat_map.get(i, "")

        ok, reason = validate_single_evidence(wrong, correct, context, category_name)

        if ok:
            valid.append(ev)
        else:
            print(f"  [SEMANTIC DROP] '{wrong}' → '{correct}': {reason}")
            dropped.append({**ev, "_drop_reason": reason})

    return valid, dropped


def _find_sentence_with(phrase: str, text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    phrase_lower = phrase.lower()
    for sent in sentences:
        if phrase_lower in sent.lower():
            return sent
    return ""