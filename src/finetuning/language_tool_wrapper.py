"""
language_tool_wrapper.py

Wrapper para el servidor local de LanguageTool.
Recibe texto transcripto (Whisper → str), llama al servidor HTTP,
filtra los errores relevantes para el tutor de inglés B1/B2 para
hispanohablantes, y devuelve una lista limpia lista para pasar al LLM.

Uso:
    from language_tool_wrapper import check_errors
    errors = check_errors("She don't know the answer")
    # → [{"wrong": "don't", "correct": "doesn't", "category": "Subject-verb agreement", "rule_id": "HE_VERB_AGR"}]
"""

import requests
import json

LT_URL = "http://localhost:8081/v2/check"

# ---------------------------------------------------------------------------
# Mapa de rule_id de LanguageTool → categoría del tutor
# Solo los errores relevantes para hispanohablantes B1/B2.
# Cualquier rule_id que NO esté aquí se descarta silenciosamente.
# ---------------------------------------------------------------------------
RELEVANT_RULES = {
    # Subject-verb agreement
    "HE_VERB_AGR":              "Subject-verb agreement: 3rd person singular",
    "HAVE_PART_AGREEMENT":      "Subject-verb agreement: have/has",
    "SVA":                      "Subject-verb agreement",
    "NON3PRS_VERB":             "Subject-verb agreement: missing 3rd person -s",
    "DOES_DOESN_T":             "Subject-verb agreement: do/does",

    # Modal verbs
    "MD_BASEFORM":              "Modal verb error: infinitive after modal",

    # Articles
    "EN_A_VS_AN":               "Wrong article: 'a' instead of 'an'",
    "R_MISSING_ARTICLE":        "Missing article",
    "MISSING_DETERMINER":       "Missing article",

    # Prepositions
    "PREPOSITION_VERB":         "Wrong preposition after verb",
    "I_AM_AGREE":               "Syntax error: 'I am agree'",

    # Tense errors
    "SINCE_FOR_AGO":            "Wrong tense: Present Perfect with past marker",
    "PRESENT_PERFECT_FOR_PAST": "Wrong tense: Present Perfect with specific past time",
    "MORFOLOGIK_RULE_EN_US":    "Spelling error",

    # Uncountable nouns — LanguageTool tiene reglas específicas para algunos
    "ADVICES":                  "Uncountable noun pluralized: 'advices'",
    "INFORMATIONS":             "Uncountable noun pluralized: 'informations'",
    "EQUIPMENTS":               "Uncountable noun pluralized: 'equipments'",
    "FURNITURES":               "Uncountable noun pluralized: 'furnitures'",

    # Confused words
    "LOOSE_LOSE":               "Confused words: 'loose' instead of 'lose'",
    "AFFECT_EFFECT":            "Confused words: 'effect' as verb instead of 'affect'",
    "BORROW_LEND":              "Confused words: 'borrow' instead of 'lend'",

    # Syntax
    "THERE_IS_NO_VERB":         "Syntax error: missing dummy subject 'it'",
    "IT_IS":                    "Syntax error: missing dummy subject 'it'",

    # Double negative
    "DOUBLE_NEGATIVE":          "Syntax error: double negative",
}

# Rule IDs que LanguageTool detecta pero que NO son errores reales
# en el contexto de un hispanohablante B1/B2 — falsos positivos comunes.
BLACKLIST_RULES = {
    "UPPERCASE_SENTENCE_START",   # Whisper a veces no capitaliza
    "PUNCTUATION_PARAGRAPH_END",  # ídem
    "COMMA_PARENTHESIS_WHITESPACE",
    "EN_QUOTES",
    "WHITESPACE_RULE",
    "DOUBLE_PUNCTUATION",
    "ENGLISH_WORD_REPEAT_RULE",   # repetición no siempre es error
    "SENTENCE_WHITESPACE",
    "UNIT_SPACE",
    "CURRENCY",
}

# Categorías de LanguageTool que SÍ queremos aunque no estén en RELEVANT_RULES
# (captura reglas gramaticales que no conocemos de antemano)
RELEVANT_CATEGORIES = {
    "GRAMMAR",
    "CONFUSED_WORDS",
    "TYPOS",
}


def check_errors(text: str, max_errors: int = 2) -> list[dict]:
    """
    Llama al servidor LanguageTool y devuelve hasta `max_errors` errores
    filtrados y formateados para el LLM.

    Args:
        text: Texto transcripto del estudiante.
        max_errors: Máximo de errores a reportar (evita overwhelm al estudiante).

    Returns:
        Lista de dicts con keys: wrong, correct, category, rule_id, reason.
        Lista vacía si no hay errores relevantes o el servidor no responde.
    """
    try:
        response = requests.post(
            LT_URL,
            data={"language": "en-US", "text": text},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        print("[LT] ERROR: LanguageTool server not running on port 8081.")
        return []
    except Exception as e:
        print(f"[LT] ERROR: {e}")
        return []

    matches = data.get("matches", [])
    errors = []

    for match in matches:
        rule_id = match.get("rule", {}).get("id", "")
        category_id = match.get("rule", {}).get("category", {}).get("id", "")

        # Descartar blacklist primero
        if rule_id in BLACKLIST_RULES:
            continue

        # Aceptar si está en RELEVANT_RULES o en RELEVANT_CATEGORIES
        is_relevant = (rule_id in RELEVANT_RULES) or (category_id in RELEVANT_CATEGORIES)
        if not is_relevant:
            continue

        # Extraer el texto incorrecto del contexto
        context_text = match.get("context", {}).get("text", "")
        offset = match.get("context", {}).get("offset", 0)
        length = match.get("context", {}).get("length", 0)
        wrong_phrase = context_text[offset: offset + length] if context_text else ""

        if not wrong_phrase:
            continue

        # Mejor corrección disponible (primera sugerencia de LT)
        replacements = match.get("replacements", [])
        correct_phrase = replacements[0].get("value", "") if replacements else ""

        if not correct_phrase:
            continue

        # Categoría para el LLM
        category = RELEVANT_RULES.get(rule_id, match.get("rule", {}).get("description", "Grammar error"))

        # Razón en lenguaje natural para el estudiante
        reason = match.get("message", "This is a grammar error.")

        errors.append({
            "wrong": wrong_phrase,
            "correct": correct_phrase,
            "category": category,
            "rule_id": rule_id,
            "reason": reason,
        })

        if len(errors) >= max_errors:
            break

    return errors


def errors_to_prompt_block(errors: list[dict]) -> str:
    """
    Convierte la lista de errores en un bloque de texto para incluir
    en el prompt del LLM.

    Returns:
        String formateado, o string vacío si no hay errores.
    """
    if not errors:
        return ""

    lines = ["CONFIRMED GRAMMAR ERRORS (mention ONLY these):"]
    for ev in errors:
        lines.append(
            f'  - Student said "{ev["wrong"]}" → correct form: "{ev["correct"]}" '
            f'— reason: {ev["reason"]}'
        )
    return "\n".join(lines)


def errors_to_json_file(errors: list[dict], path: str = "current_errors.json") -> None:
    """Guarda los errores en un archivo JSON para debugging o logging."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"errors": errors, "error_count": len(errors)}, f,
                  ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Test rápido — correr directamente para verificar la integración
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TEST_SENTENCES = [
        "She don't know the answer.",
        "I have started this job two years ago.",
        "We should to practice more every day.",
        "I need more informations about this topic.",
        "This will effect the final result.",
        "I am agree with your point.",
        "He have a good idea about the project.",
        "I think in it very often.",
        "The team are ready to start.",
        "I loose my keys every morning.",
    ]

    print("=" * 60)
    print("LanguageTool Wrapper — Test")
    print("=" * 60)

    for sentence in TEST_SENTENCES:
        errors = check_errors(sentence)
        print(f"\nINPUT:  {sentence}")
        if errors:
            for e in errors:
                print(f"  ✗ '{e['wrong']}' → '{e['correct']}'")
                print(f"    [{e['rule_id']}] {e['category']}")
        else:
            print("  ✓ No relevant errors detected")