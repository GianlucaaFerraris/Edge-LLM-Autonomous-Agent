import json
import random
import os
import re
from tqdm import tqdm
from openai import OpenAI

from context_aware_injector import context_aware_inject, CONTEXT_AWARE_CATEGORIES
from evidence_validator import validate_evidence_list

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def load_error_categories(path: str = "error_categories_pipeline.json") -> list:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = [(item[0], item[1], item[2]) for item in data if len(item) >= 3]
        print(f"[OK] Loaded {len(categories)} error categories from {path}")
        return categories
    else:
        print(f"[WARN] {path} not found — using built-in fallback categories.")
        return FALLBACK_ERROR_CATEGORIES


FALLBACK_ERROR_CATEGORIES = [
    ("Subject-verb agreement (3rd person)",
     "Remove the 's' from a third-person singular verb.",
     'Change "Technology plays a role" to "Technology play a role"'),
    ("Modal verb + to",
     "Add 'to' right after a modal verb like can, must, should, or will.",
     'Change "this can help" to "this can to help"'),
    ("Stative verb in continuous form",
     "Use a stative verb (know, understand, believe, have, need) in the -ing form.",
     'Change "I understand the concept" to "I am understanding the concept"'),
    ("Present Perfect + specific past time marker",
     "Use Present Perfect with a specific past time marker like yesterday, last year.",
     'Change "I learned about this last year" to "I have learned about this last year"'),
    ("Wrong preposition after verb",
     "Change the preposition after a verb to a wrong one.",
     'Change "it depends on the context" to "it depends of the context"'),
    ("Missing dummy subject it",
     "Remove the subject 'it' from a sentence that needs it as a dummy subject.",
     'Change "It is important to practice" to "Is important to practice"'),
    ("Uncountable noun pluralized",
     "Add a plural -s to an uncountable noun like information, knowledge, advice.",
     'Change "more information" to "more informations"'),
    ("Missing article (a/an/the)",
     "Remove a necessary article before a singular countable noun.",
     'Change "it was a great experience" to "it was great experience"'),
]


class TutorDatasetPipeline:
    def __init__(self, topics_path: str = "topics.json",
                 categories_path: str = "error_categories_pipeline.json"):
        if os.path.exists(topics_path):
            with open(topics_path, "r", encoding="utf-8") as f:
                self.topics = json.load(f)
        else:
            self.topics = [
                "Artificial Intelligence", "Remote Work", "Climate Change",
                "Social Media", "Space Exploration", "Education Technology"
            ]
        self.error_categories = load_error_categories(categories_path)
        self.model = "qwen2.5:7b"

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ").replace("#", "").replace("*", "")
        text = text.replace("/", " ").replace("\\", "").replace('"', "")
        text = re.sub(r'(Assistant|User|Tutor|Feedback|Correction|Student|Response)\s*:', '', text, flags=re.I)
        return " ".join(text.split()).strip()

    def _extract_json(self, raw: str) -> dict:
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

    def _validate_evidence(self, rewritten_text: str, evidence: list) -> list:
        """Verifica que cada 'wrong' phrase exista verbatim en el texto."""
        valid = []
        for ev in evidence:
            wrong = ev.get("wrong", "").strip()
            if wrong and wrong.lower() in rewritten_text.lower():
                valid.append(ev)
            else:
                print(f"  [VERBATIM DROP] '{wrong}' not found in rewritten text.")
        return valid

    def step_1_generate_openings(self, count: int = 10):
        openings = []
        print(f"\n--- Step 1: Generating {count} Openings ---")
        for _ in tqdm(range(count)):
            topic = random.choice(self.topics)
            prompt = (
                f"You are a friendly English tutor. Topic: '{topic}'. "
                f"Write 2 sentences in SPANISH introducing the topic naturally. "
                f"Then write ONE open-ended question in ENGLISH for a B1/B2 student. "
                f"No labels, no markdown, no bullet points, no newlines. "
                f"Output must be ready for Text-to-Speech. "
                f"Use ONLY Spanish (intro) and English (question). No Chinese characters."
            )
            res = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = self._clean_text(res.choices[0].message.content)
            openings.append({"topic": topic, "assistant_start": content})
        return openings

    def step_2_generate_perfect_responses(self, openings: list):
        print(f"\n--- Step 2: Generating Clean Student Responses ---")
        pipeline_data = []
        for item in tqdm(openings):
            prompt = (
                f"A tutor said: {item['assistant_start']} "
                f"Write a B1/B2 English student response. Two paragraphs. Zero grammar mistakes. "
                f"Natural spoken style. No markdown, no labels, no newlines. "
                f"ENGLISH ONLY. No Chinese characters."
            )
            res = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            content = self._clean_text(res.choices[0].message.content)
            item["perfect_response"] = content
            pipeline_data.append(item)
        return pipeline_data

    def _try_inject_normal(self, text: str, categories: list) -> tuple[str, list]:
        """
        Intenta inyectar errores de las categorías dadas en el texto.
        Devuelve (rewritten, evidence) o ("", []) si falla.
        """
        error_block = ""
        for i, (name, instruction, example) in enumerate(categories, 1):
            error_block += (
                f"ERROR {i}: {name}.\n"
                f"  How to inject: {instruction}\n"
                f"  Example: {example}\n"
            )

        prompt = f"""You are rewriting a student text to introduce specific grammar mistakes for a training dataset.

ORIGINAL TEXT:
{text}

ERRORS TO INJECT:
{error_block}

CRITICAL RULES:
- Before injecting each error, check that the original text has a sentence where the error fits naturally.
- If the text does NOT have a suitable sentence for a given error, SKIP that error entirely.
- Do NOT add new sentences to the text in order to force an error.
- Do NOT change any part of the text other than what is needed for the error.
- The wrong phrase must appear VERBATIM in the rewritten text — it will be verified automatically.
- Keep the meaning and overall length similar to the original.
- Output ONLY a JSON object. No explanation outside the JSON.

JSON FORMAT:
{{
  "rewritten": "full rewritten text with injected errors",
  "errors": [
    {{"wrong": "exact phrase as it appears in rewritten text", "correct": "what it should be", "reason": "one sentence explanation for a language learner"}}
  ]
}}"""

        res = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        parsed = self._extract_json(res.choices[0].message.content)
        if parsed and "rewritten" in parsed and "errors" in parsed:
            return self._clean_text(parsed["rewritten"]), parsed.get("errors", [])
        return "", []

    def step_2_5_corrupt_responses(self, pipeline_data: list):
        print(f"\n--- Step 2.5: Controlled Error Injection + Semantic Validation ---")
        corrupted_data = []
        injected_count, fallback_count = 0, 0
        semantic_drops_total = 0
        MAX_RETRIES = 3  # intentos con categorías distintas antes de declarar fallback

        # Categorías "confiables": alta probabilidad de aparecer en cualquier texto
        RELIABLE_CATEGORIES = {
            "Stative verb in continuous: 'know'",
            "Stative verb in continuous: 'understand'",
            "Stative verb in continuous: 'want'",
            "Stative verb in continuous: 'like'",
            "Stative verb in continuous: 'love'",
            "Stative verb in continuous: 'need'",
            "Stative verb in continuous: 'believe'",
            "Modal verb error: 'can to'",
            "Modal verb error: 'should to'",
            "Modal verb error: 'could to'",
            "Modal verb error: 'will to'",
            "Modal verb error: 'might to'",
            "Subject-verb agreement: missing 3rd person -s with action verbs",
            "Subject-verb agreement: 3rd person singular with 'have'",
            "Missing article: omitting 'a' before singular countable noun",
            "Missing article: omitting 'an' before vowel-sound noun",
            "Wrong article: 'the' before abstract noun used generally",
            "Uncountable noun pluralized: 'informations'",
            "Uncountable noun pluralized: 'advices'",
            "Uncountable noun pluralized: 'researches' (as mass noun)",
            "Syntax error: missing dummy subject \'it\'",
            "Syntax error: \'It has\' instead of \'There is/are\'",
            "Wrong preposition: \'depend of\' instead of \'depend on\'",
            "Wrong preposition: \'interested on\' instead of \'interested in\'",
            "Wrong preposition: \'good in\' instead of \'good at\'",
        }

        reliable_pool = [cat for cat in self.error_categories if cat[0] in RELIABLE_CATEGORIES]
        if not reliable_pool:
            reliable_pool = self.error_categories  # fallback si el set no matchea

        for item in tqdm(pipeline_data):
            item["corruption_mode"] = "controlled"
            semantic_valid = []
            attempt = 0
            used_category_names = set()

            while not semantic_valid and attempt < MAX_RETRIES:
                attempt += 1
                all_evidence = []
                final_rewritten = item["perfect_response"]

                # En el primer intento: categoría aleatoria normal
                # En reintentos: forzar categoría confiable no usada antes
                if attempt == 1:
                    n_errors = random.randint(1, 2)
                    available = [c for c in self.error_categories if c[0] not in used_category_names]
                    selected = random.sample(available, min(n_errors, len(available)))
                else:
                    available_reliable = [c for c in reliable_pool if c[0] not in used_category_names]
                    if not available_reliable:
                        break
                    selected = [random.choice(available_reliable)]
                    print(f"  [RETRY {attempt}] Trying reliable category: {selected[0][0]}")

                used_category_names.update(c[0] for c in selected)
                item["injected_errors"] = [cat[0] for cat in selected]

                ca_categories = [cat for cat in selected if cat[0] in CONTEXT_AWARE_CATEGORIES]
                normal_categories = [cat for cat in selected if cat[0] not in CONTEXT_AWARE_CATEGORIES]

                # Context-aware
                for cat in ca_categories:
                    result = context_aware_inject(final_rewritten, cat[0])
                    if result:
                        final_rewritten = result["rewritten"]
                        all_evidence.extend(result["errors"])

                # Normal injection
                if normal_categories:
                    rewritten, evidence = self._try_inject_normal(final_rewritten, normal_categories)
                    if rewritten:
                        final_rewritten = rewritten
                        all_evidence.extend(evidence)

                # Validación verbatim
                final_rewritten = self._clean_text(final_rewritten)
                verbatim_valid = self._validate_evidence(final_rewritten, all_evidence)

                # Validación semántica
                if verbatim_valid:
                    semantic_valid, dropped = validate_evidence_list(verbatim_valid, final_rewritten, [c[0] for c in selected])
                    semantic_drops_total += len(dropped)

            if semantic_valid:
                item["user_response"] = final_rewritten
                item["evidence"] = semantic_valid
                item["has_errors"] = True
                injected_count += 1
            else:
                item["user_response"] = item["perfect_response"]
                item["evidence"] = []
                item["has_errors"] = False
                item["corruption_mode"] = "fallback"
                fallback_count += 1

            corrupted_data.append(item)

        print(f"  Injected: {injected_count} | Fallbacks: {fallback_count} | Semantic drops: {semantic_drops_total}")
        return corrupted_data

    def step_3_generate_feedback(self, corrupted_data: list):
        print(f"\n--- Step 3: Surgical Tutor Feedback ---")
        final_dataset = []
        for item in tqdm(corrupted_data):

            if item["has_errors"] and item["evidence"]:
                corrections = " | ".join(
                    [f'Student said "{ev["wrong"]}" → should be "{ev["correct"]}" — reason: {ev["reason"]}'
                     for ev in item["evidence"]]
                )
                prompt = f"""You are a warm, spoken English tutor giving feedback to a B1/B2 student.

STUDENT TEXT:
{item["user_response"]}

CONFIRMED ERRORS (these are the ONLY errors you may mention — do not invent others):
{corrections}

INSTRUCTIONS:
1. In one short sentence, acknowledge the student's idea positively (max 8 words, unrelated to grammar).
2. For each confirmed error above, say naturally:
   "I noticed you said [quote ≤5 words from wrong phrase], but it should be [correct phrase] because [reason]."
3. End with one follow-up question in English about the topic. No label, just the question.

STRICT OUTPUT RULES:
- One single fluid spoken paragraph.
- No lists, no markdown, no labels, no newlines.
- Quote at most 5 words at a time when referencing the student.
- English only. No Chinese characters. Ready for Text-to-Speech."""

            else:
                prompt = f"""You are a warm, spoken English tutor.

STUDENT TEXT:
{item["user_response"]}

This student response contains no grammar errors.
1. Acknowledge the quality of the answer briefly (max 10 words).
2. Pick one specific phrase the student used well and briefly say why it works in English.
3. End with one follow-up question related to the topic.

STRICT OUTPUT RULES:
- One single fluid spoken paragraph.
- No lists, no markdown, no newlines.
- English only. Ready for Text-to-Speech."""

            res = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a friendly spoken English tutor. "
                        "You speak naturally, warmly, and concisely. "
                        "You never use bullet points or markdown. "
                        "You always output a single spoken paragraph."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            clean_content = self._clean_text(res.choices[0].message.content)

            final_dataset.append({
                "messages": [
                    {"role": "assistant", "content": item["assistant_start"]},
                    {"role": "user", "content": item["user_response"]},
                    {"role": "assistant", "content": clean_content}
                ],
                "_debug": {
                    "topic": item["topic"],
                    "corruption_mode": item.get("corruption_mode", "unknown"),
                    "injected_errors": item.get("injected_errors", []),
                    "evidence": item.get("evidence", [])
                }
            })
        return final_dataset


# -----------------------------------------------------------------------------
# QUALITY FILTER
# -----------------------------------------------------------------------------

def _feedback_reproduces_student_text(feedback: str, student: str,
                                       max_consecutive_words: int = 20) -> bool:
    feedback_words = feedback.lower().split()
    student_lower = student.lower()
    for i in range(len(feedback_words) - max_consecutive_words + 1):
        window = " ".join(feedback_words[i:i + max_consecutive_words])
        if window in student_lower:
            return True
    return False


def quality_check(entry: dict) -> tuple[bool, str]:
    feedback = entry["messages"][2]["content"]
    student = entry["messages"][1]["content"]
    assistant_start = entry["messages"][0]["content"]
    debug = entry.get("_debug", {})

    if re.search(r'[\u4e00-\u9fff]', feedback + student + assistant_start):
        return False, "Chinese characters detected"
    if len(feedback) < 80:
        return False, f"Feedback too short ({len(feedback)} chars)"
    if "?" not in feedback:
        return False, "No follow-up question in feedback"
    if len(student) < 100:
        return False, f"Student response too short ({len(student)} chars)"

    for ev in debug.get("evidence", []):
        wrong = ev.get("wrong", "").strip().lower()
        correct = ev.get("correct", "").strip().lower()
        if wrong and correct and wrong == correct:
            return False, f"Evidence wrong==correct: '{ev.get('wrong')}'"

    if _feedback_reproduces_student_text(feedback, student, max_consecutive_words=20):
        return False, "Feedback reproduces >20 consecutive words from student text"

    noticed_count = feedback.lower().count("i noticed you said")
    evidence_count = len(debug.get("evidence", []))
    if evidence_count > 0 and noticed_count > evidence_count + 1:
        return False, f"Tutor overcorrected: {noticed_count} for {evidence_count} evidence items"

    return True, "OK"


def main():
    pipeline = TutorDatasetPipeline()

    data = pipeline.step_1_generate_openings(count=10)
    data = pipeline.step_2_generate_perfect_responses(data)
    data = pipeline.step_2_5_corrupt_responses(data)
    final_dataset = pipeline.step_3_generate_feedback(data)

    passed, failed = [], []
    for entry in final_dataset:
        ok, reason = quality_check(entry)
        if ok:
            passed.append(entry)
        else:
            entry.setdefault("_debug", {})["fail_reason"] = reason
            failed.append(entry)

    print(f"\n[QC] Passed: {len(passed)} / {len(final_dataset)}  |  Failed: {len(failed)}")

    modes = {}
    for entry in passed:
        mode = entry["_debug"].get("corruption_mode", "unknown")
        modes[mode] = modes.get(mode, 0) + 1
    print(f"[QC] Breakdown: {modes}")

    clean_file = "tutor_dataset_v5_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as f:
        for entry in passed:
            f.write(json.dumps({"messages": entry["messages"]}, ensure_ascii=False) + "\n")

    debug_file = "tutor_dataset_v5_debug.jsonl"
    with open(debug_file, "w", encoding="utf-8") as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[OK] Clean  → {clean_file}")
    print(f"[OK] Debug  → {debug_file}")

    if failed:
        print(f"\n[WARN] {len(failed)} failed QC:")
        for entry in failed:
            print(f"  - [{entry['_debug'].get('corruption_mode', '?')}] "
                  f"{entry['_debug'].get('fail_reason', '?')} "
                  f"| {entry['_debug'].get('topic', '?')}")


if __name__ == "__main__":
    main()