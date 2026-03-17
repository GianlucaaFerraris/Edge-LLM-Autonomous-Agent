import json
import random
import os
import re
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

ERROR_CATEGORIES = [
    ("Subject-verb agreement", "third person singular 's' missing or wrongly added (e.g., 'Technology play' instead of 'Technology plays')"),
    ("Wrong past tense", "using base form instead of past tense for irregular verbs (e.g., 'she goed' instead of 'she went')"),
    ("Present Perfect vs Past Simple", "using Present Perfect with a specific past time marker (e.g., 'I have seen him yesterday')"),
    ("Stative verb in continuous", "using continuous form with stative verbs (e.g., 'I am knowing', 'I am understanding')"),
    ("Modal verb error", "adding 'to' after a modal verb (e.g., 'can to go', 'must to do', 'should to finish')"),
    ("False friend: actual/actually", "using 'actual' to mean 'current' or 'actually' to mean 'currently' (e.g., 'the actual president')"),
    ("False friend: assist/attend", "using 'assist' to mean 'attend a meeting' (e.g., 'I assisted the conference')"),
    ("False friend: pretend/intend", "using 'pretend' to mean 'intend' (e.g., 'I pretend to finish the project')"),
    ("Make vs Do confusion", "using 'do' with nouns that require 'make', or vice versa (e.g., 'do an effort' instead of 'make an effort')"),
    ("Wrong word: loose/lose", "using 'loose' instead of 'lose' or vice versa"),
    ("Wrong word: affect/effect", "using 'effect' as a verb or 'affect' as a noun in the wrong context"),
    ("Preposition after verb", "wrong preposition collocated with a verb (e.g., 'depend of' instead of 'depend on', 'think in' instead of 'think about')"),
    ("Time preposition", "using wrong preposition for time expressions (e.g., 'in Monday', 'at the morning', 'on the evening')"),
    ("Phrasal verb error", "translating a Spanish phrasal verb literally (e.g., 'turn down the computer' meaning 'shut down')"),
    ("Question word order", "Spanish word order in a direct question (e.g., 'Why the system is slow?' instead of 'Why is the system slow?')"),
    ("There is/are vs It has", "translating 'hay' as 'it has' (e.g., 'It has many options' instead of 'There are many options')"),
    ("Missing dummy subject 'it'", "omitting 'it' as a dummy subject (e.g., 'Is necessary to restart' instead of 'It is necessary to restart')"),
    ("I am agree", "using 'I am agree' instead of 'I agree'"),
    ("Uncountable noun pluralized", "adding plural -s to uncountable nouns (e.g., 'informations', 'knowledges', 'softwares', 'advices')"),
    ("Missing article", "omitting 'a', 'an', or 'the' where required (e.g., 'She is doctor' instead of 'She is a doctor')"),
]


class TutorDatasetPipeline:
    def __init__(self, topics_path="topics.json"):
        if os.path.exists(topics_path):
            with open(topics_path, "r", encoding="utf-8") as f:
                self.topics = json.load(f)
        else:
            self.topics = ["Artificial Intelligence", "Robotics", "Remote Work", "Climate Change"]
        self.model = "qwen2.5:7b"

    def _clean_text(self, text):
        text = text.replace("\n", " ").replace("#", "").replace("*", "")
        text = text.replace("/", " ").replace("\\", "").replace('"', "")
        text = re.sub(r'(Assistant|User|Tutor|Feedback|Correction|Student|Response):', '', text, flags=re.I)
        return " ".join(text.split())

    def _parse_json_response(self, raw: str) -> dict:
        """Extrae el primer bloque JSON válido de la respuesta del modelo."""
        raw = raw.strip()
        # Busca el primer { ... } en la respuesta
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    # -------------------------------------------------------------------------
    # STEP 1: Tutor introduces topic in Spanish and asks a question in English
    # -------------------------------------------------------------------------
    def step_1_generate_openings(self, count):
        openings = []
        print(f"--- Step 1: Generating {count} Openings ---")
        for _ in tqdm(range(count)):
            topic = random.choice(self.topics)
            prompt = (
                f"You are a spoken English tutor. Topic: {topic}. "
                f"Write ONE short paragraph in SPANISH introducing the topic naturally (2-3 sentences). "
                f"Then ask ONE open-ended question in ENGLISH for a B1/B2 student to answer. "
                f"Output only the tutor message. No labels, no markdown, no newlines. "
                f"Ready for Text-to-Speech. SPANISH intro + ENGLISH question. "
                f"DO NOT use Chinese characters."
            )
            res = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = self._clean_text(res.choices[0].message.content)
            openings.append({"topic": topic, "assistant_start": content})
        return openings

    # -------------------------------------------------------------------------
    # STEP 2: Generate a perfect B1/B2 student response (no errors yet)
    # -------------------------------------------------------------------------
    def step_2_generate_perfect_responses(self, openings):
        print(f"--- Step 2: Generating Clean Student Responses ---")
        pipeline_data = []
        for item in tqdm(openings):
            prompt = (
                f"A tutor said: {item['assistant_start']} "
                f"Write a B1/B2 student answer in ENGLISH. 2 paragraphs. No grammar mistakes. "
                f"No markdown, no labels, no newlines. Spoken style, ready for TTS. "
                f"DO NOT use Chinese characters."
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

    # -------------------------------------------------------------------------
    # STEP 2.5: Inject errors AND extract the exact wrong/right phrase pairs
    # This is the KEY fix: we get the error evidence in a structured way
    # so step_3 doesn't have to "find" anything — it just explains.
    # -------------------------------------------------------------------------
    def step_2_5_corrupt_responses(self, pipeline_data):
        print(f"--- Step 2.5: Error Injection + Evidence Extraction ---")
        corrupted_data = []
        for item in tqdm(pipeline_data):
            # Pick 1-2 error categories
            n_errors = random.randint(1, 2)
            selected = random.sample(ERROR_CATEGORIES, n_errors)
            item["injected_errors"] = [cat[0] for cat in selected]

            # Build the injection prompt asking for structured JSON output
            error_instructions = "\n".join(
                [f"ERROR {i+1}: {cat[0]} — {cat[1]}" for i, cat in enumerate(selected)]
            )

            prompt = f"""ORIGINAL TEXT: "{item['perfect_response']}"

                TASK: Rewrite the text introducing these SPECIFIC grammar errors. Then report exactly what you changed.

                ERRORS TO INJECT:
                {error_instructions}

                RULES:
                - Each error must appear clearly in the rewritten text.
                - Do not add extra errors beyond the list.
                - Return ONLY valid JSON, no markdown, no explanation outside the JSON.

                OUTPUT FORMAT (strict JSON):
                {{
                "corrupted_text": "...the full rewritten text with errors...",
                "evidence": [
                    {{"error_category": "...", "wrong_phrase": "...", "correct_phrase": "...", "reason": "..."}}
                ]
                }}"""

            res = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            raw = res.choices[0].message.content
            parsed = self._parse_json_response(raw)

            if parsed and "corrupted_text" in parsed and "evidence" in parsed:
                item["user_response"] = self._clean_text(parsed["corrupted_text"])
                item["evidence"] = parsed["evidence"]  # [{wrong_phrase, correct_phrase, reason}]
                item["has_errors"] = True
            else:
                # Fallback: use perfect response unchanged and mark as no errors
                # (rare, happens when model fails JSON format)
                print(f"  [WARN] JSON parse failed for one item, using perfect response as fallback.")
                item["user_response"] = item["perfect_response"]
                item["evidence"] = []
                item["has_errors"] = False

            corrupted_data.append(item)
        return corrupted_data

    # -------------------------------------------------------------------------
    # STEP 3: Generate surgical tutor feedback using the extracted evidence
    # The model doesn't need to "find" errors — we tell it exactly what's wrong
    # -------------------------------------------------------------------------
    def step_3_generate_feedback(self, corrupted_data):
        print(f"--- Step 3: Surgical Tutor Feedback ---")
        final_dataset = []
        for item in tqdm(corrupted_data):

            if item["has_errors"] and item["evidence"]:
                # Build a clear, unambiguous correction list for the model
                correction_lines = []
                for ev in item["evidence"]:
                    wrong = ev.get("wrong_phrase", "")
                    correct = ev.get("correct_phrase", "")
                    reason = ev.get("reason", "")
                    correction_lines.append(
                        f'- Student said "{wrong}" → should be "{correct}" because {reason}.'
                    )
                corrections_block = "\n".join(correction_lines)

                prompt = f"""You are a friendly spoken English tutor. A student answered:

                    STUDENT TEXT: "{item['user_response']}"

                    These are the CONFIRMED errors (do NOT look for other errors):
                    {corrections_block}

                    TASK:
                    1. Briefly acknowledge the student's idea (max 8 words, positive but not about grammar).
                    2. For EACH confirmed error, say: "I noticed you said [wrong phrase], but it should be [correct phrase] because [reason]."
                    3. End with ONE follow-up question in English related to the topic to keep the conversation going. Do not label it.

                    STRICT RULES:
                    - Output a single fluid paragraph. No lists, no markdown, no newlines, no labels.
                    - Quote only 1-5 words at a time when referencing the student.
                    - Do NOT say "your grammar is perfect" or "no mistakes".
                    - Ready for Text-to-Speech.
                    - ENGLISH ONLY. No Chinese characters."""

            else:
                # No errors — tutor gives positive feedback and a follow-up
                prompt = f"""You are a friendly spoken English tutor. A student answered perfectly:

                    STUDENT TEXT: "{item['user_response']}"

                    TASK:
                    1. Acknowledge the quality of the answer briefly (max 10 words).
                    2. Mention one specific thing they said well.
                    3. End with ONE follow-up question related to the topic.

                    STRICT RULES:
                    - Single fluid paragraph, no markdown, no labels, no newlines.
                    - ENGLISH ONLY. No Chinese characters. Ready for TTS."""

            res = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a friendly spoken-word English tutor at B2 level. You give corrections in a warm, natural conversational style."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2  # Low temp = consistent format
            )
            clean_content = self._clean_text(res.choices[0].message.content)

            formatted = {
                "messages": [
                    {"role": "assistant", "content": item["assistant_start"]},
                    {"role": "user", "content": item["user_response"]},
                    {"role": "assistant", "content": clean_content}
                ]
            }
            # Optional: save debug metadata separately
            formatted["_debug"] = {
                "topic": item["topic"],
                "injected_errors": item.get("injected_errors", []),
                "evidence": item.get("evidence", [])
            }
            final_dataset.append(formatted)
        return final_dataset


def main():
    pipeline = TutorDatasetPipeline()

    data = pipeline.step_1_generate_openings(count=10)
    data = pipeline.step_2_generate_perfect_responses(data)
    data = pipeline.step_2_5_corrupt_responses(data)
    final_jsonl = pipeline.step_3_generate_feedback(data)

    # Save clean dataset (for fine-tuning)
    output_file = "tutor_dataset_v4_clean.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in final_jsonl:
            entry_clean = {k: v for k, v in entry.items() if not k.startswith("_")}
            f.write(json.dumps(entry_clean, ensure_ascii=False) + "\n")

    # Save debug dataset (for inspection)
    debug_file = "tutor_dataset_v4_debug.jsonl"
    with open(debug_file, "w", encoding="utf-8") as f:
        for entry in final_jsonl:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Clean dataset: {output_file}")
    print(f"[SUCCESS] Debug dataset (with evidence): {debug_file}")


if __name__ == "__main__":
    main()