import json
import random
import os
from tqdm import tqdm
from openai import OpenAI
import re

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

ERROR_CATEGORIES = [
    # 1. VERB TENSES & AGREEMENT (20+ variantes)
    "Subject-verb agreement (e.g., third person 's' missing or extra 's')",
    "Wrong past tense conjugation (regular vs irregular verbs)",
    "Present Perfect vs Past Simple confusion (e.g., 'I have seen him yesterday')",
    "Continuous vs Simple aspect (e.g., 'I am knowing the solution')",
    "Modal verb errors (e.g., 'must to do', 'can to go', 'should had')",
    "Passive voice construction errors in descriptions",
    
    # 2. FALSE FRIENDS & VOCABULARY (25+ variantes)
    "False friend: Actual/Actually vs Current/Currently",
    "False friend: Assist vs Attend (meetings/conferences)",
    "False friend: Library vs Bookstore / Record vs Remember",
    "False friend: Pretend vs Intend / Fabric vs Factory",
    "False friend: Argument vs Plot / Resume vs Summary",
    "Confusion between 'Make' and 'Do' in technical contexts (e.g., 'do an effort')",
    "Using Spanish-sounding words (Spanglish) like 'informatization' or 'personalization'",
    "Wrong word choice: 'Loose' vs 'Lose', 'Affect' vs 'Effect', 'Rise' vs 'Raise'",
    "Confusion between 'Economic' (cheap) and 'Economical' (related to economy)",

    # 3. PREPOSITIONS & PHRASAL VERBS (15+ variantes)
    "Preposition after verb (e.g., 'depend of', 'think in', 'discuss about')",
    "Time prepositions (in/on/at confusion for deadlines or dates)",
    "Place prepositions (at the office, in the screen, on the server)",
    "Phrasal verb literalism (e.g., 'turn down the server' when meaning 'shut down')",
    "Missing 'to' in infinitive purposes (e.g., 'I go for buy')",

    # 4. SYNTAX & SENTENCE STRUCTURE (15+ variantes)
    "Word order in questions (e.g., 'Why the system is slow?')",
    "Direct translation of 'hay' (e.g., 'It has many bugs' instead of 'There are')",
    "Adjective order (e.g., 'the system blue' instead of 'the blue system')",
    "Missing 'it' as a dummy subject (e.g., 'Is necessary to...')",
    "Double negatives (e.g., 'I don't know nothing')",
    "Incorrect use of 'agree' (e.g., 'I am agree', 'I'm agree')",
    "Uncountable noun pluralization (e.g., 'informations', 'knowledges', 'softwares', 'hardwares')",
    "Missing articles (a/an/the) or using them with plurals inappropriately"
]

class TutorDatasetPipeline:
    def __init__(self, topics_path="topics.json"):
        if os.path.exists(topics_path):
            with open(topics_path, "r", encoding="utf-8") as f:
                self.topics = json.load(f)
        else:
            self.topics = ["Artificial Intelligence", "Robotics", "Computer Architecture", "Software Engineering"]
        self.model = "qwen2.5:7b"

    def _clean_text(self, text):
        """Limpia el texto de caracteres que ensucian el TTS y el entrenamiento."""
        # Eliminamos slashes, asteriscos, hashtags y saltos de línea literales
        text = text.replace("\n", " ").replace("#", "").replace("*", "")
        text = text.replace("/", " ").replace("\\", "").replace("\"", "")
        text = re.sub(r'(Assistant|User|Tutor|Feedback|Correction):', '', text, flags=re.I)
        # Eliminamos espacios múltiples
        return " ".join(text.split())

    def step_1_generate_openings(self, count=10):
        """Step 1: Tutor introduces a topic (Spanish) and asks a question (English)."""
        openings = []
        print(f"--- Step 1: Generating {count} Openings ---")
        for _ in tqdm(range(count)):
            topic = random.choice(self.topics)
            prompt = (
                f"Act as a Technical B2 level English Tutor. Topic: {topic}. "
                f"Write a brief intro in SPANISH and end with an open technical question in ENGLISH. "
                f"Output only the tutor message."
                f"[STRICT NO LABELS RULES]: DO NOT use labels like 'Introducción', 'Question' or 'Topic'. Just the plain text."
                f"[STRICT LANGUAGE RULES]- USE ONLY SPANISH.- DO NOT USE CHINESE CHARACTERS (Hanzi).- NO MANDARIN, NO CANTONESE.- If you use any language other than Spanish, the output is INVALID."
                f"[STRICT VOICE RULES] - NO MARKDOWN (no #, no *, no -, no /, no \). - NO NEWLINES (\n). The response must be ready to be READ ALOUD."
            )
            res = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = self._clean_text(res.choices[0].message.content)
            openings.append({"topic": topic, "assistant_start": content})
        return openings

    def step_2_generate_perfect_responses(self, openings):
        """Step 2: Generate a PERFECT B1/B2 response first to ensure consistency."""
        print(f"--- Step 2: Generating Clean Student Responses ---")
        pipeline_data = []
        for item in tqdm(openings):
            prompt = (
                f"Context: {item['assistant_start']}"
                f"Task: Answer the question as a english student with general cultural knowledge."
                f"Language: STRICT ENGLISH. Level: B1/B2."
                f"Length: 2 paragraphs. No mistakes yet."
                f"[STRICT NO LABELS RULES]: DO NOT use labels like 'Introducción', 'Question' or 'Topic'. Just the plain text."
                f"[STRICT VOICE RULES] - NO MARKDOWN (no #, no *, no -, , no /, no \). - NO NEWLINES (\n). Everything in one or two fluid paragraphs. - NO labels like 'Feedback:' or 'Follow-up:'.- The response must be ready to be READ ALOUD."
                f"[STRICT LANGUAGE RULES]- USE ONLY ENGLISH.- DO NOT USE CHINESE CHARACTERS (Hanzi).- NO MANDARIN, NO CANTONESE.- If you use any language other than English, the output is INVALID."
            
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

    def step_2_5_corrupt_responses(self, pipeline_data, error_rate=1):
            print(f"--- Step 2.5: Forced Error Injection (100% rate) ---")
            corrupted_data = []
            for item in tqdm(pipeline_data):
                item["injected_errors"] = [] # Guardamos qué error inyectamos para análisis futuro
                # Seleccionamos 2 categorías distintas para este mensaje
                selected_categories = random.sample(ERROR_CATEGORIES, 2)
                item["injected_errors"] = selected_categories
                
                prompt = (
                    f"ORIGINAL TEXT: {item['perfect_response']}"
                    f"TASK: You are a student. REPLACE the ORIGINAL TEXT to include these 2 SPECIFIC mistakes:"
                    f"CATEGORY 1: {selected_categories[0]}"
                    f"CATEGORY 2: {selected_categories[1]}"
                    f"INSTRUCTION: Make the errors OBVIOUS. Do not just change style; BREAK the grammar or vocabulary according to the categories."
                    f"Output ONLY the corrupted text."
                    f"[STRICT VOICE RULES] - NO MARKDOWN (no #, no *, no -, no /, no \). - NO NEWLINES (\n). Everything in one or two fluid paragraphs. - NO labels like 'Feedback:' or 'Follow-up:'.- The response must be ready to be READ ALOUD."
                    f"[STRICT LANGUAGE RULES]- USE ONLY ENGLISH.- DO NOT USE CHINESE CHARACTERS (Hanzi).- NO MANDARIN, NO CANTONESE.- If you use any language other than English, the output is INVALID."
                )
                
                res = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6 # Un poco más alto para fomentar creatividad en el error
                )
                content = self._clean_text(res.choices[0].message.content)
                item["user_response"] = content
                item["has_errors"] = True
                corrupted_data.append(item)
            return corrupted_data

    def step_3_generate_feedback(self, corrupted_data):
        print(f"--- Step 3: Surgical Tutor Feedback ---")
        final_dataset = []
        for item in tqdm(corrupted_data):
            # Le damos como pista los errores que inyectamos para que el modelo se enfoque en explicarlos, pero sin decir explícitamente "Error/Correction"
            error_hint = ", ".join(item["injected_errors"]) if item["has_errors"] else "None"

            # Prompt diseñado para generar lenguaje hablado
            prompt = f"""
                [STUDENT TEXT]: "{item['user_response']}"
                [HAS ERRORS]: {item['has_errors']}
                [TARGET ERRORS TO FIND]: {error_hint}

                [TASK]
                1. Respond as a concise technical English tutor.
                2. First, briefly validate the student's technical idea in ENGLISH (max 8 words).
                3. Identify and fix ONLY the HARD ERRORS present in [STUDENT TEXT] from the [TARGET ERRORS TO FIND] list.
                4. Format: "I noticed you said "[error]", but it should be "[fix]" because [reason]."
                5. NEVER say "Your grammar is perfect" or "Great job" regarding grammar, as this text HAS ERRORS.
                6. FINALLY, ALWAYS end with a follow-up question in ENGLISH related to the original topic to keep the conversation going. Do not label it as "Follow-up", just write the question.

                [STRICT VOICE & TTS RULES]
                - NO markdown, NO labels, NO newlines.
                - NO stylistic pedantry. Focus on logic and syntax.
                - Limit the entire response to a single fluid paragraph.
                - The output must be ready for immediate Text-to-Speech synthesis.

                [NEGATIVE CONSTRAINTS]
                - DO NOT mention "I am agree" if it is not in [STUDENT TEXT].
                - DO NOT correct words like "issue", "profoundly", or "provide" if they are used correctly.
                - DO NOT quote more than 5 words at a time.

                [STRICT LANGUAGE RULES]- USE ONLY ENGLISH.- DO NOT USE CHINESE CHARACTERS (Hanzi).- NO MANDARIN, NO CANTONESE.- If you use any language other than English, the output is INVALID.
            """
            
            res = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a friendly, spoken-word English (B2 level) tutor. You talk, you don't write lists."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            clean_content = self._clean_text(res.choices[0].message.content)

            formatted = {
                "messages": [
                    {"role": "assistant", "content": item["assistant_start"]},
                    {"role": "user", "content": item["user_response"]},
                    {"role": "assistant", "content": clean_content}
                ]
            }
            final_dataset.append(formatted)
        return final_dataset

def main():
    pipeline = TutorDatasetPipeline()
    
    data = pipeline.step_1_generate_openings(10)
    data = pipeline.step_2_generate_perfect_responses(data)
    data = pipeline.step_2_5_corrupt_responses(data)
    final_jsonl = pipeline.step_3_generate_feedback(data)
    
    output_file = "tutor_dataset_v3_clean.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in final_jsonl:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n[SUCCESS] Dataset generated: {output_file}")

if __name__ == "__main__":
    main()