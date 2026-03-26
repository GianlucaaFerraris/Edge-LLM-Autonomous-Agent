# tutor_session.py — Technical Documentation

## Overview

`tutor_session.py` is the core module for the English conversation tutor within Agenty. It operates as a stateful session manager that drives a spoken B1/B2 English practice loop for Spanish-speaking learners. The module is designed to run standalone (`python tutor_session.py`) or be orchestrated by `main.py` as one of several switchable modes.

The fundamental architectural decision: **grammar error detection is fully delegated to LanguageTool (LT), never to the LLM**. The LLM's role is restricted to generating natural, pedagogically sound feedback and follow-up questions based on errors that LT has already confirmed. This separation prevents LLM hallucination of grammar corrections and keeps inference focused on generation quality rather than detection accuracy.

---

## Dependencies

| Component | Role | Address |
|---|---|---|
| Ollama (OpenAI-compat API) | LLM inference | `http://localhost:11434/v1` |
| LanguageTool HTTP server | Grammar detection | `http://localhost:8081/v2/check` |
| `topics.json` | Topic pool (599 entries) | Resolved via directory tree walk |

**Python packages:** `openai`, `requests` (no heavy ML dependencies in this module).

---

## Module-level Constants

```python
MAX_ERRORS     = 2          # Max grammar errors surfaced per turn
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"
```

`MAX_ERRORS = 2` is a deliberate pedagogical constraint: overwhelming a student with 5+ corrections in a single turn is counterproductive. The pipeline collects all LT matches, ranks them by pedagogical priority, and reports only the top 2.

---

## Architecture & Data Flow

### Full Turn Pipeline

```
Student input (text)
        │
        ▼
  ┌─────────────────────────────────────────────────┐
  │  Language detection heuristic (_is_likely_english)   │  ← in main.py
  │  Word count pre-check (≥20 words → skip LLM)    │  ← in intent_classifier.py
  └─────────────────────────────────────────────────┘
        │
        ├──── Spanish detected ──────────────────► skip LT, errors = []
        │
        ▼
  check_errors(text)      ← LanguageTool HTTP POST
        │
        ├── collect ALL matches (no early break)
        ├── filter BLACKLIST_RULES
        ├── filter non-RELEVANT_RULES and non-RELEVANT_CATEGORIES
        ├── filter morphological false positives (prefix heuristic)
        ├── deduplicate by wrong token (seen_wrong set)
        ├── stable sort by PRIORITY tier (grammar > tense > article > spelling)
        └── return top MAX_ERRORS
        │
        ▼
  generate_feedback(topic, history, student_text, errors)
        │
        ├── errors present  → correction prompt (acknowledge + correct + follow-up)
        └── no errors       → positive reinforcement prompt (highlight + follow-up)
        │
        ▼
  TutorSession._add_history()   ← sliding window, keeps last 4 messages
  TutorSession.speak()          ← TTS hook
  TutorSession._save_log()      ← JSONL append on session end
```

---

## Key Components

### `check_errors(text: str) → list[dict]`

The critical detection layer. Sends the student's text to the local LT server and applies a multi-stage filter pipeline.

**Priority tiers** (lower = higher pedagogical value, reported first):

| Tier | Rule type | Examples |
|---|---|---|
| 1 | Subject-verb agreement | `HE_VERB_AGR`, `SVA`, `NON3PRS_VERB` |
| 2 | Modal verbs, tense errors | `MD_BASEFORM`, `SINCE_FOR_AGO`, `I_AM_AGREE` |
| 3 | Articles, uncountable nouns, confused words | `EN_A_VS_AN`, `INFORMATIONS`, `LOOSE_LOSE` |
| 4 | Spelling | `MORFOLOGIK_RULE_EN_US`, `HUNSPELL_RULE` |

**Morphological false positive filter:** LT's spelling rules occasionally suggest a derivational stem as the "correction" (e.g., `protagonized → protagonist`). The heuristic detects this by checking if one form is a prefix of the other. When true, the match is discarded silently. This only applies to `MORFOLOGIK_RULE_EN_US` and `HUNSPELL_RULE`; grammar rules are never filtered this way.

**Blacklisted rules** (noise from Whisper transcription, not real errors):

```python
UPPERCASE_SENTENCE_START, PUNCTUATION_PARAGRAPH_END,
COMMA_PARENTHESIS_WHITESPACE, EN_QUOTES, WHITESPACE_RULE,
DOUBLE_PUNCTUATION, ENGLISH_WORD_REPEAT_RULE, SENTENCE_WHITESPACE,
UNIT_SPACE, CURRENCY, EN_UNPAIRED_BRACKETS
```

**Returned dict schema:**
```python
{
    "wrong":    str,   # exact token as it appears in student text
    "correct":  str,   # first LT replacement suggestion
    "sentence": str,   # full sentence containing the error (for context injection)
    "reason":   str,   # LT's natural language explanation
    "rule_id":  str,   # LT rule ID for logging and analysis
}
```

---

### `generate_feedback(topic, history, student_text, errors) → str`

LLM generation call with two prompt branches:

**With errors:**
```
Acknowledge idea (≤8 words) → 
For each error: "In [sentence], you said '[wrong]' but it should be '[correct]' because [reason]." → 
ONE follow-up question
```

**Without errors:**
```
Acknowledge quality (≤10 words) → 
Highlight one phrase the student used well + explain why → 
ONE follow-up question
```

The system prompt enforces TTS-readiness (single spoken paragraph, no markdown, no bullet points, English only). The `SYSTEM_PROMPT` constant explicitly forbids the LLM from inventing grammar corrections beyond what is listed — this is the primary guard against LLM hallucination in the feedback path.

History is limited to the last 4 messages (`history[-4:]` in the prompt) to stay within the context budget of a 7B quantized model.

---

### `TutorSession` Class

Stateful session object. Instantiated once and optionally paused/resumed by `main.py` (session is not destroyed on mode switch).

**Key attributes:**

| Attribute | Type | Description |
|---|---|---|
| `topics` | `list[str]` | Full topic pool loaded from `topics.json` |
| `topic` | `str` | Current active topic |
| `used_topics` | `set[str]` | Topics used this session (prevents repeats) |
| `history` | `list[dict]` | Sliding window, last 4 messages |
| `turn_count` | `int` | Total turns this session |
| `error_log` | `list[dict]` | All detected errors with turn and topic metadata |
| `lt_ok` | `bool` | Whether LT is reachable (set by `main.py` at session start) |

**Topic management:**

`_pick_new_topic()` samples from `topics \ used_topics`. When the pool is exhausted, `used_topics` resets and sampling restarts from the full pool. This guarantees topic diversity across extended sessions.

**History sliding window:**

`_add_history()` enforces a hard cap of 4 messages. For a 7B INT4 model on RK3588, this keeps the prompt well within the context window and avoids the TTFT degradation that appears when the KV cache grows beyond ~512 tokens.

**Dual turn execution paths:**

- `_run_normal_turn(text)`: standalone mode — calls `check_errors()` internally.
- `_run_normal_turn_with_errors(text, errors)`: orchestrated mode — receives pre-filtered errors from `main.py` (which already ran the language detection heuristic). Avoids LT running on Spanish input.

---

### `_ask_topic_preference()`

Session initialization routine. Uses a single LLM call (temp=0, max_tokens=40) to classify the user's intent as `propose` (specific topic requested) or `random`. Falls back to random on any JSON parse failure. Empty input is interpreted as random without calling the LLM.

---

### Session Logging (`_save_log`)

Appends a JSONL record to `session_log.jsonl` at session end:

```json
{
  "turns": 12,
  "topics_covered": ["Ancient Ruins", "Social Media"],
  "total_errors": 7,
  "error_log": [
    {
      "turn": 3,
      "topic": "Ancient Ruins",
      "errors": [
        {"wrong": "the plot are", "correct": "the plot is", "rule_id": "SVA", ...}
      ]
    }
  ]
}
```

This log is the primary source for offline analysis and testing.

---

## Integration with `main.py`

When invoked from the orchestrator, the session lifecycle is:

1. `main.py` calls `ensure_running()` → starts LT subprocess if needed
2. `TutorSession()` is instantiated and stored in `ContextManager` (persists on mode switch)
3. `main.py` runs the language detection heuristic (`_is_likely_english`) before each turn
4. If English: calls `check_errors()`, passes result to `_run_normal_turn_with_errors()`
5. If Spanish: passes `errors=[]` directly (LT skipped)
6. On exit: `_save_log()` is called, `ensure_stopped()` shuts down LT subprocess

---

## Performance Characteristics (RK3588 / Qwen2.5 7B INT4)

| Operation | Typical latency |
|---|---|
| LT HTTP POST (local) | 50–150ms |
| `generate_opening` (LLM, 150 tok) | ~1.6–2.0s total |
| `generate_feedback` with errors (LLM, 300 tok) | ~3–5s total |
| `generate_feedback` no errors (LLM, 200 tok) | ~2–3s total |
| `_ask_topic_preference` LLM call (40 tok) | ~0.3–0.5s |

LT runs in a persistent JVM subprocess (managed by `language_tool_server.py`). The first call after JVM cold start takes ~15–25s; subsequent calls within the same session are consistently under 150ms.

---

## Known Limitations

**LT coverage ceiling:** LT rule-based detection is strong for morphosyntax (SVA, modals, tense markers, specific uncountable nouns) but has no coverage for:
- Collocations (`hits on the cinema` vs. `box office hits`)
- Register mismatches (`I would like to talk about` in a casual setting)
- Semantic fluency (`I preffer movies that shut my brain off` — lexically valid but stylistically weak)
- Pragmatic errors

These require an additional LLM pass for fluency scoring, which is not currently implemented due to latency constraints on the 7B model.

**Spelling correction quality:** `MORFOLOGIK_RULE_EN_US` suggestions are the nearest dictionary entry by edit distance. For Spanish-influenced nonce words (`protagonized`, `preffer`), the top suggestion is not always the intended word. The morphological prefix filter catches the worst cases but is not exhaustive. Genuine misspellings (`dont → don't`, `pirsuit → pursuit`) are handled correctly.

**Context window pressure:** The 4-message history cap means the LLM cannot track errors across more than 2 exchange rounds. A student who makes the same SVA mistake in turns 1 and 5 will not receive a "you've made this mistake before" comment.

---

## Testing Strategy

### 1. Unit Tests — `check_errors` pipeline

The most directly testable component. LT must be running on port 8081.

```python
# test_check_errors.py
import pytest
from tutor_session import check_errors

# Each case: (input_text, expected_rule_ids_in_result)
CASES = [
    # Grammar — tier 1 (should always surface over spelling)
    ("She don't know the answer.",           ["DOES_DOESN_T"]),
    ("He have a good idea.",                  ["HAVE_PART_AGREEMENT"]),
    ("The team are ready.",                   ["SVA"]),

    # Modal verbs — tier 2
    ("We should to practice every day.",      ["MD_BASEFORM"]),
    ("I have started this job two years ago.", ["SINCE_FOR_AGO"]),

    # Spelling — tier 4
    ("I dont like it.",                       ["MORFOLOGIK_RULE_EN_US"]),

    # Priority: grammar should appear before spelling
    ("He have a car and I dont care.",        # both present
     ["HAVE_PART_AGREEMENT"]),                # grammar must be first in result

    # Morphological false positive — should be filtered
    ("The movie was protagonized by The Rock.", []),  # no valid correction

    # Blacklist — should be empty
    ("she went to the store",                 []),    # UPPERCASE_SENTENCE_START filtered
]

@pytest.mark.parametrize("text,expected_rules", CASES)
def test_error_detection(text, expected_rules):
    errors = check_errors(text)
    detected_rules = [e["rule_id"] for e in errors]
    for rule in expected_rules:
        assert rule in detected_rules, f"Expected {rule} in {detected_rules} for: {text}"

def test_max_errors_cap():
    # Text with 5+ real errors — should never return more than MAX_ERRORS
    text = "She don't know and he have car and I am agree and they was there."
    errors = check_errors(text)
    assert len(errors) <= 2

def test_priority_ordering():
    # SVA (tier 1) must appear before spelling (tier 4)
    text = "The plot are dumb and I dont like it."
    errors = check_errors(text)
    assert len(errors) == 2
    assert errors[0]["rule_id"] in ("SVA", "HE_VERB_AGR", "DOES_DOESN_T", "HAVE_PART_AGREEMENT", "NON3PRS_VERB")
```

This gives you **concrete, measurable data**:
- Recall rate per rule category (how many known errors does LT actually catch?)
- False positive rate (how often does `check_errors` return something for clean input?)
- Priority ordering correctness (is tier 1 always surfaced over tier 4?)

---

### 2. Regression Tests — `generate_feedback` prompt compliance

These test LLM output quality and are inherently probabilistic, but at temp=0.0/0.2 the model is deterministic enough to be useful:

```python
def test_feedback_does_not_hallucinate_errors():
    """When errors=[], the LLM must not mention any grammar correction."""
    feedback = generate_feedback(
        topic="Social Media",
        history=[],
        student_text="I think social media has both positive and negative effects on teenagers.",
        errors=[]
    )
    # These phrases would indicate hallucinated corrections
    hallucination_markers = ["should be", "instead of", "you said", "correct form", "mistake"]
    for marker in hallucination_markers:
        assert marker.lower() not in feedback.lower(), \
            f"Possible hallucination: '{marker}' found in feedback with no errors"

def test_feedback_mentions_confirmed_error():
    errors = [{"wrong": "don't", "correct": "doesn't",
               "sentence": "She don't know the answer.",
               "reason": "3rd person singular requires -s.",
               "rule_id": "DOES_DOESN_T"}]
    feedback = generate_feedback("Daily Routines", [], "She don't know the answer.", errors)
    assert "doesn't" in feedback or "does not" in feedback.lower()
    assert "don't" in feedback  # should reference the wrong form too
```

---

### 3. End-to-End Session Benchmark

Run a scripted session using a fixed set of student inputs and measure:

```python
# benchmark_session.py
import time, json
from tutor_session import TutorSession, check_errors, generate_feedback

SCRIPTED_TURNS = [
    "I think the movies protagonized by The Rock are popular because people like action.",
    "The plot are always simple but the visual effects makes it exciting.",
    "I preffer movies with better story and good actors.",
    "Social media have a big influence on how we consume culture today.",
    "I am agree that cinema is a cultural mirror of society.",
]

results = []
for text in SCRIPTED_TURNS:
    t0 = time.perf_counter()
    errors = check_errors(text)
    lt_ms = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    feedback = generate_feedback("Cinema", [], text, errors)
    llm_ms = round((time.perf_counter() - t1) * 1000, 1)

    results.append({
        "input": text,
        "errors_detected": [(e["wrong"], e["rule_id"]) for e in errors],
        "lt_ms": lt_ms,
        "llm_ms": llm_ms,
        "feedback_word_count": len(feedback.split()),
    })

print(json.dumps(results, indent=2, ensure_ascii=False))
```

**Metrics you get from this:**
- Per-turn LT latency (ms)
- Per-turn LLM generation latency (ms)
- Errors detected vs. errors present (manual comparison)
- Feedback length distribution (are responses staying concise?)
- False negatives: errors present in input that `check_errors` missed

---

### 4. LT Coverage Analysis

Run the full `RELEVANT_RULES` dict against a known corpus of Spanish-influenced English errors to find which rules actually fire on your target population:

```python
# lt_coverage.py — Run once, save results to analyze coverage gaps
TEST_CORPUS = {
    "HE_VERB_AGR":   "She don't want to go.",
    "NON3PRS_VERB":  "He speak very fast.",
    "SINCE_FOR_AGO": "I have started this job two years ago.",
    "MD_BASEFORM":   "You should to eat more.",
    "I_AM_AGREE":    "I am agree with you.",
    "INFORMATIONS":  "Can you give me some informations?",
    "ADVICES":       "He gave me some good advices.",
    "LOOSE_LOSE":    "Don't loose your keys.",
    # ... one sentence per rule
}

for expected_rule, sentence in TEST_CORPUS.items():
    errors = check_errors(sentence)
    detected = [e["rule_id"] for e in errors]
    status = "✓" if expected_rule in detected else "✗ MISSED"
    print(f"{status}  {expected_rule}: '{sentence}' → {detected}")
```

This tells you exactly which rules in your `RELEVANT_RULES` dict LT actually catches in practice (some rules fire only under specific syntactic conditions).