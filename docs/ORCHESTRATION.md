# Orchestration Layer

> **Files:** `intent_classifier.py`, `orchestrator.py`, `context_manager.py`  
> **Responsibilities:** Intent detection, mode routing, session state management, greeting/clarification generation  
> **Classification method:** Hybrid — keyword safety net + single LLM call with structured JSON output

---

## 1. Overview

The orchestration layer is the "switchboard" of the system. Every user utterance passes through it to determine: what does the user want to do, and which subsystem should handle it? The design must satisfy three constraints simultaneously:

1. **Latency:** Classification must be fast (<2s) because it adds to every turn's total response time.
2. **Accuracy:** Misrouting a request (e.g., sending an English practice response to the engineering tutor) breaks the user experience.
3. **Robustness:** On a 7B model, classification prompts must be simple enough to produce reliable structured output.

---

## 2. Intent Classification (`intent_classifier.py`)

### 2.1 Design Evolution

The system went through two classification architectures:

**v1 (replaced):** Dual classification — `orchestrator.detect_intent()` for idle mode + `detect_intent_from_active_mode()` for active modes + per-session classifiers. This required multiple LLM calls per turn and had inconsistent behavior across modes.

**v2 (current):** **Single unified classifier** per context. One LLM call receives the full context (active mode, current topic, user text) and returns a structured JSON response. The valid actions are parameterized by mode, so the model only sees options that make sense for the current state.

### 2.2 Classification by Mode

The classifier exposes three public functions, one per operational context:

#### `classify_idle(text) → IntentResult`

Called when no mode is active (main menu). Valid actions:

| Action | Meaning |
|---|---|
| `english` | User wants to practice English |
| `engineering` | User has a technical/science question |
| `agent` | User wants a productivity action (task, calendar, etc.) |
| `unclear` | Ambiguous intent — need clarification |

#### `classify_english(text, current_topic) → IntentResult`

Called during an active English tutor session. Valid actions:

| Action | Meaning |
|---|---|
| `respond` | Continue the English conversation |
| `change_topic` | Random topic change ("otro tema", "next topic") |
| `propose_topic` | Specific topic request ("let's talk about sports") |
| `switch_engineering` | Switch to engineering tutor |
| `agent` | Agent interrupt (task, calendar, reminder, etc.) |
| `exit_mode` | Return to main menu |
| `exit_app` | Close the application |

This is the most complex classifier because it must distinguish between **responding to a conversation** (the default, most common action) and **meta-commands** (changing topics, switching modes). The prompt includes explicit examples and a critical disambiguation rule:

> "propose_topic ONLY applies when the message IS a request to change topic — a short, direct instruction. If the message is an elaborate response (≥15 words), it is ALWAYS respond."

#### `classify_engineering(text) → IntentResult`

Called during an active engineering tutor session. Valid actions:

| Action | Meaning |
|---|---|
| `respond` | Technical question or continuation |
| `switch_english` | Switch to English tutor |
| `agent` | Agent interrupt |
| `exit_mode` | Return to main menu |
| `exit_app` | Close the application |

### 2.3 Safety Net (Keyword Pre-Check)

Before calling the LLM, `_safety_check()` handles trivially obvious intents via string matching:

```python
_EXIT_APP_WORDS = {"chau", "cerrar", "quit"}
_EXIT_GENERIC = {"salir", "exit"}
_EXIT_MODE_WORDS = {"menu", "menú", "volver al menú", "stop"}
```

- **From idle:** "salir"/"exit" → `exit_app`
- **From active mode:** "salir"/"exit" → `exit_mode` (return to menu, not close app)
- **From anywhere:** "chau"/"cerrar"/"quit" → `exit_app`

This costs <0.01ms and prevents the LLM from misinterpreting exit commands — a failure mode observed during testing where the model would classify "salir" as "agent" because it interpreted it as "the user wants to do something."

### 2.4 Length Heuristic (English Mode)

A word-count heuristic provides a second layer of pre-filtering for the English classifier:

```python
if word_count >= 20:
    return IntentResult(action="respond", confidence="high")
```

**Rationale:** A response of 20+ words is virtually always a conversational response, not a meta-command. Topic changes and mode switches are expressed in short phrases ("otro tema", "pasame a ingeniería"). This heuristic eliminates an entire class of false positives where the LLM would incorrectly classify an elaborate English response as `propose_topic` because it mentioned a new subject.

A secondary check catches borderline cases (12–19 words):
```python
if action in ("propose_topic", "change_topic") and word_count >= 12:
    action = "respond"
```

### 2.5 LLM JSON Output

The LLM is prompted to return JSON:

```
Respondé SOLO con JSON válido, sin markdown:
{"action": "...", "confidence": "high"}
```

The `_llm_json()` function handles common LLM output quirks:
1. Strips markdown code fences (` ```json `)
2. Strips backticks
3. Parses JSON
4. Falls back to `{"action": "continue", "confidence": "high"}` on any failure

The `temperature=0.0` setting ensures deterministic classification — the same input always produces the same intent (modulo floating-point nondeterminism in the attention computation).

### 2.6 IntentResult Dataclass

```python
@dataclass
class IntentResult:
    action:     str          # the classified intent
    confidence: str          # "high" | "low"
    topic:      str          # extracted topic (for propose_topic)
    question:   str          # clarification question (if confidence="low")
    raw:        dict         # raw LLM JSON output (for debugging)
```

When the LLM returns `confidence: "low"`, it also provides a `question` field with a clarification prompt in Spanish. The orchestrator displays this question and uses `confirm_intent()` to interpret the user's yes/no response.

### 2.7 Confirmation Flow

```python
def confirm_intent(question: str, user_answer: str) -> bool:
```

1. **Keyword matching:** "sí"/"dale"/"claro" → True; "no"/"nope" → False
2. **LLM fallback:** For ambiguous answers, a minimal LLM call interprets the response as "confirma" or "niega"

---

## 3. Orchestrator (`orchestrator.py`)

### 3.1 Responsibilities

The orchestrator provides:

1. **Greeting generation** — time-aware welcome message mentioning available modes
2. **Clarification generation** — when intent is "unclear", asks the user to specify
3. **Return-to-mode prompts** — after an agent interrupt, asks if the user wants to resume their previous mode

### 3.2 Greeting

The greeting is generated by the LLM with a system prompt that instructs it to:
- Mention the current time
- List the three available modes naturally (no bullets, no markdown)
- Use Rioplatense voseo
- Maximum 3 sentences

This makes each session start feel natural rather than robotic.

### 3.3 Clarification

When `classify_idle()` returns "unclear", the orchestrator generates a one-sentence clarification:

```python
"El usuario dijo: '{text}'. No quedó claro qué modo quiere.
 Pedile que aclare en 1 oración, sin enumerar las opciones de nuevo."
```

The "sin enumerar las opciones de nuevo" instruction prevents the model from repeating the full menu every time it's confused — a common annoyance pattern with small LLMs.

### 3.4 Legacy Intent Detection

The file also contains the original `detect_intent()` and `detect_intent_from_active_mode()` functions. These are preserved for backwards compatibility but are **not used** in the current main.py, which routes through `intent_classifier.py` instead. The legacy functions use simpler prompts without structured JSON output and without the safety-net keyword matching.

---

## 4. Context Manager (`context_manager.py`)

### 4.1 Purpose

The Context Manager maintains **session state across mode switches**. When a user is in the Engineering tutor, switches to the Agent to add a task, and then returns to Engineering, their conversation history must be preserved.

### 4.2 Data Model

```python
@dataclass
class ModeContext:
    mode:     str                # "idle", "english", "engineering", "agent"
    history:  list[dict]         # conversation history for this mode
    session:  Any                # session instance (TutorSession, AgentSession, etc.)
    topic:    str                # current topic (for English tutor)
    metadata: dict               # mode-specific metadata
```

```python
class ContextManager:
    active_mode:    str                    # currently active mode
    previous_mode:  Optional[str]          # for "return to previous" flow
    _contexts:      dict[str, ModeContext] # per-mode state
```

### 4.3 State Transitions

```
                    set_active("english")
         idle ─────────────────────────────► english
          ▲                                      │
          │           set_active("idle")          │ set_active("agent")
          │◄─────────────────────────────────────│ (agent interrupt)
          │                                      │
          │           set_active("english")       │
          │◄─────────────────────────────────────┘
          │           (return to previous)
```

Each `set_active()` call records `previous_mode`, enabling the "return to previous" flow after an agent interrupt. The `return_to_previous()` method restores the last active mode.

### 4.4 Lazy Initialization

`ModeContext` instances are created on first access via `get(mode)`:

```python
def get(self, mode: str) -> ModeContext:
    if mode not in self._contexts:
        self._contexts[mode] = ModeContext(mode=mode)
    return self._contexts[mode]
```

This means a mode's context (and its session instance, history, etc.) only consumes memory if the user actually enters that mode. A user who only uses the English tutor never allocates Engineering or Agent contexts.

### 4.5 History Persistence

History is persisted **in-memory only** for the lifetime of the application process. There is no disk serialization. This is intentional: conversation history is ephemeral and typically not useful across application restarts on an edge device. The SQLite-backed subsystems (tasks, calendar, reminders) provide the durable state.

### 4.6 Context Clearing

```python
def clear(self, mode: str) -> None:
    if mode in self._contexts:
        del self._contexts[mode]
```

Used when the user explicitly requests a fresh start ("limpiar" / "clear") within a mode. Deleting the context forces a new `ModeContext` to be created on next access, with empty history and no session instance.

---

## 5. Complete Routing Flow

```
main.py event loop
     │
     ├─ listen() → user_text
     │
     ├─ classify_idle(user_text) → IntentResult
     │     │
     │     ├─ "exit_app" → speak("¡Hasta luego!") → exit
     │     │
     │     ├─ "unclear" → generate_clarification()
     │     │     → listen() → classify_idle() again
     │     │     → if still unclear: show full menu
     │     │
     │     ├─ "english" → _run_english(ctx)
     │     │     │
     │     │     └─ internal loop:
     │     │         ├─ classify_english(text, topic) → IntentResult
     │     │         ├─ "respond" → normal tutor turn
     │     │         ├─ "change_topic" → pick random topic
     │     │         ├─ "propose_topic" → use extracted topic
     │     │         ├─ "agent" → _handle_agent_interrupt()
     │     │         │              → agent.run_turn()
     │     │         │              → generate_return_prompt()
     │     │         │              → resume or exit
     │     │         ├─ "switch_engineering" → return "engineering"
     │     │         ├─ "exit_mode" → return "idle"
     │     │         └─ "exit_app" → return "exit_app"
     │     │
     │     ├─ "engineering" → _run_engineering(ctx)
     │     │     │
     │     │     └─ internal loop:
     │     │         ├─ classify_engineering(text) → IntentResult
     │     │         ├─ "respond" → RAG lookup + LLM generation
     │     │         ├─ "agent" → _handle_agent_interrupt()
     │     │         ├─ "switch_english" → return "english"
     │     │         ├─ "exit_mode" → return "idle"
     │     │         └─ "exit_app" → return "exit_app"
     │     │
     │     └─ "agent" → AgentSession().run_turn()
     │           → speak result
     │           → return to idle
     │
     └─ next_mode routing:
          The return value of each _run_* function becomes the next mode.
          "idle" → back to main menu
          "english" / "engineering" → direct mode switch
          "exit_app" → terminate
```

This routing architecture ensures that mode switches are **direct** — switching from English to Engineering doesn't pass through idle. The user says "pasame a ingeniería" and the system transitions immediately, preserving the fluid conversational experience.
