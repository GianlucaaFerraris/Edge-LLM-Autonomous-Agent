# Agent Session & Tool Dispatch

> **Files:** `agent_session.py`, `dispatcher.py`, `task_manager.py`, `local_calendar.py`, `reminder_manager.py`, `wa_stub.py`  
> **Persistence:** SQLite (tasks.db, calendar.db, reminders.db)  
> **Communication:** WhatsApp stub (simulated, ready for whatsapp-web.js)  
> **Design principle:** Clarify before executing; never assume missing information

---

## 1. Purpose

The Agent Session handles **actionable requests** — tasks the user wants done, not questions they want answered. It manages a personal productivity suite running entirely on local storage (SQLite) with no cloud dependencies: task lists, calendar events, reminders, WhatsApp messaging, and web search.

The agent is designed to operate in two modes:
- **Standalone:** Full interactive session with its own REPL loop
- **Intercalated (run_turn):** Executes a single action from within another mode (English tutor, Engineering tutor) and returns control to the calling mode

---

## 2. Architecture

```
User utterance
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                      AGENT SESSION                          │
│                                                             │
│  1. Inject current datetime into system prompt              │
│  2. Append user message to history                          │
│  3. Loop (max 4 rounds):                                    │
│     a. Build messages = [system] + history[-10:]            │
│     b. Call LLM (temp=0.1, max_tokens=300)                  │
│     c. Parse response for TOOL_CALL                         │
│     d. If no TOOL_CALL → prose response → exit loop         │
│     e. If TOOL_CALL → dispatch(tool, args)                  │
│        ├─ "ok"        → inject TOOL_RESULT → continue loop  │
│        ├─ "clarify"   → return question to user → exit      │
│        ├─ "error"     → inject error → continue loop        │
│        └─ "web_search" → return for orchestrator confirm    │
│                                                             │
│  4. If 4 rounds exhausted → generic response                │
└─────────────────────────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │DISPATCH │  dispatcher.py
                    │         │  validates args, detects ambiguity,
                    │         │  executes or requests clarification
                    └────┬────┘
                         │
          ┌──────────────┼──────────────────────┐
          │              │              │        │
     ┌────▼───┐    ┌─────▼────┐   ┌────▼───┐  ┌▼────────┐
     │ Tasks  │    │ Calendar │   │Reminders│  │WhatsApp │
     │ SQLite │    │  SQLite  │   │ SQLite  │  │  Stub   │
     └────────┘    └──────────┘   └─────────┘  └─────────┘
```

---

## 3. Dynamic System Prompt

The agent's system prompt is rebuilt on every LLM call via `_build_system_prompt()`. It includes:

1. **Current datetime** — injected dynamically so the model can resolve relative time references ("mañana", "el lunes que viene", "en 2 horas") without hallucinating the current date.

2. **Tool definitions** — each tool with its name, arguments, and types. This serves as the model's "function schema" — since we use text-based tool calling rather than native function calling, the schema is embedded in natural language within the system prompt.

3. **Clarification rules** — five explicit rules that instruct the model to ask before executing when information is missing, ambiguous, or when the action is destructive (web search interrupts the current session).

4. **TOOL_CALL format specification** — the exact JSON format the model must emit.

5. **Clarification examples** — concrete examples of when and how to ask for clarification.

### 3.1 Why Dynamic Datetime?

A static system prompt would require the model to "know" the current time, which it cannot after quantization-induced knowledge loss. By injecting `datetime.now()` on every call, relative time parsing becomes deterministic:

```
Fecha y hora actual: Wednesday 15/01/2025 14:30.
User: "Poneme una reunión para mañana a las 10"
→ Model resolves "mañana" as Thursday 16/01/2025, emits cal_add with date="2025-01-16"
```

---

## 4. TOOL_CALL Parsing

The `_extract_tool_call()` function implements a **robust JSON extraction** algorithm that handles edge cases from a 7B model's potentially imprecise output:

```python
1. Find "TOOL_CALL:" marker in response text
2. Find first "{" after the marker
3. Track brace depth to find matching "}"
4. Attempt JSON.parse on the extracted substring
5. Return parsed dict or None on failure
```

This depth-tracking approach is more robust than regex-based extraction because the model sometimes emits nested JSON (args containing objects) or includes text after the JSON block. It correctly handles:
- `TOOL_CALL: {"tool": "cal_add", "args": {"title": "Reunión {proyecto}"}}`
- `TOOL_CALL: {"tool": "task_add", "args": {"title": "Test"}} Listo, te lo agrego.`
- Malformed JSON → returns None → model generates a prose response instead

---

## 5. Dispatcher (`dispatcher.py`)

### 5.1 Dispatch Protocol

Every tool handler returns a standardized dict:

```python
{
    "status":   "ok" | "clarify" | "error" | "web_search",
    "result":   str,           # text for user or TOOL_RESULT injection
    "question": str | None,    # clarification question (if status="clarify")
    "tool":     str,           # tool name that was executed
    "data":     dict | None,   # structured data (web_search results)
}
```

This uniform interface allows the agent session to handle all tools identically — it only needs to check `status` and route accordingly.

### 5.2 Clarification Before Execution

The dispatcher implements **pre-execution validation** for every tool. This is a core design principle: a tool is never executed with incomplete or ambiguous arguments. Examples:

**Task/Calendar ambiguity:**
```
User: "Anotame reunión a las 3"
→ Dispatcher detects time words ("a las") in a task_add title
→ Returns clarify: "Detecté una hora en 'reunión a las 3'.
   ¿Querés agregarla como tarea pendiente o como evento al calendario?"
```

**Missing required arguments:**
```
User: "Mandá un WhatsApp"
→ wa_send called with empty contact and message
→ Returns clarify: "¿A quién le mando el mensaje?" + contact list
```

**Ambiguous contact resolution:**
```
User: "Mandá un mensaje a Juan"
→ wa_stub.resolve_contact("Juan") returns 2 matches
→ Returns clarify: "Encontré varios contactos:
   • Juan Manuel
   • Juan Carlos
   ¿A cuál querés mandarle el mensaje?"
```

### 5.3 Calendar Conflict Resolution

When `cal_add` detects a time conflict with an existing event, it doesn't just error — it suggests an alternative:

```
1. Parse the requested time slot
2. Calculate duration
3. Call cal.suggest_slot() to find the nearest free slot
4. If a slot exists on the same day → offer it
5. If no same-day slot → search subsequent days → offer first available
```

This transforms a destructive error into a constructive interaction that reduces the number of conversational round-trips.

### 5.4 Web Search as Special Case

Web search returns `status="web_search"` instead of `"ok"`. This signals the agent session that the action requires **user confirmation** before proceeding, because web search interrupts the current conversational context by entering a separate sub-session with search results as context.

---

## 6. Tool Implementations

### 6.1 Task Manager (`task_manager.py`)

SQLite-backed todo list with priority support.

**Schema:**
```sql
CREATE TABLE tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    priority    TEXT DEFAULT 'media',    -- 'alta' | 'media' | 'baja'
    done        INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,           -- ISO 8601
    done_at     TEXT                     -- ISO 8601, NULL if not done
);
```

**Operations:** `add(title, priority) → id`, `list_pending() → [dict]`, `done(task_id) → bool`, `delete(task_id) → bool`, `search(query) → [dict]`

Priority ordering uses a Python-side sort map (`alta→0, media→1, baja→2`) after the SQL query, keeping the database schema simple.

### 6.2 Local Calendar (`local_calendar.py`)

Full-featured local calendar with no external dependencies (no Google Calendar, no OAuth, no network).

**Schema:**
```sql
CREATE TABLE events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    date             TEXT NOT NULL,       -- ISO date
    start_time       TEXT NOT NULL,       -- HH:MM
    end_time         TEXT NOT NULL,       -- HH:MM
    description      TEXT DEFAULT '',
    recurrence_group INTEGER DEFAULT NULL -- groups recurring events
);
```

**Capabilities:**
- Single events: add, list, delete
- Recurring events: weekly recurrence with start/end date range, grouped by `recurrence_group` ID
- Free slot detection: finds available time blocks of a given minimum duration on a specific day
- Conflict detection: prevents overlapping events on the same day
- Smart slot suggestion: when a conflict is detected, proposes the nearest available slot
- Natural language date parsing: "hoy", "mañana", "el lunes", "lunes que viene", ISO dates, DD/MM/YYYY formats

### 6.3 Reminder Manager (`reminder_manager.py`)

Persistent reminders with a background scheduler thread.

**Schema:**
```sql
CREATE TABLE reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    remind_at   TEXT NOT NULL,    -- ISO 8601 datetime
    fired       INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);
```

**Scheduler architecture:**

```
main.py startup
     │
     └→ start_scheduler(interval_minutes=30)
         │
         └→ daemon thread: _scheduler_loop()
              │
              └→ every 30 min: _check_reminders()
                   │
                   ├→ Query: SELECT WHERE fired=0 AND remind_at <= now+35min
                   ├→ Mark matched as fired=1
                   └→ Enqueue alerts to _pending_alerts (thread-safe list)
```

The orchestrator calls `pop_alerts()` at the start of each turn, which atomically drains the alert queue and displays/speaks any pending reminders. The 35-minute check window (5 min over the 30-min interval) provides overlap to prevent missed alerts due to scheduler drift.

**Natural language datetime parsing** supports:
- ISO: "2025-08-01 15:00"
- Relative: "en 2 horas", "en 30 minutos"
- Day + time: "mañana a las 10", "el lunes a las 18:30"
- Day only (defaults to 12:00): "mañana", "el viernes"

### 6.4 WhatsApp Stub (`wa_stub.py`)

Simulated WhatsApp interface that implements the same API contract as the planned real implementation (via `whatsapp-web.js`).

**Contact resolution:** Contacts are stored in a local JSON file with names and aliases. The `resolve_contact()` function performs fuzzy matching:
```
"juanma" → matches alias → resolves to "Juan Manuel"
"juan" → matches multiple contacts → returns ambiguous result → dispatcher asks for clarification
```

**Design for replacement:** The stub's `send()` and `read()` functions return the same response structure that the real `whatsapp-web.js` integration will produce. When the real implementation is ready, only the internal body of these functions changes — the dispatcher and agent session require zero modifications.

---

## 7. Multi-Round Tool Execution

The agent session supports up to **4 rounds** of tool calls per user turn. This handles scenarios where the first tool call returns an error or where the model needs to chain tools:

```
Round 1: User asks "¿Qué tengo mañana?"
         → Model emits TOOL_CALL: cal_list(date="mañana")
         → Dispatcher returns events list
         → TOOL_RESULT injected into history

Round 2: Model processes TOOL_RESULT
         → Generates prose summary: "Mañana tenés 2 eventos..."
         → No TOOL_CALL → loop exits
```

Error recovery example:
```
Round 1: Model emits TOOL_CALL: task_done(title="comprar")
         → Dispatcher finds 3 matches → returns clarify
         → Agent returns clarification question to user

Next turn: User answers "la primera"
           → Model emits TOOL_CALL: task_done(task_id=5)
           → Dispatcher marks as done → success
```

The 4-round cap prevents infinite loops if the model gets stuck in a tool-calling cycle (a failure mode observed with 7B models when the TOOL_RESULT is ambiguous).

---

## 8. Standalone vs. Intercalated Operation

### 8.1 Standalone (`agent.run()`)

Full REPL session with its own greeting, history management, alert display, and web search sub-mode. The agent manages its own conversation loop until the user says "salir".

### 8.2 Intercalated (`agent.run_turn(user_text, return_mode)`)

Executes a single turn and returns a result dict:

```python
{
    "action":      "respond" | "web_search" | "return_to_mode",
    "text":        str,
    "search_data": dict | None,
    "return_mode": str | None,   # "english" | "engineering" | None
}
```

When called from within the English or Engineering tutor (via the orchestrator's agent interrupt handler), the `return_mode` tells the orchestrator which session to resume after the agent action completes.

This design allows seamless mid-conversation tool usage: a student practicing English can say "poneme un recordatorio para las 5" → agent handles it → returns to the English tutor with context preserved.

---

## 9. Web Search Sub-Mode

When the user confirms a web search, the agent enters a temporary sub-session:

```
1. search.search(query) → results with context
2. Build web_system prompt with search results as grounding
3. Enter mini-REPL: user can ask follow-up questions about the results
4. User says "listo" / "salir" → exit sub-mode → return to agent
```

This sub-session uses its own separate history (`web_history`) to prevent search context from polluting the main agent conversation history.

---

## 10. Inference Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | 0.1 | Near-deterministic. Tool calls must be precise JSON; creative variation in tool names or arg formats causes parse failures. |
| `max_tokens` | 300 | Agent responses are short and functional. The TOOL_CALL JSON itself is ~50–100 tokens; the natural language response after TOOL_RESULT is ~30–80 tokens. |
| History window | Last 10 messages | Sufficient for multi-round tool execution while staying within 2048-token context window. |

---

## 11. Voice I/O Integration

The `AgentSession` class exposes two hooks:

```python
def listen(self) -> str:     # defaults to input(), replaced by VoiceIO in production
def speak(self, text: str):  # defaults to print(), replaced by VoiceIO in production
```

In integrated mode (`main.py`), these are replaced with the global `VoiceIO` instance:
```python
agent.listen = listen          # → VoiceIO.listen() → Moonshine STT
agent.speak = lambda text: speak(text)  # → VoiceIO.speak_and_print() → Piper TTS
```

This dependency injection pattern keeps the agent session testable without audio hardware.
