# Agenty — Local Edge AI Assistant

A personal AI assistant that runs fully locally on a Radxa Rock 5B (RK3588) or Jetson Orin Nano.
No cloud dependencies. Everything stored in SQLite. Quantized INT4 model served by Ollama.

---

## Project Structure

```
Agenty-Edge-Assistant/
├── src/
│   ├── main.py                          ← single entry point
│   │
│   ├── english/
│   │   ├── __init__.py
│   │   └── tutor_session.py             ← English tutor + LanguageTool integration
│   │
│   ├── engineering/
│   │   ├── __init__.py
│   │   └── engineering_session.py       ← Engineering tutor (retired scientist persona)
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent_session.py             ← tool orchestrator + clarification loop
│   │   ├── dispatcher.py                ← executes tools, handles ambiguity
│   │   ├── task_manager.py              ← SQLite to-do list
│   │   ├── local_calendar.py            ← SQLite calendar (simple + recurring events)
│   │   ├── reminder_manager.py          ← SQLite reminders + 30-min background scheduler
│   │   ├── web_search.py                ← DuckDuckGo search (no API key required)
│   │   ├── wa_stub.py                   ← WhatsApp stub (swap for real integration later)
│   │   └── db/                          ← persistent databases (live on the SBC)
│   │       ├── tasks.db
│   │       ├── calendar.db
│   │       ├── reminders.db
│   │       └── wa_contacts.json
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── orchestrator.py              ← intent detection + mode routing
│   │   └── context_manager.py           ← shared state across modes
│   │
│   └── finetuning/
│       └── topics.json                  ← conversation topics for the English tutor
│
└── src/test/
    ├── conftest.py
    ├── test_modes.py
    ├── test_latency.py
    └── manual_chat.py
```

---

## Installation

### 1. Python dependencies

```bash
conda activate tutor_env
pip install openai requests duckduckgo-search
```

### 2. Register the model in Ollama (once)

```bash
cd ~/Desktop/Agenty-Edge-Assistant/model/gguf

# Edit Modelfile: replace FROM with the absolute path to the .gguf file
# FROM /home/<user>/Desktop/Agenty-Edge-Assistant/model/Qwen2.5-7B-Instruct.Q4_K_M-001.gguf

ollama create asistente -f Modelfile
ollama list   # verify "asistente" appears
```

### 3. LanguageTool (English tutor only)

```bash
# Download from https://languagetool.org/download/
java -cp languagetool-server.jar org.languagetool.server.HTTPServer \
     --port 8081 --allow-origin '*' --public
```

The assistant starts without LanguageTool — it just disables grammar error detection.

---

## Running

### Full assistant (recommended)

```bash
cd ~/Desktop/Agenty-Edge-Assistant
python src/main.py
```

### Standalone modules

```bash
# English tutor only
python src/english/tutor_session.py

# Engineering tutor only
python src/engineering/engineering_session.py

# Agent only
python src/agent/agent_session.py
```

### Tests

```bash
cd src/test
pytest test_modes.py -v
python test_latency.py
python manual_chat.py
```

---

## Orchestrator Flow

```
[main.py starts]
       ↓
[Greets user + detects intent]
       ↓
"I want to practice English"   → TutorSession (English)
"question about physics"       → EngineeringSession
"what do I have pending?"      → AgentSession (direct tool call)
"search for X"                 → AgentSession (web_search, requires confirmation)

[From any active mode — lightweight tools, no interruption]
"add task X"                       → task_add       → returns to previous mode
"what's on my calendar today?"     → cal_list       → returns to previous mode
"send WhatsApp to mom: running late"→ wa_send        → returns to previous mode
"remind me of X at 6pm"            → reminder_set   → returns to previous mode
"I finished task X, remove it"     → task_done      → returns to previous mode

[search_web — interrupts and does NOT return]
"search for X" while in English or Engineering mode
  → "This will pause your current session and you won't be able to resume it. Continue?"
  → If confirmed: enters web search mode
  → If cancelled: resumes previous mode
```

---

## Agent Tools Reference

| Tool | Description | Clarifies when... |
|------|-------------|-------------------|
| `task_add` | Add a to-do task | Title missing, or date/time detected (task vs. calendar event?) |
| `task_list` | List pending tasks | — |
| `task_done` | Mark task as done | Ambiguous which task |
| `cal_add` | Add calendar event | Date, start time, or end time missing |
| `cal_add_recurring` | Add recurring events | Day of week, time, or end date missing |
| `cal_list` | List events | — |
| `cal_delete` | Delete event | ID missing (suggests listing first) |
| `cal_free` | Find free time slots | — |
| `reminder_set` | Set a reminder | Title or datetime missing |
| `reminder_list` | List all reminders | — |
| `wa_send` | Send WhatsApp message | Contact is ambiguous → shows contact list |
| `wa_read` | Read WhatsApp messages | — |
| `search_web` | Search the web | Vague query → asks for more detail |

---

## Calendar — Usage Examples

```
"from today until August 1st, every Monday I have Systems class from 3pm to 6pm"
→ cal_add_recurring(title="Systems OS", weekday="lunes", start="15:00",
                    end="18:00", until="2025-08-01")

"what free slots do I have on Thursday for at least 2 hours?"
→ cal_free(date="jueves", duration_minutes=120)

"schedule a meeting on Wednesday at 10"
→ agent asks: "What time does it end?"
→ cal_add(title="Meeting", date="miércoles", start="10:00", end=<answer>)

"if there's a conflict, suggest another day"
→ cal.suggest_slot() automatically searches the next 7 days
```

---

## Reminders — How They Work

- The scheduler runs in a **daemon thread** every 30 minutes
- On startup, `main.py` triggers an immediate check
- Reminders due within the next 30 minutes are queued
- At the start of each orchestrator turn → `pop_alerts()` displays them
- Only reminders you explicitly added with `reminder_set` are shown

---

## WhatsApp Contacts

Edit `src/agent/db/wa_contacts.json` (auto-created on first run):

```json
[
  {"name": "Mom",         "phone": "+54911XXXXXXX", "aliases": ["mama", "ma", "mom"]},
  {"name": "Juan Manuel", "phone": "+54911XXXXXXX", "aliases": ["juanma", "juan"]}
]
```

If the contact name is ambiguous, the agent shows the full list and asks for confirmation
before sending anything.

---

## Latency Benchmarks (Laptop — NVIDIA GPU)

| Mode | TTFT | Total | TPS |
|------|------|-------|-----|
| Mode detection | 0.11s | 0.21s | 6.5 |
| Agent tool call | 0.11s | 0.62s | 10.7 |
| English tutor (short) | 0.15s | 1.34s | 27.5 |
| Engineering tutor (medium) | 0.12s | 11.35s | 25.5 |

> Rock 5B / Jetson Orin Nano performance will differ. Run `python test_latency.py`
> after deployment to get accurate numbers for the target hardware.

---

## Roadmap

- [ ] Connect real WhatsApp (`whatsapp-web.js` on localhost)
- [ ] Local RAG with `sentence-transformers` + FAISS (lecture notes, technical docs)
- [ ] TTS with Piper or Kokoro (replace `speak()` hooks)
- [ ] STT with local Whisper (replace `listen()` hooks)
- [ ] Convert model to RKLLM format to run on RK3588 NPU
- [ ] Autostart on SBC boot via `systemd` service