# Main Entry Point

> **File:** `main.py`  
> **Version:** v3 (Voice)  
> **Launch:** `python main.py [--keyboard] [--print-only]`  
> **Process lifetime:** Persistent (runs until user says "chau" or Ctrl+C)

---

## 1. Purpose

`main.py` is the top-level process that bootstraps the entire assistant, initializes hardware I/O, and runs the main event loop. It serves as the **integration point** where all subsystems — voice I/O, intent classification, tutor sessions, agent, RAG, reminders, LanguageTool — are wired together.

The file has three responsibilities:
1. **Bootstrap:** Validate prerequisites (Ollama running, model available), initialize VoiceIO, start background threads (reminder scheduler)
2. **Event loop:** Listen → classify → route → respond → repeat
3. **Mode runners:** Contain the inner loops for English and Engineering modes with full voice integration, intent classification per turn, and agent interrupt handling

---

## 2. CLI Arguments

```
python main.py                         # Full voice mode (mic + speaker)
python main.py --keyboard              # Keyboard input, speaker output
python main.py --print-only            # Mic input, text-only output
python main.py --keyboard --print-only # Pure text mode (dev/SSH)
```

These flags are passed to the `VoiceIO` constructor and affect all I/O globally. No code path inside the tutor sessions, agent, or orchestrator needs to know whether voice or text is being used — the abstraction is complete.

---

## 3. Startup Sequence

```
1. Parse CLI arguments (--keyboard, --print-only)

2. Print banner with current datetime

3. Health check: GET http://localhost:11434/api/tags
   → If Ollama is not responding: print error, exit(1)
   → This catches the most common startup failure

4. Resolve model: "asistente" (fine-tuned) or "qwen2.5:7b" (fallback)
   → Print which model was found

5. Initialize VoiceIO(use_keyboard=..., use_print=...)
   → Loads Moonshine STT model (~2-5s on ARM)
   → Loads Piper TTS voices (~1s)

6. Start reminder scheduler:
   reminders.start_scheduler(interval_minutes=30)
   → Daemon thread begins checking SQLite every 30 minutes
   → Immediate check on startup (catches overdue reminders)

7. Initialize ContextManager() — empty state

8. Generate greeting via LLM:
   generate_greeting(client, model)
   → Time-aware greeting in Rioplatense Spanish
   → speak() → audio output or text

9. Enter main event loop
```

**Total startup time on Rock 5B:** ~5–10 seconds (dominated by STT/TTS model loading and the greeting LLM call).

---

## 4. Main Event Loop

```python
while True:
    _show_alerts()          # Pop and display/speak any pending reminders
    user_text = listen()    # VoiceIO: mic → STT → text (or keyboard)
    if not user_text:
        continue

    intent = classify_idle(user_text)   # Intent classifier
    
    if intent.action == "exit_app":
        speak("¡Hasta luego!")
        ensure_stopped()    # Shut down LanguageTool if running
        break

    if intent.action == "unclear":
        # Double-classification: ask for clarification, then re-classify
        clarification = generate_clarification(user_text, client, model)
        speak(clarification)
        user_text2 = listen()
        intent = classify_idle(user_text2)
        if intent.action == "unclear":
            speak("No pude entender qué querés hacer. ...")
            continue

    # Mode routing loop — handles direct mode-to-mode switches
    next_mode = intent.action
    while next_mode and next_mode != "idle":
        if next_mode == "english":
            next_mode = _run_english(ctx)
        elif next_mode == "engineering":
            next_mode = _run_engineering(ctx)
        elif next_mode == "agent":
            # Agent runs a single turn, then returns to idle
            agent = AgentSession()
            agent.listen = listen
            agent.speak = lambda text: speak(text)
            result = agent.run_turn(user_text, return_mode=None)
            speak(result["text"])
            next_mode = None
        elif next_mode == "exit_app":
            speak("¡Hasta luego!")
            ensure_stopped()
            return
        else:
            next_mode = None
```

### 4.1 The Mode Routing Loop

The `while next_mode` loop is the mechanism for **direct mode-to-mode transitions**. When `_run_english()` returns `"engineering"` (because the user said "pasame a ingeniería"), the loop immediately calls `_run_engineering()` without returning to the idle state. This provides a seamless transition experience.

Possible return values from mode runners:
- `"idle"` → back to main event loop (user explicitly exited the mode)
- `"english"` / `"engineering"` → direct switch to another mode
- `"exit_app"` → terminate the application

### 4.2 Double-Classification for Unclear Intent

When the initial classification returns "unclear", the system doesn't give up — it generates a natural clarification question, listens for a response, and re-classifies. Only if the second attempt also fails does it fall back to explicitly listing the available options.

This two-attempt pattern significantly reduces "I don't understand" moments in practice, because the clarification question often nudges the user to rephrase more explicitly.

---

## 5. English Mode Runner (`_run_english`)

The English mode runner is the most complex because it integrates multiple subsystems:

### 5.1 Setup

```python
vio.set_mode("english")          # STT → English, TTS → English voice
ctx.set_active("english")        # Context manager tracks active mode
lt_ok = _ensure_lt_up()          # Start LanguageTool server (JVM)
```

**LanguageTool lifecycle:** The Java-based LanguageTool server is started when entering English mode and stopped when leaving. This is a deliberate memory optimization — the JVM consumes ~400–600 MB, and there's no reason to keep it running during engineering or agent modes.

### 5.2 Session Persistence

```python
if eng_ctx.session is None:
    eng_ctx.session = TutorSession()     # First entry: create new session
    eng_ctx.session._ask_topic_preference()
else:
    speak("Retomando el tutor de inglés.")  # Re-entry: resume session
```

The `TutorSession` instance is stored in the ContextManager, so switching to Engineering and back preserves the English conversation history, current topic, and session state.

### 5.3 Voice Hook Injection

```python
eng_ctx.session.speak = lambda text, total=None: _tutor_speak(text, total)
eng_ctx.session.listen = listen
```

The `TutorSession` class was designed with `speak()` and `listen()` hooks that default to `print()` and `input()`. By replacing them with VoiceIO-backed functions, the same session code works in both text and voice modes.

### 5.4 Language-Aware TTS

The `_tutor_speak()` hook auto-detects the language of each output:

```python
def _tutor_speak(text, total=None):
    # Print to console with metrics
    print(f"\n[TUTOR]: {text}")
    if total is not None:
        print(f"  ⏱  Total={total}s")
    # VoiceIO auto-detects es/en for voice selection
    vio.speak(text)
```

The English tutor emits both English (conversation prompts, feedback) and Spanish (corrections, explanations). The `_is_likely_english()` heuristic detects language by counting Spanish marker words, enabling the TTS to switch voices automatically within the same session.

### 5.5 Per-Turn Flow

```python
while True:
    _show_alerts()                          # Check reminders
    student_text = session.listen()         # VoiceIO → STT

    intent = classify_english(student_text, current_topic=session.topic)

    # Route based on intent:
    # respond → check grammar + generate feedback
    # change_topic → pick random topic
    # propose_topic → use extracted topic
    # agent → handle agent interrupt (mode switch to Spanish STT)
    # switch_engineering / exit_mode / exit_app → exit with return value
```

### 5.6 Agent Interrupt in English Mode

When the user requests an agent action while practicing English:

```python
vio.set_mode("agent")           # Switch STT to Spanish
result = _handle_agent_interrupt(student_text, "english", ctx)
vio.set_mode("english")         # Switch STT back to English
```

This temporary mode switch is necessary because agent commands are in Spanish — the STT needs to expect Spanish phonemes for accurate transcription of task names, contact names, and dates.

---

## 6. Engineering Mode Runner (`_run_engineering`)

### 6.1 Setup

```python
vio.set_mode("engineering")     # STT → Spanish, TTS → Spanish voice
ctx.set_active("engineering")   # Context manager
rag_engine = _load_rag()        # Load FAISS index + embedding model (singleton)
```

### 6.2 Streaming with Concurrent TTS

The engineering mode uses `speak_stream()` for overlapping LLM generation and TTS:

```python
token_iter = (tok for tok, _, _ in _chat_stream_iter(messages, temperature=0.3, max_tokens=600))
response = vio.speak_stream(token_iter)
```

`_chat_stream_iter()` is a variant of `_chat_stream()` that yields individual tokens instead of printing them, allowing `speak_stream()` to accumulate sentences and dispatch them to TTS as they complete.

This is the most latency-sensitive code path in the system. The generator pipeline is:
```
Ollama HTTP stream → chunk parser → token extractor → sentence accumulator → Piper TTS → speaker
```
Each stage processes data as it arrives, with no buffering beyond what's needed for sentence boundary detection.

### 6.3 RAG Integration

The engineering runner contains the full RAG integration logic (same as the standalone `engineering_session.py` but wired to VoiceIO):

```python
if rag_engine is not None:
    result = rag_engine.query_with_domain_check(user_text)
    if result.relevant:
        speak("Un momento, déjame revisar en mis libros...")
        # Build augmented prompt with RAG context
```

The "checking books" announcement is spoken through TTS, providing a natural conversational pause while the augmented prompt is constructed.

---

## 7. Global I/O Hooks

`main.py` defines two global functions that delegate to the VoiceIO singleton:

```python
def listen() -> str:
    return vio.listen()

def speak(text: str) -> None:
    vio.speak_and_print(text)
```

These are injected into every session object that needs I/O, creating a single point of control for all audio/text interaction.

---

## 8. Alert System Integration

At the top of every turn in every mode, `_show_alerts()` checks for pending reminders:

```python
def _show_alerts():
    alerts = reminders.pop_alerts()
    if not alerts:
        return
    for a in alerts:
        dt = datetime.datetime.fromisoformat(a["remind_at"])
        print(f"  ⏰  {a['title']} — {dt.strftime('%H:%M')}")
    for a in alerts:
        vio.speak(f"Recordatorio: {a['title']}", force_voice="agent")
```

The `force_voice="agent"` parameter ensures reminders are always spoken in the Spanish voice, even during an English tutor session. This prevents the jarring experience of a Spanish reminder being spoken in an English accent.

---

## 9. Shutdown

Clean shutdown involves:
1. `ensure_stopped()` — kills the LanguageTool JVM if it was started by this process
2. The reminder scheduler thread is a daemon — it terminates automatically when the main process exits
3. SQLite connections are closed automatically via Python's garbage collector and `with` context managers
4. Ollama continues running (it's an independent service)

No explicit VoiceIO shutdown is needed — audio device handles are released when the process exits.

---

## 10. Process Architecture Summary

```
main.py (PID 1 — long-running)
  │
  ├── Main thread: event loop (listen → classify → route → respond)
  │
  ├── Daemon thread: reminder scheduler (check every 30 min)
  │
  ├── Subprocess (optional): LanguageTool JVM (only during English mode)
  │     └── Killed via SIGTERM/SIGKILL on mode exit
  │
  └── External services:
        ├── Ollama (separate process, localhost:11434)
        └── (Future) whatsapp-web.js (separate process, localhost:3000)
```

Total steady-state processes: 2 (main.py + Ollama). LanguageTool adds a third only during English practice.
