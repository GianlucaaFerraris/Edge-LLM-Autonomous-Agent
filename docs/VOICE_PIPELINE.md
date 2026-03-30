# Voice I/O Pipeline (STT & TTS)

> **STT Engine:** Moonshine (edge-optimized Whisper derivative)  
> **TTS Engine:** Piper (VITS-based, CPU inference)  
> **Voice module:** `voice_io.py` (VoiceIO class)  
> **Target latency:** STT ~50ms (VAD + transcription), TTS ~20ms/sentence  
> **Hardware:** USB microphone + 3.5mm / HDMI audio output on Rock 5B / Jetson

---

## 1. Design Philosophy

The voice pipeline is designed around a principle of **sentence-level streaming**: the system does not wait for the LLM to finish generating its entire response before speaking. Instead, as tokens stream from the LLM, the system accumulates text until a sentence boundary is detected, then immediately sends that sentence to TTS while the LLM continues generating the next sentence.

This overlapping execution is critical for perceived responsiveness on edge hardware where full LLM generation can take 5–15 seconds. The user hears the first sentence within ~1–3 seconds (TTFT + first sentence accumulation + TTS), and subsequent sentences arrive with near-zero perceptual gap because TTS and LLM generation run concurrently.

```
Timeline:

LLM:    [────prefill────][──tok1──][──tok2──]...[──tokN──][──sentence 1 complete──][──tok──]...
TTS:                                                      [──speak sentence 1──]
User:   [waiting...~1.5s]                                 [hearing sentence 1]
LLM:    (continues generating)                                                   [──sentence 2 complete──]
TTS:                                                                             [──speak sentence 2──]
User:                                                                            [hearing sentence 2]
```

---

## 2. Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  MICROPHONE  │────►│  Moonshine STT  │────►│   text (str)     │
│  (USB/I2S)   │     │  VAD + Whisper  │     │                  │
└──────────────┘     │  ~50ms total    │     └────────┬─────────┘
                     └─────────────────┘              │
                                                      ▼
                                              ┌───────────────┐
                                              │  VoiceIO      │
                                              │  .listen()    │
                                              │  .speak()     │
                                              │  .speak_stream│
                                              │  .set_mode()  │
                                              └───────┬───────┘
                                                      │
                     ┌─────────────────┐              │
                     │   Piper TTS     │◄─────────────┘
                     │   VITS model    │     text (per sentence)
                     │   ~20ms/sent    │
                     └────────┬────────┘
                              │ audio (WAV PCM)
                     ┌────────▼────────┐
                     │    SPEAKER      │
                     │  (ALSA/Pulse)   │
                     └─────────────────┘
```

---

## 3. VoiceIO Class

The `VoiceIO` class provides a unified interface for all I/O operations across the system. Every component (tutor sessions, agent, orchestrator) interacts with audio through this single class, which abstracts away the hardware details.

### 3.1 Initialization

```python
vio = VoiceIO(
    use_keyboard=False,    # True → input() instead of Moonshine STT
    use_print=False,       # True → print() instead of Piper TTS
)
```

**Dev mode flags:**
- `--keyboard`: Replaces STT with `input()` for testing without a microphone
- `--print-only`: Replaces TTS with `print()` for testing without speakers
- Both flags together: Pure text mode, identical to v2 behavior

These flags enable the full pipeline to be tested on any development machine, including headless servers and SSH sessions, without modifying any application logic.

### 3.2 Mode-Aware Voice Selection

```python
vio.set_mode("english")      # STT: English model, TTS: English female voice
vio.set_mode("engineering")  # STT: Spanish model, TTS: Spanish male voice
vio.set_mode("agent")        # STT: Spanish model, TTS: Spanish male voice
vio.set_mode("idle")         # STT: Spanish model, TTS: Spanish male voice
```

Mode switching triggers two configuration changes:
1. **STT language model:** Moonshine loads language-specific acoustic models. When the user is practicing English, the STT expects English phonemes; when they switch to the agent or engineering mode, it expects Spanish.
2. **TTS voice:** Different voices for different personas reinforce the mode distinction. The English tutor uses an English-accented voice; the engineering tutor and agent use a Spanish voice.

### 3.3 Core Methods

#### `listen() → str`

Activates the microphone, runs VAD (Voice Activity Detection) to detect speech boundaries, then passes the audio segment to Moonshine for transcription. Returns the transcribed text.

In keyboard mode, falls back to `input("[VOS]: ")`.

#### `speak(text, force_voice=None)`

Sends text to Piper TTS for synthesis and plays the resulting audio. The voice is selected based on the current mode unless `force_voice` overrides it (used for reminders that should always speak in Spanish regardless of current mode).

In print-only mode, falls back to `print()`.

#### `speak_and_print(text)`

Combination method that both prints the text to the console (for logging/debugging) and speaks it through TTS. Used by the global `speak()` hook in `main.py`.

#### `speak_stream(token_iterator) → str`

The key method for streaming TTS. Accepts an iterator of tokens from the LLM's streaming response and:

1. Accumulates tokens into a buffer
2. On each sentence boundary (`.`, `!`, `?`, `:` followed by whitespace), extracts the complete sentence
3. Sends the sentence to Piper TTS immediately
4. Continues accumulating the next sentence while TTS plays
5. Returns the full concatenated text when the iterator is exhausted

```python
# Usage in engineering session:
token_iter = (tok for tok, _, _ in _chat_stream_iter(messages))
full_response = vio.speak_stream(token_iter)
```

This method is the core of the perceived-latency optimization. Without it, the user would wait for the entire LLM response (~5–15s) before hearing anything.

---

## 4. STT: Moonshine

### 4.1 Why Moonshine Over Whisper

Standard Whisper models are designed for server-grade GPUs. Even `whisper-tiny` requires ~400 MB RAM and takes ~2–5 seconds per utterance on ARM CPU. Moonshine is an edge-optimized derivative that:

- Uses a smaller encoder architecture optimized for ARM NEON instructions
- Includes built-in VAD (no separate silero-vad dependency)
- Achieves ~50ms total latency (VAD + transcription) for typical utterances (1–5 seconds of speech)
- Consumes ~200 MB RAM

### 4.2 VAD Integration

Voice Activity Detection is essential for a hands-free system. The VAD:

1. Continuously monitors the microphone input
2. Detects speech onset (energy + spectral features)
3. Records until speech offset (silence > threshold)
4. Passes the bounded audio segment to the transcription model

Without VAD, the system would either require a push-to-talk button (poor UX for a voice assistant) or would continuously transcribe ambient noise (wasting CPU and producing garbage text).

### 4.3 Language-Aware Transcription

Moonshine supports language hints that bias the acoustic model toward the expected language. The `set_mode()` call configures this:

- English mode: biased toward English phonemes and vocabulary
- Spanish mode: biased toward Spanish phonemes, voseo patterns, Argentine vocabulary

This reduces transcription errors when the user code-switches (common in the English tutor mode where the user might say Spanish meta-commands like "cambiemos de tema" between English practice sentences).

---

## 5. TTS: Piper

### 5.1 Architecture

Piper uses **VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) — a lightweight neural TTS architecture that produces natural-sounding speech at CPU-speed inference. Key properties:

- Single-model, end-to-end: text → mel spectrogram → waveform in one forward pass
- No separate vocoder needed (VITS integrates the HiFi-GAN decoder)
- ~50 MB per voice model
- ~20ms inference per sentence on ARM CPU (for sentences <50 words)

### 5.2 Voice Models

The system uses at least two Piper voice models:

| Mode | Voice | Language | Persona |
|---|---|---|---|
| English tutor | Female EN | English | Conversational tutor |
| Engineering / Agent | Male ES | Spanish | Technical mentor / assistant |

Voice selection is automatic based on the current VoiceIO mode. The English tutor's `_tutor_speak()` hook includes an additional language auto-detection heuristic to switch between Spanish and English voices within the same session:

```python
def _is_likely_english(text: str) -> bool:
    es_markers = {"quiero", "puedo", "tengo", "necesito", ...}
    words = set(text.lower().split())
    es_count = len(words & es_markers)
    if es_count >= 2: return False
    if es_count >= 1 and len(words) < 8: return False
    return True
```

This allows the English tutor to speak English sentences in an English voice and Spanish corrections/explanations in a Spanish voice, creating a natural bilingual experience.

### 5.3 Sentence Boundary Detection

For streaming TTS, accurate sentence boundary detection is important. The system uses a simple but effective heuristic: split on punctuation marks (`.`, `!`, `?`, `:`) followed by whitespace. This works well for the structured prose generated by the fine-tuned model, which consistently produces well-punctuated output.

Edge cases:
- Abbreviations ("e.g.", "Dr.") are rare in the model's output due to the conversational style enforcement in fine-tuning
- URLs and technical notation are not present (the engineering tutor doesn't write code or formulas)
- Numbered lists ("1. First...") may split at the period — acceptable because each item is spoken independently

---

## 6. Audio Hardware Configuration

### 6.1 Rock 5B

The RK3588 has built-in I2S audio interfaces but typically uses USB audio for microphone input and 3.5mm headphone jack or HDMI for output. ALSA is the default audio backend.

### 6.2 Jetson Orin Nano

Similar configuration with USB audio input and HDMI/3.5mm output. PulseAudio may be required depending on the Jetson Linux version.

### 6.3 Buffer Sizing

Audio buffer sizes are tuned for low latency:
- Input buffer: ~10ms (160 samples at 16kHz) — small enough for responsive VAD
- Output buffer: ~20ms (320 samples at 16kHz) — small enough to avoid perceptible delay between sentences

---

## 7. Per-Turn Voice Flow

Complete flow for a single turn in the engineering session with voice:

```
1. Mic → VAD detects speech → Moonshine STT → "¿Qué es backpropagation?"
   (~50ms)

2. Intent classification (LLM call) → "respond"
   (~1-3s)

3. RAG lookup → domain check → FAISS search → context found
   (~110ms)

4. speak("Un momento, déjame revisar en mis libros...")
   → Piper TTS → speaker  (~20ms synthesis, ~1s playback)

5. LLM streaming begins → tokens arrive
   → accumulate "Backpropagation es el algoritmo fundamental..."
   → sentence complete → Piper TTS → speaker
   → accumulate next sentence...
   (~TTFT 0.5-2s, then continuous)

6. LLM streaming ends → final sentence spoken
   → print metrics: TTFT, total time, RAG score
```

Total perceived latency to first audio: ~2–5 seconds (dominated by intent classification + TTFT).
Total perceived latency to full response: same as LLM generation time, because TTS runs concurrently.

---

## 8. Fallback Behavior

| Failure | Behavior |
|---|---|
| No microphone detected | Falls back to keyboard input if `--keyboard` flag set; otherwise prompts user to connect mic |
| No audio output | Falls back to print-only mode |
| Moonshine model missing | Falls back to keyboard input with warning |
| Piper model missing | Falls back to print-only output with warning |
| Audio buffer underrun | Brief silence gap, recovers automatically |

The system is designed to degrade from full voice to partial voice to text-only, never crashing due to audio hardware issues.
