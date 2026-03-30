# Engineering Tutor Session

> **File:** `engineering_session.py`  
> **Mode name:** `engineering`  
> **Dependencies:** Ollama (Qwen 2.5 7B), RAG Engine (optional), sentence-transformers, FAISS

---

## 1. Purpose & Design Philosophy

The Engineering Tutor implements a conversational mentor persona — a "wise retired engineer" — that answers conceptual questions across engineering, science, and technology domains. The design is intentionally **code-free**: the model never writes code, never performs numerical calculations, and never solves logic puzzles. Instead, it provides intuitive explanations, analogies, historical context, trade-offs, and resource recommendations.

This constraint serves two objectives:

1. **Persona coherence during fine-tuning.** By training exclusively on conceptual prose (no code blocks, no LaTeX, no numerical outputs), the model learns a narrow but deep output distribution that remains stable across temperature ranges and prompt variations.
2. **Complementarity with the Agent mode.** The Agent handles executable actions; the tutor handles knowledge transfer. This separation prevents mode confusion in a small 7B model where behavioral boundaries blur easily without explicit training signal.

The tutor supports **bilingual operation** (Spanish/English), responding in the same language the user writes in. Technical terms that are commonly used in their English form within Spanish technical discourse (e.g., "backpropagation", "deep learning", "overfitting") are preserved as-is.

---

## 2. Architecture & Data Flow

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG LOOKUP (optional)                 │
│  1. rag_engine.query_with_domain_check(question)        │
│     → embed question ONCE (~100ms)                      │
│     → domain check via dot product vs domain vector     │
│     → if in-domain: FAISS search reusing same vector    │
│     → if relevant chunks found: build augmented prompt  │
│     → if out-of-domain or no useful chunks: skip RAG    │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────▼──────────────┐
           │   PROMPT CONSTRUCTION      │
           │                            │
           │  WITH RAG:                 │
           │    system = SYSTEM_WITH_CTX│
           │    user = context + Q      │
           │    history[-6:]            │
           │                            │
           │  WITHOUT RAG:              │
           │    system = SYSTEM_BASE    │
           │    history[-8:]            │
           └─────────────┬──────────────┘
                         │
           ┌─────────────▼──────────────┐
           │   STREAMING GENERATION     │
           │   _chat_stream()           │
           │   - model resolution       │
           │   - TTFT measurement       │
           │   - token-by-token output  │
           │   - total time measurement │
           └─────────────┬──────────────┘
                         │
                         ▼
               Response + Metrics
         (TTFT, total time, RAG score)
```

---

## 3. Model Resolution Strategy

The system employs a **dual-model fallback** pattern:

```python
MODEL          = "asistente"       # fine-tuned model name in Ollama
FALLBACK_MODEL = "qwen2.5:7b"     # base model if fine-tuned is unavailable
```

At session start and on each inference call, `_resolve_model()` queries the Ollama `/api/tags` endpoint to enumerate locally available models. If the fine-tuned model (`asistente`) is registered, it is used; otherwise, the system degrades gracefully to the base Qwen 2.5 7B.

This is critical for development workflows: the base model works "well enough" to test the full pipeline before the fine-tuned GGUF is ready, and the system never crashes due to a missing model.

**Timeout:** The Ollama health check uses a 5-second timeout. On the RK3588 under load, Ollama's HTTP server can take 2–3 seconds to respond if it is actively paging model weights from disk to RAM.

---

## 4. System Prompts

Two system prompts are defined, selected dynamically based on whether RAG context is available:

### 4.1 Base System Prompt (`SYSTEM_ENGINEERING`)

Defines the persona constraints:

- Warm, direct tone — no filler phrases ("Great question!", "Sure!")
- Conceptual depth adapted to the question type: definitions are crisp, comparisons are structured, "why" questions get historical/philosophical context
- Strict prohibition: no code, no calculations, no logic puzzles, no Chinese characters
- Bilingual: responds in the language of the question
- Epistemic honesty: explicitly acknowledges uncertainty ("si no me falla la memoria", "el dato exacto no lo tengo presente")

### 4.2 RAG-Augmented Prompt (`SYSTEM_ENGINEERING_WITH_CONTEXT`)

Extends the base prompt with instructions for grounding answers in retrieved context:

- Cite sources naturally when relevant ("según el libro de Bishop sobre Deep Learning...")
- Trust retrieved context over prior knowledge when they conflict
- **Critical language rule**: RAG context may be in English (from English-language textbooks), but the response must always match the user's language. The model must translate and adapt, not copy verbatim.

### 4.3 RAG Context Injection

When RAG returns relevant chunks, the context is injected as part of the **user message**, not as a system instruction. This design choice is intentional:

```python
augmented_question = (
    f"{rag_context}\n\n"
    f"USER QUESTION: {user_question}\n\n"
    f"Answer based on the context above and your expertise. ..."
)
```

**Rationale:** In instruction-tuned models, content in the system message is treated as behavioral guidance, while content in the user message is treated as grounding information. Placing RAG context in the user message signals to the model that it should *use* the information (cite it, reason over it) rather than *obey* it (follow it as a behavioral instruction). This distinction matters for a 7B model that has limited ability to distinguish between these pragmatic roles.

---

## 5. Streaming & Latency Metrics

The `_chat_stream()` function implements token-by-token streaming with two latency measurements:

- **TTFT (Time To First Token):** Measured from the moment the OpenAI-compatible API call is initiated to the arrival of the first non-empty delta. This metric captures the model's prefill latency — the time to process the full prompt through all transformer layers before autoregressive generation begins. On the RK3588 with INT4 quantization, typical TTFT is 0.5–2 seconds depending on prompt length.

- **Total generation time:** Wall-clock time from request start to final token. Combined with the known `max_tokens` parameter (600 for engineering mode), this gives an approximation of tokens/second throughput.

```
[INGENIERO]: <streamed response>
  ⏱  TTFT=0.8s | Total=6.2s [RAG score=0.72]
```

The RAG score (top cosine similarity from FAISS) is appended to the metrics line when RAG was used, providing a per-turn diagnostic of retrieval quality without requiring separate logging infrastructure.

---

## 6. History Management

Conversation history is maintained as a list of `{"role": "user"|"assistant", "content": str}` dicts. The window sizes differ between RAG and non-RAG turns:

| Mode | History window | Rationale |
|---|---|---|
| Without RAG | Last 8 messages | Standard context for multi-turn conversation |
| With RAG | Last 6 messages (of history) + augmented user message | RAG context consumes ~200–500 tokens; reducing history compensates within the 2048-token sequence length |

When history exceeds 12 messages, it is truncated to the last 12 (FIFO). This hard cap prevents the prompt from exceeding the model's context window (2048 tokens for the fine-tuned GGUF) while preserving enough conversational coherence for follow-up questions.

---

## 7. RAG Integration Point

The RAG engine is loaded **once** at session start via `_load_rag()` (lazy singleton pattern) and reused across all turns. The integration follows a three-outcome decision tree:

```
query_with_domain_check(question)
        │
        ├─→ domain_score < threshold (0.15)
        │     → OUT OF DOMAIN: skip FAISS entirely, answer from model knowledge
        │
        ├─→ domain_score ≥ threshold, but top_score < relevance_threshold (0.40)
        │     → IN DOMAIN but NO USEFUL CONTEXT: answer from model knowledge
        │
        └─→ domain_score ≥ threshold AND top_score ≥ relevance_threshold
              → RELEVANT: inject context, announce "checking books", augment prompt
```

The "checking books" announcement (`"Un momento, déjame revisar en mis libros..."`) is a UX decision: it sets the user's expectation that the response will be grounded in a specific source, and it provides a natural pause while the augmented prompt is assembled.

When RAG is unavailable (missing index, missing dependencies), the session operates transparently without it — no error, no degradation message after the initial startup log.

---

## 8. Standalone vs. Integrated Operation

The module supports two execution modes:

- **Standalone** (`python engineering_session.py`): Runs a full interactive REPL with `input()` for text entry. Useful for isolated testing of the tutor + RAG pipeline without the orchestrator, voice I/O, or intent classification overhead.

- **Integrated** (called from `main.py`): The `run_engineering_session(existing_history)` function accepts and returns conversation history, enabling the orchestrator to persist state across mode switches. When the user exits engineering mode and later returns, the previous conversation is resumed.

Commands available in both modes:
- `salir` / `exit` / `quit` / `menu` → exit session (return to orchestrator or terminate)
- `limpiar` / `clear` / `reset` → reset conversation history to zero

---

## 9. Inference Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `temperature` | 0.3 | Low variance for factual/conceptual content. Higher values (0.5+) caused occasional hallucination of book titles and dates in testing. |
| `max_tokens` | 600 | Tutor responses average 150–300 tokens. The 600 cap provides headroom for detailed comparisons while preventing runaway generation on the SBC. |
| Streaming | Enabled | Required for acceptable UX on edge hardware where full generation can take 5–15 seconds. |

---

## 10. Error Handling

The streaming function wraps the entire generation loop in a try/except that catches any exception from the Ollama client (connection reset, timeout, OOM) and returns a formatted error string as the response text. This ensures the session loop never crashes — the user sees `[ERROR] <message>` and can continue asking questions.

This resilience is critical for an autonomous SBC system where the Ollama server may restart due to thermal throttling, kernel OOM killer activity, or manual model swaps.
