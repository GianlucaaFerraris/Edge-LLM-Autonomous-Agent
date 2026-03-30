# Fine-Tuning Pipeline

> **Files:** `finetune_qwen.py`, `generate_dataset_engineer_tutor.py`, `generate_dataset_function_calling.py`, `generate_final_dataset.py`, `generate-topics.py`  
> **Base model:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`  
> **Method:** QLoRA (4-bit base + LoRA adapters)  
> **Export format:** GGUF Q4_K_M (~4.5 GB) for Ollama  
> **Training environment:** Google Colab T4 (free tier) or local NVIDIA GPU (≥8 GB VRAM)

---

## 1. Overview

The fine-tuning pipeline transforms a general-purpose Qwen 2.5 7B-Instruct model into a domain-specialized assistant that operates in three behavioral modes: English conversational tutor, engineering knowledge mentor, and tool-calling agent. The entire process is self-contained and reproducible:

```
Phase 1: Synthetic Data Generation (on SBC or any machine with Ollama)
   qwen2.5:7b (base) → generates training examples → quality filtering

Phase 2: Dataset Assembly
   3 per-mode datasets → validation → merge + shuffle → finetune_dataset_clean.jsonl

Phase 3: QLoRA Training (on GPU — Colab T4 or local)
   Qwen2.5-7B-Instruct + LoRA adapters → train → merge → GGUF Q4_K_M

Phase 4: Deployment (back on SBC)
   GGUF → Ollama create → "asistente" model → used by all sessions
```

The key insight is that **the same base model serves as both the teacher (data generator) and the student (fine-tuning target)**. This self-distillation approach works because the quality control pipeline is strict enough to select only the outputs that already match the desired behavioral distribution, and the fine-tuning process then concentrates the model's probability mass on that distribution.

---

## 2. Synthetic Dataset Generation

### 2.1 Design Principles

Traditional fine-tuning requires human-curated datasets, which are expensive and slow to produce. This system instead uses **structured synthetic generation**: human-authored questions and constraints are combined with LLM-generated answers, then passed through automated quality filters. This is a practical implementation of the self-instruct paradigm (Wang et al., 2023) adapted for a constrained edge deployment.

The generation pipeline enforces several invariants:

- **No Chinese characters.** Qwen 2.5 occasionally emits Chinese tokens, especially for short prompts or when the base model's Chinese prior is triggered. Every generated text is filtered with a Unicode range check (`[\u4e00-\u9fff]`).
- **No code in tutor modes.** Regex-based detection of code patterns (backticks, `def`, `import`, `class`, `for ... in`, etc.) rejects any engineering tutor response containing code.
- **No filler phrases.** Opening patterns like "Great question!", "Sure!", "Absolutely!" are rejected by a compiled regex.
- **Dialect consistency.** Agent responses must use Rioplatense Spanish voseo ("podés", "tenés"). Responses containing tuteo forms ("puedes", "tienes") are rejected.

### 2.2 Engineering Tutor Dataset (`generate_dataset_engineer_tutor.py`)

**Target:** ~400 examples  
**Output:** `engineering_dataset.jsonl`

#### 2.2.1 Question Bank Architecture

Questions are organized along two axes:

**Domains (10):** software, ai_ml, deep_learning, sistemas_operativos, bases_de_datos, electronica, telecomunicaciones, robotica, fisica, quimica

**Interaction types (8):**

| Type | Description | Example |
|---|---|---|
| `def` | Clear definition with intuition | "¿Qué es un embedding vectorial?" |
| `compare` | Structured contrast between concepts | "¿Diferencia entre REST y GraphQL?" |
| `tradeoff` | Honest evaluation of both sides | "¿Cuáles son los trade-offs del fine-tuning vs RAG?" |
| `usecase` | When to use a specific technology | "¿Cuándo tiene sentido usar una base de datos en grafo?" |
| `analogy` | Real-world analogy with breakdown | "Explicame backpropagation con una analogía." |
| `why` | Historical/conceptual motivation | "¿Por qué surgieron los transformers?" |
| `mistake` | Common errors and how to avoid them | "¿Cuál es el error más común al diseñar una API REST?" |
| `resource` | Book/course recommendations | "¿Qué libros recomendás para entender ML?" |

This matrix produces ~180 unique questions. Since the target is 400 examples, the pool is cycled with reshuffling, meaning some questions appear twice but with different LLM-generated answers (temperature=0.5 ensures variation).

#### 2.2.2 Generation Pipeline

Each example is generated through this sequence:

```
1. Select (domain, question, type) from shuffled pool
2. Build type-specific prompt hint (e.g., "Build one strong analogy...")
3. Call LLM: system=SYSTEM + user=hint+question → answer
4. With 55% probability, generate a follow-up:
   a. Select follow-up from type-specific pool (bilingual)
   b. Call LLM: context=Q+A + follow-up → follow-up answer
5. Quality check the full message sequence
6. If passed: save to JSONL; if failed: increment fail counter
```

#### 2.2.3 Follow-Up System

Follow-up questions are pre-defined per interaction type, in both Spanish and English. The language of the follow-up matches the language of the original question (detected via presence of Spanish diacritics and keywords).

Examples for `compare` type:
- ES: "¿En qué escenario elegirías el segundo sobre el primero?"
- EN: "In what scenario would you choose the second over the first?"

The 55% follow-up probability creates a natural mix of single-turn (45%) and two-turn (55%) conversations in the training data, teaching the model to handle both standalone questions and contextual follow-ups.

#### 2.2.4 Quality Control Pipeline

Every generated example passes through `quality_check()`, which applies these filters:

| Check | Threshold | Applies to |
|---|---|---|
| Minimum content length | 20 chars per message | All messages |
| Chinese character detection | 0 allowed | All messages |
| Minimum answer length | 100 chars | Assistant messages |
| Code pattern detection | 0 matches | Assistant messages |
| Numerical calculation detection | 0 matches | Assistant messages |
| Filler phrase detection | 0 matches at start | Assistant messages |
| Bullet/list count | ≤2 items (unless `compare` or `resource`) | Assistant messages |
| Domain cross-contamination | No DL terms in `telecomunicaciones` | Domain-specific |

The bullet/list restriction is particularly important: it forces the model to learn **prose-based explanations** rather than falling back to the common LLM pattern of enumerating bullet points for every question.

#### 2.2.5 Checkpointing

Every 40 examples, the accumulated batch is appended to the output JSONL. This provides crash resilience during long generation runs (~2-4 hours for 400 examples on a 7B model via Ollama).

### 2.3 Agent / Function Calling Dataset (`generate_dataset_function_calling.py`)

**Target:** ~150 examples  
**Output:** `agent_dataset_clean.jsonl`

#### 2.3.1 Tool Coverage

The agent dataset teaches the model to emit structured `TOOL_CALL` JSON and interpret `TOOL_RESULT` responses for 10 tools:

| Tool | Category | Arguments |
|---|---|---|
| `search_web` | Information | `query: str` |
| `task_add` | Productivity | `title: str, notes: str?` |
| `task_list` | Productivity | `filter: str?` |
| `task_done` | Productivity | `title: str` |
| `reminder_set` | Time-based | `text: str, datetime: str` |
| `wa_send` | Communication | `contact: str, message: str` |
| `wa_read` | Communication | `contact: str?` |
| `cal_add` | Calendar | `title, datetime, duration_min?, notes?` |
| `cal_list` | Calendar | `date: str?` |
| `cal_delete` | Calendar | `title, datetime` |

#### 2.3.2 Scenario Structure

Unlike the engineering dataset which uses open-ended generation, the agent dataset uses **predefined scenarios** — complete (request, tool, args, simulated_result, final_response) tuples. This is necessary because tool-calling behavior must be exact: the model must learn the precise JSON format, the correct tool name for each intent, and the appropriate argument structure.

Each scenario produces a 5-message sequence:

```json
[
  {"role": "system",    "content": "<agent system prompt>"},
  {"role": "user",      "content": "Buscame qué es el protocolo MQTT..."},
  {"role": "assistant", "content": "TOOL_CALL: {\"tool\": \"search_web\", \"args\": {\"query\": \"...\"}}"},
  {"role": "user",      "content": "TOOL_RESULT: <simulated result>"},
  {"role": "assistant", "content": "MQTT es un protocolo de mensajería..."}
]
```

For non-tool scenarios (direct questions), the sequence is shorter (3 messages: system, user, assistant).

#### 2.3.3 TOOL_CALL Format

The system uses a **text-based tool calling convention** rather than native function calling:

```
TOOL_CALL: {"tool": "<name>", "args": {<arguments>}}
```

This design decision reflects a constraint of small models: Qwen 2.5 7B does not reliably support the OpenAI function-calling API format, especially after INT4 quantization. A text-based convention is easier to learn during fine-tuning and more robust to parse at runtime (simple string search + JSON extraction).

#### 2.3.4 Dialect Enforcement

The agent system prompt specifies Rioplatense Spanish voseo. The quality check rejects any response containing tuteo forms ("tienes", "puedes", "debes", "has") — these are detected via substring matching against a hardcoded list. This is a training-time constraint that propagates through fine-tuning: the resulting model naturally produces voseo without needing runtime prompt engineering.

### 2.4 Topic Generation (`generate-topics.py`)

**Output:** `topics.json` (~600 topics)

This script generates the topic pool for the English tutor's conversational exercises. It calls the base Qwen 2.5 7B with `response_format={"type": "json_object"}` to produce structured output.

Topics are organized across 5 clusters with ~75 categories total:

1. **Science & Tech** (15 categories): Software Engineering, AI, Robotics, Physics, etc.
2. **Society & Future** (15): AI Ethics, Futurism, Climate Change, etc.
3. **Culture & Humanities** (15): History, Literature, Mythology, etc.
4. **Nature & Science** (10): Biology, Ecology, Astronomy, etc.
5. **Lifestyle & Personal** (10+): Gastronomy, Sports, Argentina-specific topics, etc.

Each category yields 9 topics at `temperature=0.8` (high creativity). Topics are phrased as **noun phrases or headlines**, not questions, to serve as conversation seeds (e.g., "Posibles impactos de la robótica en el sector agronómico" rather than "¿Cómo afecta la robótica al agro?").

### 2.5 Final Dataset Assembly (`generate_final_dataset.py`)

**Inputs:**
- `english_tutor_dataset.jsonl` (expected: 300)
- `engineering_dataset.jsonl` (expected: 400)
- `agent_dataset_clean.jsonl` (expected: 150)

**Output:** `finetune_dataset_clean.jsonl` (~850 examples total)

#### 2.5.1 Per-Mode Validation

Each input file is validated with a mode-specific validator:

**English tutor validation:**
- Exactly 4 messages per example (system, assistant-opening, user-response, assistant-feedback)
- Role order: `[system, assistant, user, assistant]`
- Opening ≥40 chars, student response ≥60 chars, feedback ≥80 chars
- Feedback must contain a follow-up question (`?`)
- No markdown in feedback (no `*`, `#`, backtick)

**Engineering validation:**
- Minimum 3 messages
- First message must be system
- All assistant answers ≥80 chars
- No Chinese characters

**Agent validation:**
- All `TOOL_CALL:` content must parse as valid JSON with `tool` and `args` keys
- Tool names must be in the known set
- No tuteo in Spanish agent responses (voseo enforcement)
- **Hallucination detection:** when a `TOOL_RESULT` is a simple "OK" acknowledgment, the subsequent assistant response is checked for day-of-week mentions that weren't present in prior context — a common hallucination pattern where the model invents a specific day for a completed action.

#### 2.5.2 Mode Inference

Each example's mode is inferred from the system prompt content (keyword matching: "English tutor" → english, "retired engineer" → engineering, "TOOL_CALL" or "herramientas" → agent) or from the `_debug.mode` metadata field if present.

#### 2.5.3 Output

The final dataset is shuffled (seed=42 for reproducibility) and written as clean JSONL with only the `messages` field — all `_debug` metadata is stripped. This ensures the fine-tuning script sees only the training signal.

A coverage check reports pass/fail:
- ≥95%: "✅ Dataset completo. Listo para fine-tuning."
- 80–95%: "⚠️ Dataset parcial. Podés continuar pero el modelo puede quedar menos robusto."
- <80%: "❌ Dataset incompleto." → exits with error.

---

## 3. QLoRA Fine-Tuning (`finetune_qwen.py`)

### 3.1 Method

**QLoRA** (Dettmers et al., 2023) combines 4-bit NormalFloat quantization of the base model with Low-Rank Adaptation (LoRA) adapters that are trained in full precision (fp16/bf16). This achieves near-full-precision fine-tuning quality with dramatically reduced VRAM requirements.

The implementation uses **Unsloth** (Daniel Han, 2024), which provides fused kernels for QLoRA training that are 2x faster than the HuggingFace PEFT baseline while producing identical outputs.

### 3.2 Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base model | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Pre-quantized 4-bit checkpoint, saves download time |
| MAX_SEQ_LEN | 2048 | Covers 99%+ of training examples; examples exceeding this are discarded |
| LoRA rank (r) | 16 | Good balance between expressiveness and parameter count for 7B models |
| LoRA alpha | 32 | α/r = 2.0 — standard scaling factor |
| LoRA dropout | 0.05 | Mild regularization to prevent overfitting on ~850 examples |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention layers + FFN layers |
| Batch size | 2 | Per-GPU batch; reduce to 1 if OOM on Colab |
| Gradient accumulation | 4 | Effective batch = 2 × 4 = 8 |
| Epochs | 3 | Standard for small datasets; more risks overfitting |
| Learning rate | 2e-4 | Standard QLoRA LR for 7B models |
| Warmup ratio | 0.03 | ~25 warmup steps for 850 examples |
| LR scheduler | Cosine | Smooth decay to near-zero by end of training |
| Optimizer | AdamW 8-bit | 8-bit quantized optimizer states, saves ~2 GB VRAM |
| Precision | bf16 (if supported), else fp16 | bf16 preferred for numerical stability |
| Gradient checkpointing | Unsloth mode | Saves ~30% VRAM by recomputing activations instead of storing them |

### 3.3 LoRA Target Modules Explained

The LoRA adapters are applied to **all linear layers** in the transformer:

- **Attention:** `q_proj`, `k_proj`, `v_proj` (query/key/value projections), `o_proj` (output projection) — these control *what* the model attends to and *how* it aggregates information.
- **FFN:** `gate_proj`, `up_proj`, `down_proj` — these form the SwiGLU feed-forward network that applies nonlinear transformations to each token's representation.

Including FFN layers is important for behavioral fine-tuning (as opposed to just knowledge injection) because the FFN layers store the bulk of factual knowledge and output distribution patterns. With rank 16, this produces approximately **trainable params / total params ≈ 0.5–1%** of the model.

### 3.4 ChatML Formatting

Training examples are converted to **Qwen's ChatML format** before tokenization:

```
<|im_start|>system
You are a brilliant, warm retired engineer...<|im_end|>
<|im_start|>user
¿Qué es un embedding vectorial?<|im_end|>
<|im_start|>assistant
Un embedding vectorial es una representación...<|im_end|>
```

This format is critical: Qwen 2.5 was pre-trained with ChatML special tokens. Using a different format (e.g., Llama's `[INST]`) would create a distribution mismatch that degrades generation quality.

### 3.5 Training Monitoring

- Eval loss is computed every 50 steps on a 5% holdout split (seed=42)
- Checkpoints saved every 100 steps, keeping only the 2 best by eval loss (`save_total_limit=2`)
- `load_best_model_at_end=True` ensures the final model is the one with lowest eval loss, not the last checkpoint

### 3.6 GGUF Export

After training, the LoRA adapters are merged into the base model and exported to **GGUF Q4_K_M** format:

```python
model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
```

**Q4_K_M** is a mixed 4-bit quantization scheme from `llama.cpp`:
- Most layers use 4-bit quantization with 256-element groups (Q4_K)
- Attention output and FFN gate layers use higher precision (Q6_K)
- This mixed approach preserves ~98% of FP16 perplexity while achieving 4-bit compression

The resulting file is ~4.5 GB for a 7B parameter model, which fits comfortably in the RK3588's 16 GB RAM with room for the embedding model, FAISS index, and OS overhead.

### 3.7 Deployment to SBC

Post-training steps (executed on the Radxa/Jetson):

```bash
# 1. Copy GGUF to SBC
scp output/gguf/model-unsloth.Q4_K_M.gguf radxa@<IP>:~/models/

# 2. Create Ollama model with Modelfile
ollama create asistente -f Modelfile.assistant

# 3. Verify
ollama run asistente
```

The `Modelfile.assistant` specifies the `FROM` path to the GGUF and any additional parameters (temperature defaults, system prompt, stop tokens).

### 3.8 Training Time Estimates

| Hardware | Estimated Time (850 examples, 3 epochs) |
|---|---|
| Google Colab T4 (16 GB VRAM) | 45–90 minutes |
| RTX 3060 (12 GB VRAM) | 30–60 minutes |
| RTX 4090 (24 GB VRAM) | 15–25 minutes |
| CPU only | Not recommended (8+ hours) |

---

## 4. Dataset Statistics Summary

| Mode | Target Examples | Avg Tokens (approx) | Turns | Language |
|---|---|---|---|---|
| English Tutor | 300 | ~400 | Always 4 messages | English + Spanish feedback |
| Engineering | 400 | ~350 | 3 (45%) or 5 (55%) messages | Bilingual (ES/EN) |
| Agent | 150 | ~250 | 3 (no-tool) or 5 (with-tool) | Spanish (Rioplatense) |
| **Total** | **~850** | **~340** | Mixed | Mixed |

The dataset is intentionally small. For a 7B model with QLoRA, 500–1000 high-quality examples are sufficient to steer behavioral distribution without catastrophic forgetting of the base model's general capabilities. Larger datasets (5K+) risk overfitting and loss of the model's broader knowledge.
