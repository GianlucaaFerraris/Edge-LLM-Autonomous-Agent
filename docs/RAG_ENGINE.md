# RAG Engine

> **File:** `rag_engine.py`  
> **Index format:** FAISS (flat inner product)  
> **Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (~120 MB, 384-dim)  
> **Metadata:** JSON sidecar file  
> **Designed for:** ~16 GB RAM SBC (Radxa Rock 5B / Jetson Orin Nano)

---

## 1. Purpose

The RAG (Retrieval-Augmented Generation) engine augments the Engineering Tutor's responses with grounded context from indexed documents (textbooks, PDFs, technical references). Instead of relying solely on the 7B model's parametric knowledge — which is lossy, especially after INT4 quantization — the engine retrieves relevant text chunks and injects them into the LLM prompt, enabling factually grounded and source-citeable answers.

The engine is designed as an **optional, additive component**: if the FAISS index doesn't exist, or if dependencies are missing, the engineering session operates transparently without RAG. This graceful degradation is critical for an edge system that must remain functional regardless of which components are available.

---

## 2. Architecture

```
                    User Question
                         │
                    ┌────▼────┐
                    │  EMBED  │  sentence-transformers
                    │  ONCE   │  paraphrase-multilingual-MiniLM-L12-v2
                    └────┬────┘  (~100ms on ARM CPU)
                         │
                    q_vec (384-dim, L2-normalized)
                         │
              ┌──────────┼──────────────┐
              │                         │
         ┌────▼─────┐            ┌──────▼──────┐
         │  DOMAIN  │            │  FAISS      │
         │  CHECK   │            │  SEARCH     │
         │ dot(q,d) │            │  top_k=3    │
         └────┬─────┘            └──────┬──────┘
              │                         │
         score < 0.15?          scores + indices
              │                         │
         YES: return              filter by
         not relevant            threshold ≥ 0.40
              │                         │
              ▼                    ┌────▼─────┐
        RAGResult(                 │  FORMAT  │
         relevant=False)           │  CONTEXT │
                                   └────┬─────┘
                                        │
                                   RAGResult(
                                    relevant=True,
                                    context=formatted_text,
                                    chunks=[...],
                                    top_score, domain_score)
```

### 2.1 The Single-Embed Optimization

A critical design decision: the question is embedded **exactly once**, and the same vector is reused for both the domain check and the FAISS search. This might seem trivial, but it halves the embedding latency for in-domain questions:

| Approach | In-domain latency | Out-of-domain latency |
|---|---|---|
| Naive (embed twice) | ~200ms (2× embed) | ~100ms (1× embed + skip FAISS) |
| Single-embed | ~110ms (1× embed + FAISS) | ~100ms (1× embed + skip FAISS) |

On the RK3588's ARM CPU, each embedding call takes ~100ms. Saving 100ms per in-domain question is a meaningful improvement in a pipeline where total TTFT is 0.5–2s.

---

## 3. Semantic Domain Filter

### 3.1 Problem

Not every user question should trigger a FAISS search. If someone asks "¿Qué hora es?" or "Mandame un WhatsApp a Juan", searching a deep learning textbook corpus is wasted computation (and could return misleadingly "relevant" chunks about time-series or communication protocols).

### 3.2 Solution: Domain Description Vector

At load time, a comprehensive domain description text is embedded once:

```
Neural networks, deep learning, machine learning, backpropagation,
gradient descent, transformers, attention mechanism, ...
Robotics, kinematics, inverse kinematics, ...
Electronics, circuits, transistors, ...
Redes neuronales, aprendizaje profundo, ...  (Spanish equivalents)
```

This ~500-word description covers the semantic space of the indexed corpus in **both English and Spanish**. It is embedded into a single 384-dimensional vector (`_domain_vec`) and stored in memory.

### 3.3 Domain Check Mechanism

For each incoming question:

```python
domain_score = float(np.dot(q_vec, self._domain_vec.T).item())
```

This dot product between L2-normalized vectors is equivalent to cosine similarity and takes **<0.01ms** — essentially free after the embedding is already computed.

If `domain_score < DOMAIN_THRESHOLD (0.15)`, the question is deemed out-of-domain and FAISS is never consulted. The threshold of 0.15 was chosen empirically:

- "¿Qué es backpropagation?" → domain_score ≈ 0.45–0.60 (clearly in-domain)
- "¿Cuándo tiene sentido usar Redis?" → domain_score ≈ 0.20–0.30 (border, but valid)
- "Mandame un WhatsApp a Juan" → domain_score ≈ 0.03–0.08 (clearly out-of-domain)
- "¿Qué hora es?" → domain_score ≈ 0.02–0.05 (clearly out-of-domain)

### 3.4 Why Not Keyword Matching?

The previous implementation used a keyword list (`RAG_DOMAINS = {"neural", "backpropagation", ...}`). This failed for:

- Spanish queries without exact English keywords: "¿Cómo funciona la retropropagación?" → no match
- Paraphrased questions: "¿Por qué las redes profundas necesitan tantos datos?" → no keyword hit
- Tangential but valid queries: "¿Qué problema tenían los LSTM antes de los transformers?" → only partial keyword match

The semantic domain vector handles all these cases because it operates in embedding space, where semantic similarity transcends exact word matches and language boundaries.

---

## 4. FAISS Index

### 4.1 Index Type

The engine uses `faiss.IndexFlatIP` (flat index with inner product similarity). For the expected corpus size (~1,000–10,000 chunks), a flat index provides exact nearest-neighbor search in <10ms on ARM CPU. More sophisticated index types (IVF, HNSW) add complexity without meaningful speedup at this scale.

### 4.2 Vector Properties

- **Dimensionality:** 384 (MiniLM-L12-v2 output)
- **Normalization:** L2-normalized at index time and query time, making inner product equivalent to cosine similarity (range: -1 to 1)
- **Storage:** `faiss.index` binary file + `metadata.json` sidecar

### 4.3 Metadata Structure

Each vector in the FAISS index has a corresponding entry in `metadata.json`:

```json
{
  "0": {
    "text": "Backpropagation computes the gradient of the loss function...",
    "source": "Bishop_DeepLearning_Ch5",
    "chunk": 42
  },
  "1": { ... }
}
```

The metadata is loaded entirely into memory (typically <50 MB for a few thousand chunks). This avoids disk I/O during query time.

---

## 5. Embedding Model Selection

### 5.1 Why `paraphrase-multilingual-MiniLM-L12-v2`

| Criterion | Requirement | MiniLM-L12-v2 |
|---|---|---|
| Bilingual (EN/ES) | Must embed both languages into same semantic space | ✅ 50+ languages |
| Model size | Must fit in RAM alongside Qwen 7B + FAISS | ✅ ~120 MB |
| Inference speed | <200ms per query on ARM CPU | ✅ ~100ms on RK3588 |
| Quality | Good enough for textbook-level retrieval | ✅ Strong paraphrase detection |
| Self-hosted | No API calls, no network dependency | ✅ sentence-transformers local |

### 5.2 Why Not Larger Models

Models like `e5-large-v2` or `bge-large-en` produce better embeddings but consume ~1.3 GB RAM and take ~400ms per query on ARM. For the use case (textbook retrieval, not fine-grained semantic search), the quality difference does not justify the resource cost on a 16 GB SBC.

---

## 6. Query Pipeline Detail

### 6.1 `query_with_domain_check(question)` — Recommended Entry Point

This is the unified query method used by `engineering_session.py`:

```
1. Embed question → q_vec (384-dim, normalized)
2. Domain check: dot(q_vec, domain_vec) → domain_score
   - If domain_score < 0.15 → return RAGResult(relevant=False)
3. FAISS search: index.search(q_vec, top_k=3) → (scores, indices)
4. Filter: discard chunks with score < 0.40 (relevance threshold)
5. If no chunks survive filtering → return RAGResult(relevant=False)
6. Format surviving chunks into context block
7. Return RAGResult(relevant=True, context=..., chunks=..., ...)
```

### 6.2 `query(question)` — Backwards-Compatible Entry Point

Identical to `query_with_domain_check()` but **skips the domain filter** — goes straight to FAISS. Kept for CLI testing and debugging where you want to see what FAISS returns regardless of domain relevance.

---

## 7. Context Formatting

Retrieved chunks are formatted into a structured block for LLM injection:

```
RELEVANT CONTEXT FROM KNOWLEDGE BASE
(Retrieved for query: "¿Qué es la retropropagación?")

[1] Source: Bishop_DeepLearning_Ch5 (chunk 42, relevance: 0.78)
Backpropagation computes the gradient of the loss function with respect to
each weight by applying the chain rule iteratively from the output layer
back through the network...

[2] Source: Goodfellow_DL_Ch6 (chunk 15, relevance: 0.65)
The backward pass of backpropagation allows efficient computation of
gradients in computational graphs...
```

### 7.1 Word Cap

Context is truncated at `MAX_CONTEXT_WORDS = 500` words. This hard cap prevents long chunks from consuming too much of the 2048-token context window, leaving room for the system prompt, conversation history, and the model's response.

If a chunk exceeds the remaining word budget, it is truncated mid-text with an ellipsis (`...`). This is acceptable because the LLM only needs the most relevant portion — it doesn't need to see the entire source passage.

---

## 8. Relevance Thresholds

Two thresholds gate the RAG pipeline:

### 8.1 Domain Threshold (0.15)

Controls whether FAISS is consulted at all. Set low because:
- The embedding model maps diverse phrasings into overlapping regions
- False negatives (missing a relevant query) are worse than false positives (querying FAISS unnecessarily) — a wasted FAISS search costs <10ms, a missed augmentation costs answer quality

### 8.2 Relevance Threshold (0.40)

Controls whether retrieved chunks are injected into the prompt. Set moderate because:
- Below 0.40, chunks are typically tangentially related (e.g., same domain but different concept)
- Injecting low-relevance context confuses a 7B model — it may hallucinate connections between the context and the question
- Above 0.40, chunks reliably contain directly relevant information

The gap between thresholds (0.15 → 0.40) creates a "FAISS is consulted but nothing is useful" zone. This is by design: it allows the engine to report accurate diagnostic information (domain_score, top_score) without injecting noise into the prompt.

---

## 9. RAGResult Dataclass

Every query returns a `RAGResult` with full diagnostic information:

```python
@dataclass
class RAGResult:
    relevant:     bool         # True if at least one chunk passed relevance threshold
    context:      str          # Formatted text for LLM injection ("" if not relevant)
    chunks:       list[dict]   # Raw chunks with scores (for debugging)
    top_score:    float        # Best cosine similarity (0–1)
    domain_score: float        # Similarity to domain description (0–1)
    query:        str          # Original query string
```

This design enables the engineering session to make fine-grained decisions:
- `result.relevant` → whether to augment the prompt
- `result.top_score` → displayed in metrics line for per-turn diagnostics
- `result.domain_score` → available for debug logging
- `result.chunks` → available for future features (e.g., source citation in UI)

---

## 10. Singleton Pattern

The engine is instantiated once per application lifetime via `get_engine()`:

```python
_engine: Optional[RAGEngine] = None

def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.load()
    return _engine
```

This ensures:
1. The FAISS index is loaded from disk once (~50–200ms)
2. The embedding model is loaded once (~2–5s on ARM)
3. The domain vector is precomputed once
4. All subsequent queries reuse these cached resources

The `engineering_session.py` calls `_load_rag()` at session start, which wraps `get_engine()` with error handling and user-facing status messages.

---

## 11. Failure Modes & Graceful Degradation

| Failure | Behavior |
|---|---|
| FAISS index file missing | `load()` returns False; session runs without RAG |
| `faiss-cpu` not installed | ImportError caught; session runs without RAG |
| `sentence-transformers` not installed | ImportError caught; session runs without RAG |
| Embedding fails at query time | Returns empty RAGResult; session generates from model knowledge |
| FAISS search fails at query time | Returns empty RAGResult; session generates from model knowledge |
| Domain vector fails to precompute | Falls back to always querying FAISS (no domain filter) |

Every failure path is explicitly handled to ensure the system never crashes due to RAG issues. On an autonomous SBC, a crash means the user loses the entire assistant until someone manually restarts it.

---

## 12. Resource Consumption

| Resource | Quantity | Notes |
|---|---|---|
| RAM (embedding model) | ~500 MB | Loaded once, persistent |
| RAM (FAISS index) | ~10–50 MB | Depends on corpus: 4 bytes × 384 dims × N vectors |
| RAM (metadata) | ~5–30 MB | JSON, depends on chunk text length |
| Disk (index + metadata) | ~15–80 MB | Binary FAISS + JSON |
| Query latency (in-domain) | ~110 ms | 100ms embed + <10ms FAISS |
| Query latency (out-of-domain) | ~100 ms | 100ms embed + <0.01ms domain check |
| CPU during query | Single core | sentence-transformers inference is single-threaded by default |
