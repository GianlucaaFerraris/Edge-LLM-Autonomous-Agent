"""
rag_engine.py — Queries the FAISS index and returns context for the LLM.

Used by engineering_session.py to augment responses with knowledge
from indexed documents when the topic falls within the RAG domain.

Usage:
    from src.engineering.rag.rag_engine import RAGEngine

    rag = RAGEngine()         # loads index once, reuses across queries
    result = rag.query_with_domain_check("¿Qué es la retropropagación?")

    if result.relevant:
        # inject result.context into the LLM prompt
        ...
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
INDEX_FILE = _HERE / "index" / "faiss.index"
META_FILE  = _HERE / "index" / "metadata.json"

# ── Query config ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K               = 3     # number of chunks to retrieve per query
RELEVANCE_THRESHOLD = 0.4  # minimum cosine similarity score (0-1)
                             # below this → context is not injected
MAX_CONTEXT_WORDS   = 500   # hard cap on injected context length
DOMAIN_THRESHOLD    = 0.15  # minimum similarity to the domain description
                             # below this → FAISS is never consulted

# ── Domain description ────────────────────────────────────────────────────────
# This text is embedded once at load time and used as a semantic domain filter.
# It replaces the keyword-based RAG_DOMAINS check with a proper vector similarity.
# Covers both English and Spanish so bilingual queries are handled correctly.
DOMAIN_DESCRIPTION = """
Neural networks, deep learning, machine learning, backpropagation,
gradient descent, transformers, attention mechanism, computer vision,
convolutional neural networks, reinforcement learning, large language models,
fine-tuning, LoRA, QLoRA, embeddings, BERT, GPT, recommender systems,
collaborative filtering, generative models, diffusion models, GANs,
overfitting, regularization, activation functions, loss functions,
batch normalization, dropout, encoder, decoder, LSTM, RNN.
Robotics, kinematics, inverse kinematics, forward kinematics,
path planning, motion planning, SLAM, probabilistic robotics,
control systems, PID controller, actuators, sensors, trajectory.
Electronics, circuits, transistors, amplifiers, op-amps, filters,
signal processing, impedance, microcontrollers, FPGA, PWM, ADC, DAC,
oscillators, voltage, current, resistors, capacitors, inductors.
Linear algebra, probability, statistics, Bayesian inference,
Markov chains, differential equations, Fourier transform, Laplace,
eigenvalues, matrix operations, data structures, algorithms.
Redes neuronales, aprendizaje profundo, aprendizaje automático,
retropropagación, descenso de gradiente, sobreajuste, regularización,
función de activación, función de pérdida, normalización por lotes,
codificador, decodificador, ajuste fino, modelo generativo.
Robótica, cinemática, cinemática inversa, cinemática directa,
planificación de trayectoria, planificación de movimiento,
sistemas de control, sensor, actuador, trayectoria.
Electrónica, circuitos, transistores, amplificadores, filtros,
procesamiento de señales, impedancia, microcontroladores,
tensión, corriente, resistencias, capacitores, inductores.
Álgebra lineal, probabilidad, estadística, inferencia bayesiana,
cadenas de Markov, ecuaciones diferenciales, transformada de Fourier,
estructuras de datos, algoritmos, sistemas operativos.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """
    Result of a RAG query.

    Attributes:
        relevant:      True if at least one chunk exceeded the relevance threshold.
        context:       Formatted text ready to inject into the LLM prompt.
                       Empty string if not relevant.
        chunks:        Raw retrieved chunks with scores (for debugging).
        top_score:     Cosine similarity of the best matching chunk (0-1).
        domain_score:  Cosine similarity against the domain description (0-1).
        query:         The original query string.
    """
    relevant:     bool
    context:      str
    chunks:       list[dict]
    top_score:    float
    domain_score: float
    query:        str

    def __repr__(self):
        status = (
            f"relevant (score={self.top_score:.3f})"
            if self.relevant
            else f"not relevant (score={self.top_score:.3f})"
        )
        return f"RAGResult({status}, domain={self.domain_score:.3f}, {len(self.chunks)} chunks)"


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Loads the FAISS index once and exposes query methods.

    Design decisions:
    - Lazy loading: index is only loaded when first query() is called.
    - Singleton-friendly: instantiate once in engineering_session.py,
      reuse across the whole session.
    - Threshold filtering: chunks below RELEVANCE_THRESHOLD are discarded
      so the LLM never receives irrelevant context that could confuse it.
    - Semantic domain filter: instead of keyword matching, a domain description
      is embedded once at load time. query_with_domain_check() uses this vector
      to decide if FAISS should be consulted at all — handles Spanish and
      natural-language phrasings that don't contain exact English keywords.
    - Single embed: query_with_domain_check() embeds the question only once
      and reuses the same vector for both the domain check and the FAISS search,
      keeping latency identical to the original query() path.
    """

    def __init__(self,
                 index_file:        Path  = INDEX_FILE,
                 meta_file:         Path  = META_FILE,
                 model_name:        str   = EMBEDDING_MODEL,
                 top_k:             int   = TOP_K,
                 threshold:         float = RELEVANCE_THRESHOLD,
                 domain_threshold:  float = DOMAIN_THRESHOLD):

        self.index_file       = index_file
        self.meta_file        = meta_file
        self.model_name       = model_name
        self.top_k            = top_k
        self.threshold        = threshold
        self.domain_threshold = domain_threshold

        self._index      = None
        self._metadata   = None
        self._model      = None
        self._domain_vec = None   # precomputed domain embedding
        self._ready      = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Loads the FAISS index, metadata, embedding model, and domain vector.
        Returns True if successful, False if index doesn't exist yet.
        """
        if self._ready:
            return True

        if not self.index_file.exists() or not self.meta_file.exists():
            print(f"  [RAG] Index not found at {self.index_file}")
            print(f"        Run build_index.py first.")
            return False

        # Load FAISS index
        try:
            import faiss
            self._index = faiss.read_index(str(self.index_file))
            print(f"  [RAG] Index loaded: {self._index.ntotal} vectors")
        except ImportError:
            print("  [RAG] faiss-cpu not installed: pip install faiss-cpu")
            return False
        except Exception as e:
            print(f"  [RAG] Error loading index: {e}")
            return False

        # Load metadata
        try:
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        except Exception as e:
            print(f"  [RAG] Error loading metadata: {e}")
            return False

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except ImportError:
            print("  [RAG] sentence-transformers not installed: pip install sentence-transformers")
            return False
        except Exception as e:
            print(f"  [RAG] Error loading model: {e}")
            return False

        # Precompute domain vector — paid once, reused every query
        try:
            self._domain_vec = self._model.encode(
                [DOMAIN_DESCRIPTION],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            print(f"  [RAG] Domain vector precomputed (threshold={self.domain_threshold})")
        except Exception as e:
            print(f"  [RAG] Warning: could not precompute domain vector: {e}")
            # Non-fatal: query_with_domain_check will fall back to always querying FAISS

        self._ready = True
        return True

    # ── Query (original — kept for backwards compatibility and CLI) ───────────

    def query(self, question: str) -> RAGResult:
        """
        Retrieves the most relevant chunks for a given question.
        Does NOT perform a domain check — goes straight to FAISS.
        Kept for backwards compatibility and direct CLI testing.

        For session use, prefer query_with_domain_check().
        """
        empty = RAGResult(
            relevant=False, context="", chunks=[],
            top_score=0.0, domain_score=0.0, query=question
        )

        if not self._ready and not self.load():
            return empty

        try:
            q_vec = self._model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        except Exception as e:
            print(f"  [RAG] Embedding error: {e}")
            return empty

        return self._search_and_build(q_vec, question, domain_score=0.0)

    # ── Query with domain check (recommended for session use) ─────────────────

    def query_with_domain_check(self, question: str) -> RAGResult:
        """
        Unified entry point for session use.

        Embeds the question ONCE and reuses the same vector for:
          1. Domain check  — dot product against precomputed domain vector (~0ms)
          2. FAISS search  — only if domain check passes (<10ms)

        Latency vs. original flow:
          - Question IS in domain:  identical (~100ms embed + <10ms FAISS)
          - Question NOT in domain: +~100ms vs. keyword check (was ~0ms),
            but this cost only matters for out-of-domain questions where
            the LLM answers immediately anyway.

        Args:
            question: User's question in any language.

        Returns:
            RAGResult. If domain_score < domain_threshold, returns
            relevant=False without consulting FAISS.
        """
        empty = RAGResult(
            relevant=False, context="", chunks=[],
            top_score=0.0, domain_score=0.0, query=question
        )

        if not self._ready and not self.load():
            return empty

        # ── Embed once ────────────────────────────────────────────────────────
        try:
            q_vec = self._model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        except Exception as e:
            print(f"  [RAG] Embedding error: {e}")
            return empty

        # ── Domain check ──────────────────────────────────────────────────────
        domain_score = 0.0
        if self._domain_vec is not None:
            domain_score = float(np.dot(q_vec, self._domain_vec.T).item())
            if domain_score < self.domain_threshold:
                # Out of domain — skip FAISS entirely
                return RAGResult(
                    relevant=False, context="", chunks=[],
                    top_score=0.0, domain_score=domain_score, query=question
                )
        # If domain_vec failed to load, we skip the check and always query FAISS

        # ── FAISS search — reuses q_vec, no re-embedding ──────────────────────
        return self._search_and_build(q_vec, question, domain_score=domain_score)

    # ── Shared FAISS search + result builder ──────────────────────────────────

    def _search_and_build(self, q_vec: np.ndarray,
                          question: str,
                          domain_score: float) -> RAGResult:
        """
        Runs FAISS search with a precomputed query vector and builds the result.
        Separated so both query() and query_with_domain_check() share the logic.
        """
        empty = RAGResult(
            relevant=False, context="", chunks=[],
            top_score=0.0, domain_score=domain_score, query=question
        )

        try:
            scores, indices = self._index.search(q_vec, self.top_k)
            scores  = scores[0].tolist()
            indices = indices[0].tolist()
        except Exception as e:
            print(f"  [RAG] Search error: {e}")
            return empty

        retrieved = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            if score < self.threshold:
                continue
            meta = self._metadata.get(str(idx))
            if not meta:
                continue
            retrieved.append({
                "text":   meta["text"],
                "source": meta["source"],
                "chunk":  meta["chunk"],
                "score":  round(float(score), 4),
            })

        if not retrieved:
            top_score = scores[0] if scores else 0.0
            return RAGResult(
                relevant=False, context="", chunks=[],
                top_score=float(top_score), domain_score=domain_score, query=question
            )

        context   = self._format_context(retrieved, question)
        top_score = retrieved[0]["score"]

        return RAGResult(
            relevant=True,
            context=context,
            chunks=retrieved,
            top_score=top_score,
            domain_score=domain_score,
            query=question,
        )

    # ── Context formatter ─────────────────────────────────────────────────────

    def _format_context(self, chunks: list[dict], question: str) -> str:
        """
        Formats retrieved chunks into a context block for the LLM.

        Designed to:
        - Tell the LLM WHERE the info comes from (source + chunk index)
        - Be concise — respects MAX_CONTEXT_WORDS
        - Not overwhelm the LLM with too much text
        """
        lines = [
            "RELEVANT CONTEXT FROM KNOWLEDGE BASE",
            f"(Retrieved for query: \"{question}\")\n",
        ]

        word_count = 0
        for i, chunk in enumerate(chunks, 1):
            chunk_words = chunk["text"].split()

            remaining = MAX_CONTEXT_WORDS - word_count
            if remaining <= 0:
                break
            if len(chunk_words) > remaining:
                chunk_words = chunk_words[:remaining]
                truncated = True
            else:
                truncated = False

            text = " ".join(chunk_words)
            if truncated:
                text += "..."

            lines.append(
                f"[{i}] Source: {chunk['source']} (chunk {chunk['chunk']}, "
                f"relevance: {chunk['score']:.2f})\n{text}\n"
            )
            word_count += len(chunk_words)

        return "\n".join(lines)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def index_size(self) -> int:
        return self._index.ntotal if self._index else 0

    def get_stats(self) -> dict:
        """Returns index statistics for debugging."""
        if not self._ready:
            return {"ready": False}
        sources = set()
        for v in self._metadata.values():
            sources.add(v.get("source", "unknown"))
        return {
            "ready":            True,
            "vectors":          self._index.ntotal,
            "sources":          sorted(sources),
            "model":            self.model_name,
            "threshold":        self.threshold,
            "domain_threshold": self.domain_threshold,
            "top_k":            self.top_k,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton for use across the session
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    """Returns the global RAGEngine instance (lazy init)."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.load()
    return _engine


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is backpropagation?"
    print(f"\nQuery: {query}\n")

    engine = get_engine()
    if not engine.is_ready:
        print("Index not ready. Run build_index.py first.")
        sys.exit(1)

    print(f"Index stats: {engine.get_stats()}\n")

    result = engine.query_with_domain_check(query)
    print(f"Result: {result}")
    if result.relevant:
        print(f"\n{'─'*60}")
        print(result.context)
    elif result.domain_score > 0:
        print(f"Domain score {result.domain_score:.3f} below threshold "
              f"({engine.domain_threshold}) — out of domain, skipped FAISS.")
    else:
        print(f"Top FAISS score {result.top_score:.3f} below relevance "
              f"threshold ({engine.threshold}) — no useful context found.")