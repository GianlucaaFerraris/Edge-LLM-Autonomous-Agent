"""
engineering_session.py — Engineering tutor session with RAG support.

RAG flow:
    1. rag_engine.query_with_domain_check(question)
       → embeds the question ONCE
       → semantic domain check against precomputed domain vector (~0ms after embed)
       → if in domain: FAISS search reusing same vector (<10ms)
       → if result.relevant: inject context + announce "checking books"
       → if result.not_relevant: answer directly (RAG found nothing useful)
       → if out of domain: answer directly (domain score too low)
"""

import sys
import time

import requests
from openai import OpenAI

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

SYSTEM_ENGINEERING = (
    "You are a brilliant, warm retired engineer and scientist with decades of experience "
    "across software, AI, electronics, physics, chemistry, and systems engineering. "
    "You explain concepts with clarity, depth, and genuine enthusiasm — like a mentor who "
    "loves sharing knowledge. You NEVER write code, never do numerical calculations, and "
    "never solve logic puzzles. Instead, you explain the intuition, the trade-offs, the "
    "history, the analogies, and the real-world implications of technical concepts. "
    "You adapt your depth to what is being asked: a definition gets a crisp explanation, "
    "a comparison gets a structured contrast, a 'why' gets philosophy and context. "
    "You are direct and concrete — no filler phrases like 'Great question!' or 'Sure!'. "
    "If asked in Spanish, answer in Spanish. If asked in English, answer in English. "
    "No bullet-point lists unless the question is explicitly a comparison or enumeration. "
    "No Chinese characters. "
    "When stating specific facts (dates, names, measurements, records), if you are not "
    "completely certain, acknowledge it explicitly: 'si no me falla la memoria', 'creo que', "
    "'el dato exacto no lo tengo presente, pero...' — never substitute an uncertain fact "
    "with a wrong one."
)

# System prompt used when RAG context is available
SYSTEM_ENGINEERING_WITH_CONTEXT = (
    SYSTEM_ENGINEERING +
    "\n\nWhen you receive a RELEVANT CONTEXT FROM KNOWLEDGE BASE section, "
    "use it to ground your answer. Cite the source naturally if relevant "
    "(e.g., 'según el libro de Bishop sobre Deep Learning...'). "
    "If the context contradicts your prior knowledge, trust the context. "
    "If the context is only partially relevant, use what's useful and fill "
    "the rest with your expertise.\n\n"
    "CRITICAL LANGUAGE RULE: The knowledge base context may be in English, "
    "but you MUST ALWAYS respond in the same language the user is writing in. "
    "If the user writes in Spanish, your ENTIRE response must be in Spanish — "
    "translate and adapt the English context, do not copy it verbatim. "
    "Technical terms (e.g., 'backpropagation', 'deep learning') can stay in English "
    "if they are commonly used that way in Spanish technical discourse."
)


def _resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def _chat_stream(messages, temperature=0.3, max_tokens=600) -> tuple[str, float, float]:
    """Streaming with latency metrics."""
    model = _resolve_model()
    t_start = time.perf_counter()
    first_token_t = None
    full_text = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta and first_token_t is None:
                first_token_t = time.perf_counter()
            full_text += delta
            print(delta, end="", flush=True)
    except Exception as e:
        full_text = f"[ERROR] {e}"
        print(full_text)
    t_end = time.perf_counter()
    ttft  = (first_token_t - t_start) if first_token_t else (t_end - t_start)
    return full_text, round(ttft, 2), round(t_end - t_start, 2)


def _load_rag():
    """
    Lazy-loads the RAG engine.
    Returns the engine instance or None if unavailable.

    Note: no longer returns a separate should_use_rag function —
    domain filtering is now handled inside query_with_domain_check().
    """
    try:
        from src.engineering.rag.rag_engine import get_engine
        engine = get_engine()
        if engine.is_ready:
            stats = engine.get_stats()
            print(f"  [RAG] Ready — {stats['vectors']} vectors from "
                  f"{len(stats['sources'])} source(s)")
            print(f"  [RAG] Domain threshold={stats['domain_threshold']} | "
                  f"Relevance threshold={stats['threshold']}")
            return engine
        else:
            print("  [RAG] Index not found — run build_index.py to enable RAG")
            return None
    except ImportError:
        print("  [RAG] Dependencies missing — pip install sentence-transformers faiss-cpu")
        return None
    except Exception as e:
        print(f"  [RAG] Could not load: {e}")
        return None


def _build_rag_prompt(user_question: str, rag_context: str,
                      history: list[dict]) -> list[dict]:
    """
    Builds the message list for a RAG-augmented response.
    The context is injected as part of the user message so the
    LLM treats it as grounding information, not a system instruction.
    """
    augmented_question = (
        f"{rag_context}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        f"Answer based on the context above and your expertise. "
        f"If the context directly addresses the question, use it. "
        f"If not, rely on your knowledge. "
        f"IMPORTANT: Respond in the SAME language the user question is written in. "
        f"If the question is in Spanish, your entire answer must be in Spanish."
    )
    messages = [{"role": "system", "content": SYSTEM_ENGINEERING_WITH_CONTEXT}]
    messages.extend(history[-6:])  # slightly more history for RAG turns
    messages.append({"role": "user", "content": augmented_question})
    return messages


def run_engineering_session(existing_history: list = None) -> list:
    """
    Runs an engineering tutor session with RAG support.
    Accepts existing history to resume a session.
    Returns the history on exit (preserved by main.py).
    """
    history = existing_history or []
    model   = _resolve_model()

    print(f"\n{'═'*60}")
    print(f"  TUTOR DE INGENIERÍA  |  Modelo: {model}")
    print(f"  'salir' → menú  |  'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")

    # Load RAG engine — single instance reused across the whole session
    rag_engine    = _load_rag()
    rag_available = rag_engine is not None

    if history:
        print("[INGENIERO]: Retomando donde lo dejamos. ¿Qué más querés explorar?\n")
    else:
        print("[INGENIERO]: Hola, soy tu tutor de ingeniería. ¿Sobre qué concepto querés hablar?\n")

    while True:
        try:
            user_text = input("[VOS]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in ("salir", "exit", "quit", "menu", "menú"):
            break
        if user_text.lower() in ("limpiar", "clear", "reset"):
            history = []
            print("  [INFO] Contexto limpiado.\n")
            continue

        # ── RAG lookup ────────────────────────────────────────────────────────
        used_rag    = False
        rag_context = ""

        if rag_available:
            t_rag  = time.perf_counter()
            result = rag_engine.query_with_domain_check(user_text)
            t_rag  = round(time.perf_counter() - t_rag, 2)

            if result.relevant:
                # In domain + useful context found → augment the prompt
                print(f"  [RAG] {len(result.chunks)} fragmento(s) encontrado(s) "
                      f"(relevance={result.top_score:.2f}, "
                      f"domain={result.domain_score:.2f}, {t_rag}s)")
                rag_context = result.context
                used_rag    = True
                print("\n[INGENIERO]: ", end="", flush=True)
                print("Un momento, déjame revisar en mis libros... ", end="", flush=True)
            # Silently skip if out of domain or no relevant chunks found.
            # Uncomment the lines below for debug logging:
            # elif result.domain_score > 0 and result.domain_score < rag_engine.domain_threshold:
            #     print(f"  [RAG] Fuera de dominio (domain={result.domain_score:.2f})")
            # elif result.domain_score >= rag_engine.domain_threshold:
            #     print(f"  [RAG] En dominio pero sin contexto útil (top={result.top_score:.2f})")

        # ── Build prompt and generate response ────────────────────────────────
        if not used_rag:
            print("\n[INGENIERO]: ", end="", flush=True)

        history.append({"role": "user", "content": user_text})

        if used_rag:
            messages = _build_rag_prompt(user_text, rag_context, history[:-1])
        else:
            messages = (
                [{"role": "system", "content": SYSTEM_ENGINEERING}]
                + history[-8:]
            )

        response, ttft, total = _chat_stream(messages)

        rag_tag = f" [RAG score={result.top_score:.2f}]" if used_rag else ""
        print(f"\n  ⏱  TTFT={ttft}s | Total={total}s{rag_tag}\n")

        history.append({"role": "assistant", "content": response})
        if len(history) > 12:
            history = history[-12:]

    return history


if __name__ == "__main__":
    run_engineering_session()