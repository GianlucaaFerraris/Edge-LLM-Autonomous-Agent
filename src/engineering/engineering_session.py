"""
engineering_session.py — Sesión del tutor de ingeniería.

Módulo standalone extraído de manual_chat.py.
Puede correrse directamente o ser llamado desde main.py.
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


def _resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def _chat_stream(messages, temperature=0.3, max_tokens=600) -> tuple[str, float, float]:
    """Streaming con métricas de latencia."""
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


def run_engineering_session(existing_history: list = None) -> list:
    """
    Corre una sesión del tutor de ingeniería.
    Acepta historial previo para reanudar.
    Retorna el historial al salir (para que main.py lo conserve).
    """
    history = existing_history or []

    print(f"\n{'═'*60}")
    print(f"  TUTOR DE INGENIERÍA  |  Modelo: {_resolve_model()}")
    print(f"  'salir' → menú | 'limpiar' → nuevo contexto")
    print(f"{'═'*60}\n")

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

        history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": SYSTEM_ENGINEERING}] + history[-8:]

        print("\n[INGENIERO]: ", end="", flush=True)
        response, ttft, total = _chat_stream(messages)
        print(f"\n  ⏱  TTFT={ttft}s | Total={total}s\n")

        history.append({"role": "assistant", "content": response})
        if len(history) > 12:
            history = history[-12:]

    return history


if __name__ == "__main__":
    run_engineering_session()