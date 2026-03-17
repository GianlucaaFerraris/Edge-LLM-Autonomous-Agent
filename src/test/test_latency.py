"""
test_latency.py — Benchmark de latencia e inferencia.

Mide tiempo de primera respuesta (TTFT) y throughput (tokens/s).
Genera un reporte en consola y guarda un JSON con los resultados.

Correr con:
    pytest test_latency.py -v -s

O directamente:
    python test_latency.py
"""

import json
import time
import statistics
import datetime
import pytest
import requests
from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/v1"
MODEL      = "asistente"
RESULTS_FILE = "latency_results.json"

# ── Casos de prueba por modo ───────────────────────────────────────────────────
LATENCY_CASES = [
    {
        "id":   "english_short",
        "mode": "english_tutor",
        "system": (
            "You are a warm, friendly spoken English tutor for Spanish-speaking B1/B2 students. "
            "Single spoken paragraph, TTS-ready, no markdown."
        ),
        "user": "Start a conversation about travel. 2 sentences in Spanish, then one English question.",
        "max_tokens": 150,
        "expected_max_s": 15,
    },
    {
        "id":   "engineering_medium",
        "mode": "engineering",
        "system": (
            "You are a brilliant, warm retired engineer. Explain with clarity and depth. "
            "No code, no calculations. Answer in Spanish."
        ),
        "user": "¿Qué es la impedancia en un circuito eléctrico?",
        "max_tokens": 400,
        "expected_max_s": 30,
    },
    {
        "id":   "agent_tool_call",
        "mode": "agent",
        "system": (
            "Sos un asistente personal. Respondés en español rioplatense. "
            "Cuando necesitás una herramienta: TOOL_CALL: {\"tool\": \"nombre\", \"args\": {...}}"
        ),
        "user": "¿Qué tengo pendiente?",
        "max_tokens": 100,
        "expected_max_s": 10,
    },
    {
        "id":   "mode_detection",
        "mode": "selector",
        "system": "",
        "user": (
            "El usuario dijo: 'quiero practicar inglés'\n"
            "Respondé con exactamente una palabra: english_tutor, engineering, agent, o unclear."
        ),
        "max_tokens": 5,
        "expected_max_s": 5,
    },
]

N_RUNS = 3  # repeticiones por caso para promediar


def measure_call(client, model, messages, max_tokens) -> dict:
    """
    Hace una llamada y mide:
      - ttft: tiempo hasta recibir el primer token (streaming)
      - total_time: tiempo total
      - tokens: tokens generados (aproximado por palabras si no hay contador)
    """
    t_start = time.perf_counter()
    first_token_t = None
    full_text = ""

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and first_token_t is None:
            first_token_t = time.perf_counter()
        full_text += delta

    t_end = time.perf_counter()

    total_time = t_end - t_start
    ttft       = (first_token_t - t_start) if first_token_t else total_time
    # Estimación de tokens: ~0.75 palabras/token es heurística para español/inglés
    est_tokens = len(full_text.split()) / 0.75

    return {
        "ttft_s":      round(ttft, 3),
        "total_s":     round(total_time, 3),
        "tokens_est":  round(est_tokens),
        "tps":         round(est_tokens / max(total_time, 0.001), 1),
        "response":    full_text[:200],  # preview
    }


def resolve_model(client) -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else "qwen2.5:7b"
    except Exception:
        return "qwen2.5:7b"


# ── Pytest version ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        pytest.skip("Ollama no está corriendo")
    return OpenAI(base_url=OLLAMA_URL, api_key="ollama")


@pytest.fixture(scope="module")
def model(client):
    return resolve_model(client)


@pytest.mark.parametrize("case", LATENCY_CASES, ids=[c["id"] for c in LATENCY_CASES])
def test_latency(client, model, case):
    """Verifica que cada caso responda dentro del tiempo máximo esperado."""
    messages = []
    if case["system"]:
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["user"]})

    result = measure_call(client, model, messages, case["max_tokens"])

    print(f"\n  [{case['id']}]")
    print(f"    TTFT:       {result['ttft_s']:.2f}s")
    print(f"    Total:      {result['total_s']:.2f}s")
    print(f"    Tokens est: {result['tokens_est']}")
    print(f"    TPS est:    {result['tps']}")
    print(f"    Preview:    {result['response'][:80]}...")

    assert result["total_s"] <= case["expected_max_s"], (
        f"[{case['id']}] Demasiado lento: {result['total_s']:.1f}s "
        f"(máximo esperado: {case['expected_max_s']}s)"
    )


# ── Modo standalone: reporte completo ────────────────────────────────────────

def run_full_benchmark():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("[ERROR] Ollama no está corriendo en localhost:11434")
        return

    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    model  = resolve_model(client)

    print(f"\n{'═'*60}")
    print(f"  BENCHMARK DE LATENCIA — {model}")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Corridas por caso: {N_RUNS}")
    print(f"{'═'*60}\n")

    all_results = {}

    for case in LATENCY_CASES:
        print(f"[{case['id']}] modo={case['mode']}")
        messages = []
        if case["system"]:
            messages.append({"role": "system", "content": case["system"]})
        messages.append({"role": "user", "content": case["user"]})

        runs = []
        for i in range(N_RUNS):
            r = measure_call(client, model, messages, case["max_tokens"])
            runs.append(r)
            status = "✅" if r["total_s"] <= case["expected_max_s"] else "❌"
            print(f"  run {i+1}: TTFT={r['ttft_s']:.2f}s | total={r['total_s']:.2f}s | "
                  f"~{r['tokens_est']}tok | {r['tps']} tok/s  {status}")

        totals = [r["total_s"] for r in runs]
        ttfts  = [r["ttft_s"]  for r in runs]
        tps    = [r["tps"]     for r in runs]

        summary = {
            "total_avg_s":  round(statistics.mean(totals), 3),
            "total_min_s":  round(min(totals), 3),
            "total_max_s":  round(max(totals), 3),
            "ttft_avg_s":   round(statistics.mean(ttfts), 3),
            "tps_avg":      round(statistics.mean(tps), 1),
            "expected_max_s": case["expected_max_s"],
            "passed":       statistics.mean(totals) <= case["expected_max_s"],
        }
        all_results[case["id"]] = summary

        icon = "✅" if summary["passed"] else "❌"
        print(f"  → avg={summary['total_avg_s']:.2f}s | "
              f"TTFT avg={summary['ttft_avg_s']:.2f}s | "
              f"TPS avg={summary['tps_avg']} {icon}\n")

    # Reporte final
    print(f"\n{'─'*60}")
    print("  RESUMEN")
    print(f"{'─'*60}")
    passed = sum(1 for v in all_results.values() if v["passed"])
    total  = len(all_results)
    for case_id, s in all_results.items():
        icon = "✅" if s["passed"] else "❌"
        print(f"  {icon} {case_id:<25} avg={s['total_avg_s']:.2f}s  "
              f"TPS={s['tps_avg']}")
    print(f"\n  Resultado: {passed}/{total} casos dentro del tiempo esperado")

    # Guardar JSON
    output = {
        "model":     model,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_runs":    N_RUNS,
        "results":   all_results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados guardados en: {RESULTS_FILE}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    run_full_benchmark()
