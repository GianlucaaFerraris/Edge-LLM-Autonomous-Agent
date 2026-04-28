"""
benchmark_suites.py — las 5 suites de test.

Cada suite retorna una lista de dicts (una fila por iteración) que después
el reporter agrega a estadísticos. El formato tabular facilita dumpear a CSV
para pasar a pandas/Excel si se quiere análisis más fino.

Suites:
  1. suite_llm:      inferencia pura, sin intent ni RAG
  2. suite_intent:   latencia del clasificador de intents
  3. suite_rag:      comparación directa RAG vs no-RAG con mismo prompt
  4. suite_e2e:      turno completo intent → (RAG?) → chat
  5. suite_thermal:  loop continuo para stress térmico
"""

import sys
import time

from benchmark_lib import ollama_chat_stream


MODEL = "asistente"
SYSTEM_NEUTRAL = "Sos un asistente útil. Respondé en español, breve y claro."


# ─────────────────────────────────────────────────────────────────────────────
# Suite 1: LLM puro
# ─────────────────────────────────────────────────────────────────────────────

def suite_llm(prompts, iterations, warmup=2, progress=None):
    """
    Mide inferencia bruta del LLM. No pasa por intent ni RAG.

    El objetivo es caracterizar el modelo en sí: TTFT puro, TPS puro,
    sin ningún overhead del pipeline de Agenty. Es el baseline contra
    el que se comparan las otras suites.
    """
    results = []
    total_runs = len(prompts) * (iterations + warmup)
    run = 0

    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_NEUTRAL},
            {"role": "user",   "content": p["text"]},
        ]

        # Warmup: descarta N iteraciones para estabilizar KV cache y caches CUDA
        for _ in range(warmup):
            run += 1
            if progress: progress(run, total_runs, f"warmup {p['id']}")
            ollama_chat_stream(MODEL, messages, max_tokens=400)

        for i in range(iterations):
            run += 1
            if progress: progress(run, total_runs, f"{p['id']} iter {i+1}")
            r = ollama_chat_stream(MODEL, messages, max_tokens=400)
            results.append({
                "suite":          "llm",
                "prompt_id":      p["id"],
                "prompt_lang":    p.get("lang", "es"),
                "iteration":      i + 1,
                "ttft_s":         round(r.ttft_s, 4),
                "total_s":        round(r.total_s, 4),
                "tokens":         r.eval_count,
                "prompt_tokens":  r.prompt_eval_count,
                "tps_server":     round(r.tps_server, 2),
                "tps_client":     round(r.tps_client, 2),
                "prompt_eval_ms": round(r.prompt_eval_duration_ns / 1e6, 2),
                "error":          r.error,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Suite 2: Intent classifier
# ─────────────────────────────────────────────────────────────────────────────

def suite_intent(prompts_idle, prompts_engineering, iterations,
                 warmup=2, progress=None):
    """
    Mide la latencia que el clasificador de intents le agrega a cada turno.

    Es crítico porque se ejecuta ANTES del LLM principal, y si tarda mucho
    la UX se degrada. En Agenty el clasificador hace un mini call al mismo
    LLM con max_tokens bajo — la optimización depende de esa latencia.
    """
    try:
        from src.orchestrator.intent_classifier import (
            classify_idle, classify_engineering,
        )
    except ImportError as e:
        print(f"\n[ERROR] No se puede importar intent_classifier: {e}",
              file=sys.stderr)
        return []

    results = []
    cases = [
        ("idle",        classify_idle,        prompts_idle),
        ("engineering", classify_engineering, prompts_engineering),
    ]

    for mode, fn, prompts in cases:
        for p in prompts:
            # Warmup específico por prompt (evita overhead del primer call)
            for _ in range(warmup):
                try: fn(p["text"])
                except Exception: pass

            for i in range(iterations):
                t0 = time.perf_counter()
                try:
                    intent = fn(p["text"])
                    action = intent.action
                    conf   = intent.confidence
                    err    = ""
                except Exception as e:
                    action = "error"
                    conf   = "unknown"
                    err    = str(e)
                elapsed = time.perf_counter() - t0

                results.append({
                    "suite":       "intent",
                    "mode":        mode,
                    "prompt_id":   p["id"],
                    "iteration":   i + 1,
                    "latency_s":   round(elapsed, 4),
                    "predicted":   action,
                    "expected":    p.get("expected", ""),
                    "correct":     action == p.get("expected", action),
                    "confidence":  conf,
                    "error":       err,
                })
                if progress:
                    progress(len(results), -1, f"intent/{mode}/{p['id']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Suite 3: RAG vs no-RAG (A/B en mismo prompt)
# ─────────────────────────────────────────────────────────────────────────────

def suite_rag(prompts, iterations, warmup=2, progress=None):
    """
    Para cada prompt técnico, corre dos veces:
      A) sin RAG: LLM con solo el SYSTEM_ENGINEERING
      B) con RAG: LLM con chunks FAISS inyectados al prompt

    Mide overhead del RAG (embed + FAISS) y diferencia en latencia total
    y prompt inflation. Permite cuantificar el trade-off RAG vs baseline.
    """
    try:
        from src.engineering.engineering_session import (
            SYSTEM_ENGINEERING, _load_rag, _build_rag_prompt,
        )
    except ImportError as e:
        print(f"\n[ERROR] Cannot import engineering_session: {e}",
              file=sys.stderr)
        return []

    rag_engine = _load_rag()
    if rag_engine is None:
        print("\n[WARN] RAG engine no disponible. Suite RAG saltada.")
        return []

    results = []
    total = len(prompts) * iterations
    run = 0

    for p in prompts:
        text = p["text"]
        messages_no_rag = [
            {"role": "system", "content": SYSTEM_ENGINEERING},
            {"role": "user",   "content": text},
        ]

        # Warmup ambas ramas (cargan embeddings, FAISS, KV caches)
        for _ in range(warmup):
            ollama_chat_stream(MODEL, messages_no_rag, max_tokens=200)
            try: rag_engine.query_with_domain_check(text)
            except Exception: pass

        for i in range(iterations):
            run += 1
            if progress: progress(run, total, f"rag/{p['id']} iter {i+1}")

            # ── A: sin RAG ──
            r_no = ollama_chat_stream(MODEL, messages_no_rag, max_tokens=500)

            # ── B: con RAG ──
            t_rag = time.perf_counter()
            try:
                rag_result = rag_engine.query_with_domain_check(text)
            except Exception as e:
                rag_result = None
                print(f"\n[WARN] RAG query failed: {e}")
            rag_time = time.perf_counter() - t_rag

            if rag_result and rag_result.relevant:
                messages_rag = _build_rag_prompt(text, rag_result.context, [])
            else:
                messages_rag = messages_no_rag  # fallback: RAG decidió no fire

            r_yes = ollama_chat_stream(MODEL, messages_rag, max_tokens=500)

            results.append({
                "suite":            "rag",
                "prompt_id":        p["id"],
                "iteration":        i + 1,
                "expected_rag":     p.get("expected_rag", True),
                # Sin RAG
                "noRAG_ttft_s":     round(r_no.ttft_s, 4),
                "noRAG_total_s":    round(r_no.total_s, 4),
                "noRAG_tokens":     r_no.eval_count,
                "noRAG_tps":        round(r_no.tps_server, 2),
                "noRAG_prompt_tokens": r_no.prompt_eval_count,
                # RAG metadata
                "rag_overhead_s":    round(rag_time, 4),
                "rag_triggered":     bool(rag_result and rag_result.relevant),
                "rag_top_score":     round(getattr(rag_result, "top_score", 0), 3) if rag_result else 0,
                "rag_domain_score":  round(getattr(rag_result, "domain_score", 0), 3) if rag_result else 0,
                "rag_n_chunks":      len(getattr(rag_result, "chunks", [])) if rag_result else 0,
                # Con RAG
                "RAG_ttft_s":       round(r_yes.ttft_s, 4),
                "RAG_total_s":      round(r_yes.total_s, 4),
                "RAG_tokens":       r_yes.eval_count,
                "RAG_tps":          round(r_yes.tps_server, 2),
                "RAG_prompt_tokens": r_yes.prompt_eval_count,
                # Deltas
                "total_overhead_s": round(rag_time + (r_yes.total_s - r_no.total_s), 4),
                "prompt_inflation": r_yes.prompt_eval_count - r_no.prompt_eval_count,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Suite 4: End-to-end (turno completo)
# ─────────────────────────────────────────────────────────────────────────────

def suite_e2e(prompts, iterations, warmup=1, progress=None):
    """
    Simula un turno completo en modo engineering:
      1. classify_engineering(user_text) → action
      2. si action == 'respond': RAG lookup
      3. chat stream al LLM con contexto

    Reporta tiempo por fase. Es el número que más importa para UX porque
    es lo que siente el usuario por cada input que le manda al asistente.
    """
    try:
        from src.orchestrator.intent_classifier import classify_engineering
        from src.engineering.engineering_session import (
            SYSTEM_ENGINEERING, _load_rag, _build_rag_prompt,
        )
    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}", file=sys.stderr)
        return []

    rag_engine = _load_rag()
    results = []
    total = len(prompts) * iterations
    run = 0

    for p in prompts:
        text = p["text"]

        for _ in range(warmup):
            try:
                classify_engineering(text)
                if rag_engine: rag_engine.query_with_domain_check(text)
            except Exception: pass

        for i in range(iterations):
            run += 1
            if progress: progress(run, total, f"e2e/{p['id']} iter {i+1}")

            t_turn = time.perf_counter()

            # Fase 1: intent
            t1 = time.perf_counter()
            try:
                intent = classify_engineering(text)
                intent_action = intent.action
            except Exception as e:
                intent_action = "error"
                print(f"\n[WARN] intent failed: {e}")
            t_intent = time.perf_counter() - t1

            # Fase 2: RAG (solo si intent == respond)
            rag_time  = 0.0
            rag_fired = False
            messages = [
                {"role": "system", "content": SYSTEM_ENGINEERING},
                {"role": "user",   "content": text},
            ]
            if rag_engine and intent_action == "respond":
                t2 = time.perf_counter()
                try:
                    rag_result = rag_engine.query_with_domain_check(text)
                    rag_time = time.perf_counter() - t2
                    if rag_result.relevant:
                        rag_fired = True
                        messages = _build_rag_prompt(text, rag_result.context, [])
                except Exception: pass

            # Fase 3: LLM
            r = ollama_chat_stream(MODEL, messages, max_tokens=500)

            t_total = time.perf_counter() - t_turn

            results.append({
                "suite":         "e2e",
                "prompt_id":     p["id"],
                "iteration":     i + 1,
                "intent_action": intent_action,
                "intent_s":      round(t_intent, 4),
                "rag_s":         round(rag_time, 4),
                "rag_fired":     rag_fired,
                "ttft_s":        round(r.ttft_s, 4),
                "llm_total_s":   round(r.total_s, 4),
                "tokens":        r.eval_count,
                "tps":           round(r.tps_server, 2),
                "turn_total_s":  round(t_total, 4),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Suite 5: Thermal stress
# ─────────────────────────────────────────────────────────────────────────────

def suite_thermal(prompts, duration_s=600, progress=None):
    """
    Ejecuta inferencias en loop durante 'duration_s' segundos.

    Objetivo: detectar degradación de TPS por thermal throttling.
    Comparar TPS del primer cuartil vs último cuartil del run muestra
    cuánto baja el rendimiento cuando la Jetson se calienta.

    En la Orin Nano el throttling empieza ~85°C en tj. El suite NO tiene
    que forzar temperaturas altas — solo sostener carga y ver qué pasa
    con la temperatura que la jetson llega naturalmente.

    Usa varios prompts para evitar que el KV cache optimice el mismo query.
    """
    if not prompts:
        return []

    results = []
    t_start = time.monotonic()
    t_end   = t_start + duration_s
    iteration = 0
    prompt_idx = 0

    while time.monotonic() < t_end:
        iteration += 1
        p = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1

        messages = [
            {"role": "system", "content": SYSTEM_NEUTRAL},
            {"role": "user",   "content": p["text"]},
        ]

        elapsed = time.monotonic() - t_start
        if progress:
            progress(int(elapsed), duration_s,
                     f"iter {iteration} ({p['id']})")

        r = ollama_chat_stream(MODEL, messages, max_tokens=300)

        results.append({
            "suite":            "thermal",
            "iteration":        iteration,
            "t_since_start_s":  round(elapsed, 2),
            "prompt_id":        p["id"],
            "ttft_s":           round(r.ttft_s, 4),
            "total_s":          round(r.total_s, 4),
            "tokens":           r.eval_count,
            "tps":              round(r.tps_server, 2),
        })

    return results