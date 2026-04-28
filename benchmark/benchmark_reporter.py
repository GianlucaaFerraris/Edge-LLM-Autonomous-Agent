"""
benchmark_reporter.py — generación de reports.

Toma los resultados crudos de las suites y produce:
  - raw_data.json:  todo el contenido sin agregación
  - <suite>.csv:    una tabla por suite, importable en pandas/Excel
  - tegrastats.csv: serie temporal del monitor
  - system_info.json: snapshot de la Jetson
  - report.md:      reporte legible con tablas listas para el informe

Las funciones agg_* agregan estadísticos por suite siguiendo las
métricas que típicamente se reportan en benchmarking de LLMs:
mean, median, p95 (tail), std, min, max.
"""

import csv
import json
import os

from benchmark_lib import summarize


# ─────────────────────────────────────────────────────────────────────────────
# Persistencia
# ─────────────────────────────────────────────────────────────────────────────

def save_raw(results: dict, outdir: str):
    """Dump completo del objeto results a JSON para auditoría."""
    path = os.path.join(outdir, "raw_data.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def save_csv(suite_name: str, rows: list, outdir: str):
    """Tabla CSV con todas las iteraciones de una suite."""
    if not rows:
        return None
    path = os.path.join(outdir, f"{suite_name}.csv")
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Agregación
# ─────────────────────────────────────────────────────────────────────────────

def _stats(rows, field):
    """Summarize de un campo numérico ignorando nulls/strings vacíos."""
    vals = []
    for r in rows:
        v = r.get(field)
        if v is None: continue
        if isinstance(v, bool): continue
        if isinstance(v, str) and not v: continue
        if isinstance(v, (int, float)):
            vals.append(v)
    return summarize(vals)


def agg_llm(rows):
    if not rows: return {}
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    return {
        "global": {
            "n":              len(rows),
            "ttft_s":         _stats(rows, "ttft_s"),
            "total_s":        _stats(rows, "total_s"),
            "tokens":         _stats(rows, "tokens"),
            "tps_server":     _stats(rows, "tps_server"),
            "tps_client":     _stats(rows, "tps_client"),
            "prompt_eval_ms": _stats(rows, "prompt_eval_ms"),
        },
        "by_prompt": {
            pid: {
                "n":           len(pr),
                "ttft_s":      _stats(pr, "ttft_s"),
                "total_s":     _stats(pr, "total_s"),
                "tps_server":  _stats(pr, "tps_server"),
                "tokens":      _stats(pr, "tokens"),
            } for pid, pr in by_prompt.items()
        },
    }


def agg_intent(rows):
    if not rows: return {}
    correct = sum(1 for r in rows if r.get("correct"))
    modes = sorted(set(r["mode"] for r in rows))
    by_mode = {}
    for m in modes:
        mr = [r for r in rows if r["mode"] == m]
        mc = sum(1 for r in mr if r.get("correct"))
        by_mode[m] = {
            "n":          len(mr),
            "latency_s":  _stats(mr, "latency_s"),
            "accuracy":   round(100 * mc / max(1, len(mr)), 1),
        }
    return {
        "global": {
            "n":          len(rows),
            "latency_s":  _stats(rows, "latency_s"),
            "accuracy":   round(100 * correct / len(rows), 1),
        },
        "by_mode": by_mode,
    }


def agg_rag(rows):
    if not rows: return {}
    fired = [r for r in rows if r.get("rag_triggered")]
    return {
        "global": {
            "n":                   len(rows),
            "rag_fired_pct":       round(100 * len(fired) / len(rows), 1),
            "rag_overhead_s":      _stats(rows, "rag_overhead_s"),
            "noRAG_ttft_s":        _stats(rows, "noRAG_ttft_s"),
            "noRAG_total_s":       _stats(rows, "noRAG_total_s"),
            "noRAG_tps":           _stats(rows, "noRAG_tps"),
            "noRAG_prompt_tokens": _stats(rows, "noRAG_prompt_tokens"),
            "RAG_ttft_s":          _stats(rows, "RAG_ttft_s"),
            "RAG_total_s":         _stats(rows, "RAG_total_s"),
            "RAG_tps":             _stats(rows, "RAG_tps"),
            "RAG_prompt_tokens":   _stats(rows, "RAG_prompt_tokens"),
            "prompt_inflation":    _stats(rows, "prompt_inflation"),
            "total_overhead_s":    _stats(rows, "total_overhead_s"),
            "rag_top_score":       _stats(rows, "rag_top_score"),
        }
    }


def agg_e2e(rows):
    if not rows: return {}
    return {
        "global": {
            "n":              len(rows),
            "intent_s":       _stats(rows, "intent_s"),
            "rag_s":          _stats(rows, "rag_s"),
            "ttft_s":         _stats(rows, "ttft_s"),
            "llm_total_s":    _stats(rows, "llm_total_s"),
            "turn_total_s":   _stats(rows, "turn_total_s"),
            "tps":            _stats(rows, "tps"),
            "rag_fired_pct":  round(100*sum(1 for r in rows
                                             if r.get("rag_fired"))/len(rows), 1),
        }
    }


def agg_thermal(rows):
    """Agregación temporal: compara primer cuartil vs último para ver caída."""
    if not rows: return {}
    n = len(rows)
    q1_size = max(1, n // 4)
    q1 = rows[:q1_size]
    q4 = rows[-q1_size:]

    tps_q1  = _stats(q1, "tps")
    tps_q4  = _stats(q4, "tps")
    ttft_q1 = _stats(q1, "ttft_s")
    ttft_q4 = _stats(q4, "ttft_s")

    tps_drop_pct = 0.0
    if tps_q1["mean"] > 0:
        tps_drop_pct = 100 * (tps_q1["mean"] - tps_q4["mean"]) / tps_q1["mean"]

    return {
        "global": {
            "n":          n,
            "duration_s": rows[-1]["t_since_start_s"] if rows else 0,
            "ttft_s":     _stats(rows, "ttft_s"),
            "total_s":    _stats(rows, "total_s"),
            "tps":        _stats(rows, "tps"),
        },
        "degradation": {
            "first_quarter_tps_mean":  round(tps_q1["mean"], 2),
            "last_quarter_tps_mean":   round(tps_q4["mean"], 2),
            "tps_drop_pct":            round(tps_drop_pct, 1),
            "first_quarter_ttft_mean": round(ttft_q1["mean"], 4),
            "last_quarter_ttft_mean":  round(ttft_q4["mean"], 4),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(s, unit="", prec=3):
    """'mean=X.XX (median=..., p95=..., std=..., n=N)' para un summarize()."""
    if not s or s.get("n", 0) == 0:
        return "—"
    return (f"mean={s['mean']:.{prec}f}{unit}, "
            f"median={s['median']:.{prec}f}, "
            f"p95={s['p95']:.{prec}f}, "
            f"std={s['std']:.{prec}f}, "
            f"n={s['n']}")


def _tbl(headers, rows):
    out  = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out + "\n"


def build_markdown(results, system_info, tegrastats_summary, outdir):
    md = []
    md.append("# Agenty — Benchmark Report (Jetson Orin Nano 8GB)\n")
    md.append(f"*Generated: {system_info.get('timestamp', '-')}*")
    cfg = results.get("config", {})
    md.append(f"*Preset: **{cfg.get('preset', '-')}**, "
              f"iterations: **{cfg.get('iterations', '-')}**, "
              f"warmup: **{cfg.get('warmup', '-')}**, "
              f"thermal: **{cfg.get('thermal_s', '-')} s***\n")

    # 1. System
    md.append("## 1. Sistema\n")
    md.append(f"- **L4T:** `{system_info.get('l4t', '-')}`")
    md.append(f"- **Power mode:** `{system_info.get('power_mode', '-')}`")
    ov = (system_info.get('ollama_version', '') or '').splitlines()
    md.append(f"- **Ollama:** `{ov[0] if ov else '-'}`")
    md.append("")
    md.append("**Memory:**")
    md.append("```")
    md.extend(system_info.get("memory", []))
    md.append("```\n")

    # 2. LLM
    agg = results.get("llm_agg", {})
    if agg.get("global"):
        g = agg["global"]
        md.append("## 2. LLM puro — inferencia sin intent ni RAG\n")
        md.append("Mediciones del modelo base (Qwen 2.5 7B Q4_K_M) sin "
                  "ningún pipeline de Agenty por encima. Baseline del hardware.\n")
        md.append(f"- **TTFT (Time To First Token):** {_fmt(g['ttft_s'], ' s', 3)}")
        md.append(f"- **Tiempo total de generación:** {_fmt(g['total_s'], ' s', 3)}")
        md.append(f"- **Tokens generados:** {_fmt(g['tokens'], ' tok', 1)}")
        md.append(f"- **TPS (server, Ollama):** {_fmt(g['tps_server'], ' tok/s', 2)}")
        md.append(f"- **TPS (client, wall-clock):** {_fmt(g['tps_client'], ' tok/s', 2)}")
        md.append(f"- **Prompt eval (prefill):** {_fmt(g['prompt_eval_ms'], ' ms', 1)}\n")

        md.append("### Por prompt\n")
        rows = []
        for pid, st in agg.get("by_prompt", {}).items():
            rows.append([
                pid,
                f"{st['ttft_s']['mean']:.3f} ± {st['ttft_s']['std']:.3f}",
                f"{st['total_s']['mean']:.2f}",
                f"{st['tokens']['mean']:.0f}",
                f"{st['tps_server']['mean']:.1f}",
            ])
        md.append(_tbl(["prompt_id", "TTFT (s)", "Total (s)", "Tokens", "TPS"], rows))

    # 3. Intent
    agg = results.get("intent_agg", {})
    if agg.get("global"):
        g = agg["global"]
        md.append("## 3. Intent classifier\n")
        md.append("Latencia que agrega el clasificador de intents a cada turno "
                  "del usuario (antes de invocar el LLM principal).\n")
        md.append(f"- **Latencia global:** {_fmt(g['latency_s'], ' s', 3)}")
        md.append(f"- **Accuracy global:** {g['accuracy']}%\n")

        rows = []
        for m, st in agg.get("by_mode", {}).items():
            rows.append([
                m,
                f"{st['latency_s']['mean']:.3f}",
                f"{st['latency_s']['median']:.3f}",
                f"{st['latency_s']['p95']:.3f}",
                f"{st['accuracy']}%",
                st['n'],
            ])
        md.append(_tbl(
            ["mode", "mean (s)", "median (s)", "p95 (s)", "accuracy", "n"],
            rows))

    # 4. RAG
    agg = results.get("rag_agg", {})
    if agg.get("global"):
        g = agg["global"]
        md.append("## 4. RAG vs no-RAG\n")
        md.append("A/B directo con el mismo prompt: primero sin RAG (solo "
                  "SYSTEM_ENGINEERING), después con RAG (chunks FAISS "
                  "inyectados al prompt).\n")
        md.append(f"- **RAG disparó:** {g['rag_fired_pct']}% de queries (resto filtradas por thresholds de dominio/relevancia)")
        md.append(f"- **Score de relevancia medio (cuando disparó):** {g['rag_top_score']['mean']:.3f}")
        md.append(f"- **Overhead RAG (embed + FAISS):** {_fmt(g['rag_overhead_s'], ' s', 4)}")
        md.append(f"- **Prompt inflation:** {_fmt(g['prompt_inflation'], ' tokens', 1)}")
        md.append(f"- **Overhead total por turno (vs no-RAG):** {_fmt(g['total_overhead_s'], ' s', 3)}\n")

        md.append("### Comparación directa\n")
        rows = [
            ["TTFT (s)",        f"{g['noRAG_ttft_s']['mean']:.3f}",       f"{g['RAG_ttft_s']['mean']:.3f}"],
            ["Total (s)",       f"{g['noRAG_total_s']['mean']:.2f}",      f"{g['RAG_total_s']['mean']:.2f}"],
            ["TPS",             f"{g['noRAG_tps']['mean']:.1f}",          f"{g['RAG_tps']['mean']:.1f}"],
            ["Prompt tokens",   f"{g['noRAG_prompt_tokens']['mean']:.0f}", f"{g['RAG_prompt_tokens']['mean']:.0f}"],
        ]
        md.append(_tbl(["Métrica", "sin RAG", "con RAG"], rows))

    # 5. E2E
    agg = results.get("e2e_agg", {})
    if agg.get("global"):
        g = agg["global"]
        md.append("## 5. End-to-end (turno completo)\n")
        md.append("Simulación del flujo real: `classify_engineering` → "
                  "(si respond) `rag_engine.query` → `ollama_chat_stream`.\n")
        md.append(f"- **Intent classification:** {_fmt(g['intent_s'], ' s', 3)}")
        md.append(f"- **RAG lookup:** {_fmt(g['rag_s'], ' s', 4)}")
        md.append(f"- **LLM TTFT:** {_fmt(g['ttft_s'], ' s', 3)}")
        md.append(f"- **LLM total:** {_fmt(g['llm_total_s'], ' s', 2)}")
        md.append(f"- **Turn total (lo que siente el usuario):** {_fmt(g['turn_total_s'], ' s', 2)}")
        md.append(f"- **RAG fired:** {g['rag_fired_pct']}%\n")

    # 6. Thermal
    agg = results.get("thermal_agg", {})
    if agg.get("global"):
        g = agg["global"]
        d = agg.get("degradation", {})
        md.append("## 6. Thermal stress test\n")
        md.append(f"Carga sostenida durante {g['duration_s']:.0f} segundos "
                  f"({g['n']} iteraciones). Compara TPS del primer cuartil "
                  "(Jetson fría) contra el último (caliente) para detectar "
                  "degradación por thermal throttling.\n")

        md.append(f"- **TPS 1er cuartil (fría):** {d.get('first_quarter_tps_mean', '-')} tok/s")
        md.append(f"- **TPS 4º cuartil (caliente):** {d.get('last_quarter_tps_mean', '-')} tok/s")
        md.append(f"- **Caída de TPS:** {d.get('tps_drop_pct', 0)}%")
        md.append(f"- **TTFT 1er cuartil:** {d.get('first_quarter_ttft_mean', 0):.3f} s")
        md.append(f"- **TTFT 4º cuartil:** {d.get('last_quarter_ttft_mean', 0):.3f} s\n")

    # 7. Tegrastats
    if tegrastats_summary and tegrastats_summary.get("available"):
        t = tegrastats_summary
        md.append("## 7. Tegrastats (durante thermal test)\n")
        md.append(f"- **Duración monitoreada:** {t['duration_s']:.0f} s "
                  f"({t['n_samples']} samples @ 1 Hz)")
        md.append(f"- **Throttle events (tj ≥ 85 °C sostenido ≥ 2 s):** "
                  f"{t['throttle_events']}")
        md.append(f"- **% tiempo con tj ≥ 85 °C:** "
                  f"{t['tj_above_85_pct']:.1f}%\n")

        rows = [
            ["RAM used (MB)",        f"{t['ram_used_mb']['mean']:.0f}",    f"{t['ram_used_mb']['max']}"],
            ["Swap used (MB)",       f"{t['swap_used_mb']['mean']:.0f}",   f"{t['swap_used_mb']['max']}"],
            ["CPU temp (°C)",        f"{t['cpu_temp_c']['mean']:.1f}",     f"{t['cpu_temp_c']['max']:.1f}"],
            ["GPU temp (°C)",        f"{t['gpu_temp_c']['mean']:.1f}",     f"{t['gpu_temp_c']['max']:.1f}"],
            ["TJ temp (°C)",         f"{t['tj_temp_c']['mean']:.1f}",      f"{t['tj_temp_c']['max']:.1f}"],
            ["GPU utilization %",    f"{t['gr3d_pct']['mean']:.1f}",       f"{t['gr3d_pct']['max']}"],
            ["GPU+SOC power (mW)",   f"{t['vdd_gpu_soc_mw']['mean']:.0f}", f"{t['vdd_gpu_soc_mw']['max']}"],
            ["CPU power (mW)",       f"{t['vdd_cpu_cv_mw']['mean']:.0f}",  f"{t['vdd_cpu_cv_mw']['max']}"],
            ["VIN total 5V (mW)",    f"{t['vin_total_mw']['mean']:.0f}",   f"{t['vin_total_mw']['max']}"],
        ]
        md.append(_tbl(["Métrica", "mean", "max"], rows))

    # Footer
    md.append("---\n")
    md.append("*Datos crudos en `raw_data.json`, CSVs por suite y "
              "`tegrastats.csv` con serie temporal completa en el mismo directorio.*")

    path = os.path.join(outdir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(md))
    return path