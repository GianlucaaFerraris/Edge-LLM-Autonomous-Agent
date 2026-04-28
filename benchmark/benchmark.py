#!/usr/bin/env python3
"""
benchmark.py — entry point del suite de benchmarks Agenty.

Diseñado para correr en Jetson Orin Nano 8GB (JetPack 6.2.2+).
Texto puro — no usa voice_io, no abre mic/parlantes.

Uso:
    # Correr suite completo con preset standard (10 iter, thermal 10min)
    python benchmark.py

    # Presets:
    python benchmark.py --quick      # 5 iter, thermal 3min
    python benchmark.py --thorough   # 20 iter, thermal 20min

    # Solo algunas suites:
    python benchmark.py --suites llm,rag
    python benchmark.py --suites llm,intent,e2e --skip-thermal

    # Override manual:
    python benchmark.py --iterations 15 --thermal-s 600

Output:
    <este_dir>/results/<timestamp>_<preset>/
        raw_data.json      - todos los datos
        system_info.json   - snapshot del sistema
        llm.csv, intent.csv, rag.csv, e2e.csv, thermal.csv
        tegrastats.csv     - serie temporal del monitor
        report.md          - report con tablas listas para el informe

Asumiendo estructura de directorios:
    Agenty-Edge-Assistant/
    ├── src/                   (tu código)
    └── benchmarks/            (este directorio)
        ├── benchmark.py       (este archivo)
        ├── benchmark_lib.py
        ├── benchmark_suites.py
        ├── benchmark_reporter.py
        ├── prompts.json
        └── results/           (se crea solo)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                          # Agenty-Edge-Assistant/
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_lib import (
    TegrastatsMonitor, get_system_info, ollama_warmup,
)
from benchmark_suites import (
    suite_llm, suite_intent, suite_rag, suite_e2e, suite_thermal,
)
from benchmark_reporter import (
    save_raw, save_csv, build_markdown,
    agg_llm, agg_intent, agg_rag, agg_e2e, agg_thermal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "quick":    {"iterations":  5, "warmup": 1, "thermal_s":  180},
    "standard": {"iterations": 10, "warmup": 2, "thermal_s":  600},
    "thorough": {"iterations": 20, "warmup": 3, "thermal_s": 1200},
}


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def progress_fn(prefix):
    def _fn(cur, total, msg=""):
        if total and total > 0:
            bar = "=" * min(30, int(30 * cur / total))
            sys.stdout.write(f"\r  [{prefix:>6}] [{bar:<30}] {cur}/{total} {msg[:32]:<32}")
        else:
            sys.stdout.write(f"\r  [{prefix:>6}] {cur} {msg[:40]:<40}")
        sys.stdout.flush()
    return _fn


def hr(title):
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────

def preflight(model: str, needs_rag: bool, needs_intent: bool) -> bool:
    """Chequeo previo: Ollama vivo, modelo disponible, imports OK."""
    import requests
    hr("Preflight")
    ok = True

    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        print(f"  Ollama alive. Modelos registrados: {names}")
        if model not in names:
            print(f"  [FAIL] Modelo '{model}' no registrado. Registralo con:")
            print(f"         ollama create {model} -f Modelfile")
            ok = False
        else:
            print(f"  [OK]   Modelo '{model}' disponible")
    except Exception as e:
        print(f"  [FAIL] Ollama no responde: {e}")
        print(f"         sudo systemctl start ollama")
        return False

    if needs_intent:
        try:
            from src.orchestrator.intent_classifier import classify_idle  # noqa
            print("  [OK]   intent_classifier importable")
        except ImportError as e:
            print(f"  [WARN] intent_classifier no importable: {e}")
            print("         Suite 'intent' y 'e2e' serán omitidas")

    if needs_rag:
        try:
            from src.engineering.engineering_session import _load_rag
            rag = _load_rag()
            if rag:
                print("  [OK]   RAG engine cargado (index FAISS encontrado)")
            else:
                print("  [WARN] RAG engine devolvió None (index no encontrado)")
        except ImportError as e:
            print(f"  [WARN] engineering_session no importable: {e}")
            print("         Suite 'rag' será omitida")

    import shutil
    if shutil.which("tegrastats"):
        print("  [OK]   tegrastats disponible (thermal monitoring habilitado)")
    else:
        print("  [WARN] tegrastats no encontrado (thermal stats limitados)")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agenty benchmark suite (Jetson Orin Nano)")
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        default="standard",
                        help="Preset de config (default: standard)")
    parser.add_argument("--quick",    action="store_const", const="quick",
                        dest="preset")
    parser.add_argument("--thorough", action="store_const", const="thorough",
                        dest="preset")
    parser.add_argument("--iterations", type=int,
                        help="Override # iteraciones por prompt")
    parser.add_argument("--warmup",     type=int,
                        help="Override # warmup iterations")
    parser.add_argument("--thermal-s",  type=int,
                        help="Duración del thermal test en segundos")
    parser.add_argument("--suites", default="all",
                        help="llm,intent,rag,e2e,thermal (coma) o 'all'")
    parser.add_argument("--skip-thermal", action="store_true")
    parser.add_argument("--prompts",
                        default=os.path.join(_HERE, "prompts.json"))
    parser.add_argument("--output-dir",
                        default=os.path.join(_HERE, "results"))
    parser.add_argument("--model", default="asistente")
    parser.add_argument("--no-preflight", action="store_true")
    args = parser.parse_args()

    # Resolver config
    cfg = PRESETS[args.preset].copy()
    if args.iterations is not None: cfg["iterations"] = args.iterations
    if args.warmup     is not None: cfg["warmup"]     = args.warmup
    if args.thermal_s  is not None: cfg["thermal_s"]  = args.thermal_s

    suites = (set(s.strip() for s in args.suites.split(","))
              if args.suites != "all"
              else {"llm", "intent", "rag", "e2e", "thermal"})
    if args.skip_thermal:
        suites.discard("thermal")

    # Header
    hr("AGENTY BENCHMARK — Jetson Orin Nano 8GB")
    print(f"  Preset:     {args.preset}")
    print(f"  Iterations: {cfg['iterations']}  (warmup: {cfg['warmup']})")
    print(f"  Suites:     {sorted(suites)}")
    print(f"  Model:      {args.model}")
    if "thermal" in suites:
        print(f"  Thermal:    {cfg['thermal_s']} s ({cfg['thermal_s']/60:.1f} min)")

    # Preflight
    needs_rag    = "rag"    in suites or "e2e" in suites
    needs_intent = "intent" in suites or "e2e" in suites
    if not args.no_preflight:
        if not preflight(args.model, needs_rag, needs_intent):
            print("\n  Abortando. Corregí los issues arriba o corré con --no-preflight")
            return 1

    # Load prompts
    try:
        with open(args.prompts) as f:
            prompts = json.load(f)
    except Exception as e:
        print(f"  [ERROR] No se puede leer prompts: {e}")
        return 1

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.output_dir, f"{ts}_{args.preset}")
    os.makedirs(outdir, exist_ok=True)
    print(f"  Output:     {outdir}")

    # System info snapshot
    sys_info = get_system_info()
    with open(os.path.join(outdir, "system_info.json"), "w") as f:
        json.dump(sys_info, f, indent=2, default=str)

    # Warmup global
    hr("Warmup global — carga del modelo a RAM")
    t0 = time.perf_counter()
    ollama_warmup(args.model, n=2)
    print(f"  Modelo cargado en {time.perf_counter()-t0:.1f} s")

    results = {}
    t_start = time.perf_counter()

    # ── 1. LLM ──
    if "llm" in suites:
        hr("[1/5] Suite LLM — inferencia pura")
        pr = (prompts.get("general", [])
            + prompts.get("engineering_es", [])[:3]
            + prompts.get("engineering_en", [])[:2]
            + prompts.get("off_domain", []))
        t = time.perf_counter()
        rows = suite_llm(pr, cfg["iterations"], cfg["warmup"],
                         progress=progress_fn("llm"))
        print(f"\n  {len(rows)} iteraciones — {time.perf_counter()-t:.1f} s")
        results["llm"] = rows
        results["llm_agg"] = agg_llm(rows)
        save_csv("llm", rows, outdir)

    # ── 2. Intent ──
    if "intent" in suites:
        hr("[2/5] Suite Intent classifier")
        t = time.perf_counter()
        rows = suite_intent(
            prompts.get("intent_idle", []),
            prompts.get("intent_engineering", []),
            cfg["iterations"], cfg["warmup"],
            progress=progress_fn("intent"),
        )
        print(f"\n  {len(rows)} iteraciones — {time.perf_counter()-t:.1f} s")
        results["intent"] = rows
        results["intent_agg"] = agg_intent(rows)
        save_csv("intent", rows, outdir)

    # ── 3. RAG ──
    if "rag" in suites:
        hr("[3/5] Suite RAG vs no-RAG")
        pr = (prompts.get("engineering_es", [])
            + prompts.get("engineering_en", []))
        t = time.perf_counter()
        rows = suite_rag(pr, cfg["iterations"], cfg["warmup"],
                         progress=progress_fn("rag"))
        print(f"\n  {len(rows)} iteraciones — {time.perf_counter()-t:.1f} s")
        results["rag"] = rows
        results["rag_agg"] = agg_rag(rows)
        save_csv("rag", rows, outdir)

    # ── 4. E2E ──
    if "e2e" in suites:
        hr("[4/5] Suite End-to-end")
        pr = (prompts.get("engineering_es", [])[:3]
            + prompts.get("engineering_en", [])[:2])
        t = time.perf_counter()
        rows = suite_e2e(pr, cfg["iterations"], cfg["warmup"],
                         progress=progress_fn("e2e"))
        print(f"\n  {len(rows)} iteraciones — {time.perf_counter()-t:.1f} s")
        results["e2e"] = rows
        results["e2e_agg"] = agg_e2e(rows)
        save_csv("e2e", rows, outdir)

    # ── 5. Thermal (último, con tegrastats paralelo) ──
    tegra_summary = {"available": False}
    if "thermal" in suites:
        hr(f"[5/5] Suite Thermal stress ({cfg['thermal_s']} s)")
        mon = TegrastatsMonitor(interval_ms=1000)
        mon.start()
        pr = (prompts.get("engineering_es", [])
            + prompts.get("engineering_en", [])
            + prompts.get("general", []))
        t = time.perf_counter()
        rows = suite_thermal(pr, cfg["thermal_s"],
                              progress=progress_fn("therm"))
        elapsed = time.perf_counter() - t
        print(f"\n  {len(rows)} iteraciones — {elapsed:.1f} s "
              f"({elapsed/60:.1f} min)")
        mon.stop()

        tegra_summary = mon.summary()
        results["thermal"] = rows
        results["thermal_agg"] = agg_thermal(rows)
        save_csv("thermal", rows, outdir)

        raw = mon.raw_samples()
        if raw:
            save_csv("tegrastats", raw, outdir)

    # Persist consolidated
    results["system_info"]       = sys_info
    results["tegrastats_summary"] = tegra_summary
    results["config"]            = {"preset": args.preset, **cfg}
    save_raw(results, outdir)

    md_path = build_markdown(results, sys_info, tegra_summary, outdir)

    total = time.perf_counter() - t_start
    hr("DONE")
    print(f"  Tiempo total: {total/60:.1f} min")
    print(f"  Output dir:   {outdir}")
    print(f"  Report MD:    {md_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())