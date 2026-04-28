#!/usr/bin/env python3
"""
plots.py — Post-procesamiento: genera gráficos del informe desde los CSVs.

Lee los CSVs producidos por benchmark.py y genera PNGs para el informe.
Dependencias: pandas, numpy, matplotlib. NO seaborn (evitar dependencia pesada).

Los gráficos están agrupados por categoría de análisis:
  - Thermal/power (necesitan tegrastats.csv)
  - Latency budget (E2E breakdown)
  - RAG cost-benefit (RAG vs noRAG)
  - Distribuciones (TTFT, TPS) — informan reliability
  - Bilingual performance (ES vs EN) — segmenta llm.csv por lang
  - Cold vs warm — primera iter vs resto
  - Prefill vs decode — descompone TTFT vs decode time

Uso:
    # Gráfica el resultado más reciente
    python plots.py --latest

    # Gráfica un resultado específico
    python plots.py --dir results/20251028_143022_standard

    # Lista resultados disponibles
    python plots.py --list

Output: results/<timestamp>_<preset>/plots/*.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ─────────────────────────────────────────────────────────────────────────────
# Style — custom rcParams en lugar de seaborn (sin dependencias extras)
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.figsize":   (10, 5.5),
    "figure.dpi":       110,
    "savefig.dpi":      130,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "legend.framealpha": 0.92,
})

# Paleta consistente — colores asignados por concepto, no por orden
COLOR = {
    "tps":       "#1f77b4",   # azul
    "ttft":      "#ff7f0e",   # naranja
    "decode":    "#2ca02c",   # verde
    "intent":    "#9467bd",   # púrpura
    "rag":       "#8c564b",   # marrón
    "tj":        "#d62728",   # rojo (thermal junction = lo crítico)
    "gpu_temp":  "#e377c2",   # rosa
    "cpu_temp":  "#ff9896",   # rojo claro
    "ram":       "#17becf",   # cyan
    "swap":      "#bcbd22",   # amarillo verdoso
    "power_gpu": "#1f77b4",
    "power_cpu": "#ff7f0e",
    "power_tot": "#2ca02c",
    "no_rag":    "#7f7f7f",
    "with_rag":  "#1f77b4",
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV loaders — robustos a archivos faltantes
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    """Carga CSV o retorna None si no existe / está vacío."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"  [WARN] No se pudo leer {path}: {e}")
        return None


def load_all(result_dir):
    return {
        "llm":     load_csv(os.path.join(result_dir, "llm.csv")),
        "intent":  load_csv(os.path.join(result_dir, "intent.csv")),
        "rag":     load_csv(os.path.join(result_dir, "rag.csv")),
        "e2e":     load_csv(os.path.join(result_dir, "e2e.csv")),
        "thermal": load_csv(os.path.join(result_dir, "thermal.csv")),
        "tegra":   load_csv(os.path.join(result_dir, "tegrastats.csv")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. THERMAL & POWER — el core del informe edge
# ─────────────────────────────────────────────────────────────────────────────

def plot_thermal_curve(thermal_df, tegra_df, outdir):
    """
    Plot bandera del informe: TPS (eje izq) + tj temp (eje der) sobre el tiempo.
    Muestra correlación visual entre throttling y caída de TPS.
    """
    if thermal_df is None or len(thermal_df) < 2:
        return None

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    ax1.scatter(thermal_df["t_since_start_s"], thermal_df["tps"],
                color=COLOR["tps"], alpha=0.5, s=18, label="TPS")
    if len(thermal_df) >= 6:
        win = max(3, len(thermal_df) // 12)
        rolling = thermal_df["tps"].rolling(window=win, center=True).mean()
        ax1.plot(thermal_df["t_since_start_s"], rolling,
                 color=COLOR["tps"], lw=2, label=f"TPS rolling avg (n={win})")

    ax1.set_xlabel("Tiempo desde inicio (s)")
    ax1.set_ylabel("Tokens/s (server)", color=COLOR["tps"])
    ax1.tick_params(axis="y", labelcolor=COLOR["tps"])
    ax1.grid(True, alpha=0.3)

    if tegra_df is not None and "tj_temp" in tegra_df.columns:
        tj_data = tegra_df[tegra_df["tj_temp"] > 0]
        if len(tj_data) > 0:
            ax2.plot(tj_data["t"], tj_data["tj_temp"],
                     color=COLOR["tj"], lw=1.4, alpha=0.8, label="tj@ (°C)")
            ax2.axhline(85, color=COLOR["tj"], ls=":", alpha=0.5,
                        label="Throttle threshold (85 °C)")

    ax2.set_ylabel("Temperatura tj (°C)", color=COLOR["tj"])
    ax2.tick_params(axis="y", labelcolor=COLOR["tj"])
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Thermal stress — TPS vs temperatura junction")
    out = os.path.join(outdir, "01_thermal_curve.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_temperature_profile(tegra_df, outdir):
    """Tres temperaturas (CPU, GPU, TJ) sobre el tiempo del thermal test."""
    if tegra_df is None or len(tegra_df) < 2:
        return None

    fig, ax = plt.subplots()
    if "cpu_temp" in tegra_df.columns:
        ax.plot(tegra_df["t"], tegra_df["cpu_temp"], color=COLOR["cpu_temp"],
                lw=1.3, label="CPU")
    if "gpu_temp" in tegra_df.columns:
        ax.plot(tegra_df["t"], tegra_df["gpu_temp"], color=COLOR["gpu_temp"],
                lw=1.3, label="GPU")
    if "tj_temp" in tegra_df.columns:
        ax.plot(tegra_df["t"], tegra_df["tj_temp"], color=COLOR["tj"],
                lw=1.6, label="TJ (junction)")

    ax.axhline(85, color=COLOR["tj"], ls=":", alpha=0.5,
               label="Throttle (85 °C)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title("Perfil térmico durante carga sostenida")
    ax.legend()

    out = os.path.join(outdir, "02_temperature_profile.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_power_profile(tegra_df, outdir):
    """Power consumption por rail durante la carga."""
    if tegra_df is None or len(tegra_df) < 2:
        return None

    fig, ax = plt.subplots()
    if "vdd_gpu_soc" in tegra_df.columns:
        ax.plot(tegra_df["t"], tegra_df["vdd_gpu_soc"]/1000.0,
                color=COLOR["power_gpu"], lw=1.3, label="VDD_GPU_SOC (W)")
    if "vdd_cpu_cv" in tegra_df.columns:
        ax.plot(tegra_df["t"], tegra_df["vdd_cpu_cv"]/1000.0,
                color=COLOR["power_cpu"], lw=1.3, label="VDD_CPU_CV (W)")
    if "vin_total" in tegra_df.columns:
        ax.plot(tegra_df["t"], tegra_df["vin_total"]/1000.0,
                color=COLOR["power_tot"], lw=1.6, label="VIN_SYS_5V0 (total, W)")

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Potencia (W)")
    ax.set_title("Consumo de potencia durante carga sostenida")
    ax.legend()

    out = os.path.join(outdir, "03_power_profile.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_memory_profile(tegra_df, outdir):
    """RAM + swap usage. Muestra headroom respecto a los 8GB totales."""
    if tegra_df is None or len(tegra_df) < 2:
        return None
    if "ram_used" not in tegra_df.columns:
        return None

    fig, ax = plt.subplots()
    ram_total_gb = tegra_df["ram_total"].max() / 1024.0

    ax.fill_between(tegra_df["t"], 0, tegra_df["ram_used"]/1024.0,
                    color=COLOR["ram"], alpha=0.55, label="RAM usado (GB)")
    if "swap_used" in tegra_df.columns and tegra_df["swap_used"].max() > 0:
        ax.fill_between(tegra_df["t"],
                        tegra_df["ram_used"]/1024.0,
                        (tegra_df["ram_used"] + tegra_df["swap_used"])/1024.0,
                        color=COLOR["swap"], alpha=0.45, label="Swap usado (GB)")

    ax.axhline(ram_total_gb, color="black", ls="--", alpha=0.6,
               label=f"RAM total ({ram_total_gb:.1f} GB)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Memoria (GB)")
    ax.set_title("Uso de memoria durante carga sostenida")
    ax.legend(loc="lower right")
    ax.set_ylim(0, ram_total_gb * 1.05)

    out = os.path.join(outdir, "04_memory_profile.png")
    plt.savefig(out)
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. LATENCY BUDGET (E2E breakdown)
# ─────────────────────────────────────────────────────────────────────────────

def plot_e2e_breakdown(e2e_df, outdir):
    """
    Stacked bar por prompt_id mostrando dónde se va el tiempo en un turno:
      intent + RAG + TTFT + decode = turn_total

    Es el chart más útil para el informe porque muestra el bottleneck.
    """
    if e2e_df is None or len(e2e_df) == 0:
        return None

    grp = e2e_df.groupby("prompt_id").agg({
        "intent_s":     "mean",
        "rag_s":        "mean",
        "ttft_s":       "mean",
        "llm_total_s":  "mean",
        "turn_total_s": "mean",
    }).reset_index()
    grp["decode_s"] = (grp["llm_total_s"] - grp["ttft_s"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(grp))

    bars_intent = ax.bar(x, grp["intent_s"],
                         color=COLOR["intent"], label="Intent classifier")
    bars_rag    = ax.bar(x, grp["rag_s"], bottom=grp["intent_s"],
                         color=COLOR["rag"], label="RAG (embed + FAISS)")
    bottom_ttft = grp["intent_s"] + grp["rag_s"]
    bars_ttft   = ax.bar(x, grp["ttft_s"], bottom=bottom_ttft,
                         color=COLOR["ttft"], label="LLM prefill (TTFT)")
    bottom_dec  = bottom_ttft + grp["ttft_s"]
    bars_dec    = ax.bar(x, grp["decode_s"], bottom=bottom_dec,
                         color=COLOR["decode"], label="LLM decode")

    for i, total in enumerate(grp["turn_total_s"]):
        ax.text(i, total * 1.015, f"{total:.1f}s",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(grp["prompt_id"], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Tiempo (s)")
    ax.set_title("Latency budget — descomposición de un turno end-to-end")
    ax.legend(loc="upper left")

    out = os.path.join(outdir, "05_e2e_breakdown.png")
    plt.savefig(out)
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAG cost-benefit
# ─────────────────────────────────────────────────────────────────────────────

def plot_rag_comparison(rag_df, outdir):
    """
    Grouped bar: 4 métricas × 2 condiciones (RAG vs noRAG).
    Permite ver de un vistazo el costo del RAG en cada dimensión.
    """
    if rag_df is None or len(rag_df) == 0:
        return None

    metrics = {
        "TTFT (s)":         ("noRAG_ttft_s",    "RAG_ttft_s"),
        "Total (s)":        ("noRAG_total_s",   "RAG_total_s"),
        "Tokens out":       ("noRAG_tokens",    "RAG_tokens"),
        "Prompt tokens":    ("noRAG_prompt_tokens", "RAG_prompt_tokens"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, (name, (col_no, col_yes)) in zip(axes, metrics.items()):
        no_mean  = rag_df[col_no].mean()
        no_std   = rag_df[col_no].std()
        yes_mean = rag_df[col_yes].mean()
        yes_std  = rag_df[col_yes].std()

        ax.bar([0, 1], [no_mean, yes_mean],
               yerr=[no_std, yes_std],
               color=[COLOR["no_rag"], COLOR["with_rag"]],
               capsize=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["sin RAG", "con RAG"], fontsize=9)
        ax.set_title(name)
        ax.grid(True, alpha=0.3, axis="y")

        for i, v in enumerate([no_mean, yes_mean]):
            ax.text(i, v, f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("RAG vs no-RAG — comparación A/B (mean ± std)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(outdir, "06_rag_comparison.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_rag_overhead_breakdown(rag_df, outdir):
    """Histograma del overhead total por turno que agrega el RAG."""
    if rag_df is None or len(rag_df) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Izq: overhead RAG puro (embed + FAISS)
    ax = axes[0]
    ax.hist(rag_df["rag_overhead_s"], bins=20,
            color=COLOR["rag"], alpha=0.75, edgecolor="black")
    mean = rag_df["rag_overhead_s"].mean()
    ax.axvline(mean, color="red", ls="--", lw=1.5,
               label=f"mean = {mean*1000:.0f} ms")
    ax.set_xlabel("Overhead embed + FAISS (s)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Overhead RAG puro (antes del LLM)")
    ax.legend()

    # Der: prompt inflation (tokens extra que mete el RAG)
    ax = axes[1]
    ax.hist(rag_df["prompt_inflation"], bins=20,
            color=COLOR["with_rag"], alpha=0.75, edgecolor="black")
    mean = rag_df["prompt_inflation"].mean()
    ax.axvline(mean, color="red", ls="--", lw=1.5,
               label=f"mean = {mean:.0f} tokens")
    ax.set_xlabel("Tokens extra en prompt")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Prompt inflation por contexto RAG")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(outdir, "07_rag_overhead_breakdown.png")
    plt.savefig(out)
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. DISTRIBUCIONES — informan reliability / variance
# ─────────────────────────────────────────────────────────────────────────────

def plot_ttft_distribution(llm_df, outdir):
    """Histograma + box plot del TTFT — captura la variabilidad."""
    if llm_df is None or len(llm_df) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    ax.hist(llm_df["ttft_s"], bins=20,
            color=COLOR["ttft"], alpha=0.75, edgecolor="black")
    p50 = llm_df["ttft_s"].median()
    p95 = llm_df["ttft_s"].quantile(0.95)
    ax.axvline(p50, color="green", ls="--", lw=1.5, label=f"p50 = {p50:.3f}s")
    ax.axvline(p95, color="red",   ls="--", lw=1.5, label=f"p95 = {p95:.3f}s")
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de TTFT — Time To First Token")
    ax.legend()

    ax = axes[1]
    bp = ax.boxplot(llm_df["ttft_s"], vert=True, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=COLOR["ttft"], alpha=0.6))
    ax.set_xticklabels(["TTFT"])
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Box plot")

    plt.tight_layout()
    out = os.path.join(outdir, "08_ttft_distribution.png")
    plt.savefig(out)
    plt.close()
    return out


def plot_tps_distribution(llm_df, outdir):
    """Histograma + box plot del TPS."""
    if llm_df is None or len(llm_df) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    ax.hist(llm_df["tps_server"], bins=20,
            color=COLOR["tps"], alpha=0.75, edgecolor="black")
    mean = llm_df["tps_server"].mean()
    p5   = llm_df["tps_server"].quantile(0.05)
    ax.axvline(mean, color="red", ls="--", lw=1.5, label=f"mean = {mean:.1f}")
    ax.axvline(p5,   color="orange", ls="--", lw=1.5, label=f"p5 = {p5:.1f}")
    ax.set_xlabel("Tokens/s (server)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de TPS")
    ax.legend()

    ax = axes[1]
    ax.boxplot(llm_df["tps_server"], vert=True, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=COLOR["tps"], alpha=0.6))
    ax.set_xticklabels(["TPS"])
    ax.set_ylabel("Tokens/s")
    ax.set_title("Box plot")

    plt.tight_layout()
    out = os.path.join(outdir, "09_tps_distribution.png")
    plt.savefig(out)
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. BILINGUAL — segmentación ES vs EN
# ─────────────────────────────────────────────────────────────────────────────

def plot_bilingual_comparison(llm_df, outdir):
    """
    Si hay prompts EN y ES, comparar TPS y TTFT por idioma.
    Valida que el tokenizer no penalice un idioma sobre el otro.
    """
    if llm_df is None or "prompt_lang" not in llm_df.columns:
        return None
    langs = llm_df["prompt_lang"].dropna().unique()
    if len(langs) < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # TPS por idioma
    ax = axes[0]
    data = [llm_df[llm_df["prompt_lang"] == l]["tps_server"].values for l in langs]
    bp = ax.boxplot(data, labels=langs, patch_artist=True)
    for patch, lang in zip(bp["boxes"], langs):
        patch.set_facecolor(COLOR["tps"] if lang == "es" else COLOR["ttft"])
        patch.set_alpha(0.6)
    ax.set_ylabel("TPS (server)")
    ax.set_title("TPS por idioma del prompt")

    # TTFT por idioma
    ax = axes[1]
    data = [llm_df[llm_df["prompt_lang"] == l]["ttft_s"].values for l in langs]
    bp = ax.boxplot(data, labels=langs, patch_artist=True)
    for patch, lang in zip(bp["boxes"], langs):
        patch.set_facecolor(COLOR["tps"] if lang == "es" else COLOR["ttft"])
        patch.set_alpha(0.6)
    ax.set_ylabel("TTFT (s)")
    ax.set_title("TTFT por idioma del prompt")

    fig.suptitle("Performance bilingüe — ES vs EN", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(outdir, "10_bilingual_comparison.png")
    plt.savefig(out)
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREFILL vs DECODE — descompone el costo del LLM
# ─────────────────────────────────────────────────────────────────────────────

def plot_prefill_vs_decode(llm_df, outdir):
    """
    Scatter: tiempo de prefill (eje X) vs tiempo de decode (eje Y).
    Si el cluster cae en la diagonal, prefill y decode pesan parecido.
    Si cae arriba: decode-bound (lo normal con responses largas).
    Si cae abajo: prefill-bound (responses cortas o prompts gigantes).
    """
    if llm_df is None or "prompt_eval_ms" not in llm_df.columns:
        return None

    prefill_s = llm_df["prompt_eval_ms"] / 1000.0
    decode_s  = llm_df["total_s"] - llm_df["ttft_s"]

    fig, ax = plt.subplots()
    sc = ax.scatter(prefill_s, decode_s,
                    c=llm_df["tokens"], cmap="viridis",
                    s=45, alpha=0.75, edgecolor="black", linewidth=0.4)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Tokens generados")

    # Diagonal y=x para referencia
    lim = max(prefill_s.max(), decode_s.max()) * 1.1
    ax.plot([0, lim], [0, lim], ls=":", color="gray", alpha=0.6, label="y = x")

    ax.set_xlabel("Tiempo de prefill (s)")
    ax.set_ylabel("Tiempo de decode (s)")
    ax.set_title("Prefill vs decode — descomposición del costo LLM")
    ax.legend()

    out = os.path.join(outdir, "11_prefill_vs_decode.png")
    plt.savefig(out)
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTENT classifier
# ─────────────────────────────────────────────────────────────────────────────

def plot_intent_latency(intent_df, outdir):
    """Box plot de latencia por modo del intent classifier."""
    if intent_df is None or len(intent_df) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Latencia por modo
    ax = axes[0]
    modes = sorted(intent_df["mode"].unique())
    data = [intent_df[intent_df["mode"] == m]["latency_s"].values for m in modes]
    bp = ax.boxplot(data, labels=modes, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR["intent"])
        patch.set_alpha(0.65)
    ax.set_ylabel("Latencia (s)")
    ax.set_title("Latencia del intent classifier por modo")

    # Accuracy por modo
    ax = axes[1]
    acc = []
    for m in modes:
        sub = intent_df[intent_df["mode"] == m]
        acc.append(100 * sub["correct"].sum() / len(sub) if len(sub) > 0 else 0)
    bars = ax.bar(modes, acc, color=COLOR["decode"], alpha=0.75)
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Accuracy del intent classifier por modo")

    plt.tight_layout()
    out = os.path.join(outdir, "12_intent_latency.png")
    plt.savefig(out)
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Listado de resultados
# ─────────────────────────────────────────────────────────────────────────────

def list_results(base):
    if not os.path.isdir(base):
        print(f"  No existe el directorio: {base}")
        return
    dirs = sorted([d for d in os.listdir(base)
                   if os.path.isdir(os.path.join(base, d))])
    if not dirs:
        print(f"  No hay resultados en {base}")
        return
    print(f"\n  Resultados en {base}:")
    for d in dirs:
        full = os.path.join(base, d)
        files = [f for f in os.listdir(full) if f.endswith(".csv")]
        print(f"    {d}    ({len(files)} CSVs)")


def find_latest(base):
    if not os.path.isdir(base):
        return None
    dirs = sorted([d for d in os.listdir(base)
                   if os.path.isdir(os.path.join(base, d))])
    return os.path.join(base, dirs[-1]) if dirs else None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(here, "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        help="Directorio de resultados específico")
    parser.add_argument("--latest", action="store_true",
                        help="Usar el resultado más reciente")
    parser.add_argument("--list",   action="store_true",
                        help="Listar resultados disponibles")
    parser.add_argument("--base",   default=default_results,
                        help="Base directory de resultados")
    args = parser.parse_args()

    if args.list:
        list_results(args.base)
        return 0

    if args.latest:
        result_dir = find_latest(args.base)
        if not result_dir:
            print(f"  No hay resultados en {args.base}")
            return 1
    elif args.dir:
        result_dir = args.dir
    else:
        result_dir = find_latest(args.base)
        if not result_dir:
            print(f"  Usá --dir o --latest. Resultados disponibles:")
            list_results(args.base)
            return 1

    if not os.path.isdir(result_dir):
        print(f"  No existe: {result_dir}")
        return 1

    print(f"\n  Procesando: {result_dir}")

    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    dfs = load_all(result_dir)
    print("  CSVs encontrados:")
    for name, df in dfs.items():
        print(f"    {name:<10} {'OK ('+str(len(df))+' rows)' if df is not None else 'no disponible'}")

    print("\n  Generando gráficos...")

    plotters = [
        ("Thermal curve",       lambda: plot_thermal_curve(dfs["thermal"], dfs["tegra"], plots_dir)),
        ("Temperature profile", lambda: plot_temperature_profile(dfs["tegra"], plots_dir)),
        ("Power profile",       lambda: plot_power_profile(dfs["tegra"], plots_dir)),
        ("Memory profile",      lambda: plot_memory_profile(dfs["tegra"], plots_dir)),
        ("E2E breakdown",       lambda: plot_e2e_breakdown(dfs["e2e"], plots_dir)),
        ("RAG comparison",      lambda: plot_rag_comparison(dfs["rag"], plots_dir)),
        ("RAG overhead",        lambda: plot_rag_overhead_breakdown(dfs["rag"], plots_dir)),
        ("TTFT distribution",   lambda: plot_ttft_distribution(dfs["llm"], plots_dir)),
        ("TPS distribution",    lambda: plot_tps_distribution(dfs["llm"], plots_dir)),
        ("Bilingual comparison",lambda: plot_bilingual_comparison(dfs["llm"], plots_dir)),
        ("Prefill vs decode",   lambda: plot_prefill_vs_decode(dfs["llm"], plots_dir)),
        ("Intent latency",      lambda: plot_intent_latency(dfs["intent"], plots_dir)),
    ]

    generated = []
    for name, fn in plotters:
        try:
            path = fn()
            if path:
                print(f"    OK    {name:<25} → {os.path.basename(path)}")
                generated.append(path)
            else:
                print(f"    skip  {name:<25} (datos no disponibles)")
        except Exception as e:
            print(f"    FAIL  {name:<25} ({e})")

    print(f"\n  {len(generated)} gráficos generados en: {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())