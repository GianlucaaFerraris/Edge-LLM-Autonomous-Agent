"""
benchmark_lib.py — infraestructura común para el suite de benchmarks.

Contiene:
  - statistics helpers (percentile, summarize)
  - get_system_info(): snapshot de la Jetson (L4T, power mode, RAM)
  - TegrastatsMonitor: captura tegrastats en thread paralelo
  - ollama_chat_stream(): wrapper /api/chat con métricas nativas de Ollama

Diseño:
  - Sin dependencias de voice_io — todo texto puro.
  - Métricas server-side preferidas (Ollama reporta eval_count/eval_duration
    en el chunk final del stream, que son exactas a nivel GPU).
  - TTFT medido client-side porque incluye queue + prefill (lo que siente el user).
"""

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from statistics import mean, median, stdev
from typing import Optional

import requests


OLLAMA_URL = "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def percentile(data, p):
    """P-th percentile (0-100) por interpolación lineal."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(samples):
    """Resumen estadístico estándar: n, min, mean, median, p95, max, std."""
    if not samples:
        return {"n": 0, "min": 0, "mean": 0, "median": 0, "p95": 0,
                "max": 0, "std": 0}
    return {
        "n":      len(samples),
        "min":    min(samples),
        "mean":   mean(samples),
        "median": median(samples),
        "p95":    percentile(samples, 95),
        "max":    max(samples),
        "std":    stdev(samples) if len(samples) > 1 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# System info — snapshot al inicio del benchmark para el report
# ─────────────────────────────────────────────────────────────────────────────

def _safe_run(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def get_system_info():
    info = {}

    # L4T release
    try:
        with open("/etc/nv_tegra_release") as f:
            info["l4t"] = f.readline().strip()
    except Exception:
        info["l4t"] = "unknown"

    # Power mode (nvpmodel -q no necesita sudo para lectura)
    out = _safe_run(["nvpmodel", "-q"])
    m = re.search(r"NV Power Mode:\s*(\S+)", out)
    info["power_mode"] = m.group(1) if m else "unknown"

    # jetson_clocks status (si es accesible)
    jc = _safe_run(["jetson_clocks", "--show"])
    if jc:
        info["jetson_clocks_show"] = jc.splitlines()[:10]

    # Memory
    out = _safe_run(["free", "-m"])
    info["memory"] = out.splitlines()[:3]

    # Ollama
    info["ollama_version"] = _safe_run(["ollama", "--version"])
    info["ollama_ps"]      = _safe_run(["ollama", "ps"])

    # Kernel
    info["uname"]     = _safe_run(["uname", "-a"])
    info["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return info


# ─────────────────────────────────────────────────────────────────────────────
# TegrastatsMonitor — serie temporal de temp/freq/power
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TegraSample:
    t:           float = 0.0   # tiempo relativo desde start (s)
    ram_used:    int   = 0     # MB
    ram_total:   int   = 0
    swap_used:   int   = 0
    swap_total:  int   = 0
    cpu_temp:    float = 0.0   # °C
    gpu_temp:    float = 0.0
    tj_temp:     float = 0.0   # thermal junction (indicador de throttle)
    soc_temp:    float = 0.0
    gr3d_pct:    int   = 0     # GPU utilization 0-100
    gr3d_freq:   int   = 0     # MHz
    vdd_gpu_soc: int   = 0     # mW
    vdd_cpu_cv:  int   = 0     # mW
    vin_total:   int   = 0     # mW (VIN_SYS_5V0)


class TegrastatsMonitor:
    """
    Captura tegrastats en un thread paralelo.

    tegrastats imprime una línea por intervalo con formato tipo:
      '... RAM 2340/7620MB (lfb 15x4MB) SWAP 0/15552MB (cached 0MB)
       CPU [24%@729,15%@...] GR3D_FREQ 0%@[408] cpu@45.5C gpu@45C tj@46C
       VDD_GPU_SOC 580mW VDD_CPU_CV 430mW VIN_SYS_5V0 4200mW'

    Uso:
        mon = TegrastatsMonitor(interval_ms=1000)
        mon.start()
        ...carga de trabajo...
        mon.stop()
        summary = mon.summary()
        raw = mon.raw_samples()
    """

    RE_RAM     = re.compile(r"RAM (\d+)/(\d+)MB")
    RE_SWAP    = re.compile(r"SWAP (\d+)/(\d+)MB")
    RE_CPUTEMP = re.compile(r"cpu@([\d.]+)C")
    RE_GPUTEMP = re.compile(r"gpu@([\d.]+)C")
    RE_TJTEMP  = re.compile(r"tj@([\d.]+)C")
    RE_SOCTEMP = re.compile(r"soc[0-9]@([\d.]+)C")
    RE_GR3D    = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+)\]?)?")
    RE_VDD_GPU = re.compile(r"VDD_GPU_SOC (\d+)mW")
    RE_VDD_CPU = re.compile(r"VDD_CPU_CV (\d+)mW")
    RE_VIN     = re.compile(r"VIN_SYS_5V0 (\d+)mW")

    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.samples: list[TegraSample] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._t0 = None

    def _parse(self, line: str) -> Optional[TegraSample]:
        try:
            s = TegraSample()
            m = self.RE_RAM.search(line)
            if m: s.ram_used, s.ram_total = int(m.group(1)), int(m.group(2))
            m = self.RE_SWAP.search(line)
            if m: s.swap_used, s.swap_total = int(m.group(1)), int(m.group(2))
            m = self.RE_CPUTEMP.search(line)
            if m: s.cpu_temp = float(m.group(1))
            m = self.RE_GPUTEMP.search(line)
            if m: s.gpu_temp = float(m.group(1))
            m = self.RE_TJTEMP.search(line)
            if m: s.tj_temp = float(m.group(1))
            m = self.RE_SOCTEMP.search(line)
            if m: s.soc_temp = float(m.group(1))
            m = self.RE_GR3D.search(line)
            if m:
                s.gr3d_pct = int(m.group(1))
                if m.group(2): s.gr3d_freq = int(m.group(2))
            m = self.RE_VDD_GPU.search(line)
            if m: s.vdd_gpu_soc = int(m.group(1))
            m = self.RE_VDD_CPU.search(line)
            if m: s.vdd_cpu_cv = int(m.group(1))
            m = self.RE_VIN.search(line)
            if m: s.vin_total = int(m.group(1))
            return s
        except Exception:
            return None

    def _reader(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            s = self._parse(line)
            if s:
                s.t = time.monotonic() - self._t0
                self.samples.append(s)

    def start(self):
        self._t0 = time.monotonic()
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except FileNotFoundError:
            print("[WARN] tegrastats not available — skipping thermal monitoring")
            self._proc = None
            return
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try: self._proc.kill()
                except Exception: pass
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {"available": False, "n_samples": 0}

        def col(attr):
            return [getattr(s, attr) for s in self.samples
                    if getattr(s, attr) not in (0, 0.0)]

        # Detectar throttle events: tj >= 85°C por >=2 samples consecutivos.
        # Orin Nano empieza a throttlear el GPU alrededor de 85°C, hard limit
        # alrededor de 97°C. Por debajo de 85°C no hay throttling perceptible.
        tj_series = [s.tj_temp for s in self.samples if s.tj_temp > 0]
        throttle_events = 0
        i = 0
        while i < len(tj_series) - 1:
            if tj_series[i] >= 85.0 and tj_series[i+1] >= 85.0:
                throttle_events += 1
                while i < len(tj_series) and tj_series[i] >= 85.0:
                    i += 1
            else:
                i += 1

        return {
            "available":       True,
            "n_samples":       len(self.samples),
            "duration_s":      round(self.samples[-1].t, 1),
            "ram_used_mb":     summarize(col("ram_used")),
            "swap_used_mb":    summarize(col("swap_used")),
            "cpu_temp_c":      summarize(col("cpu_temp")),
            "gpu_temp_c":      summarize(col("gpu_temp")),
            "tj_temp_c":       summarize(col("tj_temp")),
            "gr3d_pct":        summarize(col("gr3d_pct")),
            "vdd_gpu_soc_mw":  summarize(col("vdd_gpu_soc")),
            "vdd_cpu_cv_mw":   summarize(col("vdd_cpu_cv")),
            "vin_total_mw":    summarize(col("vin_total")),
            "throttle_events": throttle_events,
            "tj_above_85_pct": (100.0 * sum(1 for v in tj_series if v >= 85.0)
                                / len(tj_series)) if tj_series else 0.0,
        }

    def raw_samples(self):
        return [asdict(s) for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# Ollama native client — métricas exactas del servidor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    ttft_s:         float = 0.0   # client-side wall clock hasta primer token
    total_s:        float = 0.0   # client-side total
    eval_count:     int   = 0     # tokens generados (server)
    prompt_eval_count: int = 0    # tokens del prompt (server)
    eval_duration_ns:  int = 0    # tiempo puro de decode (server)
    prompt_eval_duration_ns: int = 0  # prefill (server)
    load_duration_ns: int  = 0    # carga del modelo (~0 si ya en RAM)
    tps_server:     float = 0.0   # tokens/s medidos por Ollama
    tps_client:     float = 0.0   # tokens/s desde wall clock
    response_text:  str   = ""
    error:          str   = ""


def ollama_chat_stream(model: str, messages: list, temperature: float = 0.3,
                       max_tokens: int = 600, timeout: int = 300) -> InferenceResult:
    """
    POST /api/chat con stream=True.

    Ollama envía un chunk por token. El último chunk (done=true) incluye:
      - eval_count:               tokens generados
      - eval_duration:            tiempo puro de decode en nanosegundos
      - prompt_eval_count:        tokens del prompt
      - prompt_eval_duration:     prefill time en ns
      - load_duration:            tiempo de carga del modelo (0 si ya estaba)
      - total_duration:           wall time total (incluye queue, network, etc.)

    Las métricas server-side son las que debés citar en el informe porque
    no dependen del cliente; las client-side son útiles para TTFT (UX real).
    """
    res = InferenceResult()
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   True,
        "options":  {"temperature": temperature, "num_predict": max_tokens},
    }

    t0 = time.perf_counter()
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload,
                           stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    res.eval_count              = data.get("eval_count", 0)
                    res.prompt_eval_count       = data.get("prompt_eval_count", 0)
                    res.eval_duration_ns        = data.get("eval_duration", 0)
                    res.prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
                    res.load_duration_ns        = data.get("load_duration", 0)
                    break

                content = data.get("message", {}).get("content", "")
                if content:
                    if res.ttft_s == 0.0:
                        res.ttft_s = time.perf_counter() - t0
                    res.response_text += content

    except Exception as e:
        res.error = str(e)
        return res

    res.total_s = time.perf_counter() - t0

    if res.eval_duration_ns > 0:
        res.tps_server = res.eval_count / (res.eval_duration_ns / 1e9)
    if res.total_s > res.ttft_s > 0:
        res.tps_client = res.eval_count / (res.total_s - res.ttft_s)

    return res


def ollama_warmup(model: str, n: int = 2):
    """Precarga el modelo a RAM. Primera inferencia siempre paga model load."""
    for _ in range(n):
        ollama_chat_stream(
            model,
            [{"role": "user", "content": "hi"}],
            temperature=0.0, max_tokens=5, timeout=120,
        )