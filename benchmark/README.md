# Agenty Benchmarks — Jetson Orin Nano 8GB

Suite de benchmarks en texto puro (no voice_io) para caracterizar el asistente Agenty en Jetson Orin Nano 8GB y producir datos para documentación.

## Estructura

```
benchmarks/
├── benchmark.py              # entry point / CLI
├── benchmark_lib.py          # TegrastatsMonitor + OllamaClient + stats
├── benchmark_suites.py       # 5 suites: llm, intent, rag, e2e, thermal
├── benchmark_reporter.py     # JSON + CSV + Markdown
├── prompts.json              # dataset de test
└── results/                  # (se crea solo) outputs por corrida
    └── YYYYMMDD_HHMMSS_<preset>/
        ├── raw_data.json
        ├── system_info.json
        ├── llm.csv
        ├── intent.csv
        ├── rag.csv
        ├── e2e.csv
        ├── thermal.csv
        ├── tegrastats.csv    # serie temporal
        └── report.md         # tablas para el informe
```

Debe quedar al mismo nivel que `src/`:

```
Agenty-Edge-Assistant/
├── src/
└── benchmarks/       <-- acá
```

## Instalación

Solo necesita `requests` (que ya tenés por Ollama):

```bash
pip install requests
```

Chequear que `tegrastats` está en PATH:

```bash
which tegrastats  # /usr/bin/tegrastats
```

## Uso

```bash
cd ~/Desktop/Agenty-Edge-Assistant/benchmarks

# Preset standard (lo que pediste: 10 iter, thermal 10 min) — ~1 h
python benchmark.py

# Suite completa más rápido — ~30 min
python benchmark.py --quick

# Suite exhaustiva — ~2-3 h
python benchmark.py --thorough

# Solo algunas suites (útil para iterar)
python benchmark.py --suites llm
python benchmark.py --suites llm,rag --skip-thermal

# Override manual
python benchmark.py --iterations 15 --thermal-s 600
```

Antes de correr:
- Asegurate de cerrar el `main.py` de Agenty si lo tenés abierto (Ollama se lleva la RAM).
- Idealmente cerrá el desktop (`sudo systemctl isolate multi-user.target`) para maximizar RAM libre durante el test.
- Si tenés `jetson_clocks` activo, documentalo en el informe. El benchmark captura el power mode actual automáticamente.

## Qué mide cada suite

### 1. LLM puro
Invoca `/api/chat` de Ollama directo, sin intent ni RAG. Mide el modelo base contra el hardware:
- **TTFT** (Time To First Token): client-side, desde el POST hasta recibir el primer token. Incluye queue + prefill.
- **TPS** (Tokens Per Second): server-side (reportado por Ollama en `eval_duration`) — medición exacta del decode sin contaminar con red/parseo.
- **Prompt eval (prefill)**: server-side, tiempo de procesar el prompt antes de empezar a generar.
- **Tokens generados**: `eval_count` del server.

### 2. Intent classifier
Mide la latencia que agrega el clasificador de intents a cada turno, antes del LLM principal. Es crítico porque corre en TODOS los mensajes del user. Valida `accuracy` contra `expected` en `prompts.json`.

### 3. RAG vs no-RAG
A/B directo: corre el mismo prompt con y sin RAG. Reporta:
- **RAG overhead**: embed (MiniLM) + FAISS search. Lo que tarda ANTES del LLM.
- **Prompt inflation**: cuántos tokens extra mete el RAG en el prompt.
- **Total overhead**: `rag_overhead + (RAG_total - noRAG_total)`. Es lo que cuesta agregar RAG al pipeline.
- **RAG fired %**: qué porcentaje de queries pasaron ambos thresholds (domain + relevance).

### 4. End-to-end
Simula un turno real: intent → (RAG si aplica) → chat stream. Reporta el breakdown y el `turn_total_s`, que es lo que siente el usuario.

### 5. Thermal stress
Loop de N segundos haciendo inferencias continuas con varios prompts distintos (para evitar que el KV cache optimice el mismo query). `TegrastatsMonitor` corre en paralelo a 1 Hz capturando temp/freq/power. Reporta:
- **Degradación**: TPS del primer cuartil vs último cuartil del run. Si cae más de ~10% es thermal throttling.
- **Throttle events**: momentos donde `tj@` (thermal junction) sostuvo ≥85 °C por ≥2 s consecutivos.
- **Power**: mean/max de `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0`.

## Métricas que vas a ver en el report

Para cada métrica numérica, se reporta: `mean`, `median`, `p95`, `std`, `min`, `max`, `n`.

**Por qué p95 y no solo mean**: en edge hardware con thermal throttling, la media esconde los outliers. El `p95` captura el tail (5% peor) que es donde se ve el throttling esporádico. Citar `mean ± std` Y `p95` en el informe es lo académicamente correcto.

**Por qué TPS server-side**: Ollama te dice exactamente cuántos tokens generó y en cuánto tiempo puro de decode (sin TTFT, sin network, sin tokenización del prompt). El TPS client-side está contaminado por todo eso. Para comparar contra benchmarks de la industria, siempre usar server-side.

## Output para el informe

El `report.md` tiene tablas en formato Markdown listas para copiar a un documento. Si usás LaTeX, las podés convertir con pandoc:

```bash
pandoc report.md -o report.tex
```

El `raw_data.json` contiene todo sin agregar, para análisis posterior con pandas si querés armar gráficos:

```python
import pandas as pd
df = pd.read_csv("thermal.csv")
df.plot(x="t_since_start_s", y="tps")    # curva de degradación
```

Y el `tegrastats.csv` es una serie temporal completa con una fila por segundo, perfecta para plotear temperatura vs tiempo.

## Troubleshooting

**"Modelo 'asistente' no registrado"**: Registralo desde tu Modelfile (`ollama create asistente -f Modelfile`) o pasá otro modelo con `--model qwen2.5:7b`.

**"engineering_session no importable"**: Verificá que el benchmark esté al mismo nivel que `src/`. Si está en otro lado, ajustá `_ROOT` en `benchmark.py`.

**"RAG engine devolvió None"**: El índice FAISS no está. Buildeá con `python src/engineering/rag/build_index.py`. Suite RAG y parte de E2E se saltean sin romper el resto.

**"tegrastats no encontrado"**: Está en `/usr/bin/tegrastats` en JetPack. Si no lo encuentra, agregalo al PATH. Sin tegrastats el thermal test igual corre pero sin captura de temp/power.

**CUDA OOM durante el test**: Si seguís viendo fallos de alloc CUDA después del upgrade a JetPack 6.2.2, agregá las variables de entorno a Ollama:

```bash
sudo systemctl edit ollama
# [Service]
# Environment="OLLAMA_FLASH_ATTENTION=1"
# Environment="OLLAMA_NUM_PARALLEL=1"
# Environment="OLLAMA_CONTEXT_LENGTH=2048"
sudo systemctl daemon-reload && sudo systemctl restart ollama
```