# Testing del Asistente Fine-tuneado

## Setup (una sola vez)

### 1. Registrar el modelo en Ollama

```bash
cd ~/Desktop/Agenty-Edge-Assitant/model/gguf
ollama create asistente -f Modelfile
```

Verificar que quedó registrado:
```bash
ollama list
```

### 2. Instalar dependencias de test

```bash
conda activate tutor_env
pip install pytest openai requests
```

---

## Estructura

```
src/test/
  conftest.py       # fixtures compartidos (pytest los carga automáticamente)
  test_modes.py     # unit tests de los 3 modos + selector
  test_latency.py   # benchmark de latencia
  manual_chat.py    # chat interactivo para validación manual
```

---

## Correr los tests

### Unit tests completos
```bash
cd ~/Desktop/Agenty-Edge-Assitant/src/test
pytest test_modes.py -v
```

### Solo un modo
```bash
pytest test_modes.py -v -k "english"
pytest test_modes.py -v -k "engineering"
pytest test_modes.py -v -k "agent"
pytest test_modes.py -v -k "selector"
```

### Benchmark de latencia (pytest)
```bash
pytest test_latency.py -v -s
```

### Benchmark completo con reporte (standalone)
```bash
python test_latency.py
# Genera latency_results.json con el resumen
```

---

## Validación manual

### Menú interactivo
```bash
python manual_chat.py
```

### Ir directo a un modo
```bash
python manual_chat.py --mode english
python manual_chat.py --mode engineering
python manual_chat.py --mode agent
```

**Comandos dentro del chat:**
- `salir` → volver al menú
- `limpiar` → borrar el contexto actual (nuevo turno sin historial)

Cada respuesta muestra:
- `TTFT`: tiempo hasta el primer token (lo que percibe el usuario)
- `Total`: tiempo total de la respuesta

---

## Valores de latencia esperados (T4 → notebook NVIDIA)

| Caso              | TTFT esperado | Total esperado |
|-------------------|---------------|----------------|
| Detección de modo | < 1s          | < 5s           |
| English (corto)   | < 2s          | < 15s          |
| Engineering (med) | < 3s          | < 30s          |
| Agent tool call   | < 2s          | < 10s          |

> En la Rock 5B los tiempos serán distintos — este benchmark sirve como baseline.

---

## Si `asistente` no está disponible

Los tests y el chat caen automáticamente al modelo `qwen2.5:7b` (fallback).
Para forzar el fallback mientras probás:
```bash
ollama list  # verificar modelos disponibles
```
