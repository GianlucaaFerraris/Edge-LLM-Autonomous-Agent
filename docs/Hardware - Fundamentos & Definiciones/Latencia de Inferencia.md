## <font color="#de7802">1. Definición Técnica</font>

Es el componente más pesado de la latencia total. Se divide en dos métricas fundamentales que utilizaremos para nuestro [Benchmark](Benchmark.md) en [NVIDIA Jetson Orin Nano](../NVIDIA%20-%20Conceptos%20&%20SBC/NVIDIA%20Jetson%20Orin%20Nano.md) o la Rock 5 Model B.

### <font color="#fac08f">A. TTFT (Time To First Token)</font>

Es el tiempo que tarda el modelo en procesar el [Prompt](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Prompt.md) y emitir el **primer carácter**. 
* **Fórmula:** $\text{TTFT} = T_{prefill}$
* Representa la fase donde el hardware lee toda la entrada y "entiende" el contexto. En la Jetson o Rock 5B, un prompt muy largo (ej. un texto para corregir) aumentará drásticamente el TTFT.

### <font color="#fac08f">B. TPOT (Time Per Output Token)</font>

Es el tiempo promedio entre la generación de cada palabra subsiguiente al primer caracter. 
* **Fórmula:** $\text{TPOT} = \frac{1}{\text{Tokens per Second}}$
* Si el TPOT es de 150ms, la velocidad será de ~6.6 [Token](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Token.md)/s, lo cual es similar a la velocidad de lectura humana rápida.

Relacionandolo con [KV Caching](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/KV%20Caching.md):

| Fase     | Metrica de Rendimiento             | Descripción                                                        |
| -------- | ---------------------------------- | ------------------------------------------------------------------ |
| Prefill  | **TTFT** (_Time to First Token_)   | El modelo procesa todo tu _prompt_ de entrada de una sola vez.     |
| Decoding | **TPOT** (_Time Per Output Token_) | El modelo genera la respuesta palabra por palabra (token a token). |

---
## <font color="#de7802">2. Cuellos de Botella: Compute vs Memory Bound</font>

* **Fase de Prefill (TTFT):** Suele estar limitada por el cómputo (**Compute-bound**). Aquí importan los T[FLOPS](FLOPS.md) de la placa.
* **Fase de Decoding (TPOT):** Está limitada por la memoria (**Memory-bound**). Aquí importa el [Ancho de Banda de Memoria](Ancho%20de%20Banda%20de%20Memoria.md) (GB/s).

---
## <font color="#de7802">3. Implicación en el Proyecto</font>

Para nuestro proyecto:
1. **Meta de TTFT:** < 1.0 segundos (para que la respuesta parezca inmediata).
2. **Meta de TPOT:** < 200 ms (para que el habla del asistente sea fluida y no entrecortada).
3. **Estrategia:** Usaremos [KV Caching](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/KV%20Caching.md) (Almacenamiento de claves y valores) para que el modelo no tenga que re-procesar toda la conversación en cada turno, reduciendo la latencia acumulada.