## <font color="#de7802">1. Definición Técnica y Origen del Problema</font>

En la arquitectura de los [Transformers](../Transformers/Transformers.md) (como Llama, Qwen o GPT), el proceso de [Inferencia](../Hardware%20-%20Fundamentos%20&%20Definiciones/Inferencia.md) es **autoregresivo**. Esto significa que para generar el [Token](Token.md) $N$, el modelo debe procesar no solo tu pregunta inicial, sino también todos los tokens previos generados por él mismo ($N-1$, $N-2$, etc.).

Sin **KV Caching**, en cada nuevo paso de generación, el modelo recalcula innecesariamente las matrices de **Key (K)** y **Value (V)** de todos los tokens anteriores en cada una de las capas de atención. Esto genera una carga computacional de orden $O(n^2)$, donde el tiempo de procesamiento se dispara exponencialmente a medida que la conversación crece.

## <font color="#de7802">2. El Mecanismo de Inferencia con KV Cache</font>

El **KV Caching** es una técnica de optimización que almacena en la memoria RAM los tensores de "Key" y "Value" ya calculados.

1. **Fase de Prefill:** El modelo procesa el _prompt_ inicial, calcula los pares K y V de cada token y los guarda en un búfer de memoria.
2. **Fase de Decoding:** Para generar el siguiente token, el modelo solo calcula el par K/V del **último** token generado y lo concatena al caché existente.
Relacionandolo con la parte de [Latencia de Inferencia](../Hardware%20-%20Fundamentos%20&%20Definiciones/Latencia%20de%20Inferencia.md):

| Fase     | Métrica de Rendimiento       | Descripción                                                        |
| -------- | ---------------------------- | ------------------------------------------------------------------ |
| Prefill  | TTFT (Time to First Token)   | El modelo procesa todo tu _prompt_ de entrada de una sola vez.     |
| Decoding | TPOT (Time per Output Token) | El modelo genera la respuesta palabra por palabra (token a token). |

## <font color="#de7802">3. Simbiosis con el Proyecto (Rockchip RK3588)</font>

En nuestro agente de ingeniería, el KV Cache es el puente entre la **NPU** y la **RAM**:

- **Aceleración de Inferencia:** Al no recalcular el pasado, la NPU se enfoca exclusivamente en la matriz del nuevo token, manteniendo los **tokens/s** estables.
- **Limitación de** [Ancho de Banda de Memoria](../Hardware%20-%20Fundamentos%20&%20Definiciones/Ancho%20de%20Banda%20de%20Memoria.md): El KV Cache reside en la LPDDR4x. Cada vez que generamos un token, la NPU debe leer el caché completo. Por eso, un caché muy grande satura el bus de 33.8 GB/s.