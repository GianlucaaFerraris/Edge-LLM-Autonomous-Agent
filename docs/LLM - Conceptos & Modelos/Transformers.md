
### <font color="#de7802">1. Ontología y Arquitectura del Transformer</font>

El Transformer, introducido en _Vaswani et al. (2017)_ ([link del paper](https://arxiv.org/abs/1706.03762)), supuso un cambio de paradigma al abandonar las dependencias secuenciales de las Redes Neuronales Recurrentes (RNN) y las LSTMs en favor de un mecanismo de **Atención Paralelizable**.

#### <font color="#fbd5b5">1.1. El Mecanismo de Auto-Atención (Self-Attention)</font>

El núcleo del Transformer es el cálculo de la **Atención de Producto Punto Escalada**. Definimos tres matrices de pesos entrenables: Consultas ($Q$), Claves ($K$) y Valores ($V$).

$$\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

- **Escalamiento ($1/\sqrt{d_k}$):** Previene la saturación del gradiente en la función softmax cuando las dimensiones del modelo son elevadas.
- **Multi-Head Attention (MHA):** Permite que el modelo proyecte de forma paralela el espacio de representación en diferentes subespacios, capturando dependencias sintácticas y semánticas simultáneamente.

#### <font color="#fbd5b5">1.2. Feed-Forward Networks (FFN) y Capas de Normalización</font>

Cada bloque de atención es seguido por una red _position-wise feed-forward_:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

En modelos modernos como [QWen 2.5 7B](QWen%202.5%207B.md) (que estamos usando), se utilizan variantes como **SwiGLU** para mejorar la expresividad no lineal del modelo.

### <font color="#de7802">2. Relación con los LLMs Modernos (Qwen/Llama)</font>

Los [LLM](LLM.md) actuales han evolucionado hacia arquitecturas de **Solo Decodificador** (_Decoder-only_), eliminando la parte del codificador del Transformer original.

- **Modelos Causales:** Predicen el token $t+1$ basándose únicamente en los tokens $t_{<1}$, utilizando una máscara causal en la matriz de atención para evitar la fuga de información del futuro.
- **Escalabilidad:** Gracias a la paralelización del entrenamiento, se han podido escalar hasta los billones de parámetros ($3B \space o \space 7B$ en nuestro caso), donde emergen capacidades de razonamiento técnico.
### <font color="#de7802">3. Implementación en el Proyecto: De Predator a Radxa Rock 5B</font>

#### <font color="#fbd5b5">3.1. Optimización en Host (RTX 4050) via Unsloth</font>

Para realizar el [Fine Tuning](Fine%20Tuning.md) de forma eficiente en hardware de consumo (6GB VRAM), aplicamos:

1. [QLoRA](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/QLoRA.md) **(Quantized Low-Rank Adaptation):** Congelamos los pesos del modelo base en **4-bit NormalFloat (NF4)**.
	- Insertamos matrices de rango bajo ($A$ y $B$) en las capas de atención. Solo estas matrices son actualizables, reduciendo drásticamente la memoria necesaria para los gradientes.
2. **Fused Kernels (Triton):** Unsloth optimiza la operación de Softmax y el cálculo de la pérdida integrándolos en un solo Kernel de GPU, evitando costosos movimientos de datos entre registros.
#### <font color="#fbd5b5">3.2. Despliegue en Edge (Radxa Rock 5B) via RKLLM</font>

El despliegue en la NPU del chip RK3588 requiere una transpilación del grafo computacional:

1. **Exportación:** El modelo se convierte de PyTorch/Safetensors a un formato intermedio (ONNX o directamente a RKLLM).
2. [Cuantización](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Cuantización.md)** Post-Entrenamiento (PTQ):** La NPU de Rockchip utiliza aceleradores INT8/INT4. El Toolkit de RKLLM calibra los pesos para que el modelo corra en los **6 TOPS** de la NPU sin saturar el bus de memoria.
3. [Inferencia](../Hardware%20-%20Fundamentos%20&%20Definiciones/Inferencia.md): El uso de la NPU libera la CPU (Cortex-A76/A55) para tareas de sistema, permitiendo que el tutor responda en tiempo real.

### <font color="#de7802">4. Glosario Técnico para el Proyecto</font>

- **Tokenizer:** El algoritmo (BPE en Qwen) que segmenta el texto en unidades numéricas.
- **Context Window:** El límite de tokens que el modelo puede "atender" a la vez (2048 en nuestra config inicial).
- **VRAM Overhead:** Memoria consumida no solo por los pesos, sino por el [KV Caching](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/KV%20Caching.md) generado durante la inferencia.