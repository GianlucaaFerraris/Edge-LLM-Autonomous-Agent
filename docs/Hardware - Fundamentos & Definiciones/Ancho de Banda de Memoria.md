## <font color="#de7802">1. Definición Técnica</font>

El **Ancho de Banda de Memoria** es la velocidad máxima a la que los datos pueden ser leídos o escritos en la memoria RAM por el procesador (CPU/GPU/NPU). Se mide habitualmente en Gigabytes por segundo (**GB/s**).

Su fórmula teórica es:
$$\text{BW} = \text{Frecuencia del Bus} \times \text{Ancho del Bus (bits)} \times \text{Operaciones por ciclo} / 8$$

---
## <font color="#de7802">2. Capacidad vs Ancho de Banda de Memoria</font>

No se debe confundir el tamaño de la memoria RAM (ej. 16GB) con su ancho de banda (ej. 32 GB/s). Para la generación de texto (LLMs), el **tamaño** define si el modelo _entra_ en la placa, pero el **ancho de banda** define qué tan _rápido_ se procesa el modelo.

Para esto es bueno trabajar con la analogía de un tanque de agua. Imagina que la *memoria RAM* es un tanque de agua que alimenta a tu *procesador* (la canilla).
- **Capacidad de RAM:** Son los 8GB o 16GB. Esto es el tamaño del tanque. Determina cuánta agua (datos) puedes guardar. Si el tanque es muy chico, el modelo no cabe y el sistema falla (Out of Memory).  Cuando enciendes tu asistente, el modelo (digamos que pesa 5 GB entonces ocupa eso y te quedan 11GB) se carga **una sola vez** desde el SSD a la RAM y se queda ahí sentado. 
- **Ancho de Banda (lo que estamos analizando):** Es el **diámetro del caño** que conecta el tanque con la canilla. Se mide en **GB/s**. Determina cuánta agua puede pasar por segundo. Es en otras palabras la velocidad en el que el procesador puede leer los GB almacenados en la RAM (tanque).

Por esto mismo, el ancho de banda de memoria es un concepto fundamental al analizar que modelo vamos a elegir para nuestro dispositivo. Ya que mientras mas grande sea el modelo no solo va a dejar menos espacio en la RAM para el resto de tareas, sino que los [Token](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Token.md) por segundo de respuesta van a ser menores ya que el procesador tiene que leer el modelo entero cada vez que responde (recomiendo ver [Latencia de Inferencia](Latencia%20de%20Inferencia.md)).

---
## <font color="#de7802">3. ¿Por qué es el cuello de botella en Machine Learning?</font>

En la [Inferencia](Inferencia.md) de [LLM](../LLM%20-%20Conceptos%20&%20Modelos/LLM.md), nos enfrentamos a un problema de **Baja Intensidad Aritmética**. Esto significa que el procesador termina sus cálculos mucho más rápido de lo que la RAM puede enviarle los pesos del modelo.

### <font color="#fbd5b5">El Ciclo de Carga del Token:</font>

Para generar una sola palabra, el hardware debe:
1. Mover **todos** los parámetros del modelo (ej. 4.8 GB para [QWen 2.5 7B](../LLM%20-%20Conceptos%20&%20Modelos/QWen%202.5%207B.md)) desde la RAM hasta los registros del procesador.
2. Realizar el cálculo matemático (esto es casi instantáneo).
3. Emitir el [Token](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Token.md).

Si tu ancho de banda es de 32 GB/s ([Radxa Rock 5 Model B](../Radxa%20-%20Conceptos%20&%20SBC/Radxa%20Rock%205%20Model%20B.md)) y tu modelo pesa 4.8 GB, el límite físico absoluto es de:
$$\text{Tokens/s} \approx \frac{32 \text{ GB/s}}{4.8 \text{ GB}} \approx 6.6 \text{ tokens/s}$$
---
## <font color="#de7802">4. Relevancia en Edge AI (Rock 5B vs Jetson)</font>

En dispositivos de borde (ver [Edge AI](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Edge%20AI.md)), el ancho de banda es limitado comparado con las GPUs de escritorio (una RTX 4050 como la que uso yo tiene ~160 GB/s).
* [Radxa Rock 5 Model B](../Radxa%20-%20Conceptos%20&%20SBC/Radxa%20Rock%205%20Model%20B.md): Utiliza LPDDR4x. Es eficiente, pero su bus de datos es estrecho, lo que limita la velocidad de conversación.
* [NVIDIA Jetson Orin Nano](../NVIDIA%20-%20Conceptos%20&%20SBC/NVIDIA%20Jetson%20Orin%20Nano.md): Utiliza LPDDR5 con un bus más ancho (~68 GB/s). Esto permite que el mismo modelo [QWen 2.5 7B](../LLM%20-%20Conceptos%20&%20Modelos/QWen%202.5%207B.md) corra al **doble de velocidad** que en la Radxa, simplemente porque el procesador no está "esperando" tanto los datos.
---
## <font color="#de7802">5. Estrategias de Mitigación en el Proyecto</font>

Para superar el "Muro de la Memoria" en nuestra implementación:
1. [Cuantización](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/Cuantización.md) (4-bit o 3-bit): Al reducir el tamaño del modelo, disminuimos los GB que deben viajar por el bus en cada paso.
2. [KV Caching](../Machine%20Learning%20-%20Fundamentos%20&%20Definiciones/KV%20Caching.md):** Reutilizamos cálculos previos para no tener que volver a leer todo el historial de la conversación desde la RAM.
3. **Optimización de RAM:** Cerrar servicios innecesarios en Linux para asegurar que el bus de datos esté dedicado exclusivamente al proceso de inferencia.