## <font color="#de7802">1. Definición y Relevancia</font>

**Edge AI** es el paradigma de computación que consiste en ejecutar algoritmos de Inteligencia Artificial directamente en dispositivos locales (hardware embebido), sin necesidad de conectividad a la nube o procesamiento en servidores externos.

En el contexto de este proyecto, el "Edge" está representado por la plataforma [NVIDIA Jetson Orin Nano](../NVIDIA%20-%20Conceptos%20&%20SBC/NVIDIA%20Jetson%20Orin%20Nano.md).

---
## <font color="#de7802">2. Pilares Técnicos del Proyecto en el Edge</font>

### <font color="#fac08f">A. Latencia y Determinismo</font>

Al procesar la [Inferencia](../Hardware%20-%20Fundamentos%20&%20Definiciones/Inferencia.md) del LLM localmente, eliminamos la [Latencia](../Hardware%20-%20Fundamentos%20&%20Definiciones/Latencia.md) de red ($L_{network}$). La latencia total depende exclusivamente del rendimiento del hardware:

$$\text{Latencia Total} = \text{Latencia de Inferencia} + \text{Latencia de Pre-procesamiento}$$

Esto por ejemplo es crítico para la **Práctica de Inglés** de nuestro proyecto, donde un retraso superior a los 2 segundos rompería el flujo de la conversación natural. Ademas, si en un futuro necesitamos que el agente de ia diriga algun elemento en movimiento como un robot tiene las mismas implicancias una latencia del orden de los segundos.

### <font color="#fac08f">B. Privacidad y Seguridad (Data Sovereignty)</font>

Toda la información procesada (transcripciones de voz y correcciones gramaticales) permanece dentro de la red local. Esto cumple con los estándares más estrictos de privacidad, ya que no hay exposición de datos personales a APIs de terceros (como OpenAI o Google).

### <font color="#fac08f">C. Eficiencia Energética y Costo Operativo</font>

A diferencia de los modelos en la nube que consumen kilowatts en centros de datos, este proyecto busca la optimización en dispositivos que consumen entre **15W y 30W**, permitiendo una asistencia 24/7 con un impacto energético mínimo.

---

## <font color="#de7802">3. El Desafío del "Resource-Constrained Computing"</font>

Implementar un modelo de la familia [QWen 2.5 7B](../LLM%20-%20Conceptos%20&%20Modelos/QWen%202.5%207B.md) en el Edge nos enfrenta a tres limitaciones físicas que este proyecto documentará:

1. **Compute Bound:** La capacidad de cálculo bruto medida en [FLOPS](../Hardware%20-%20Fundamentos%20&%20Definiciones/FLOPS.md).
2. **Memory Bound:** El cuello de botella del ancho de banda de memoria ($\text{GB/s}$) que limita los [Token](Token.md)/segundo.
3. **Thermal Throttling:** La gestión de calor en dispositivos compactos sin ventilación industrial.

## <font color="#de7802">4. Técnicas de Adaptación para Edge AI</font>

Para que un LLM sea viable en mi [NVIDIA Jetson Orin Nano](../NVIDIA%20-%20Conceptos%20&%20SBC/NVIDIA%20Jetson%20Orin%20Nano.md), el proyecto aplicará:

- **Cuantización:** Reducción de la precisión de los pesos (ej. de FP16 a INT4).
- [Fine Tuning](../LLM%20-%20Conceptos%20&%20Modelos/Fine%20Tuning.md) ([QLoRA](QLoRA.md)): Adaptación del modelo sin re-entrenar todos sus parámetros.
- **Inferencia Acelerada:** Uso de núcleos [CUDA](../NVIDIA%20-%20Conceptos%20&%20SBC/CUDA.md) (NVIDIA) para desplazar la carga de trabajo fuera de la CPU general.