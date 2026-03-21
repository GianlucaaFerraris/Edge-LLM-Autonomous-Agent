## <font color="#de7802">1. Fundamentos: De la Precisión a la Eficiencia</font>

La cuantización es el proceso de reducir la precisión de los números que representan los pesos (_weights_) del modelo [LLM](../LLM%20-%20Conceptos%20&%20Modelos/LLM.md). Originalmente, los modelos se entrenan en **FP32** (32 bits por número). Sin embargo, esto es inviable para [Edge AI](Edge%20AI.md). Por esto es que la cuantización cuando nuestros recursos de hardware son limitados juega un rol importante.

Reducir el numero de bits implica que el modelo LLM resultante ocupe menos espacio en nuestra memoria (ya sea para almacenarlo permanentemente en el disco sólido o cuando lo cargamos a la RAM). Además, las operaciones como multiplicación de matrices pueden ser realizadas mucho más rapidamente con aritmética de enteros. 

## <font color="#de7802">2. Niveles de Cuantización</font>

- **FP16/BF16 (16 bits):** El estándar de oro para entrenamiento. Alta fidelidad, pero ocupa mucha memoria (2 Bytes por parámetro).
- **INT8 (8 bits):** Reduce el peso a la mitad (1 Byte por parámetro). Es el "punto dulce" nativo para la NPU del RK3588 (6 TOPS).
- **INT4 / Q4 (4 bits):** Reduce el peso a ~0,7 Bytes por parámetro. Permite meter un modelo de 7B (7 mil millones de parámetros) en solo 4,8 GB de RAM.

## <font color="#de7802">3. El Proceso Matemático</font>

Cuantizar no es simplemente "borrar decimales". Se utiliza un factor de escala ```float32``` positivo ($S$) y un punto cero ($Z$) para mapear un rango de valores flotantes a un rango entero. De esta forma podemos obtener el valor cuantizado ```int8``` a través del numero inicial flotante $W_{float}$ :
$$x = S \left(\frac{W_{float}}{S} + Z\right)$$

De esta forma los valores dentro del rango $[a,b]$ conservan su valor entero mientras que los que se encuentran fuera son recotrados a su valor mas cercano. Es muy usado en los métodos de cuantización perder uno de los $256$ numeros posibles dentro de un `int8` para que el rango  $[a,b]=[-127,127]$ de forma que $Z=0$. Se pierde un poco de precisión pero evitamos realizar muchas operaciones de suma para cuantizar y trabajar. Leer este [paper](https://arxiv.org/pdf/1712.05877) si queres saber más al respecto.

La **granularidad** de la cuantización es un factor determinante en la fidelidad del modelo, pudiendo aplicarse de forma *per-tensor* (array multidimensional), donde un único par de parámetros de escala y punto cero ($S, Z$) se utiliza para todo el bloque de datos, o de forma *per-channel*, permitiendo una precisión mucho mayor al asignar un par $(S, Z)$ específico para cada canal o dimensión del tensor, lo cual reduce el error de redondeo a costa de un ligero incremento en el uso de memoria. 

Para determinar el rango dinámico $[a, b]$ de los valores de punto flotante que se transformarán en enteros, se recurre a la **calibración**: un proceso que en los pesos es directo (ya que los valores son conocidos), pero que en las activaciones requiere estrategias específicas. La **cuantización dinámica post-entrenamiento** calcula estos rangos en tiempo de ejecución, ofreciendo buena precisión pero con un _overhead_ de [Latencia](../Hardware%20-%20Fundamentos%20&%20Definiciones/Latencia.md); por el contrario, la **cuantización estática** los precalcula pasando un dataset de calibración (generalmente unos 200 ejemplos representativos) a través de observadores que registran los valores para definir los rangos mediante técnicas como **Min-Max** (ideal para pesos), **Moving Average** (ideal para activaciones) o métodos de **Histograma** basados en Entropía, Error Cuadrático Medio (MSE) o Percentiles para minimizar la pérdida de información. En escenarios de máxima exigencia, se utiliza el **Quantization Aware Training (QAT)**, que simula el error de cuantización durante el entrenamiento para que el modelo aprenda a compensarlo. El flujo de trabajo estándar para llevar un modelo a **INT8** consiste en identificar los operadores más pesados (como las multiplicaciones de matrices), probar primero la cuantización dinámica, escalar a la estática si se requiere más velocidad, y finalmente recurrir al QAT si la degradación de la precisión es inaceptable tras la conversión final de los operadores a sus contrapartes enteras.

## <font color="#de7802">4. Tipos de Cuantización en el Proyecto (RKLLM)</font>

Nuestro stack tecnológico utiliza principalmente:

- **w8a8:** Pesos en 8 bits, Activaciones en 8 bits. Máxima precisión para el Tutor de Ingeniería.
- **w4a16:** Pesos en 4 bits, Activaciones en 16 bits. Esta técnica guarda los datos en 4 bits para que viajen rápido por la RAM, pero cuando la NPU los procesa, los eleva a 16 bits para que el cálculo matemático sea más preciso.

## <font color="#de7802">5. Ventajas y Degradación</font>

1. **Velocidad:** Al reducir los bits, reducimos la cantidad de datos que deben cruzar el bus de memoria (el cuello de botella). Por lo tanto la velocidad de transporte de datos aumenta.
2. **Eficiencia Energética:** Mover menos bits consume menos energía.
3. **Pérdida de Calidad** ([Perplexity](Perplexity.md)): Cuantos menos bits usamos, más "ruido" introducimos. Un modelo INT4 puede empezar a confundir conceptos técnicos muy específicos que un modelo FP16 entendería perfectamente.