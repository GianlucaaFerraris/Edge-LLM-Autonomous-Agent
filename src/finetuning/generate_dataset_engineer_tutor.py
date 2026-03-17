"""
generate_dataset_engineering_tutor.py

Genera el dataset de fine-tuning para el modo Engineering Tutor.

Filosofía: "el sabio retirado"
  - Respuestas técnicamente profundas, cálidas y sin arrogancia
  - SIN código, SIN fórmulas matemáticas, SIN cálculos numéricos
  - Conceptual puro: definiciones, comparaciones, analogías, trade-offs,
    recursos, casos de uso, errores comunes, historia del concepto
  - Adapta la profundidad al tipo de pregunta
  - Responde en el idioma de la pregunta (español/inglés)

Tipos de interacción cubiertos:
  def | compare | tradeoff | usecase | analogy | why | mistake | resource

Dominios: software, ai_ml, deep_learning, sistemas_operativos,
          bases_de_datos, electronica, telecomunicaciones,
          robotica, fisica, quimica

Uso:
    python generate_dataset_engineering_tutor.py
    python generate_dataset_engineering_tutor.py --target 20   # test
    python generate_dataset_engineering_tutor.py --target 400  # produccion

Salida: engineering_dataset.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
from tqdm import tqdm
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/v1"
MODEL      = "qwen2.5:7b"
TARGET     = 400
OUTPUT     = "engineering_dataset.jsonl"
CHECKPOINT = 40

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a brilliant, warm retired engineer and scientist with decades of experience "
    "across software, AI, electronics, physics, chemistry, and systems engineering. "
    "You explain concepts with clarity, depth, and genuine enthusiasm — like a mentor "
    "who loves sharing knowledge. "
    "You NEVER write code, never do numerical calculations, and never solve logic puzzles. "
    "Instead, you explain the intuition, the trade-offs, the history, the analogies, "
    "and the real-world implications of technical concepts. "
    "You adapt your depth to what is being asked: a definition gets a crisp explanation, "
    "a comparison gets a structured contrast, a 'why' gets philosophy and context. "
    "You are direct and concrete — no filler phrases like 'Great question!' or 'Sure!'. "
    "If asked in Spanish, answer in Spanish. If asked in English, answer in English. "
    "No bullet-point lists unless the question is explicitly a comparison or enumeration. "
    "No Chinese characters."
)

# ── Banco de preguntas ────────────────────────────────────────────────────────
# Formato: (pregunta, tipo)
# Tipos: def | compare | tradeoff | usecase | analogy | why | mistake | resource

QUESTIONS = {
    "software": [
        ("¿Qué es una lista enlazada y cuándo tiene sentido usarla?", "def"),
        ("¿Qué es un deadlock?", "def"),
        ("¿Qué es el principio de responsabilidad única?", "def"),
        ("¿Qué es la inmutabilidad en programación?", "def"),
        ("¿Qué es un sistema de tipos y para qué sirve?", "def"),
        ("¿Qué es la programación funcional?", "def"),
        ("What is eventual consistency in distributed systems?", "def"),
        ("What is a race condition?", "def"),
        ("What is idempotency and why does it matter in APIs?", "def"),
        ("What is a message queue?", "def"),
        ("¿Diferencia entre REST y GraphQL?", "compare"),
        ("¿Diferencia entre SQL y NoSQL? ¿Cuándo usar cada uno?", "compare"),
        ("¿Diferencia entre proceso e hilo?", "compare"),
        ("¿Diferencia entre arquitectura monolítica y microservicios?", "compare"),
        ("What is the difference between TCP and UDP?", "compare"),
        ("What is the difference between a stack and a heap in memory?", "compare"),
        ("What is the difference between horizontal and vertical scaling?", "compare"),
        ("¿Diferencia entre sincrónico y asincrónico en programación?", "compare"),
        ("¿Cuáles son las ventajas y desventajas de usar microservicios?", "tradeoff"),
        ("¿Qué se pierde cuando usás un ORM en vez de SQL crudo?", "tradeoff"),
        ("What are the trade-offs of using a NoSQL database?", "tradeoff"),
        ("What do you lose with eventual consistency vs strong consistency?", "tradeoff"),
        ("¿Cuándo tiene sentido usar una base de datos en grafo?", "usecase"),
        ("¿Cuándo usarías una cola de mensajes en vez de una llamada directa?", "usecase"),
        ("When would you use a binary search tree over a hash map?", "usecase"),
        ("Explicame qué es un índice de base de datos con una analogía.", "analogy"),
        ("Explicame el patrón Observer con una analogía de la vida real.", "analogy"),
        ("Explain recursion using a real-world analogy.", "analogy"),
        ("Explain the CAP theorem using an analogy.", "analogy"),
        ("¿Por qué existe Docker? ¿Qué problema resuelve realmente?", "why"),
        ("¿Por qué surgieron los microservicios? ¿Qué problema tenían los monolitos?", "why"),
        ("Why was version control invented? What problem does it solve?", "why"),
        ("¿Cuál es el error más común al diseñar una API REST?", "mistake"),
        ("¿Qué errores comete la gente cuando empieza a usar microservicios?", "mistake"),
        ("What is the most common mistake people make with database indexes?", "mistake"),
        ("¿Qué recursos recomendás para entender sistemas distribuidos?", "resource"),
        ("What books or resources would you recommend to deeply understand algorithms?", "resource"),
    ],

    "ai_ml": [
        ("¿Qué es el overfitting y por qué ocurre realmente?", "def"),
        ("¿Qué es una función de pérdida y qué papel cumple?", "def"),
        ("¿Qué es un embedding vectorial?", "def"),
        ("¿Qué es RAG y por qué es importante?", "def"),
        ("¿Qué es la cuantización de modelos?", "def"),
        ("What is gradient descent intuitively?", "def"),
        ("What is a confusion matrix?", "def"),
        ("What is transfer learning?", "def"),
        ("¿Diferencia entre supervised, unsupervised y reinforcement learning?", "compare"),
        ("¿Diferencia entre fine-tuning y prompting?", "compare"),
        ("¿Diferencia entre precisión y recall? ¿Cuándo importa más cada uno?", "compare"),
        ("What is the difference between bagging and boosting?", "compare"),
        ("What is the difference between a parameter and a hyperparameter?", "compare"),
        ("¿Cuáles son los trade-offs del fine-tuning vs RAG?", "tradeoff"),
        ("What do you lose when you quantize a model?", "tradeoff"),
        ("¿Cuándo usar un modelo de regresión vs clasificación?", "usecase"),
        ("¿Cuándo tiene sentido usar LoRA en vez de un fine-tuning completo?", "usecase"),
        ("Explicame backpropagation con una analogía.", "analogy"),
        ("Explicame la atención (attention mechanism) con una analogía.", "analogy"),
        ("Explain the vanishing gradient problem with an analogy.", "analogy"),
        ("¿Por qué surgieron los transformers? ¿Qué problema tenían los LSTM?", "why"),
        ("Why does dropout help prevent overfitting intuitively?", "why"),
        ("¿Cuál es el error más común al evaluar un modelo de ML?", "mistake"),
        ("What is the most common mistake when splitting train and test data?", "mistake"),
        ("¿Qué libros o recursos recomendás para entender ML desde los fundamentos?", "resource"),
        ("What resources would you recommend to understand transformers deeply?", "resource"),
    ],

    "deep_learning": [
        ("¿Qué es una red neuronal convolucional y por qué funciona para imágenes?", "def"),
        ("¿Qué es un autoencoder?", "def"),
        ("¿Qué es el batch normalization?", "def"),
        ("¿Qué es un GAN y cómo funciona conceptualmente?", "def"),
        ("What is a residual connection and why does it help deep networks?", "def"),
        ("What is self-supervised learning?", "def"),
        ("¿Diferencia entre CNN y Vision Transformer?", "compare"),
        ("¿Diferencia entre LSTM y Transformer para secuencias?", "compare"),
        ("What is the difference between encoder-only, decoder-only, and encoder-decoder models?", "compare"),
        ("¿Cuándo usarías un RNN y cuándo un Transformer?", "usecase"),
        ("¿Cuándo tiene sentido usar un autoencoder?", "usecase"),
        ("Explicame qué es el mecanismo de atención con una analogía.", "analogy"),
        ("Explain what a convolutional filter does using a real-world analogy.", "analogy"),
        ("¿Por qué el deep learning necesita tantos datos para funcionar bien?", "why"),
        ("¿Por qué se usan capas residuales en redes muy profundas?", "why"),
        ("¿Qué errores comete la gente al entrenar su primera red neuronal?", "mistake"),
        ("¿Qué recursos recomendás para aprender deep learning desde cero?", "resource"),
        ("What books would you recommend to understand neural networks from first principles?", "resource"),
    ],

    "sistemas_operativos": [
        ("¿Qué es un sistema operativo y qué problema resuelve realmente?", "def"),
        ("¿Qué es la memoria virtual?", "def"),
        ("¿Qué es un semáforo en SO?", "def"),
        ("¿Qué es el scheduler de un SO?", "def"),
        ("¿Qué es una syscall?", "def"),
        ("What is a context switch?", "def"),
        ("What is the difference between a process and a thread at the OS level?", "compare"),
        ("¿Diferencia entre memoria heap y stack desde la perspectiva del SO?", "compare"),
        ("¿Diferencia entre un kernel monolítico y un microkernel?", "compare"),
        ("What are the trade-offs of preemptive vs cooperative scheduling?", "tradeoff"),
        ("¿Cuándo tiene sentido usar hilos en vez de procesos?", "usecase"),
        ("Explicame la memoria virtual con una analogía.", "analogy"),
        ("Explain what a mutex is using a real-world analogy.", "analogy"),
        ("¿Por qué existe la memoria virtual? ¿Qué problema resuelve?", "why"),
        ("¿Cuál es el error más común al programar con múltiples hilos?", "mistake"),
        ("¿Qué libros recomendás para entender cómo funciona un SO por dentro?", "resource"),
    ],

    "bases_de_datos": [
        ("¿Qué es una transacción en bases de datos?", "def"),
        ("¿Qué es ACID?", "def"),
        ("¿Qué es una clave foránea?", "def"),
        ("¿Qué es un índice y cómo funciona internamente?", "def"),
        ("¿Qué es el teorema CAP?", "def"),
        ("What is database normalization?", "def"),
        ("What is the difference between OLAP and OLTP databases?", "compare"),
        ("¿Diferencia entre un índice B-tree y un hash index?", "compare"),
        ("¿Diferencia entre una base de datos relacional y una documental?", "compare"),
        ("¿Diferencia entre una base de datos en grafo y una relacional?", "compare"),
        ("What are the trade-offs of denormalization?", "tradeoff"),
        ("¿Qué se gana y se pierde al particionar una base de datos?", "tradeoff"),
        ("¿Cuándo usarías una base de datos de series temporales?", "usecase"),
        ("¿Cuándo tiene sentido usar Redis en vez de una base de datos relacional?", "usecase"),
        ("Explicame qué es ACID con una analogía bancaria.", "analogy"),
        ("¿Por qué existe el modelo relacional? ¿Qué problema resolvió Codd?", "why"),
        ("¿Cuál es el error más común al diseñar un esquema de base de datos?", "mistake"),
        ("¿Qué recursos recomendás para entender bases de datos en profundidad?", "resource"),
    ],

    "electronica": [
        ("¿Qué es PWM y cómo se usa para controlar la velocidad de un motor?", "def"),
        ("¿Qué es un ADC y un DAC?", "def"),
        ("¿Qué es un optoacoplador y para qué sirve?", "def"),
        ("¿Qué es un regulador LDO?", "def"),
        ("What is a pull-up resistor and why do you need one?", "def"),
        ("What is a bypass capacitor?", "def"),
        ("¿Diferencia entre microcontrolador y microprocesador?", "compare"),
        ("¿Diferencia entre I2C y SPI? ¿Cuándo usar cada uno?", "compare"),
        ("What is the difference between a MOSFET and a BJT as a switch?", "compare"),
        ("¿Cuándo usarías I2C sobre SPI en un diseño real?", "usecase"),
        ("¿Cuándo tiene sentido usar un optoacoplador?", "usecase"),
        ("Explicame cómo funciona un transistor como interruptor con una analogía.", "analogy"),
        ("¿Por qué los circuitos digitales modernos operan a voltajes tan bajos?", "why"),
        ("¿Cuál es el error más común al conectar periféricos I2C?", "mistake"),
        ("¿Qué recursos recomendás para aprender electrónica analógica y digital?", "resource"),
    ],

    "telecomunicaciones": [
        ("¿Qué es el modelo OSI y por qué tiene sentido dividirlo en capas?", "def"),
        ("¿Qué es la latencia y cómo se diferencia del ancho de banda?", "def"),
        ("¿Qué es QoS en redes?", "def"),
        ("¿Qué es MIMO en comunicaciones inalámbricas?", "def"),
        ("¿Qué es la modulación y para qué sirve?", "def"),
        ("What is packet switching and why did it replace circuit switching?", "def"),
        ("¿Diferencia entre WiFi y LoRa para IoT?", "compare"),
        ("¿Diferencia entre latencia y throughput?", "compare"),
        ("What is the difference between HTTP and WebSocket?", "compare"),
        ("¿Diferencia entre 4G y 5G en términos conceptuales?", "compare"),
        ("¿Cuándo usarías MQTT en vez de HTTP para IoT?", "usecase"),
        ("¿Cuándo tiene sentido usar LoRa en vez de WiFi?", "usecase"),
        ("Explicame cómo funciona el DNS con una analogía.", "analogy"),
        ("Explain how TCP ensures reliable delivery using an analogy.", "analogy"),
        ("¿Por qué se inventó el modelo en capas para las redes?", "why"),
        ("¿Por qué existe UDP si es menos confiable que TCP?", "why"),
        ("¿Cuál es el error más común al diagnosticar problemas de red?", "mistake"),
        ("¿Qué recursos recomendás para entender redes desde los fundamentos?", "resource"),
    ],

    "robotica": [
        ("¿Qué es la cinemática directa e inversa?", "def"),
        ("¿Qué es SLAM en robótica?", "def"),
        ("¿Qué es la odometría?", "def"),
        ("¿Qué es ROS y qué problema resuelve?", "def"),
        ("What is path planning in robotics?", "def"),
        ("¿Diferencia entre sensor LIDAR y cámara RGB-D?", "compare"),
        ("¿Diferencia entre un robot reactivo y uno deliberativo?", "compare"),
        ("¿Diferencia entre un servo motor y un stepper motor?", "compare"),
        ("¿Cuándo usarías un filtro de Kalman?", "usecase"),
        ("¿Cuándo tiene sentido usar LIDAR sobre cámaras?", "usecase"),
        ("Explicame el control PID con una analogía de la vida diaria.", "analogy"),
        ("Explain SLAM using a real-world analogy.", "analogy"),
        ("¿Por qué es tan difícil la cinemática inversa para brazos robóticos?", "why"),
        ("¿Cuál es el error más común al implementar un controlador PID?", "mistake"),
        ("¿Qué recursos recomendás para aprender robótica desde cero?", "resource"),
    ],

    "fisica": [
        ("¿Qué es la entropía intuitivamente?", "def"),
        ("¿Qué es la impedancia?", "def"),
        ("¿Qué es el efecto Hall?", "def"),
        ("¿Qué es la transformada de Fourier conceptualmente?", "def"),
        ("¿Qué es la resonancia?", "def"),
        ("What is electromagnetic induction?", "def"),
        ("What is the photoelectric effect?", "def"),
        ("¿Diferencia entre calor y temperatura?", "compare"),
        ("¿Diferencia entre corriente alterna y continua?", "compare"),
        ("¿Diferencia entre campo eléctrico y magnético?", "compare"),
        ("¿Cuándo importa la impedancia de salida de una fuente?", "usecase"),
        ("Explicame la transformada de Fourier con una analogía musical.", "analogy"),
        ("Explain electromagnetic waves using a real-world analogy.", "analogy"),
        ("Explain entropy using an analogy.", "analogy"),
        ("¿Por qué la relatividad especial dice que nada puede superar la velocidad de la luz?", "why"),
        ("¿Por qué la mecánica cuántica es tan contraintuitiva?", "why"),
        ("¿Qué recursos recomendás para entender física de forma intuitiva?", "resource"),
        ("What books made physics click for you as a teacher?", "resource"),
    ],

    "quimica": [
        ("¿Qué es la electronegatividad y cómo afecta los enlaces?", "def"),
        ("¿Qué es un electrolito?", "def"),
        ("¿Qué es la entalpía de reacción?", "def"),
        ("¿Qué es la oxidación y la reducción conceptualmente?", "def"),
        ("What is a catalyst and how does it work?", "def"),
        ("What is pH intuitively?", "def"),
        ("¿Diferencia entre enlace covalente y iónico?", "compare"),
        ("¿Diferencia entre reacción exotérmica y endotérmica?", "compare"),
        ("¿Cuándo importa la pureza de un electrolito en una batería?", "usecase"),
        ("Explicame la electronegatividad con una analogía.", "analogy"),
        ("Explain how a battery works conceptually using an analogy.", "analogy"),
        ("¿Por qué el agua es tan buena disolvente?", "why"),
        ("¿Por qué las baterías de litio son tan densas energéticamente?", "why"),
        ("¿Qué recursos recomendás para entender química de forma intuitiva?", "resource"),
    ],
}

# ── Follow-ups por tipo ───────────────────────────────────────────────────────
# Formato: (español, english)
FOLLOWUPS_BY_TYPE = {
    "def": [
        ("¿Podés darme un ejemplo concreto de cuándo esto importa en la práctica?", "Can you give me a concrete example of when this matters in practice?"),
        ("¿Cuál es el malentendido más común sobre este concepto?", "What is the most common misconception about this concept?"),
        ("¿Cómo lo explicarías a alguien sin background técnico?", "How would you explain this to someone with no technical background?"),
        ("¿Qué pasa cuando se ignora esto en un sistema real?", "What goes wrong when this is ignored in a real system?"),
    ],
    "compare": [
        ("¿En qué escenario elegirías el segundo sobre el primero?", "In what scenario would you choose the second over the first?"),
        ("¿Hay algún caso donde ambos conviven en el mismo sistema?", "Is there a case where both coexist in the same system?"),
        ("¿Cuál de los dos tiende a producir errores más difíciles de detectar?", "Which one tends to produce harder-to-find bugs or problems?"),
    ],
    "tradeoff": [
        ("¿Qué señales te dirían que elegiste mal?", "What signs would tell you that you made the wrong choice?"),
        ("¿Hay alguna alternativa que evite ambos extremos?", "Is there an alternative that avoids both extremes?"),
        ("¿Cómo sabrías que el trade-off vale la pena en un proyecto real?", "How would you know the trade-off is worth it in a real project?"),
    ],
    "usecase": [
        ("¿Y cuándo definitivamente NO lo usarías?", "And when would you definitely NOT use it?"),
        ("¿Qué señales en los requerimientos te harían pensar en esta solución?", "What signals in the requirements would make you think of this solution?"),
        ("¿Qué pasa si lo usás fuera de su caso ideal?", "What happens if you use it outside its ideal case?"),
    ],
    "analogy": [
        ("¿Dónde se rompe la analogía?", "Where does the analogy break down?"),
        ("¿Tenés otra analogía que capture un aspecto distinto?", "Do you have another analogy that captures a different aspect?"),
    ],
    "why": [
        ("¿Qué problemas nuevos introdujo la solución?", "What new problems did the solution introduce?"),
        ("¿Cómo evolucionó esto desde sus inicios?", "How has this evolved since its beginnings?"),
    ],
    "mistake": [
        ("¿Cómo detectarías ese error antes de que llegue a producción?", "How would you catch that mistake before it reaches production?"),
        ("¿Qué hábito o práctica evitaría ese error?", "What habit or practice would prevent that mistake?"),
    ],
    "resource": [
        ("¿Por dónde empezarías si solo tenés una hora por día?", "Where would you start if you only have one hour a day?"),
        ("¿Qué camino seguirías para pasar de intermedio a experto en esto?", "What path would you follow to go from intermediate to expert?"),
    ],
}

FOLLOWUPS_GENERIC = [
    ("¿Podés profundizar en la parte que más confunde a la gente?", "Can you go deeper on the part that confuses people the most?"),
    ("¿Cómo se relaciona esto con otros conceptos del área?", "How does this relate to other concepts in the field?"),
]


def _is_spanish(text: str) -> bool:
    return bool(re.search(r'[áéíóúüñ¿¡]', text)) or \
           any(w in text.lower() for w in
               ["qué", "cómo", "cuál", "diferencia", "explicame",
                "para qué", "cuándo", "por qué", "podés", "tenés"])


def _call(messages: list, temperature: float = 0.5, max_tokens: int = 500) -> str:
    try:
        res = client.chat.completions.create(
            model=MODEL, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        text = res.choices[0].message.content or ""
        if re.search(r'[\u4e00-\u9fff]', text):
            return ""
        return text.strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""


def generate_answer(question: str, qtype: str) -> str:
    lang = "Spanish" if _is_spanish(question) else "English"

    type_hints = {
        "def":      "Give a clear, deep definition. Explain the intuition behind it, not just what it is. Include why it matters.",
        "compare":  "Structure the comparison clearly. Focus on the conceptual differences and when each shines. Avoid tables or bullet lists.",
        "tradeoff": "Explain both sides honestly. When does each side hurt you? Give a concrete scenario for each.",
        "usecase":  "Describe the scenario where this makes sense. What signals in a problem lead you to this choice?",
        "analogy":  "Build one strong analogy from everyday life. Explain it fully. Then note where the analogy breaks down.",
        "why":      "Explain the historical or conceptual motivation. What problem existed before? What did this solve?",
        "mistake":  "Describe the mistake vividly. Why does it happen? What does it look like in practice? How do you avoid it?",
        "resource": "Recommend specific books, courses, or authors by name. Explain why each one is valuable and in what order to tackle them.",
    }

    hint = type_hints.get(qtype, "Answer with depth and clarity.")
    prompt = (
        f"A student asks: '{question}'\n\n"
        f"Instruction: {hint}\n\n"
        f"Answer in {lang}. Be the wise retired mentor. "
        f"Do NOT write code. Do NOT do numerical calculations. "
        f"Do NOT name specific syntax, keywords, or commands — explain concepts, not syntax. "
        f"Prose only. No numbered lists. No bullet lists unless the question explicitly asks for a comparison. "
        f"No bold headers. No filler opener. Start directly with the answer. Under 250 words."
    )
    return _call([
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ])


def generate_followup(question: str, answer: str, qtype: str) -> tuple[str, str]:
    is_es   = _is_spanish(question)
    pool    = FOLLOWUPS_BY_TYPE.get(qtype, FOLLOWUPS_GENERIC)
    fu_es, fu_en = random.choice(pool)
    followup = fu_es if is_es else fu_en
    lang     = "Spanish" if is_es else "English"

    prompt = (
        f"Previous question: '{question}'\n"
        f"Your answer: '{answer}'\n\n"
        f"Student follow-up: '{followup}'\n\n"
        f"Answer the follow-up in {lang}. Stay deep and conceptual. "
        f"No code, no calculations, no syntax names. No numbered lists, no bold headers. "
        f"No filler opener. Under 200 words."
    )
    fa = _call([
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ])
    return followup, fa


# ── QC ────────────────────────────────────────────────────────────────────────
CODE_PATTERN   = re.compile(r'```|`\w|\bdef |\bclass |\bimport |for \w+ in |while True|print\(|==>|<->|train_test_split|sklearn\.|torch\.|tf\.')
CALC_PATTERN   = re.compile(r'\b\d+\s*[+\-*/]\s*\d+\b')
FILLER_PATTERN = re.compile(
    r'^(sure[,!]|great question|absolutely[,!]|of course[,!]|certainly[,!]'
    r'|claro[,!]|por supuesto[,!]|desde luego[,!])',
    re.IGNORECASE)
SIGNOFF_PATTERN = re.compile(
    r'(espero que (esta|esto|este)|hope (this|that) helps|espero haber)',
    re.IGNORECASE)


def quality_check(msgs: list, qtype: str, domain: str = "") -> tuple[bool, str]:
    for m in msgs:
        if not m.get("content") or len(m["content"]) < 20:
            return False, f"Empty/short: role={m['role']}"
        if re.search(r'[\u4e00-\u9fff]', m["content"]):
            return False, "Chinese characters"

    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    for m in assistant_msgs:
        c = m["content"]
        if len(c) < 100:
            return False, f"Answer too short ({len(c)} chars)"
        if CODE_PATTERN.search(c):
            return False, "Contains code syntax"
        if CALC_PATTERN.search(c):
            return False, "Contains numerical calculation"
        if FILLER_PATTERN.search(c):
            return False, "Starts with filler phrase"
        # No listas de bullets puras (salvo compare y resource)
        bullet_lines = sum(1 for line in c.splitlines()
                           if line.strip().startswith(("- ", "* ", "• ")))
        # Detectar listas numeradas con bold: "1. **X**" o "1. X"
        numbered_headers = sum(1 for line in c.splitlines()
                               if re.match(r'^\s*\d+\.\s', line.strip()))
        list_count = bullet_lines + numbered_headers
        if list_count > 2 and qtype not in ("compare", "resource"):
            return False, f"Too many list items ({list_count}) for type '{qtype}'"

    # ── Domain sanity: telecomunicaciones no debería hablar de DL ──────────────
    DL_TERMS = re.compile(
        r'\b(relu|backprop|gradient descent|epoch|batch size|'
        r'red neuronal artificial|neural network|overfitting|loss function)\b',
        re.IGNORECASE)
    if domain == "telecomunicaciones":
        for m in assistant_msgs:
            if DL_TERMS.search(m["content"]):
                return False, "telecomunicaciones answer contains DL terminology"

    return True, "OK"


def _save(examples: list) -> None:
    with open(OUTPUT, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(target: int) -> None:
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        print("[OK] Ollama conectado")
    except Exception:
        print("[ERROR] Ollama no está corriendo."); sys.exit(1)

    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)
        print(f"[INFO] Output anterior eliminado: {OUTPUT}")

    pool = []
    for domain, qs in QUESTIONS.items():
        for question, qtype in qs:
            pool.append((domain, question, qtype))

    total_unique = len(pool)
    print(f"[INFO] Pool: {total_unique} preguntas únicas en {len(QUESTIONS)} dominios")

    random.shuffle(pool)
    full_pool = list(pool)
    while len(full_pool) < target:
        extra = list(pool)
        random.shuffle(extra)
        full_pool.extend(extra)

    passed, failed, attempts = [], 0, 0
    q_idx = 0
    domain_counts, type_counts, turn_counts = {}, {}, {1: 0, 2: 0}

    with tqdm(total=target, desc="engineering") as pbar:
        while len(passed) < target:
            attempts += 1
            domain, question, qtype = full_pool[q_idx % len(full_pool)]
            q_idx += 1

            answer = generate_answer(question, qtype)
            if not answer:
                failed += 1; continue

            has_followup = random.random() < 0.55
            n_turns = 1
            if has_followup:
                followup, fa = generate_followup(question, answer, qtype)
                if fa:
                    msgs = [
                        {"role": "system",    "content": SYSTEM},
                        {"role": "user",      "content": question},
                        {"role": "assistant", "content": answer},
                        {"role": "user",      "content": followup},
                        {"role": "assistant", "content": fa},
                    ]
                    n_turns = 2
                else:
                    msgs = [
                        {"role": "system",    "content": SYSTEM},
                        {"role": "user",      "content": question},
                        {"role": "assistant", "content": answer},
                    ]
            else:
                msgs = [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": answer},
                ]

            ok, reason = quality_check(msgs, qtype, domain)
            if not ok:
                failed += 1; continue

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            type_counts[qtype]    = type_counts.get(qtype, 0) + 1
            turn_counts[n_turns]  = turn_counts.get(n_turns, 0) + 1

            ex = {
                "messages": msgs,
                "_debug": {
                    "mode":      "engineering",
                    "domain":    domain,
                    "qtype":     qtype,
                    "question":  question,
                    "turns":     n_turns,
                    "qc_reason": reason,
                }
            }
            passed.append(ex)
            pbar.update(1)
            pbar.set_postfix(fail=failed, dom=domain[:6], qtype=qtype[:5],
                             rate=f"{len(passed)/attempts*100:.0f}%")

            if len(passed) % CHECKPOINT == 0:
                _save(passed[-CHECKPOINT:])
                print(f"\n  [CHECKPOINT] {len(passed)}/{target} guardados")

            if attempts > target * 6:
                print(f"\n[WARN] Parando con {len(passed)}")
                break

    remainder = len(passed) % CHECKPOINT
    if remainder:
        _save(passed[-remainder:])

    print(f"\n{'='*58}")
    print(f"  Passed:        {len(passed)}")
    print(f"  Failed:        {failed}")
    print(f"  Success rate:  {len(passed)/max(attempts,1)*100:.1f}%")
    print(f"  Output:        {OUTPUT}")
    print(f"\n  Por dominio:")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {d:<30s} {c}")
    print(f"\n  Por tipo:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<12s} {c}")
    print(f"\n  1 turno: {turn_counts.get(1,0)} | 2 turnos: {turn_counts.get(2,0)}")
    print(f"{'='*58}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset del Engineering Tutor — sabio retirado"
    )
    parser.add_argument("--target", type=int, default=TARGET,
                        help=f"Numero de ejemplos (default: {TARGET})")
    args = parser.parse_args()
    main(args.target)