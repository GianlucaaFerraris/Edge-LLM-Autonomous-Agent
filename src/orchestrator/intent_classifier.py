"""
intent_classifier.py — Clasificador unificado de intent.

Reemplaza la doble clasificación (orchestrator.detect_intent +
detect_intent_from_active_mode + tutor.classify_intent) con un
ÚNICO call al LLM que recibe el contexto completo:
  - modo activo actual
  - tema actual (si aplica)
  - mensaje del usuario

Retorna un intent estructurado que el caller consume directamente,
sin necesidad de re-clasificar.

Diseño:
  - Un solo prompt parametrizado por modo activo.
  - El prompt lista SOLO los intents válidos para ese modo.
  - Respuesta JSON con action + confidence + metadata.
  - Fallback seguro: si el LLM falla, retorna "continue" (no rompe nada).
  - Sin keyword matching hardcodeado (excepto "salir"/"exit" como safety net).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
from openai import OpenAI

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


def _resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def _llm_json(prompt: str, max_tokens: int = 100) -> dict:
    """Llama al LLM y parsea JSON. Fallback seguro."""
    try:
        res = client.chat.completions.create(
            model=_resolve_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = res.choices[0].message.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        return json.loads(raw)
    except Exception:
        return {"action": "continue", "confidence": "high"}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    action:     str               # ver acciones válidas por modo abajo
    confidence: str = "high"      # "high" | "low"
    topic:      str = ""          # para propose_topic / change_topic
    question:   str = ""          # pregunta de clarificación si confidence=low
    raw:        dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts por modo
# ─────────────────────────────────────────────────────────────────────────────

# Acciones válidas por modo:
#   idle:        english, engineering, agent, unclear
#   english:     respond, change_topic, propose_topic, exit_mode,
#                switch_engineering, agent, exit_app
#   engineering: respond, exit_mode, switch_english, agent, exit_app
#   agent:       (el agente tiene su propio pipeline de tools)

_PROMPT_IDLE = """Clasificá el mensaje del usuario con EXACTAMENTE una de estas opciones.

Mensaje: "{text}"

Opciones:
- "english":      quiere practicar inglés, hablar en inglés, mejorar gramática, tutor de inglés
- "engineering":  pregunta técnica, quiere el tutor de ingeniería, consultar algo de ciencia/tech
- "agent":        quiere hacer algo concreto: tarea, calendario, recordatorio, WhatsApp, búsqueda web
- "unclear":      no queda claro qué quiere

Respondé SOLO con JSON válido, sin markdown:
{{"action": "...", "confidence": "high"}}
"""

_PROMPT_ENGLISH = """El usuario está practicando inglés con un tutor conversacional.
Tema ACTUAL de la conversación: "{topic}"
Mensaje del usuario: "{text}"

Tu tarea: clasificar QUÉ QUIERE HACER el usuario.

Opciones:
- "respond":             está respondiendo o continuando la conversación en inglés.
                         Incluye cualquier respuesta sustantiva a la pregunta del tutor,
                         aunque mencione el tema actual u otros temas relacionados.
- "change_topic":        quiere cambiar a un tema ALEATORIO. Frase corta y directa:
                         "otro tema", "cambiemos", "next topic", "algo diferente".
- "propose_topic":       en una frase CORTA y DIRECTA pide practicar un tema NUEVO y DISTINTO
                         al actual. Ej: "vamos con deportes", "quiero practicar sobre tecnología".
                         NO aplica si el usuario está RESPONDIENDO aunque su respuesta mencione topics.
- "switch_engineering":  quiere DEJAR el inglés e ir al TUTOR DE INGENIERÍA.
- "agent":               quiere una herramienta del agente (tarea, calendario, recordatorio, WhatsApp).
- "exit_mode":           quiere salir del tutor de inglés y volver al menú.
- "exit_app":            quiere cerrar la aplicación completamente.

REGLA CRÍTICA — respond vs propose_topic:
"propose_topic" SOLO aplica cuando el mensaje ES una solicitud de cambio de tema,
es decir, una instrucción corta y directa del tipo "quiero practicar X" o "hablemos de Y".
Si el mensaje es una respuesta elaborada (más de ~15 palabras) que desarrolla ideas
sobre el tema actual o cualquier otro tema, es SIEMPRE "respond", sin excepción.
Un mensaje largo nunca es propose_topic.

EJEMPLOS respond (respuestas elaboradas → SIEMPRE respond):
- "I think ancient ruins are important because they show us how civilizations..." → respond
- "In my opinion, social media has both positive and negative effects on..." → respond
- "I would say that climate change is one of the most important issues..." → respond
- "History is fascinating because we can learn from past mistakes and..." → respond

EJEMPLOS propose_topic (solicitudes cortas y directas de cambio):
- "quiero practicar sobre robótica" → propose_topic
- "let's talk about sports" → propose_topic
- "cambiemos a tecnología" → propose_topic

EJEMPLOS switch_engineering (pide salir del inglés):
- "pasame al tutor de ingeniería" → switch_engineering
- "quiero consultar algo técnico" → switch_engineering
- "modo ingeniería" → switch_engineering

Respondé SOLO con JSON válido:
{{"action": "...", "confidence": "high"}}

Si action es "propose_topic", agregá el campo "topic" con el tema extraído (1-6 palabras en inglés):
{{"action": "propose_topic", "confidence": "high", "topic": "..."}}

Si no estás seguro, usá confidence "low" y agregá "question" con una pregunta corta en español:
{{"action": "...", "confidence": "low", "question": "..."}}
"""

_PROMPT_ENGINEERING = """El usuario está en el tutor de ingeniería (consultas técnicas en español).
Mensaje del usuario: "{text}"

Clasificá QUÉ QUIERE HACER:

- "respond":          está haciendo una pregunta técnica o continuando la conversación
- "switch_english":   quiere SALIR de ingeniería e ir a PRACTICAR INGLÉS
                      (dice "quiero practicar inglés", "pasame al inglés",
                      "tutor de inglés", "english mode")
- "agent":            quiere una herramienta (tarea, calendario, recordatorio, WhatsApp, búsqueda web)
- "exit_mode":        quiere salir al menú principal
- "exit_app":         quiere cerrar la aplicación

IMPORTANTE: Si el usuario pregunta algo técnico sobre inglés (gramática, verbos,
etc.) como consulta de ingeniería, eso es "respond", no "switch_english".

Respondé SOLO con JSON válido:
{{"action": "...", "confidence": "high"}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Safety net — solo para los casos más obvios
# ─────────────────────────────────────────────────────────────────────────────

_EXIT_APP_WORDS = {"chau", "cerrar", "quit"}
# "salir" y "exit" son exit_mode cuando hay un modo activo, exit_app solo desde idle
_EXIT_GENERIC = {"salir", "exit"}
_EXIT_MODE_WORDS = {"menu", "menú", "volver al menú", "volver al menu", "stop"}


def _safety_check(text: str, mode: str) -> Optional[IntentResult]:
    """
    Chequeo rápido sin LLM para los casos triviales.
    Retorna None si no matchea (→ se usa el LLM).
    """
    t = text.lower().strip()

    if mode == "idle":
        if t in _EXIT_APP_WORDS or t in _EXIT_GENERIC:
            return IntentResult(action="exit_app", confidence="high")
        return None

    # En un modo activo
    if t in _EXIT_APP_WORDS:
        return IntentResult(action="exit_app", confidence="high")
    if t in _EXIT_MODE_WORDS or t in _EXIT_GENERIC:
        return IntentResult(action="exit_mode", confidence="high")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def classify_idle(text: str) -> IntentResult:
    """Clasifica intent cuando no hay modo activo."""
    safe = _safety_check(text, "idle")
    if safe:
        return safe

    prompt = _PROMPT_IDLE.format(text=text)
    data = _llm_json(prompt, max_tokens=30)
    action = data.get("action", "unclear")
    if action not in ("english", "engineering", "agent", "unclear"):
        action = "unclear"
    return IntentResult(
        action=action,
        confidence=data.get("confidence", "high"),
        question=data.get("question", ""),
        raw=data,
    )


def classify_english(text: str, current_topic: str = "") -> IntentResult:
    """Clasifica intent dentro del tutor de inglés."""
    safe = _safety_check(text, "english")
    if safe:
        return safe

    # Heurística de longitud: respuestas largas (≥20 palabras) son casi
    # siempre "respond". propose_topic y change_topic son frases cortas y
    # directas. Esto evita que el LLM malinterprete respuestas elaboradas
    # sobre el tema actual como solicitudes de cambio de tema.
    word_count = len(text.split())
    if word_count >= 20:
        return IntentResult(action="respond", confidence="high")

    prompt = _PROMPT_ENGLISH.format(text=text, topic=current_topic)
    data = _llm_json(prompt, max_tokens=60)
    action = data.get("action", "respond")
    valid = {"respond", "change_topic", "propose_topic",
             "switch_engineering", "agent", "exit_mode", "exit_app"}
    if action not in valid:
        action = "respond"

    # Sanity check: propose_topic y change_topic no deberían venir de
    # respuestas largas, aunque el LLM los clasifique así.
    if action in ("propose_topic", "change_topic") and word_count >= 12:
        action = "respond"

    return IntentResult(
        action=action,
        confidence=data.get("confidence", "high"),
        topic=data.get("topic", ""),
        question=data.get("question", ""),
        raw=data,
    )


def classify_engineering(text: str) -> IntentResult:
    """Clasifica intent dentro del tutor de ingeniería."""
    safe = _safety_check(text, "engineering")
    if safe:
        return safe

    prompt = _PROMPT_ENGINEERING.format(text=text)
    data = _llm_json(prompt, max_tokens=40)
    action = data.get("action", "respond")
    valid = {"respond", "switch_english", "agent", "exit_mode", "exit_app"}
    if action not in valid:
        action = "respond"

    return IntentResult(
        action=action,
        confidence=data.get("confidence", "high"),
        question=data.get("question", ""),
        raw=data,
    )


def confirm_intent(question: str, user_answer: str) -> bool:
    """
    Cuando el clasificador retorna confidence=low, se le hace la pregunta
    al usuario. Esta función interpreta si la respuesta confirma o niega.
    """
    ans = user_answer.lower().strip()

    CONFIRM = {"sí", "si", "dale", "yes", "claro", "obvio", "exacto",
               "correcto", "eso", "ok", "bueno", "confirmo"}
    DENY = {"no", "nope", "nel", "para nada", "no es eso", "nah", "no quiero"}

    if ans in CONFIRM:
        return True
    if ans in DENY:
        return False

    # Ambiguo → LLM
    prompt = (
        f"El asistente preguntó: \"{question}\"\n"
        f"El usuario respondió: \"{user_answer}\"\n\n"
        f"¿El usuario confirmó o negó? Respondé exactamente: confirma o niega."
    )
    try:
        res = client.chat.completions.create(
            model=_resolve_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        raw = res.choices[0].message.content.strip().lower()
        return "confirma" in raw
    except Exception:
        return False