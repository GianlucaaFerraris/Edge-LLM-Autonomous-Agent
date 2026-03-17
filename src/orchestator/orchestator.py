"""
orchestrator.py — Orquestador principal.

Detecta el intent del usuario y despacha al modo correcto.
Maneja la interrupción limpia desde cualquier modo hacia el agente.
"""

import datetime
import json
import re
from typing import Optional

import requests
from openai import OpenAI

from .context_manager import ContextManager

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


# ── Intent detection ──────────────────────────────────────────────────────────

INTENT_PROMPT = """Clasificá el mensaje del usuario con EXACTAMENTE una palabra.

Mensaje: "{text}"

Opciones:
- english:      quiere practicar inglés, hablar en inglés, mejorar gramática
- engineering:  pregunta técnica (software, IA, física, química, electrónica, robótica)
- agent:        quiere hacer algo concreto (tarea, calendario, recordatorio, WhatsApp, búsqueda)
- unclear:      no queda claro

Respondé con exactamente una palabra: english, engineering, agent, o unclear."""

AGENT_LIGHTWEIGHT_TOOLS = {
    "task_add", "task_list", "task_done",
    "cal_add", "cal_add_recurring", "cal_list", "cal_delete", "cal_free",
    "reminder_set", "reminder_list",
    "wa_send", "wa_read",
}


def _resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def _chat_silent(messages: list[dict], max_tokens: int = 10) -> str:
    model = _resolve_model()
    try:
        res = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return res.choices[0].message.content.strip().lower()
    except Exception:
        return "unclear"


def detect_intent(text: str) -> str:
    """Detecta el intent del mensaje."""
    result = _chat_silent(
        [{"role": "user", "content": INTENT_PROMPT.format(text=text)}],
        max_tokens=5
    )
    if "english"     in result: return "english"
    if "engineering" in result: return "engineering"
    if "agent"       in result: return "agent"
    return "unclear"


def detect_intent_from_active_mode(text: str, active_mode: str) -> str:
    """
    Desde un modo activo (english/engineering), detecta si el mensaje
    es una interrupción de agente o sigue siendo parte del modo actual.
    """
    prompt = (
        f"El usuario está en modo '{active_mode}' y dijo: \"{text}\"\n\n"
        f"¿Es esto una solicitud de agente (tarea, calendario, recordatorio, WhatsApp, búsqueda web)\n"
        f"o es parte normal de la conversación del modo {active_mode}?\n\n"
        f"Respondé con exactamente una palabra: agent o continue."
    )
    result = _chat_silent([{"role": "user", "content": prompt}], max_tokens=5)
    return "agent" if "agent" in result else "continue"


# ── Greeting ──────────────────────────────────────────────────────────────────

SELECTOR_SYSTEM = (
    "Sos un asistente personal que corre localmente en una Radxa Rock 5B. "
    "Respondés siempre en español con voseo rioplatense natural. "
    "Al iniciar, saludás brevemente mencionando la hora y preguntás qué quiere hacer el usuario. "
    "Mencionás las opciones disponibles de forma natural y concisa:\n"
    "  • Practicar inglés (tutor con corrección)\n"
    "  • Consultar algo técnico (ingeniero experto)\n"
    "  • Gestionar tareas, calendario, recordatorios, WhatsApp o buscar algo\n"
    "Máximo 3 oraciones. Sin bullets. Sin markdown."
)


def generate_greeting(client, model: str) -> str:
    now = datetime.datetime.now()
    hora = now.strftime("%H:%M")
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SELECTOR_SYSTEM},
            {"role": "user",   "content": f"Son las {hora}. Iniciá la sesión."},
        ],
        temperature=0.5,
        max_tokens=120,
    )
    return res.choices[0].message.content.strip()


def generate_clarification(text: str, client, model: str) -> str:
    """Pide aclaración cuando el intent no quedó claro."""
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SELECTOR_SYSTEM},
            {"role": "user",   "content": (
                f"El usuario dijo: \"{text}\". "
                f"No quedó claro qué modo quiere. "
                f"Pedile que aclare en 1 oración, sin enumerar las opciones de nuevo."
            )},
        ],
        temperature=0.3,
        max_tokens=60,
    )
    return res.choices[0].message.content.strip()


def generate_return_prompt(mode: str, client, model: str) -> str:
    """Pregunta si el usuario quiere volver al modo anterior."""
    mode_names = {"english": "el tutor de inglés", "engineering": "el tutor de ingeniería"}
    name = mode_names.get(mode, mode)
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            f"Preguntale al usuario en 1 oración con voseo rioplatense "
            f"si quiere retomar {name} o si necesita algo más."
        )}],
        temperature=0.3,
        max_tokens=40,
    )
    return res.choices[0].message.content.strip()