"""
agent_session.py — Sesión del agente con herramientas.

Puede correr en dos modos:
  - Standalone: AgentSession().run()
  - Intercalado: AgentSession().run_turn(user_text, return_mode)
    → ejecuta una acción y retorna al modo anterior

Maneja:
  - Clarificación antes de ejecutar
  - Tools livianas (ejecutan y retornan al modo anterior)
  - search_web (inline: ejecuta, inyecta contexto, LLM sintetiza respuesta)
  - Recordatorios pendientes al inicio de cada turno
  - Hora actual inyectada en el system prompt
"""

import datetime
import json
import re
import sys
import time
from typing import Optional

import requests
from openai import OpenAI

from .dispatcher import dispatch
from . import reminder_manager as reminders

OLLAMA_URL     = "http://localhost:11434/v1"
MODEL          = "asistente"
FALLBACK_MODEL = "qwen2.5:7b"

# Ubicación del usuario — usada para búsquedas locales.
# El LLM la recibe en el system prompt y debe incluirla en queries de search_web.
USER_LOCATION = "Córdoba Capital, Argentina"

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


# ── System prompt dinámico (con hora actual) ──────────────────────────────────

def _build_system_prompt() -> str:
    now = datetime.datetime.now()
    return (
        f"Sos un asistente personal de IA que corre localmente en una Radxa Rock 5B. "
        f"Hablás siempre en español rioplatense natural. "
        f"Usás voseo correctamente: 'podés', 'tenés', 'querés', 'hacé', 'decime', 'avisame'. "
        f"NUNCA usás 'vos' como muletilla al inicio de las oraciones. "
        f"Tu tono es directo, amigable y conciso.\n\n"
        f"Fecha y hora actual: {now.strftime('%A %d/%m/%Y %H:%M')}.\n"
        f"Ubicación del usuario: {USER_LOCATION}.\n\n"
        f"HERRAMIENTAS DISPONIBLES:\n"
        f"- task_add(title, priority?): agregá una tarea. priority: 'alta'/'media'/'baja'.\n"
        f"- task_list(): mostrá todas las tareas pendientes.\n"
        f"- task_done(task_id?, title?): marcá una tarea como completada.\n"
        f"- reminder_set(title, datetime_str): configurá un recordatorio.\n"
        f"- reminder_list(): mostrá todos los recordatorios pendientes.\n"
        f"- wa_send(contact, message): enviá un mensaje de WhatsApp.\n"
        f"- wa_read(contact?): leé mensajes de WhatsApp recientes.\n"
        f"- cal_add(title, date, start, end, description?): agregá un evento.\n"
        f"- cal_add_recurring(title, weekday, start, end, from?, until): eventos recurrentes.\n"
        f"- cal_list(date?, from?, to?): mostrá eventos del calendario.\n"
        f"- cal_delete(event_id): eliminá un evento.\n"
        f"- cal_free(date?, duration_minutes?): encontrá huecos libres.\n"
        f"- search_web(query): buscá en internet y usá los resultados para responder.\n\n"
        f"REGLAS CRÍTICAS:\n"
        f"1. Si te falta información para ejecutar una tool, preguntá ANTES de ejecutar.\n"
        f"2. Si un contacto de WhatsApp es ambiguo, mostrá la lista y preguntá.\n"
        f"3. Si fecha u hora no quedaron claras, preguntá.\n"
        f"4. Si hay ambigüedad entre tarea y evento de calendario, preguntá.\n"
        f"5. Para search_web con búsquedas locales (restaurantes, comercios, servicios, "
        f"   eventos), incluí siempre '{USER_LOCATION}' en la query. "
        f"   Ejemplo: si el usuario pide 'restaurantes chinos', la query debe ser "
        f"   'restaurantes comida china {USER_LOCATION}'.\n"
        f"6. Cuando recibas un TOOL_RESULT de search_web, leé los resultados y respondé "
        f"   directamente al usuario con la información encontrada. No volvás a llamar "
        f"   a search_web para la misma consulta.\n"
        f"7. Cuando usés una herramienta, respondés EXACTAMENTE así:\n"
        f"   TOOL_CALL: {{\"tool\": \"nombre\", \"args\": {{...}}}}\n"
        f"8. Si no necesitás herramienta, respondés directo en prosa.\n\n"
        f"EJEMPLOS DE CLARIFICACIÓN CORRECTA:\n"
        f"Usuario: 'mandá un whatsapp a Juanma' → preguntá: '¿Querés mandarle a Juan Manuel? [mostrar contactos]'\n"
        f"Usuario: 'el miércoles que viene reunión a las...' (sin hora fin) → '¿A qué hora termina?'\n"
    )


# ── Parser de TOOL_CALL ───────────────────────────────────────────────────────

def _extract_tool_call(response: str) -> Optional[dict]:
    idx = response.find("TOOL_CALL:")
    if idx == -1:
        return None
    start = response.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(response[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(response[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Modelo ────────────────────────────────────────────────────────────────────

def _resolve_model() -> str:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return MODEL if MODEL in names else FALLBACK_MODEL
    except Exception:
        return FALLBACK_MODEL


def _chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 500) -> str:
    """
    max_tokens aumentado a 500 (antes 300) para dar al LLM espacio suficiente
    para sintetizar respuestas a partir de resultados de búsqueda.
    """
    model = _resolve_model()
    res = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content.strip()


# ── Alertas de recordatorios ──────────────────────────────────────────────────

def _show_pending_alerts() -> None:
    alerts = reminders.pop_alerts()
    if not alerts:
        return
    print("\n" + "⏰" * 20)
    for a in alerts:
        dt = datetime.datetime.fromisoformat(a["remind_at"])
        print(f"  ⏰ RECORDATORIO: {a['title']} — {dt.strftime('%H:%M')}")
    print("⏰" * 20 + "\n")


# ── Sesión standalone ─────────────────────────────────────────────────────────

class AgentSession:

    def __init__(self):
        self.history = []
        self.model   = _resolve_model()

    def listen(self) -> str:
        """Hook para Whisper en producción."""
        try:
            return input("[VOS]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "salir"

    def speak(self, text: str) -> None:
        """Hook para TTS en producción."""
        print(f"\n[AGENTE]: {text}\n")

    def _process_turn(self, user_text: str) -> dict:
        """
        Procesa un turno completo incluyendo búsquedas web inline.

        Flujo para search_web:
          1. LLM genera TOOL_CALL search_web con query geolocalizada
          2. Dispatcher ejecuta la búsqueda y retorna status="ok" con
             el bloque de resultados formateado para el LLM
          3. Se inyecta como TOOL_RESULT en el historial
          4. El LLM itera una vuelta más y sintetiza la respuesta final
          5. Se retorna como respuesta normal (action="respond")

        No hay más flujo de "web_search" separado — todo es inline.

        Retorna:
            {
                "action": "respond" | "exit",
                "text": str,
                "search_data": None,   ← mantenido por compatibilidad con main.py
            }
        """
        self.history.append({"role": "user", "content": user_text})

        # Loop interno: hasta 5 rondas (una extra para síntesis post-search)
        for round_n in range(5):
            messages = [{"role": "system", "content": _build_system_prompt()}] + self.history[-12:]
            response = _chat(messages)

            tool_call = _extract_tool_call(response)

            if not tool_call:
                # Respuesta en prosa — fin del turno
                self.history.append({"role": "assistant", "content": response})
                return {"action": "respond", "text": response, "search_data": None}

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            # Registrar el TOOL_CALL en historial
            self.history.append({"role": "assistant", "content": response})

            # Ejecutar tool
            result = dispatch(tool_name, tool_args)

            if result["status"] == "clarify":
                # El dispatcher necesita más info → responder con la pregunta
                self.history.append({"role": "assistant", "content": result["question"]})
                return {"action": "respond", "text": result["question"], "search_data": None}

            if result["status"] == "error":
                error_msg = result["result"]
                self.history.append({"role": "user", "content": f"TOOL_RESULT: {error_msg}"})
                # Continuar loop para que el modelo responda el error
                continue

            # status == "ok" — tool ejecutada con éxito (incluye search_web)
            # Inyectar resultado en historial y continuar el loop.
            # Para search_web: el TOOL_RESULT contiene el bloque de resultados;
            # el LLM lo leerá en la próxima iteración y sintetizará la respuesta.
            self.history.append({
                "role":    "user",
                "content": f"TOOL_RESULT: {result['result']}"
            })

        # Si se agotaron los rounds, dar respuesta genérica
        return {"action": "respond", "text": "Procesé lo que pediste.", "search_data": None}

    def run(self) -> None:
        """Sesión standalone completa."""
        print(f"\n{'═'*60}")
        print(f"  AGENTE  |  Modelo: {self.model}")
        print(f"  'salir' → menú | 'limpiar' → nuevo contexto")
        print(f"{'═'*60}\n")
        self.speak("¡Hola! ¿Qué necesitás?")

        while True:
            _show_pending_alerts()

            user_text = self.listen()
            if not user_text:
                continue
            if user_text.lower() in ("salir", "exit", "quit"):
                self.speak("¡Hasta luego!")
                break
            if user_text.lower() in ("limpiar", "clear"):
                self.history = []
                print("  [INFO] Contexto limpiado.\n")
                continue

            result = self._process_turn(user_text)
            self.speak(result["text"])

            if len(self.history) > 24:
                self.history = self.history[-24:]

    def run_turn(self, user_text: str, return_mode: str = None) -> dict:
        """
        Ejecuta UN turno del agente desde otro modo (tutor inglés, ingeniería).
        Retorna:
            {
                "action": "respond" | "return_to_mode",
                "text": str,
                "search_data": None,
                "return_mode": str | None,
            }
        """
        _show_pending_alerts()
        result = self._process_turn(user_text)
        result["return_mode"] = return_mode

        # Siempre retornar al modo anterior — ya no hay interrupción por web_search
        result["action"] = "return_to_mode"

        return result