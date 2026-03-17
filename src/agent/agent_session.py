"""
agent_session.py — Sesión del agente con herramientas.

Puede correr en dos modos:
  - Standalone: AgentSession().run()
  - Intercalado: AgentSession().run_turn(user_text, return_mode)
    → ejecuta una acción y retorna al modo anterior

Maneja:
  - Clarificación antes de ejecutar
  - Tools livianas (ejecutan y retornan al modo anterior)
  - search_web (pesada: pregunta antes de abandonar la sesión actual)
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
        f"Fecha y hora actual: {now.strftime('%A %d/%m/%Y %H:%M')}.\n\n"
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
        f"- search_web(query): buscá en internet (abandona la sesión actual).\n\n"
        f"REGLAS CRÍTICAS:\n"
        f"1. Si te falta información para ejecutar una tool, preguntá ANTES de ejecutar.\n"
        f"2. Si un contacto de WhatsApp es ambiguo, mostrá la lista y preguntá.\n"
        f"3. Si fecha u hora no quedaron claras, preguntá.\n"
        f"4. Si hay ambigüedad entre tarea y evento de calendario, preguntá.\n"
        f"5. search_web es especial: avisá que va a interrumpir la sesión actual.\n"
        f"6. Cuando usés una herramienta, respondés EXACTAMENTE así:\n"
        f"   TOOL_CALL: {{\"tool\": \"nombre\", \"args\": {{...}}}}\n"
        f"7. Si no necesitás herramienta, respondés directo en prosa.\n\n"
        f"EJEMPLOS DE CLARIFICACIÓN CORRECTA:\n"
        f"Usuario: 'mandá un whatsapp a Juanma' → preguntá: '¿Querés mandarle a Juan Manuel? [mostrar contactos]'\n"
        f"Usuario: 'el miércoles que viene reunión a las...' (sin hora fin) → '¿A qué hora termina?'\n"
        f"Usuario: 'buscame X' → '¿Estás seguro? Buscar en internet va a pausar la sesión actual.'\n"
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


def _chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 300) -> str:
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
        Procesa un turno completo.
        Retorna:
            {
                "action": "respond" | "web_search" | "exit",
                "text": str,
                "search_data": dict | None,
            }
        """
        self.history.append({"role": "user", "content": user_text})

        # Loop interno: hasta 4 rondas de tool calls
        for _ in range(4):
            messages = [{"role": "system", "content": _build_system_prompt()}] + self.history[-10:]
            response = _chat(messages)

            tool_call = _extract_tool_call(response)

            if not tool_call:
                # Respuesta en prosa — fin del turno
                self.history.append({"role": "assistant", "content": response})
                return {"action": "respond", "text": response, "search_data": None}

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            # Ejecutar tool
            result = dispatch(tool_name, tool_args)

            if result["status"] == "clarify":
                # El dispatcher necesita más info → responder con la pregunta
                self.history.append({"role": "assistant", "content": response})
                self.history.append({"role": "assistant", "content": result["question"]})
                return {"action": "respond", "text": result["question"], "search_data": None}

            if result["status"] == "web_search":
                # Tool pesada — retornar para que el orquestador confirme
                self.history.append({"role": "assistant", "content": response})
                return {
                    "action":      "web_search",
                    "text":        result["result"],
                    "search_data": result["data"],
                }

            if result["status"] == "error":
                error_msg = result["result"]
                self.history.append({"role": "assistant", "content": response})
                self.history.append({"role": "user", "content": f"TOOL_RESULT: {error_msg}"})
                # Continuar loop para que el modelo responda el error
                continue

            # Tool ejecutada con éxito — inyectar resultado y continuar
            self.history.append({"role": "assistant", "content": response})
            self.history.append({"role": "user", "content": f"TOOL_RESULT: {result['result']}"})

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

            if result["action"] == "web_search":
                self.speak(result["text"])
                confirm = self.listen()
                if confirm.lower() in ("sí", "si", "dale", "yes", "sí", "confirmo"):
                    self._enter_web_mode(result["search_data"])
                else:
                    self.speak("Entendido, cancelé la búsqueda. ¿Qué más necesitás?")
            else:
                self.speak(result["text"])

            if len(self.history) > 20:
                self.history = self.history[-20:]

    def run_turn(self, user_text: str, return_mode: str = None) -> dict:
        """
        Ejecuta UN turno del agente desde otro modo (tutor inglés, ingeniería).
        Retorna:
            {
                "action": "respond" | "web_search" | "return_to_mode",
                "text": str,
                "search_data": dict | None,
                "return_mode": str | None,
            }
        """
        _show_pending_alerts()
        result = self._process_turn(user_text)
        result["return_mode"] = return_mode

        if result["action"] == "respond":
            result["action"] = "return_to_mode"

        return result

    def _enter_web_mode(self, search_data: dict) -> None:
        """
        Modo tutor con contexto de búsqueda web.
        El agente usa los resultados como base para responder preguntas.
        """
        context = search_data["search_result"]["context"]
        query   = search_data["query"]

        web_system = (
            f"Sos un asistente que acaba de buscar en internet sobre: \"{query}\".\n"
            f"Estos son los resultados encontrados:\n\n{context}\n\n"
            f"Usá esta información para responder las preguntas del usuario. "
            f"Hablás en español rioplatense natural. "
            f"Si el usuario dice 'listo', 'ya está' o 'salir', cerrá el modo de búsqueda."
        )
        web_history = []

        print(f"\n  [BÚSQUEDA] Entrando en modo tutor con contexto de: \"{query}\"")
        print(f"  Escribí 'listo' cuando termines.\n")

        self.speak(f"Listo, encontré información sobre \"{query}\". ¿Qué querés saber?")

        while True:
            user_text = self.listen()
            if not user_text:
                continue
            if user_text.lower() in ("listo", "ya está", "ya esta", "salir", "exit"):
                self.speak("Cerrando modo búsqueda. ¿Necesitás algo más?")
                break

            web_history.append({"role": "user", "content": user_text})
            messages = [{"role": "system", "content": web_system}] + web_history[-6:]
            response = _chat(messages, temperature=0.3, max_tokens=500)
            web_history.append({"role": "assistant", "content": response})
            self.speak(response)