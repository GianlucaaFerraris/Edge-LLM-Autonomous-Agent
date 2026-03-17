"""
test_modes.py — Unit tests para los 3 modos del asistente.

Correr con:
    pytest test_modes.py -v

Para correr solo un modo:
    pytest test_modes.py -v -k "english"
    pytest test_modes.py -v -k "engineering"
    pytest test_modes.py -v -k "agent"
"""

import json
import re
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — English Tutor
# ══════════════════════════════════════════════════════════════════════════════

class TestEnglishTutor:
    """Valida el comportamiento del tutor de inglés."""

    def _ask(self, chat_fn, sys_english, user_content):
        return chat_fn(
            [{"role": "system", "content": sys_english},
             {"role": "user",   "content": user_content}],
            temperature=0.3,
            max_tokens=300,
        )

    def test_responds_in_english(self, chat_fn, sys_english):
        """La respuesta principal debe estar en inglés."""
        prompt = (
            "Start a new English conversation about: 'daily routines'.\n"
            "Open with 2 sentences in Spanish introducing the topic, "
            "then ask ONE open question in English. No markdown."
        )
        response = self._ask(chat_fn, sys_english, prompt)
        # Heurística: al menos 60% de las palabras son ASCII (proxy de inglés dominante)
        words = response.split()
        ascii_words = [w for w in words if all(ord(c) < 128 for c in w)]
        ratio = len(ascii_words) / max(len(words), 1)
        assert ratio > 0.60, (
            f"Se esperaba mayoría de texto en inglés, ratio ASCII={ratio:.2f}\n{response}"
        )

    def test_no_markdown(self, chat_fn, sys_english):
        """No debe haber markdown (bullets, headers, bold)."""
        prompt = (
            "Start a new English conversation about: 'technology'.\n"
            "Open with 2 sentences in Spanish, then ONE question in English. No markdown."
        )
        response = self._ask(chat_fn, sys_english, prompt)
        forbidden = ["**", "##", "- ", "* ", "1. ", "```"]
        for marker in forbidden:
            assert marker not in response, (
                f"Se encontró markdown '{marker}' en la respuesta del tutor:\n{response}"
            )

    def test_single_paragraph(self, chat_fn, sys_english):
        """La respuesta debe ser un único párrafo (sin saltos de línea dobles)."""
        prompt = (
            "Start a new English conversation about: 'food'.\n"
            "Single spoken paragraph, no markdown, TTS-ready."
        )
        response = self._ask(chat_fn, sys_english, prompt)
        assert "\n\n" not in response, (
            f"Se esperaba un solo párrafo sin saltos dobles:\n{response}"
        )

    def test_no_chinese_characters(self, chat_fn, sys_english):
        """No debe haber caracteres chinos."""
        prompt = "Start a conversation about music. Single paragraph."
        response = self._ask(chat_fn, sys_english, prompt)
        chinese = re.findall(r'[\u4e00-\u9fff]', response)
        assert not chinese, f"Se encontraron caracteres chinos: {chinese}\n{response}"

    def test_ends_with_question(self, chat_fn, sys_english):
        """La respuesta debe terminar con una pregunta (signo ?)."""
        prompt = (
            "The student said: 'I like playing guitar in my free time.'\n"
            "No grammar errors found. Give warm feedback and end with ONE follow-up question."
        )
        response = self._ask(chat_fn, sys_english, prompt)
        assert "?" in response, (
            f"Se esperaba una pregunta de seguimiento:\n{response}"
        )

    def test_error_feedback_only_listed_errors(self, chat_fn, sys_english):
        """Al dar feedback de errores, solo debe mencionar los errores indicados."""
        prompt = (
            "The student said: \"I go to the store yesterday.\"\n\n"
            "LanguageTool found these errors:\n"
            "  - Error: 'go' → 'went' | Possible agreement error.\n\n"
            "Give warm feedback mentioning ONLY the listed error. "
            "End with ONE follow-up question. Single paragraph, no markdown."
        )
        response = self._ask(chat_fn, sys_english, prompt)
        # Debe mencionar el error concreto
        assert "went" in response.lower() or "go" in response.lower(), (
            f"La respuesta no parece mencionar el error de conjugación:\n{response}"
        )
        assert "?" in response, "Falta la pregunta de seguimiento."

    def test_no_filler_phrases(self, chat_fn, sys_english):
        """No debe usar frases de relleno típicas."""
        prompt = "Start a conversation about sports. Single paragraph."
        response = self._ask(chat_fn, sys_english, prompt)
        fillers = ["Great question!", "Sure!", "Of course!", "Certainly!"]
        for filler in fillers:
            assert filler.lower() not in response.lower(), (
                f"Se encontró frase de relleno '{filler}':\n{response}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — Engineering Tutor
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineeringTutor:
    """Valida el comportamiento del tutor de ingeniería."""

    def _ask(self, chat_fn, sys_engineering, user_content, lang="es"):
        return chat_fn(
            [{"role": "system", "content": sys_engineering},
             {"role": "user",   "content": user_content}],
            temperature=0.3,
            max_tokens=500,
        )

    def test_responds_in_spanish_when_asked_in_spanish(self, chat_fn, sys_engineering):
        """Si la pregunta es en español, la respuesta debe ser en español."""
        response = self._ask(chat_fn, sys_engineering, "¿Qué es la entropía?")
        # Detectar palabras comunes en español
        spanish_markers = ["la ", "el ", "es ", "en ", "que ", "de ", "un ", "una "]
        hits = sum(1 for m in spanish_markers if m in response.lower())
        assert hits >= 3, (
            f"Se esperaba respuesta en español (hits={hits}):\n{response}"
        )

    def test_responds_in_english_when_asked_in_english(self, chat_fn, sys_engineering):
        """Si la pregunta es en inglés, la respuesta debe ser en inglés."""
        response = self._ask(chat_fn, sys_engineering, "What is entropy?")
        english_markers = ["the ", "is ", "of ", "in ", "and ", "to ", "a ", "that "]
        hits = sum(1 for m in english_markers if m in response.lower())
        assert hits >= 4, (
            f"Se esperaba respuesta en inglés (hits={hits}):\n{response}"
        )

    def test_no_code_in_response(self, chat_fn, sys_engineering):
        """No debe escribir bloques de código."""
        response = self._ask(chat_fn, sys_engineering,
                             "Explicame cómo funciona un algoritmo de ordenamiento.")
        code_markers = ["```", "def ", "for i in", "int main", "function("]
        for marker in code_markers:
            assert marker not in response, (
                f"Se encontró código ('{marker}') en respuesta del ingeniero:\n{response}"
            )

    def test_no_filler_phrases(self, chat_fn, sys_engineering):
        """No debe usar frases de relleno."""
        response = self._ask(chat_fn, sys_engineering, "¿Qué es un transistor?")
        fillers = ["¡Excelente pregunta!", "¡Claro!", "¡Por supuesto!", "Great question!"]
        for filler in fillers:
            assert filler.lower() not in response.lower(), (
                f"Se encontró frase de relleno '{filler}':\n{response}"
            )

    def test_no_chinese_characters(self, chat_fn, sys_engineering):
        """No debe haber caracteres chinos."""
        response = self._ask(chat_fn, sys_engineering, "¿Qué es la mecánica cuántica?")
        chinese = re.findall(r'[\u4e00-\u9fff]', response)
        assert not chinese, f"Se encontraron caracteres chinos: {chinese}"

    def test_definition_is_concise(self, chat_fn, sys_engineering):
        """Una pregunta de definición debe dar una respuesta directa, no una novela."""
        response = self._ask(chat_fn, sys_engineering, "¿Qué es un mutex?")
        word_count = len(response.split())
        assert word_count < 300, (
            f"Respuesta muy larga para una definición ({word_count} palabras):\n{response}"
        )

    def test_comparison_uses_structure(self, chat_fn, sys_engineering):
        """Una pregunta de comparación puede usar algo de estructura."""
        response = self._ask(
            chat_fn, sys_engineering,
            "¿Diferencias entre TCP y UDP? Listá las principales."
        )
        # Debe mencionar ambos protocolos
        assert "tcp" in response.lower() or "TCP" in response, "No menciona TCP"
        assert "udp" in response.lower() or "UDP" in response, "No menciona UDP"

    def test_does_not_solve_math_problems(self, chat_fn, sys_engineering):
        """No debe calcular, sino explicar la intuición."""
        response = self._ask(
            chat_fn, sys_engineering,
            "¿Cuánto es la raíz cuadrada de 144?"
        )
        # No debe simplemente dar "12" como respuesta única
        assert len(response.split()) > 5, (
            f"Se esperaba una explicación, no un cálculo directo:\n{response}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODO 3 — Agente con herramientas
# ══════════════════════════════════════════════════════════════════════════════

class TestAgent:
    """Valida el comportamiento del agente y su uso de herramientas."""

    def _ask(self, chat_fn, sys_agent, messages_extra):
        messages = [{"role": "system", "content": sys_agent}] + messages_extra
        return chat_fn(messages, temperature=0.1, max_tokens=200)

    def _extract_tool_call(self, response):
        """
        Extrae el JSON del TOOL_CALL balanceando llaves.
        Más robusto que un regex — maneja args anidados correctamente.
        """
        marker = "TOOL_CALL:"
        idx = response.find(marker)
        if idx == -1:
            return None

        # Buscar el inicio del JSON
        start = response.find("{", idx)
        if start == -1:
            return None

        # Balancear llaves para encontrar el cierre correcto
        depth = 0
        for i, ch in enumerate(response[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = response[start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
        return None

    def test_task_list_triggers_tool_call(self, chat_fn, sys_agent):
        """Pedir las tareas pendientes debe generar un TOOL_CALL a task_list."""
        response = self._ask(chat_fn, sys_agent, [
            {"role": "user", "content": "¿Qué tengo pendiente?"}
        ])
        tool = self._extract_tool_call(response)
        assert tool is not None, f"Se esperaba un TOOL_CALL:\n{response}"
        assert tool.get("tool") == "task_list", (
            f"Se esperaba task_list, se obtuvo: {tool.get('tool')}"
        )

    def test_task_add_triggers_tool_call_with_title(self, chat_fn, sys_agent):
        """Agregar una tarea debe generar TOOL_CALL con el título correcto."""
        response = self._ask(chat_fn, sys_agent, [
            {"role": "user", "content": "Agregá una tarea: comprar leche"}
        ])
        tool = self._extract_tool_call(response)
        assert tool is not None, f"Se esperaba un TOOL_CALL:\n{response}"
        assert tool.get("tool") == "task_add", (
            f"Se esperaba task_add, se obtuvo: {tool.get('tool')}"
        )
        args = tool.get("args", {})
        assert "leche" in str(args).lower(), (
            f"El título de la tarea debería mencionar 'leche': {args}"
        )

    def test_calendar_add_triggers_tool_call(self, chat_fn, sys_agent):
        """Agendar un evento debe generar TOOL_CALL a cal_add."""
        response = self._ask(chat_fn, sys_agent, [
            {"role": "user", "content": "Agendame una reunión mañana a las 10am"}
        ])
        tool = self._extract_tool_call(response)
        assert tool is not None, f"Se esperaba un TOOL_CALL:\n{response}"
        assert tool.get("tool") in ("cal_add", "reminder_set"), (
            f"Se esperaba cal_add o reminder_set, se obtuvo: {tool.get('tool')}"
        )

    def test_tool_call_json_is_valid(self, chat_fn, sys_agent):
        """El JSON del TOOL_CALL debe ser parseable."""
        response = self._ask(chat_fn, sys_agent, [
            {"role": "user", "content": "Buscá el clima en Buenos Aires"}
        ])
        if "TOOL_CALL" in response:
            tool = self._extract_tool_call(response)
            assert tool is not None, (
                f"El TOOL_CALL tiene JSON inválido:\n{response}"
            )

    def test_responds_after_tool_result(self, chat_fn, sys_agent):
        """Después de un TOOL_RESULT el agente debe responder en prosa, sin otro TOOL_CALL."""
        response = self._ask(chat_fn, sys_agent, [
            {"role": "user",      "content": "¿Qué tengo pendiente?"},
            {"role": "assistant", "content": 'TOOL_CALL: {"tool": "task_list", "args": {}}'},
            {"role": "user",      "content": (
                "TOOL_RESULT: [ ] Comprar yerba\n[ ] Llamar al médico\n[ ] Pagar alquiler"
            )},
        ])
        # No debe haber otro TOOL_CALL en la respuesta final
        assert "TOOL_CALL" not in response, (
            f"Se esperaba respuesta en prosa, no otro TOOL_CALL:\n{response}"
        )
        # Debe mencionar al menos una tarea
        items = ["yerba", "médico", "alquiler"]
        found = any(item in response.lower() for item in items)
        assert found, f"La respuesta no menciona ninguna tarea:\n{response}"

    def test_responds_in_spanish_rioplatense(self, chat_fn, sys_agent):
        """El agente debe responder en español con voseo."""
        response = self._ask(chat_fn, sys_agent, [
            {"role": "user",      "content": "Hola, ¿cómo andás?"},
        ])
        # El agente puede emitir TOOL_CALL o responder directamente
        # Si responde en prosa, debe ser en español
        if "TOOL_CALL" not in response:
            spanish_markers = ["hola", "bien", "puedo", "ayudarte", "claro", "cómo"]
            hits = sum(1 for m in spanish_markers if m in response.lower())
            assert hits >= 1, (
                f"La respuesta no parece estar en español:\n{response}"
            )

    def test_no_tool_call_for_conversational_query(self, chat_fn, sys_agent):
        """Una pregunta conversacional simple no debe generar TOOL_CALL."""
        response = self._ask(chat_fn, sys_agent, [
            {"role": "user", "content": "¿Cuántas herramientas tenés disponibles?"}
        ])
        assert "TOOL_CALL" not in response, (
            f"Una pregunta conversacional no debería disparar un TOOL_CALL:\n{response}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODO SELECTOR — Detección de modo
# ══════════════════════════════════════════════════════════════════════════════

MODE_DETECT_PROMPT = """El usuario dijo: "{text}"

¿Qué modo quiere usar?
- english_tutor: practicar inglés, hablar en inglés, mejorar gramática
- engineering: pregunta técnica de software, IA, física, química, robótica, electrónica
- agent: hacer algo concreto (buscar, agendar, recordatorio, WhatsApp, tareas)
- unclear: no queda claro

Respondé con exactamente una palabra: english_tutor, engineering, agent, o unclear."""


class TestModeDetector:
    """Valida que el selector de modos clasifique correctamente."""

    def _detect(self, chat_fn, model_name, text):
        res = chat_fn(
            [{"role": "user", "content": MODE_DETECT_PROMPT.format(text=text)}],
            temperature=0.0,
            max_tokens=5,
        ).lower()
        if "english" in res:   return "english_tutor"
        if "engineer" in res:  return "engineering"
        if "agent" in res:     return "agent"
        return "unclear"

    @pytest.mark.parametrize("text,expected", [
        ("quiero practicar inglés",           "english_tutor"),
        ("let's talk in English",             "english_tutor"),
        ("¿qué es un transistor?",            "engineering"),
        ("explicame la diferencia entre TCP y UDP", "engineering"),
        ("agregá una tarea: llamar al médico", "agent"),
        ("¿qué tengo en el calendario mañana?", "agent"),
        ("buscame el clima en Córdoba",        "agent"),
    ])
    def test_mode_detection(self, chat_fn, model_name, text, expected):
        detected = self._detect(chat_fn, model_name, text)
        assert detected == expected, (
            f"Texto: '{text}'\n"
            f"  Esperado: {expected}\n"
            f"  Detectado: {detected}"
        )