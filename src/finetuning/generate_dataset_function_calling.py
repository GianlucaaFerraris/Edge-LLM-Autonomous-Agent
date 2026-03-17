"""

generate_agent_dataset.py

Genera 150 ejemplos de conversación para el modo agente del asistente edge.
Tools cubiertas:
  - search_web
  - task_add / task_list / task_done
  - reminder_set
  - wa_send / wa_read
  - cal_add / cal_list / cal_delete

Formato de salida: JSONL compatible con Ollama / Unsloth fine-tuning.
Idioma principal: español.
Herramientas: 1 por turno (simple y confiable para un 7B).

Estructura de cada ejemplo:
  messages[0] = system prompt del agente
  messages[1] = user request
  messages[2] = TOOL_CALL del modelo
  messages[3] = TOOL_RESULT (simulado)
  messages[4] = respuesta final del modelo

Para ejemplos sin tool (preguntas directas):
  messages[0] = system
  messages[1] = user
  messages[2] = respuesta directa

NOTA CALENDARIO:
  Se usa calendario LOCAL (SQLite) en vez de Google Calendar.
  Sin OAuth, sin red, sin dependencias externas — 100% local en SBC.
  Base de datos: ~/.local/share/edge_assistant/calendar.db
  Schema:
    CREATE TABLE events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        title     TEXT NOT NULL,
        datetime  TEXT NOT NULL,  -- ISO 8601 p.ej. "2025-03-20 15:00"
        duration  INTEGER DEFAULT 60,  -- minutos
        notes     TEXT DEFAULT ""
    );
"""


import json
import os
import random
import re
import sys
from tqdm import tqdm
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL   = "http://localhost:11434/v1"
MODEL        = "qwen2.5:7b"
TARGET       = 150
OUTPUT_CLEAN = "agent_dataset_clean.jsonl"
OUTPUT_DEBUG = "agent_dataset_debug.jsonl"
CHECKPOINT   = 25

client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# ---------------------------------------------------------------------------
# System prompt del agente
# ---------------------------------------------------------------------------
SYSTEM_AGENT = """Sos un asistente personal de IA que corre localmente en una SBC (Single Board Computer). Respondés siempre en español rioplatense. Tenés acceso a las siguientes herramientas:

  - search_web(query: str) → busca en internet y devuelve resultados relevantes
  - task_add(title: str, notes: str?) → agrega una tarea a la lista de pendientes
  - task_list(filter: str?) → lista las tareas pendientes, opcionalmente filtradas
  - task_done(title: str) → marca una tarea como completada
  - reminder_set(text: str, datetime: str) → crea un recordatorio para fecha/hora específica
  - wa_send(contact: str, message: str) → envía un mensaje de WhatsApp
  - wa_read(contact: str?) → lee mensajes recientes de WhatsApp
  - cal_add(title: str, datetime: str, duration_min: int?, notes: str?) → agrega evento al calendario local (SQLite)
  - cal_list(date: str?) → lista eventos del calendario local para una fecha o rango
  - cal_delete(title: str, datetime: str) → elimina un evento del calendario local

Cuando el usuario te pide hacer algo, emitís exactamente un TOOL_CALL en este formato:
TOOL_CALL: {"tool": "<nombre>", "args": {<argumentos>}}

Después de recibir el TOOL_RESULT, respondés de forma natural y concisa en español rioplatense.
Si no necesitás ninguna herramienta para responder, respondés directamente sin TOOL_CALL."""

# ---------------------------------------------------------------------------
# Escenarios predefinidos por tool
# Estructura: (user_request, tool, args, simulated_result, final_response)
# ---------------------------------------------------------------------------

SCENARIOS = {

    # ── search_web ──────────────────────────────────────────────────────────
    "search_web": [
        (
            "Buscame qué es el protocolo MQTT y para qué se usa.",
            "search_web",
            {"query": "protocolo MQTT explicación uso"},
            "MQTT es un protocolo de mensajería ligero basado en el modelo publicador/suscriptor, diseñado para dispositivos con recursos limitados e IoT. Usa TCP/IP y un broker central para distribuir mensajes.",
            "MQTT es un protocolo de mensajería liviano ideal para IoT. Funciona con un modelo publicador/suscriptor donde los dispositivos se conectan a un broker central que distribuye los mensajes. Es muy eficiente en redes con poca banda ancha.",
        ),
        (
            "¿Cuál es el precio actual del dólar en Argentina?",
            "search_web",
            {"query": "precio dólar Argentina hoy"},
            "Dólar oficial: $1.050 | Dólar blue: $1.230 | Dólar MEP: $1.195 (actualizado hoy)",
            "Según los últimos datos: el dólar oficial está a $1.050, el blue a $1.230 y el MEP a $1.195.",
        ),
        (
            "Buscame recetas fáciles con pollo y arroz.",
            "search_web",
            {"query": "recetas fáciles pollo arroz"},
            "Arroz con pollo al limón, Pollo al curry con arroz basmati, Cazuela de pollo y arroz con verduras. Todas listas en menos de 40 minutos.",
            "Encontré tres recetas fáciles: arroz con pollo al limón, pollo al curry con arroz basmati, y cazuela de pollo y arroz con verduras. Las tres se preparan en menos de 40 minutos.",
        ),
        (
            "¿Qué temperatura máxima soporta la Radxa Rock 5B?",
            "search_web",
            {"query": "Radxa Rock 5B temperatura máxima operación"},
            "La Radxa Rock 5B opera de 0°C a 80°C según especificaciones oficiales. Se recomienda disipador activo para uso continuo bajo carga.",
            "La Radxa Rock 5B soporta hasta 80°C según las especificaciones oficiales. Para uso continuo bajo carga, se recomienda usar disipador activo.",
        ),
        (
            "Buscame noticias de hoy sobre inteligencia artificial.",
            "search_web",
            {"query": "noticias inteligencia artificial hoy"},
            "Google DeepMind presenta nuevo modelo de proteínas. Meta lanza Llama 4 con contexto de 1M tokens. OpenAI anuncia GPT-5 para empresas.",
            "Las noticias de hoy en IA: DeepMind presentó un nuevo modelo para proteínas, Meta lanzó Llama 4 con contexto de 1 millón de tokens, y OpenAI anunció GPT-5 para uso empresarial.",
        ),
        (
            "¿Cómo se hace para instalar Ollama en Ubuntu?",
            "search_web",
            {"query": "instalar Ollama Ubuntu comandos"},
            "curl -fsSL https://ollama.ai/install.sh | sh — luego ollama serve para iniciar el servidor y ollama pull <modelo> para descargar modelos.",
            "Para instalar Ollama en Ubuntu: ejecutá `curl -fsSL https://ollama.ai/install.sh | sh`, luego `ollama serve` para iniciar el servidor, y `ollama pull <modelo>` para descargar el modelo que quieras.",
        ),
        (
            "¿Cuáles son los mejores frameworks de Python para hacer APIs REST?",
            "search_web",
            {"query": "mejores frameworks Python API REST 2025"},
            "FastAPI (el más popular por rendimiento y tipado), Flask (simple y flexible), Django REST Framework (completo para proyectos grandes).",
            "Los mejores frameworks Python para APIs REST son: FastAPI (el más popular hoy, muy rápido y con tipado automático), Flask (simple y flexible para proyectos pequeños), y Django REST Framework (ideal para proyectos grandes con mucha funcionalidad incluida).",
        ),
        (
            "Buscame el horario del colectivo 88 en Córdoba.",
            "search_web",
            {"query": "horario colectivo 88 Córdoba Argentina"},
            "Línea 88 Córdoba: primer servicio 5:30hs, último servicio 23:15hs. Frecuencia pico: cada 12 minutos. Frecuencia valle: cada 20 minutos.",
            "La línea 88 en Córdoba sale desde las 5:30 hasta las 23:15. En horas pico pasa cada 12 minutos, y en horario valle cada 20 minutos.",
        ),
        (
            "¿Qué es LoRA y cómo se usa para fine-tuning?",
            "search_web",
            {"query": "LoRA fine-tuning LLM explicación"},
            "LoRA (Low-Rank Adaptation) añade matrices de bajo rango a las capas del transformer, reduciendo parámetros entrenables hasta 10.000x. Se usa con Unsloth, HuggingFace PEFT, o llama.cpp.",
            "LoRA es una técnica que agrega matrices de bajo rango a las capas del transformer, reduciendo los parámetros entrenables hasta 10.000 veces. Podés usarlo con Unsloth, HuggingFace PEFT o llama.cpp para fine-tunear sin necesitar mucha VRAM.",
        ),
        (
            "¿Qué películas están en cartelera esta semana en Córdoba?",
            "search_web",
            {"query": "cartelera cine Córdoba esta semana"},
            "Cines Hoyts: Mission Impossible 8, Inside Out 3, Dune 3. Showcase: Avengers: Secret Wars, Minecraft Movie.",
            "Esta semana en los cines de Córdoba: en Hoyts pasan Mission Impossible 8, Inside Out 3 y Dune 3. En Showcase están Avengers: Secret Wars y la película de Minecraft.",
        ),
    ],

    # ── task_add ────────────────────────────────────────────────────────────
    "task_add": [
        (
            "Agregame una tarea para revisar el fine-tuning del modelo.",
            "task_add",
            {"title": "Revisar fine-tuning del modelo", "notes": ""},
            "OK: tarea agregada",
            "Listo, agregué 'Revisar fine-tuning del modelo' a tu lista de tareas.",
        ),
        (
            "Anotame que tengo que comprar yerba y azúcar.",
            "task_add",
            {"title": "Comprar yerba y azúcar", "notes": "supermercado"},
            "OK: tarea agregada",
            "Anotado. Agregué 'Comprar yerba y azúcar' a tus pendientes.",
        ),
        (
            "Recordame que tengo que llamar al médico para pedir turno.",
            "task_add",
            {"title": "Llamar al médico para pedir turno", "notes": ""},
            "OK: tarea agregada",
            "Agregué 'Llamar al médico para pedir turno' a tu lista de tareas.",
        ),
        (
            "Agregame una tarea: preparar el dataset para el fine-tuning antes del viernes.",
            "task_add",
            {"title": "Preparar dataset para fine-tuning", "notes": "antes del viernes"},
            "OK: tarea agregada",
            "Listo, agregué la tarea 'Preparar dataset para fine-tuning' con nota 'antes del viernes'.",
        ),
        (
            "Anotame que tengo que pagar el alquiler.",
            "task_add",
            {"title": "Pagar alquiler", "notes": ""},
            "OK: tarea agregada",
            "Anotado. Agregué 'Pagar alquiler' a tus pendientes.",
        ),
        (
            "Tengo que actualizar el sistema operativo de la Radxa, anotámelo.",
            "task_add",
            {"title": "Actualizar sistema operativo Radxa", "notes": ""},
            "OK: tarea agregada",
            "Agregué 'Actualizar sistema operativo Radxa' a tu lista de tareas.",
        ),
        (
            "Anotame que tengo que enviarle el presupuesto a Juan.",
            "task_add",
            {"title": "Enviar presupuesto a Juan", "notes": ""},
            "OK: tarea agregada",
            "Listo, 'Enviar presupuesto a Juan' está en tu lista de pendientes.",
        ),
        (
            "Agregá una tarea para revisar las métricas de entrenamiento del modelo.",
            "task_add",
            {"title": "Revisar métricas de entrenamiento", "notes": "modelo qwen2.5:7b"},
            "OK: tarea agregada",
            "Agregué 'Revisar métricas de entrenamiento' con nota sobre el modelo qwen2.5:7b.",
        ),
    ],

    # ── task_list ────────────────────────────────────────────────────────────
    "task_list": [
        (
            "¿Qué tengo pendiente?",
            "task_list",
            {"filter": ""},
            "[ ] Revisar fine-tuning del modelo\n[ ] Comprar yerba y azúcar\n[ ] Llamar al médico\n[ ] Pagar alquiler",
            "Tus tareas pendientes son: revisar el fine-tuning, comprar yerba y azúcar, llamar al médico, y pagar el alquiler.",
        ),
        (
            "Mostrame todas mis tareas.",
            "task_list",
            {"filter": ""},
            "[ ] Preparar dataset fine-tuning\n[ ] Actualizar Radxa OS\n[x] Instalar LanguageTool",
            "Tenés dos tareas pendientes: preparar el dataset para fine-tuning y actualizar el OS de la Radxa. Ya completaste la instalación de LanguageTool.",
        ),
        (
            "¿Tengo algo que hacer hoy?",
            "task_list",
            {"filter": "hoy"},
            "[ ] Pagar alquiler (vence hoy)\n[ ] Enviar presupuesto a Juan",
            "Para hoy tenés: pagar el alquiler (vence hoy) y enviar el presupuesto a Juan.",
        ),
        (
            "Listame las tareas relacionadas con el modelo.",
            "task_list",
            {"filter": "modelo"},
            "[ ] Revisar fine-tuning del modelo\n[ ] Revisar métricas de entrenamiento",
            "Las tareas relacionadas con el modelo son: revisar el fine-tuning y revisar las métricas de entrenamiento.",
        ),
        (
            "¿Qué me falta hacer esta semana?",
            "task_list",
            {"filter": "semana"},
            "[ ] Preparar dataset para fine-tuning (antes del viernes)\n[ ] Llamar al médico\n[ ] Comprar yerba y azúcar",
            "Esta semana te quedan: preparar el dataset antes del viernes, llamar al médico y hacer las compras.",
        ),
    ],

    # ── task_done ────────────────────────────────────────────────────────────
    "task_done": [
        (
            "Ya llamé al médico, marcámelo como hecho.",
            "task_done",
            {"title": "Llamar al médico para pedir turno"},
            "OK: tarea marcada como completada",
            "Perfecto, marqué 'Llamar al médico' como completada.",
        ),
        (
            "Terminé de preparar el dataset, marcá esa tarea como lista.",
            "task_done",
            {"title": "Preparar dataset para fine-tuning"},
            "OK: tarea marcada como completada",
            "Excelente, marqué 'Preparar dataset para fine-tuning' como completada.",
        ),
        (
            "Pagué el alquiler.",
            "task_done",
            {"title": "Pagar alquiler"},
            "OK: tarea marcada como completada",
            "Listo, 'Pagar alquiler' marcado como completado.",
        ),
        (
            "Ya le mandé el presupuesto a Juan, podés tildar esa tarea.",
            "task_done",
            {"title": "Enviar presupuesto a Juan"},
            "OK: tarea marcada como completada",
            "Perfecto, marqué 'Enviar presupuesto a Juan' como completada.",
        ),
        (
            "Actualicé la Radxa, esa tarea ya está.",
            "task_done",
            {"title": "Actualizar sistema operativo Radxa"},
            "OK: tarea marcada como completada",
            "Bien, 'Actualizar sistema operativo Radxa' marcada como completada.",
        ),
    ],

    # ── reminder_set ─────────────────────────────────────────────────────────
    "reminder_set": [
        (
            "Recordame mañana a las 9 que tengo reunión con el equipo.",
            "reminder_set",
            {"text": "Reunión con el equipo", "datetime": "mañana 09:00"},
            "OK: recordatorio creado para mañana 09:00",
            "Listo, te recuerdo mañana a las 9 que tenés reunión con el equipo.",
        ),
        (
            "Poneme un recordatorio para el viernes a las 18hs de ir al gimnasio.",
            "reminder_set",
            {"text": "Ir al gimnasio", "datetime": "viernes 18:00"},
            "OK: recordatorio creado para viernes 18:00",
            "Recordatorio creado: el viernes a las 18 te aviso que tenés que ir al gimnasio.",
        ),
        (
            "Recordame en 2 horas revisar el entrenamiento del modelo.",
            "reminder_set",
            {"text": "Revisar entrenamiento del modelo", "datetime": "+2h"},
            "OK: recordatorio creado para +2 horas",
            "Te recuerdo en 2 horas que tenés que revisar el entrenamiento del modelo.",
        ),
        (
            "Avisame el lunes a las 10 que tengo que pagar la factura de luz.",
            "reminder_set",
            {"text": "Pagar factura de luz", "datetime": "lunes 10:00"},
            "OK: recordatorio creado para lunes 10:00",
            "Listo, el lunes a las 10 te aviso que tenés que pagar la factura de luz.",
        ),
        (
            "Recordame todos los días a las 8 tomar el medicamento.",
            "reminder_set",
            {"text": "Tomar medicamento", "datetime": "diario 08:00"},
            "OK: recordatorio recurrente creado para 08:00 diario",
            "Creé un recordatorio diario a las 8 para que tomes el medicamento.",
        ),
        (
            "Recordame el 25 de marzo a las 12 que es el cumpleaños de mamá.",
            "reminder_set",
            {"text": "Cumpleaños de mamá", "datetime": "2025-03-25 12:00"},
            "OK: recordatorio creado para 25/03 12:00",
            "Anotado, el 25 de marzo a las 12 te recuerdo el cumpleaños de tu mamá.",
        ),
        (
            "Poneme una alarma para mañana a las 7:30 para levantarme.",
            "reminder_set",
            {"text": "Levantarse", "datetime": "mañana 07:30"},
            "OK: recordatorio creado para mañana 07:30",
            "Listo, mañana a las 7:30 te despierto.",
        ),
        (
            "Recordame dentro de 30 minutos sacar la ropa del lavarropas.",
            "reminder_set",
            {"text": "Sacar ropa del lavarropas", "datetime": "+30min"},
            "OK: recordatorio creado para +30 minutos",
            "En 30 minutos te aviso que saques la ropa del lavarropas.",
        ),
    ],

    # ── wa_send ──────────────────────────────────────────────────────────────
    "wa_send": [
        (
            "Mandále un WhatsApp a Juan que llego a las 19.",
            "wa_send",
            {"contact": "Juan", "message": "Llego a las 19."},
            "OK: mensaje enviado a Juan",
            "Listo, le mandé un mensaje a Juan avisándole que llegás a las 19.",
        ),
        (
            "Decile a mamá por WhatsApp que hoy no voy a cenar.",
            "wa_send",
            {"contact": "Mamá", "message": "Hoy no voy a cenar."},
            "OK: mensaje enviado a Mamá",
            "Le avisé a tu mamá por WhatsApp que hoy no vas a cenar.",
        ),
        (
            "Mandá un mensaje al grupo Trabajo que mañana la reunión es a las 10.",
            "wa_send",
            {"contact": "Trabajo", "message": "Mañana la reunión es a las 10."},
            "OK: mensaje enviado al grupo Trabajo",
            "Mandé el mensaje al grupo Trabajo avisando que la reunión de mañana es a las 10.",
        ),
        (
            "Escribile a María que ya terminé el informe.",
            "wa_send",
            {"contact": "María", "message": "Ya terminé el informe."},
            "OK: mensaje enviado a María",
            "Listo, le avisé a María que terminaste el informe.",
        ),
        (
            "Mandále a Pedro que no puedo ir el sábado.",
            "wa_send",
            {"contact": "Pedro", "message": "No puedo ir el sábado."},
            "OK: mensaje enviado a Pedro",
            "Le mandé el mensaje a Pedro avisándole que no podés ir el sábado.",
        ),
        (
            "Avisale a Lucía por WhatsApp que el pedido está listo para retirar.",
            "wa_send",
            {"contact": "Lucía", "message": "El pedido está listo para retirar."},
            "OK: mensaje enviado a Lucía",
            "Le avisé a Lucía que el pedido está listo para retirar.",
        ),
        (
            "Mandá al grupo Familia que el asado es el domingo a las 13.",
            "wa_send",
            {"contact": "Familia", "message": "El asado es el domingo a las 13."},
            "OK: mensaje enviado al grupo Familia",
            "Avisé al grupo Familia que el asado es el domingo a las 13.",
        ),
        (
            "Decile a Carlos que lo llamo en un rato.",
            "wa_send",
            {"contact": "Carlos", "message": "Te llamo en un rato."},
            "OK: mensaje enviado a Carlos",
            "Le mandé a Carlos que lo llamás en un rato.",
        ),
    ],

    # ── wa_read ──────────────────────────────────────────────────────────────
    "wa_read": [
        (
            "¿Tengo mensajes nuevos en WhatsApp?",
            "wa_read",
            {"contact": ""},
            "Juan (hace 5min): '¿A qué hora llegás?' | Mamá (hace 1h): 'Llamame cuando puedas' | Trabajo (hace 2h): 'Reunión cancelada para mañana'",
            "Tenés 3 mensajes nuevos: Juan pregunta a qué hora llegás (hace 5 minutos), tu mamá pide que la llames cuando puedas (hace 1 hora), y el grupo Trabajo avisó que la reunión de mañana está cancelada.",
        ),
        (
            "Leeme los mensajes de Juan.",
            "wa_read",
            {"contact": "Juan"},
            "Juan (10:30): '¿Quedamos a las 19?' | Juan (11:15): '¿A qué hora llegás?'",
            "Juan te mandó dos mensajes: a las 10:30 preguntó si quedaban a las 19, y a las 11:15 preguntó a qué hora llegás.",
        ),
        (
            "¿Qué me mandó mamá?",
            "wa_read",
            {"contact": "Mamá"},
            "Mamá (hace 1h): 'Llamame cuando puedas, es importante'",
            "Tu mamá te mandó un mensaje hace una hora diciendo que la llames cuando puedas, que es importante.",
        ),
        (
            "Mostrame los últimos mensajes del grupo Trabajo.",
            "wa_read",
            {"contact": "Trabajo"},
            "Pedro (09:00): 'Buen día a todos' | Ana (09:45): 'El informe está listo' | Carlos (11:00): 'Reunión cancelada para mañana'",
            "En el grupo Trabajo: Pedro saludó a las 9, Ana avisó que el informe está listo a las 9:45, y Carlos canceló la reunión de mañana a las 11.",
        ),
        (
            "¿Tengo mensajes sin leer?",
            "wa_read",
            {"contact": ""},
            "4 mensajes sin leer: María (x2), Familia (x1), Pedro (x1)",
            "Tenés 4 mensajes sin leer: 2 de María, 1 del grupo Familia y 1 de Pedro.",
        ),
    ],

    # ── cal_add — calendario local SQLite ────────────────────────────────────
    "cal_add": [
        (
            "Agendame una reunión mañana a las 15 con el equipo de 1 hora.",
            "cal_add",
            {"title": "Reunión con el equipo", "datetime": "mañana 15:00", "duration_min": 60, "notes": ""},
            "OK: evento creado",
            "Listo, agendé 'Reunión con el equipo' para mañana a las 15, duración 1 hora.",
        ),
        (
            "Agregame al calendario el dentista el jueves a las 10.",
            "cal_add",
            {"title": "Dentista", "datetime": "jueves 10:00", "duration_min": 45, "notes": ""},
            "OK: evento creado",
            "Agendé el dentista para el jueves a las 10.",
        ),
        (
            "Poneme en el calendario que el sábado a las 13 es el asado en lo de Pedro.",
            "cal_add",
            {"title": "Asado en lo de Pedro", "datetime": "sábado 13:00", "duration_min": 180, "notes": "llevar bebidas"},
            "OK: evento creado",
            "Agregué 'Asado en lo de Pedro' al calendario para el sábado a las 13, duración estimada 3 horas.",
        ),
        (
            "Agendame el curso de Python online el lunes y miércoles de 19 a 21.",
            "cal_add",
            {"title": "Curso Python online", "datetime": "lunes 19:00", "duration_min": 120, "notes": "también miércoles"},
            "OK: evento creado",
            "Agendé el curso de Python los lunes de 19 a 21. Agregá manualmente el miércoles con la misma hora.",
        ),
        (
            "Meteme en el calendario 'Entrega proyecto' para el viernes a las 23:59.",
            "cal_add",
            {"title": "Entrega proyecto", "datetime": "viernes 23:59", "duration_min": 0, "notes": "deadline"},
            "OK: evento creado",
            "Listo, 'Entrega proyecto' está en tu calendario para el viernes a las 23:59.",
        ),
        (
            "Agendame una llamada con el cliente el martes a las 11:30, 30 minutos.",
            "cal_add",
            {"title": "Llamada con cliente", "datetime": "martes 11:30", "duration_min": 30, "notes": ""},
            "OK: evento creado",
            "Agendé 'Llamada con cliente' para el martes a las 11:30, duración 30 minutos.",
        ),
    ],

    # ── cal_list — calendario local SQLite ───────────────────────────────────
    "cal_list": [
        (
            "¿Qué tengo en el calendario hoy?",
            "cal_list",
            {"date": "hoy"},
            "09:00 - Reunión de equipo (1h) | 14:00 - Llamada con cliente (30min) | 18:00 - Gimnasio",
            "Hoy tenés: reunión de equipo a las 9, llamada con el cliente a las 14, y gimnasio a las 18.",
        ),
        (
            "¿Qué tengo esta semana?",
            "cal_list",
            {"date": "semana"},
            "Lunes 15:00 - Reunión equipo | Miércoles 10:00 - Dentista | Viernes 23:59 - Entrega proyecto",
            "Esta semana tenés: reunión de equipo el lunes a las 15, dentista el miércoles a las 10, y la entrega del proyecto el viernes a las 23:59.",
        ),
        (
            "¿Tengo algo el jueves?",
            "cal_list",
            {"date": "jueves"},
            "10:00 - Dentista (45min)",
            "El jueves tenés solo el dentista a las 10, 45 minutos.",
        ),
        (
            "Mostrame mi agenda del fin de semana.",
            "cal_list",
            {"date": "fin de semana"},
            "Sábado 13:00 - Asado en lo de Pedro (3h) | Domingo - Sin eventos",
            "El sábado tenés el asado en lo de Pedro a las 13. El domingo está libre.",
        ),
        (
            "¿Qué tengo mañana?",
            "cal_list",
            {"date": "mañana"},
            "15:00 - Reunión con el equipo (1h) | 19:00 - Curso Python online (2h)",
            "Mañana tenés: reunión con el equipo a las 15 y el curso de Python de 19 a 21.",
        ),
    ],

    # ── cal_delete — calendario local SQLite ──────────────────────────────────
    "cal_delete": [
        (
            "Cancelá la reunión de mañana a las 15.",
            "cal_delete",
            {"title": "Reunión con el equipo", "datetime": "mañana 15:00"},
            "OK: evento eliminado",
            "Listo, cancelé la reunión de mañana a las 15.",
        ),
        (
            "Borrá el dentista del jueves, lo reprogramé.",
            "cal_delete",
            {"title": "Dentista", "datetime": "jueves 10:00"},
            "OK: evento eliminado",
            "Eliminé el turno del dentista del jueves. Cuando tengas la nueva fecha, la agendo.",
        ),
        (
            "Sacame del calendario la llamada con el cliente del martes.",
            "cal_delete",
            {"title": "Llamada con cliente", "datetime": "martes 11:00"},
            "OK: evento eliminado",
            "Eliminé la llamada con el cliente del martes.",
        ),
        (
            "Cancelá el asado del sábado.",
            "cal_delete",
            {"title": "Asado en lo de Pedro", "datetime": "sábado 13:00"},
            "OK: evento eliminado",
            "Cancelé el asado del sábado en el calendario.",
        ),
    ],

    # ── sin tool — respuestas directas ───────────────────────────────────────
    "no_tool": [
        (
            "¿Qué herramientas tenés disponibles?",
            None, None, None,
            "Puedo ayudarte con: buscar en internet, gestionar tu lista de tareas pendientes, "
            "poner recordatorios, enviar y leer mensajes de WhatsApp, y organizar tu calendario "
            "(ver, agregar y eliminar eventos). ¿Qué necesitás?",
        ),
        (
            "¿Podés mandarme un mail?",
            None, None, None,
            "Por ahora no tengo acceso al correo electrónico, pero puedo mandarte mensajes por "
            "WhatsApp, agregar tareas, poner recordatorios o buscar información. "
            "¿Te ayudo con algo de eso?",
        ),
        (
            "¿Podés controlar dispositivos de mi casa?",
            None, None, None,
            "Todavía no tengo integración con domótica o dispositivos del hogar. "
            "Mis herramientas actuales son: búsqueda web, tareas, recordatorios, "
            "WhatsApp y calendario. ¿Necesitás algo de eso?",
        ),
        (
            "Hola, ¿cómo estás?",
            None, None, None,
            "¡Hola! Estoy listo para ayudarte. Puedo buscar información, gestionar tus tareas, "
            "mandarte recordatorios, leer y enviar WhatsApps, o ver tu calendario. ¿Qué necesitás?",
        ),
        (
            "¿Podés hacer reservas en restaurantes?",
            None, None, None,
            "No tengo esa función directamente, pero puedo buscarte el número o web del restaurante "
            "para que vos hagas la reserva, o agendarte un recordatorio para llamar. ¿Te ayudo así?",
        ),
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(messages: list, temperature: float = 0.3, max_tokens: int = 300) -> str:
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = res.choices[0].message.content or ""
        if re.search(r'[\u4e00-\u9fff]', text):
            return ""
        return text.strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""


SYSTEM_VARY = (
    "Sos un asistente personal que responde SIEMPRE en español rioplatense usando voseo (vos, tenés, podés, hacés) no usás tuteo (tú, tienes, puedes) o expresiones españolas como vale (usa listo para eso). "
    "(vos, tenés, podés, hacés). Nunca uses tuteo (tú, tienes, puedes). "
    "Respondés de forma natural, concisa y directa."
)

def _vary_response(base_response: str, user_request: str) -> str:
    """
    Genera una variación natural de la respuesta base usando el LLM.
    Usa el system prompt correcto para mantener voseo consistente.
    """
    prompt = (
        f"Reescribí esta respuesta de forma ligeramente distinta pero igualmente natural. "
        f"Mantené la misma información. Máximo 2 oraciones. Sin markdown.\n\n"
        f"Original: {base_response}\n\n"
        f"Reescrita:"
    )
    varied = _call(
        [{"role": "system", "content": SYSTEM_VARY},
         {"role": "user",   "content": prompt}],
        temperature=0.6, max_tokens=150
    )
    return varied if varied and len(varied) > 20 else base_response


def _vary_user_request(base_request: str) -> str:
    """Genera una variación del pedido del usuario para más diversidad."""
    prompt = (
        f"Reescribí este pedido de usuario de forma ligeramente distinta pero equivalente. "
        f"Mismo idioma y misma intención. Una sola oración. Sin markdown.\n\n"
        f"Original: {base_request}\n\n"
        f"Reescrito:"
    )
    varied = _call(
        [{"role": "system", "content": SYSTEM_VARY},
         {"role": "user",   "content": prompt}],
        temperature=0.7, max_tokens=80
    )
    return varied if varied and len(varied) > 10 else base_request


def build_example(scenario_tuple: tuple, tool_key: str, vary: bool = True) -> dict:
    """
    Construye un ejemplo JSONL a partir de un escenario.
    vary=True genera variaciones del texto para más diversidad.
    """
    user_req, tool, args, result, final = scenario_tuple

    # Variar el 60% de los ejemplos para diversidad
    if vary and random.random() < 0.6:
        user_req = _vary_user_request(user_req)
        final    = _vary_response(final, user_req)

    if tool is None:
        # Respuesta directa sin tool
        msgs = [
            {"role": "system",    "content": SYSTEM_AGENT},
            {"role": "user",      "content": user_req},
            {"role": "assistant", "content": final},
        ]
    else:
        tool_call_str  = f'TOOL_CALL: {json.dumps({"tool": tool, "args": args}, ensure_ascii=False)}'
        tool_result_str = f'TOOL_RESULT: {result}'

        msgs = [
            {"role": "system",    "content": SYSTEM_AGENT},
            {"role": "user",      "content": user_req},
            {"role": "assistant", "content": tool_call_str},
            {"role": "user",      "content": tool_result_str},
            {"role": "assistant", "content": final},
        ]

    return {
        "messages": msgs,
        "_debug": {
            "mode":     "agent",
            "tool":     tool_key,
            "original": scenario_tuple[0],
        }
    }


def _qc(example: dict) -> tuple[bool, str]:
    msgs = example["messages"]
    for m in msgs:
        if not m.get("content", "").strip():
            return False, "Empty content"
        if re.search(r'[\u4e00-\u9fff]', m["content"]):
            return False, "Chinese characters"

    # El TOOL_CALL debe tener formato correcto
    tool_key = example["_debug"]["tool"]
    if tool_key not in ("no_tool", None):
        tool_msgs = [m for m in msgs if m["role"] == "assistant"
                     and "TOOL_CALL:" in m["content"]]
        if not tool_msgs:
            return False, "Missing TOOL_CALL in assistant message"
        try:
            tc_str = tool_msgs[0]["content"].replace("TOOL_CALL:", "").strip()
            tc     = json.loads(tc_str)
            if "tool" not in tc or "args" not in tc:
                return False, "Invalid TOOL_CALL format"
        except json.JSONDecodeError:
            return False, "TOOL_CALL is not valid JSON"

    # La respuesta final debe tener sustancia
    final_msgs = [m for m in msgs
                  if m["role"] == "assistant" and "TOOL_CALL:" not in m["content"]]
    if not final_msgs or len(final_msgs[-1]["content"]) < 20:
        return False, "Final response too short"

    # Detectar gramática rota generada por el LLM al variar respuestas
    final_lower = final_msgs[-1]["content"].lower()
    broken_patterns = [
        "podés que",      # "Podés que te ayude" — mezcla de registros
        "puedo que",      # similar
        "puede que te",   # confusión modal
        "vos me podés",   # construcción rara
    ]
    for pat in broken_patterns:
        if pat in final_lower:
            return False, f"Gramática rota: '{pat}' detectado"

    # Detectar respuestas de confirmación demasiado indirectas
    indirect_confirms = [
        "lo que tenés que hacer es",
        "lo que hay que hacer es",
    ]
    for pat in indirect_confirms:
        if pat in final_lower:
            return False, f"Respuesta demasiado indirecta para confirmación: '{pat}'"

    # Detectar tuteo en respuestas — inconsistente con el voseo del system prompt
    final_text = final_msgs[-1]["content"].lower()
    tuteo_markers = [" tienes ", " puedes ", " debes ", " has ", "tú ", " tu "]
    if any(m in f" {final_text} " for m in tuteo_markers):
        return False, "Tuteo detected — should use voseo"

    # Detectar alucinaciones: la respuesta final no debe inventar info
    # que no esté en el TOOL_RESULT (ej: inventar día de la semana)
    tool_result_msgs = [m for m in msgs if m["role"] == "user"
                        and "TOOL_RESULT:" in m["content"]]
    if tool_result_msgs and final_msgs:
        result_text = tool_result_msgs[-1]["content"].lower()
        final_text_low = final_msgs[-1]["content"].lower()
        # Si el resultado es solo "ok: ..." (sin info extra), la respuesta
        # no debería mencionar días específicos que no estaban en el user request
        if result_text.strip().startswith("tool_result: ok") or            result_text.strip() == "tool_result: ok: evento eliminado" or            result_text.strip() == "tool_result: ok: tarea marcada como completada":
            days = ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"]
            invented = [d for d in days if d in final_text_low
                        and d not in " ".join(m["content"].lower() for m in msgs[:-1])]
            if invented:
                return False, f"Alucinación: inventó días {invented} no presentes en contexto"

    return True, "OK"


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

def generate_agent_dataset(target: int = TARGET) -> list[dict]:
    """
    Genera el dataset del agente distribuyendo ejemplos entre todas las tools.
    Cada escenario base se usa múltiples veces con variaciones.
    """

    # Calcular cuántos ejemplos por tool
    all_tools = list(SCENARIOS.keys())
    per_tool  = target // len(all_tools)
    extra     = target % len(all_tools)

    tool_targets = {t: per_tool for t in all_tools}
    for t in random.sample(all_tools, extra):
        tool_targets[t] += 1

    passed = []
    failed = []

    print(f"\n{'='*60}")
    print(f"  Generando dataset agente | Target: {target}")
    print(f"  Distribución por tool:")
    for t, n in tool_targets.items():
        print(f"    {t:30s}: {n} ejemplos")
    print(f"{'='*60}")

    with tqdm(total=target, desc="agent") as pbar:
        for tool_key, tool_target in tool_targets.items():
            scenarios = SCENARIOS[tool_key]
            generated = 0
            attempts  = 0

            while generated < tool_target:
                attempts += 1
                if attempts > tool_target * 10:
                    print(f"  [WARN] {tool_key}: demasiados intentos, parando")
                    break

                # Rotar escenarios base y variar
                scenario = scenarios[generated % len(scenarios)]
                vary     = attempts > 1  # primera vuelta sin variar, resto con variación

                ex = build_example(scenario, tool_key, vary=vary)
                ok, reason = _qc(ex)

                if ok:
                    ex["_debug"]["qc_reason"] = "OK"
                    passed.append(ex)
                    generated += 1
                    pbar.update(1)

                    # Checkpoint
                    if len(passed) % CHECKPOINT == 0:
                        _checkpoint_save(passed[-CHECKPOINT:])
                else:
                    ex["_debug"]["qc_reason"] = reason
                    failed.append(ex)

    # Guardar el resto
    remainder = len(passed) % CHECKPOINT
    if remainder > 0:
        _checkpoint_save(passed[-remainder:])

    print(f"\n  Passed: {len(passed)} | Failed: {len(failed)}")
    return passed


def _checkpoint_save(examples: list[dict]) -> None:
    with open(OUTPUT_CLEAN, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex["messages"]},
                               ensure_ascii=False) + "\n")
    with open(OUTPUT_DEBUG, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        print("[OK] Ollama conectado")
    except Exception:
        print("[ERROR] Ollama no está corriendo.")
        sys.exit(1)

    # Limpiar archivos previos
    for f in [OUTPUT_CLEAN, OUTPUT_DEBUG]:
        if os.path.exists(f):
            os.remove(f)

    examples = generate_agent_dataset(TARGET)

    # Shuffle final
    random.shuffle(examples)
    with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex["messages"]},
                               ensure_ascii=False) + "\n")

    print(f"\n[DONE] {OUTPUT_CLEAN}: {len(examples)} ejemplos")
    print(f"[DONE] {OUTPUT_DEBUG}: con metadata completa")

    # Distribución final
    counts = {}
    for ex in examples:
        t = ex["_debug"]["tool"]
        counts[t] = counts.get(t, 0) + 1
    print("\nDistribución final:")
    for tool, count in sorted(counts.items()):
        print(f"  {tool:30s}: {count}")


if __name__ == "__main__":
    main()