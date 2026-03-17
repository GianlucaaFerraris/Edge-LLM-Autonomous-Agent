"""
dispatcher.py — Ejecuta tools del agente y maneja ambigüedad.

Antes de ejecutar cualquier tool, verifica si hay información faltante
y genera preguntas de clarificación cuando es necesario.

Retorna siempre:
    {
        "status":   "ok" | "clarify" | "error" | "web_search",
        "result":   str,           ← texto para mostrar al usuario
        "question": str | None,    ← pregunta de clarificación si status="clarify"
        "tool":     str,           ← nombre de la tool ejecutada
        "data":     dict | None,   ← datos estructurados (para web_search)
    }
"""

import datetime
import re
from typing import Optional

from . import task_manager as tasks
from . import local_calendar as cal
from . import reminder_manager as reminders
from . import wa_stub as wa
from . import web_search as search


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(tool: str, result: str, data: dict = None) -> dict:
    return {"status": "ok", "tool": tool, "result": result, "question": None, "data": data}

def _clarify(tool: str, question: str) -> dict:
    return {"status": "clarify", "tool": tool, "result": "", "question": question, "data": None}

def _error(tool: str, msg: str) -> dict:
    return {"status": "error", "tool": tool, "result": msg, "question": None, "data": None}

def _web(tool: str, result: str, data: dict) -> dict:
    return {"status": "web_search", "tool": tool, "result": result, "question": None, "data": data}


# ── Dispatcher principal ──────────────────────────────────────────────────────

def dispatch(tool: str, args: dict) -> dict:
    """
    Ejecuta una tool con sus argumentos.
    Valida y clarifica antes de ejecutar.
    """
    handlers = {
        "task_add":     _task_add,
        "task_list":    _task_list,
        "task_done":    _task_done,
        "cal_add":      _cal_add,
        "cal_add_recurring": _cal_add_recurring,
        "cal_list":     _cal_list,
        "cal_delete":   _cal_delete,
        "cal_free":     _cal_free,
        "reminder_set": _reminder_set,
        "reminder_list": _reminder_list,
        "wa_send":      _wa_send,
        "wa_read":      _wa_read,
        "search_web":   _search_web,
    }
    handler = handlers.get(tool)
    if not handler:
        return _error(tool, f"Tool desconocida: '{tool}'")
    return handler(args)


# ── task ──────────────────────────────────────────────────────────────────────

def _task_add(args: dict) -> dict:
    title    = args.get("title", "").strip()
    priority = args.get("priority", "media")

    if not title:
        return _clarify("task_add", "¿Cómo se llama la tarea que querés agregar?")

    # Detectar ambigüedad: ¿es una tarea o un evento del calendario?
    time_words = ["a las", "el lunes", "el martes", "el miércoles", "mañana", "hoy a"]
    if any(w in title.lower() for w in time_words):
        return _clarify(
            "task_add",
            f"Detecté una hora o fecha en '{title}'. "
            f"¿Querés agregarla como tarea pendiente o como evento al calendario?"
        )

    task_id = tasks.add(title, priority)
    return _ok("task_add", f"Listo, agregué la tarea #{task_id}: \"{title}\" [{priority}].")


def _task_list(args: dict) -> dict:
    pending = tasks.list_pending()
    return _ok("task_list", tasks.format_list(pending))


def _task_done(args: dict) -> dict:
    task_id = args.get("task_id") or args.get("id")

    if task_id is None:
        # Intentar buscar por título
        title = args.get("title", "").strip()
        if title:
            matches = tasks.search(title)
            if not matches:
                return _error("task_done", f"No encontré ninguna tarea con '{title}'.")
            if len(matches) > 1:
                options = "\n".join(f"  #{t['id']} {t['title']}" for t in matches)
                return _clarify("task_done", f"Encontré varias tareas:\n{options}\n¿Cuál querés marcar como hecha? (decime el número)")
            task_id = matches[0]["id"]
        else:
            return _clarify("task_done", "¿Qué tarea querés marcar como completada? Decime el número o el nombre.")

    if tasks.done(int(task_id)):
        return _ok("task_done", f"Tarea #{task_id} marcada como completada. ✅")
    return _error("task_done", f"No encontré la tarea #{task_id}.")


# ── calendar ──────────────────────────────────────────────────────────────────

def _cal_add(args: dict) -> dict:
    title = args.get("title", "").strip()
    date  = args.get("date", "").strip()
    start = args.get("start", "").strip()
    end   = args.get("end", "").strip()
    desc  = args.get("description", "")

    if not title:
        return _clarify("cal_add", "¿Cómo se llama el evento que querés agregar?")
    if not date:
        return _clarify("cal_add", f"¿Para qué día es '{title}'?")
    if not start:
        return _clarify("cal_add", f"¿A qué hora empieza '{title}'?")
    if not end:
        return _clarify("cal_add", f"¿A qué hora termina '{title}'?")

    try:
        event_id = cal.add_event(title, date, start, end, desc)
        parsed_date = cal._parse_date(date)
        return _ok("cal_add",
                   f"Evento agregado: \"{title}\" el {parsed_date.strftime('%d/%m')} "
                   f"de {cal._parse_time(start).strftime('%H:%M')} a {cal._parse_time(end).strftime('%H:%M')}. (ID: {event_id})")
    except ValueError as e:
        err = str(e)
        if "Conflicto" in err:
            # Ofrecer slot alternativo
            try:
                start_t = cal._parse_time(start)
                end_t   = cal._parse_time(end)
                duration = (end_t.hour * 60 + end_t.minute) - (start_t.hour * 60 + start_t.minute)
                suggestion = cal.suggest_slot(title, duration, date)
                if suggestion:
                    if suggestion["is_original_date"]:
                        return _clarify(
                            "cal_add",
                            f"{err}\n"
                            f"Hay un hueco libre ese día a las {suggestion['start']}. ¿Lo agendo ahí?"
                        )
                    else:
                        return _clarify(
                            "cal_add",
                            f"{err}\n"
                            f"El día pedido no tiene lugar. El próximo hueco disponible es el "
                            f"{suggestion['date']} a las {suggestion['start']}. ¿Lo agendo ahí?"
                        )
            except Exception:
                pass
        return _error("cal_add", err)


def _cal_add_recurring(args: dict) -> dict:
    title    = args.get("title", "").strip()
    weekday  = args.get("weekday", "").strip()
    start    = args.get("start", "").strip()
    end      = args.get("end", "").strip()
    from_d   = args.get("from", "hoy")
    until_d  = args.get("until", "").strip()
    desc     = args.get("description", "")

    if not title:
        return _clarify("cal_add_recurring", "¿Cómo se llama el evento recurrente?")
    if not weekday:
        return _clarify("cal_add_recurring", f"¿Qué día de la semana se repite '{title}'?")
    if not start:
        return _clarify("cal_add_recurring", f"¿A qué hora empieza '{title}'?")
    if not end:
        return _clarify("cal_add_recurring", f"¿A qué hora termina '{title}'?")
    if not until_d:
        return _clarify("cal_add_recurring", f"¿Hasta qué fecha se repite '{title}'?")

    try:
        ids = cal.add_recurring(title, weekday, start, end, from_d, until_d, desc)
        return _ok("cal_add_recurring",
                   f"Agregué {len(ids)} eventos: \"{title}\" todos los {weekday} "
                   f"de {start} a {end}, hasta el {until_d}.")
    except ValueError as e:
        return _error("cal_add_recurring", str(e))


def _cal_list(args: dict) -> dict:
    date      = args.get("date", "hoy")
    date_from = args.get("from")
    date_to   = args.get("to")

    try:
        if date_from and date_to:
            events = cal.list_events(date_from=date_from, date_to=date_to)
            header = f"Eventos del {date_from} al {date_to}:"
        else:
            events = cal.list_events(date=date)
            parsed = cal._parse_date(date)
            header = f"Eventos del {parsed.strftime('%d/%m/%Y')}:"

        if not events:
            return _ok("cal_list", f"{header}\n  No hay eventos.")
        return _ok("cal_list", f"{header}\n{cal.format_events(events, show_date=bool(date_from))}")
    except ValueError as e:
        return _error("cal_list", str(e))


def _cal_delete(args: dict) -> dict:
    event_id = args.get("event_id") or args.get("id")
    if not event_id:
        return _clarify("cal_delete", "¿Cuál es el ID del evento que querés eliminar? Pedime la lista primero si no lo sabés.")
    if cal.delete_event(int(event_id)):
        return _ok("cal_delete", f"Evento #{event_id} eliminado.")
    return _error("cal_delete", f"No encontré el evento #{event_id}.")


def _cal_free(args: dict) -> dict:
    date     = args.get("date", "hoy")
    duration = int(args.get("duration_minutes", 60))
    try:
        slots = cal.find_free_slots(date, duration)
        parsed = cal._parse_date(date)
        if not slots:
            return _ok("cal_free", f"No hay huecos de {duration} minutos el {parsed.strftime('%d/%m')}.")
        lines = [f"Huecos libres el {parsed.strftime('%d/%m')} (mínimo {duration} min):"]
        for s in slots:
            lines.append(f"  🟢 {s['start']} – {s['end']} ({s['duration_minutes']} min)")
        return _ok("cal_free", "\n".join(lines))
    except ValueError as e:
        return _error("cal_free", str(e))


# ── reminder ──────────────────────────────────────────────────────────────────

def _reminder_set(args: dict) -> dict:
    title       = args.get("title", "").strip()
    datetime_str = args.get("datetime_str", "").strip()

    if not title:
        return _clarify("reminder_set", "¿De qué te tengo que recordar?")
    if not datetime_str:
        return _clarify("reminder_set", f"¿Para cuándo es el recordatorio de '{title}'?")

    try:
        rid = reminders.add(title, datetime_str)
        return _ok("reminder_set", f"Recordatorio #{rid} configurado: \"{title}\" — {datetime_str}. ⏰")
    except ValueError as e:
        return _error("reminder_set", str(e))


def _reminder_list(args: dict) -> dict:
    pending = reminders.list_all_pending()
    return _ok("reminder_list", reminders.format_reminders(pending))


# ── WhatsApp ──────────────────────────────────────────────────────────────────

def _wa_send(args: dict) -> dict:
    contact = args.get("contact", "").strip()
    message = args.get("message", "").strip()

    if not contact:
        return _clarify("wa_send",
                        f"¿A quién le mando el mensaje?\n{wa.format_contacts()}")
    if not message:
        return _clarify("wa_send", f"¿Qué le mando a {contact}?")

    # Verificar si el contacto es ambiguo
    matches = wa.resolve_contact(contact)
    if not matches:
        return _clarify("wa_send",
                        f"No encontré '{contact}' en tus contactos.\n"
                        f"{wa.format_contacts()}\n¿A quién querés mandarle el mensaje?")
    if len(matches) > 1:
        names = "\n".join(f"  • {m['name']}" for m in matches)
        return _clarify("wa_send",
                        f"Encontré varios contactos que coinciden con '{contact}':\n{names}\n"
                        f"¿A cuál querés mandarle el mensaje?")

    result = wa.send(contact, message)
    if result["success"]:
        return _ok("wa_send",
                   f"Mensaje enviado a {result['contact']}: \"{message}\" ✅"
                   + (" (simulado)" if result.get("simulated") else ""))
    return _error("wa_send", result.get("error", "Error desconocido"))


def _wa_read(args: dict) -> dict:
    contact = args.get("contact", "").strip() or None
    result = wa.read(contact)
    if not result.get("success"):
        return _error("wa_read", "No pude leer los mensajes.")
    msgs = result.get("messages", [])
    if not msgs:
        return _ok("wa_read", "No hay mensajes recientes.")
    lines = ["Mensajes recientes:"]
    for m in msgs:
        lines.append(f"  💬 {m['from']} ({m['time']}): {m['text']}")
    suffix = " (simulado)" if result.get("simulated") else ""
    return _ok("wa_read", "\n".join(lines) + suffix)


# ── web search ────────────────────────────────────────────────────────────────

def _search_web(args: dict) -> dict:
    query = args.get("query", "").strip()
    if not query:
        return _clarify("search_web",
                        "¿Qué querés que busque? Dame más detalles para hacer una buena búsqueda.")

    result = search.search(query)
    summary = search.format_results_summary(result)

    if not result["success"]:
        return _error("search_web", result["error"])

    return _web("search_web", summary, {"search_result": result, "query": query})