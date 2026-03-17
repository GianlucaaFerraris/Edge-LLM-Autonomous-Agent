"""
local_calendar.py — Calendario local SQLite.

Soporta:
  - Eventos simples
  - Eventos recurrentes (daily / weekly / días específicos en rango)
  - Query de eventos por día o rango
  - Detección de huecos libres
  - Inserción inteligente (propone slot si hay conflicto)
"""

import sqlite3
import datetime
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "db", "calendar.db")

DAYS_ES = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
}


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT    NOT NULL,
                date         TEXT    NOT NULL,
                start_time   TEXT    NOT NULL,
                end_time     TEXT    NOT NULL,
                description  TEXT    DEFAULT '',
                recurrence_group INTEGER DEFAULT NULL
            )
        """)


_init()


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> datetime.date:
    """Parsea fechas: ISO, 'hoy', 'mañana', 'el lunes', 'lunes que viene'."""
    s = date_str.strip().lower()
    today = datetime.date.today()

    if s in ("hoy", "today"):
        return today
    if s in ("mañana", "manana", "tomorrow"):
        return today + datetime.timedelta(days=1)
    if s in ("pasado mañana", "pasado manana"):
        return today + datetime.timedelta(days=2)

    # "el lunes", "lunes que viene", "el próximo martes"
    for name, weekday in DAYS_ES.items():
        if name in s:
            days_ahead = (weekday - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # "el lunes" cuando hoy es lunes → próximo
            if "pasado" in s:
                days_ahead += 7
            return today + datetime.timedelta(days=days_ahead)

    # ISO: YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue

    raise ValueError(f"No pude entender la fecha: '{date_str}'")


def _parse_time(time_str: str) -> datetime.time:
    """Parsea hora: HH:MM, HHhs, 'las 15', etc."""
    s = time_str.strip().lower().replace("hs", "").replace("h", "").strip()
    for fmt in ("%H:%M", "%H"):
        try:
            return datetime.datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"No pude entender la hora: '{time_str}'")


def _to_minutes(t: datetime.time) -> int:
    return t.hour * 60 + t.minute


def _overlaps(s1: str, e1: str, s2: str, e2: str) -> bool:
    """True si los intervalos [s1,e1) y [s2,e2) se solapan."""
    return s1 < e2 and s2 < e1


# ── API pública ───────────────────────────────────────────────────────────────

def add_event(title: str, date: str, start: str, end: str,
              description: str = "", recurrence_group: int = None) -> int:
    """Agrega un evento. Retorna el id. Lanza ValueError si hay conflicto."""
    d = _parse_date(date).isoformat()
    s = _parse_time(start).strftime("%H:%M")
    e = _parse_time(end).strftime("%H:%M")
    if s >= e:
        raise ValueError(f"La hora de inicio ({s}) debe ser anterior al fin ({e}).")

    # Verificar conflictos
    conflicts = get_conflicts(d, s, e)
    if conflicts:
        names = ", ".join(c["title"] for c in conflicts)
        raise ValueError(f"Conflicto con: {names}")

    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO events (title, date, start_time, end_time, description, recurrence_group) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (title, d, s, e, description, recurrence_group)
        )
        return cur.lastrowid


def add_recurring(title: str, weekday_name: str, start: str, end: str,
                  from_date: str, until_date: str,
                  description: str = "") -> list[int]:
    """
    Agrega eventos recurrentes para un día de la semana en un rango de fechas.
    Ej: todos los lunes de 15:00 a 18:00 desde hoy hasta el 1 de agosto.
    Retorna lista de ids creados.
    """
    wd_name = weekday_name.lower().strip()
    if wd_name not in DAYS_ES:
        raise ValueError(f"Día no reconocido: '{weekday_name}'. Usá: lunes, martes, etc.")

    target_wd = DAYS_ES[wd_name]
    d_from    = _parse_date(from_date)
    d_until   = _parse_date(until_date)

    if d_from > d_until:
        raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin.")

    # Encontrar el primer día objetivo
    days_ahead = (target_wd - d_from.weekday()) % 7
    current = d_from + datetime.timedelta(days=days_ahead)

    # Obtener un grupo id compartido para todos los eventos de esta serie
    with _conn() as conn:
        row = conn.execute("SELECT MAX(recurrence_group) FROM events").fetchone()
        max_group = row[0] or 0
    group_id = max_group + 1

    ids = []
    skipped = []
    while current <= d_until:
        try:
            eid = add_event(title, current.isoformat(), start, end, description, group_id)
            ids.append(eid)
        except ValueError as e:
            skipped.append(f"{current.isoformat()}: {e}")
        current += datetime.timedelta(weeks=1)

    if skipped:
        print(f"  [CAL] {len(skipped)} fechas omitidas por conflicto.")
    return ids


def list_events(date: str = None, date_from: str = None, date_to: str = None) -> list[dict]:
    """Lista eventos de un día o rango. Default: hoy."""
    with _conn() as conn:
        if date_from and date_to:
            d_from = _parse_date(date_from).isoformat()
            d_to   = _parse_date(date_to).isoformat()
            rows = conn.execute(
                "SELECT * FROM events WHERE date BETWEEN ? AND ? ORDER BY date, start_time",
                (d_from, d_to)
            ).fetchall()
        else:
            d = _parse_date(date or "hoy").isoformat()
            rows = conn.execute(
                "SELECT * FROM events WHERE date = ? ORDER BY start_time",
                (d,)
            ).fetchall()
    return [dict(r) for r in rows]


def delete_event(event_id: int) -> bool:
    """Elimina un evento por id."""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        return cur.rowcount > 0


def delete_series(recurrence_group: int) -> int:
    """Elimina todos los eventos de una serie recurrente. Retorna cantidad."""
    with _conn() as conn:
        cur = conn.execute(
            "DELETE FROM events WHERE recurrence_group = ?", (recurrence_group,)
        )
        return cur.rowcount


def get_conflicts(date: str, start: str, end: str) -> list[dict]:
    """Retorna eventos que se solapan con el intervalo dado."""
    d = _parse_date(date).isoformat() if not date.count("-") == 2 else date
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE date = ? AND start_time < ? AND end_time > ?",
            (d, end, start)
        ).fetchall()
    return [dict(r) for r in rows]


def find_free_slots(date: str, duration_minutes: int = 60,
                    day_start: str = "08:00", day_end: str = "22:00") -> list[dict]:
    """
    Encuentra huecos libres en un día dado.
    Retorna lista de {start, end, duration_minutes}.
    """
    events = list_events(date)
    d_start = _parse_time(day_start)
    d_end   = _parse_time(day_end)

    # Construir timeline de bloques ocupados
    busy = []
    for ev in events:
        s = datetime.time.fromisoformat(ev["start_time"])
        e = datetime.time.fromisoformat(ev["end_time"])
        busy.append((_to_minutes(s), _to_minutes(e)))
    busy.sort()

    # Merge solapados
    merged = []
    for s, e in busy:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Encontrar huecos
    slots = []
    cursor = _to_minutes(d_start)
    end_min = _to_minutes(d_end)

    for bs, be in merged:
        if bs > cursor and (bs - cursor) >= duration_minutes:
            slots.append({
                "start": f"{cursor//60:02d}:{cursor%60:02d}",
                "end":   f"{bs//60:02d}:{bs%60:02d}",
                "duration_minutes": bs - cursor,
            })
        cursor = max(cursor, be)

    if end_min > cursor and (end_min - cursor) >= duration_minutes:
        slots.append({
            "start": f"{cursor//60:02d}:{cursor%60:02d}",
            "end":   f"{end_min//60:02d}:{end_min%60:02d}",
            "duration_minutes": end_min - cursor,
        })

    return slots


def suggest_slot(title: str, duration_minutes: int, preferred_date: str,
                 day_start: str = "08:00", day_end: str = "22:00") -> Optional[dict]:
    """
    Intenta encontrar un slot para un evento de `duration_minutes` en `preferred_date`.
    Si no hay lugar, busca en los próximos 7 días.
    Retorna {date, start, end} o None.
    """
    base_date = _parse_date(preferred_date)
    for delta in range(8):
        check_date = base_date + datetime.timedelta(days=delta)
        slots = find_free_slots(check_date.isoformat(), duration_minutes, day_start, day_end)
        if slots:
            best = slots[0]
            return {
                "date":  check_date.isoformat(),
                "start": best["start"],
                "end":   best["end"],
                "is_original_date": delta == 0,
                "days_shifted": delta,
            }
    return None


def format_events(events: list[dict], show_date: bool = False) -> str:
    """Formatea eventos para mostrar al usuario."""
    if not events:
        return "No hay eventos."
    lines = []
    for ev in events:
        date_prefix = f"{ev['date']} " if show_date else ""
        desc = f" — {ev['description']}" if ev.get("description") else ""
        lines.append(f"  📅 #{ev['id']} {date_prefix}{ev['start_time']}-{ev['end_time']} {ev['title']}{desc}")
    return "\n".join(lines)