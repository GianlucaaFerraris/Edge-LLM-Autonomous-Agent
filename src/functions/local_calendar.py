"""
local_calendar.py

Calendario local usando SQLite — reemplaza Google Calendar.
Sin OAuth, sin red, sin dependencias externas.
Corre 100% en la SBC.

Base de datos: ~/.local/share/edge_assistant/calendar.db

Uso desde el dispatcher del agente:
    from local_calendar import cal_add, cal_list, cal_delete

Uso directo para testing:
    python local_calendar.py
"""

import sqlite3
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_DIR  = Path.home() / ".local" / "share" / "edge_assistant"
DB_PATH = DB_DIR / "calendar.db"


def _get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT NOT NULL,
            datetime  TEXT NOT NULL,
            duration  INTEGER DEFAULT 60,
            notes     TEXT DEFAULT ''
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Parseo de fechas en español natural
# ---------------------------------------------------------------------------
DAYS_ES = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
}

MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def parse_datetime(text: str) -> str | None:
    """
    Convierte expresiones de fecha/hora en español a ISO 8601.
    Ejemplos:
      "mañana 15:00"         → "2025-03-21 15:00"
      "viernes 18:00"        → "2025-03-22 18:00"
      "2025-03-25 12:00"     → "2025-03-25 12:00"  (ya es ISO, pasa directo)
      "+2h"                  → "2025-03-20 16:35"
      "+30min"               → "2025-03-20 15:05"
      "diario 08:00"         → "diario 08:00"  (pasa directo — APScheduler lo maneja)
    """
    text = text.strip().lower()
    now  = datetime.now()

    # Ya es ISO — pasa directo
    if re.match(r'\d{4}-\d{2}-\d{2}', text):
        return text

    # Relativo: +Nh o +Nmin
    m = re.match(r'\+(\d+)(h|min)', text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta   = timedelta(hours=n) if unit == "h" else timedelta(minutes=n)
        return (now + delta).strftime("%Y-%m-%d %H:%M")

    # Recurrente — devolver tal cual para que APScheduler lo procese
    if text.startswith("diario"):
        return text

    # Extraer hora si existe
    time_match = re.search(r'(\d{1,2})[:\.](\d{2})', text)
    hour, minute = (int(time_match.group(1)), int(time_match.group(2))) \
                   if time_match else (0, 0)

    # Hoy / mañana
    if "hoy" in text:
        base = now
    elif "mañana" in text or "manana" in text:
        base = now + timedelta(days=1)

    # Día de la semana
    elif any(d in text for d in DAYS_ES):
        day_name = next(d for d in DAYS_ES if d in text)
        target_wd = DAYS_ES[day_name]
        current_wd = now.weekday()
        delta_days = (target_wd - current_wd) % 7
        if delta_days == 0:
            delta_days = 7  # si es hoy, ir al próximo
        base = now + timedelta(days=delta_days)

    # Fecha tipo "25 de marzo"
    elif any(m in text for m in MONTHS_ES):
        month_name = next(m for m in MONTHS_ES if m in text)
        month_num  = MONTHS_ES[month_name]
        day_match  = re.search(r'(\d{1,2})', text)
        day_num    = int(day_match.group(1)) if day_match else 1
        year       = now.year if month_num >= now.month else now.year + 1
        base = datetime(year, month_num, day_num)
    else:
        return None  # no se pudo parsear

    return base.replace(hour=hour, minute=minute, second=0,
                        microsecond=0).strftime("%Y-%m-%d %H:%M")


def _parse_date_range(date_str: str) -> tuple[str, str]:
    """
    Convierte un string de fecha/rango a (fecha_inicio, fecha_fin) en ISO.
    Ejemplos:
      "hoy"           → ("2025-03-20", "2025-03-20")
      "mañana"        → ("2025-03-21", "2025-03-21")
      "semana"        → ("2025-03-20", "2025-03-26")
      "fin de semana" → ("2025-03-22", "2025-03-23")
      "jueves"        → ("2025-03-27", "2025-03-27")
    """
    text = (date_str or "hoy").strip().lower()
    now  = datetime.now()

    if "fin de semana" in text:
        # Próximo sábado y domingo
        days_to_sat = (5 - now.weekday()) % 7 or 7
        sat = now + timedelta(days=days_to_sat)
        sun = sat + timedelta(days=1)
        return sat.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")

    if "semana" in text:
        # Lunes a domingo de la semana actual
        mon = now - timedelta(days=now.weekday())
        sun = mon + timedelta(days=6)
        return mon.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")

    # Fecha única (hoy, mañana, día de semana, etc.)
    dt_str = parse_datetime(text + " 00:00")
    if dt_str:
        date_only = dt_str[:10]
        return date_only, date_only

    # Fallback — hoy
    today = now.strftime("%Y-%m-%d")
    return today, today


# ---------------------------------------------------------------------------
# API pública — estas funciones llama el dispatcher del agente
# ---------------------------------------------------------------------------

def cal_add(title: str, datetime_str: str,
            duration_min: int = 60, notes: str = "") -> str:
    """
    Agrega un evento al calendario local.
    Returns: mensaje de confirmación o error.
    """
    dt = parse_datetime(datetime_str)
    if not dt:
        return f"ERROR: no pude interpretar la fecha '{datetime_str}'"

    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO events (title, datetime, duration, notes) VALUES (?,?,?,?)",
            (title, dt, duration_min, notes)
        )
        conn.commit()
        return f"OK: evento '{title}' creado para {dt}"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        conn.close()


def cal_list(date_str: str = "hoy") -> str:
    """
    Lista eventos del calendario para una fecha o rango.
    Returns: string formateado con los eventos, o mensaje vacío.
    """
    start, end = _parse_date_range(date_str)
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT title, datetime, duration, notes
               FROM events
               WHERE date(datetime) BETWEEN ? AND ?
               ORDER BY datetime""",
            (start, end)
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"Sin eventos entre {start} y {end}."

    lines = []
    for row in rows:
        dt    = datetime.fromisoformat(row["datetime"])
        label = dt.strftime("%H:%M")
        dur   = f" ({row['duration']}min)" if row["duration"] else ""
        note  = f" — {row['notes']}" if row["notes"] else ""
        lines.append(f"{label} - {row['title']}{dur}{note}")

    return "\n".join(lines)


def cal_delete(title: str, datetime_str: str) -> str:
    """
    Elimina un evento por título y fecha/hora.
    Returns: confirmación o error.
    """
    dt = parse_datetime(datetime_str)
    if not dt:
        return f"ERROR: no pude interpretar la fecha '{datetime_str}'"

    # Buscar por título y fecha (tolerante a diferencias de minutos)
    date_only = dt[:10]
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT id, title, datetime FROM events
               WHERE lower(title) LIKE ? AND date(datetime) = ?""",
            (f"%{title.lower()}%", date_only)
        ).fetchall()

        if not rows:
            return f"No encontré ningún evento llamado '{title}' para {date_only}."

        # Eliminar el primero que matchee
        conn.execute("DELETE FROM events WHERE id = ?", (rows[0]["id"],))
        conn.commit()
        return f"OK: eliminado '{rows[0]['title']}' del {rows[0]['datetime']}"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        conn.close()


def cal_list_all() -> str:
    """Lista todos los eventos futuros — útil para debug."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT title, datetime, duration, notes
               FROM events
               WHERE datetime >= datetime('now')
               ORDER BY datetime
               LIMIT 20"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "No hay eventos futuros en el calendario."

    lines = []
    for row in rows:
        dt  = row["datetime"]
        dur = f" ({row['duration']}min)" if row["duration"] else ""
        lines.append(f"{dt} - {row['title']}{dur}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI de testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"DB: {DB_PATH}\n")

    # Test básico
    print("── Agregando eventos de prueba ──")
    print(cal_add("Reunión de equipo", "mañana 15:00", duration_min=60))
    print(cal_add("Dentista", "viernes 10:00", duration_min=45))
    print(cal_add("Asado en lo de Pedro", "sábado 13:00", duration_min=180, notes="llevar bebidas"))
    print(cal_add("Entrega proyecto", "viernes 23:59", duration_min=0, notes="deadline"))

    print("\n── Listando hoy ──")
    print(cal_list("hoy"))

    print("\n── Listando esta semana ──")
    print(cal_list("semana"))

    print("\n── Listando viernes ──")
    print(cal_list("viernes"))

    print("\n── Eliminando dentista ──")
    print(cal_delete("Dentista", "viernes 10:00"))

    print("\n── Todos los eventos futuros ──")
    print(cal_list_all())