"""
reminder_manager.py — Recordatorios persistentes + scheduler en background.

El scheduler corre en un thread daemon cada 30 minutos.
Cuando detecta un recordatorio próximo (dentro de los próximos 30 min),
lo encola en `pending_alerts` para que el orquestador lo muestre
al inicio del próximo turno.

Uso:
    from agent.reminder_manager import add, list_upcoming, start_scheduler, pop_alerts
"""

import sqlite3
import datetime
import os
import threading
import time
from typing import Callable, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "db", "reminders.db")

# Cola thread-safe de alertas pendientes para mostrar al usuario
_pending_alerts: list[dict] = []
_alerts_lock = threading.Lock()

# Callback opcional que se llama cuando hay una alerta (para integrar con el orquestador)
_alert_callback: Optional[Callable[[list[dict]], None]] = None


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                remind_at   TEXT    NOT NULL,
                fired       INTEGER DEFAULT 0,
                created_at  TEXT    NOT NULL
            )
        """)


_init()


# ── API pública ───────────────────────────────────────────────────────────────

def add(title: str, datetime_str: str) -> int:
    """
    Agrega un recordatorio.
    datetime_str puede ser ISO (2025-08-01 15:00) o relativo ('mañana a las 10').
    Retorna el id.
    """
    remind_at = _parse_datetime(datetime_str)
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO reminders (title, remind_at, created_at) VALUES (?, ?, ?)",
            (title, remind_at.isoformat(), datetime.datetime.now().isoformat())
        )
        return cur.lastrowid


def list_upcoming(hours: int = 24) -> list[dict]:
    """Lista recordatorios pendientes en las próximas N horas."""
    now    = datetime.datetime.now()
    until  = now + datetime.timedelta(hours=hours)
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE fired = 0 AND remind_at BETWEEN ? AND ? ORDER BY remind_at",
            (now.isoformat(), until.isoformat())
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_pending() -> list[dict]:
    """Lista todos los recordatorios pendientes."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE fired = 0 ORDER BY remind_at"
        ).fetchall()
    return [dict(r) for r in rows]


def delete(reminder_id: int) -> bool:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        return cur.rowcount > 0


def pop_alerts() -> list[dict]:
    """
    Retorna y limpia la cola de alertas pendientes.
    Llamar al inicio de cada turno del orquestador.
    """
    with _alerts_lock:
        alerts = list(_pending_alerts)
        _pending_alerts.clear()
    return alerts


def set_alert_callback(fn: Callable[[list[dict]], None]):
    """Registra un callback que se llama inmediatamente cuando hay alertas."""
    global _alert_callback
    _alert_callback = fn


def format_reminders(reminders: list[dict]) -> str:
    if not reminders:
        return "No tenés recordatorios pendientes."
    lines = []
    for r in reminders:
        dt = datetime.datetime.fromisoformat(r["remind_at"])
        lines.append(f"  ⏰ #{r['id']} {dt.strftime('%d/%m %H:%M')} — {r['title']}")
    return "Recordatorios:\n" + "\n".join(lines)


# ── Scheduler interno ─────────────────────────────────────────────────────────

def _check_reminders():
    """Chequea recordatorios próximos y los encola si corresponde."""
    now      = datetime.datetime.now()
    window   = now + datetime.timedelta(minutes=35)  # margen de 5 min sobre los 30

    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE fired = 0 AND remind_at <= ?",
            (window.isoformat(),)
        ).fetchall()

    if not rows:
        return

    fired_ids = []
    alerts = []
    for row in rows:
        remind_at = datetime.datetime.fromisoformat(row["remind_at"])
        # Solo alertar si está dentro de la ventana Y no es del pasado lejano
        if remind_at >= now - datetime.timedelta(hours=1):
            alerts.append(dict(row))
        fired_ids.append(row["id"])

    if fired_ids:
        with _conn() as conn:
            conn.execute(
                f"UPDATE reminders SET fired = 1 WHERE id IN ({','.join('?'*len(fired_ids))})",
                fired_ids
            )

    if alerts:
        with _alerts_lock:
            _pending_alerts.extend(alerts)
        if _alert_callback:
            try:
                _alert_callback(alerts)
            except Exception:
                pass


def _scheduler_loop(interval_seconds: int = 1800):
    """Loop del scheduler. Corre en thread daemon."""
    while True:
        try:
            _check_reminders()
        except Exception as e:
            print(f"  [REMINDER] Error en scheduler: {e}")
        time.sleep(interval_seconds)


def start_scheduler(interval_minutes: int = 30):
    """Arranca el thread scheduler. Llamar una sola vez al inicio."""
    t = threading.Thread(
        target=_scheduler_loop,
        args=(interval_minutes * 60,),
        daemon=True,
        name="reminder-scheduler"
    )
    t.start()
    # Chequeo inmediato al arrancar
    _check_reminders()
    print(f"  [REMINDER] Scheduler iniciado (cada {interval_minutes} min)")


# ── Helpers de parseo ─────────────────────────────────────────────────────────

_DAYS_ES = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
}


def _parse_datetime(s: str) -> datetime.datetime:
    """
    Parsea strings como:
      '2025-08-01 15:00'
      'mañana a las 10'
      'el lunes a las 18:30'
      'en 2 horas'
    """
    s = s.strip().lower()
    now = datetime.datetime.now()

    # Relativo: "en X horas / minutos"
    if s.startswith("en "):
        parts = s.split()
        try:
            amount = int(parts[1])
            unit   = parts[2] if len(parts) > 2 else "horas"
            if "hora" in unit:
                return now + datetime.timedelta(hours=amount)
            if "minuto" in unit or "min" in unit:
                return now + datetime.timedelta(minutes=amount)
        except (ValueError, IndexError):
            pass

    # Extraer hora si viene "a las HH:MM" o "a las HH"
    time_part = None
    import re
    m = re.search(r'a las (\d{1,2}(?::\d{2})?)', s)
    if m:
        t_str = m.group(1)
        fmt = "%H:%M" if ":" in t_str else "%H"
        time_part = datetime.datetime.strptime(t_str, fmt).time()
        s_base = s[:m.start()].strip()
    else:
        s_base = s

    # Fecha base
    if "hoy" in s_base or not s_base:
        base_date = now.date()
    elif "mañana" in s_base or "manana" in s_base:
        base_date = now.date() + datetime.timedelta(days=1)
    else:
        base_date = None
        for name, wd in _DAYS_ES.items():
            if name in s_base:
                days_ahead = (wd - now.weekday()) % 7 or 7
                base_date = now.date() + datetime.timedelta(days=days_ahead)
                break

    if base_date is None:
        # Intentar ISO directo
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"No pude interpretar la fecha/hora: '{s}'")

    if time_part:
        return datetime.datetime.combine(base_date, time_part)
    else:
        # Sin hora específica → mediodía
        return datetime.datetime.combine(base_date, datetime.time(12, 0))