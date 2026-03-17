"""
task_manager.py — Todolist persistente en SQLite.

Operaciones:
    add(title, priority)   → task_id
    list_pending()         → lista de dicts
    done(task_id)          → bool
    delete(task_id)        → bool
    search(query)          → lista de dicts
"""

import sqlite3
import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "db", "tasks.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                priority    TEXT    DEFAULT 'media',
                done        INTEGER DEFAULT 0,
                created_at  TEXT    NOT NULL,
                done_at     TEXT
            )
        """)


_init()


# ── API pública ───────────────────────────────────────────────────────────────

def add(title: str, priority: str = "media") -> int:
    """Agrega una tarea. Retorna el id asignado."""
    priority = priority.lower()
    if priority not in ("alta", "media", "baja"):
        priority = "media"
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (title, priority, created_at) VALUES (?, ?, ?)",
            (title, priority, datetime.datetime.now().isoformat())
        )
        return cur.lastrowid


def list_pending() -> list[dict]:
    """Lista tareas pendientes ordenadas por prioridad."""
    order = {"alta": 0, "media": 1, "baja": 2}
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, priority, created_at FROM tasks WHERE done = 0 ORDER BY created_at"
        ).fetchall()
    tasks = [dict(r) for r in rows]
    tasks.sort(key=lambda t: order.get(t["priority"], 1))
    return tasks


def done(task_id: int) -> bool:
    """Marca una tarea como completada. Retorna True si existía."""
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE tasks SET done = 1, done_at = ? WHERE id = ? AND done = 0",
            (datetime.datetime.now().isoformat(), task_id)
        )
        return cur.rowcount > 0


def delete(task_id: int) -> bool:
    """Elimina una tarea. Retorna True si existía."""
    with _conn() as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount > 0


def search(query: str) -> list[dict]:
    """Busca tareas pendientes por título (case-insensitive)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, priority FROM tasks WHERE done = 0 AND LOWER(title) LIKE ?",
            (f"%{query.lower()}%",)
        ).fetchall()
    return [dict(r) for r in rows]


def format_list(tasks: list[dict]) -> str:
    """Formatea la lista de tareas para mostrar al usuario."""
    if not tasks:
        return "No tenés tareas pendientes."
    icons = {"alta": "🔴", "media": "🟡", "baja": "🟢"}
    lines = []
    for t in tasks:
        icon = icons.get(t["priority"], "⚪")
        lines.append(f"  {icon} #{t['id']} {t['title']} [{t['priority']}]")
    return "Tareas pendientes:\n" + "\n".join(lines)