# Persistent Storage Subsystems

> **Files:** `task_manager.py`, `local_calendar.py`, `reminder_manager.py`, `wa_stub.py`  
> **Backend:** SQLite3 (stdlib, zero dependencies)  
> **Database path:** `<module_dir>/db/*.db`  
> **Design constraint:** 100% local, no network, no OAuth, no cloud sync

---

## 1. Design Rationale

All persistent state in Agenty is stored in local SQLite databases. This is a deliberate architectural choice for an edge device:

- **No network dependency.** The assistant must function with zero internet connectivity. Google Calendar, Todoist, and similar cloud services require OAuth flows, API keys, and network access — all of which add failure modes to an autonomous SBC.
- **No server process.** SQLite is an embedded database — it's a library, not a server. There is no daemon to start, no port to manage, no connection pooling. The Python `sqlite3` module is part of the standard library.
- **Crash resilience.** SQLite uses WAL (Write-Ahead Logging) by default, providing ACID transactions. A power failure mid-write won't corrupt the database.
- **Minimal resource overhead.** Each database file is typically <1 MB. SQLite's memory footprint for these workloads is negligible (<5 MB total across all databases).

---

## 2. Task Manager (`task_manager.py`)

### 2.1 Schema

```sql
CREATE TABLE tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    priority    TEXT DEFAULT 'media',    -- 'alta' | 'media' | 'baja'
    done        INTEGER DEFAULT 0,       -- 0=pending, 1=completed
    created_at  TEXT NOT NULL,           -- ISO 8601
    done_at     TEXT                     -- ISO 8601, NULL until completed
);
```

### 2.2 API

| Function | Signature | Description |
|---|---|---|
| `add` | `(title: str, priority: str) → int` | Creates task, returns auto-incremented ID |
| `list_pending` | `() → list[dict]` | All tasks where `done=0`, sorted by priority |
| `done` | `(task_id: int) → bool` | Sets `done=1` and `done_at`, returns True if task existed |
| `delete` | `(task_id: int) → bool` | Hard delete, returns True if task existed |
| `search` | `(query: str) → list[dict]` | Case-insensitive LIKE search on title |
| `format_list` | `(tasks: list[dict]) → str` | Human-readable formatting with priority icons |

### 2.3 Priority System

Three levels with visual indicators:
- 🔴 `alta` (high)
- 🟡 `media` (medium) — default
- 🟢 `baja` (low)

Sorting is handled in Python after the SQL query via a map (`alta→0, media→1, baja→2`). This keeps the SQL simple and avoids database-level enum constraints.

### 2.4 Connection Pattern

Every operation opens a fresh connection via `_conn()`, wrapped in a `with` statement for automatic commit/rollback:

```python
def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
```

This pattern avoids long-lived connections (which can cause "database is locked" errors in multi-threaded scenarios) while maintaining simplicity. For the workload profile (a few reads/writes per minute at most), connection overhead is negligible.

---

## 3. Local Calendar (`local_calendar.py`)

### 3.1 Schema

```sql
CREATE TABLE events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    date             TEXT NOT NULL,       -- YYYY-MM-DD
    start_time       TEXT NOT NULL,       -- HH:MM
    end_time         TEXT NOT NULL,       -- HH:MM
    description      TEXT DEFAULT '',
    recurrence_group INTEGER DEFAULT NULL -- links recurring events
);
```

### 3.2 Natural Language Date Parsing

The calendar implements a custom date parser (`_parse_date()`) that handles:

| Input | Resolved to |
|---|---|
| `"hoy"` / `"today"` | `date.today()` |
| `"mañana"` / `"tomorrow"` | `today + 1 day` |
| `"el lunes"` / `"martes"` etc. | Next occurrence of that weekday |
| `"lunes que viene"` | Next occurrence (same as above, "que viene" stripped) |
| `"2025-03-20"` | ISO date directly |
| `"20/03/2025"` | DD/MM/YYYY format |

Spanish day names are mapped via a dict:
```python
DAYS_ES = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
}
```

Note the duplicate entries without accent marks — this handles cases where the STT transcription or user input omits diacritics.

### 3.3 Conflict Detection

When adding an event, the calendar checks for time overlaps:

```
Existing:  [10:00 ─── 11:30]
New:              [11:00 ─── 12:00]
           → Conflict detected
```

The overlap check compares `start_time` and `end_time` as string-parsed `datetime.time` objects. Two events conflict if:
```
new_start < existing_end AND new_end > existing_start
```

### 3.4 Free Slot Detection

`find_free_slots(date, min_duration_minutes)` scans a day's events and returns available time blocks:

```
Events on 2025-03-20:
  [09:00-10:00] Meeting
  [14:00-15:30] Call

Working hours: [08:00 - 20:00]

Free slots (≥60 min):
  🟢 08:00 – 09:00 (60 min)
  🟢 10:00 – 14:00 (240 min)
  🟢 15:30 – 20:00 (270 min)
```

This feeds into the conflict resolution flow: when `cal_add` detects a conflict, it calls `suggest_slot()` to propose the nearest available block.

### 3.5 Recurring Events

`add_recurring(title, weekday, start, end, from_date, until_date)` creates multiple events linked by a `recurrence_group` ID:

```
add_recurring("Standup", "lunes", "09:00", "09:30", "2025-01-06", "2025-03-31")
→ Creates 13 events (every Monday), all sharing the same recurrence_group
```

This "materialized recurrence" approach (storing individual events rather than a recurrence rule) simplifies querying and deletion at the cost of slightly more storage — acceptable for personal calendar volumes.

---

## 4. Reminder Manager (`reminder_manager.py`)

### 4.1 Schema

```sql
CREATE TABLE reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    remind_at   TEXT NOT NULL,    -- ISO 8601 datetime
    fired       INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);
```

### 4.2 Scheduler Architecture

The reminder system uses a **polling scheduler** on a daemon thread:

```
┌─────────────────────────────────────────────┐
│           MAIN THREAD                       │
│  ┌──────────────────────────────────────┐   │
│  │  On each turn: pop_alerts()          │   │
│  │  → Drain _pending_alerts queue       │   │
│  │  → Display/speak each alert          │   │
│  └──────────────────────────────────────┘   │
└─────────────────────┬───────────────────────┘
                      │ reads (thread-safe)
┌─────────────────────▼───────────────────────┐
│          DAEMON THREAD                      │
│  _scheduler_loop(interval=1800s):           │
│    1. Query: remind_at <= now + 35min       │
│       AND fired = 0                         │
│    2. Mark matched as fired = 1             │
│    3. Append to _pending_alerts             │
│    4. Call _alert_callback if registered    │
│    5. Sleep 30 minutes                      │
│    6. Repeat                                │
└─────────────────────────────────────────────┘
```

**Thread safety:** The `_pending_alerts` list is protected by a `threading.Lock`. The scheduler thread writes under the lock; the main thread reads and clears under the same lock via `pop_alerts()`.

### 4.3 Timing Window

The scheduler checks every 30 minutes with a 35-minute lookahead window (5-minute overlap). This ensures that a reminder set for 14:32 is not missed if the scheduler last ran at 14:00 and next runs at 14:30:

```
14:00 — scheduler runs, checks up to 14:35 → finds reminder at 14:32 → fires it
14:30 — next scheduler run (14:32 already fired=1, no duplicate)
```

### 4.4 Natural Language Datetime Parsing

The `_parse_datetime()` function handles:

| Input | Resolved to |
|---|---|
| `"2025-08-01 15:00"` | Direct ISO parse |
| `"en 2 horas"` | `now + 2h` |
| `"en 30 minutos"` | `now + 30min` |
| `"mañana a las 10"` | `tomorrow @ 10:00` |
| `"el lunes a las 18:30"` | `next Monday @ 18:30` |
| `"mañana"` (no time) | `tomorrow @ 12:00` (default noon) |

The parser first extracts the time component via regex (`a las (\d{1,2}(?::\d{2})?)`), then resolves the date component using the same day-name lookup table as the calendar.

### 4.5 Alert Callback

An optional callback can be registered via `set_alert_callback(fn)`. This allows the orchestrator to receive immediate notification when a reminder fires, rather than waiting for the next `pop_alerts()` call. Currently unused in `main.py` (the polling pattern is sufficient for the 30-minute interval), but available for future real-time notification features.

---

## 5. WhatsApp Stub (`wa_stub.py`)

### 5.1 Purpose

The WhatsApp module is a **fully functional stub** that simulates the send/read API contract. It allows the entire agent pipeline (intent → dispatcher → tool → response) to be tested end-to-end without requiring a real WhatsApp connection.

### 5.2 Contact Storage

Contacts are stored in a JSON file (`db/wa_contacts.json`):

```json
[
    {"name": "Mamá", "phone": "+54911XXXXXXX", "aliases": ["mama", "ma"]},
    {"name": "Juan Manuel", "phone": "+54911XXXXXXX", "aliases": ["juanma", "juan manuel", "juan"]}
]
```

The alias system enables natural-language contact resolution:
- "mamá" → exact alias match → "Mamá"
- "juanma" → alias match → "Juan Manuel"
- "juan" → matches both "Juan Manuel" and potentially "Juan Carlos" → ambiguous → dispatcher asks

### 5.3 API Contract

```python
send(contact_name: str, message: str) → {
    "success": bool,
    "contact": str,         # resolved full name
    "phone": str,
    "message": str,
    "timestamp": str,       # ISO 8601
    "simulated": True,      # flag for UI display
    # OR on failure:
    "error": str,
    "ambiguous": bool,      # True if multiple matches
    "matches": [str],       # list of matching names
}

read(contact_name: str = None) → {
    "success": bool,
    "messages": [{"from": str, "text": str, "time": str}],
    "simulated": True,
}
```

### 5.4 Path to Real Implementation

The stub is designed for drop-in replacement. When `whatsapp-web.js` is integrated:

1. `send()` → HTTP POST to the local whatsapp-web.js server
2. `read()` → HTTP GET from the local whatsapp-web.js server
3. Contact resolution remains the same (local JSON)
4. The dispatcher requires zero changes
5. Remove `"simulated": True` from responses

---

## 6. Database File Locations

All databases are stored in a `db/` subdirectory relative to the module file:

```
src/agent/
  ├── db/
  │   ├── tasks.db
  │   ├── calendar.db
  │   ├── reminders.db
  │   └── wa_contacts.json
  ├── task_manager.py
  ├── local_calendar.py
  ├── reminder_manager.py
  └── wa_stub.py
```

The `db/` directory is auto-created by each module's `_init()` function if it doesn't exist. This means a fresh deployment requires no manual setup — databases are initialized with empty tables on first access.

---

## 7. Data Durability Guarantees

| Subsystem | Persistence | Survives reboot | Survives power failure |
|---|---|---|---|
| Tasks | SQLite WAL | ✅ | ✅ (WAL journaling) |
| Calendar | SQLite WAL | ✅ | ✅ |
| Reminders | SQLite WAL | ✅ | ✅ |
| Contacts | JSON file | ✅ | ✅ (written atomically) |
| Conversation history | In-memory only | ❌ | ❌ |
| Tutor session state | In-memory only | ❌ | ❌ |

This durability profile matches the use case: productivity data (tasks, events, reminders) must survive crashes; conversational context is ephemeral.
