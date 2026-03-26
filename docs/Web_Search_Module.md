# `web_search` Module

**Location:** `src/agent/web_search.py`  
**Runtime:** Python 3.10+ · ARM (RK3588 / Jetson Orin Nano) · x86_64  
**Dependencies:** `ddgs`, `lxml`, `beautifulsoup4`, `requests`

---

## Overview

`web_search` is the data-retrieval layer for the agent's `search_web` tool. It exposes three search strategies with automatic fallback, and two LLM-ready context formatters. The `dispatcher` calls this module directly — the agent never touches it.

```
User query
    │
    ▼
dispatcher._search_web()
    │
    ├── _is_place_query() == True
    │       │
    │       ├── search_places()          ← DDG Maps, structured data, ~1s
    │       │       │
    │       │       ├── ≥ 3 results ──► format_places_for_llm()  ──► TOOL_RESULT
    │       │       │
    │       │       └── < 3 results ──► search_and_fetch(fetch_top_n=2)
    │       │                                   │
    │       │                                   └──► format_results_for_llm() ──► TOOL_RESULT
    │       │
    │       └── (places error) ──────► search_and_fetch(fetch_top_n=2)
    │
    └── _is_place_query() == False
            │
            └── search_and_fetch(fetch_top_n=1) ──► format_results_for_llm() ──► TOOL_RESULT
```

The `TOOL_RESULT` block is injected into the agent's message history. The LLM reads it on the next loop iteration and synthesizes a final response directly — no separate "web mode" session.

---

## Dependencies

```bash
pip install ddgs lxml beautifulsoup4
```

| Package | Role | ARM note |
|---|---|---|
| `ddgs` | DDG text search + Maps API | Pure Python, no issues |
| `lxml` | HTML parser for BeautifulSoup | ~3× faster than `html.parser` on RK3588. Install via apt if pip fails: `sudo apt install python3-lxml` |
| `beautifulsoup4` | HTML → plain text extraction | Required for `fetch_page_content()` |
| `requests` | HTTP client for page fetch | Standard, already in most venvs |

> **Note:** `ddgs` is the renamed successor of `duckduckgo_search`. The module includes a fallback import for environments still running the old package name.

---

## Public API

### `search_places(query, location="", max_results=8) → dict`

Queries DuckDuckGo Maps and returns structured place data. No HTTP fetch of external pages — DDG returns the data directly.

**When to use:** any query seeking a physical location (restaurant, pharmacy, gym, hotel, etc.).

**Args:**

| Param | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | Full search query including location hint. e.g. `"restaurantes asiáticos Córdoba Capital Argentina"` |
| `location` | `str` | `""` | Geographic bias passed to `ddgs.maps(place=...)`. Redundant if already in `query`, but reinforces the bias. |
| `max_results` | `int` | `8` | Max places to retrieve. DDG Maps may return fewer regardless. |

**Returns:**

```python
{
    "success":  bool,
    "query":    str,
    "location": str,
    "places": [
        {
            "name":     str,    # Business name
            "address":  str,    # Street address — may be empty
            "phone":    str,    # May be empty
            "category": str,    # e.g. "Chinese Restaurant"
            "hours":    str,    # Formatted string, e.g. "Mo-Fr: 12:00-23:00"
            "rating":   float,  # 0.0–5.0, or None if unavailable
            "url":      str,    # Business website, may be empty
            "lat":      float,
            "lng":      float,
        },
        ...
    ],
    "error": str | None
}
```

**Known limitation:** DDG Maps coverage is uneven for mid-size Argentine cities. Central Córdoba has reasonable coverage; peripheral neighborhoods or niche categories may return 0–2 results. The dispatcher's cascade handles this automatically.

---

### `search_and_fetch(query, max_results=5, fetch_top_n=2, region="es-ar") → dict`

Two-stage search: DDG text search for URLs + HTML fetch of the top N pages.

**Stage 1 — DDG text search:** returns `{title, url, snippet}` for `max_results` pages.

**Stage 2 — HTML fetch:** downloads `fetch_top_n` pages, strips noise tags (`<script>`, `<nav>`, `<footer>`, etc.), deduplicates consecutive lines, and truncates to 4000 chars per page.

**Args:**

| Param | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | Search query |
| `max_results` | `int` | `5` | Total DDG results to retrieve |
| `fetch_top_n` | `int` | `2` | Pages to actually fetch and parse. Set to `1` for general queries, `2` for place queries where the first page may fail bot-detection. |
| `region` | `str` | `"es-ar"` | DDG region bias |

**Returns:** same schema as `search()`, with `"context"` replaced by an enriched block containing both snippets and extracted page content. `format_results_for_llm()` detects this and returns it directly.

**Latency budget (RK3588, 100 Mbps link):**

| Stage | Time |
|---|---|
| DDG text search | ~0.5–1.5s |
| fetch × 1 page | ~1–3s |
| fetch × 2 pages | ~2–6s |
| BeautifulSoup parse | ~50–150ms per page |
| **Total (fetch_top_n=1)** | **~2–4s** |
| **Total (fetch_top_n=2)** | **~3–7s** |

**Bot-detection behavior:** Sites like TripAdvisor may return a CAPTCHA or empty HTML. `fetch_page_content()` returns `"[FETCH_ERROR] ..."` in this case. The enriched context block falls back gracefully to the DDG snippet for that URL.

**JS-rendered content:** Sites that build their DOM client-side (React, Vue) return a near-empty HTML shell. `fetch_page_content()` will extract very little useful text. This is the primary failure mode for modern directory sites. DDG snippets are used as the fallback.

---

### `search(query, max_results=5, region="es-ar") → dict`

Plain DDG text search — no page fetch. Returns snippets only.

Used internally by `search_and_fetch()` as Stage 1. Exposed publicly for cases where snippets are sufficient and latency must be minimized.

**Returns:**

```python
{
    "success": bool,
    "query":   str,
    "results": [{"title": str, "url": str, "snippet": str}, ...],
    "context": str,   # legacy formatted block
    "error":   str | None
}
```

---

### `fetch_page_content(url, timeout=8, max_chars=4000) → str`

Downloads a single URL and extracts clean text via BeautifulSoup.

**Args:**

| Param | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | — | Target URL |
| `timeout` | `int` | `8` | HTTP timeout in seconds. Conservative for edge deployments on variable network. |
| `max_chars` | `int` | `4000` | Hard cap on output length. At ~4 chars/token ≈ 1000 tokens — enough for a full list of businesses without saturating the LLM context window. |

**Returns:** extracted text string, or `"[FETCH_ERROR] ..."` on any failure (network, HTTP 4xx/5xx, parse error, missing dependency).

**User-Agent:** Uses a Googlebot UA string. Accepted by most directory and listing sites. Not a guarantee against bot-detection — Cloudflare-protected sites will still block.

---

## Context Formatters

### `format_places_for_llm(result, max_places=5) → str`

Formats `search_places()` output as a structured `TOOL_RESULT` block.

Output example:

```
[PLACES_RESULTS] Búsqueda: "restaurantes asiáticos" — Córdoba Capital, Argentina
4 lugar(es) encontrado(s). Usá estos datos para responder directamente.

[1] China Garden
    Dirección:    Av. Vélez Sársfield 248, Córdoba
    Teléfono:     0351-422-XXXX
    Categoría:    Chinese Restaurant
    Calificación: 4.3/5
    Horarios:     Mo-Su: 12:00-15:00, 20:00-23:30

[2] ...
```

The closing instruction explicitly tells the LLM to respond directly with the data and not reference search engine URLs.

---

### `format_results_for_llm(result, max_results=5) → str`

Formats `search()` or `search_and_fetch()` output as a `TOOL_RESULT` block.

**Detection logic:** if `result["context"]` already contains `"[SEARCH_RESULTS]"` (set by `search_and_fetch()`), it is returned as-is. Otherwise a basic snippet block is constructed. This avoids double-formatting.

---

## Dispatcher Integration

`dispatcher._search_web()` owns the routing logic. `web_search` has no awareness of the agent or the dispatcher — it is a pure data module.

```python
# dispatcher.py — relevant constants
_MAPS_MIN_RESULTS = 3   # threshold to skip fetch fallback
_PLACE_KEYWORDS   = {...} # set used by _is_place_query()
```

The `_PLACE_KEYWORDS` set is a O(n) membership check — effectively O(1) on a set of ~40 strings. No LLM call, no regex — sub-microsecond.

The `USER_LOCATION` constant in `agent_session.py` is injected into the system prompt, which causes the LLM to append the location to place queries before calling `search_web`. The dispatcher does not parse or inject location itself.

---

## Configuration Reference

| Constant | Location | Default | Effect |
|---|---|---|---|
| `USER_LOCATION` | `agent_session.py` | `"Córdoba Capital, Argentina"` | Appended to place queries by the LLM |
| `_MAPS_MIN_RESULTS` | `dispatcher.py` | `3` | Min DDG Maps results to skip fetch fallback |
| `fetch_top_n` (places) | `dispatcher._search_web` | `2` | Pages fetched on Maps fallback |
| `fetch_top_n` (general) | `dispatcher._search_web` | `1` | Pages fetched for non-place queries |
| `max_chars` | `fetch_page_content()` | `4000` | Per-page text cap (~1000 tokens) |
| `timeout` | `fetch_page_content()` | `8s` | HTTP fetch timeout |

---

## Error Handling

| Condition | Behavior |
|---|---|
| `ddgs` not installed | Returns `{"success": False, "error": "ddgs no instalado..."}` — dispatcher returns `_error()` to agent |
| DDG network error | Same as above |
| `beautifulsoup4` not installed | `fetch_page_content()` returns `"[FETCH_ERROR] beautifulsoup4 no instalado"` — context block uses snippet fallback |
| HTTP 4xx / 5xx on fetch | `fetch_page_content()` returns `"[FETCH_ERROR] HTTPError: ..."` — snippet fallback |
| Bot-detection / CAPTCHA | Page parses but extracts near-empty text — snippet fallback (no explicit detection) |
| JS-rendered page | Same as bot-detection case |
| DDG Maps returns 0 results | `search_places()` returns `{"success": True, "places": []}` — dispatcher falls through to `search_and_fetch()` |
| All strategies return empty | `dispatcher._search_web()` returns `_error()` with descriptive message to agent |

All errors are non-fatal and propagate up cleanly. The agent always receives either a usable context block or an error string — no exceptions reach the session loop.

---

## Known Limitations

1. **DDG Maps coverage in Argentina:** Reliable for major cities and central areas. Sparse for suburbs and niche business categories. Mitigated by the `_MAPS_MIN_RESULTS` fallback threshold.

2. **Price freshness:** DDG text snippets are cached. For e-commerce price queries (MercadoLibre, Frávega), prices in snippets may lag the live site by 1–7 days. The system has no mechanism to detect stale prices — the LLM should caveat this when responding.

3. **JS-rendered content:** `fetch_page_content()` uses a static HTTP client. Sites requiring JavaScript execution (most modern SPAs) will return near-empty extracts. Adding Playwright/Pyppeteer would resolve this but introduces unacceptable RAM overhead on a 16 GB SBC running a 7B LLM concurrently.

4. **`lxml` on ARM:** Must be installed via system package manager on some distros:
   ```bash
   sudo apt install python3-lxml
   # then: pip install lxml  (links against system lib)
   ```

---

## Testing

Quick integration test — run from project root:

```bash
# Verify DDG text search
python3 -c "
from src.agent.web_search import search
r = search('air fryer Ninja precio Argentina')
print(r['success'], len(r['results']), 'results')
for x in r['results'][:2]: print(' -', x['title'])
"

# Verify DDG Maps
python3 -c "
from src.agent.web_search import search_places
r = search_places('restaurantes chinos', location='Córdoba Capital, Argentina')
print(r['success'], len(r['places']), 'places')
for p in r['places'][:3]: print(' -', p['name'], '|', p['address'])
"

# Verify fetch pipeline
python3 -c "
from src.agent.web_search import search_and_fetch
r = search_and_fetch('restaurantes asiáticos Córdoba Argentina', fetch_top_n=1)
print(r['success'], len(r['context']), 'chars of context')
"
```

Expected output for a working installation:

```
True 5 results
 - Air Fryer Ninja AF101 - MercadoLibre
 - Frávega - Ninja Air Fryer...

True 4 places
 - China Garden | Av. Vélez Sársfield 248, Córdoba
 - ...

True 3842 chars of context
```