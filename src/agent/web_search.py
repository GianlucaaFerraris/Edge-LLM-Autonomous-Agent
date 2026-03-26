"""
web_search.py — Búsqueda web con DuckDuckGo (sin API key).

Estrategia en cascada para queries de lugares físicos:
  1. DDG Maps (search_places)     → datos estructurados, sin fetch, ~1s
  2. Si Maps < min_results        → search_and_fetch (DDG text + HTML scraping)
  3. Si fetch falla / JS-only     → snippets DDG como último recurso

Para queries generales (precios, noticias, conceptos):
  - search_and_fetch directamente con fetch_top_n=1

Dependencias:
    pip install ddgs lxml beautifulsoup4
"""

import datetime
from typing import Optional

try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDGS_AVAILABLE = True
    except ImportError:
        _DDGS_AVAILABLE = False

try:
    import lxml  # noqa: F401
    _HTML_PARSER = "lxml"
except ImportError:
    _HTML_PARSER = "html.parser"

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; Googlebot/2.1; "
        "+http://www.google.com/bot.html)"
    ),
    "Accept":          "text/html,application/xhtml+xml",
    "Accept-Language": "es-AR,es;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}

_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "svg", "form", "button", "meta",
]


# ─────────────────────────────────────────────────────────────────────────────
# DDG Maps — datos estructurados
# ─────────────────────────────────────────────────────────────────────────────

def search_places(query: str, location: str = "", max_results: int = 8) -> dict:
    """
    Búsqueda de locales/negocios via DuckDuckGo Maps.
    Retorna datos estructurados: nombre, dirección, teléfono, horarios, rating.
    Sin fetch de páginas — latencia ~1s.

    Retorna:
        {
            success:  bool,
            query:    str,
            location: str,
            places:   [{name, address, phone, category, hours, rating, url, lat, lng}],
            error:    str | None
        }
    """
    if not _DDGS_AVAILABLE:
        return {
            "success": False, "query": query, "location": location,
            "places": [], "error": "ddgs no instalado: pip install ddgs",
        }

    full_query = f"{query} {location}".strip() if location else query

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.maps(full_query, place=location, max_results=max_results))
    except Exception as e:
        return {
            "success": False, "query": query, "location": location,
            "places": [], "error": str(e),
        }

    places = []
    for r in raw:
        hours_raw = r.get("hours", {})
        if isinstance(hours_raw, dict):
            hours_str = ", ".join(f"{k}: {v}" for k, v in hours_raw.items()) if hours_raw else ""
        else:
            hours_str = str(hours_raw) if hours_raw else ""

        places.append({
            "name":     r.get("title", ""),
            "address":  r.get("address", ""),
            "phone":    r.get("phone", ""),
            "category": r.get("category", ""),
            "hours":    hours_str,
            "rating":   r.get("rating"),
            "url":      r.get("url", "") or r.get("website", ""),
            "lat":      r.get("latitude"),
            "lng":      r.get("longitude"),
        })

    # Filtrar resultados sin nombre
    places = [p for p in places if p["name"].strip()]

    return {
        "success":  True,
        "query":    query,
        "location": location,
        "places":   places,
        "error":    None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fetch + extracción de texto
# ─────────────────────────────────────────────────────────────────────────────

def fetch_page_content(url: str, timeout: int = 8, max_chars: int = 4000) -> str:
    """
    Descarga una URL y extrae texto legible via BeautifulSoup.
    Retorna texto truncado a max_chars, o "[FETCH_ERROR] ..." si falla.
    """
    try:
        import requests as _req
        resp = _req.get(url, headers=_FETCH_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        return f"[FETCH_ERROR] {type(e).__name__}: {e}"

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "[FETCH_ERROR] beautifulsoup4 no instalado: pip install beautifulsoup4"

    try:
        soup = BeautifulSoup(resp.text, _HTML_PARSER)
    except Exception as e:
        return f"[FETCH_ERROR] parse error: {e}"

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    raw_lines = soup.get_text(separator="\n", strip=True).splitlines()
    lines = [l.strip() for l in raw_lines if len(l.strip()) > 3]

    # Colapsar duplicados consecutivos (frecuentes en Tripadvisor)
    deduped = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
            prev = line

    text = "\n".join(deduped)
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# DDG text search base
# ─────────────────────────────────────────────────────────────────────────────

def search(query: str, max_results: int = 5, region: str = "es-ar") -> dict:
    """
    Búsqueda de texto DDG. Retorna snippets sin fetch de páginas.
    """
    if not _DDGS_AVAILABLE:
        return {
            "success": False, "query": query, "results": [],
            "context": "", "error": "ddgs no instalado: pip install ddgs",
        }

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, region=region, max_results=max_results))
    except Exception as e:
        return {"success": False, "query": query, "results": [], "context": "", "error": str(e)}

    results = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]

    context_parts = [
        f"Resultados de búsqueda para: \"{query}\" ({datetime.date.today().isoformat()})\n"
    ]
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] {r['title']}\n{r['snippet']}\nFuente: {r['url']}\n")

    return {
        "success": True, "query": query, "results": results,
        "context": "\n".join(context_parts), "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DDG text + fetch de contenido real
# ─────────────────────────────────────────────────────────────────────────────

def search_and_fetch(query: str,
                     max_results: int = 5,
                     fetch_top_n: int = 2,
                     region: str = "es-ar") -> dict:
    """
    Búsqueda DDG + fetch HTML de las top N páginas.
    Fallback automático a snippets si el fetch falla (bot-detection, JS-only).
    """
    base = search(query, max_results=max_results, region=region)
    if not base["success"] or not base["results"]:
        return base

    results     = base["results"]
    fetch_count = min(fetch_top_n, len(results))

    fetched_pages = []
    for i in range(fetch_count):
        url = results[i]["url"]
        print(f"  [SEARCH] Fetching [{i+1}/{fetch_count}]: {url[:70]}...")
        content = fetch_page_content(url)
        if content.startswith("[FETCH_ERROR]"):
            print(f"  [SEARCH] {content}")
            fetched_pages.append(None)
        else:
            print(f"  [SEARCH] {len(content)} chars extraídos.")
            fetched_pages.append(content)

    lines = [
        f"[SEARCH_RESULTS] Query: \"{query}\" — {datetime.date.today().isoformat()}",
        f"Total resultados DDG: {len(results)}",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"{'─'*50}")
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        if r["snippet"]:
            lines.append(f"    Snippet: {r['snippet']}")

        page_idx = i - 1
        if page_idx < len(fetched_pages) and fetched_pages[page_idx]:
            lines.append(f"    [Contenido extraído de la página]:")
            for line in fetched_pages[page_idx].splitlines()[:80]:
                lines.append(f"    {line}")
        lines.append("")

    lines.append("─" * 50)
    lines.append(
        "Con la información anterior, respondé la pregunta del usuario de forma "
        "directa y concreta. Si encontrás nombres de locales, direcciones, ratings "
        "o teléfonos en el contenido de las páginas, usá esos datos reales. "
        "No uses markdown. Respondé en español rioplatense."
    )

    return {
        "success": True, "query": query, "results": results,
        "context": "\n".join(lines), "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formateadores para el LLM
# ─────────────────────────────────────────────────────────────────────────────

def format_places_for_llm(result: dict, max_places: int = 5) -> str:
    """
    Formatea resultado de search_places() como contexto LLM-ready.
    Datos estructurados directos — el LLM puede responder sin inferir nada.
    """
    if not result.get("success"):
        return f"[PLACES_ERROR] {result.get('error', 'error desconocido')}"

    places = result["places"][:max_places]

    if not places:
        return (
            f"[PLACES_EMPTY] DDG Maps no encontró resultados para "
            f"\"{result['query']}\" en {result.get('location', 'la ubicación')}."
        )

    loc = result.get("location", "")
    lines = [
        f"[PLACES_RESULTS] Búsqueda: \"{result['query']}\""
        + (f" — {loc}" if loc else ""),
        f"{len(places)} lugar(es) encontrado(s). Usá estos datos para responder directamente.\n",
    ]

    for i, p in enumerate(places, 1):
        lines.append(f"[{i}] {p['name']}")
        if p["address"]:
            lines.append(f"    Dirección:    {p['address']}")
        if p["phone"]:
            lines.append(f"    Teléfono:     {p['phone']}")
        if p["category"]:
            lines.append(f"    Categoría:    {p['category']}")
        if p["rating"] is not None:
            lines.append(f"    Calificación: {p['rating']}/5")
        if p["hours"]:
            lines.append(f"    Horarios:     {p['hours']}")
        if p["url"]:
            lines.append(f"    Web:          {p['url']}")
        lines.append("")

    lines.append(
        "Respondé al usuario nombrando cada lugar con dirección y teléfono. "
        "No menciones URLs de buscadores. Si algún dato falta, omitilo. "
        "No uses markdown. Respondé en español rioplatense."
    )
    return "\n".join(lines)


def format_results_for_llm(result: dict, max_results: int = 5) -> str:
    """
    Formatea resultado de search() o search_and_fetch() como contexto LLM-ready.
    Si el contexto ya está enriquecido (search_and_fetch), lo retorna directo.
    """
    if not result.get("success"):
        return f"[SEARCH_ERROR] {result.get('error', 'error desconocido')}"

    # search_and_fetch ya construyó el contexto enriquecido
    if result.get("context") and "[SEARCH_RESULTS]" in result["context"]:
        return result["context"]

    # Fallback: bloque básico con snippets
    results = result["results"][:max_results]
    if not results:
        return f"[SEARCH] Sin resultados para: \"{result['query']}\""

    lines = [
        f"[SEARCH_RESULTS] Query: \"{result['query']}\" — {datetime.date.today().isoformat()}",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet'][:600]}")
        if r["url"]:
            lines.append(f"    Fuente: {r['url']}")
        lines.append("")

    lines.append(
        "Respondé la pregunta del usuario con esta información. "
        "Sé directo y concreto. Respondé en español rioplatense."
    )
    return "\n".join(lines)


# ── Legacy ────────────────────────────────────────────────────────────────────

def refine_query(raw_query: str, client, model: str) -> str:
    prompt = (
        f"Reformulá esta consulta para buscarla en internet de forma efectiva.\n"
        f"Consulta original: \"{raw_query}\"\n"
        f"Respondé con SOLO la query reformulada, sin explicaciones, máximo 10 palabras."
    )
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=30,
        )
        return res.choices[0].message.content.strip().strip('"')
    except Exception:
        return raw_query


def format_results_summary(search_result: dict, max_snippets: int = 3) -> str:
    """Legacy — mantenido por compatibilidad."""
    if not search_result.get("success"):
        return f"No pude buscar: {search_result.get('error', 'error desconocido')}"
    results = search_result["results"][:max_snippets]
    if not results:
        return "No encontré resultados."
    lines = [f"Encontré esto sobre \"{search_result['query']}\":\n"]
    for r in results:
        lines.append(f"• {r['title']}")
        if r["snippet"]:
            snippet = r["snippet"][:200] + "..." if len(r["snippet"]) > 200 else r["snippet"]
            lines.append(f"  {snippet}")
    return "\n".join(lines)