"""
web_search.py — Búsqueda web con DuckDuckGo (sin API key).

Usa la librería `duckduckgo_search` (ddgs), que funciona en ARM (Rock 5B, Jetson).
Instalación: pip install duckduckgo-search

Retorna resultados limpios listos para inyectar como contexto en el LLM.
"""

import datetime
from typing import Optional

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


def search(query: str, max_results: int = 5, region: str = "es-ar") -> dict:
    """
    Busca en DuckDuckGo.

    Retorna:
        {
            success: bool,
            query: str,
            results: [{title, url, snippet}],
            context: str,   ← texto listo para inyectar al LLM
            error: str      ← solo si success=False
        }
    """
    if not _DDGS_AVAILABLE:
        return {
            "success": False,
            "query":   query,
            "results": [],
            "context": "",
            "error":   "duckduckgo_search no está instalado. Corré: pip install duckduckgo-search",
        }

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, region=region, max_results=max_results))
    except Exception as e:
        return {
            "success": False,
            "query":   query,
            "results": [],
            "context": "",
            "error":   str(e),
        }

    results = []
    for r in raw:
        results.append({
            "title":   r.get("title", ""),
            "url":     r.get("href", ""),
            "snippet": r.get("body", ""),
        })

    # Construir contexto para el LLM
    context_parts = [
        f"Resultados de búsqueda para: \"{query}\" ({datetime.date.today().isoformat()})\n"
    ]
    for i, r in enumerate(results, 1):
        context_parts.append(f"[{i}] {r['title']}\n{r['snippet']}\nFuente: {r['url']}\n")

    return {
        "success": True,
        "query":   query,
        "results": results,
        "context": "\n".join(context_parts),
        "error":   None,
    }


def refine_query(raw_query: str, client, model: str) -> str:
    """
    Usa el LLM para reformular una query vaga en algo más buscable.
    Llamar cuando el usuario dice algo como "buscame eso de antes".
    """
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
    """Formatea resultados para mostrar al usuario antes de entrar en modo tutor."""
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