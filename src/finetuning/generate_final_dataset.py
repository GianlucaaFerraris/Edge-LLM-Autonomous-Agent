"""
create_final_dataset.py

Valida y combina los 3 datasets en uno solo listo para fine-tuning.

Uso:
    python create_final_dataset.py
    python create_final_dataset.py --dry-run   # solo valida, no genera output

Inputs esperados:
    english_tutor_dataset.jsonl   (300 ejemplos)
    engineering_dataset.jsonl     (150 ejemplos)
    agent_dataset_clean.jsonl     (150 ejemplos)

Output:
    finetune_final.jsonl          — solo campo "messages", listo para Unsloth
    finetune_final_report.txt     — reporte de validación
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FILES = {
    "english_tutor": "english_tutor_dataset.jsonl",
    "engineering":   "engineering_dataset.jsonl",
    "agent":         "agent_dataset_clean.jsonl",
}
EXPECTED_COUNTS = {
    "english_tutor": 300,
    "engineering":   400,
    "agent":         150,
}
OUTPUT_JSONL   = "finetune_dataset_clean.jsonl"
OUTPUT_REPORT  = "finetune_final_report.txt"

# ── Validadores por modo ──────────────────────────────────────────────────────

KNOWN_TOOLS = {
    "search_web", "task_add", "task_list", "task_done",
    "reminder_set", "wa_send", "wa_read",
    "cal_add", "cal_list", "cal_delete",
    # herramientas del agente genérico (generate_all_datasets.py)
    "run_command", "read_file", "write_file", "list_files",
    "get_weather", "set_reminder", "memory_store", "memory_recall",
}

TUTEO_HARD = [" tienes ", " puedes ", " debes ", " has "]

def _has_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def _all_content(msgs: list) -> str:
    return " ".join(m["content"] for m in msgs)


def validate_english_tutor(ex: dict, idx: int) -> list[str]:
    errors = []
    msgs   = ex.get("messages", [])

    if len(msgs) != 4:
        errors.append(f"[{idx}] Expected 4 messages, got {len(msgs)}")
        return errors

    roles = [m["role"] for m in msgs]
    if roles != ["system", "assistant", "user", "assistant"]:
        errors.append(f"[{idx}] Wrong role order: {roles}")

    opening  = msgs[1]["content"]
    student  = msgs[2]["content"]
    feedback = msgs[3]["content"]

    if _has_chinese(_all_content(msgs)):
        errors.append(f"[{idx}] Chinese characters")
    if len(opening) < 40:
        errors.append(f"[{idx}] Opening too short ({len(opening)} chars)")
    if len(student) < 60:
        errors.append(f"[{idx}] Student too short ({len(student)} chars)")
    if len(feedback) < 80:
        errors.append(f"[{idx}] Feedback too short ({len(feedback)} chars)")
    if "?" not in feedback:
        errors.append(f"[{idx}] No follow-up question in feedback")
    if re.search(r'[*#`]', feedback):
        errors.append(f"[{idx}] Markdown in feedback")

    return errors


def validate_engineering(ex: dict, idx: int) -> list[str]:
    errors = []
    msgs   = ex.get("messages", [])

    if len(msgs) < 3:
        errors.append(f"[{idx}] Too few messages: {len(msgs)}")
        return errors
    if msgs[0]["role"] != "system":
        errors.append(f"[{idx}] First message not system")
    if _has_chinese(_all_content(msgs)):
        errors.append(f"[{idx}] Chinese characters")

    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    for m in assistant_msgs:
        if len(m["content"]) < 80:
            errors.append(f"[{idx}] Assistant answer too short ({len(m['content'])} chars)")

    return errors


def validate_agent(ex: dict, idx: int) -> list[str]:
    errors = []
    msgs   = ex.get("messages", [])

    if len(msgs) < 3:
        errors.append(f"[{idx}] Too few messages: {len(msgs)}")
        return errors
    if msgs[0]["role"] != "system":
        errors.append(f"[{idx}] First message not system")
    if _has_chinese(_all_content(msgs)):
        errors.append(f"[{idx}] Chinese characters")

    # Validar TOOL_CALLs
    for m in msgs:
        if m["role"] == "assistant" and "TOOL_CALL:" in m["content"]:
            tc_raw = m["content"].replace("TOOL_CALL:", "").strip()
            try:
                tc = json.loads(tc_raw)
                if "tool" not in tc:
                    errors.append(f"[{idx}] TOOL_CALL missing 'tool' key")
                if "args" not in tc:
                    errors.append(f"[{idx}] TOOL_CALL missing 'args' key")
                tool_name = tc.get("tool", "")
                if tool_name and tool_name not in KNOWN_TOOLS:
                    errors.append(f"[{idx}] Unknown tool: '{tool_name}'")
            except json.JSONDecodeError as e:
                errors.append(f"[{idx}] Invalid TOOL_CALL JSON: {e} — content: {tc_raw[:60]}")

    # Detectar tuteo duro en respuestas del agente (que deben ser en voseo)
    # Solo aplica si el system prompt está en español
    system_content = msgs[0]["content"].lower()
    if "respondés" in system_content or "español" in system_content:
        final_msgs = [m for m in msgs
                      if m["role"] == "assistant" and "TOOL_CALL:" not in m["content"]]
        for m in final_msgs:
            text = " " + m["content"].lower() + " "
            found = [t for t in TUTEO_HARD if t in text]
            if found:
                errors.append(f"[{idx}] Tuteo detected in agent response: {found}")

    # Detectar alucinación: respuesta post-OK inventa días de la semana
    for j, m in enumerate(msgs):
        content = m.get("content", "")
        if "TOOL_RESULT:" in content and j + 1 < len(msgs):
            result   = content.lower()
            response = msgs[j + 1]["content"].lower()
            is_ok_only = (
                "ok: evento eliminado" in result or
                "ok: tarea marcada" in result or
                "ok: tarea agregada" in result or
                result.strip() in ("tool_result: ok", "ok")
            )
            if is_ok_only:
                days      = ["lunes", "martes", "miércoles", "jueves",
                             "viernes", "sábado", "domingo"]
                prior_ctx = " ".join(mm["content"].lower()
                                     for mm in msgs[:j]).lower()
                invented  = [d for d in days
                             if d in response and d not in prior_ctx]
                if invented:
                    errors.append(f"[{idx}] Hallucinated days in post-OK response: {invented}")

    return errors


# ── Inferir modo desde system prompt ────────────────────────────────────────
def infer_mode(ex: dict) -> str:
    """Intenta detectar el modo aunque el ejemplo no tenga _debug."""
    msgs = ex.get("messages", [])
    if not msgs:
        return "unknown"
    sp = msgs[0].get("content", "").lower()
    debug_mode = ex.get("_debug", {}).get("mode", "")
    if debug_mode:
        return debug_mode
    if "english tutor" in sp or "spoken english" in sp or "tts-ready" in sp:
        return "english_tutor"
    if "engineering tutor" in sp or "senior engineering" in sp:
        return "engineering"
    if "tool_call" in sp or "herramientas" in sp or "sbc" in sp:
        return "agent"
    return "unknown"


# ── Carga y validación ────────────────────────────────────────────────────────
def load_and_validate(filepath: str, mode: str) -> tuple[list[dict], list[str], dict]:
    """Carga un JSONL, valida cada ejemplo, devuelve (válidos, errores, stats)."""
    path = Path(filepath)
    if not path.exists():
        print(f"  [MISSING] {filepath}")
        return [], [f"File not found: {filepath}"], {}

    examples = []
    parse_errors = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors.append(f"Line {lineno}: JSON parse error — {e}")

    if parse_errors:
        print(f"  [PARSE ERRORS] {filepath}: {len(parse_errors)} líneas inválidas")

    validator = {
        "english_tutor": validate_english_tutor,
        "engineering":   validate_engineering,
        "agent":         validate_agent,
    }.get(mode, lambda ex, idx: [])

    validation_errors = []
    valid_examples    = []
    for i, ex in enumerate(examples):
        errs = validator(ex, i + 1)
        if errs:
            validation_errors.extend(errs)
        else:
            valid_examples.append(ex)

    stats = {
        "total":   len(examples),
        "valid":   len(valid_examples),
        "invalid": len(examples) - len(valid_examples),
        "parse_errors": len(parse_errors),
    }
    return valid_examples, validation_errors + parse_errors, stats


# ── Análisis del dataset final ────────────────────────────────────────────────
def analyze_final(examples: list[dict]) -> dict:
    stats = defaultdict(int)
    seq_lengths = []

    for ex in examples:
        mode = infer_mode(ex)
        stats[f"mode_{mode}"] += 1
        msgs = ex.get("messages", [])
        stats[f"turns_{len(msgs)}"] += 1
        total_chars = sum(len(m["content"]) for m in msgs)
        seq_lengths.append(total_chars)

    stats["avg_chars"]  = int(sum(seq_lengths) / len(seq_lengths)) if seq_lengths else 0
    stats["max_chars"]  = max(seq_lengths) if seq_lengths else 0
    stats["min_chars"]  = min(seq_lengths) if seq_lengths else 0

    # Aproximar tokens (1 token ≈ 4 chars)
    stats["avg_tokens_approx"] = stats["avg_chars"] // 4
    stats["max_tokens_approx"] = stats["max_chars"] // 4

    return dict(stats)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(dry_run: bool) -> None:
    print(f"\n{'='*60}")
    print(f"  CREATE FINAL DATASET")
    print(f"  dry_run={dry_run}")
    print(f"{'='*60}\n")

    all_valid   = []
    all_errors  = []
    report_lines = []

    # ── Cargar y validar cada archivo ────────────────────────────────────────
    for mode, filepath in INPUT_FILES.items():
        print(f"[{mode}] Cargando {filepath}...")
        valid, errors, stats = load_and_validate(filepath, mode)

        expected = EXPECTED_COUNTS.get(mode, 0)
        pct      = stats.get("valid", 0) / expected * 100 if expected else 0
        status   = "✅" if stats.get("valid", 0) >= expected * 0.95 else \
                   "⚠️ " if stats.get("valid", 0) >= expected * 0.80 else "❌"

        print(f"  {status} Total: {stats.get('total', 0)} | "
              f"Válidos: {stats.get('valid', 0)}/{expected} ({pct:.0f}%) | "
              f"Inválidos: {stats.get('invalid', 0)}")

        if errors:
            print(f"  Primeros 5 errores:")
            for e in errors[:5]:
                print(f"    → {e}")
            if len(errors) > 5:
                print(f"    ... y {len(errors) - 5} más")

        all_valid.extend(valid)
        all_errors.extend(errors)

        report_lines.append(f"\n[{mode}] {filepath}")
        report_lines.append(f"  Total: {stats.get('total', 0)}")
        report_lines.append(f"  Válidos: {stats.get('valid', 0)}/{expected} ({pct:.0f}%)")
        report_lines.append(f"  Inválidos: {stats.get('invalid', 0)}")
        report_lines.append(f"  Parse errors: {stats.get('parse_errors', 0)}")
        for e in errors[:20]:
            report_lines.append(f"  ERROR: {e}")

    # ── Estadísticas del dataset combinado ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  DATASET COMBINADO: {len(all_valid)} ejemplos")

    analysis = analyze_final(all_valid)
    mode_lines = [(k, v) for k, v in analysis.items() if k.startswith("mode_")]
    for k, v in sorted(mode_lines):
        print(f"  {k.replace('mode_', ''):20s}: {v}")
    print(f"  {'avg_tokens_approx':20s}: {analysis.get('avg_tokens_approx', 0)}")
    print(f"  {'max_tokens_approx':20s}: {analysis.get('max_tokens_approx', 0)}")

    report_lines.append(f"\n[COMBINADO]")
    report_lines.append(f"  Total válidos: {len(all_valid)}")
    report_lines.append(f"  Total errores: {len(all_errors)}")
    for k, v in sorted(analysis.items()):
        report_lines.append(f"  {k}: {v}")

    # ── Verificar balance ────────────────────────────────────────────────────
    total_expected = sum(EXPECTED_COUNTS.values())
    coverage = len(all_valid) / total_expected * 100
    print(f"\n  Cobertura total: {len(all_valid)}/{total_expected} ({coverage:.0f}%)")

    if coverage < 80:
        print(f"\n  ❌ Dataset incompleto ({coverage:.0f}%). "
              f"Regenerá los archivos faltantes antes de fine-tunear.")
        if not dry_run:
            sys.exit(1)
    elif coverage < 95:
        print(f"\n  ⚠️  Dataset parcial ({coverage:.0f}%). "
              f"Podés continuar pero el modelo puede quedar menos robusto.")
    else:
        print(f"\n  ✅ Dataset completo ({coverage:.0f}%). Listo para fine-tuning.")

    # ── Generar output ────────────────────────────────────────────────────────
    if not dry_run:
        print(f"\n[GENERANDO] Mezclando y guardando en {OUTPUT_JSONL}...")
        random.shuffle(all_valid)

        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for ex in all_valid:
                # Output limpio: solo "messages", sin "_debug"
                clean = {"messages": ex["messages"]}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        size_mb = os.path.getsize(OUTPUT_JSONL) / 1e6
        print(f"  ✅ {OUTPUT_JSONL} — {len(all_valid)} líneas — {size_mb:.1f} MB")

        # Guardar reporte
        with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"  ✅ {OUTPUT_REPORT} — reporte de validación")

        # Mostrar primeras 2 líneas para verificación rápida
        print(f"\n[PREVIEW] Primeros 2 ejemplos del dataset final:")
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                ex = json.loads(line)
                msgs = ex["messages"]
                mode = infer_mode({"messages": msgs})
                print(f"\n  [{i+1}] mode={mode} | turns={len(msgs)}")
                for m in msgs[:2]:
                    preview = m["content"][:80].replace("\n", " ")
                    print(f"    [{m['role']:10s}] {preview}...")

    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Ejemplos válidos:  {len(all_valid)}")
    print(f"  Errores totales:   {len(all_errors)}")
    if not dry_run:
        print(f"  Output:            {OUTPUT_JSONL}")
        print(f"  Reporte:           {OUTPUT_REPORT}")
    print(f"\nSiguiente paso:")
    print(f"  python finetune_qwen.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo valida los archivos sin generar output")
    args = parser.parse_args()
    main(dry_run=args.dry_run)