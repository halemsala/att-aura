"""Aura Fase 2.2 — Backtester: calibracao das probs emitidas vs resultados.

Read-only sobre a base vetorial (vec_cases_meta). Advisory/paper.
Brier = media de (prob - resultado)^2 | 0 = perfeito | 0.25 = chute 50/50.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0 + 1e-9)]


def _fetch_resolved(vm) -> List[Tuple[str, float, int]]:
    out = []
    with vm._lock:
        conn = vm._connect()
        rows = conn.execute(
            "SELECT created_at, analysis_prob, outcome FROM vec_cases_meta "
            "WHERE outcome IN ('win','loss') AND analysis_prob IS NOT NULL "
            "ORDER BY created_at").fetchall()
    for r in rows:
        p = float(r["analysis_prob"])
        if 0.0 <= p <= 1.0:
            out.append((str(r["created_at"]), p, 1 if r["outcome"] == "win" else 0))
    return out


def _brier(samples) -> float:
    return sum((p - y) ** 2 for _, p, y in samples) / len(samples)


def _log_loss(samples, eps: float = 1e-6) -> float:
    total = 0.0
    for _, p, y in samples:
        pc = min(max(p, eps), 1.0 - eps)
        total += -math.log(pc if y == 1 else 1.0 - pc)
    return total / len(samples)


def _reliability(samples) -> List[Dict]:
    bins = []
    for lo, hi in _BINS:
        sel = [(p, y) for _, p, y in samples if lo <= p < hi]
        if sel:
            n = len(sel)
            bins.append({
                "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": n,
                "avg_prob": round(sum(p for p, _ in sel) / n, 3),
                "actual_rate": round(sum(y for _, y in sel) / n, 3),
            })
        else:
            bins.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": 0})
    return bins


def _walk_forward(samples, folds: int = 3) -> Dict:
    n = len(samples)
    if n < folds * 5:
        return {"folds": 0, "note": "amostra insuficiente para walk-forward"}
    size = n // folds
    chunks = []
    for i in range(folds):
        chunk = samples[i * size:(i + 1) * size] if i < folds - 1 else samples[i * size:]
        chunks.append({"fold": i + 1, "n": len(chunk),
                       "brier": round(_brier(chunk), 4)})
    return {"folds": folds, "chunks": chunks}


def backtest_report(vm=None, min_resolved: int = 5) -> Dict:
    if vm is None:
        from agents.vector_memory import get_vector_memory
        vm = get_vector_memory()
    try:
        samples = _fetch_resolved(vm)
    except Exception as e:
        return {"status": "error", "note": f"fetch_error: {e}"}
    try:
        stats = vm.stats()
        outcomes = stats.get("outcomes", {}) or {}
        total = stats.get("total_cases", 0)
    except Exception:
        outcomes, total = {}, 0
    coverage = {
        "total_cases": total,
        "outcomes": outcomes,
        "resolved_com_prob": len(samples),
    }
    if len(samples) < min_resolved:
        return {"status": "insufficient_data",
                "note": f"resolved_com_prob={len(samples)} < min={min_resolved}; "
                        "backtester acorda quando o fluxo resolver outcomes",
                "coverage": coverage}
    return {
        "status": "ok",
        "n": len(samples),
        "brier": round(_brier(samples), 4),
        "brier_reference": {"perfeito": 0.0, "chute_50_50": 0.25},
        "log_loss": round(_log_loss(samples), 4),
        "reliability": _reliability(samples),
        "walk_forward": _walk_forward(samples),
        "coverage": coverage,
        "disclaimer": "medicao de calibracao (paper/advisory); nao e recomendacao",
    }
