from __future__ import annotations

import logging

from fastapi import APIRouter, Body

LOG = logging.getLogger("aura.rag.routes")

# --- FASE 1: bridge anti-shadowing (engine/agents pode sombrear agents/) ---
try:
    from engine import aura_agents_bridge
    aura_agents_bridge.ensure()
except Exception:
    pass
router = APIRouter()


@router.get("/api/rag/health")
async def rag_health():
    try:
        from agents.rag_analog import rag_health
        return rag_health()
    except Exception as e:
        return {"enabled": False, "status": "error", "note": str(e)}


@router.get("/api/rag/stats")
async def rag_stats():
    try:
        from agents.vector_memory import get_vector_memory
        return get_vector_memory().stats()
    except Exception as e:
        return {"status": "error", "note": str(e)}


@router.post("/api/rag/search")
async def rag_search(payload: dict = Body(...)):
    try:
        from agents.rag_analog import build_analog_context
        ctx = build_analog_context(payload or {})
        return ctx if ctx is not None else {"status": "disabled"}
    except Exception as e:
        return {"status": "error", "note": str(e)}



@router.get("/api/rag/backtest")
async def rag_backtest():
    """Fase 2.2: calibracao das probs emitidas (read-only sobre o RAG)."""
    try:
        from agents.rag_backtest import backtest_report
        return backtest_report()
    except Exception as e:
        return {"status": "error", "note": str(e)}
