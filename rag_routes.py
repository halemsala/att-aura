from __future__ import annotations

import logging

from fastapi import APIRouter, Body

LOG = logging.getLogger("aura.rag.routes")
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
