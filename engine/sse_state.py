from __future__ import annotations

import asyncio
import hashlib
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter()
_snapshot_fn = None


def init_stream(snapshot_provider) -> None:
    global _snapshot_fn
    _snapshot_fn = snapshot_provider


@router.get("/api/ui/state/stream")
async def ui_state_stream(request: Request):
    async def event_gen():
        last_hash = None
        last_keepalive = time.monotonic()
        while True:
            if await request.is_disconnected():
                break
            snap = None
            if _snapshot_fn is not None:
                try:
                    snap = _snapshot_fn()
                    if asyncio.iscoroutine(snap):
                        snap = await snap
                except Exception:
                    snap = None
            if snap is not None:
                try:
                    payload = json.dumps(snap, ensure_ascii=False, default=str)
                except Exception:
                    payload = None
                if payload:
                    h = hashlib.md5(payload.encode("utf-8")).hexdigest()
                    if h != last_hash:
                        last_hash = h
                        yield f"data: {payload}\n\n"
            if time.monotonic() - last_keepalive >= 15:
                last_keepalive = time.monotonic()
                yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    headers = {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)
