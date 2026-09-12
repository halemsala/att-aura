from __future__ import annotations

import json
import re
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
TRACE_FILE = LOG_DIR / "traces.jsonl"
SERVICE = "engine"

_trace_id: ContextVar[Optional[str]] = ContextVar("aura_trace_id", default=None)

_start_ts = time.time()
_lock = threading.Lock()
_counters = {}
_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
_hist = {"buckets": {b: 0 for b in _BUCKETS}, "sum": 0.0, "count": 0}


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _inc(name: str, labels: Optional[dict] = None, value: float = 1.0) -> None:
    key = (name, tuple(sorted((labels or {}).items())))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + value


def _log_line(record: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _norm_path(path: str, route_path: Optional[str]) -> str:
    if route_path:
        return route_path
    p = (path or "/").split("?")[0]
    return re.sub(r"\d+", ":id", p)


class TraceAndMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace = request.headers.get("x-trace-id") or new_trace_id()
        token = _trace_id.set(trace)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Trace-Id"] = trace
            return response
        finally:
            dur = time.perf_counter() - t0
            _trace_id.reset(token)
            route_path = None
            try:
                route_path = getattr(request.scope.get("route"), "path", None)
            except Exception:
                pass
            path = _norm_path(request.url.path, route_path)
            _inc("aura_http_requests_total",
                 {"path": path, "status": str(status), "method": request.method})
            with _lock:
                _hist["sum"] += dur
                _hist["count"] += 1
                for b in _BUCKETS:
                    if dur <= b:
                        _hist["buckets"][b] += 1
            _log_line({
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "svc": SERVICE, "trace_id": trace,
                "method": request.method, "path": path,
                "status": status, "dur_ms": round(dur * 1000, 2),
            })


def _render_metrics() -> str:
    lines = [
        "# HELP aura_uptime_seconds Segundos desde o carregamento do modulo",
        "# TYPE aura_uptime_seconds gauge",
        f"aura_uptime_seconds {round(time.time() - _start_ts, 3)}",
        "# HELP aura_http_requests_total Requisicoes HTTP por rota",
        "# TYPE aura_http_requests_total counter",
    ]
    with _lock:
        for (name, labels), value in sorted(_counters.items()):
            lab = ",".join(k + "=" + chr(34) + v + chr(34) for k, v in labels)
            lines.append(f"{name}{{{lab}}} {int(value)}")
        lines += [
            "# HELP aura_http_request_duration_seconds Duracao das requisicoes",
            "# TYPE aura_http_request_duration_seconds histogram",
        ]
        for b in _BUCKETS:
            lines.append(f"aura_http_request_duration_seconds_bucket{{le=" + chr(34) + f"{b}" + chr(34) + f"}} {_hist['buckets'][b]}")
        lines.append(f"aura_http_request_duration_seconds_bucket{{le=" + chr(34) + "+Inf" + chr(34) + f"}} {_hist['count']}")
        lines.append(f"aura_http_request_duration_seconds_sum {round(_hist['sum'], 6)}")
        lines.append(f"aura_http_request_duration_seconds_count {_hist['count']}")
    return "\n".join(lines) + "\n"


router = APIRouter()


@router.get("/api/obs/metrics")
async def metrics():
    return Response(content=_render_metrics(),
                    media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/api/obs/health")
async def obs_health():
    with _lock:
        return {
            "service": SERVICE,
            "trace_file": str(TRACE_FILE),
            "requests_total": _hist["count"],
            "uptime_sec": round(time.time() - _start_ts, 1),
            "current_trace_id": get_trace_id(),
        }


def install(app) -> bool:
    try:
        app.add_middleware(TraceAndMetricsMiddleware)
        app.include_router(router)
        return True
    except Exception:
        return False
