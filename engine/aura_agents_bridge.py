"""Aura Fase 1 — Bridge anti-shadowing do pacote agents.
No processo do Engine, sys.path pode conter engine/ antes da raiz,
fazendo 'import agents' resolver engine/agents (que nao tem rag_*).
Este bridge registra os modulos da RAIZ em sys.modules (precedencia
total sobre o path), tornando os imports imunes ao shadowing.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "agents"

# ordem importa: vector_memory primeiro (rag_analog/rag_hooks dependem dele)
_MODULES = ("vector_memory", "rag_analog", "rag_backtest", "glm_analysis_agent", "rag_hooks")


def _ensure_pkg() -> None:
    if "agents" not in sys.modules:
        pkg = types.ModuleType("agents")
        pkg.__path__ = [str(AGENTS)]
        sys.modules["agents"] = pkg


def _load(name: str):
    key = f"agents.{name}"
    if key in sys.modules:
        return sys.modules[key]
    _ensure_pkg()
    path = AGENTS / f"{name}.py"
    if not path.exists():
        raise ImportError(f"{path} nao existe")
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure():
    ok = []
    for name in _MODULES:
        try:
            _load(name)
            ok.append(name)
        except Exception as e:
            print(f"[rag_bridge] {name}: {type(e).__name__}: {e}")
    return ok
