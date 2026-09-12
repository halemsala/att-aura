from __future__ import annotations

from typing import Dict, Optional

from agents.vector_memory import SearchResult, get_vector_memory

DISCLAIMER = "evidencia estatistica auxiliar; nao e recomendacao de aposta"


def build_analog_context(features: Dict, k: Optional[int] = None,
                         vm=None) -> Optional[Dict]:
    mem = vm or get_vector_memory()
    if mem is None or not getattr(mem, "enabled", False):
        return None
    try:
        res = mem.search(features, k=k)
    except Exception as e:
        res = SearchResult(status="error", note=f"rag_exception:{type(e).__name__}")
    ctx = res.to_dict()
    ctx["note"] = f"{res.note} | {DISCLAIMER}" if res.note else DISCLAIMER
    ctx["disclaimer"] = DISCLAIMER
    return ctx


def format_prompt_block(ctx: Optional[Dict]) -> str:
    if not ctx:
        return "[CONTEXTO ANALOGICO] desativado — prosseguir sem contexto análogo."
    status = ctx.get("status", "unknown")
    if status == "ok":
        a = ctx.get("analogous", {})
        total = ctx.get("n_indexed", 0)
        used = a.get("n_used", 0)
        rate = a.get("over_rate")
        rate_txt = f"{rate*100:.1f}%" if isinstance(rate, float) else "n/d"
        lines = [
            "=== CONTEXTO ANALOGICO (RAG local) ===",
            f"Status: ok | Casos indexados: {total} | Análogos usados: {used}",
            f"Similaridade média: {a.get('avg_similarity')} | Confiança: {a.get('confidence')}",
            f"Histórico: over {a.get('wins')} | under {a.get('losses')} | "
            f"push {a.get('pushes')} → taxa over {rate_txt} (excl. push)",
            f"Total final médio de cantos: {a.get('avg_final_corners')} | "
            f"Prob. média emitida: {a.get('avg_analysis_prob')}",
            f"NOTA: {ctx.get('disclaimer', DISCLAIMER)}",
        ]
        return "\n".join(lines)
    if status == "insufficient_evidence":
        return (f"=== CONTEXTO ANALOGICO ===\n"
                f"Status: evidência insuficiente ({ctx.get('note','')}) — "
                f"prosseguir sem contexto análogo.")
    return (f"=== CONTEXTO ANALOGICO ===\n"
            f"Status: {status} ({ctx.get('note','')}) — ignorar contexto, prosseguir.")


def rag_health() -> Dict:
    try:
        return get_vector_memory().health()
    except Exception as e:
        return {"enabled": False, "status": "error", "note": str(e)}
