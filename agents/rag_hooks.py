"""
Aura Fase 1 — Hooks RAG (integracoes A-D) via wraps fail-safe.
Se qualquer provider/modulo falhar, o sistema original roda intacto.
"""
import logging
from pathlib import Path

LOG = logging.getLogger("aura.rag.hooks")


def _features_from_view(view):
    if not isinstance(view, dict):
        return {}
    f = {}
    lookup = {
        "home": ("home", "mandante"),
        "away": ("away", "visitante"),
        "minute": ("minute", "minuto"),
        "score_home": ("score_home", "placar_home"),
        "score_away": ("score_away", "placar_away"),
        "pressure": ("pressure", "pressure_gauge"),
    }
    for canon, keys in lookup.items():
        for k in keys:
            v = view.get(k)
            if v is not None:
                f[canon] = v
                break
    metrics = view.get("metrics") if isinstance(view.get("metrics"), dict) else {}
    corners = metrics.get("corners") if isinstance(metrics.get("corners"), dict) else {}
    if corners.get("home") is not None:
        f["corners_home"] = corners["home"]
    if corners.get("away") is not None:
        f["corners_away"] = corners["away"]
    if view.get("market"):
        f["market"] = view["market"]
    return f


def _fixture_of(view):
    if isinstance(view, dict):
        return str(view.get("fixture_id") or view.get("match_id") or "")
    return ""


# ------------------------- Hooks A + B (ReasoningEngine) -------------------------
try:
    from agents.rag_analog import build_analog_context, format_prompt_block
    from agents.reasoning_engine import ReasoningEngine

    _orig_build_prompt = ReasoningEngine._build_prompt

    def _build_prompt_with_rag(self, view, feats):
        prompt = _orig_build_prompt(self, view, feats)
        try:
            ctx = None
            if isinstance(view, dict):
                ctx = view.get("analog_context")
                if ctx is None:
                    ctx = build_analog_context(_features_from_view(view))
                    view["analog_context"] = ctx  # cache p/ JSON consolidado
            prompt = prompt + "\n\n" + format_prompt_block(ctx)
        except Exception:
            pass
        return prompt

    ReasoningEngine._build_prompt = _build_prompt_with_rag

    _orig_analyze = ReasoningEngine.analyze

    async def _analyze_with_rag(self, view):
        analysis = await _orig_analyze(self, view)
        try:
            if isinstance(analysis, dict) and isinstance(view, dict):
                analysis.setdefault("analog_context", view.get("analog_context"))
        except Exception:
            pass
        return analysis

    ReasoningEngine.analyze = _analyze_with_rag
    _hook_ab = True
    LOG.info("hooks A/B ativos: _build_prompt + analyze com RAG")
except Exception:
    _hook_ab = False
    LOG.warning("hooks A/B inativos (fail-safe)")

# ------------------------- Hooks C + D (agente GLM) -------------------------
_glm_cls = None
try:
    import agents.glm_analysis_agent as _glm_mod
    for _name in dir(_glm_mod):
        _obj = getattr(_glm_mod, _name)
        if (isinstance(_obj, type)
                and hasattr(_obj, "glm_analyze")
                and hasattr(_obj, "_journal_decision")
                and hasattr(_obj, "resolve_match")):
            _glm_cls = _obj
            break

    if _glm_cls is not None:
        from agents import vector_memory
        from agents.vector_memory import canonical_text, sanitize_features

        def _content_exists(vm, text):
            try:
                with vm._lock:
                    conn = vm._connect()
                    row = conn.execute(
                        "SELECT 1 FROM vec_cases_meta WHERE embed_text=? LIMIT 1",
                        (text,)).fetchone()
                    return row is not None
            except Exception:
                return False

        _orig_journal = _glm_cls._journal_decision

        def _journal_with_rag(self, analysis, view):
            try:
                feats = _features_from_view(view)
                if feats.get("home") and feats.get("away"):
                    vm = vector_memory.get_vector_memory()
                    text = canonical_text(sanitize_features(feats))
                    if not _content_exists(vm, text):
                        prob = None
                        if isinstance(analysis, dict):
                            for k in ("prob", "p_5m", "probability", "p"):
                                v = analysis.get(k)
                                if isinstance(v, (int, float)):
                                    prob = float(v)
                                    break
                        decision = ""
                        if isinstance(analysis, dict):
                            decision = str(analysis.get("decision") or "")
                        case_id = vm.index_case(
                            feats, outcome="pending",
                            extra={"analysis_prob": prob,
                                   "decision": decision,
                                   "round_id": _fixture_of(view)})
                        if isinstance(view, dict):
                            view["rag_case_id"] = case_id
            except Exception:
                pass
            return _orig_journal(self, analysis, view)

        _glm_cls._journal_decision = _journal_with_rag

        _orig_resolve = _glm_cls.resolve_match

        def _map_outcome(outcome):
            if not isinstance(outcome, dict):
                return None, None
            res = str(outcome.get("result") or outcome.get("call")
                      or outcome.get("outcome") or "").strip().lower()
            corners = (outcome.get("final_corners") or outcome.get("corners_total")
                       or outcome.get("total_corners"))
            mapping = {"win": "win", "green": "win", "vitoria": "win",
                       "loss": "loss", "red": "loss", "derrota": "loss",
                       "push": "push", "devolvido": "push",
                       "void": "void", "anulado": "void"}
            return mapping.get(res), corners

        async def _resolve_with_rag(self, fixture_id, outcome):
            try:
                oc, fc = _map_outcome(outcome)
                if oc:
                    vm = vector_memory.get_vector_memory()
                    with vm._lock:
                        conn = vm._connect()
                        rows = conn.execute(
                            "SELECT case_id FROM vec_cases_meta WHERE round_id=?",
                            (str(fixture_id),)).fetchall()
                    for r in rows:
                        vm.update_outcome(int(r["case_id"]), oc, final_corners=fc)
                    if rows:
                        LOG.info("hook D: %d casos do fixture %s -> %s",
                                 len(rows), fixture_id, oc)
            except Exception:
                pass
            return await _orig_resolve(self, fixture_id, outcome)

        _glm_cls.resolve_match = _resolve_with_rag
        _hook_cd = True
        LOG.info("hooks C/D ativos na classe %s", _glm_cls.__name__)
    else:
        _hook_cd = False
except Exception:
    _hook_cd = False
    LOG.warning("hooks C/D inativos (fail-safe)")
