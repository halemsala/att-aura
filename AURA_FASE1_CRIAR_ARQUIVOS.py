# -*- coding: utf-8 -*-
"""
AURA_FASE1_CRIAR_ARQUIVOS.py — Instalador dos arquivos da Fase 1 (Aura).

Uso:
    python AURA_FASE1_CRIAR_ARQUIVOS.py           cria os arquivos que faltam
    python AURA_FASE1_CRIAR_ARQUIVOS.py --force   sobrescreve os existentes

Cria a estrutura completa (relativa a pasta onde este script estiver):
    agents/ vector_memory.py, rag_analog.py
    engine/ rag_routes.py, sse_state.py
    bridge/ telegram_alerts.py
    scripts/ aura_telegram_watchdog.py, aura_rag_migrate_history.py
    desktop/ui/matriz_v22/assets/ aura-stream.js
    tests/ (5 suites)
    skills/aura-rag-analog/SKILL.md
    config/ aura_rag.json, aura_telegram.json, aura_watchdog.json
    AURA_FASE1_INSTALL.bat, AURA_FASE1_TEST.bat, AURA_RAG_MIGRATE.bat,
    AURA_TELEGRAM_WATCHDOG.bat, AURA_FASE1_CHECKFILES.bat
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {}

# ============================================================ agents/vector_memory.py

FILES["agents/vector_memory.py"] = r'''from __future__ import annotations

import collections
import json
import logging
import math
import sqlite3
import struct
import threading
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("aura.rag")

try:
    import sqlite_vec  # pip install sqlite-vec
    HAS_SQLITE_VEC = True
except Exception:
    HAS_SQLITE_VEC = False

ALLOWED_KEYS = (
    "sport", "league", "home", "away", "minute", "score_home", "score_away",
    "corners_home", "corners_away", "line", "line_type", "market",
    "horizon_min", "pressure", "source", "stage",
)
SENSITIVE_HINTS = ("token", "cookie", "password", "senha", "api_key", "secret", "authorization")
RESOLVED_OUTCOMES = {"win", "loss", "push"}
VALID_OUTCOMES = {"win", "loss", "push", "void", "pending", "blocked"}


def _norm_text(s: str) -> str:
    try:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    return s.strip().lower()[:64]


def _int(v):
    try:
        return int(v)
    except Exception:
        return None


def _float(v):
    try:
        return float(v)
    except Exception:
        return None


def sanitize_features(features: Dict) -> Dict:
    out: Dict = {}
    for key, value in (features or {}).items():
        lk = str(key).lower()
        if any(h in lk for h in SENSITIVE_HINTS):
            LOG.warning("RAG: campo sensivel ignorado no embedding: %s", key)
            continue
        if key in ALLOWED_KEYS and value is not None:
            out[key] = _norm_text(value) if isinstance(value, str) else value
    return out


def canonical_text(san: Dict) -> str:
    parts = []
    for k in sorted(san):
        v = san[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def _normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n <= 0.0:
        return list(vec)
    return [x / n for x in vec]


def _post_json(url: str, payload: Dict, timeout: float) -> Optional[Dict]:
    if not url:
        return None
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class OllamaEmbedding:
    def __init__(self, cfg: Dict):
        self.url = str(cfg.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/")
        self.model = cfg.get("ollama_model", "nomic-embed-text")
        self._name = f"ollama/{self.model}"

    @property
    def name(self) -> str:
        return self._name

    def embed(self, text: str, timeout: float) -> Optional[List[float]]:
        v = _post_json(self.url + "/api/embed", {"model": self.model, "input": text}, timeout)
        if v and isinstance(v.get("embeddings"), list) and v["embeddings"]:
            e = v["embeddings"][0]
            if isinstance(e, list) and e:
                return [float(x) for x in e]
        v = _post_json(self.url + "/api/embeddings", {"model": self.model, "prompt": text}, timeout)
        if v and isinstance(v.get("embedding"), list) and v["embedding"]:
            return [float(x) for x in v["embedding"]]
        return None


class LMStudioEmbedding:
    def __init__(self, cfg: Dict):
        self.url = str(cfg.get("lmstudio_url", "http://127.0.0.1:1234/v1")).rstrip("/")
        self.model = cfg.get("lmstudio_model", "text-embedding-nomic-embed-text-v1.5")
        self._name = f"lmstudio/{self.model}"

    @property
    def name(self) -> str:
        return self._name

    def embed(self, text: str, timeout: float) -> Optional[List[float]]:
        v = _post_json(self.url + "/embeddings", {"model": self.model, "input": text}, timeout)
        if v and isinstance(v.get("data"), list) and v["data"]:
            e = v["data"][0].get("embedding")
            if isinstance(e, list) and e:
                return [float(x) for x in e]
        return None


class ProviderChain:
    def __init__(self, cfg: Dict):
        self.timeout = float(cfg.get("timeout_s", 2.0))
        self.providers = []
        for name in cfg.get("provider_order", ["ollama", "lmstudio"]):
            if name == "ollama":
                self.providers.append(OllamaEmbedding(cfg))
            elif name == "lmstudio":
                self.providers.append(LMStudioEmbedding(cfg))

    def embed(self, text: str) -> Tuple[Optional[List[float]], Optional[str]]:
        for p in self.providers:
            try:
                v = p.embed(text, self.timeout)
            except Exception:
                v = None
            if v:
                return v, p.name
        return None, None


class _LRUCache:
    def __init__(self, cap: int):
        self._d = collections.OrderedDict()
        self._cap = max(1, int(cap))

    def get(self, key: str):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self._cap:
            self._d.popitem(last=False)


@dataclass
class SearchResult:
    status: str = "insufficient_evidence"
    note: str = ""
    n_indexed: int = 0
    k_found: int = 0
    n_used: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    blocked: int = 0
    pending: int = 0
    over_rate: Optional[float] = None
    avg_similarity: Optional[float] = None
    avg_final_corners: Optional[float] = None
    avg_analysis_prob: Optional[float] = None
    confidence: float = 0.0
    provider: Optional[str] = None
    cases: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        d["analogous"] = {
            "wins": self.wins, "losses": self.losses, "pushes": self.pushes,
            "n_used": self.n_used, "over_rate": self.over_rate,
            "avg_similarity": self.avg_similarity,
            "avg_final_corners": self.avg_final_corners,
            "avg_analysis_prob": self.avg_analysis_prob,
            "confidence": self.confidence,
        }
        return d


class VectorMemory:
    def __init__(self, db_path, provider, enabled: bool = True, search_k: int = 12,
                 oversample: int = 3, min_cases: int = 5, max_age_days: int = 180,
                 cache_size: int = 256, migrate_batch: int = 2000):
        self.db_path = Path(db_path)
        self.provider = provider
        self.enabled = bool(enabled)
        self.search_k = max(1, int(search_k))
        self.oversample = max(1, int(oversample))
        self.min_cases = max(1, int(min_cases))
        self.max_age_days = int(max_age_days)
        self.migrate_batch = int(migrate_batch)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._cache = _LRUCache(cache_size)
        self._last_provider: Optional[str] = None
        self._vec_ext = HAS_SQLITE_VEC

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=5.0)
        if self._vec_ext:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except Exception:
                self._vec_ext = False
                LOG.warning("RAG: sqlite-vec indisponivel; busca vetorial desativada (meta preservada)")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        self._conn = conn
        self._ensure_meta_tables()
        return conn

    def _ensure_meta_tables(self) -> None:
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS vec_cases_meta (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            embed_text TEXT NOT NULL,
            sport TEXT, league TEXT, home TEXT, away TEXT,
            minute INTEGER, score_home INTEGER, score_away INTEGER,
            corners_home INTEGER, corners_away INTEGER,
            line REAL, line_type TEXT, horizon_min INTEGER,
            pressure REAL, source TEXT, stage TEXT,
            outcome TEXT DEFAULT 'pending',
            final_corners INTEGER,
            analysis_prob REAL, ev REAL, round_id TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            extra_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_meta_outcome ON vec_cases_meta(outcome);
        CREATE TABLE IF NOT EXISTS rag_meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        self._conn.commit()

    def _vec_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_cases'"
        ).fetchone()
        return row is not None

    def _stored_dim(self) -> Optional[int]:
        row = self._conn.execute(
            "SELECT value FROM rag_meta WHERE key='embedding_dim'").fetchone()
        if row is None:
            return None
        try:
            return int(row["value"])
        except Exception:
            return None

    def _create_vec_table(self, dim: int) -> None:
        dim = int(dim)
        if not (1 <= dim <= 4096):
            raise ValueError(f"dim invalida: {dim}")
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_cases "
            f"USING vec0(case_id INTEGER PRIMARY KEY, embedding float[{dim}])")
        self._conn.execute(
            "INSERT OR REPLACE INTO rag_meta(key,value) VALUES('embedding_dim',?)",
            (str(dim),))
        self._conn.commit()

    def _migrate_dim(self, new_dim: int) -> None:
        rows = self._conn.execute(
            "SELECT case_id, embed_text FROM vec_cases_meta LIMIT ?",
            (self.migrate_batch,)).fetchall()
        self._conn.execute("DROP TABLE IF EXISTS vec_cases")
        self._conn.execute("DELETE FROM rag_meta WHERE key='embedding_dim'")
        self._create_vec_table(new_dim)
        reindexed = 0
        for r in rows:
            vec, _ = self._embed(r["embed_text"])
            if vec is None:
                continue
            norm = _normalize(vec)
            blob = struct.pack(f"<{len(norm)}f", *norm)
            self._conn.execute(
                "INSERT OR REPLACE INTO vec_cases(case_id, embedding) VALUES (?,?)",
                (r["case_id"], blob))
            reindexed += 1
        self._conn.commit()
        LOG.info("RAG: migracao de dimensao concluida (%d/%d reindexados)", reindexed, len(rows))

    def _ensure_vec_for(self, dim: int) -> None:
        stored = self._stored_dim()
        if self._vec_exists() and stored == dim:
            return
        if self._vec_exists() and stored is not None and stored != dim:
            self._migrate_dim(dim)
            return
        self._create_vec_table(dim)

    def _embed(self, text: str) -> Tuple[Optional[List[float]], Optional[str]]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached, self._last_provider
        vec, pname = self.provider.embed(text)
        if vec:
            self._cache.put(text, vec)
            self._last_provider = pname
        return vec, pname

    def index_case(self, features: Dict, outcome: str = "pending",
                   extra: Optional[Dict] = None) -> Optional[int]:
        if not self.enabled:
            return None
        try:
            san = sanitize_features(features)
            if not san:
                LOG.warning("RAG: nenhuma feature valida para indexar")
                return None
            text = canonical_text(san)
            with self._lock:
                vec, pname = self._embed(text)
                if vec is None:
                    LOG.warning("RAG: embedding indisponivel; caso nao indexado")
                    return None
                conn = self._connect()
                self._ensure_vec_for(len(vec))
                ex = dict(extra or {})
                cur = conn.execute(
                    """INSERT INTO vec_cases_meta
                    (embed_text, sport, league, home, away, minute, score_home, score_away,
                     corners_home, corners_away, line, line_type, horizon_min, pressure,
                     source, stage, outcome, analysis_prob, ev, round_id, created_at, extra_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (text,
                     san.get("sport"), san.get("league"), san.get("home"), san.get("away"),
                     _int(san.get("minute")), _int(san.get("score_home")), _int(san.get("score_away")),
                     _int(san.get("corners_home")), _int(san.get("corners_away")),
                     _float(san.get("line")), san.get("line_type"), _int(san.get("horizon_min")),
                     _float(san.get("pressure")), san.get("source"), san.get("stage"),
                     outcome if outcome in VALID_OUTCOMES else "pending",
                     _float(ex.pop("analysis_prob", None)), _float(ex.pop("ev", None)),
                     str(ex.pop("round_id", "") or ""),
                     datetime.now().isoformat(timespec="seconds"),
                     json.dumps(ex, ensure_ascii=False, default=str) if ex else None))
                case_id = cur.lastrowid
                norm = _normalize(vec)
                blob = struct.pack(f"<{len(norm)}f", *norm)
                conn.execute("DELETE FROM vec_cases WHERE case_id=?", (case_id,))
                conn.execute("INSERT INTO vec_cases(case_id, embedding) VALUES (?,?)",
                             (case_id, blob))
                conn.commit()
                return int(case_id)
        except Exception:
            LOG.exception("RAG: falha ao indexar caso")
            return None

    def update_outcome(self, case_id: int, outcome: str,
                       final_corners=None, final_score=None) -> bool:
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"outcome invalido: {outcome}")
        try:
            with self._lock:
                conn = self._connect()
                fc = _int(final_corners)
                if final_score is not None and isinstance(final_score, (tuple, list)) and len(final_score) == 2:
                    ex_row = conn.execute(
                        "SELECT extra_json FROM vec_cases_meta WHERE case_id=?",
                        (case_id,)).fetchone()
                    ex = json.loads(ex_row["extra_json"]) if ex_row and ex_row["extra_json"] else {}
                    ex["final_score"] = [_int(final_score[0]), _int(final_score[1])]
                    conn.execute("UPDATE vec_cases_meta SET extra_json=? WHERE case_id=?",
                                 (json.dumps(ex, ensure_ascii=False), case_id))
                resolved = datetime.now().isoformat(timespec="seconds") if outcome in RESOLVED_OUTCOMES else None
                cur = conn.execute(
                    "UPDATE vec_cases_meta SET outcome=?, final_corners=?, resolved_at=? WHERE case_id=?",
                    (outcome, fc, resolved, case_id))
                conn.commit()
                return cur.rowcount > 0
        except Exception:
            LOG.exception("RAG: falha ao atualizar outcome case_id=%s", case_id)
            return False

    def search(self, features: Dict, k: Optional[int] = None) -> SearchResult:
        if not self.enabled:
            return SearchResult(status="disabled", note="rag_disabled")
        try:
            san = sanitize_features(features)
            text = canonical_text(san)
            with self._lock:
                vec, pname = self._embed(text)
                if vec is None:
                    return SearchResult(status="error", note="embedding_unavailable",
                                        provider=pname)
                conn = self._connect()
                total = conn.execute("SELECT COUNT(*) c FROM vec_cases_meta").fetchone()["c"]
                if not self._vec_ext or not self._vec_exists() or self._stored_dim() is None:
                    return SearchResult(status="insufficient_evidence",
                                        note="vec_table_empty_or_unavailable",
                                        n_indexed=total, provider=pname)
                dim = self._stored_dim()
                if len(vec) != dim:
                    return SearchResult(status="error", note="dim_mismatch_reindex_pending",
                                        n_indexed=total, provider=pname)
                norm = _normalize(vec)
                blob = struct.pack(f"<{len(norm)}f", *norm)
                kq = max(10, (k or self.search_k) * self.oversample)
                try:
                    rows = conn.execute(
                        "SELECT rowid, distance FROM vec_cases "
                        "WHERE embedding MATCH :q AND k = :k ORDER BY distance",
                        {"q": blob, "k": kq}).fetchall()
                except sqlite3.OperationalError:
                    return SearchResult(status="error", note="vec_query_error",
                                        n_indexed=total, provider=pname)
                if not rows:
                    return SearchResult(status="insufficient_evidence", note="no_cases",
                                        n_indexed=total, provider=pname)
                ids = [int(r["rowid"]) for r in rows]
                dist = {int(r["rowid"]): float(r["distance"]) for r in rows}
                qmarks = ",".join("?" * len(ids))
                meta_rows = conn.execute(
                    f"SELECT * FROM vec_cases_meta WHERE case_id IN ({qmarks})", ids).fetchall()
                k_target = k or self.search_k
                now = datetime.now()
                cases: List[Dict] = []
                for mr in sorted(meta_rows, key=lambda r: dist.get(r["case_id"], 999.0))[:k_target]:
                    d = dist.get(mr["case_id"], 1.0)
                    sim = max(0.0, min(1.0, 1.0 - (d * d) / 2.0))
                    age_days = None
                    try:
                        age_days = (now - datetime.fromisoformat(mr["created_at"])).days
                    except Exception:
                        pass
                    cases.append({
                        "case_id": mr["case_id"], "league": mr["league"],
                        "home": mr["home"], "away": mr["away"], "minute": mr["minute"],
                        "corners": (mr["corners_home"], mr["corners_away"]),
                        "line": mr["line"], "horizon_min": mr["horizon_min"],
                        "outcome": mr["outcome"], "final_corners": mr["final_corners"],
                        "analysis_prob": mr["analysis_prob"], "ev": mr["ev"],
                        "similarity": round(sim, 3),
                        "created_at": mr["created_at"],
                        "stale": bool(age_days is not None and age_days > self.max_age_days),
                    })
                fresh = [c for c in cases if not c["stale"]]
                resolved = [c for c in fresh if c["outcome"] in RESOLVED_OUTCOMES]
                wins = sum(1 for c in resolved if c["outcome"] == "win")
                losses = sum(1 for c in resolved if c["outcome"] == "loss")
                pushes = sum(1 for c in resolved if c["outcome"] == "push")
                denom = wins + losses
                over_rate = round(wins / denom, 3) if denom > 0 else None
                sims = [c["similarity"] for c in resolved] or [c["similarity"] for c in cases]
                avg_sim = round(sum(sims) / len(sims), 3) if sims else None
                finals = [c["final_corners"] for c in resolved if c["final_corners"] is not None]
                avg_fc = round(sum(finals) / len(finals), 2) if finals else None
                probs = [c["analysis_prob"] for c in resolved if c["analysis_prob"] is not None]
                avg_p = round(sum(probs) / len(probs), 3) if probs else None
                conf = 0.0
                if denom > 0 and avg_sim is not None:
                    conf = round(min(1.0, denom / 10.0) * (0.5 + 0.5 * avg_sim), 3)
                if len(resolved) < self.min_cases:
                    status = "insufficient_evidence"
                    note = f"n_resolved={len(resolved)} < min_cases={self.min_cases}"
                else:
                    status = "ok"
                    note = "evidencia_analogica_disponivel"
                return SearchResult(
                    status=status, note=note, n_indexed=total,
                    k_found=len(rows), n_used=len(cases),
                    wins=wins, losses=losses, pushes=pushes,
                    blocked=sum(1 for c in fresh if c["outcome"] == "blocked"),
                    pending=sum(1 for c in fresh if c["outcome"] == "pending"),
                    over_rate=over_rate, avg_similarity=avg_sim,
                    avg_final_corners=avg_fc, avg_analysis_prob=avg_p,
                    confidence=conf, provider=pname, cases=cases)
        except Exception:
            LOG.exception("RAG: falha na busca")
            return SearchResult(status="error", note="search_exception")

    def stats(self) -> Dict:
        try:
            with self._lock:
                conn = self._connect()
                total = conn.execute("SELECT COUNT(*) c FROM vec_cases_meta").fetchone()["c"]
                by_outcome = {r["outcome"]: r["c"] for r in conn.execute(
                    "SELECT outcome, COUNT(*) c FROM vec_cases_meta GROUP BY outcome")}
                rng = conn.execute(
                    "SELECT MIN(created_at) a, MAX(created_at) b FROM vec_cases_meta").fetchone()
                return {"total_cases": total, "outcomes": by_outcome,
                        "oldest": rng["a"], "newest": rng["b"],
                        "dim": self._stored_dim(), "vec_available": self._vec_ext,
                        "provider": self._last_provider}
        except Exception:
            return {"total_cases": 0, "error": "stats_unavailable"}

    def health(self) -> Dict:
        return {"enabled": self.enabled, "vec_available": self._vec_ext,
                "db": str(self.db_path), **self.stats()}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    @classmethod
    def from_config(cls, cfg: Dict) -> "VectorMemory":
        db = cfg.get("db_path", "data/aura_vector_memory.sqlite3")
        db_path = Path(db) if Path(db).is_absolute() else ROOT / db
        return cls(db_path=db_path, provider=ProviderChain(cfg),
                   enabled=bool(cfg.get("enabled", True)),
                   search_k=cfg.get("search_k", 12),
                   oversample=cfg.get("oversample", 3),
                   min_cases=cfg.get("min_cases_for_stats", 5),
                   max_age_days=cfg.get("max_age_days", 180),
                   cache_size=cfg.get("cache_size", 256),
                   migrate_batch=cfg.get("migrate_batch_limit", 2000))


_INSTANCE = None
_SINGLETON_LOCK = threading.Lock()


def get_vector_memory() -> VectorMemory:
    global _INSTANCE
    if _INSTANCE is None:
        with _SINGLETON_LOCK:
            if _INSTANCE is None:
                cfg: Dict = {}
                try:
                    p = ROOT / "config" / "aura_rag.json"
                    if p.exists():
                        cfg = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    LOG.exception("RAG: config invalida; usando defaults")
                _INSTANCE = VectorMemory.from_config(cfg)
    return _INSTANCE
'''

# ============================================================ agents/rag_analog.py

FILES["agents/rag_analog.py"] = r'''from __future__ import annotations

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
'''

# ============================================================ engine/rag_routes.py

FILES["engine/rag_routes.py"] = r'''from __future__ import annotations

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
'''

# ============================================================ engine/sse_state.py

FILES["engine/sse_state.py"] = r'''from __future__ import annotations

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
'''

# ============================================================ desktop JS

FILES["desktop/ui/matriz_v22/assets/aura-stream.js"] = r'''(function (global) {
  "use strict";

  function AuraStream() {}

  AuraStream.start = function (opts) {
    opts = opts || {};
    var url = opts.url || "";
    var pollUrl = opts.pollUrl || "";
    var onState = typeof opts.onState === "function" ? opts.onState : function () {};
    var pollIntervalMs = opts.pollIntervalMs || 1000;
    var maxReconnectMs = opts.maxReconnectMs || 30000;

    var st = { mode: "idle", es: null, pollTimer: null, stopped: false, attempts: 0 };

    function dispatch(data) {
      try { onState(data); }
      catch (e) { if (global.console && global.console.error) global.console.error("AuraStream onState:", e); }
    }

    function startPolling() {
      if (st.pollTimer || st.stopped) return;
      st.mode = "polling";
      (function tick() {
        if (st.stopped) return;
        fetch(pollUrl, { cache: "no-store" })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (j) { if (j) dispatch(j); })
          .catch(function () {})
          .finally(function () {
            if (!st.stopped) st.pollTimer = setTimeout(tick, pollIntervalMs);
          });
      })();
    }

    function stopPolling() {
      if (st.pollTimer) { clearTimeout(st.pollTimer); st.pollTimer = null; }
    }

    function connect() {
      if (st.stopped) return;
      if (typeof global.EventSource !== "function" || !url) { startPolling(); return; }
      var es;
      try { es = new global.EventSource(url); }
      catch (e) { startPolling(); return; }
      st.es = es;
      st.mode = "connecting";
      es.onopen = function () { st.mode = "stream"; st.attempts = 0; stopPolling(); };
      es.onmessage = function (ev) {
        try { dispatch(JSON.parse(ev.data)); } catch (e) {}
      };
      es.onerror = function () {
        try { es.close(); } catch (e) {}
        st.es = null;
        st.mode = "reconnecting";
        startPolling();
        if (st.stopped) return;
        var delay = Math.min(maxReconnectMs, 1000 * Math.pow(2, st.attempts++));
        setTimeout(connect, delay);
      };
    }

    connect();

    return {
      stop: function () {
        st.stopped = true;
        stopPolling();
        if (st.es) { try { st.es.close(); } catch (e) {} st.es = null; }
      },
      get mode() { return st.mode; }
    };
  };

  global.AuraStream = AuraStream;
})(window);
'''

# ============================================================ bridge/telegram_alerts.py

FILES["bridge/telegram_alerts.py"] = r'''from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("aura.telegram")
LEVELS = {"info": 0, "warn": 1, "error": 2, "critical": 3}


def _http_send(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status < 400


class TelegramAlerter:
    def __init__(self, cfg_path=None, sender: Optional[Callable[[str], bool]] = None):
        self.cfg_path = Path(cfg_path) if cfg_path else ROOT / "config" / "aura_telegram.json"
        self.cfg = self._load()
        self._sender = sender
        self._q = queue.Queue(maxsize=200)
        self._recent = {}
        self._window = deque()
        self._dropped = 0
        self._lock = threading.Lock()
        self._thread = None
        if self.cfg.get("enabled"):
            self._thread = threading.Thread(
                target=self._worker, name="aura-telegram", daemon=True)
            self._thread.start()

    def _load(self) -> dict:
        try:
            if self.cfg_path.exists():
                return json.loads(self.cfg_path.read_text(encoding="utf-8"))
        except Exception:
            LOG.exception("telegram: config invalida")
        return {"enabled": False}

    def alert(self, level: str, title: str, body: str = "") -> bool:
        if not self.cfg.get("enabled"):
            return False
        lvl = level if level in LEVELS else "info"
        min_lvl = self.cfg.get("min_level", "warn")
        if LEVELS[lvl] < LEVELS.get(min_lvl, 1):
            return False
        key = hashlib.md5((lvl + "|" + title).encode("utf-8")).hexdigest()
        now = time.time()
        with self._lock:
            last = self._recent.get(key)
            dd = float(self.cfg.get("dedupe_minutes", 10)) * 60.0
            if last is not None and dd > 0 and now - last < dd:
                return False
            while self._window and now - self._window[0] > 3600.0:
                self._window.popleft()
            if len(self._window) >= int(self.cfg.get("rate_limit_per_hour", 20)):
                self._dropped += 1
                return False
            self._window.append(now)
            self._recent[key] = now
        text = f"[AURA][{lvl.upper()}] {title}"
        if body:
            text += "\n" + str(body)[:3500]
        try:
            self._q.put_nowait((text, key))
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def _worker(self) -> None:
        token = str(self.cfg.get("bot_token", ""))
        chat_id = str(self.cfg.get("chat_id", ""))
        while True:
            item = self._q.get()
            if item is None:
                break
            text, key = item
            ok = False
            for _ in range(2):
                try:
                    if self._sender is not None:
                        ok = bool(self._sender(text))
                    else:
                        ok = _http_send(token, chat_id, text)
                    if ok:
                        break
                except Exception:
                    ok = False
                time.sleep(5)
            if not ok:
                LOG.warning("telegram: falha ao enviar: %s", text[:80])
                with self._lock:
                    self._recent.pop(key, None)

    def stats(self) -> dict:
        with self._lock:
            return {"enabled": bool(self.cfg.get("enabled")),
                    "queued": self._q.qsize(), "dropped": self._dropped}


_INSTANCE = None
_LOCK = threading.Lock()


def get_alerter() -> TelegramAlerter:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = TelegramAlerter()
    return _INSTANCE


def alert(level: str, title: str, body: str = "") -> bool:
    try:
        return get_alerter().alert(level, title, body)
    except Exception:
        return False
'''

# ============================================================ scripts/aura_telegram_watchdog.py

FILES["scripts/aura_telegram_watchdog.py"] = r'''from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SERVICES = [
    {"name": "alfred",  "host": "127.0.0.1", "port": 8791, "path": "/health",     "expect": True},
    {"name": "engine",  "host": "127.0.0.1", "port": 8765, "path": "/api/health", "expect": True},
    {"name": "bridge",  "host": "127.0.0.1", "port": 8080, "path": None,          "expect": True},
    {"name": "jarvis",  "host": "127.0.0.1", "port": 8099, "path": None,          "expect": False},
    {"name": "ollama",  "host": "127.0.0.1", "port": 11434, "path": None,         "expect": False},
    {"name": "hermes",  "host": "127.0.0.1", "port": 8777, "path": "/health",     "expect": False},
]


def _load_config() -> dict:
    p = ROOT / "config" / "aura_watchdog.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _merge_services(cfg: dict) -> List[dict]:
    overrides = {s.get("name"): s for s in cfg.get("services", []) if isinstance(s, dict)}
    out = []
    for svc in DEFAULT_SERVICES:
        merged = dict(svc)
        ov = overrides.get(svc["name"])
        if ov:
            merged.update({k: v for k, v in ov.items() if k != "name"})
        out.append(merged)
    return out


def _probe_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _probe_http(host: str, port: int, path: str, timeout: float = 2.0) -> bool:
    try:
        url = f"http://{host}:{port}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def default_prober(svc: dict) -> bool:
    if not _probe_tcp(svc["host"], svc["port"]):
        return False
    if svc.get("path"):
        return _probe_http(svc["host"], svc["port"], svc["path"])
    return True


class Watchdog:
    def __init__(self, services: List[dict], alerter=None,
                 prober: Optional[object] = None,
                 interval_s: int = 60, remind_after_min: int = 30):
        self.services = services
        self.alerter = alerter
        self.prober = prober or default_prober
        self.interval_s = int(interval_s)
        self.remind_after_s = int(remind_after_min) * 60
        self._prev: Dict[str, bool] = {}
        self._down_since: Dict[str, float] = {}
        self._reminded = set()

    def _notify(self, level: str, title: str, body: str) -> None:
        if self.alerter is None:
            print(f"[watchdog][{level}] {title}: {body}")
            return
        try:
            self.alerter.alert(level, title, body)
        except Exception:
            pass

    def run_once(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        now = time.time()
        for svc in self.services:
            name = svc["name"]
            up = False
            try:
                up = bool(self.prober(svc))
            except Exception:
                up = False
            result[name] = "up" if up else "down"
            was = self._prev.get(name)
            if up:
                if was is False:
                    self._notify("info", "servico_up",
                                 f"{name} ({svc['host']}:{svc['port']}) respondeu novamente")
                self._down_since.pop(name, None)
                self._reminded.discard(name)
            else:
                if was is None:
                    if svc.get("expect", False):
                        self._notify("error", "servico_down",
                                     f"{name} ({svc['host']}:{svc['port']}) sem resposta")
                        self._down_since[name] = now
                    else:
                        self._down_since[name] = now
                elif was is True:
                    self._notify("error", "servico_down",
                                 f"{name} ({svc['host']}:{svc['port']}) caiu")
                    self._down_since[name] = now
                    self._reminded.discard(name)
                else:
                    since = self._down_since.get(name)
                    if (since is not None and self.remind_after_s > 0
                            and now - since >= self.remind_after_s
                            and name not in self._reminded
                            and svc.get("expect", False)):
                        self._notify("warn", "servico_persistindo_down",
                                     f"{name} down ha mais de {self.remind_after_s//60} min")
                        self._reminded.add(name)
            self._prev[name] = up
        return result

    def run_forever(self) -> None:
        print(f"[watchdog] iniciado (intervalo {self.interval_s}s, "
              f"{len(self.services)} servicos, Ctrl+C para sair)")
        while True:
            try:
                st = self.run_once()
                print("[watchdog] " + " ".join(f"{k}={v}" for k, v in st.items()))
            except KeyboardInterrupt:
                print("[watchdog] encerrado")
                return
            except Exception as e:
                print(f"[watchdog] erro no ciclo: {e}")
            time.sleep(self.interval_s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aura watchdog de servicos")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = _load_config()
    if args.config:
        try:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except Exception:
            print(f"[watchdog] config invalida: {args.config}")
            return 2

    services = _merge_services(cfg)
    try:
        from bridge.telegram_alerts import get_alerter
        alerter = get_alerter()
    except Exception:
        alerter = None

    wd = Watchdog(services, alerter=alerter,
                  interval_s=args.interval or cfg.get("interval_s", 60),
                  remind_after_min=cfg.get("remind_after_min", 30))

    if args.once:
        st = wd.run_once()
        for name, status in st.items():
            print(f"{name:8s} {status}")
        return 0 if all(v == "up" for v in st.values()) else 1

    wd.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# ============================================================ scripts/aura_rag_migrate_history.py

FILES["scripts/aura_rag_migrate_history.py"] = r'''from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.vector_memory import VectorMemory, canonical_text, sanitize_features

CANONICAL_ALIASES = {
    "sport": ("esporte", "sport"),
    "league": ("liga", "league", "competition", "comp", "competicao", "campeonato"),
    "home": ("casa", "mandante", "home", "hometeam", "teamhome", "timecasa", "timedacasa"),
    "away": ("fora", "visitante", "away", "awayteam", "teamaway", "timefora"),
    "minute": ("minuto", "minute", "min", "matchminute", "minjogo"),
    "score_home": ("placarcasa", "scorehome", "goalshome", "golscasa", "golcasa"),
    "score_away": ("placarfora", "scoreaway", "goalsaway", "golsfora", "golfora"),
    "corners_home": ("escanteioscasa", "cornershome", "ctcasa", "esccasa", "corncasa"),
    "corners_away": ("escanteiosfora", "cornersaway", "ctfora", "escfora", "cornfora"),
    "line": ("linha", "line", "ahline", "handicap", "linhaasiatica", "asianline"),
    "line_type": ("tipolinha", "linetype"),
    "market": ("mercado", "market", "tipomercado"),
    "horizon_min": ("horizonte", "horizontemin", "horizon", "horizonmin", "janelamin"),
    "pressure": ("pressao", "pressure", "indicepressao"),
    "source": ("fonte", "source", "host", "site", "origem"),
    "stage": ("estagio", "stage", "fase"),
}

OUTCOME_COLS = ("resultado", "result", "outcome", "resulttype", "hit", "acerto",
                "statusresultado", "veredito")
FINAL_CORNERS_COLS = ("finalcorners", "totalcorners", "cantosfinal", "escanteiosfinal",
                      "escanteiostotal", "totalescanteios", "finaltotal", "cornersfinal")
PROB_COLS = ("analysisprob", "prob", "probabilidade", "pover", "confianca", "probemitida")
EV_COLS = ("ev", "evesperado", "expectedvalue")
ROUND_COLS = ("roundid", "matchid", "gameid", "jogoid", "idmatch", "idjogo")

OUTCOME_VALUES = {
    "win": ("win", "over", "green", "vitoria", "ganhou", "hit", "acerto", "positivo", "1"),
    "loss": ("loss", "under", "red", "derrota", "perdeu", "miss", "negativo", "0"),
    "push": ("push", "anulado", "neutro", "devolvido"),
    "void": ("void", "cancelado", "cancel"),
    "blocked": ("blocked", "bloqueado", "bloqueio", "rejeitado", "rejected"),
    "pending": ("pending", "pendente", "aberto", "emandamento", "open"),
}

IDENTITY = {"league", "home", "away", "sport"}
EVIDENCE = {"minute", "corners_home", "corners_away", "line", "market",
            "horizon_min", "pressure", "score_home", "score_away"}
IGNORED_TABLES = {"vec_cases", "vec_cases_meta", "rag_meta", "rag_migration_log"}


def _norm_col(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _build_alias_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for canon, aliases in CANONICAL_ALIASES.items():
        for a in aliases:
            m.setdefault(a, canon)
        m.setdefault(_norm_col(canon), canon)
    return m


ALIAS_TO_CANON = _build_alias_map()


def normalize_outcome(raw) -> str:
    if raw is None:
        return "pending"
    s = str(raw).strip().lower()
    for canon, aliases in OUTCOME_VALUES.items():
        if s in aliases:
            return canon
    return "pending"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class TablePlan:
    name: str
    features_map: Dict[str, str] = field(default_factory=dict)
    outcome_col: Optional[str] = None
    final_corners_col: Optional[str] = None
    prob_col: Optional[str] = None
    ev_col: Optional[str] = None
    round_col: Optional[str] = None
    row_count: int = 0
    has_rowid: bool = True

    @property
    def canon_set(self) -> set:
        return set(self.features_map.values())

    def is_candidate(self, min_features: int) -> bool:
        c = self.canon_set
        return (len(c) >= min_features and bool(c & IDENTITY) and bool(c & EVIDENCE))


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def plan_table(conn: sqlite3.Connection, table: str) -> TablePlan:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({_quote(table)})")]
    plan = TablePlan(name=table)
    for col in cols:
        n = _norm_col(col)
        if n in ALIAS_TO_CANON:
            plan.features_map[col] = ALIAS_TO_CANON[n]
        elif n in OUTCOME_COLS and plan.outcome_col is None:
            plan.outcome_col = col
        elif n in FINAL_CORNERS_COLS and plan.final_corners_col is None:
            plan.final_corners_col = col
        elif n in PROB_COLS and plan.prob_col is None:
            plan.prob_col = col
        elif n in EV_COLS and plan.ev_col is None:
            plan.ev_col = col
        elif n in ROUND_COLS and plan.round_col is None:
            plan.round_col = col
    plan.row_count = conn.execute(
        f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
    try:
        conn.execute(f"SELECT rowid FROM {_quote(table)} LIMIT 1").fetchone()
        plan.has_rowid = True
    except sqlite3.OperationalError:
        plan.has_rowid = False
    return plan


def list_user_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'").fetchall()
    return [r[0] for r in rows if r[0] not in IGNORED_TABLES]


@dataclass
class TableReport:
    table: str
    scanned: int = 0
    migrated: int = 0
    skipped_pk: int = 0
    skipped_dup: int = 0
    skipped_empty: int = 0
    failed: int = 0
    outcomes: Dict[str, int] = field(default_factory=dict)


class HistoryMigrator:
    def __init__(self, vm: VectorMemory, source_db: Path,
                 tables: Optional[List[str]] = None, limit: int = 5000,
                 batch: int = 100, min_features: int = 3,
                 skip_dup: bool = True, echo=print):
        self.vm = vm
        self.source_db = Path(source_db)
        self.table_filter = tables
        self.limit = int(limit)
        self.batch = int(batch)
        self.min_features = int(min_features)
        self.skip_dup = skip_dup
        self.echo = echo
        self._src_name = self.source_db.name

    def _connect_source(self) -> sqlite3.Connection:
        uri = f"file:{self.source_db.resolve().as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=5.0)

    def _connect_log(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.vm.db_path), timeout=10.0,
                               check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS rag_migration_log (
            source_db TEXT NOT NULL,
            source_table TEXT NOT NULL,
            source_pk TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            migrated_at TEXT NOT NULL,
            PRIMARY KEY (source_db, source_table, source_pk)
        );
        """)
        conn.commit()
        return conn

    def _already_logged(self, log: sqlite3.Connection, table: str, pk: str) -> bool:
        try:
            row = log.execute(
                "SELECT 1 FROM rag_migration_log "
                "WHERE source_db=? AND source_table=? AND source_pk=? LIMIT 1",
                (self._src_name, table, pk)).fetchone()
            return row is not None
        except sqlite3.OperationalError:
            return False

    def _content_exists(self, log: sqlite3.Connection, text: str) -> bool:
        if not self.skip_dup:
            return False
        try:
            row = log.execute(
                "SELECT 1 FROM vec_cases_meta WHERE embed_text=? LIMIT 1",
                (text,)).fetchone()
            return row is not None
        except sqlite3.OperationalError:
            return False

    def plan(self) -> List[TablePlan]:
        conn = self._connect_source()
        try:
            tables = list_user_tables(conn)
            if self.table_filter:
                tables = [t for t in tables if t in set(self.table_filter)]
                missing = set(self.table_filter) - set(tables)
                for m in missing:
                    self.echo(f"[migrate] AVISO: tabela pedida nao existe: {m}")
            plans = [plan_table(conn, t) for t in tables]
            candidates = [p for p in plans if p.is_candidate(self.min_features)]
            for p in plans:
                if p not in candidates and (not self.table_filter or p.name in set(self.table_filter)):
                    self.echo(f"[migrate] tabela ignorada (sem features suficientes): {p.name} "
                              f"[{len(p.canon_set)} canonicos: {sorted(p.canon_set)}]")
            if self.table_filter:
                forced = set(self.table_filter)
                candidates = [p for p in plans if p.name in forced] or candidates
            return candidates
        finally:
            conn.close()

    def migrate(self, dry_run: bool = False) -> List[TableReport]:
        reports: List[TableReport] = []
        plans = self.plan()
        if not plans:
            self.echo("[migrate] nenhuma tabela candidata. Rode --inspect e envie o "
                      "output para ajustar o mapeamento de colunas.")
            return reports
        src = self._connect_source()
        log = None if dry_run else self._connect_log()
        remaining = self.limit
        try:
            for plan in plans:
                if remaining <= 0:
                    break
                rep = self._migrate_table(src, log, plan, remaining, dry_run)
                reports.append(rep)
                remaining -= rep.migrated + rep.skipped_pk + rep.skipped_dup + rep.skipped_empty
        finally:
            src.close()
            if log is not None:
                log.close()
        return reports

    def _migrate_table(self, src, log, plan: TablePlan, limit: int,
                       dry_run: bool) -> TableReport:
        rep = TableReport(table=plan.name)
        self.echo(f"[migrate] {plan.name}: {plan.row_count} linhas, "
                  f"features={sorted(plan.canon_set)}")
        if plan.has_rowid:
            query = f"SELECT rowid AS __pk, * FROM {_quote(plan.name)} ORDER BY rowid"
        else:
            query = f"SELECT * FROM {_quote(plan.name)}"
        cur = src.execute(query)
        cols = [d[0] for d in cur.description]
        samples_shown = 0
        pending_commit = 0
        while True:
            rows = cur.fetchmany(self.batch)
            if not rows:
                break
            for row in rows:
                if rep.scanned >= limit:
                    break
                rec = dict(zip(cols, row))
                pk = str(rec.get("__pk", f"pos{rep.scanned}"))
                rep.scanned += 1

                features = {}
                for orig, canon in plan.features_map.items():
                    v = rec.get(orig)
                    if v is not None:
                        features[canon] = v
                san = sanitize_features(features)
                text = canonical_text(san)
                if not text:
                    rep.skipped_empty += 1
                    continue

                outcome = normalize_outcome(rec.get(plan.outcome_col)) \
                    if plan.outcome_col else "pending"

                if dry_run:
                    rep.outcomes[outcome] = rep.outcomes.get(outcome, 0) + 1
                    if samples_shown < 3:
                        self.echo(f"  [dry-run] pk={pk} outcome={outcome} :: {text}")
                        samples_shown += 1
                    continue

                if self._already_logged(log, plan.name, pk):
                    rep.skipped_pk += 1
                    continue
                if self._content_exists(log, text):
                    rep.skipped_dup += 1
                    try:
                        log.execute(
                            "INSERT OR IGNORE INTO rag_migration_log VALUES (?,?,?,?,?)",
                            (self._src_name, plan.name, pk,
                             hashlib.md5(text.encode("utf-8")).hexdigest(),
                             datetime.now().isoformat(timespec="seconds")))
                        pending_commit += 1
                    except sqlite3.Error:
                        pass
                    continue

                extra = {"migrated_from": f"{self._src_name}:{plan.name}#{pk}",
                         "source_pk": pk}
                if plan.prob_col and rec.get(plan.prob_col) is not None:
                    extra["analysis_prob"] = _num(rec.get(plan.prob_col))
                if plan.ev_col and rec.get(plan.ev_col) is not None:
                    extra["ev"] = _num(rec.get(plan.ev_col))
                if plan.round_col and rec.get(plan.round_col) is not None:
                    extra["round_id"] = str(rec.get(plan.round_col))

                case_id = self.vm.index_case(features, outcome=outcome, extra=extra)
                if case_id is None:
                    rep.failed += 1
                    continue

                fc = rec.get(plan.final_corners_col) if plan.final_corners_col else None
                if fc is not None and outcome in ("win", "loss", "push", "void"):
                    self.vm.update_outcome(case_id, outcome, final_corners=fc)

                rep.migrated += 1
                rep.outcomes[outcome] = rep.outcomes.get(outcome, 0) + 1
                try:
                    log.execute(
                        "INSERT OR IGNORE INTO rag_migration_log VALUES (?,?,?,?,?)",
                        (self._src_name, plan.name, pk,
                         hashlib.md5(text.encode("utf-8")).hexdigest(),
                         datetime.now().isoformat(timespec="seconds")))
                    pending_commit += 1
                except sqlite3.Error:
                    rep.failed += 1

                if pending_commit >= self.batch:
                    log.commit()
                    pending_commit = 0
                    self.echo(f"  ... {rep.scanned} processadas "
                              f"(migradas={rep.migrated})")
            if rep.scanned >= limit:
                break
        if log is not None and pending_commit:
            log.commit()
        self.echo(f"[migrate] {plan.name}: scanned={rep.scanned} "
                  f"migradas={rep.migrated} skip_pk={rep.skipped_pk} "
                  f"skip_dup={rep.skipped_dup} vazias={rep.skipped_empty} "
                  f"falhas={rep.failed} outcomes={rep.outcomes}")
        return rep


def _probe_provider(vm: VectorMemory) -> bool:
    try:
        r = vm.provider.embed("aura probe")
    except Exception:
        return False
    vec = r[0] if isinstance(r, tuple) else r
    return bool(vec)


def inspect(data_dir: Path, source: Optional[Path] = None) -> None:
    dbs = [source] if source else sorted(data_dir.glob("*.sqlite3"))
    print(f"[inspect] diretorio: {data_dir}")
    for db in dbs[:10]:
        print(f"\n=== {db.name} ({db.stat().st_size // 1024} KB) ===")
        try:
            uri = f"file:{db.resolve().as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        except sqlite3.Error as e:
            print(f"  erro: {e}")
            continue
        try:
            for t in list_user_tables(conn)[:30]:
                plan = plan_table(conn, t)
                cand = "CANDIDATA" if plan.is_candidate(3) else "ignorar"
                print(f"  {t:32s} rows={plan.row_count:<8} {cand}")
                if plan.features_map:
                    print(f"    features: {plan.features_map}")
                if plan.outcome_col:
                    print(f"    outcome col: {plan.outcome_col}")
        finally:
            conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Migracao do historico para o RAG")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", default=None)
    ap.add_argument("--table", action="append", default=None)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--min-features", type=int, default=3)
    ap.add_argument("--no-skip-dup", action="store_true")
    args = ap.parse_args()

    data_dir = ROOT / "data"
    source = Path(args.source) if args.source else data_dir / "aura_live_learning.sqlite3"

    if args.inspect:
        inspect(data_dir, Path(source) if args.source else None)
        return 0

    if not source.exists():
        print(f"[migrate] origem nao encontrada: {source}")
        return 2

    from agents.vector_memory import get_vector_memory
    vm = get_vector_memory()
    health = vm.health()
    if not health.get("enabled"):
        print("[migrate] RAG desligado em config/aura_rag.json (enabled=false). Aborted.")
        return 2
    if not health.get("vec_available"):
        print("[migrate] sqlite-vec indisponivel. Rode AURA_FASE1_INSTALL.bat. Aborted.")
        return 2
    if not _probe_provider(vm):
        print("[migrate] provider de embedding sem resposta (Ollama 11434 / "
              "LM Studio 1234). Suba um deles ou ajuste config/aura_rag.json. Aborted.")
        return 2
    if source.resolve() == Path(vm.db_path).resolve():
        print("[migrate] origem e igual ao banco vetorial de destino. Aborted.")
        return 2

    print(f"[migrate] origem: {source}")
    print(f"[migrate] destino: {vm.db_path} (dim={health.get('dim')}, "
          f"casos atuais={health.get('total_cases')})")

    mig = HistoryMigrator(vm, source, tables=args.table, limit=args.limit,
                          batch=args.batch, min_features=args.min_features,
                          skip_dup=not args.no_skip_dup)
    t0 = datetime.now()
    reports = mig.migrate(dry_run=args.dry_run)
    dt = (datetime.now() - t0).total_seconds()

    if args.dry_run:
        print(f"\n[migrate] DRY-RUN concluido em {dt:.1f}s — nada foi escrito. "
              f"Remova --dry-run para migrar de verdade.")
        return 0

    total_m = sum(r.migrated for r in reports)
    total_s = sum(r.skipped_pk + r.skipped_dup for r in reports)
    print(f"\n[migrate] concluido em {dt:.1f}s: migradas={total_m} "
          f"ignoradas(idempotencia)={total_s}")
    print(f"[migrate] estado final do RAG: {vm.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# ============================================================ tests

FILES["tests/test_vector_memory.py"] = r'''import hashlib
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.vector_memory import VectorMemory, canonical_text, sanitize_features

try:
    import sqlite_vec  # noqa: F401
    HAS_VEC = True
except Exception:
    HAS_VEC = False


class StaticTestProvider:
    def __init__(self, dim=16):
        self.dim = dim
        self.name = "test-static"

    def embed(self, text, timeout=None):
        v = [0.0] * self.dim
        for tok in text.replace("=", " ").replace("|", " ").split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


class BrokenProvider:
    def __init__(self):
        self.name = "broken"

    def embed(self, text, timeout=None):
        return None


def make_features(minute=68, ch=5, ca=3, league="serie-a", home="timea", away="timeb"):
    return {"league": league, "home": home, "away": away, "minute": minute,
            "corners_home": ch, "corners_away": ca, "line": 1.0, "horizon_min": 5}


class TestSanitize(unittest.TestCase):
    def test_sensitive_keys_removed(self):
        f = {"league": "x", "token": "SEGREDO", "api_key": "123", "minute": 10}
        san = sanitize_features(f)
        text = canonical_text(san)
        self.assertNotIn("SEGREDO", text)
        self.assertNotIn("123", text)
        self.assertIn("league=x", text)

    def test_canonical_stable_regardless_of_order(self):
        a = {"league": "x", "minute": 5, "line": 1.0}
        b = {"line": 1.0, "minute": 5, "league": "x"}
        self.assertEqual(canonical_text(sanitize_features(a)),
                         canonical_text(sanitize_features(b)))


@unittest.skipUnless(HAS_VEC, "sqlite-vec nao instalado")
class TestVectorMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aura_rag_test_"))
        self.db = self.tmp / "db.sqlite3"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mem(self, provider=None, **kw):
        return VectorMemory(db_path=self.db,
                            provider=provider or StaticTestProvider(), **kw)

    def test_index_and_search_finds_similar(self):
        mem = self._mem()
        fa = make_features(minute=68)
        fb = make_features(minute=70)
        fc = make_features(league="premier", home="timex", away="timey", minute=10)
        ida = mem.index_case(fa, extra={"analysis_prob": 0.55})
        idb = mem.index_case(fb)
        idc = mem.index_case(fc)
        self.assertIsNotNone(ida)
        res = mem.search(make_features(minute=69), k=2)
        ids = [c["case_id"] for c in res.cases]
        self.assertIn(idb, ids)
        self.assertNotIn(idc, ids)

    def test_stats_and_over_rate(self):
        mem = self._mem()
        for _ in range(4):
            cid = mem.index_case(make_features(minute=65))
            mem.update_outcome(cid, "win", final_corners=11)
        cid = mem.index_case(make_features(minute=66))
        mem.update_outcome(cid, "loss", final_corners=8)
        cid = mem.index_case(make_features(minute=67))
        mem.update_outcome(cid, "push", final_corners=10)
        res = mem.search(make_features(minute=66), k=10)
        self.assertEqual(res.wins, 4)
        self.assertEqual(res.losses, 1)
        self.assertEqual(res.pushes, 1)
        self.assertAlmostEqual(res.over_rate, 0.8, places=3)

    def test_update_outcome_flips_pending(self):
        mem = self._mem()
        cid = mem.index_case(make_features())
        res = mem.search(make_features(), k=5)
        self.assertGreaterEqual(res.pending, 1)
        self.assertTrue(mem.update_outcome(cid, "win", final_corners=12))
        res = mem.search(make_features(), k=5)
        self.assertGreaterEqual(res.wins, 1)

    def test_insufficient_evidence_on_empty_db(self):
        mem = self._mem()
        res = mem.search(make_features())
        self.assertEqual(res.status, "insufficient_evidence")
        self.assertEqual(res.n_indexed, 0)

    def test_provider_failure_is_fail_safe(self):
        mem = self._mem(provider=BrokenProvider())
        self.assertIsNone(mem.index_case(make_features()))
        res = mem.search(make_features())
        self.assertEqual(res.status, "error")

    def test_dim_migration_reindexes(self):
        mem8 = self._mem(provider=StaticTestProvider(dim=8))
        mem8.index_case(make_features(minute=60))
        mem8.index_case(make_features(minute=61))
        mem8.close()
        mem16 = self._mem(provider=StaticTestProvider(dim=16))
        mem16.index_case(make_features(minute=62))
        res = mem16.search(make_features(minute=61), k=5)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.n_indexed, 3)
        mem16.close()


class TestNoVecFailSafe(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aura_rag_novec_"))
        self.db = self.tmp / "db.sqlite3"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @unittest.skipIf(HAS_VEC, "sqlite-vec instalado; teste de degradacao simulado via flag")
    def test_search_without_vec_returns_degraded(self):
        mem = VectorMemory(self.db, StaticTestProvider())
        res = mem.search(make_features())
        self.assertIn(res.status, ("insufficient_evidence", "error"))
        mem.close()


if __name__ == "__main__":
    unittest.main()
'''

FILES["tests/test_rag_analog.py"] = r'''import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rag_analog import build_analog_context, format_prompt_block
from agents.vector_memory import SearchResult


class DisabledVM:
    enabled = False
    def search(self, *a, **k):
        raise AssertionError("nao deveria ser chamado")


class FakeVM:
    enabled = True
    def search(self, features, k=None):
        return SearchResult(status="ok", note="evidencia_analogica_disponivel",
                            n_indexed=100, k_found=9, n_used=9, wins=5, losses=3,
                            pushes=1, over_rate=0.625, avg_similarity=0.81,
                            avg_final_corners=11.2, avg_analysis_prob=0.58,
                            confidence=0.54, provider="test",
                            cases=[{"case_id": 1, "outcome": "win", "similarity": 0.9}])


class InsufficientVM:
    enabled = True
    def search(self, features, k=None):
        return SearchResult(status="insufficient_evidence",
                            note="n_resolved=2 < min_cases=5", n_indexed=10, n_used=2)


class TestRagAnalog(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(build_analog_context({"minute": 60}, vm=DisabledVM()))

    def test_context_schema(self):
        ctx = build_analog_context({"minute": 60}, vm=FakeVM())
        for key in ("status", "note", "analogous", "cases", "disclaimer"):
            self.assertIn(key, ctx)
        self.assertIn("nao e recomendacao", ctx["disclaimer"])

    def test_format_block_ok(self):
        ctx = build_analog_context({}, vm=FakeVM())
        block = format_prompt_block(ctx)
        self.assertIn("62.5%", block)
        self.assertIn("nao e recomendacao", block.lower())

    def test_format_block_insufficient(self):
        ctx = build_analog_context({}, vm=InsufficientVM())
        block = format_prompt_block(ctx)
        self.assertIn("insuficiente", block.lower())

    def test_format_block_none(self):
        self.assertIn("desativado", format_prompt_block(None).lower())


if __name__ == "__main__":
    unittest.main()
'''

FILES["tests/test_telegram_alerts.py"] = r'''import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge.telegram_alerts import TelegramAlerter


class FakeSender:
    def __init__(self):
        self.sent = []

    def __call__(self, text):
        self.sent.append(text)
        return True


def _cfg(tmp, **kw):
    base = {"enabled": True, "bot_token": "x", "chat_id": "1",
            "min_level": "info", "dedupe_minutes": 10, "rate_limit_per_hour": 1000}
    base.update(kw)
    p = Path(tmp) / "tg.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    return str(p)


def _wait(fake, n=1, timeout=3.0):
    t0 = time.time()
    while len(fake.sent) < n and time.time() - t0 < timeout:
        time.sleep(0.02)
    return len(fake.sent) >= n


class TestTelegramAlerts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aura_tg_test_")

    def test_alert_delivered(self):
        fake = FakeSender()
        a = TelegramAlerter(cfg_path=_cfg(self.tmp), sender=fake)
        self.assertTrue(a.alert("warn", "teste_titulo", "corpo"))
        self.assertTrue(_wait(fake, 1))
        self.assertIn("[AURA][WARN] teste_titulo", fake.sent[0])

    def test_dedupe_same_title(self):
        fake = FakeSender()
        a = TelegramAlerter(cfg_path=_cfg(self.tmp, dedupe_minutes=10), sender=fake)
        self.assertTrue(a.alert("warn", "mesmo_titulo"))
        self.assertFalse(a.alert("warn", "mesmo_titulo"))
        self.assertTrue(_wait(fake, 1))
        self.assertEqual(len(fake.sent), 1)

    def test_min_level_filters_info(self):
        fake = FakeSender()
        a = TelegramAlerter(cfg_path=_cfg(self.tmp, min_level="warn"), sender=fake)
        self.assertFalse(a.alert("info", "x"))
        self.assertTrue(a.alert("warn", "y"))
        self.assertTrue(_wait(fake, 1))

    def test_disabled(self):
        p = Path(self.tmp) / "off.json"
        p.write_text(json.dumps({"enabled": False}), encoding="utf-8")
        a = TelegramAlerter(cfg_path=str(p), sender=FakeSender())
        self.assertFalse(a.alert("error", "x"))

    def test_rate_limit(self):
        fake = FakeSender()
        a = TelegramAlerter(cfg_path=_cfg(self.tmp, rate_limit_per_hour=2,
                                          dedupe_minutes=0), sender=fake)
        self.assertTrue(a.alert("warn", "a1"))
        self.assertTrue(a.alert("warn", "a2"))
        self.assertFalse(a.alert("warn", "a3"))
        self.assertTrue(_wait(fake, 2))

    def test_body_truncated(self):
        fake = FakeSender()
        a = TelegramAlerter(cfg_path=_cfg(self.tmp), sender=fake)
        a.alert("warn", "grande", "x" * 5000)
        self.assertTrue(_wait(fake, 1))
        self.assertLessEqual(len(fake.sent[0]), 3600)


if __name__ == "__main__":
    unittest.main()
'''

FILES["tests/test_telegram_watchdog.py"] = r'''import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aura_telegram_watchdog import Watchdog

SVCS = [{"name": "svc_a", "host": "127.0.0.1", "port": 1, "path": None, "expect": True},
        {"name": "svc_b", "host": "127.0.0.1", "port": 2, "path": None, "expect": True}]


class FakeAlerter:
    def __init__(self):
        self.calls = []

    def alert(self, level, title, body):
        self.calls.append((level, title))


def _make_prober(state):
    def prober(svc):
        return state.get(svc["name"], False)
    return prober


class TestWatchdog(unittest.TestCase):
    def test_transition_down_and_recovery(self):
        al = FakeAlerter()
        state = {"svc_a": True, "svc_b": True}
        wd = Watchdog(SVCS, alerter=al, prober=_make_prober(state), interval_s=1)
        st = wd.run_once()
        self.assertEqual(st, {"svc_a": "up", "svc_b": "up"})
        self.assertEqual(al.calls, [])

        state["svc_a"] = False
        wd.run_once()
        downs = [c for c in al.calls if c[1] == "servico_down"]
        self.assertEqual(len(downs), 1)

        state["svc_a"] = True
        wd.run_once()
        ups = [c for c in al.calls if c[1] == "servico_up"]
        self.assertEqual(len(ups), 1)

    def test_initial_down_alerts_when_expected(self):
        al = FakeAlerter()
        wd = Watchdog(SVCS, alerter=al, prober=_make_prober({}), interval_s=1)
        st = wd.run_once()
        self.assertEqual(st["svc_a"], "down")
        downs = [c for c in al.calls if c[1] == "servico_down"]
        self.assertEqual(len(downs), 2)

    def test_unexpected_service_silent_on_initial_down(self):
        al = FakeAlerter()
        svcs = [{"name": "hermes", "host": "127.0.0.1", "port": 8777,
                 "path": None, "expect": False}]
        wd = Watchdog(svcs, alerter=al, prober=_make_prober({}), interval_s=1)
        wd.run_once()
        self.assertEqual(al.calls, [])


if __name__ == "__main__":
    unittest.main()
'''

FILES["tests/test_rag_migrate.py"] = r'''import hashlib
import math
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.vector_memory import VectorMemory
from scripts.aura_rag_migrate_history import HistoryMigrator

try:
    import sqlite_vec  # noqa: F401
    HAS_VEC = True
except Exception:
    HAS_VEC = False


class StaticTestProvider:
    def __init__(self, dim=16):
        self.dim = dim
        self.name = "test-static"

    def embed(self, text, timeout=None):
        v = [0.0] * self.dim
        for tok in text.replace("=", " ").replace("|", " ").split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


FIXTURE_SQL = """
CREATE TABLE learning (
    id INTEGER PRIMARY KEY,
    liga TEXT, mandante TEXT, visitante TEXT,
    minuto INTEGER, escanteios_casa INTEGER, escanteios_fora INTEGER,
    linha REAL, horizonte INTEGER, resultado TEXT, cantos_final INTEGER, prob REAL
);
INSERT INTO learning VALUES (1, 'serie-a', 'aa', 'bb', 70, 5, 3, 1.0, 5, 'win', 11, 0.55);
INSERT INTO learning VALUES (2, 'serie-a', 'aa', 'bb', 75, 6, 3, 1.0, 5, 'loss', 8, 0.52);
INSERT INTO learning VALUES (3, 'serie-a', 'cc', 'dd', 80, 4, 4, 1.5, 10, 'push', 10, 0.50);
CREATE TABLE irrelevant (id INTEGER PRIMARY KEY, note TEXT);
INSERT INTO irrelevant VALUES (1, 'sem features de jogo');
"""


@unittest.skipUnless(HAS_VEC, "sqlite-vec nao instalado")
class TestHistoryMigrator(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aura_mig_test_"))
        self.src = self.tmp / "aura_live_learning.sqlite3"
        conn = sqlite3.connect(self.src)
        conn.executescript(FIXTURE_SQL)
        conn.commit()
        conn.close()
        self.vm = VectorMemory(db_path=self.tmp / "vec.sqlite3",
                               provider=StaticTestProvider())

    def tearDown(self):
        self.vm.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_finds_candidate_and_ignores_irrelevant(self):
        mig = HistoryMigrator(self.vm, self.src)
        plans = mig.plan()
        names = [p.name for p in plans]
        self.assertIn("learning", names)
        self.assertNotIn("irrelevant", names)
        plan = next(p for p in plans if p.name == "learning")
        self.assertEqual(plan.features_map["liga"], "league")
        self.assertEqual(plan.features_map["mandante"], "home")
        self.assertEqual(plan.features_map["minuto"], "minute")
        self.assertEqual(plan.outcome_col, "resultado")

    def test_migrate_applies_outcomes(self):
        mig = HistoryMigrator(self.vm, self.src)
        reports = mig.migrate()
        self.assertEqual(reports[0].migrated, 3)
        self.assertEqual(self.vm.stats()["total_cases"], 3)
        res = self.vm.search({"league": "serie-a", "home": "aa", "away": "bb",
                              "minute": 72, "line": 1.0, "horizon_min": 5}, k=5)
        self.assertEqual(res.wins, 1)
        self.assertEqual(res.losses, 1)
        self.assertEqual(res.pushes, 1)

    def test_rerun_is_idempotent(self):
        mig = HistoryMigrator(self.vm, self.src)
        mig.migrate()
        reports2 = HistoryMigrator(self.vm, self.src).migrate()
        self.assertEqual(reports2[0].migrated, 0)
        self.assertEqual(reports2[0].skipped_pk, 3)
        self.assertEqual(self.vm.stats()["total_cases"], 3)

    def test_dry_run_writes_nothing(self):
        mig = HistoryMigrator(self.vm, self.src)
        reports = mig.migrate(dry_run=True)
        self.assertGreaterEqual(reports[0].scanned, 3)
        self.assertEqual(self.vm.stats()["total_cases"], 0)

    def test_limit_respected(self):
        mig = HistoryMigrator(self.vm, self.src, limit=2)
        reports = mig.migrate()
        self.assertEqual(reports[0].migrated, 2)


if __name__ == "__main__":
    unittest.main()
'''

# ============================================================ skill

FILES["skills/aura-rag-analog/SKILL.md"] = r'''# Skill: aura-rag-analog

## Objetivo
Operar e diagnosticar a memoria vetorial de casos analogos (RAG local) do Aura.

## Quando usar
- Antes de rodas longas de analise: checar /api/rag/health
- Quando o contexto analogico vier vazio ou "insufficient_evidence"
- Para responder "em casos parecidos, qual foi a taxa historica?"

## Comandos
- Health:  curl http://127.0.0.1:8765/api/rag/health
- Stats:   curl http://127.0.0.1:8765/api/rag/stats
- Busca manual: POST /api/rag/search {"league":"...","minute":70,"line":1.0}
- Migrar historico: AURA_RAG_MIGRATE.bat (dry-run antes, sempre)
- Watchdog: AURA_TELEGRAM_WATCHDOG.bat

## Como ler o resultado
- status=ok: analogous.over_rate = taxa over historica (excl. push);
  confidence sobe com n>=10 resolvidos e similaridade media alta.
- status=insufficient_evidence: NAO e erro. Base pequena ou poucos analogos.
  Prosseguir sem contexto analogico.
- status=error: provider caiu. Analise continua sem RAG (fail-safe por design).

## Regras de seguranca
1. RAG e evidencia advisory: nunca publicar recomendacao baseada apenas nele.
2. Campos sensiveis (token/cookie/senha/api_key) sao bloqueados antes do embed.
3. Nao editar o banco vetorial manualmente; usar migrador ou indexacao automatica.
4. Migracao e idempotente: reexecutar nao duplica (log por PK + hash de conteudo).

## Diagnostico rapido
1. /api/rag/health: enabled=true e vec_available=true?
2. total_cases=0? Rodar AURA_RAG_MIGRATE.bat.
3. provider null? Ollama (11434) ou LM Studio (1234) ligado?
4. Sempre insufficient_evidence? Normal com base pequena (min_cases=5).
'''

# ============================================================ configs

FILES["config/aura_rag.json"] = r'''{
  "enabled": true,
  "provider_order": ["ollama", "lmstudio"],
  "ollama_url": "http://127.0.0.1:11434",
  "ollama_model": "nomic-embed-text",
  "lmstudio_url": "http://127.0.0.1:1234/v1",
  "lmstudio_model": "text-embedding-nomic-embed-text-v1.5",
  "db_path": "data/aura_vector_memory.sqlite3",
  "search_k": 12,
  "oversample": 3,
  "min_cases_for_stats": 5,
  "max_age_days": 180,
  "timeout_s": 2.0,
  "cache_size": 256,
  "migrate_batch_limit": 2000
}
'''

FILES["config/aura_telegram.json"] = r'''{
  "enabled": false,
  "bot_token": "",
  "chat_id": "",
  "min_level": "warn",
  "rate_limit_per_hour": 20,
  "dedupe_minutes": 10
}
'''

FILES["config/aura_watchdog.json"] = r'''{
  "interval_s": 60,
  "remind_after_min": 30,
  "services": [
    {"name": "hermes", "expect": false},
    {"name": "jarvis", "expect": false},
    {"name": "ollama", "expect": false}
  ]
}
'''

# ============================================================ bats

FILES["AURA_FASE1_INSTALL.bat"] = r'''@echo off
cd /d C:\aura
echo [FASE1] Instalando sqlite-vec...
python -m pip install --upgrade sqlite-vec
where ollama >nul 2>nul
if %errorlevel%==0 (
    echo [FASE1] Baixando modelo de embedding local...
    ollama pull nomic-embed-text
) else (
    echo [FASE1] AVISO: Ollama nao encontrado no PATH. Baixe nomic-embed-text manualmente
    echo        ou configure provider_order=["lmstudio"] em config\aura_rag.json
)
echo [FASE1] Concluido.
pause
'''

FILES["AURA_FASE1_TEST.bat"] = r'''@echo off
cd /d C:\aura
echo [FASE1] Compilacao dos modulos novos...
python -m compileall -q agents\vector_memory.py agents\rag_analog.py engine\rag_routes.py engine\sse_state.py bridge\telegram_alerts.py scripts\aura_telegram_watchdog.py scripts\aura_rag_migrate_history.py
if errorlevel 1 (echo [FASE1] FALHA na compilacao & exit /b 1)
echo [FASE1] Suite de testes (offline, nao precisa de Ollama)...
python -m unittest tests.test_vector_memory tests.test_rag_analog tests.test_telegram_alerts tests.test_telegram_watchdog tests.test_rag_migrate -v
if errorlevel 1 (echo [FASE1] FALHA nos testes & exit /b 1)
echo [FASE1] APROVADO.
pause
'''

FILES["AURA_RAG_MIGRATE.bat"] = r'''@echo off
cd /d C:\aura
echo [FASE1] Migracao do historico para o RAG — DRY-RUN primeiro:
python scripts\aura_rag_migrate_history.py --dry-run
echo.
set /p CONF=Executar migracao REAL agora? (S/N): 
if /i not "%CONF%"=="S" (echo [FASE1] Abortado pelo operador. & exit /b 0)
python scripts\aura_rag_migrate_history.py
echo.
python -c "import sys; sys.path.insert(0,'.'); from agents.vector_memory import get_vector_memory; print(get_vector_memory().stats())"
pause
'''

FILES["AURA_TELEGRAM_WATCHDOG.bat"] = r'''@echo off
cd /d C:\aura
python scripts\aura_telegram_watchdog.py
pause
'''

FILES["AURA_FASE1_CHECKFILES.bat"] = r'''@echo off
cd /d C:\aura
echo === Verificacao de arquivos da FASE 1 ===
set FALTANDO=0
for %%f in (
 "agents\vector_memory.py"
 "agents\rag_analog.py"
 "engine\rag_routes.py"
 "engine\sse_state.py"
 "desktop\ui\matriz_v22\assets\aura-stream.js"
 "bridge\telegram_alerts.py"
 "scripts\aura_telegram_watchdog.py"
 "scripts\aura_rag_migrate_history.py"
 "tests\test_vector_memory.py"
 "tests\test_rag_analog.py"
 "tests\test_telegram_alerts.py"
 "tests\test_telegram_watchdog.py"
 "tests\test_rag_migrate.py"
 "skills\aura-rag-analog\SKILL.md"
 "config\aura_rag.json"
 "config\aura_telegram.json"
 "config\aura_watchdog.json"
 "AURA_FASE1_INSTALL.bat"
 "AURA_FASE1_TEST.bat"
 "AURA_RAG_MIGRATE.bat"
 "AURA_TELEGRAM_WATCHDOG.bat"
) do (
 if exist %%f (echo   OK    %%f) else (echo   FALTA %%f & set FALTANDO=1)
)
if "%FALTANDO%"=="1" (echo. & echo *** Ha arquivos faltando. *** ) else (echo. & echo *** Todos os arquivos da Fase 1 presentes. ***)
pause
'''

# ============================================================ main


def main() -> int:
    force = "--force" in sys.argv
    created, skipped = 0, 0
    for rel, content in FILES.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            print(f"PULA  {rel} (ja existe; use --force para sobrescrever)")
            skipped += 1
            continue
        target.write_text(content, encoding="utf-8")
        print(f"OK    {rel}")
        created += 1
    print(f"\nConcluido: {created} criados, {skipped} preservados.")
    print(f"Pasta base: {ROOT}")
    print("Proximo passo: copie esta estrutura para C:\\aura (robocopy) e rode")
    print("AURA_FASE1_CHECKFILES.bat e AURA_FASE1_TEST.bat la dentro.")
    return 0


if __name__ == "__main__":
    sys.exit(main())