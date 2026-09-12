from __future__ import annotations

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
