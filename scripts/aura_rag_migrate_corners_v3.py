"""Aura Fase 1 — Migrador v3 (real): corner_observations_v2 (JSON) -> RAG vetorial.
--dry-run: so planeja. --real: migra de verdade (idempotente, dedupe, sem locks).
"""
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.vector_memory import get_vector_memory, canonical_text, sanitize_features

SRC = ROOT / "data" / "aura_live_learning.sqlite3"
SRC_NAME = "aura_live_learning.sqlite3"
TABLE = "corner_observations_v2"
FLUSH_EVERY = 50


def g(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def build_case(payload, analysis):
    features = {}
    for canon, path in [
        ("home", ("home",)), ("away", ("away",)), ("minute", ("minute",)),
        ("score_home", ("score_home",)), ("score_away", ("score_away",)),
        ("pressure", ("pressure_gauge",)), ("stage", ("status",)),
    ]:
        v = g(payload, *path)
        if v is not None:
            features[canon] = v
    ch = g(payload, "metrics", "corners", "home")
    ca = g(payload, "metrics", "corners", "away")
    if ch is not None:
        features["corners_home"] = ch
    if ca is not None:
        features["corners_away"] = ca
    mkt = g(analysis, "market")
    if mkt:
        features["market"] = mkt
    features["source"] = "sokkerpro"
    outcome = "pending" if g(analysis, "eligible") else "blocked"
    prob = first_not_none(g(analysis, "p_5m"), g(analysis, "prob"),
                          g(analysis, "p"), g(analysis, "probability"))
    return features, outcome, prob


def main(mode: str) -> int:
    real = (mode == "--real")
    vm = get_vector_memory()
    h = vm.health()
    if not h.get("enabled") or not h.get("vec_available"):
        print("ABORT: RAG indisponivel (rodar AURA_FASE1_INSTALL.bat).")
        return 2
    vec, pname = vm.provider.embed("probe")
    if not vec:
        print("ABORT: LM Studio sem resposta (ligue o servidor).")
        return 2
    print(f"provider: {pname} | dim: {h.get('dim')} | casos atuais: {h.get('total_cases')}")

    conn = sqlite3.connect(f"file:{SRC.as_posix()}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT capture_key, fixture_id, payload, analysis FROM corner_observations_v2"
    ).fetchall()
    conn.close()
    print(f"origem: {TABLE} ({len(rows)} rows) — modo {'REAL' if real else 'DRY-RUN'}")

    log = None
    pending_log = []
    if real:
        log = sqlite3.connect(str(vm.db_path), timeout=10.0, check_same_thread=False)
        log.execute("PRAGMA journal_mode=WAL")
        log.executescript("""
        CREATE TABLE IF NOT EXISTS rag_migration_log (
            source_db TEXT NOT NULL, source_table TEXT NOT NULL,
            source_pk TEXT NOT NULL, content_hash TEXT NOT NULL,
            migrated_at TEXT NOT NULL,
            PRIMARY KEY (source_db, source_table, source_pk));
        """)
        log.commit()

    def flush():
        if log is not None and pending_log:
            log.executemany(
                "INSERT OR IGNORE INTO rag_migration_log VALUES (?,?,?,?,?)",
                list(pending_log))
            log.commit()
            pending_log.clear()

    stats = {"scanned": 0, "migradas": 0, "skip_pk": 0, "skip_dup": 0,
             "no_identity": 0, "falhas": 0, "pending": 0, "blocked": 0}
    seen_texts = set()
    samples = []
    for rk, fixture, payload_s, analysis_s in rows:
        stats["scanned"] += 1
        try:
            payload = json.loads(payload_s)
            analysis = json.loads(analysis_s) if analysis_s else {}
        except Exception:
            stats["falhas"] += 1
            continue
        features, outcome, prob = build_case(payload, analysis)
        if not (features.get("home") and features.get("away")):
            stats["no_identity"] += 1
            continue
        san = sanitize_features(features)
        text = canonical_text(san)
        if not text:
            stats["no_identity"] += 1
            continue

        if not real:
            seen_texts.add(text)
            stats["pending" if outcome == "pending" else "blocked"] += 1
            if len(samples) < 3:
                samples.append((fixture, rk, text, outcome, prob))
            continue

        pk = str(rk)
        try:
            if log.execute(
                "SELECT 1 FROM rag_migration_log WHERE source_db=? AND source_table=? "
                "AND source_pk=? LIMIT 1",
                (SRC_NAME, TABLE, pk)).fetchone():
                stats["skip_pk"] += 1
                continue
            if log.execute(
                "SELECT 1 FROM vec_cases_meta WHERE embed_text=? LIMIT 1",
                (text,)).fetchone():
                stats["skip_dup"] += 1
                pending_log.append((SRC_NAME, TABLE, pk,
                                    hashlib.md5(text.encode()).hexdigest(),
                                    datetime.now().isoformat(timespec="seconds")))
                continue
        except sqlite3.OperationalError:
            pass

        extra = {"round_id": str(fixture or ""), "analysis_prob": prob,
                 "capture_key": pk, "window": g(analysis, "window", "primary"),
                 "call": g(analysis, "call"),
                 "reason": str(g(analysis, "reason") or "")[:120]}
        case_id = vm.index_case(features, outcome=outcome, extra=extra)
        if case_id is None:
            stats["falhas"] += 1
            continue
        stats["migradas"] += 1
        stats["pending" if outcome == "pending" else "blocked"] += 1
        pending_log.append((SRC_NAME, TABLE, pk,
                            hashlib.md5(text.encode()).hexdigest(),
                            datetime.now().isoformat(timespec="seconds")))
        if stats["migradas"] % FLUSH_EVERY == 0:
            flush()
            print(f"  ... {stats['scanned']} escaneadas (migradas={stats['migradas']})")
    flush()

    print(f"\nRELATORIO: {stats}")
    if not real:
        print(f"casos UNICOS (pos-dedupe): {len(seen_texts)}")
        for fixture, rk, text, outcome, prob in samples:
            print(f"  fixture={fixture} outcome={outcome} prob={prob}")
            print(f"    {text}")
        print("\n(dry-run: nada escrito)")
        return 0

    print(f"\nRAG final: {vm.stats()}")
    print("\n=== POS-TESTE: busca de casos analogos reais ===")
    res = vm.search({"home": "Chelsea", "away": "Leeds United", "minute": 82,
                     "score_home": 4, "score_away": 2, "source": "sokkerpro"}, k=5)
    print(f"status={res.status} | n_indexed={res.n_indexed} | recuperados={len(res.cases)}")
    for c in res.cases[:5]:
        print(f"  sim={c['similarity']} | {c['home']} x {c['away']} | min={c['minute']} "
              f"| placar {c['corners'][0]}-{c['corners'][1]} corners | outcome={c['outcome']}")
    if log is not None:
        log.close()
    vm.close()
    print("\nMIGRACAO v3 CONCLUIDA.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "--dry-run"))
