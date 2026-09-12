from __future__ import annotations

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
