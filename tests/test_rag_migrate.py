import hashlib
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
