import hashlib
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rag_backtest import backtest_report, _brier, _log_loss
from agents.vector_memory import VectorMemory

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


@unittest.skipUnless(HAS_VEC, "sqlite-vec nao instalado")
class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="aura_bt_"))
        self.vm = VectorMemory(db_path=self.tmp / "bt.sqlite3",
                               provider=StaticTestProvider())

    def tearDown(self):
        self.vm.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add(self, i, prob, outcome):
        self.vm.index_case(
            {"home": f"time{i}", "away": f"rival{i}", "minute": 70 + i,
             "score_home": 1, "score_away": 1},
            outcome=outcome, extra={"analysis_prob": prob})

    def test_brier_matematica_exata(self):
        self.assertAlmostEqual(_brier([("", 1.0, 1), ("", 0.0, 0)]), 0.0)
        self.assertAlmostEqual(_brier([("", 0.8, 1), ("", 0.6, 0)]), 0.20)
        self.assertAlmostEqual(_log_loss([("", 1.0, 1), ("", 0.0, 0)]), 0.0, places=5)

    def test_insufficient_data(self):
        self._add(1, 0.8, "win")
        rep = backtest_report(vm=self.vm, min_resolved=5)
        self.assertEqual(rep["status"], "insufficient_data")
        self.assertEqual(rep["coverage"]["resolved_com_prob"], 1)

    def test_push_e_pending_excluidos(self):
        for i in range(6):
            self._add(i, 0.7, "win")
        self._add(10, 0.7, "push")     # excluido
        self._add(11, 0.9, "pending")  # excluido
        rep = backtest_report(vm=self.vm, min_resolved=5)
        self.assertEqual(rep["status"], "ok")
        self.assertEqual(rep["n"], 6)
        self.assertAlmostEqual(rep["brier"], 0.09, places=3)

    def test_reliability_bins(self):
        self._add(1, 0.85, "win")   # bin 0.8-1.0, rate 1.0
        self._add(2, 0.9, "loss")   # bin 0.8-1.0, rate 0.5
        self._add(3, 0.55, "win")   # bin 0.4-0.6
        self._add(4, 0.45, "win")
        self._add(5, 0.35, "loss")
        self._add(6, 0.65, "win")
        rep = backtest_report(vm=self.vm, min_resolved=5)
        self.assertEqual(rep["status"], "ok")
        b81 = next(b for b in rep["reliability"] if b["bin"] == "0.8-1.0")
        self.assertEqual(b81["n"], 2)
        self.assertEqual(b81["actual_rate"], 0.5)

    def test_walk_forward(self):
        for i in range(30):
            self._add(i, 0.6 if i % 2 else 0.4, "win" if i % 2 else "loss")
        rep = backtest_report(vm=self.vm, min_resolved=5)
        self.assertEqual(rep["walk_forward"]["folds"], 3)
        self.assertEqual(sum(c["n"] for c in rep["walk_forward"]["chunks"]), 30)


if __name__ == "__main__":
    unittest.main()
