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
