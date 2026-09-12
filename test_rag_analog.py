import sys
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
