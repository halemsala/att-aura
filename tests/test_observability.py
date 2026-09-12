import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine import aura_observability as obs


class TestObservability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aura_obs_")
        self._orig = obs.TRACE_FILE
        obs.TRACE_FILE = Path(self.tmp) / "traces.jsonl"
        self.app = FastAPI()
        obs.install(self.app)

        @self.app.get("/ping")
        def ping():
            return {"trace": obs.get_trace_id()}

        self.client = TestClient(self.app)

    def tearDown(self):
        obs.TRACE_FILE = self._orig

    def test_trace_id_gerado_e_contexto(self):
        r = self.client.get("/ping")
        tid = r.headers.get("x-trace-id")
        self.assertTrue(tid and len(tid) == 12)
        self.assertEqual(r.json()["trace"], tid)

    def test_trace_id_propagado_do_header(self):
        r = self.client.get("/ping", headers={"X-Trace-Id": "abc123def456"})
        self.assertEqual(r.headers.get("x-trace-id"), "abc123def456")
        self.assertEqual(r.json()["trace"], "abc123def456")

    def test_metrics_formato_prometheus(self):
        self.client.get("/ping")
        txt = self.client.get("/api/obs/metrics").text
        self.assertIn("aura_http_requests_total", txt)
        self.assertIn("path=/ping", txt.replace(chr(34), ""))
        self.assertIn("aura_http_request_duration_seconds_count", txt)
        self.assertIn("aura_uptime_seconds", txt)
        for line in txt.splitlines():
            if line and not line.startswith("#"):
                self.assertEqual(len(line.rsplit(" ", 1)), 2, line)

    def test_log_jsonl_valido(self):
        self.client.get("/ping")
        self.assertTrue(obs.TRACE_FILE.exists())
        import json as js
        last = obs.TRACE_FILE.read_text(encoding="utf-8").strip().splitlines()[-1]
        rec = js.loads(last)
        for k in ("ts", "trace_id", "method", "path", "status", "dur_ms"):
            self.assertIn(k, rec)


if __name__ == "__main__":
    unittest.main()
