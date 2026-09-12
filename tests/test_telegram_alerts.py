import json
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
