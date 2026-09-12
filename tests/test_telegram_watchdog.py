import sys
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
