from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SERVICES = [
    {"name": "alfred",  "host": "127.0.0.1", "port": 8791, "path": "/health",     "expect": True},
    {"name": "engine",  "host": "127.0.0.1", "port": 8765, "path": "/api/health", "expect": True},
    {"name": "bridge",  "host": "127.0.0.1", "port": 8080, "path": None,          "expect": True},
    {"name": "jarvis",  "host": "127.0.0.1", "port": 8099, "path": None,          "expect": False},
    {"name": "ollama",  "host": "127.0.0.1", "port": 11434, "path": None,         "expect": False},
    {"name": "hermes",  "host": "127.0.0.1", "port": 8777, "path": "/health",     "expect": False},
]


def _load_config() -> dict:
    p = ROOT / "config" / "aura_watchdog.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _merge_services(cfg: dict) -> List[dict]:
    overrides = {s.get("name"): s for s in cfg.get("services", []) if isinstance(s, dict)}
    out = []
    for svc in DEFAULT_SERVICES:
        merged = dict(svc)
        ov = overrides.get(svc["name"])
        if ov:
            merged.update({k: v for k, v in ov.items() if k != "name"})
        out.append(merged)
    return out


def _probe_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _probe_http(host: str, port: int, path: str, timeout: float = 2.0) -> bool:
    try:
        url = f"http://{host}:{port}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def default_prober(svc: dict) -> bool:
    if not _probe_tcp(svc["host"], svc["port"]):
        return False
    if svc.get("path"):
        return _probe_http(svc["host"], svc["port"], svc["path"])
    return True


class Watchdog:
    def __init__(self, services: List[dict], alerter=None,
                 prober: Optional[object] = None,
                 interval_s: int = 60, remind_after_min: int = 30):
        self.services = services
        self.alerter = alerter
        self.prober = prober or default_prober
        self.interval_s = int(interval_s)
        self.remind_after_s = int(remind_after_min) * 60
        self._prev: Dict[str, bool] = {}
        self._down_since: Dict[str, float] = {}
        self._reminded = set()

    def _notify(self, level: str, title: str, body: str) -> None:
        if self.alerter is None:
            print(f"[watchdog][{level}] {title}: {body}")
            return
        try:
            self.alerter.alert(level, title, body)
        except Exception:
            pass

    def run_once(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        now = time.time()
        for svc in self.services:
            name = svc["name"]
            up = False
            try:
                up = bool(self.prober(svc))
            except Exception:
                up = False
            result[name] = "up" if up else "down"
            was = self._prev.get(name)
            if up:
                if was is False:
                    self._notify("info", "servico_up",
                                 f"{name} ({svc['host']}:{svc['port']}) respondeu novamente")
                self._down_since.pop(name, None)
                self._reminded.discard(name)
            else:
                if was is None:
                    if svc.get("expect", False):
                        self._notify("error", "servico_down",
                                     f"{name} ({svc['host']}:{svc['port']}) sem resposta")
                        self._down_since[name] = now
                    else:
                        self._down_since[name] = now
                elif was is True:
                    self._notify("error", "servico_down",
                                 f"{name} ({svc['host']}:{svc['port']}) caiu")
                    self._down_since[name] = now
                    self._reminded.discard(name)
                else:
                    since = self._down_since.get(name)
                    if (since is not None and self.remind_after_s > 0
                            and now - since >= self.remind_after_s
                            and name not in self._reminded
                            and svc.get("expect", False)):
                        self._notify("warn", "servico_persistindo_down",
                                     f"{name} down ha mais de {self.remind_after_s//60} min")
                        self._reminded.add(name)
            self._prev[name] = up
        return result

    def run_forever(self) -> None:
        print(f"[watchdog] iniciado (intervalo {self.interval_s}s, "
              f"{len(self.services)} servicos, Ctrl+C para sair)")
        while True:
            try:
                st = self.run_once()
                print("[watchdog] " + " ".join(f"{k}={v}" for k, v in st.items()))
            except KeyboardInterrupt:
                print("[watchdog] encerrado")
                return
            except Exception as e:
                print(f"[watchdog] erro no ciclo: {e}")
            time.sleep(self.interval_s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aura watchdog de servicos")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = _load_config()
    if args.config:
        try:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except Exception:
            print(f"[watchdog] config invalida: {args.config}")
            return 2

    services = _merge_services(cfg)
    try:
        from bridge.telegram_alerts import get_alerter
        alerter = get_alerter()
    except Exception:
        alerter = None

    wd = Watchdog(services, alerter=alerter,
                  interval_s=args.interval or cfg.get("interval_s", 60),
                  remind_after_min=cfg.get("remind_after_min", 30))

    if args.once:
        st = wd.run_once()
        for name, status in st.items():
            print(f"{name:8s} {status}")
        return 0 if all(v == "up" for v in st.values()) else 1

    wd.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
