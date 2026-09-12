from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("aura.telegram")
LEVELS = {"info": 0, "warn": 1, "error": 2, "critical": 3}


def _http_send(token: str, chat_id: str, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status < 400


class TelegramAlerter:
    def __init__(self, cfg_path=None, sender: Optional[Callable[[str], bool]] = None):
        self.cfg_path = Path(cfg_path) if cfg_path else ROOT / "config" / "aura_telegram.json"
        self.cfg = self._load()
        self._sender = sender
        self._q = queue.Queue(maxsize=200)
        self._recent = {}
        self._window = deque()
        self._dropped = 0
        self._lock = threading.Lock()
        self._thread = None
        if self.cfg.get("enabled"):
            self._thread = threading.Thread(
                target=self._worker, name="aura-telegram", daemon=True)
            self._thread.start()

    def _load(self) -> dict:
        try:
            if self.cfg_path.exists():
                return json.loads(self.cfg_path.read_text(encoding="utf-8"))
        except Exception:
            LOG.exception("telegram: config invalida")
        return {"enabled": False}

    def alert(self, level: str, title: str, body: str = "") -> bool:
        if not self.cfg.get("enabled"):
            return False
        lvl = level if level in LEVELS else "info"
        min_lvl = self.cfg.get("min_level", "warn")
        if LEVELS[lvl] < LEVELS.get(min_lvl, 1):
            return False
        key = hashlib.md5((lvl + "|" + title).encode("utf-8")).hexdigest()
        now = time.time()
        with self._lock:
            last = self._recent.get(key)
            dd = float(self.cfg.get("dedupe_minutes", 10)) * 60.0
            if last is not None and dd > 0 and now - last < dd:
                return False
            while self._window and now - self._window[0] > 3600.0:
                self._window.popleft()
            if len(self._window) >= int(self.cfg.get("rate_limit_per_hour", 20)):
                self._dropped += 1
                return False
            self._window.append(now)
            self._recent[key] = now
        text = f"[AURA][{lvl.upper()}] {title}"
        if body:
            text += "\n" + str(body)[:3500]
        try:
            self._q.put_nowait((text, key))
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def _worker(self) -> None:
        token = str(self.cfg.get("bot_token", ""))
        chat_id = str(self.cfg.get("chat_id", ""))
        while True:
            item = self._q.get()
            if item is None:
                break
            text, key = item
            ok = False
            for _ in range(2):
                try:
                    if self._sender is not None:
                        ok = bool(self._sender(text))
                    else:
                        ok = _http_send(token, chat_id, text)
                    if ok:
                        break
                except Exception:
                    ok = False
                time.sleep(5)
            if not ok:
                LOG.warning("telegram: falha ao enviar: %s", text[:80])
                with self._lock:
                    self._recent.pop(key, None)

    def stats(self) -> dict:
        with self._lock:
            return {"enabled": bool(self.cfg.get("enabled")),
                    "queued": self._q.qsize(), "dropped": self._dropped}


_INSTANCE = None
_LOCK = threading.Lock()


def get_alerter() -> TelegramAlerter:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = TelegramAlerter()
    return _INSTANCE


def alert(level: str, title: str, body: str = "") -> bool:
    try:
        return get_alerter().alert(level, title, body)
    except Exception:
        return False
