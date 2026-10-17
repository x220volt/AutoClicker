"""Bounded UI state delivery and independent hang diagnostics."""
import copy
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict


class LatestUpdates:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = OrderedDict()

    def put(self, key, callback):
        with self.lock:
            self.items[key] = callback

    def pop(self):
        with self.lock:
            return self.items.popitem(last=False)[1] if self.items else None

    def __len__(self):
        with self.lock:
            return len(self.items)


class HangMonitor:
    def __init__(self, directory):
        self.logger = logging.Logger("autoclicker.runtime")
        try:
            os.makedirs(directory, exist_ok=True)
            handler = RotatingFileHandler(os.path.join(directory, "runtime.log"),
                                          maxBytes=2_000_000, backupCount=2, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.logger.addHandler(handler)
        except OSError:
            self.logger.addHandler(logging.NullHandler())
        self.last_beat = time.monotonic()
        self.state = {}
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._watch, daemon=True)
        self.worker.start()

    def beat(self, **state):
        self.last_beat = time.monotonic()
        self.state = state

    def _watch(self):
        last_report = 0.0
        while not self.stop.wait(1):
            now = time.monotonic()
            if now - self.last_beat > 5 and now - last_report > 30:
                last_report = now
                stacks = "\n".join(
                    f"Thread {ident}\n{''.join(traceback.format_stack(frame))}"
                    for ident, frame in sys._current_frames().items())
                self.logger.warning("UI heartbeat stalled: %s\n%s", self.state, stacks)

    def close(self):
        self.stop.set()
        self.worker.join(1.5)
        for handler in self.logger.handlers:
            handler.close()


class SettingsWriter:
    """A single writer coalesces pending snapshots and drains before shutdown."""
    def __init__(self, persist, logger):
        self.persist = persist
        self.logger = logger
        self.condition = threading.Condition()
        self.pending = None
        self.stopping = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, *snapshot):
        with self.condition:
            if self.stopping:
                return False
            self.pending = copy.deepcopy(snapshot)
            self.condition.notify()
        return True

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.stopping)
                if self.pending is None:
                    return
                snapshot, self.pending = self.pending, None
            try:
                if self.persist(*snapshot) is False:
                    self.logger.error("Settings save failed")
            except Exception:
                self.logger.exception("Settings save failed")

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.worker.join()
