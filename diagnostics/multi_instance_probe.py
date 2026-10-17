"""Run bounded, synthetic hang diagnostics without ADB or user configuration."""
import json
import queue
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from autoclicker.core.clicker import AutoClicker
from autoclicker.gui.main_window import App
from autoclicker.gui.runtime import LatestUpdates


def diagnose():
    results = {}
    app = SimpleNamespace(_closing=False, _ui_queue=queue.Queue(), after=Mock(),
                          _latest_updates=LatestUpdates(), _diagnostics=Mock(), tab_frames={})
    app._drain_ui_queue = lambda: App._drain_ui_queue(app)
    callback_entered = threading.Event()
    produced = threading.Event()

    def modal_callback():
        callback_entered.set()
        produced.wait(2)
        # Simulate a modal callback that has not returned to the pump.
        results['blocked_callback_queue_depth'] = app._ui_queue.qsize()
        results['pump_rescheduled_while_callback_blocked'] = app.after.called

    def producer():
        callback_entered.wait(2)
        for _ in range(5 * 100):
            App.post_to_ui(app, lambda: None)
        produced.set()

    app._ui_queue.put(modal_callback)
    worker = threading.Thread(target=producer)
    worker.start()
    App._drain_ui_queue(app)
    worker.join()

    app._ui_queue = queue.Queue()
    for _ in range(100):
        app._ui_queue.put(lambda: time.sleep(0.002))
    started = time.perf_counter()
    App._drain_ui_queue(app)
    results['100_callbacks_at_2ms_pump_block_ms'] = round((time.perf_counter()-started)*1000, 1)

    with tempfile.TemporaryDirectory(prefix='multinstance-', dir=Path(__file__).parent) as folder:
        base = Path(folder)
        (base / 'templates').mkdir()
        (base / 'fallback_templates').mkdir()
        clicker = AutoClicker(base_dir=folder, logger=lambda _: None)
        acquired = threading.Event()

        def hold_connection_lock():
            with clicker._device_lock:
                acquired.set()
                time.sleep(0.25)

        worker = threading.Thread(target=hold_connection_lock)
        worker.start()
        acquired.wait(2)
        started = time.perf_counter()
        clicker.disconnect()
        results['disconnect_blocked_by_device_lock_ms'] = round((time.perf_counter()-started)*1000, 1)
        worker.join()

        rng = np.random.default_rng(1234)
        template = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        path = base / 'templates' / 'sample.png'
        cv2.imencode('.png', template)[1].tofile(str(path))
        AutoClicker.invalidate_template_cache(path)
        for count in (5, 8):
            barrier = threading.Barrier(count)

            def match_worker(index):
                matcher = AutoClicker(base_dir=folder, logger=lambda _: None)
                screen = np.zeros((720, 1280, 3), dtype=np.uint8)
                screen[200:232, 300:332] = template
                barrier.wait(timeout=10)
                for _ in range(30):
                    match = matcher.find_template(screen, str(path), threshold=0.99, filename='sample.png')
                    if match is None or match[:2] != (316, 216):
                        raise AssertionError((index, match))
                return 30

            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=count) as executor:
                matches = sum(executor.map(match_worker, range(count)))
            results[f'{count}_workers_synthetic_matching'] = {
                'successful_matches': matches,
                'seconds': round(time.perf_counter()-started, 3),
                'resolution': '1280x720',
                'adb_exercised': False,
            }
        AutoClicker.invalidate_template_cache()
    return results


if __name__ == '__main__':
    print(json.dumps(diagnose(), indent=2))
