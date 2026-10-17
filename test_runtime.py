import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from autoclicker.core.clicker import AutoClicker
from autoclicker.gui.main_window import App
from autoclicker.gui.runtime import LatestUpdates, SettingsWriter, HangMonitor


class RuntimeTests(unittest.TestCase):
    def test_latest_state_is_coalesced(self):
        updates = LatestUpdates()
        for count in range(10000):
            updates.put("counter", lambda value=count: value)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates.pop()(), 9999)

    def test_pump_yields_and_logs_callback_errors(self):
        app = SimpleNamespace(_closing=False, _ui_queue=queue.Queue(),
            _latest_updates=LatestUpdates(), _diagnostics=Mock(), after=Mock(),
            _drain_ui_queue=Mock(), tab_frames={})
        app._ui_queue.put(lambda: (_ for _ in ()).throw(ValueError("test")))
        for _ in range(100):
            app._ui_queue.put(lambda: time.sleep(0.003))
        App._drain_ui_queue(app)
        self.assertGreater(app._ui_queue.qsize(), 80)
        app.after.assert_called_once()
        app._diagnostics.logger.exception.assert_called_once()

    def test_writer_coalesces_and_drains_before_close(self):
        entered, release = threading.Event(), threading.Event()
        values = []
        def persist(value):
            entered.set()
            release.wait(2)
            values.append(value)
        writer = SettingsWriter(persist, Mock())
        writer.submit(0)
        self.assertTrue(entered.wait(2))
        for value in range(1, 100):
            writer.submit(value)
        release.set()
        writer.close()
        self.assertEqual(values, [0, 99])
        self.assertFalse(writer.worker.is_alive())

    def test_watchdog_records_stacks_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            monitor = HangMonitor(folder)
            monitor.last_beat = time.monotonic() - 10
            deadline = time.monotonic() + 3
            path = Path(folder, "runtime.log")
            while time.monotonic() < deadline and not path.stat().st_size:
                time.sleep(0.05)
            monitor.close()
            self.assertIn("UI heartbeat stalled", path.read_text(encoding="utf-8"))
            self.assertFalse(monitor.worker.is_alive())

    def test_capture_limits_concurrency_and_cancels_waiter(self):
        with tempfile.TemporaryDirectory() as folder:
            clickers = [AutoClicker(base_dir=folder, logger=lambda _: None) for _ in range(8)]
            lock = threading.Lock()
            active = 0
            peak = 0
            def capture(grayscale):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.04)
                with lock:
                    active -= 1
                return "frame"
            for clicker in clickers:
                clicker._capture_screen_unlocked = capture
            results = []
            workers = [threading.Thread(target=lambda c=c: results.append(c.capture_screen()))
                       for c in clickers]
            for worker in workers: worker.start()
            for worker in workers: worker.join(3)
            self.assertEqual(results, ["frame"] * 8)
            self.assertLessEqual(peak, 3)
            clicker = clickers[0]
            clicker._capture_lock.acquire()
            worker = threading.Thread(target=clicker.capture_screen)
            worker.start()
            clicker.request_shutdown()
            worker.join(1)
            clicker._capture_lock.release()
            self.assertFalse(worker.is_alive())

    def test_disconnect_does_not_wait_for_connect_and_rejects_stale_result(self):
        with tempfile.TemporaryDirectory() as folder:
            clicker = AutoClicker(base_dir=folder, logger=lambda _: None)
            entered, release = threading.Event(), threading.Event()
            def ensure():
                entered.set()
                release.wait(2)
                return True
            clicker._ensure_adb_server = ensure
            clicker._run_adb_stage = Mock(return_value=SimpleNamespace(stdout="127.0.0.1:5555 device", returncode=0))
            worker = threading.Thread(target=clicker.start_adb_server)
            worker.start()
            self.assertTrue(entered.wait(2))
            started = time.perf_counter()
            clicker.disconnect()
            self.assertLess(time.perf_counter()-started, 0.2)
            release.set()
            worker.join(2)
            self.assertIsNone(clicker.device)

    def test_socket_size_limit_and_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            clicker = AutoClicker(base_dir=folder, logger=lambda _: None)
            sock = Mock()
            sock.recv.return_value = b"x" * (1024 * 1024)
            with self.assertRaises(OSError):
                clicker._read_capture_payload(sock)
            clicker._capture_context.deadline = time.monotonic() - 1
            with self.assertRaises(TimeoutError):
                clicker._read_capture_payload(sock)


if __name__ == "__main__":
    unittest.main()
