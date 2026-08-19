import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from main import (
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_SIMILARITY_THRESHOLD,
    TeraboxClicker,
)
from cropping_tool import normalize_filename, save_template


class TeraboxClickerCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="terabox-clicker-test-")
        Path(self.temp_dir, "templates").mkdir()
        Path(self.temp_dir, "fallback_templates").mkdir()

    def tearDown(self):
        TeraboxClicker._flush_pending_counts(
            os.path.join(self.temp_dir, "config.json")
        )
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def _write_png(path, image):
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise AssertionError("test image encoding failed")
        encoded.tofile(str(path))

    def _pattern(self, size=12):
        rng = np.random.default_rng(12345)
        return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)

    def test_config_is_validated_and_template_order_is_deduplicated(self):
        template_path = Path(self.temp_dir, "templates", "UPPER.PNG")
        self._write_png(template_path, self._pattern())
        config = {
            "device_address": "emulator-9999",
            "scan_interval": "invalid",
            "no_match_timeout": 0,
            "similarity_threshold": 2,
            "enable_random_click": "false",
            "match_grayscale": "off",
            "template_order": ["UPPER.PNG", "UPPER.PNG", "missing.png"],
            "template_counts": {"UPPER.PNG": -5},
            "template_actions": {"UPPER.PNG": "invalid"},
            "template_offsets": {"UPPER.PNG": ["3", "4"]},
        }
        Path(self.temp_dir, "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

        clicker = TeraboxClicker(base_dir=self.temp_dir, logger=lambda _: None)

        self.assertEqual(clicker.device_address, "emulator-9999")
        self.assertEqual(clicker.scan_interval, DEFAULT_SCAN_INTERVAL)
        self.assertEqual(clicker.no_match_timeout, 0)
        self.assertEqual(
            clicker.similarity_threshold, DEFAULT_SIMILARITY_THRESHOLD
        )
        self.assertFalse(clicker.enable_random_click)
        self.assertFalse(clicker.match_grayscale)
        self.assertEqual(clicker.template_order, ["UPPER.PNG"])
        self.assertEqual(clicker.template_counts["UPPER.PNG"], 0)
        self.assertEqual(clicker.template_actions["UPPER.PNG"], "click")
        self.assertEqual(clicker.template_offsets["UPPER.PNG"], [3, 4])

    def test_template_decode_is_cached_and_click_coordinates_are_clamped(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "pattern.png")
        self._write_png(template_path, template)
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.template_offsets["pattern.png"] = [1000, -1000]

        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[20:32, 30:42] = template
        original_fromfile = np.fromfile
        with patch("main.np.fromfile", wraps=original_fromfile) as fromfile:
            first = clicker.find_template(
                screen, str(template_path), threshold=0.99,
                filename="pattern.png"
            )
            second = clicker.find_template(
                screen, str(template_path), threshold=0.99,
                filename="pattern.png"
            )
            clicker.match_grayscale = False
            third = clicker.find_template(
                screen, str(template_path), threshold=0.99,
                filename="pattern.png"
            )

        self.assertIsNotNone(first)
        self.assertEqual(first[:2], (99, 0))
        self.assertEqual(second[:2], (99, 0))
        self.assertEqual(third[:2], (99, 0))
        self.assertEqual(fromfile.call_count, 2)

    def test_flat_template_uses_stable_matching_method(self):
        template = np.full((8, 8, 3), 255, dtype=np.uint8)
        template_path = Path(self.temp_dir, "templates", "flat.png")
        self._write_png(template_path, template)
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        screen = np.zeros((30, 30, 3), dtype=np.uint8)
        screen[10:18, 12:20] = template

        match = clicker.find_template(
            screen, str(template_path), threshold=0.99, filename="flat.png"
        )

        self.assertIsNotNone(match)
        self.assertEqual(match[:2], (16, 14))

    def test_count_deltas_from_multiple_instances_are_merged(self):
        template_path = Path(self.temp_dir, "templates", "count.png")
        self._write_png(template_path, self._pattern())
        first = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-1",
            logger=lambda _: None,
        )
        self.assertTrue(first.save_config())
        second = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-2",
            logger=lambda _: None,
        )

        self.assertEqual(first._record_match("count.png"), 1)
        self.assertEqual(second._record_match("count.png"), 2)
        self.assertEqual(second._record_match("count.png"), 3)
        self.assertTrue(first.flush_counts())

        config = TeraboxClicker.read_config(first.config_path)
        self.assertEqual(config["template_counts"]["count.png"], 3)

        second.reset_counts()
        config = TeraboxClicker.read_config(first.config_path)
        self.assertEqual(config["template_counts"]["count.png"], 0)

    def test_corrupt_config_is_not_overwritten(self):
        config_path = Path(self.temp_dir, "config.json")
        corrupt_text = "{ definitely-not-json"
        config_path.write_text(corrupt_text, encoding="utf-8")
        messages = []
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=messages.append,
        )

        self.assertFalse(clicker.save_config())
        self.assertEqual(config_path.read_text(encoding="utf-8"), corrupt_text)
        self.assertTrue(any("설정" in message for message in messages))

    def test_adb_connection_is_not_reported_for_unlisted_device(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        command_result = subprocess.CompletedProcess([], 0, "", "")
        clicker._run_adb = Mock(return_value=command_result)

        with patch("main.AdbClient") as client_class:
            client_class.return_value.devices.return_value = []
            connected = clicker.start_adb_server()

        self.assertFalse(connected)
        self.assertIsNone(clicker.device)

    def test_stop_interrupts_a_long_scan_interval(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=30,
            logger=lambda _: None,
        )
        clicker.device = object()
        scanned = threading.Event()

        def run_once():
            scanned.set()
            return False

        clicker.run_once = run_once
        worker = threading.Thread(target=clicker.start_loop, daemon=True)
        worker.start()
        self.assertTrue(scanned.wait(1.0))

        started = time.monotonic()
        clicker.stop_loop()
        worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 1.0)


    def test_template_filename_validation(self):
        self.assertEqual(normalize_filename("button"), "button.png")
        self.assertEqual(normalize_filename("button.PNG"), "button.PNG")
        for invalid in ("../escape", "folder/name", "CON", "bad?.png"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalize_filename(invalid)

    def test_cropping_save_updates_file_and_config_atomically(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        image = self._pattern()
        save_path = save_template(
            clicker, image, "created", (7, 9)
        )

        self.assertTrue(Path(save_path).is_file())
        self.assertIn("created.png", clicker.template_order)
        self.assertEqual(clicker.template_offsets["created.png"], [7, 9])
        self.assertEqual(clicker.template_actions["created.png"], "click")
        config = TeraboxClicker.read_config(clicker.config_path)
        self.assertEqual(config["template_offsets"]["created.png"], [7, 9])
        self.assertFalse(
            list(Path(self.temp_dir, "templates").glob("*.tmp"))
        )

    def test_device_failure_invalidates_device_reference(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        mock_device = Mock()
        mock_device.shell.side_effect = RuntimeError("ADB socket closed")
        clicker.device = mock_device

        self.assertFalse(clicker.click(100, 200))
        self.assertIsNone(clicker.device)

        clicker.device = mock_device
        self.assertFalse(clicker.go_back())
        self.assertIsNone(clicker.device)

    def test_toggle_and_reset_actions(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.template_order = ["t1.png"]
        clicker.template_actions["t1.png"] = "click"
        
        self.assertEqual(clicker.toggle_action("t1.png"), "double_click")
        self.assertEqual(clicker.toggle_action("t1.png"), "back")
        self.assertEqual(clicker.toggle_action("t1.png"), "click")

        clicker.fallback_template_order = ["fb1.png"]
        clicker.fallback_template_actions["fb1.png"] = "click"
        self.assertEqual(clicker.toggle_fallback_action("fb1.png"), "double_click")

    def test_no_match_action_dispatch(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        mock_device = Mock()
        clicker.device = mock_device

        clicker.no_match_action = "custom_click"
        clicker.no_match_coords = [120, 340]
        clicker.no_match_interval = 0.1
        clicker.last_match_time = time.monotonic() - 10
        clicker.last_random_click_time = time.monotonic() - 10

        screen = np.zeros((400, 300, 3), dtype=np.uint8)
        clicker._handle_no_match_action(screen, time.monotonic())

        mock_device.shell.assert_called_with("input tap 120 340")

    def test_timers_status_calculation(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
            no_match_timeout=60,
        )
        clicker.no_match_action = "fallback_list"
        clicker.no_match_interval = 30
        now = time.monotonic()
        clicker.last_match_time = now - 10
        clicker.last_random_click_time = now - 10
        clicker.last_action_time = now - 20

        status = clicker.get_timers_status()
        self.assertAlmostEqual(status["no_match_elapsed"], 10, delta=0.5)
        self.assertAlmostEqual(status["no_match_remaining"], 20, delta=0.5)
        self.assertAlmostEqual(status["timeout_elapsed"], 20, delta=0.5)
        self.assertAlmostEqual(status["timeout_remaining"], 40, delta=0.5)
        self.assertEqual(status["no_match_action"], "fallback_list")

    def test_normalize_device_serials(self):
        # 127.0.0.1:5555 covers emulator-5554, so emulator-5554 is deduplicated.
        # emulator-5556 has no corresponding IP, so it is safely kept as-is.
        raw = ["emulator-5554", "127.0.0.1:5555", "emulator-5556", "192.168.1.100:5555", "RF8N10XXXXX"]
        normalized = TeraboxClicker.normalize_device_serials(raw)
        self.assertEqual(
            normalized,
            ["127.0.0.1:5555", "192.168.1.100:5555", "emulator-5556", "RF8N10XXXXX"]
        )

    def test_execute_template(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "test_btn.png")
        self._write_png(template_path, template)

        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        mock_device = Mock()
        clicker.device = mock_device
        clicker.template_order = ["test_btn.png"]
        clicker.template_actions = {"test_btn.png": "click"}
        clicker.template_counts = {"test_btn.png": 0}

        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[20:32, 30:42] = template
        clicker.capture_screen = Mock(return_value=screen)

        success = clicker.execute_template("test_btn.png", is_fallback=False)
        self.assertTrue(success)
        mock_device.shell.assert_called_with("input tap 36 26")
        self.assertEqual(clicker.template_counts["test_btn.png"], 1)

    def test_fallback_final_action_execution(self):
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        mock_device = Mock()
        clicker.device = mock_device
        clicker.no_match_action = "fallback_list"
        clicker.fallback_final_action = "custom_double_click"
        clicker.fallback_final_coords = [250, 450]
        clicker.no_match_interval = 0.1
        clicker.last_match_time = time.monotonic() - 10
        clicker.last_random_click_time = time.monotonic() - 10

        screen = np.zeros((800, 600, 3), dtype=np.uint8)
        clicker._handle_no_match_action(screen, time.monotonic())

        self.assertEqual(mock_device.shell.call_count, 2)
        mock_device.shell.assert_called_with("input tap 250 450")

    def test_timeout_callback_is_triggered_only_once_until_match(self):
        callback_mock = Mock()
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
            no_match_timeout=5,
            on_timeout_callback=callback_mock,
        )
        clicker.last_action_time = time.monotonic() - 10
        self.assertFalse(clicker._timeout_alert_active)

        # Simulate timeout condition in start_loop check
        timeout = clicker.no_match_timeout
        if timeout > 0 and time.monotonic() - clicker.last_action_time >= timeout:
            if not clicker._timeout_alert_active:
                clicker._timeout_alert_active = True
                clicker.on_timeout_callback(timeout)

        self.assertTrue(clicker._timeout_alert_active)
        self.assertEqual(callback_mock.call_count, 1)

        # Second loop iteration under same condition should NOT call callback again
        if timeout > 0 and time.monotonic() - clicker.last_action_time >= timeout:
            if not clicker._timeout_alert_active:
                clicker._timeout_alert_active = True
                clicker.on_timeout_callback(timeout)

        self.assertEqual(callback_mock.call_count, 1)

        # Successful match resets _timeout_alert_active
        now = time.monotonic()
        clicker.last_action_time = now
        clicker._timeout_alert_active = False
        self.assertFalse(clicker._timeout_alert_active)

    def test_double_click_interval_configuration(self):
        config = {
            "device_address": "emulator-test",
            "double_click_interval": 0.35,
        }
        Path(self.temp_dir, "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        self.assertEqual(clicker.double_click_interval, 0.35)

        mock_device = Mock()
        clicker.device = mock_device
        waited_delays = []
        clicker._wait_after_action = lambda sec: (waited_delays.append(sec), True)[1]
        
        # Test default delay uses double_click_interval
        clicker.double_click(100, 200)
        self.assertEqual(mock_device.shell.call_count, 2)
        self.assertEqual(waited_delays, [0.35])

        # Test explicit delay override
        waited_delays.clear()
        clicker.double_click(100, 200, delay=2.5)
        self.assertEqual(waited_delays, [2.5])

    def test_template_delays_and_execution_wait(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "delayed_btn.png")
        self._write_png(template_path, template)

        config = {
            "device_address": "emulator-test",
            "template_order": ["delayed_btn.png"],
            "template_delays": {"delayed_btn.png": 1.5},
        }
        Path(self.temp_dir, "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        clicker = TeraboxClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        self.assertEqual(clicker.template_delays.get("delayed_btn.png"), 1.5)

        mock_device = Mock()
        clicker.device = mock_device
        waited_delays = []
        clicker._wait_after_action = lambda sec: (waited_delays.append(sec), True)[1]

        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[20:32, 30:42] = template
        clicker.capture_screen = Mock(return_value=screen)

        # Test run_once waits configured delay before action
        success = clicker.run_once()
        self.assertTrue(success)
        mock_device.shell.assert_called_with("input tap 36 26")
        # First wait is the pre-action delay (1.5), second is post-action cooldown (2.0)
        self.assertIn(1.5, waited_delays)

        # Test setting delay and saving
        clicker.set_template_delay("delayed_btn.png", 3.0)
        self.assertEqual(clicker.template_delays["delayed_btn.png"], 3.0)
        reloaded = TeraboxClicker.read_config(clicker.config_path)
        self.assertEqual(reloaded["template_delays"]["delayed_btn.png"], 3.0)

        # Test deleting template removes delay entry
        clicker.delete_template("delayed_btn.png")
        self.assertNotIn("delayed_btn.png", clicker.template_delays)


if __name__ == "__main__":
    unittest.main()

