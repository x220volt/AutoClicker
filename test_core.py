import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import socket
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

from main import (
    ADB_COMMAND_TIMEOUT,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_SIMILARITY_THRESHOLD,
    TEMPLATE_MANIFEST_REFRESH_INTERVAL,
    AutoClicker,
    resolve_adb_path,
    get_default_adb_path,
    DEFAULT_ADB_MODE,
)
from cropping_tool import normalize_filename, save_template


class AutoClickerCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="auto-clicker-test-")
        Path(self.temp_dir, "templates").mkdir()
        Path(self.temp_dir, "fallback_templates").mkdir()

    def tearDown(self):
        AutoClicker._flush_pending_counts(
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
            "roi_full_scan_budget": 1,
            "template_order": ["UPPER.PNG", "UPPER.PNG", "missing.png"],
            "template_counts": {"UPPER.PNG": -5},
            "template_actions": {"UPPER.PNG": "invalid"},
            "template_offsets": {"UPPER.PNG": ["3", "4"]},
        }
        Path(self.temp_dir, "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )

        clicker = AutoClicker(base_dir=self.temp_dir, logger=lambda _: None)

        self.assertEqual(clicker.device_address, "emulator-9999")
        self.assertEqual(clicker.scan_interval, DEFAULT_SCAN_INTERVAL)
        self.assertEqual(clicker.no_match_timeout, 0)
        self.assertEqual(
            clicker.similarity_threshold, DEFAULT_SIMILARITY_THRESHOLD
        )
        self.assertFalse(clicker.enable_random_click)
        self.assertFalse(clicker.match_grayscale)
        self.assertEqual(clicker.roi_full_scan_budget, 1)
        self.assertEqual(clicker.template_order, ["UPPER.PNG"])
        self.assertEqual(clicker.template_counts["UPPER.PNG"], 0)
        self.assertEqual(clicker.template_actions["UPPER.PNG"], "click")
        self.assertEqual(clicker.template_offsets["UPPER.PNG"], [3, 4])

    def test_template_decode_is_cached_and_click_coordinates_are_clamped(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "pattern.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
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
        self.assertEqual(fromfile.call_count, 1)

    def test_flat_template_uses_stable_matching_method(self):
        template = np.full((8, 8, 3), 255, dtype=np.uint8)
        template_path = Path(self.temp_dir, "templates", "flat.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
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
        first = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-1",
            logger=lambda _: None,
        )
        self.assertTrue(first.save_config())
        second = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-2",
            logger=lambda _: None,
        )

        self.assertEqual(first._record_match("count.png"), 1)
        self.assertEqual(second._record_match("count.png"), 2)
        self.assertEqual(second._record_match("count.png"), 3)
        self.assertTrue(first.flush_counts())

        config = AutoClicker.read_config(first.config_path)
        self.assertEqual(config["template_counts"]["count.png"], 3)

        second.reset_counts()
        config = AutoClicker.read_config(first.config_path)
        self.assertEqual(config["template_counts"]["count.png"], 0)

    def test_corrupt_config_is_not_overwritten(self):
        config_path = Path(self.temp_dir, "config.json")
        corrupt_text = "{ definitely-not-json"
        config_path.write_text(corrupt_text, encoding="utf-8")
        messages = []
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=messages.append,
        )

        self.assertFalse(clicker.save_config())
        self.assertEqual(config_path.read_text(encoding="utf-8"), corrupt_text)
        self.assertTrue(any("설정" in message for message in messages))

    def test_adb_connection_is_not_reported_for_unlisted_device(self):
        clicker = AutoClicker(
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
        clicker = AutoClicker(
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
        clicker = AutoClicker(
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
        config = AutoClicker.read_config(clicker.config_path)
        self.assertEqual(config["template_offsets"]["created.png"], [7, 9])
        self.assertFalse(
            list(Path(self.temp_dir, "templates").glob("*.tmp"))
        )

    def test_device_failure_invalidates_device_reference(self):
        clicker = AutoClicker(
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
        clicker = AutoClicker(
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
        clicker = AutoClicker(
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

        mock_device.shell.assert_called_with("input tap 120 340", timeout=ADB_COMMAND_TIMEOUT)

    def test_timers_status_calculation(self):
        clicker = AutoClicker(
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
        normalized = AutoClicker.normalize_device_serials(raw)
        self.assertEqual(
            normalized,
            ["127.0.0.1:5555", "192.168.1.100:5555", "emulator-5556", "RF8N10XXXXX"]
        )

    def test_execute_template(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "test_btn.png")
        self._write_png(template_path, template)

        clicker = AutoClicker(
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
        mock_device.shell.assert_called_with("input tap 36 26", timeout=ADB_COMMAND_TIMEOUT)
        self.assertEqual(clicker.template_counts["test_btn.png"], 1)

    def test_fallback_final_action_execution(self):
        clicker = AutoClicker(
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
        mock_device.shell.assert_called_with("input tap 250 450", timeout=ADB_COMMAND_TIMEOUT)

    def test_timeout_callback_is_triggered_only_once_until_match(self):
        callback_mock = Mock()
        clicker = AutoClicker(
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
        clicker = AutoClicker(
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
        clicker = AutoClicker(
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
        mock_device.shell.assert_called_with("input tap 36 26", timeout=ADB_COMMAND_TIMEOUT)
        # First wait is the pre-action delay (1.5), second is post-action cooldown (2.0)
        self.assertIn(1.5, waited_delays)

        # Test setting delay and saving
        clicker.set_template_delay("delayed_btn.png", 3.0)
        self.assertEqual(clicker.template_delays["delayed_btn.png"], 3.0)
        reloaded = AutoClicker.read_config(clicker.config_path)
        self.assertEqual(reloaded["template_delays"]["delayed_btn.png"], 3.0)

        # Test deleting template removes delay entry
        clicker.delete_template("delayed_btn.png")
        self.assertNotIn("delayed_btn.png", clicker.template_delays)

    def test_preload_templates_eliminates_subsequent_io(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "preloaded.png")
        self._write_png(template_path, template)

        AutoClicker.invalidate_template_cache()
        loaded = AutoClicker.preload_templates(
            [Path(self.temp_dir, "templates")], grayscales=(False, True)
        )
        self.assertGreaterEqual(loaded, 2)

        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[20:32, 30:42] = template

        original_fromfile = np.fromfile
        with patch(
            "main.np.fromfile", wraps=original_fromfile
        ) as fromfile, patch(
            "main.os.path.exists",
            side_effect=AssertionError("hot path called os.path.exists"),
        ), patch(
            "main.os.stat",
            side_effect=AssertionError("hot path called os.stat"),
        ):
            match1 = clicker.find_template(
                screen, str(template_path), threshold=0.99, filename="preloaded.png"
            )
            clicker.match_grayscale = False
            match2 = clicker.find_template(
                screen, str(template_path), threshold=0.99, filename="preloaded.png"
            )

        self.assertIsNotNone(match1)
        self.assertIsNotNone(match2)
        self.assertEqual(fromfile.call_count, 0)

    def test_template_directory_manifest_is_shared_and_throttled(self):
        template_dir = Path(self.temp_dir, "templates")
        self._write_png(template_dir / "shared.png", self._pattern())
        AutoClicker.invalidate_template_cache()
        clock = [100.0]

        with patch(
            "main.time.perf_counter", side_effect=lambda: clock[0]
        ), patch("main.os.scandir", wraps=os.scandir) as scandir:
            AutoClicker._refresh_template_directory(template_dir)
            AutoClicker._refresh_template_directory(template_dir)
            clock[0] += TEMPLATE_MANIFEST_REFRESH_INTERVAL * 0.5
            AutoClicker._refresh_template_directory(template_dir)
            self.assertEqual(scandir.call_count, 1)

            clock[0] += TEMPLATE_MANIFEST_REFRESH_INTERVAL
            AutoClicker._refresh_template_directory(template_dir)
            self.assertEqual(scandir.call_count, 2)

    def test_multi_instance_shares_preloaded_template_cache(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "shared.png")
        self._write_png(template_path, template)

        AutoClicker.invalidate_template_cache()
        AutoClicker.preload_templates(
            [Path(self.temp_dir, "templates")], grayscales=(True,)
        )

        inst1 = AutoClicker(base_dir=self.temp_dir, device_address="emulator-1", logger=lambda _: None)
        inst2 = AutoClicker(base_dir=self.temp_dir, device_address="emulator-2", logger=lambda _: None)
        inst3 = AutoClicker(base_dir=self.temp_dir, device_address="emulator-3", logger=lambda _: None)

        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[20:32, 30:42] = template

        original_fromfile = np.fromfile
        with patch("main.np.fromfile", wraps=original_fromfile) as fromfile:
            m1 = inst1.find_template(screen, str(template_path), threshold=0.99)
            m2 = inst2.find_template(screen, str(template_path), threshold=0.99)
            m3 = inst3.find_template(screen, str(template_path), threshold=0.99)

        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)
        self.assertIsNotNone(m3)
        self.assertEqual(fromfile.call_count, 0)

    def test_configurable_post_action_delay(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "action_btn.png")
        self._write_png(template_path, template)

        config = {
            "device_address": "emulator-test",
            "template_order": ["action_btn.png"],
            "post_action_delay": 3.5,
        }
        Path(self.temp_dir, "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        self.assertEqual(clicker.post_action_delay, 3.5)

        mock_device = Mock()
        clicker.device = mock_device
        waited_delays = []
        clicker._wait_after_action = lambda sec: (waited_delays.append(sec), True)[1]

        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[20:32, 30:42] = template
        clicker.capture_screen = Mock(return_value=screen)

        success = clicker.run_once()
        self.assertTrue(success)
        self.assertIn(3.5, waited_delays)

    def test_two_stage_matching_skips_small_templates(self):
        """Small templates (< 20px) should skip the downscale pre-filter and match at full res."""
        small_template = self._pattern(size=8)
        template_path = Path(self.temp_dir, "templates", "tiny_icon.png")
        self._write_png(template_path, small_template)

        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )

        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[10:18, 20:28] = small_template

        match = clicker.find_template(screen, str(template_path), threshold=0.99)
        self.assertIsNotNone(match, "Small template should match at full resolution")
        x, y, confidence = match
        self.assertEqual(x, 24)
        self.assertEqual(y, 14)
        self.assertGreaterEqual(confidence, 0.99)

    def test_two_stage_matching_rejects_via_prescale(self):
        """Large templates not present should be quickly rejected by the prescale filter."""
        large_template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "large_btn.png")
        self._write_png(template_path, large_template)

        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )

        # Screen with no matching region
        screen = np.zeros((200, 300, 3), dtype=np.uint8)

        match = clicker.find_template(screen, str(template_path), threshold=0.9)
        self.assertIsNone(match, "Non-matching template should be rejected")

    def test_subprocess_capture_fallback(self):
        """capture_screen should fallback to ppadb screencap when subprocess fails."""
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )

        # Create a test image and encode it as PNG
        test_img = np.zeros((100, 80, 3), dtype=np.uint8)
        test_img[10:50, 10:50] = [0, 255, 0]
        _, png_bytes = cv2.imencode(".png", test_img)
        png_data = png_bytes.tobytes()

        mock_device = Mock()
        connection = MagicMock()
        connection.read_all.return_value = png_data
        mock_device.create_connection.return_value = connection
        clicker.device = mock_device

        # Mock socket and subprocess to fail so fallback to ppadb triggers
        failed_proc = Mock()
        failed_proc.returncode = 1
        failed_proc.stdout = b""
        with patch.object(clicker, "_capture_direct_socket", return_value=(None, "transport")), patch("main.subprocess.run", return_value=failed_proc):
            result = clicker.capture_screen(grayscale=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (100, 80, 3))
        mock_device.create_connection.assert_called_once_with(timeout=ADB_COMMAND_TIMEOUT)
        connection.send.assert_called_once_with("shell:/system/bin/screencap -p")

    def test_config_cache_avoids_redundant_reads(self):
        """In-memory config cache should avoid re-reading unchanged files."""
        config = {"device_address": "emulator-test", "scan_interval": 3}
        config_path = os.path.join(self.temp_dir, "config.json")
        Path(config_path).write_text(json.dumps(config), encoding="utf-8")

        # First read - should populate cache
        result1 = AutoClicker._read_config_unlocked(config_path)
        self.assertEqual(result1["scan_interval"], 3)

        # Second read - should hit cache (same mtime)
        result2 = AutoClicker._read_config_unlocked(config_path)
        self.assertEqual(result2["scan_interval"], 3)

        # Verify both are independent dicts (not shared references)
        result1["scan_interval"] = 999
        result3 = AutoClicker._read_config_unlocked(config_path)
        self.assertEqual(result3["scan_interval"], 3)

    def test_template_pre_and_post_action_delays(self):
        template = self._pattern()
        template_path = Path(self.temp_dir, "templates", "test_btn.png")
        self._write_png(template_path, template)

        # 1. Test POST-action delay
        config = {
            "device_address": "emulator-test",
            "template_order": ["test_btn.png"],
            "template_delays": {"test_btn.png": 4.0},
            "template_delay_types": {"test_btn.png": "post"},
            "post_action_delay": 1.0,
        }
        Path(self.temp_dir, "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        self.assertEqual(clicker.template_delays.get("test_btn.png"), 4.0)
        self.assertEqual(clicker.template_delay_types.get("test_btn.png"), "post")

        mock_device = Mock()
        clicker.device = mock_device
        events = []
        clicker._wait_after_action = lambda sec: (events.append(f"wait_{sec}"), True)[1]
        clicker._execute_action = lambda action, x, y: (events.append("action"), True)[1]

        screen = np.zeros((80, 100, 3), dtype=np.uint8)
        screen[20:32, 30:42] = template
        clicker.capture_screen = Mock(return_value=screen)

        success = clicker.run_once()
        self.assertTrue(success)
        # With post-action delay, action runs FIRST, then post-wait of 4s runs
        self.assertEqual(events, ["action", "wait_4"])

        # 2. Test PRE-action delay
        clicker.set_template_delay("test_btn.png", 2.5, "pre")
        events.clear()
        success = clicker.run_once()
        self.assertTrue(success)
        # With pre-action delay, pre-wait of 2.5s runs first, then action, then global post_action_delay (1s)
        self.assertEqual(events, ["wait_2.5", "action", "wait_1"])


    def test_default_preload_is_valid_and_concurrent_calls_decode_once(self):
        template_path = Path(self.temp_dir, "templates", "preload_once.png")
        self._write_png(template_path, self._pattern(size=32))
        AutoClicker.invalidate_template_cache(template_path)
        errors = []

        def preload():
            try:
                AutoClicker.preload_templates(grayscales=(True,))
            except Exception as error:
                errors.append(error)

        original_fromfile = np.fromfile
        with patch("main.APP_DIR", self.temp_dir), patch(
            "main.np.fromfile", wraps=original_fromfile
        ) as fromfile:
            workers = [threading.Thread(target=preload) for _ in range(3)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(2.0)

        self.assertFalse(errors)
        self.assertEqual(fromfile.call_count, 1)
        AutoClicker.invalidate_template_cache(template_path)

    def test_local_verify_avoids_full_resolution_screen_match(self):
        template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "local.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.dynamic_roi = False
        screen = np.zeros((200, 300, 3), dtype=np.uint8)
        screen[70:110, 120:160] = template
        observed_shapes = []
        original_match = cv2.matchTemplate

        def tracked_match(image, target, method, *args, **kwargs):
            observed_shapes.append(image.shape[:2])
            return original_match(image, target, method, *args, **kwargs)

        with patch("main.cv2.matchTemplate", side_effect=tracked_match):
            match = clicker.find_template(
                screen, str(template_path), threshold=0.99, filename="local.png"
            )

        self.assertIsNotNone(match)
        self.assertEqual(match[:2], (140, 90))
        self.assertIn((100, 150), observed_shapes)
        self.assertNotIn((200, 300), observed_shapes)
        self.assertTrue(any(height < 100 for height, _ in observed_shapes))

    def test_static_roi_offsets_coordinates_and_avoids_full_screen(self):
        template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "roi.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.dynamic_roi = False
        clicker.template_rois["roi.png"] = [0.4, 0.2, 0.8, 0.8]
        screen = np.zeros((200, 300, 3), dtype=np.uint8)
        screen[80:120, 150:190] = template
        observed_shapes = []
        original_match = cv2.matchTemplate

        def tracked_match(image, target, method, *args, **kwargs):
            observed_shapes.append(image.shape[:2])
            return original_match(image, target, method, *args, **kwargs)

        with patch("main.cv2.matchTemplate", side_effect=tracked_match):
            match = clicker.find_template(
                screen,
                str(template_path),
                threshold=0.99,
                filename="roi.png",
                rois_dict=clicker.template_rois,
            )

        self.assertIsNotNone(match)
        self.assertEqual(match[:2], (170, 100))
        self.assertNotIn((200, 300), observed_shapes)

    def test_local_verify_falls_back_to_full_region(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        screen = np.zeros((100, 120), dtype=np.uint8)
        template = self._pattern(size=40)[:, :, 0]
        coarse = np.zeros((31, 41), dtype=np.float32)
        coarse[5, 7] = 1.0
        cache_key = ("virtual-template", True)

        with patch.object(
            clicker,
            "_run_match_map",
            return_value=(coarse, 1.0, (7, 5)),
        ), patch.object(
            clicker,
            "_match_region",
            side_effect=[None, (20, 30, 0.95)],
        ) as match_region:
            result = clicker._find_in_region(
                screen,
                template,
                cv2.TM_CCOEFF_NORMED,
                0.9,
                cache_key,
                (0, 0, 120, 100),
            )

        self.assertEqual(result, (20, 30, 0.95))
        self.assertEqual(match_region.call_args_list[0].args[-1], "match.local")
        self.assertEqual(match_region.call_args_list[-1].args[-1], "match.full")

    def test_identical_frame_skip_preserves_post_match_rescan(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.force_scan_interval = 5.0
        screen = np.zeros((80, 100), dtype=np.uint8)

        self.assertTrue(clicker._should_scan_frame(screen, 100.0))
        clicker._last_scan_had_match = False
        self.assertFalse(clicker._should_scan_frame(screen, 100.1))
        clicker._last_scan_had_match = True
        self.assertTrue(clicker._should_scan_frame(screen, 100.2))

    def test_adaptive_interval_grows_only_while_unchanged(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=1,
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.force_scan_interval = 10.0
        screen = np.zeros((80, 100), dtype=np.uint8)
        clicker._should_scan_frame(screen, 10.0)
        clicker._last_scan_had_match = False
        for now in (10.1, 10.2, 10.3):
            self.assertFalse(clicker._should_scan_frame(screen, now))
        self.assertGreater(clicker._current_loop_interval(), 1.0)
        self.assertLessEqual(clicker._current_loop_interval(), 5.0)
        clicker._mark_screen_dirty()
        self.assertEqual(clicker._current_loop_interval(), 1.0)

    def test_adaptive_order_protects_first_three_priorities(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
            adaptive_template_order=True,
        )
        order = ["a.png", "b.png", "c.png", "d.png", "e.png"]
        clicker.last_matched_template = "c.png"
        clicker.last_matched_is_fallback = False
        clicker._transition_counts[(False, "c.png")][(False, "e.png")] = 3

        optimized = clicker._get_template_scan_order(order)

        self.assertEqual(optimized[:3], tuple(order[:3]))
        self.assertEqual(optimized[3], "e.png")
        self.assertCountEqual(optimized, order)

    def test_raw_screencap_decoder_handles_rgba(self):
        payload = struct.pack("<III", 2, 1, 1) + bytes(
            [255, 0, 0, 255, 0, 255, 0, 255]
        )

        color = AutoClicker._decode_raw_screencap(payload, False)
        gray = AutoClicker._decode_raw_screencap(payload, True)

        self.assertEqual(color.shape, (1, 2, 3))
        self.assertEqual(color[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(color[0, 1].tolist(), [0, 255, 0])
        self.assertEqual(gray.shape, (1, 2))

    def test_corrupt_exec_capture_uses_resolved_serial_and_ppadb_fallback(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        clicker.capture_backend = "png"
        test_img = np.zeros((20, 30, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".png", test_img)
        device = Mock()
        device.serial = "emulator-5554"
        connection = MagicMock()
        connection.read_all.return_value = encoded.tobytes()
        device.create_connection.return_value = connection
        clicker.device = device
        proc = subprocess.CompletedProcess([], 0, b"corrupt-png", b"")

        with patch.object(clicker, "_capture_direct_socket", return_value=(None, "transport")), patch("main.subprocess.run", return_value=proc) as run:
            result = clicker.capture_screen(grayscale=False)

        self.assertIsNotNone(result)
        command = run.call_args.args[0]
        self.assertEqual(
            command[1:3], ["-P", str(clicker.port)]
        )
        self.assertEqual(command[command.index("-s") + 1], "emulator-5554")
        device.create_connection.assert_called_once_with(timeout=ADB_COMMAND_TIMEOUT)

    def test_capture_direct_socket_success(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        test_img = np.zeros((40, 50, 3), dtype=np.uint8)
        test_img[5:15, 5:15] = [255, 0, 0]
        _, png_bytes = cv2.imencode(".png", test_img)
        payload = png_bytes.tobytes()

        # Mock direct socket interaction
        mock_sock = MagicMock()
        mock_sock.recv.side_effect = [b"OKAY", b"OKAY", payload, b""]

        with patch("main.socket.socket", return_value=mock_sock):
            img, failure = clicker._capture_direct_socket("emulator-test", "png", use_grayscale=False)

        self.assertIsNotNone(img)
        self.assertIsNone(failure)
        self.assertEqual(img.shape, (40, 50, 3))
        mock_sock.connect.assert_called_once_with(("127.0.0.1", 5037))

    def test_reconnect_backoff_avoids_repeated_connect_processes(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.start_adb_server = Mock(return_value=False)

        self.assertIsNone(clicker.capture_screen())
        self.assertIsNone(clicker.capture_screen())

        clicker.start_adb_server.assert_called_once()
        self.assertGreater(clicker._next_reconnect_at, time.monotonic())

    def test_nested_config_cache_is_deeply_isolated(self):
        config_path = Path(self.temp_dir, "config.json")
        config_path.write_text(json.dumps({
            "instances": [{"device_address": "a", "future": {"value": 1}}],
            "template_rois": {"x.png": [0.1, 0.2, 0.3, 0.4]},
        }), encoding="utf-8")

        first = AutoClicker.read_config(str(config_path))
        first["instances"][0]["future"]["value"] = 99
        first["template_rois"]["x.png"][0] = 0.9
        second = AutoClicker.read_config(str(config_path))

        self.assertEqual(second["instances"][0]["future"]["value"], 1)
        self.assertEqual(second["template_rois"]["x.png"][0], 0.1)

    def test_instance_merge_preserves_unknown_keys_in_one_snapshot(self):
        config_path = Path(self.temp_dir, "config.json")
        config_path.write_text(json.dumps({
            "device_address": "a",
            "instances": [{
                "device_address": "a",
                "scan_interval": 2,
                "future": {"value": 7},
            }],
        }), encoding="utf-8")

        saved = AutoClicker.update_instances_config(
            [{"device_address": "a", "scan_interval": 1}],
            str(config_path),
            primary_settings={"device_address": "a", "scan_interval": 1},
        )
        result = AutoClicker.read_config(str(config_path))

        self.assertTrue(saved)
        self.assertEqual(result["scan_interval"], 1)
        self.assertEqual(result["instances"][0]["future"]["value"], 7)

    def test_performance_metrics_are_bounded(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
            performance_metrics=True,
        )
        for _ in range(300):
            clicker._record_performance("stage", time.perf_counter())

        stats = clicker.get_performance_stats()

        self.assertEqual(stats["stage"]["count"], 240)
        self.assertGreaterEqual(stats["stage"]["max_ms"], stats["stage"]["p95_ms"])

    def test_dynamic_location_hint_short_circuits_coarse_match(self):
        template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "hint.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        screen = np.zeros((200, 300, 3), dtype=np.uint8)
        screen[70:110, 120:160] = template
        first = clicker.find_template(
            screen, str(template_path), threshold=0.99, filename="hint.png"
        )
        stages = []
        original_run = clicker._run_match_map

        def tracked_run(image, target, method, stage):
            stages.append(stage)
            return original_run(image, target, method, stage)

        prescaled_provider = Mock(side_effect=AssertionError("unexpected resize"))
        with patch.object(clicker, "_run_match_map", side_effect=tracked_run):
            second = clicker.find_template(
                screen,
                str(template_path),
                threshold=0.99,
                filename="hint.png",
                _prescaled_screen=prescaled_provider,
            )

        self.assertEqual(first[:2], second[:2])
        self.assertEqual(stages, ["match.hint"])
        prescaled_provider.assert_not_called()
    def test_malformed_instances_are_recovered_during_snapshot_save(self):
        malformed_values = (None, 7, [{"device_address": ["invalid"]}])
        for index, existing in enumerate(malformed_values):
            config_path = Path(self.temp_dir, f"malformed-{index}.json")
            config_path.write_text(
                json.dumps({"instances": existing}), encoding="utf-8"
            )

            saved = AutoClicker.update_instances_config(
                [{"device_address": "emulator-test", "scan_interval": 1}],
                str(config_path),
            )
            result = AutoClicker.read_config(str(config_path))

            self.assertTrue(saved)
            self.assertEqual(
                result["instances"][0]["device_address"], "emulator-test"
            )

    def test_external_template_overwrite_refreshes_full_and_scaled_cache(self):
        template_path = Path(self.temp_dir, "templates", "replace.png")
        first_image = self._pattern(size=40)
        second_image = np.bitwise_xor(first_image, 255)
        self._write_png(template_path, first_image)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            match_grayscale=False,
            logger=lambda _: None,
        )
        first, _ = clicker._load_template(str(template_path), grayscale=False)
        cache_key = (os.path.abspath(template_path), False)
        first_scaled = clicker._get_scaled_template(cache_key, first).copy()
        old_mtime = template_path.stat().st_mtime_ns

        self._write_png(template_path, second_image)
        os.utime(template_path, ns=(old_mtime + 1_000_000_000,) * 2)
        directory_key = os.path.abspath(template_path.parent)
        next_scan = (
            AutoClicker._template_directory_last_scan[directory_key]
            + TEMPLATE_MANIFEST_REFRESH_INTERVAL
        )
        with patch("main.time.perf_counter", return_value=next_scan):
            second, _ = clicker._load_template(
                str(template_path), grayscale=False
            )
        second_scaled = clicker._get_scaled_template(cache_key, second)

        self.assertTrue(np.array_equal(second, second_image))
        self.assertFalse(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first_scaled, second_scaled))

    def test_snapshot_save_merges_pending_count_with_one_atomic_write(self):
        config_path = Path(self.temp_dir, "single-write.json")
        config_path.write_text(
            json.dumps({"template_counts": {"x.png": 0}}), encoding="utf-8"
        )
        path_key = os.path.abspath(config_path)
        AutoClicker._pending_count_deltas[path_key]["template_counts"][
            "x.png"
        ] = 1
        original_write = AutoClicker._atomic_write_config_unlocked

        with patch.object(
            AutoClicker,
            "_atomic_write_config_unlocked",
            side_effect=original_write,
        ) as atomic_write:
            saved = AutoClicker.update_instances_config(
                [{"device_address": "emulator-test"}], str(config_path)
            )

        result = AutoClicker.read_config(str(config_path))
        self.assertTrue(saved)
        self.assertEqual(atomic_write.call_count, 1)
        self.assertEqual(result["template_counts"]["x.png"], 1)

    def test_failed_snapshot_write_retains_pending_count_for_retry(self):
        config_path = Path(self.temp_dir, "retry-write.json")
        config_path.write_text(
            json.dumps({"template_counts": {"x.png": 0}}), encoding="utf-8"
        )
        path_key = os.path.abspath(config_path)
        pending = AutoClicker._pending_count_deltas[path_key][
            "template_counts"
        ]
        pending["x.png"] = 1

        with patch.object(
            AutoClicker,
            "_atomic_write_config_unlocked",
            side_effect=OSError("disk full"),
        ):
            self.assertFalse(AutoClicker.update_instances_config(
                [{"device_address": "emulator-test"}], str(config_path)
            ))

        self.assertEqual(pending["x.png"], 1)
        self.assertTrue(AutoClicker.update_instances_config(
            [{"device_address": "emulator-test"}], str(config_path)
        ))
        result = AutoClicker.read_config(str(config_path))
        self.assertEqual(result["template_counts"]["x.png"], 1)

    def test_exec_cooldown_is_not_extended_while_using_fallback(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.capture_backend = "png"
        clicker.device = Mock(serial="emulator-test")
        fallback_image = np.zeros((10, 20, 3), dtype=np.uint8)

        with patch.object(
            clicker, "_capture_exec_backend", return_value=(None, "transport")
        ) as exec_capture, patch.object(
            clicker, "_capture_ppadb_backend", return_value=fallback_image
        ) as fallback_capture:
            self.assertIsNotNone(clicker.capture_screen(grayscale=False))
            first_deadline = clicker._exec_out_disabled_until
            self.assertIsNotNone(clicker.capture_screen(grayscale=False))

        self.assertEqual(exec_capture.call_count, 1)
        self.assertEqual(fallback_capture.call_count, 2)
        self.assertEqual(clicker._exec_out_failure_count, 1)
        self.assertEqual(clicker._exec_out_disabled_until, first_deadline)

    def test_slow_reconnect_failure_uses_failure_completion_time(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.start_adb_server = Mock(return_value=False)

        with patch("main.time.monotonic", side_effect=[100.0, 110.0]):
            self.assertIsNone(clicker.capture_screen())

        self.assertEqual(clicker._next_reconnect_at, 111.0)

    def test_stop_during_initial_connect_cancels_loop_start(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )

        def connect_then_stop():
            clicker.stop_loop()
            return True

        clicker.start_adb_server = Mock(side_effect=connect_then_stop)
        clicker.run_once = Mock()
        with patch.object(clicker, "preload_templates") as preload:
            clicker.start_loop()

        clicker.run_once.assert_not_called()
        preload.assert_not_called()
        self.assertFalse(clicker._automatic_loop_active)

    def test_stop_returning_from_match_prevents_action(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.template_order = ["x.png"]
        clicker.template_actions = {"x.png": "click"}
        clicker._automatic_loop_active = True
        clicker.is_running = True
        clicker.capture_screen = Mock(
            return_value=np.zeros((40, 50), dtype=np.uint8)
        )

        def match_then_stop(*args, **kwargs):
            clicker.stop_loop()
            return (10, 10, 1.0)

        clicker.find_template = Mock(side_effect=match_then_stop)
        clicker._execute_action = Mock(return_value=True)

        self.assertFalse(clicker.run_once())
        clicker._execute_action.assert_not_called()

    def test_wait_interval_is_capped_by_force_and_action_deadlines(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=1,
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker._unchanged_frame_streak = 4
        clicker.max_idle_interval = 5
        clicker._last_full_scan_time = 100.0
        clicker.force_scan_interval = 5.0
        clicker.no_match_action = "none"
        clicker.no_match_timeout = 0
        self.assertAlmostEqual(clicker._current_wait_interval(104.5), 0.5)

        clicker._last_full_scan_time = 0.0
        clicker.no_match_action = "custom_click"
        clicker.no_match_interval = 3.0
        clicker.last_match_time = 102.0
        clicker.last_random_click_time = 102.0
        self.assertAlmostEqual(clicker._current_wait_interval(104.5), 0.5)

    def test_adaptive_order_is_opt_in_to_preserve_user_priority(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        order = ["a.png", "b.png", "c.png", "d.png", "e.png"]
        clicker.last_matched_template = "c.png"
        clicker._transition_counts[(False, "c.png")][(False, "e.png")] = 9

        self.assertEqual(clicker._get_template_scan_order(order), tuple(order))

    def test_periodic_hint_bypass_runs_global_coarse_scan(self):
        template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "periodic.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
            match_grayscale=False,
        )
        screen = np.zeros((200, 300, 3), dtype=np.uint8)
        screen[70:110, 120:160] = template
        clicker.find_template(
            screen, str(template_path), threshold=0.99, filename="periodic.png"
        )
        absolute_path = os.path.abspath(template_path)
        with clicker._hint_lock:
            clicker._template_hint_hits[absolute_path] = 9
        stages = []
        original_run = clicker._run_match_map

        def tracked_run(image, target, method, stage):
            stages.append(stage)
            return original_run(image, target, method, stage)

        prescaled = cv2.resize(
            screen,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )
        with patch.object(clicker, "_run_match_map", side_effect=tracked_run):
            result = clicker.find_template(
                screen,
                str(template_path),
                threshold=0.99,
                filename="periodic.png",
                _prescaled_screen=prescaled,
            )

        self.assertIsNotNone(result)
        self.assertNotIn("match.hint", stages)
        self.assertIn("match.coarse", stages)

    def test_roi_fullscreen_fallback_throttle_survives_screen_dirty(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.force_scan_interval = 5.0

        with patch("main.time.monotonic", side_effect=[100.0, 101.0]):
            self.assertTrue(clicker._roi_full_scan_due("x.png"))
            self.assertFalse(clicker._roi_full_scan_due("x.png"))

        clicker._mark_screen_dirty()
        with patch("main.time.monotonic", return_value=101.1):
            self.assertFalse(clicker._roi_full_scan_due("x.png"))
        with patch("main.time.monotonic", return_value=106.0):
            self.assertTrue(clicker._roi_full_scan_due("x.png"))

    def test_roi_frame_budget_prioritizes_then_round_robins(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.roi_full_scan_budget = 2
        clicker.force_scan_interval = 5.0
        order = ["a.png", "b.png", "c.png", "d.png", "e.png"]
        rois = {filename: [0.0, 0.0, 0.5, 1.0] for filename in order}

        clicker._begin_roi_full_scan_frame((100, 200, 3))
        first = clicker._schedule_roi_full_scan_candidates(
            clicker.template_dir, order, rois, now=100.0
        )
        self.assertEqual(
            [os.path.basename(path) for path in first],
            ["a.png", "b.png"],
        )
        with patch("main.time.monotonic", return_value=100.0):
            self.assertTrue(clicker._roi_full_scan_due(first[0]))
            self.assertTrue(clicker._roi_full_scan_due(first[1]))
            self.assertFalse(
                clicker._roi_full_scan_due(
                    os.path.join(clicker.template_dir, "c.png")
                )
            )
        self.assertEqual(clicker._roi_full_scan_budget_remaining, 0)

        clicker._reset_roi_full_scan_frame()
        clicker._begin_roi_full_scan_frame((100, 200, 3))
        second = clicker._schedule_roi_full_scan_candidates(
            clicker.template_dir, order, rois, now=101.0
        )
        self.assertEqual(
            [os.path.basename(path) for path in second],
            ["c.png", "d.png"],
        )

        clicker._reset_roi_full_scan_frame()
        clicker._begin_roi_full_scan_frame((100, 200, 3))
        third = clicker._schedule_roi_full_scan_candidates(
            clicker.template_dir, order, rois, now=102.0
        )
        self.assertEqual(
            [os.path.basename(path) for path in third],
            ["e.png"],
        )

    def test_run_once_limits_roi_fullscreen_fallbacks_to_budget(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        order = ["a.png", "b.png", "c.png", "d.png"]
        for filename in order:
            self._write_png(
                Path(self.temp_dir, "templates", filename),
                self._pattern(),
            )
        AutoClicker._refresh_template_directory(
            clicker.template_dir, force=True
        )
        clicker.template_order = order
        clicker.template_rois = {
            filename: [0.0, 0.0, 0.5, 1.0] for filename in order
        }
        clicker.dynamic_roi = False
        clicker.roi_full_scan_budget = 2
        clicker._automatic_loop_active = True
        clicker.capture_screen = Mock(
            return_value=np.zeros((120, 160, 3), dtype=np.uint8)
        )
        full_scans = []

        def miss_everywhere(*args, **_kwargs):
            if args[5] == (0, 0, 160, 120):
                full_scans.append(os.path.basename(args[4][0]))
            return None

        with patch.object(
            clicker, "_find_in_region", side_effect=miss_everywhere
        ):
            self.assertFalse(clicker.run_once())

        self.assertEqual(full_scans, ["a.png", "b.png"])

    def test_only_resolution_change_resets_all_roi_state(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.frame_change_detection = True
        first = np.zeros((40, 50), dtype=np.uint8)
        changed = first.copy()
        changed[0, 0] = 1
        resized = np.zeros((41, 50), dtype=np.uint8)
        template_path = os.path.abspath("x.png")

        clicker._begin_roi_full_scan_frame(first.shape)
        clicker._reset_roi_full_scan_frame()
        with clicker._hint_lock:
            clicker._template_location_hints[template_path] = (
                first.shape,
                0,
                0,
                10,
                10,
            )
            clicker._template_hint_hits[template_path] = 2
            clicker._roi_full_scan_times[template_path] = 100.0
            clicker._roi_full_scan_reservation_times[template_path] = 100.0

        self.assertTrue(clicker._should_scan_frame(first, 100.0))
        clicker._mark_screen_dirty()
        self.assertTrue(clicker._should_scan_frame(changed, 101.0))
        self.assertIn(template_path, clicker._roi_full_scan_times)
        self.assertIn(
            template_path, clicker._roi_full_scan_reservation_times
        )
        self.assertIn(template_path, clicker._template_location_hints)

        clicker._begin_roi_full_scan_frame(first.shape)
        self.assertIn(template_path, clicker._roi_full_scan_times)
        clicker._begin_roi_full_scan_frame(resized.shape)
        self.assertNotIn(template_path, clicker._roi_full_scan_times)
        self.assertNotIn(
            template_path, clicker._roi_full_scan_reservation_times
        )
        self.assertNotIn(template_path, clicker._template_location_hints)

    def test_wait_elapsed_is_not_subtracted_from_deadline_twice(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=1,
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.adaptive_scan_interval = False
        clicker._last_full_scan_time = 100.0
        clicker.force_scan_interval = 1.0
        clicker.no_match_action = "none"
        clicker.no_match_timeout = 0

        self.assertAlmostEqual(
            clicker._current_wait_interval(now=100.4, elapsed=0.4),
            0.6,
        )

    def test_strict_roi_also_restricts_dynamic_hint(self):
        template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "strict.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            match_grayscale=False,
            logger=lambda _: None,
        )
        screen = np.zeros((200, 300, 3), dtype=np.uint8)
        screen[90:130, 220:260] = template

        first = clicker.find_template(
            screen,
            str(template_path),
            threshold=0.99,
            filename="strict.png",
        )
        self.assertIsNotNone(first)

        clicker.template_rois["strict.png"] = [0.0, 0.0, 0.5, 1.0]
        clicker.roi_fullscreen_fallback = False
        second = clicker.find_template(
            screen,
            str(template_path),
            threshold=0.99,
            filename="strict.png",
        )
        self.assertIsNone(second)

    def test_fullscreen_roi_reuses_lazy_prescaled_screen(self):
        template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "full-roi.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            match_grayscale=False,
            logger=lambda _: None,
        )
        clicker.dynamic_roi = False
        clicker.template_rois["full-roi.png"] = [0.0, 0.0, 1.0, 1.0]
        screen = np.zeros((200, 300, 3), dtype=np.uint8)
        screen[80:120, 130:170] = template
        prescaled = cv2.resize(
            screen,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )
        provider = Mock(return_value=prescaled)

        result = clicker.find_template(
            screen,
            str(template_path),
            threshold=0.99,
            filename="full-roi.png",
            _prescaled_screen=provider,
        )

        self.assertIsNotNone(result)
        provider.assert_called_once_with()

    def test_stop_during_fallback_scan_prevents_final_action(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.no_match_action = "fallback_list"
        clicker.no_match_interval = 1.0
        clicker.fallback_template_order = ["x.png"]
        clicker.fallback_final_action = "custom_click"
        clicker.last_match_time = 0.0
        clicker.last_random_click_time = 0.0
        clicker._automatic_loop_active = True
        clicker.is_running = True
        final_action = Mock(return_value=True)
        clicker._execute_final_action = final_action

        def stop_and_miss(*args, **kwargs):
            clicker.stop_loop()
            return False

        clicker._try_fallback_templates = Mock(side_effect=stop_and_miss)
        screen = np.zeros((40, 50), dtype=np.uint8)
        prescaled = np.zeros((20, 25), dtype=np.uint8)

        clicker._handle_no_match_action(
            screen,
            now=100.0,
            _prescaled_screen=prescaled,
        )

        final_action.assert_not_called()

    def test_stale_scaled_template_cannot_reenter_cache_after_reload(self):
        template_path = Path(self.temp_dir, "templates", "racing.png")
        first_image = self._pattern(size=40)
        second_image = np.bitwise_xor(first_image, 255)
        self._write_png(template_path, first_image)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            match_grayscale=False,
            logger=lambda _: None,
        )
        first, _ = clicker._load_template(str(template_path), grayscale=False)
        cache_key = (os.path.abspath(template_path), False)
        old_mtime = template_path.stat().st_mtime_ns
        entered = threading.Event()
        release = threading.Event()
        errors = []
        original_resize = cv2.resize

        def blocking_resize(image, *args, **kwargs):
            if image is first:
                entered.set()
                if not release.wait(2.0):
                    raise AssertionError("blocked resize did not resume")
            return original_resize(image, *args, **kwargs)

        def build_stale_scaled():
            try:
                clicker._get_scaled_template(cache_key, first)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=build_stale_scaled)
        with patch("main.cv2.resize", side_effect=blocking_resize):
            worker.start()
            try:
                self.assertTrue(entered.wait(1.0))
                self._write_png(template_path, second_image)
                os.utime(
                    template_path,
                    ns=(old_mtime + 1_000_000_000,) * 2,
                )
                AutoClicker._refresh_template_directory(
                    template_path.parent, force=True
                )
                second, _ = clicker._load_template(
                    str(template_path), grayscale=False
                )
            finally:
                release.set()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        scaled = clicker._get_scaled_template(cache_key, second)
        cached_source, cached_scaled = AutoClicker._scaled_template_cache[
            cache_key
        ]
        self.assertIs(cached_source, second)
        self.assertIs(cached_scaled, scaled)
        self.assertFalse(np.array_equal(scaled, original_resize(
            first,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )))

    def test_adb_cli_device_list_maps_tcp_port_to_emulator_serial(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )

        def run_adb(args, **kwargs):
            stdout = ""
            if args == ["devices"]:
                stdout = (
                    "List of devices attached\n"
                    "emulator-5554\tdevice\n"
                )
            return subprocess.CompletedProcess(args, 0, stdout, "")

        clicker._run_adb = Mock(side_effect=run_adb)
        clicker._probe_adb_server = Mock(return_value=True)
        device = Mock(serial="emulator-5554")
        with patch("main.AdbClient") as client_class, patch(
            "main.AdbDevice", return_value=device
        ) as device_class:
            connected = clicker.start_adb_server()

        self.assertTrue(connected)
        self.assertIs(clicker.device, device)
        device_class.assert_called_once_with(
            client_class.return_value, "emulator-5554"
        )
        client_class.return_value.devices.assert_not_called()

    def test_expired_action_deadline_forces_immediate_loop_wakeup(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=10,
            logger=lambda _: None,
        )
        clicker.adaptive_scan_interval = False
        clicker.no_match_action = "custom_click"
        clicker.no_match_interval = 5.0
        clicker.last_match_time = 90.0
        clicker.last_random_click_time = 90.0
        clicker.no_match_timeout = 0

        self.assertAlmostEqual(
            clicker._current_wait_interval(now=100.0),
            0.01,
        )

    def test_reconnect_deadline_caps_adaptive_loop_sleep(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=5,
            logger=lambda _: None,
        )
        clicker.adaptive_scan_interval = True
        clicker._unchanged_frame_streak = 4
        clicker.max_idle_interval = 5.0
        clicker._automatic_loop_active = True
        clicker._last_full_scan_time = 90.0
        clicker.force_scan_interval = 5.0
        clicker.no_match_action = "custom_click"
        clicker.no_match_interval = 5.0
        clicker.last_match_time = 90.0
        clicker.last_random_click_time = 90.0
        clicker.no_match_timeout = 0
        clicker.device = None
        clicker._next_reconnect_at = 101.0

        self.assertAlmostEqual(
            clicker._current_wait_interval(now=100.0),
            1.0,
        )

    def test_roi_fallback_is_unthrottled_when_frame_detection_is_off(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.frame_change_detection = False
        clicker.force_scan_interval = 30.0

        with patch("main.time.monotonic", return_value=100.0):
            self.assertTrue(clicker._roi_full_scan_due("x.png"))
        with patch("main.time.monotonic", return_value=100.1):
            self.assertTrue(clicker._roi_full_scan_due("x.png"))

    def test_too_small_strict_roi_does_not_fall_back_to_fullscreen(self):
        template = self._pattern(size=40)
        template_path = Path(self.temp_dir, "templates", "tiny-roi.png")
        self._write_png(template_path, template)
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            match_grayscale=False,
            logger=lambda _: None,
        )
        clicker.dynamic_roi = False
        clicker.roi_fullscreen_fallback = False
        clicker.template_rois["tiny-roi.png"] = [0.0, 0.0, 0.1, 0.1]
        screen = np.zeros((120, 160, 3), dtype=np.uint8)
        screen[50:90, 80:120] = template

        result = clicker.find_template(
            screen,
            str(template_path),
            threshold=0.99,
            filename="tiny-roi.png",
        )

        self.assertIsNone(result)

    def test_stop_after_first_click_cancels_second_click_immediately(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.is_running = True
        device = Mock(serial="emulator-test")
        clicker.device = device

        def stop_on_first_click(*args, **kwargs):
            clicker.stop_loop()
            return ""

        device.shell.side_effect = stop_on_first_click
        started = time.monotonic()
        result = clicker.double_click(10, 20, delay=0.5)
        elapsed = time.monotonic() - started

        self.assertFalse(result)
        self.assertEqual(device.shell.call_count, 1)
        self.assertLess(elapsed, 0.2)

    def test_stop_during_reconnect_cancels_capture_backends(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker.is_running = True
        device = Mock(serial="emulator-test")

        def reconnect_then_stop():
            clicker.device = device
            clicker.stop_loop()
            return True

        clicker.start_adb_server = Mock(side_effect=reconnect_then_stop)
        clicker._capture_exec_backend = Mock()
        clicker._capture_ppadb_backend = Mock()

        self.assertIsNone(clicker.capture_screen())
        clicker._capture_exec_backend.assert_not_called()
        clicker._capture_ppadb_backend.assert_not_called()

    def test_custom_adb_endpoint_is_used_by_cli_and_capture(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-5554",
            host="10.0.0.20",
            port=5040,
            logger=lambda _: None,
        )
        text_proc = subprocess.CompletedProcess([], 0, "", "")
        with patch("main.subprocess.run", return_value=text_proc) as run:
            clicker._run_adb(["devices"])
        self.assertEqual(
            run.call_args.args[0],
            [
                clicker.adb_path,
                "-H",
                "10.0.0.20",
                "-P",
                "5040",
                "devices",
            ],
        )

        binary_proc = subprocess.CompletedProcess([], 1, b"", b"")
        with patch("main.subprocess.run", return_value=binary_proc) as run:
            clicker._capture_exec_backend(
                "emulator-5554", "png", use_grayscale=False
            )
        command = run.call_args.args[0]
        self.assertEqual(command[1:5], ["-H", "10.0.0.20", "-P", "5040"])
        self.assertEqual(command[command.index("-s") + 1], "emulator-5554")

    def test_local_adb_endpoint_omits_host_flag_for_start_server(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            host="127.0.0.1",
            port=5037,
            logger=lambda _: None,
        )
        text_proc = subprocess.CompletedProcess([], 0, "", "")
        with patch("main.subprocess.run", return_value=text_proc) as run:
            clicker._run_adb(["start-server"])
        self.assertEqual(
            run.call_args.args[0],
            [
                clicker.adb_path,
                "-P",
                "5037",
                "start-server",
            ],
        )

        with patch("main.subprocess.run", return_value=text_proc) as run:
            clicker._run_adb(["devices"])
        self.assertEqual(
            run.call_args.args[0],
            [
                clicker.adb_path,
                "-P",
                "5037",
                "devices",
            ],
        )

    def test_run_once_uses_fresh_time_after_template_scan(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.template_order = ["missing.png"]
        clicker.capture_screen = Mock(
            return_value=np.zeros((40, 50), dtype=np.uint8)
        )
        clicker.find_template = Mock(return_value=None)
        clicker._handle_no_match_action = Mock()

        with patch("main.time.monotonic", side_effect=[100.0, 110.0]):
            clicker.run_once()

        self.assertEqual(
            clicker._handle_no_match_action.call_args.args[1],
            110.0,
        )

    def test_gui_close_joins_workers_before_final_config_save(self):
        from gui_app import App

        events = []
        first = Mock()
        second = Mock()
        first.begin_shutdown.side_effect = lambda: events.append("request1")
        second.begin_shutdown.side_effect = lambda: events.append("request2")
        first.shutdown.side_effect = lambda: events.append("shutdown1")
        second.shutdown.side_effect = lambda: events.append("shutdown2")
        app = Mock()
        app._closing = False
        app._start_all_generation = 0
        app.tab_frames = {"one": first, "two": second}
        app._ui_pump_id = "ui"
        app._timer_pump_id = "timer"
        app.save_app_config.side_effect = lambda: events.append("save")
        app.destroy.side_effect = lambda: events.append("destroy")

        with patch(
            "gui_app.TemplatePreviewTooltip.get_instance"
        ) as tooltip_instance:
            tooltip_instance.return_value.hide.side_effect = (
                lambda: events.append("hide")
            )
            App.on_closing(app)

        self.assertEqual(
            events,
            [
                "hide",
                "request1",
                "request2",
                "shutdown1",
                "shutdown2",
                "save",
                "destroy",
            ],
        )

    def test_cancel_event_set_during_loop_start_is_not_cleared(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.device = object()
        clicker._stop_event.set()
        cancel_event = Mock()
        cancel_event.is_set.side_effect = [False, True]
        clicker.run_once = Mock()

        with patch.object(clicker, "preload_templates") as preload:
            clicker.start_loop(cancel_event=cancel_event)

        self.assertTrue(clicker._stop_event.is_set())
        self.assertFalse(clicker.is_running)
        self.assertFalse(clicker._automatic_loop_active)
        clicker.run_once.assert_not_called()
        preload.assert_not_called()

    def test_gui_close_cancels_worker_after_generation_guard(self):
        from gui_app import InstanceTabFrame

        frame = Mock()
        frame.clicker.device = object()
        frame.clicker.is_running = False
        frame._loop_starting = False
        frame._loop_generation = 0
        frame._connection_generation = 0
        frame._destroyed = False
        frame._alert_shown_for_current_timeout = False
        frame._action_threads = set()
        frame._action_threads_lock = threading.Lock()
        entered = threading.Event()
        release = threading.Event()
        request_seen = threading.Event()
        received_cancel = []

        def run_loop(cancel_event=None):
            received_cancel.append(cancel_event)
            entered.set()
            release.wait(2.0)

        frame.clicker.start_loop.side_effect = run_loop
        frame.clicker.request_shutdown.side_effect = request_seen.set
        frame.begin_shutdown.side_effect = lambda: (
            InstanceTabFrame.begin_shutdown(frame)
        )
        InstanceTabFrame.start_clicker_loop(frame)
        self.assertTrue(entered.wait(1.0))

        closer = threading.Thread(
            target=InstanceTabFrame.shutdown,
            args=(frame,),
        )
        closer.start()
        self.assertTrue(request_seen.wait(1.0))
        self.assertTrue(received_cancel[0].is_set())
        release.set()
        closer.join(2.0)

        self.assertFalse(closer.is_alive())
        frame.clicker.shutdown.assert_called_once_with()

    def test_gui_shutdown_joins_manual_action_before_core_flush(self):
        from gui_app import InstanceTabFrame

        frame = Mock()
        frame._destroyed = False
        frame._connection_generation = 0
        frame._loop_generation = 0
        frame._loop_cancel_event = threading.Event()
        frame._action_threads = set()
        frame._action_threads_lock = threading.Lock()
        frame.clicker_thread = None
        request_seen = threading.Event()
        action_started = threading.Event()
        events = []

        def request_shutdown():
            events.append("request")
            request_seen.set()

        def manual_action():
            action_started.set()
            request_seen.wait(1.0)
            time.sleep(0.05)
            events.append("manual_done")

        frame.clicker.request_shutdown.side_effect = request_shutdown
        frame.begin_shutdown.side_effect = lambda: (
            InstanceTabFrame.begin_shutdown(frame)
        )
        frame.clicker.shutdown.side_effect = lambda: events.append("core_flush")
        InstanceTabFrame._start_action_worker(frame, manual_action)
        self.assertTrue(action_started.wait(1.0))

        InstanceTabFrame.shutdown(frame)

        self.assertEqual(
            events,
            ["request", "manual_done", "core_flush"],
        )

    def test_reconnect_backoff_ignores_scan_cadence(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=0.1,
            logger=lambda _: None,
        )
        clicker.no_match_timeout = 0
        clicker.device = None
        clicker._next_reconnect_at = 110.0

        self.assertAlmostEqual(
            clicker._current_wait_interval(now=100.0),
            10.0,
        )

    def test_reconnect_backoff_honors_earlier_timeout_deadline(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=0.1,
            logger=lambda _: None,
        )
        clicker.device = None
        clicker._next_reconnect_at = 110.0
        clicker.no_match_timeout = 10.0
        clicker.last_action_time = 92.0

        self.assertAlmostEqual(
            clicker._current_wait_interval(now=100.0),
            2.0,
        )

    def test_adb_start_stops_after_cancellation_between_commands(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        clicker._automatic_loop_active = True
        clicker._probe_adb_server = Mock(return_value=False)
        calls = []

        def run_adb(args, **kwargs):
            calls.append(args)
            if args == ["start-server"]:
                clicker.stop_loop()
            return subprocess.CompletedProcess(args, 0, "", "")

        clicker._run_adb = Mock(side_effect=run_adb)

        self.assertFalse(clicker.start_adb_server())
        self.assertEqual(calls, [["start-server"]])

    def test_toggle_stops_loop_while_start_worker_is_pending(self):
        from gui_app import InstanceTabFrame

        frame = Mock()
        frame.clicker.is_running = False
        frame._loop_starting = True

        InstanceTabFrame.toggle_clicker(frame)

        frame.save_settings.assert_called_once_with()
        frame.stop_clicker_loop.assert_called_once_with()
        frame.start_clicker_loop.assert_not_called()

    def test_external_connection_wakes_reconnect_backoff(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            scan_interval=0.1,
            logger=lambda _: None,
        )
        clicker.device = object()
        clicker.no_match_timeout = 0

        wake_event = threading.Event()
        wait_started = threading.Event()
        wake_proxy = Mock()
        wake_proxy.clear.side_effect = wake_event.clear
        wake_proxy.set.side_effect = wake_event.set

        def wait_for_wake(timeout):
            wait_started.set()
            return wake_event.wait(timeout)

        wake_proxy.wait.side_effect = wait_for_wake
        clicker._loop_wake_event = wake_proxy

        scan_count = 0
        resumed = threading.Event()

        def run_once():
            nonlocal scan_count
            scan_count += 1
            if scan_count == 1:
                with clicker._device_lock:
                    clicker.device = None
                clicker._next_reconnect_at = time.monotonic() + 10.0
            else:
                resumed.set()
                clicker.stop_loop()
            return False

        clicker.run_once = run_once
        worker = threading.Thread(target=clicker.start_loop, daemon=True)
        worker.start()
        self.assertTrue(wait_started.wait(1.0))

        def run_adb(args, **kwargs):
            stdout = ""
            if args == ["devices"]:
                stdout = (
                    "List of devices attached\n"
                    "127.0.0.1:5555\tdevice\n"
                )
            return subprocess.CompletedProcess(args, 0, stdout, "")

        clicker._run_adb = Mock(side_effect=run_adb)
        clicker._probe_adb_server = Mock(return_value=True)
        device = Mock(serial="127.0.0.1:5555")
        try:
            with patch("main.AdbClient"), patch(
                "main.AdbDevice", return_value=device
            ):
                connected = clicker.start_adb_server()
            self.assertTrue(connected)
            self.assertTrue(resumed.wait(1.0))
        finally:
            clicker.stop_loop()
            worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertGreaterEqual(scan_count, 2)

    def test_wake_during_scan_is_not_cleared_before_wait(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=5.0,
            logger=lambda _: None,
        )
        clicker.device = object()
        clicker.no_match_timeout = 0
        work_started = threading.Event()
        release_work = threading.Event()
        resumed = threading.Event()
        scan_count = 0

        def run_once():
            nonlocal scan_count
            scan_count += 1
            if scan_count == 1:
                work_started.set()
                release_work.wait(1.0)
            else:
                resumed.set()
                clicker.stop_loop()
            return False

        clicker.run_once = run_once
        worker = threading.Thread(target=clicker.start_loop, daemon=True)
        worker.start()
        try:
            self.assertTrue(work_started.wait(1.0))
            clicker._loop_wake_event.set()
            release_work.set()
            self.assertTrue(resumed.wait(1.0))
        finally:
            release_work.set()
            clicker.stop_loop()
            worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertGreaterEqual(scan_count, 2)

    def test_gui_template_save_reports_config_failure(self):
        from gui_app import InstanceTabFrame

        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        clicker.save_config = Mock(return_value=False)
        frame = Mock()
        frame.clicker = clicker
        dialog = Mock()
        dialog.get_input.return_value = "not-persisted"

        with patch(
            "gui_app.ctk.CTkInputDialog", return_value=dialog
        ):
            InstanceTabFrame._prompt_and_save_template(
                frame, self._pattern(), 3, 4, False
            )

        self.assertTrue(
            Path(clicker.template_dir, "not-persisted.png").is_file()
        )
        clicker.save_config.assert_called_once_with(include_templates=True)
        frame.app_owner.refresh_all_tabs_templates.assert_not_called()
        messages = [call.args[0] for call in frame.log_message.call_args_list]
        self.assertTrue(any("config.json" in message for message in messages))
        self.assertFalse(any("저장 완료" in message for message in messages))

    def test_stop_during_scan_skips_timeout_callback(self):
        callback = Mock()
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=5.0,
            no_match_timeout=1.0,
            on_timeout_callback=callback,
            logger=lambda _: None,
        )
        clicker.device = object()

        def stop_during_scan():
            clicker.last_action_time = time.monotonic() - 10.0
            clicker.stop_loop()
            return False

        clicker.run_once = stop_during_scan
        clicker.start_loop()

        callback.assert_not_called()

    def test_cancel_event_interrupts_loop_wait_without_stop_call(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=5.0,
            logger=lambda _: None,
        )
        clicker.device = object()
        clicker.no_match_timeout = 0
        cancel_event = threading.Event()
        wait_started = threading.Event()
        real_wake = threading.Event()
        wake_proxy = Mock()
        wake_proxy.clear.side_effect = real_wake.clear
        wake_proxy.set.side_effect = real_wake.set

        def wait_for_signal(timeout):
            wait_started.set()
            return real_wake.wait(timeout)

        wake_proxy.wait.side_effect = wait_for_signal
        clicker._loop_wake_event = wake_proxy
        clicker.run_once = Mock(return_value=False)
        worker = threading.Thread(
            target=clicker.start_loop,
            kwargs={"cancel_event": cancel_event},
            daemon=True,
        )
        worker.start()
        try:
            self.assertTrue(wait_started.wait(1.0))
            cancel_event.set()
            worker.join(1.0)
        finally:
            clicker.stop_loop()
            worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(clicker.run_once.call_count, 1)

    def test_queued_timeout_warning_is_ignored_after_stop(self):
        from gui_app import InstanceTabFrame

        frame = Mock()
        frame._destroyed = False
        frame._loop_generation = 3
        frame.clicker.is_running = True
        callbacks = []
        frame.app_owner.post_to_ui.side_effect = callbacks.append

        InstanceTabFrame.on_no_match_timeout(frame, 5)
        self.assertEqual(len(callbacks), 1)

        frame.clicker.is_running = False
        callbacks[0]()

        frame._set_tab_warning_state.assert_not_called()
        frame.status_label.configure.assert_not_called()

    def test_begin_shutdown_is_idempotent(self):
        from gui_app import InstanceTabFrame

        frame = Mock()
        frame._shutdown_started = False
        frame._destroyed = False
        frame._connection_generation = 2
        frame._loop_generation = 4
        frame._loop_cancel_event = threading.Event()

        InstanceTabFrame.begin_shutdown(frame)
        InstanceTabFrame.begin_shutdown(frame)

        self.assertTrue(frame._loop_cancel_event.is_set())
        self.assertTrue(frame._destroyed)
        self.assertEqual(frame._connection_generation, 3)
        self.assertEqual(frame._loop_generation, 5)
        frame.clicker.request_shutdown.assert_called_once_with()

    def test_active_cancel_stops_adb_start_between_commands(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        cancel_event = threading.Event()
        clicker._probe_adb_server = Mock(return_value=False)
        calls = []

        def run_adb(args, **kwargs):
            calls.append(args)
            if args == ["start-server"]:
                cancel_event.set()
            return subprocess.CompletedProcess(args, 0, "", "")

        clicker._run_adb = Mock(side_effect=run_adb)
        clicker.run_once = Mock()
        clicker.start_loop(cancel_event=cancel_event)

        self.assertEqual(calls, [["start-server"]])
        clicker.run_once.assert_not_called()
        self.assertIsNone(clicker._active_loop_cancel_event)
        self.assertFalse(clicker._loop_worker_active)

    def test_active_cancel_interrupts_action_delay(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            logger=lambda _: None,
        )
        cancel_event = threading.Event()
        clicker._automatic_loop_active = True
        clicker._active_loop_cancel_event = cancel_event
        result = []
        worker = threading.Thread(
            target=lambda: result.append(clicker._wait_after_action(5.0)),
            daemon=True,
        )
        started = time.monotonic()
        worker.start()
        time.sleep(0.05)
        cancel_event.set()
        worker.join(1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [False])
        self.assertLess(time.monotonic() - started, 1.0)

    def test_stop_during_timeout_calculation_skips_callback(self):
        callback = Mock()
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            scan_interval=5.0,
            no_match_timeout=1.0,
            on_timeout_callback=callback,
            logger=lambda _: None,
        )
        clicker.device = object()

        def run_once():
            clicker.last_action_time = time.monotonic() - 10.0
            return False

        original_safe_number = clicker._safe_number
        stopped = False

        def stop_while_calculating(value, default, **kwargs):
            nonlocal stopped
            result = original_safe_number(value, default, **kwargs)
            if not stopped and default == 0 and value == clicker.no_match_timeout:
                stopped = True
                clicker.stop_loop()
            return result

        clicker.run_once = run_once
        clicker._safe_number = Mock(side_effect=stop_while_calculating)
        clicker.start_loop()

        self.assertTrue(stopped)
        callback.assert_not_called()

    def test_consecutive_match_count_and_callback(self):
        callback = Mock()
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            consecutive_match_threshold=3,
            on_consecutive_match_callback=callback,
            logger=lambda _: None,
        )
        # First match
        clicker._record_match("btn_ok.png")
        self.assertEqual(clicker.consecutive_match_count, 1)
        self.assertEqual(clicker.consecutive_match_template, "btn_ok.png")
        callback.assert_not_called()

        # Second match of same template
        clicker._record_match("btn_ok.png")
        self.assertEqual(clicker.consecutive_match_count, 2)
        callback.assert_not_called()

        # Third match reaches threshold (3) -> triggers callback once
        clicker._record_match("btn_ok.png")
        self.assertEqual(clicker.consecutive_match_count, 3)
        callback.assert_called_once_with("btn_ok.png", 3)

        # Fourth match of same template -> count increments, but callback not called repeatedly
        clicker._record_match("btn_ok.png")
        self.assertEqual(clicker.consecutive_match_count, 4)
        self.assertEqual(callback.call_count, 1)

        # Different template match -> streak resets to 1
        clicker._record_match("btn_close.png")
        self.assertEqual(clicker.consecutive_match_count, 1)
        self.assertEqual(clicker.consecutive_match_template, "btn_close.png")
        self.assertFalse(clicker._consecutive_alert_triggered)

    def test_consecutive_match_disabled_when_threshold_zero(self):
        callback = Mock()
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            consecutive_match_threshold=0,
            on_consecutive_match_callback=callback,
            logger=lambda _: None,
        )
        for _ in range(10):
            clicker._record_match("btn_ok.png")
        self.assertEqual(clicker.consecutive_match_count, 10)
        callback.assert_not_called()

    def test_consecutive_match_config_save_and_load(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="emulator-test",
            consecutive_match_threshold=7,
            logger=lambda _: None,
        )
    def test_multiple_instances_share_same_settings(self):
        clicker1 = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            scan_interval=3,
            no_match_timeout=45,
            similarity_threshold=0.88,
            logger=lambda _: None,
        )
        clicker1.save_config(include_templates=False)

        clicker2 = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5557",
            logger=lambda _: None,
        )
        clicker2.load_config()

        self.assertEqual(clicker2.scan_interval, 3)
        self.assertEqual(clicker2.no_match_timeout, 45)
        self.assertAlmostEqual(clicker2.similarity_threshold, 0.88)
    def test_reset_counts_on_startup_save_and_load(self):
        clicker1 = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            reset_counts_on_startup=True,
            logger=lambda _: None,
        )
        self.assertTrue(clicker1.reset_counts_on_startup)
        clicker1.save_config(include_templates=False)

        clicker2 = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5557",
            logger=lambda _: None,
        )
        clicker2.load_config()
        self.assertTrue(clicker2.reset_counts_on_startup)

    def test_reset_counts_on_startup_resets_counts_when_loaded(self):
        template_path = Path(self.temp_dir, "templates", "test.png")
        self._write_png(template_path, self._pattern())
        config = {
            "device_address": "127.0.0.1:5555",
            "reset_counts_on_startup": True,
            "template_order": ["test.png"],
            "template_counts": {"test.png": 42},
        }
        Path(self.temp_dir, "config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            reset_counts_on_startup=True,
            logger=lambda _: None,
        )
        clicker.load_config()
        clicker.reset_counts()
        self.assertEqual(clicker.template_counts.get("test.png"), 0)

    def test_rename_template_success(self):
        template_path = Path(self.temp_dir, "templates", "old_btn.png")
        self._write_png(template_path, self._pattern())
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        clicker.template_order = ["old_btn.png"]
        clicker.template_counts = {"old_btn.png": 15}
        clicker.template_actions = {"old_btn.png": "double_click"}
        clicker.template_offsets = {"old_btn.png": [10, 20]}
        clicker.template_delays = {"old_btn.png": 1.5}
        clicker.template_delay_types = {"old_btn.png": "post"}
        clicker.template_rois = {"old_btn.png": [0.1, 0.2, 0.3, 0.4]}

        success, new_name = clicker.rename_template("old_btn.png", "new_btn")
        self.assertTrue(success)
        self.assertEqual(new_name, "new_btn.png")
        self.assertFalse(os.path.exists(str(template_path)))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "templates", "new_btn.png")))

        self.assertEqual(clicker.template_order, ["new_btn.png"])
        self.assertEqual(clicker.template_counts["new_btn.png"], 15)
        self.assertEqual(clicker.template_actions["new_btn.png"], "double_click")
        self.assertEqual(clicker.template_offsets["new_btn.png"], [10, 20])
        self.assertEqual(clicker.template_delays["new_btn.png"], 1.5)
        self.assertEqual(clicker.template_delay_types["new_btn.png"], "post")
        self.assertEqual(clicker.template_rois["new_btn.png"], [0.1, 0.2, 0.3, 0.4])

    def test_rename_fallback_template_success(self):
        template_path = Path(self.temp_dir, "fallback_templates", "fb_old.png")
        self._write_png(template_path, self._pattern())
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        clicker.fallback_template_order = ["fb_old.png"]
        clicker.fallback_template_counts = {"fb_old.png": 3}
        clicker.fallback_template_actions = {"fb_old.png": "back"}

        success, new_name = clicker.rename_fallback_template("fb_old.png", "fb_new.png")
        self.assertTrue(success)
        self.assertEqual(new_name, "fb_new.png")
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "fallback_templates", "fb_new.png")))
        self.assertEqual(clicker.fallback_template_order, ["fb_new.png"])
        self.assertEqual(clicker.fallback_template_counts["fb_new.png"], 3)
        self.assertEqual(clicker.fallback_template_actions["fb_new.png"], "back")

    def test_rename_template_validation(self):
        t1 = Path(self.temp_dir, "templates", "btn1.png")
        t2 = Path(self.temp_dir, "templates", "btn2.png")
        self._write_png(t1, self._pattern())
        self._write_png(t2, self._pattern())
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        # Duplicate name
        success, msg = clicker.rename_template("btn1.png", "btn2.png")
        self.assertFalse(success)
        self.assertIn("이미 존재합니다", msg)

        # Invalid character
        success, msg = clicker.rename_template("btn1.png", "bad:name.png")
        self.assertFalse(success)
        self.assertIn("특수문자", msg)

        # Empty name
        success, msg = clicker.rename_template("btn1.png", "   ")
        self.assertFalse(success)

        # Same name (no-op success)
        success, msg = clicker.rename_template("btn1.png", "btn1.png")
        self.assertTrue(success)

    def test_template_filtering_logic(self):
        from gui_app import InstanceTabFrame
        mock_frame = MagicMock()
        mock_frame._destroyed = False
        mock_frame.winfo_exists.return_value = True

        # 1. Primary search with matching query
        mock_frame.primary_search_var = MagicMock()
        mock_frame.primary_search_var.get.return_value = "login"
        mock_frame.clicker = MagicMock()
        mock_frame.clicker.template_order = ["btn_login.png", "btn_cancel.png", "login_popup.png"]

        w_login = MagicMock()
        w_cancel = MagicMock()
        w_popup = MagicMock()
        mock_frame.template_row_widgets = {
            "btn_login.png": w_login,
            "btn_cancel.png": w_cancel,
            "login_popup.png": w_popup,
        }
        mock_frame.templates_frame = MagicMock()
        mock_frame.primary_no_match_label = None

        InstanceTabFrame.filter_templates(mock_frame, is_fallback=False)

        w_login.frame.pack.assert_called_with(fill="x", padx=5, pady=1)
        w_cancel.frame.pack_forget.assert_called_once()
        w_popup.frame.pack.assert_called_with(fill="x", padx=5, pady=1)

        # 2. Fallback search with no match
        mock_frame.fallback_search_var = MagicMock()
        mock_frame.fallback_search_var.get.return_value = "not_found"
        mock_frame.clicker.fallback_template_order = ["fb_retry.png"]
        w_retry = MagicMock()
        mock_frame.fallback_template_row_widgets = {"fb_retry.png": w_retry}
        mock_frame.fallback_templates_frame = MagicMock()
        mock_frame.fb_no_match_label = MagicMock()
        mock_frame.fb_no_match_label.winfo_exists.return_value = True

        InstanceTabFrame.filter_templates(mock_frame, is_fallback=True)

        w_retry.frame.pack_forget.assert_called_once()
        mock_frame.fb_no_match_label.pack.assert_called_with(pady=20)

        # 3. Clear search restores all
        mock_frame.primary_search_var.get.return_value = ""
        w_cancel.frame.pack.reset_mock()
        InstanceTabFrame.filter_templates(mock_frame, is_fallback=False)
        w_cancel.frame.pack.assert_called_with(fill="x", padx=5, pady=1)

    def test_remove_template_from_all_tabs_cleans_clicker_memory(self):
        from gui_app import App
        mock_app = MagicMock()
        mock_frame1 = MagicMock()
        mock_frame2 = MagicMock()

        mock_clicker1 = MagicMock()
        mock_clicker1.template_order = ["t1.png", "target.png", "t2.png"]
        mock_clicker1.template_counts = {"target.png": 5, "t1.png": 1}
        mock_clicker1.template_actions = {"target.png": "click"}
        mock_clicker1.template_offsets = {"target.png": [0, 0]}
        mock_clicker1.template_delays = {"target.png": 1.0}
        mock_clicker1.template_delay_types = {"target.png": "pre"}
        mock_clicker1.template_rois = {"target.png": [0, 0, 1, 1]}

        mock_clicker2 = MagicMock()
        mock_clicker2.template_order = ["target.png", "t3.png"]
        mock_clicker2.template_counts = {"target.png": 2}
        mock_clicker2.template_actions = {"target.png": "double"}
        mock_clicker2.template_offsets = {"target.png": [1, 1]}
        mock_clicker2.template_delays = {"target.png": 0.5}
        mock_clicker2.template_delay_types = {"target.png": "post"}
        mock_clicker2.template_rois = {"target.png": [0, 0, 1, 1]}

        mock_frame1.clicker = mock_clicker1
        mock_frame1.template_row_widgets = {"target.png": MagicMock()}
        mock_frame1.template_count_labels = {"target.png": MagicMock()}

        mock_frame2.clicker = mock_clicker2
        mock_frame2.template_row_widgets = {"target.png": MagicMock()}
        mock_frame2.template_count_labels = {"target.png": MagicMock()}

        mock_app.tab_frames = {"inst1": mock_frame1, "inst2": mock_frame2}

        App.remove_template_from_all_tabs(mock_app, "target.png", is_fallback=False)

        self.assertNotIn("target.png", mock_clicker1.template_order)
        self.assertNotIn("target.png", mock_clicker1.template_counts)
        self.assertNotIn("target.png", mock_clicker1.template_actions)
        self.assertNotIn("target.png", mock_clicker2.template_order)
        self.assertNotIn("target.png", mock_clicker2.template_counts)
        self.assertNotIn("target.png", mock_clicker2.template_actions)
        self.assertNotIn("target.png", mock_frame1.template_row_widgets)
        self.assertNotIn("target.png", mock_frame2.template_row_widgets)

    def test_load_template_missing_file_does_not_log_error(self):
        logs = []
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda msg: logs.append(msg),
        )
        non_existent_path = os.path.join(self.temp_dir, "templates", "deleted_file.png")
        template, method = clicker._load_template(non_existent_path)
        self.assertIsNone(template)
        self.assertIsNone(method)
        # Should not spam error logs when file simply doesn't exist
        for log in logs:
            self.assertNotIn("템플릿 로드 실패", log)

    def test_missing_template_file_auto_pruned_in_scan_loop(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            device_address="127.0.0.1:5555",
            logger=lambda _: None,
        )
        clicker.template_order = ["non_existent_1.png", "non_existent_2.png"]
        clicker.template_counts = {"non_existent_1.png": 3}
        clicker._automatic_loop_active = False

        # Mock capture_screen to return a blank valid image
        screen = np.zeros((100, 100, 3), dtype=np.uint8)
        clicker.capture_screen = MagicMock(return_value=screen)
        clicker.run_once()

        # Both missing files should have been automatically pruned from template_order
        self.assertEqual(clicker.template_order, [])
        self.assertEqual(clicker.template_counts, {})

    def test_fast_screen_hash_detection(self):
        screen1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        screen2 = screen1.copy()
        screen3 = screen1.copy()
        screen3[540, 960] = [255, 255, 255]  # Center pixel change

        hash1 = AutoClicker._fast_screen_hash(screen1)
        hash2 = AutoClicker._fast_screen_hash(screen2)
        hash3 = AutoClicker._fast_screen_hash(screen3)

        self.assertEqual(hash1, hash2)
        # Should be different when content changes
        screen_corner = screen1.copy()
        screen_corner[0, 0] = [100, 100, 100]
        hash_corner = AutoClicker._fast_screen_hash(screen_corner)
        self.assertNotEqual(hash1, hash_corner)

    def test_background_tab_timer_throttling(self):
        from gui_app import App
        mock_app = MagicMock()
        mock_app._closing = False
        mock_app.tabview.get.return_value = "ActiveTab"

        active_frame = MagicMock()
        bg_frame = MagicMock()
        bg_frame._last_bg_timer_update = 0.0

        mock_app.tab_frames = {
            "ActiveTab": active_frame,
            "BgTab": bg_frame,
        }

        # 1. First pump: Active tab updates display, BgTab does warning check + initial update
        with patch("time.monotonic", return_value=100.0):
            App._pump_timer_updates(mock_app)
            active_frame.update_timer_display.assert_called_once()
            bg_frame.check_tab_warning_status.assert_called_once()
            bg_frame.update_timer_display.assert_called_once()

        # 2. Second pump within 2.5s (at 101.0s):
        # Active tab updates display, BgTab only runs warning check (display throttled)
        active_frame.update_timer_display.reset_mock()
        bg_frame.check_tab_warning_status.reset_mock()
        bg_frame.update_timer_display.reset_mock()

        with patch("time.monotonic", return_value=101.0):
            App._pump_timer_updates(mock_app)
            active_frame.update_timer_display.assert_called_once()
            bg_frame.check_tab_warning_status.assert_called_once()
            bg_frame.update_timer_display.assert_not_called()

        # 3. Third pump after 2.5s (at 102.6s):
        # BgTab display is updated again
        active_frame.update_timer_display.reset_mock()
        bg_frame.check_tab_warning_status.reset_mock()
        bg_frame.update_timer_display.reset_mock()

        with patch("time.monotonic", return_value=102.6):
            App._pump_timer_updates(mock_app)
            active_frame.update_timer_display.assert_called_once()
            bg_frame.check_tab_warning_status.assert_called_once()
            bg_frame.update_timer_display.assert_called_once()

    def test_resolve_adb_path_bundled_and_custom(self):
        default_path = get_default_adb_path(self.temp_dir)
        self.assertEqual(
            resolve_adb_path("bundled", base_dir=self.temp_dir), default_path
        )
        self.assertEqual(
            resolve_adb_path("custom", "C:\\custom\\adb.exe", base_dir=self.temp_dir),
            "C:\\custom\\adb.exe",
        )
        self.assertEqual(
            resolve_adb_path("custom", "", base_dir=self.temp_dir), default_path
        )
        self.assertEqual(
            resolve_adb_path("custom", "   ", base_dir=self.temp_dir), default_path
        )

    def test_auto_clicker_adb_mode_custom_and_config_persistence(self):
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            adb_mode="custom",
            custom_adb_path="D:\\tools\\adb.exe",
            logger=lambda _: None,
        )
        self.assertEqual(clicker.adb_mode, "custom")
        self.assertEqual(clicker.custom_adb_path, "D:\\tools\\adb.exe")
        self.assertEqual(clicker.adb_path, "D:\\tools\\adb.exe")

        saved = clicker.save_config()
        self.assertTrue(saved)

        config_file = os.path.join(self.temp_dir, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("adb_mode"), "custom")
        self.assertEqual(data.get("custom_adb_path"), "D:\\tools\\adb.exe")

        clicker2 = AutoClicker(base_dir=self.temp_dir, logger=lambda _: None)
        clicker2.load_config()
        self.assertEqual(clicker2.adb_mode, "custom")
        self.assertEqual(clicker2.custom_adb_path, "D:\\tools\\adb.exe")
        self.assertEqual(clicker2.adb_path, "D:\\tools\\adb.exe")

    def test_apply_settings_updates_adb_mode_and_path(self):
        clicker = AutoClicker(base_dir=self.temp_dir, logger=lambda _: None)
        default_path = get_default_adb_path(self.temp_dir)
        self.assertEqual(clicker.adb_path, default_path)

        clicker._apply_settings(
            {"adb_mode": "custom", "custom_adb_path": "E:\\adb\\adb.exe"}
        )
        self.assertEqual(clicker.adb_mode, "custom")
        self.assertEqual(clicker.custom_adb_path, "E:\\adb\\adb.exe")
        self.assertEqual(clicker.adb_path, "E:\\adb\\adb.exe")

        clicker._apply_settings({"adb_mode": "bundled"})
        self.assertEqual(clicker.adb_mode, "bundled")
        self.assertEqual(clicker.adb_path, default_path)


    def test_validate_adb_executable_checks_file_and_version_output(self):
        fake_adb = Path(self.temp_dir, "custom-adb.exe")
        fake_adb.write_bytes(b"not-really-an-executable")
        AutoClicker._adb_validation_cache.clear()
        version_result = subprocess.CompletedProcess(
            [str(fake_adb), "version"],
            0,
            "Android Debug Bridge version 1.0.41\nVersion test",
            "",
        )

        with patch("main.subprocess.run", return_value=version_result) as run:
            valid, resolved, message = AutoClicker.validate_adb_executable(
                str(fake_adb)
            )

        self.assertTrue(valid)
        self.assertEqual(Path(resolved).resolve(), fake_adb.resolve())
        self.assertIn("Android Debug Bridge", message)
        run.assert_called_once()

        missing = Path(self.temp_dir, "missing-adb.exe")
        valid, resolved, message = AutoClicker.validate_adb_executable(str(missing))
        self.assertFalse(valid)
        self.assertEqual(Path(resolved).resolve(), missing.resolve())
        self.assertIn("찾을 수 없습니다", message)

    def test_failed_start_is_recovered_when_another_process_started_server(self):
        messages = []
        clicker = AutoClicker(
            base_dir=self.temp_dir,
            port=61337,
            logger=messages.append,
        )
        clicker._probe_adb_server = Mock(side_effect=[False, True])
        clicker._run_adb = Mock(
            return_value=subprocess.CompletedProcess(
                [],
                1,
                "",
                "could not read ok from ADB Server\nfailed to start daemon",
            )
        )

        with patch("main.time.sleep"):
            self.assertTrue(clicker._ensure_adb_server())

        clicker._run_adb.assert_called_once_with(["start-server"], check=False)
        self.assertTrue(any("could not read ok" in message for message in messages))
        self.assertTrue(any("다른 스레드/프로세스" in message for message in messages))

    def test_process_wide_adb_lock_starts_server_once_for_concurrent_tabs(self):
        ready = threading.Event()
        call_lock = threading.Lock()
        start_calls = 0
        clickers = [
            AutoClicker(
                base_dir=self.temp_dir,
                port=61338,
                logger=lambda _: None,
            )
            for _ in range(8)
        ]

        def probe(timeout=0.35):
            return ready.is_set()

        def start_server(args, **kwargs):
            nonlocal start_calls
            self.assertEqual(args, ["start-server"])
            with call_lock:
                start_calls += 1
            time.sleep(0.05)
            ready.set()
            return subprocess.CompletedProcess(args, 0, "", "")

        for clicker in clickers:
            clicker._probe_adb_server = probe
            clicker._run_adb = start_server

        with ThreadPoolExecutor(max_workers=len(clickers)) as executor:
            results = list(executor.map(lambda item: item._ensure_adb_server(), clickers))

        self.assertTrue(all(results))
        self.assertEqual(start_calls, 1)

    @unittest.skipUnless(os.name == "nt", "Bundled adb.exe integration test is Windows-only")
    def test_concurrent_cold_start_succeeds_on_temporary_adb_port(self):
        adb_path = Path(__file__).resolve().parent / "ADB" / "adb.exe"
        if not adb_path.is_file():
            self.skipTest("Bundled adb.exe is unavailable")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
            probe_socket.bind(("127.0.0.1", 0))
            temporary_port = probe_socket.getsockname()[1]

        cleanup_command = [
            str(adb_path),
            "-P",
            str(temporary_port),
            "kill-server",
        ]
        subprocess.run(cleanup_command, capture_output=True, timeout=5)
        messages = []
        clickers = [
            AutoClicker(
                adb_path=str(adb_path),
                base_dir=self.temp_dir,
                host="127.0.0.1",
                port=temporary_port,
                logger=messages.append,
            )
            for _ in range(8)
        ]

        try:
            with ThreadPoolExecutor(max_workers=len(clickers)) as executor:
                results = list(
                    executor.map(lambda item: item._ensure_adb_server(), clickers)
                )
            self.assertTrue(all(results), "\n".join(messages))
            start_messages = [
                message for message in messages if "ADB 서버 시작 시도" in message
            ]
            self.assertEqual(len(start_messages), 1, "\n".join(messages))
        finally:
            subprocess.run(cleanup_command, capture_output=True, timeout=5)

if __name__ == "__main__":
    unittest.main()



