import cv2
import numpy as np

# Multi-instance CPU/threading optimization: limit OpenCV internal worker threads
# to 1 to prevent thread explosion (e.g. 5 instances * 12 threads = 60 threads)
# and disable OpenCL context conflicts across background threads.
try:
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from ppadb.client import Client as AdbClient
from ppadb.device import Device as AdbDevice
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
VALID_TEMPLATE_ACTIONS = {"click", "double_click", "click_click", "double", "back"}
VALID_DELAY_TYPES = {"pre", "post"}
VALID_NO_MATCH_ACTIONS = {
    "none", "fallback_list", "random_click", "custom_click",
    "custom_double_click", "double_click", "click_click", "back",
}
CONFIG_FILENAME = "config.json"
TEMPLATE_DIR_NAME = "templates"
FALLBACK_TEMPLATE_DIR_NAME = "fallback_templates"
COUNT_FLUSH_INTERVAL = 5.0
TEMPLATE_MANIFEST_REFRESH_INTERVAL = 1.0
ADB_COMMAND_TIMEOUT = 10.0
ADB_SERVER_PROBE_TIMEOUT = 0.35
ADB_SERVER_START_ATTEMPTS = 3
ADB_SERVER_RETRY_DELAYS = (0.15, 0.35)
ADB_VERSION_TIMEOUT = 5.0
PRESCALE_FACTOR = 0.5
PRESCALE_MIN_TEMPLATE_DIM = 20
LOCAL_VERIFY_MARGIN = 12
LOCAL_VERIFY_TOP_K = 3
HINT_FULL_SCAN_INTERVAL = 10
PERFORMANCE_SAMPLE_LIMIT = 240
DEFAULT_FORCE_SCAN_INTERVAL = 5.0
DEFAULT_ROI_FULL_SCAN_BUDGET = 2
ROI_PRIORITY_TEMPLATE_COUNT = 3
DEFAULT_MAX_IDLE_INTERVAL = 5.0
DEFAULT_CAPTURE_BACKEND = "auto"
VALID_CAPTURE_BACKENDS = {"auto", "png", "raw"}
DEFAULT_ADB_MODE = "bundled"
DEFAULT_CUSTOM_ADB_PATH = ""
VALID_ADB_MODES = {"bundled", "custom"}


def get_app_dir():
    """Return the writable directory next to the app/script, independent of cwd."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def get_default_adb_path(base_dir=None):
    """Return the bundled ADB executable when available, extracted to a persistent path in onefile mode."""
    if base_dir:
        custom_base_adb = os.path.join(base_dir, "ADB", "adb.exe")
        if os.path.exists(custom_base_adb):
            return custom_base_adb

    app_dir_adb = os.path.join(get_app_dir(), "ADB", "adb.exe")
    if os.path.exists(app_dir_adb):
        return app_dir_adb

    if hasattr(sys, "_MEIPASS"):
        meipass_adb_dir = os.path.join(sys._MEIPASS, "ADB")
        meipass_adb_exe = os.path.join(meipass_adb_dir, "adb.exe")
        if os.path.exists(meipass_adb_exe):
            persistent_adb_dir = os.path.join(
                os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                "AutoClicker",
                "ADB",
            )
            persistent_adb_exe = os.path.join(persistent_adb_dir, "adb.exe")
            try:
                os.makedirs(persistent_adb_dir, exist_ok=True)
                for item in os.listdir(meipass_adb_dir):
                    src_file = os.path.join(meipass_adb_dir, item)
                    dst_file = os.path.join(persistent_adb_dir, item)
                    if os.path.isfile(src_file):
                        if (
                            not os.path.exists(dst_file)
                            or os.path.getsize(src_file) != os.path.getsize(dst_file)
                        ):
                            try:
                                shutil.copy2(src_file, dst_file)
                            except Exception:
                                pass
                if os.path.exists(persistent_adb_exe):
                    return persistent_adb_exe
            except Exception:
                pass
            return meipass_adb_exe

    return "adb.exe"


def resolve_adb_path(adb_mode=DEFAULT_ADB_MODE, custom_adb_path="", base_dir=None):
    """Return the executable path based on ADB mode ('bundled' vs 'custom')."""
    if adb_mode == "custom" and custom_adb_path and str(custom_adb_path).strip():
        return os.path.expandvars(os.path.expanduser(str(custom_adb_path).strip()))
    return get_default_adb_path(base_dir)


ADB_HOST = "127.0.0.1"
ADB_PORT = 5037
DEVICE_ADDRESS = "127.0.0.1:5555"
DEFAULT_SCAN_INTERVAL = 1
DEFAULT_NO_MATCH_TIMEOUT = 120
DEFAULT_SIMILARITY_THRESHOLD = 0.9
DEFAULT_MATCH_GRAYSCALE = True
DEFAULT_ENABLE_RANDOM_CLICK = False
DEFAULT_RANDOM_CLICK_INTERVAL = 30
DEFAULT_DOUBLE_CLICK_INTERVAL = 1.0
DEFAULT_POST_ACTION_DELAY = 2.0
DEFAULT_CONSECUTIVE_MATCH_THRESHOLD = 0
DEFAULT_RESET_COUNTS_ON_STARTUP = False
APP_DIR = get_app_dir()
CONFIG_PATH = os.path.join(APP_DIR, CONFIG_FILENAME)
ADB_PATH = get_default_adb_path(APP_DIR)


class AutoClicker:
    """Thread-safe ADB screen matcher used by both the CLI and GUI."""

    _config_lock = threading.RLock()
    _template_cache_lock = threading.RLock()
    _template_manifest_lock = threading.RLock()
    _preload_lock = threading.Lock()
    _template_cache = {}
    _scaled_template_cache = {}
    _template_directory_manifests = {}
    _template_directory_last_scan = {}
    _template_directory_generations = defaultdict(int)
    _template_file_generations = defaultdict(int)
    _pending_count_deltas = defaultdict(
        lambda: {
            "template_counts": defaultdict(int),
            "fallback_template_counts": defaultdict(int),
        }
    )
    _count_flush_timers = {}
    _live_counts = defaultdict(
        lambda: {
            "template_counts": {},
            "fallback_template_counts": {},
        }
    )
    _config_cache = {}
    _config_cache_mtime = {}
    _save_debounce_timers = {}

    # ADB is one process-wide service per host/port. Per-instance locks cannot
    # protect a cold start when the GUI connects several tabs at once.
    _adb_server_locks_guard = threading.Lock()
    _adb_server_locks = {}
    _adb_validation_lock = threading.Lock()
    _adb_validation_cache = {}
    def __init__(
        self,
        adb_path=None,
        host=ADB_HOST,
        port=ADB_PORT,
        device_address=None,
        scan_interval=None,
        no_match_timeout=None,
        similarity_threshold=None,
        enable_random_click=None,
        random_click_interval=None,
        double_click_interval=None,
        on_timeout_callback=None,
        logger=None,
        on_match_callback=None,
        base_dir=None,
        match_grayscale=None,
        post_action_delay=None,
        local_verify=None,
        frame_change_detection=None,
        adaptive_scan_interval=None,
        adaptive_template_order=None,
        capture_backend=None,
        performance_metrics=None,
        consecutive_match_threshold=None,
        on_consecutive_match_callback=None,
        reset_counts_on_startup=None,
        adb_mode=None,
        custom_adb_path=None,
    ):
        self.base_dir = os.path.abspath(base_dir or APP_DIR)
        self.adb_mode = adb_mode if adb_mode in VALID_ADB_MODES else DEFAULT_ADB_MODE
        self.custom_adb_path = str(custom_adb_path or "").strip()
        self._adb_path_explicit = adb_path is not None
        if self._adb_path_explicit:
            self.adb_path = adb_path
        else:
            self.adb_path = resolve_adb_path(
                self.adb_mode, self.custom_adb_path, self.base_dir
            )
        self.host = host
        self.port = port
        self._device_address_explicit = device_address is not None
        self.device_address = str(device_address or DEVICE_ADDRESS).strip()
        self.reset_counts_on_startup = (
            bool(reset_counts_on_startup)
            if reset_counts_on_startup is not None
            else DEFAULT_RESET_COUNTS_ON_STARTUP
        )
        self.scan_interval = (
            scan_interval if scan_interval is not None else DEFAULT_SCAN_INTERVAL
        )
        self.no_match_timeout = (
            no_match_timeout
            if no_match_timeout is not None
            else DEFAULT_NO_MATCH_TIMEOUT
        )
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else DEFAULT_SIMILARITY_THRESHOLD
        )
        self.match_grayscale = (
            match_grayscale
            if match_grayscale is not None
            else DEFAULT_MATCH_GRAYSCALE
        )
        self.enable_random_click = (
            enable_random_click
            if enable_random_click is not None
            else DEFAULT_ENABLE_RANDOM_CLICK
        )
        self.random_click_interval = (
            random_click_interval
            if random_click_interval is not None
            else DEFAULT_RANDOM_CLICK_INTERVAL
        )
        self.double_click_interval = (
            double_click_interval
            if double_click_interval is not None
            else DEFAULT_DOUBLE_CLICK_INTERVAL
        )
        self.post_action_delay = (
            post_action_delay
            if post_action_delay is not None
            else DEFAULT_POST_ACTION_DELAY
        )
        self.local_verify = True if local_verify is None else bool(local_verify)
        self.local_verify_margin = LOCAL_VERIFY_MARGIN
        self.local_verify_top_k = LOCAL_VERIFY_TOP_K
        self.dynamic_roi = True
        self.roi_fullscreen_fallback = True
        self.roi_full_scan_budget = DEFAULT_ROI_FULL_SCAN_BUDGET
        self.frame_change_detection = (
            True if frame_change_detection is None else bool(frame_change_detection)
        )
        self.force_scan_interval = DEFAULT_FORCE_SCAN_INTERVAL
        self.adaptive_scan_interval = (
            True if adaptive_scan_interval is None else bool(adaptive_scan_interval)
        )
        self.max_idle_interval = DEFAULT_MAX_IDLE_INTERVAL
        self.adaptive_template_order = (
            False if adaptive_template_order is None else bool(adaptive_template_order)
        )
        self.capture_backend = str(
            capture_backend or DEFAULT_CAPTURE_BACKEND
        ).strip().lower()
        if self.capture_backend not in VALID_CAPTURE_BACKENDS:
            self.capture_backend = DEFAULT_CAPTURE_BACKEND
        self.performance_metrics = (
            False if performance_metrics is None else bool(performance_metrics)
        )
        self._optimization_defaults = {
            "local_verify": self.local_verify,
            "frame_change_detection": self.frame_change_detection,
            "adaptive_scan_interval": self.adaptive_scan_interval,
            "adaptive_template_order": self.adaptive_template_order,
            "capture_backend": self.capture_backend,
            "performance_metrics": self.performance_metrics,
        }
        self.no_match_action = "none"
        self.no_match_interval = self.random_click_interval
        self.no_match_coords = [500, 500]
        self.fallback_final_action = "none"
        self.fallback_final_coords = [500, 500]
        self.consecutive_match_threshold = (
            self._safe_int(
                consecutive_match_threshold,
                DEFAULT_CONSECUTIVE_MATCH_THRESHOLD,
                minimum=0,
            )
            if consecutive_match_threshold is not None
            else DEFAULT_CONSECUTIVE_MATCH_THRESHOLD
        )
        self.on_consecutive_match_callback = on_consecutive_match_callback
        self.consecutive_match_count = 0
        self.consecutive_match_template = None
        self._consecutive_alert_triggered = False
        self.on_timeout_callback = on_timeout_callback
        self.on_match_callback = on_match_callback
        self.logger = logger if logger else print

        now = time.monotonic()
        self.last_action_time = now
        self.last_match_time = now
        self.last_random_click_time = now
        self.last_matched_template = None
        self.last_matched_is_fallback = False
        self._timeout_alert_active = False
        self._last_frame_signature = None
        self._last_full_scan_time = 0.0
        self._last_scan_had_match = False
        self._force_full_scan = True
        self._unchanged_frame_streak = 0
        self._automatic_loop_active = False
        self._template_location_hints = {}
        self._hint_lock = threading.RLock()
        self._template_hint_hits = defaultdict(int)
        self._roi_full_scan_times = {}
        self._roi_full_scan_reservation_times = {}
        self._roi_full_scan_round_robin = {}
        self._roi_full_scan_allowed = set()
        self._roi_full_scan_budget_remaining = 0
        self._roi_full_scan_frame_active = False
        self._roi_screen_shape = None
        self._transition_counts = defaultdict(Counter)
        self._adaptive_order_cycle = 0
        self._perf_lock = threading.Lock()
        self._perf_samples = defaultdict(
            lambda: deque(maxlen=PERFORMANCE_SAMPLE_LIMIT)
        )
        self._exec_out_disabled_until = 0.0
        self._exec_out_failure_count = 0
        self._raw_capture_disabled_until = 0.0
        self._next_reconnect_at = 0.0
        self._reconnect_delay = 1.0

        self.client = None
        self.device = None
        self.is_running = False
        self._stop_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._loop_wake_event = threading.Event()
        self._active_loop_cancel_event = None
        self._loop_worker_active = False
        self._device_lock = threading.RLock()
        self._loop_state_lock = threading.RLock()
        self._seen_template_directory_generations = {}

        self.template_dir = os.path.join(self.base_dir, TEMPLATE_DIR_NAME)
        self.fallback_template_dir = os.path.join(
            self.base_dir, FALLBACK_TEMPLATE_DIR_NAME
        )
        self.config_path = os.path.join(self.base_dir, CONFIG_FILENAME)
        self.template_order = []
        self.template_counts = {}
        self.template_actions = {}
        self.template_offsets = {}
        self.template_delays = {}
        self.template_delay_types = {}
        self.template_rois = {}
        self.fallback_template_order = []
        self.fallback_template_counts = {}
        self.fallback_template_actions = {}
        self.fallback_template_offsets = {}
        self.fallback_template_delays = {}
        self.fallback_template_delay_types = {}
        self.fallback_template_rois = {}

        os.makedirs(self.template_dir, exist_ok=True)
        os.makedirs(self.fallback_template_dir, exist_ok=True)
        self.load_config()

    def log(self, message):
        try:
            self.logger(message)
        except Exception:
            pass

    @staticmethod
    def _safe_number(value, default, minimum=None, maximum=None):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number):
            return default
        if minimum is not None and number < minimum:
            return default
        if maximum is not None and number > maximum:
            return default
        return int(number) if number.is_integer() else number

    @staticmethod
    def _safe_int(value, default=0, minimum=0):
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return number if number >= minimum else default

    @staticmethod
    def _safe_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "on", "1"}:
                return True
            if normalized in {"false", "no", "off", "0", ""}:
                return False
        return default

    @staticmethod
    def _safe_coords(value, default=None):
        fallback = list(default or [500, 500])
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return fallback
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _safe_offsets(value):
        if not isinstance(value, dict):
            return {}
        result = {}
        for filename, coords in value.items():
            if not isinstance(filename, str):
                continue
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                try:
                    result[filename] = [int(coords[0]), int(coords[1])]
                except (TypeError, ValueError):
                    pass
        return result

    @staticmethod
    def _safe_delays(value):
        if not isinstance(value, dict):
            return {}
        result = {}
        for filename, delay_val in value.items():
            if not isinstance(filename, str):
                continue
            try:
                d = float(delay_val)
                if math.isfinite(d) and d >= 0:
                    result[filename] = int(d) if d.is_integer() else round(d, 2)
            except (TypeError, ValueError):
                pass
        return result

    @staticmethod
    def _safe_delay_types(value):
        if not isinstance(value, dict):
            return {}
        result = {}
        for filename, timing in value.items():
            if not isinstance(filename, str):
                continue
            if timing in VALID_DELAY_TYPES:
                result[filename] = timing
        return result


    @staticmethod
    def _safe_rois(value):
        """Validate normalized [left, top, right, bottom] template regions."""
        if not isinstance(value, dict):
            return {}
        result = {}
        for filename, bounds in value.items():
            if not isinstance(filename, str) or not isinstance(bounds, (list, tuple)):
                continue
            if len(bounds) != 4:
                continue
            try:
                left, top, right, bottom = (float(item) for item in bounds)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(item) for item in (left, top, right, bottom)):
                continue
            if 0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0:
                result[filename] = [left, top, right, bottom]
        return result

    def _record_performance(self, stage, started_at):
        if not self.performance_metrics:
            return 0.0
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        with self._perf_lock:
            self._perf_samples[stage].append(elapsed_ms)
        return elapsed_ms

    def get_performance_stats(self, reset=False):
        """Return bounded rolling timing statistics without emitting scan-time logs."""
        with self._perf_lock:
            snapshots = {
                stage: tuple(samples)
                for stage, samples in self._perf_samples.items()
                if samples
            }
            if reset:
                self._perf_samples.clear()
        result = {}
        for stage, samples in snapshots.items():
            ordered = sorted(samples)
            p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
            result[stage] = {
                "count": len(samples),
                "avg_ms": sum(samples) / len(samples),
                "p95_ms": ordered[p95_index],
                "max_ms": max(samples),
            }
        return result

    @staticmethod
    def _fast_screen_hash(screen):
        """Compute an ultra-fast 64-bit sampling hash across an 8x8 grid of sample points.
        Eliminates the multi-megabyte whole-buffer CRC32 scan, completing in ~0.005ms.
        """
        shape = screen.shape
        h, w = shape[:2]
        if h <= 8 or w <= 8:
            contiguous = screen if screen.flags.c_contiguous else np.ascontiguousarray(screen)
            return (
                shape,
                contiguous.dtype.str,
                zlib.crc32(memoryview(contiguous)) & 0xFFFFFFFF,
            )

        # 8x8 uniformly spaced sample points including edges and interior
        ys = np.linspace(0, h - 1, 8, dtype=np.int32)
        xs = np.linspace(0, w - 1, 8, dtype=np.int32)
        sample = screen[ys[:, None], xs]
        sample_bytes = sample.tobytes()

        # 64-bit FNV-1a hash over the compact sample buffer (64~192 bytes)
        h64 = 0xcbf29ce484222325
        for b in sample_bytes:
            h64 = ((h64 ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF

        return (shape, screen.dtype.str, h64)

    @classmethod
    def _frame_signature(cls, screen):
        return cls._fast_screen_hash(screen)

    def _should_scan_frame(self, screen, now):
        if not self.frame_change_detection or not self._automatic_loop_active:
            self._last_full_scan_time = now
            self._unchanged_frame_streak = 0
            return True

        signature = self._frame_signature(screen)
        unchanged = signature == self._last_frame_signature
        self._last_frame_signature = signature
        force_due = now - self._last_full_scan_time >= self.force_scan_interval
        can_skip = (
            unchanged
            and not self._force_full_scan
            and not force_due
            and not self._last_scan_had_match
        )
        if can_skip:
            self._unchanged_frame_streak += 1
            return False

        self._last_full_scan_time = now
        self._force_full_scan = False
        if not unchanged:
            self._unchanged_frame_streak = 0
        return True

    def _mark_screen_dirty(self):
        self._force_full_scan = True
        self._last_frame_signature = None
        self._unchanged_frame_streak = 0

    def _current_loop_interval(self):
        base = float(self._safe_number(
            self.scan_interval, DEFAULT_SCAN_INTERVAL, minimum=0.1
        ))
        if not self.adaptive_scan_interval or self._unchanged_frame_streak <= 0:
            return base
        maximum = max(base, float(self.max_idle_interval))
        multiplier = 1.5 ** min(self._unchanged_frame_streak, 4)
        return min(maximum, base * multiplier)

    def _current_wait_interval(self, now=None, elapsed=0.0):
        """Return cadence remaining, capped by deadlines measured from now."""
        now = time.monotonic() if now is None else now
        interval = max(
            0.01,
            self._current_loop_interval() - max(0.0, float(elapsed)),
        )
        remaining = []
        reconnect_pending = (
            self.device is None
            and self._next_reconnect_at > now
        )
        if not reconnect_pending:
            remaining.append(interval)

        if not reconnect_pending and (
            self.frame_change_detection
            and self._automatic_loop_active
            and self._last_full_scan_time > 0
        ):
            force_remaining = self.force_scan_interval - (
                now - self._last_full_scan_time
            )
            remaining.append(max(0.01, force_remaining))

        if (
            not reconnect_pending
            and self.no_match_action != "none"
            and self.no_match_interval > 0
        ):
            no_match_elapsed = min(
                now - self.last_match_time,
                now - self.last_random_click_time,
            )
            action_remaining = self.no_match_interval - no_match_elapsed
            remaining.append(max(0.01, action_remaining))

        if self.no_match_timeout > 0 and not self._timeout_alert_active:
            timeout_remaining = self.no_match_timeout - (
                now - self.last_action_time
            )
            remaining.append(max(0.01, timeout_remaining))

        if reconnect_pending:
            remaining.append(
                max(0.01, self._next_reconnect_at - now)
            )

        return max(0.01, min(remaining))

    def _wait_for_loop_wake(self, timeout, cancel_event=None):
        timeout = max(0.0, float(timeout))
        if cancel_event is None:
            return self._loop_wake_event.wait(timeout)

        deadline = time.monotonic() + timeout
        while True:
            if cancel_event.is_set():
                self.stop_loop()
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._loop_wake_event.wait(min(0.1, remaining)):
                return True

    def _get_template_scan_order(self, order, is_fallback=False):
        base_order = tuple(order)
        if not self.adaptive_template_order or len(base_order) <= 3:
            return base_order

        self._adaptive_order_cycle += 1
        if self._adaptive_order_cycle % 10 == 0 or not self.last_matched_template:
            return base_order

        previous = (bool(self.last_matched_is_fallback), self.last_matched_template)
        transitions = self._transition_counts.get(previous)
        if not transitions:
            return base_order

        protected = base_order[:3]
        tail = list(base_order[3:])
        original_index = {filename: index for index, filename in enumerate(tail)}
        predicted = [
            filename
            for filename in tail
            if transitions.get((bool(is_fallback), filename), 0) >= 3
        ]
        predicted.sort(
            key=lambda filename: (
                -transitions[(bool(is_fallback), filename)],
                original_index[filename],
            )
        )
        if not predicted:
            return base_order
        predicted_set = set(predicted)
        return protected + tuple(predicted) + tuple(
            filename for filename in tail if filename not in predicted_set
        )
    @classmethod
    def _read_config_unlocked(cls, config_path):
        if not os.path.exists(config_path):
            return {}
        try:
            mtime = os.path.getmtime(config_path)
        except OSError:
            mtime = None
        cached_mtime = cls._config_cache_mtime.get(config_path)
        if cached_mtime is not None and mtime == cached_mtime and config_path in cls._config_cache:
            return copy.deepcopy(cls._config_cache[config_path])
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("config.json 최상위 값은 객체여야 합니다.")
        cls._config_cache[config_path] = copy.deepcopy(config)
        cls._config_cache_mtime[config_path] = mtime
        return copy.deepcopy(config)

    @classmethod
    def _atomic_write_config_unlocked(cls, config_path, config, do_fsync=True):
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        temp_path = (
            f"{config_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with open(temp_path, "w", encoding="utf-8", newline="\n") as file:
                json.dump(config, file, ensure_ascii=False, indent=4)
                file.flush()
                if do_fsync:
                    os.fsync(file.fileno())
            os.replace(temp_path, config_path)
            cls._config_cache[config_path] = copy.deepcopy(config)
            try:
                cls._config_cache_mtime[config_path] = os.path.getmtime(config_path)
            except OSError:
                cls._config_cache_mtime.pop(config_path, None)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @classmethod
    def read_config(cls, config_path=CONFIG_PATH):
        config_path = os.path.abspath(config_path)
        with cls._config_lock:
            try:
                return cls._read_config_unlocked(config_path)
            except (OSError, ValueError, json.JSONDecodeError):
                return {}

    @classmethod
    def _flush_pending_counts_unlocked(cls, config_path):
        config = cls._read_config_unlocked(config_path)
        snapshots = cls._merge_pending_counts_unlocked(config_path, config)
        if not any(snapshots.values()):
            return True

        cls._atomic_write_config_unlocked(config_path, config)
        cls._ack_pending_counts_unlocked(config_path, snapshots)
        return True

    @classmethod
    def _merge_pending_counts_unlocked(cls, config_path, config):
        """Merge a stable delta snapshot into config without writing it yet."""
        pending = cls._pending_count_deltas[config_path]
        snapshots = {}
        for count_key in ("template_counts", "fallback_template_counts"):
            snapshots[count_key] = dict(pending[count_key])
            counts = config.get(count_key, {})
            if not isinstance(counts, dict):
                counts = {}
            else:
                counts = dict(counts)
            for filename, delta in snapshots[count_key].items():
                current = cls._safe_int(counts.get(filename, 0))
                counts[filename] = current + delta
            config[count_key] = counts
        return snapshots

    @classmethod
    def _ack_pending_counts_unlocked(cls, config_path, snapshots):
        pending = cls._pending_count_deltas[config_path]
        for count_key, deltas in snapshots.items():
            for filename, delta in deltas.items():
                pending[count_key][filename] -= delta
                if pending[count_key][filename] <= 0:
                    del pending[count_key][filename]

    @classmethod
    def _schedule_count_flush_unlocked(cls, config_path):
        timer = cls._count_flush_timers.get(config_path)
        if timer is not None and timer.is_alive():
            return

        timer = threading.Timer(
            COUNT_FLUSH_INTERVAL, cls._flush_pending_counts, args=(config_path,)
        )
        timer.daemon = True
        cls._count_flush_timers[config_path] = timer
        timer.start()

    @classmethod
    def _flush_pending_counts(cls, config_path):
        config_path = os.path.abspath(config_path)
        with cls._config_lock:
            cls._count_flush_timers.pop(config_path, None)
            try:
                return cls._flush_pending_counts_unlocked(config_path)
            except (OSError, ValueError, json.JSONDecodeError):
                pending = cls._pending_count_deltas[config_path]
                if any(pending[key] for key in pending):
                    cls._schedule_count_flush_unlocked(config_path)
                return False

    def flush_counts(self):
        if not self._flush_pending_counts(self.config_path):
            self.log("클릭 카운트 저장 실패: config.json 상태를 확인하세요.")
            return False
        return True

    @classmethod
    def update_instances_config(
        cls, instances, config_path=CONFIG_PATH, primary_settings=None
    ):
        """Merge per-device settings and persist the GUI snapshot in one write."""
        config_path = os.path.abspath(config_path)
        if not isinstance(instances, (list, tuple)):
            instances = []
        with cls._config_lock:
            try:
                config = cls._read_config_unlocked(config_path)
                snapshots = cls._merge_pending_counts_unlocked(
                    config_path, config
                )
                existing = config.get("instances", [])
                if not isinstance(existing, list):
                    existing = []
                existing = [
                    item
                    for item in existing
                    if isinstance(item, dict)
                    and isinstance(item.get("device_address"), str)
                ]
                existing_by_address = {
                    item.get("device_address"): dict(item)
                    for item in existing
                    if isinstance(item, dict) and item.get("device_address")
                }
                instance_list = []
                for instance in instances:
                    if isinstance(instance, dict):
                        address = instance.get("device_address")
                        merged = dict(existing_by_address.get(address, {}))
                        merged.update(instance)
                    elif isinstance(instance, str):
                        address = instance
                        merged = dict(existing_by_address.get(address, {"device_address": address}))
                    else:
                        address = None
                        merged = None
                    if address and address not in [i.get("device_address") for i in instance_list]:
                        instance_list.append(merged)
                if isinstance(primary_settings, dict):
                    config.update(primary_settings)
                if instance_list:
                    config["instances"] = instance_list
                cls._atomic_write_config_unlocked(
                    config_path, config, do_fsync=False
                )
                cls._ack_pending_counts_unlocked(config_path, snapshots)
                return True
            except (OSError, ValueError, json.JSONDecodeError):
                return False

    def _apply_settings(self, source):
        if not isinstance(source, dict):
            return

        self.scan_interval = self._safe_number(
            source.get("scan_interval", self.scan_interval),
            self.scan_interval,
            minimum=0.1,
        )
        self.no_match_timeout = self._safe_number(
            source.get("no_match_timeout", self.no_match_timeout),
            self.no_match_timeout,
            minimum=0,
        )
        self.similarity_threshold = self._safe_number(
            source.get("similarity_threshold", self.similarity_threshold),
            self.similarity_threshold,
            minimum=0.01,
            maximum=1.0,
        )
        self.match_grayscale = self._safe_bool(
            source.get("match_grayscale"), self.match_grayscale
        )
        self.enable_random_click = self._safe_bool(
            source.get("enable_random_click"), self.enable_random_click
        )
        self.random_click_interval = self._safe_number(
            source.get("random_click_interval", self.random_click_interval),
            self.random_click_interval,
            minimum=0.1,
        )
        self.double_click_interval = self._safe_number(
            source.get("double_click_interval", self.double_click_interval),
            self.double_click_interval,
            minimum=0.01,
        )
        self.post_action_delay = self._safe_number(
            source.get("post_action_delay", getattr(self, "post_action_delay", DEFAULT_POST_ACTION_DELAY)),
            getattr(self, "post_action_delay", DEFAULT_POST_ACTION_DELAY),
            minimum=0.0,
        )
        defaults = self._optimization_defaults
        self.local_verify = self._safe_bool(
            source.get("local_verify"), defaults["local_verify"]
        )
        self.local_verify_margin = self._safe_int(
            source.get("local_verify_margin", LOCAL_VERIFY_MARGIN),
            LOCAL_VERIFY_MARGIN,
            minimum=2,
        )
        self.local_verify_top_k = min(10, self._safe_int(
            source.get("local_verify_top_k", LOCAL_VERIFY_TOP_K),
            LOCAL_VERIFY_TOP_K,
            minimum=1,
        ))
        self.dynamic_roi = self._safe_bool(
            source.get("dynamic_roi"), True
        )
        self.roi_fullscreen_fallback = self._safe_bool(
            source.get("roi_fullscreen_fallback"), True
        )
        self.roi_full_scan_budget = self._safe_int(
            source.get("roi_full_scan_budget", DEFAULT_ROI_FULL_SCAN_BUDGET),
            DEFAULT_ROI_FULL_SCAN_BUDGET,
            minimum=1,
        )
        self.frame_change_detection = self._safe_bool(
            source.get("frame_change_detection"),
            defaults["frame_change_detection"],
        )
        self.force_scan_interval = self._safe_number(
            source.get("force_scan_interval", DEFAULT_FORCE_SCAN_INTERVAL),
            DEFAULT_FORCE_SCAN_INTERVAL,
            minimum=0.5,
        )
        self.adaptive_scan_interval = self._safe_bool(
            source.get("adaptive_scan_interval"),
            defaults["adaptive_scan_interval"],
        )
        self.max_idle_interval = self._safe_number(
            source.get("max_idle_interval", DEFAULT_MAX_IDLE_INTERVAL),
            DEFAULT_MAX_IDLE_INTERVAL,
            minimum=0.1,
        )
        self.adaptive_template_order = self._safe_bool(
            source.get("adaptive_template_order"),
            defaults["adaptive_template_order"],
        )
        self.performance_metrics = self._safe_bool(
            source.get("performance_metrics"),
            defaults["performance_metrics"],
        )
        backend = str(
            source.get("capture_backend", defaults["capture_backend"])
        ).strip().lower()
        self.capture_backend = (
            backend if backend in VALID_CAPTURE_BACKENDS else DEFAULT_CAPTURE_BACKEND
        )

        action = source.get("no_match_action")
        if action not in VALID_NO_MATCH_ACTIONS:
            action = "random_click" if self.enable_random_click else "none"
        self.no_match_action = action
        self.no_match_interval = self._safe_number(
            source.get("no_match_interval", self.random_click_interval),
            self.random_click_interval,
            minimum=0.1,
        )
        self.no_match_coords = self._safe_coords(
            source.get("no_match_coords"), self.no_match_coords
        )
        fb_action = source.get("fallback_final_action", "none")
        if fb_action not in VALID_NO_MATCH_ACTIONS:
            fb_action = "none"
        self.fallback_final_action = fb_action
        self.fallback_final_coords = self._safe_coords(
            source.get("fallback_final_coords"), self.fallback_final_coords
        )
        self.consecutive_match_threshold = self._safe_int(
            source.get(
                "consecutive_match_threshold",
                getattr(
                    self,
                    "consecutive_match_threshold",
                    DEFAULT_CONSECUTIVE_MATCH_THRESHOLD,
                ),
            ),
            getattr(
                self,
                "consecutive_match_threshold",
                DEFAULT_CONSECUTIVE_MATCH_THRESHOLD,
            ),
            minimum=0,
        )
        self.reset_counts_on_startup = self._safe_bool(
            source.get("reset_counts_on_startup"),
            getattr(self, "reset_counts_on_startup", DEFAULT_RESET_COUNTS_ON_STARTUP),
        )

        adb_mode_val = source.get("adb_mode")
        if adb_mode_val in VALID_ADB_MODES:
            self.adb_mode = adb_mode_val
        elif "adb_mode" in source and isinstance(adb_mode_val, str):
            cleaned_mode = adb_mode_val.strip().lower()
            if cleaned_mode in VALID_ADB_MODES:
                self.adb_mode = cleaned_mode

        if "custom_adb_path" in source and source["custom_adb_path"] is not None:
            self.custom_adb_path = str(source["custom_adb_path"]).strip()

        if not getattr(self, "_adb_path_explicit", False):
            self.adb_path = resolve_adb_path(
                self.adb_mode, self.custom_adb_path, self.base_dir
            )

    @staticmethod
    def _sync_order(saved_order, current_files):
        current_set = set(current_files)
        result = []
        seen = set()
        if isinstance(saved_order, list):
            for filename in saved_order:
                if (
                    isinstance(filename, str)
                    and filename in current_set
                    and filename not in seen
                ):
                    result.append(filename)
                    seen.add(filename)
        result.extend(filename for filename in current_files if filename not in seen)
        return result

    @classmethod
    def _drop_template_cache_entries_unlocked(cls, absolute_path):
        for cache_key in tuple(cls._template_cache):
            cached_path = (
                cache_key[0] if isinstance(cache_key, tuple) else cache_key
            )
            if cached_path == absolute_path:
                cls._template_cache.pop(cache_key, None)
                cls._scaled_template_cache.pop(cache_key, None)

    @classmethod
    def _refresh_template_directory(cls, directory, force=False):
        """Refresh one shared directory manifest at most once per interval."""
        absolute_dir = os.path.abspath(directory)
        now = time.perf_counter()
        manifest = cls._template_directory_manifests.get(absolute_dir)
        last_scan = cls._template_directory_last_scan.get(absolute_dir, 0.0)
        if (
            not force
            and manifest is not None
            and now - last_scan < TEMPLATE_MANIFEST_REFRESH_INTERVAL
        ):
            return manifest

        with cls._template_manifest_lock:
            now = time.perf_counter()
            manifest = cls._template_directory_manifests.get(absolute_dir)
            last_scan = cls._template_directory_last_scan.get(
                absolute_dir, 0.0
            )
            if (
                not force
                and manifest is not None
                and now - last_scan < TEMPLATE_MANIFEST_REFRESH_INTERVAL
            ):
                return manifest

            current = {}
            try:
                entries = os.scandir(absolute_dir)
            except OSError:
                entries = None
            if entries is not None:
                with entries:
                    for entry in entries:
                        if not entry.name.lower().endswith(IMAGE_EXTENSIONS):
                            continue
                        try:
                            if not entry.is_file():
                                continue
                            stat_result = entry.stat()
                        except OSError:
                            continue
                        absolute_path = os.path.abspath(entry.path)
                        current[absolute_path] = (
                            stat_result.st_mtime_ns,
                            stat_result.st_size,
                        )

            previous = manifest or {}
            changed_paths = {
                path
                for path in set(previous).union(current)
                if previous.get(path) != current.get(path)
            }
            if changed_paths:
                cls._template_directory_generations[absolute_dir] += 1
                with cls._template_cache_lock:
                    for absolute_path in changed_paths:
                        cls._template_file_generations[absolute_path] += 1
                        cls._drop_template_cache_entries_unlocked(absolute_path)
            cls._template_directory_manifests[absolute_dir] = current
            cls._template_directory_last_scan[absolute_dir] = now
            return current

    @classmethod
    def _template_is_registered(cls, template_path):
        absolute_path = os.path.abspath(template_path)
        manifest = cls._refresh_template_directory(
            os.path.dirname(absolute_path)
        )
        return absolute_path in manifest

    @classmethod
    def _list_template_files(cls, directory):
        manifest = cls._refresh_template_directory(directory)
        return sorted(os.path.basename(path) for path in manifest)

    def _remember_template_directory_generations(self):
        for directory in (self.template_dir, self.fallback_template_dir):
            absolute_dir = os.path.abspath(directory)
            self._seen_template_directory_generations[absolute_dir] = (
                self._template_directory_generations.get(absolute_dir, 0)
            )

    @staticmethod
    def _prune_template_collection(manifest, order, mappings):
        available = {os.path.basename(path) for path in manifest}
        missing = {filename for filename in order if filename not in available}
        if not missing:
            return False
        order[:] = [filename for filename in order if filename not in missing]
        for mapping in mappings:
            for filename in missing:
                mapping.pop(filename, None)
        return True

    def _refresh_template_manifests(self):
        changed = False
        collections = (
            (
                self.template_dir,
                self.template_order,
                (
                    self.template_counts,
                    self.template_actions,
                    self.template_offsets,
                    self.template_delays,
                    self.template_delay_types,
                    self.template_rois,
                ),
            ),
            (
                self.fallback_template_dir,
                self.fallback_template_order,
                (
                    self.fallback_template_counts,
                    self.fallback_template_actions,
                    self.fallback_template_offsets,
                    self.fallback_template_delays,
                    self.fallback_template_delay_types,
                    self.fallback_template_rois,
                ),
            ),
        )
        for directory, order, mappings in collections:
            absolute_dir = os.path.abspath(directory)
            manifest = self._refresh_template_directory(absolute_dir)
            generation = self._template_directory_generations.get(
                absolute_dir, 0
            )
            previous = self._seen_template_directory_generations.get(
                absolute_dir
            )
            self._seen_template_directory_generations[absolute_dir] = generation
            if previous is not None and previous != generation:
                changed = True
            if self._prune_template_collection(manifest, order, mappings):
                changed = True
        if changed:
            self._force_full_scan = True
        return changed

    def _sync_template_collection(
        self,
        config,
        order_key,
        counts_key,
        actions_key,
        offsets_key,
        delays_key,
        delay_types_key,
        directory,
    ):
        current_files = self._list_template_files(directory)
        order = self._sync_order(config.get(order_key, []), current_files)

        raw_counts = config.get(counts_key, {})
        if not isinstance(raw_counts, dict):
            raw_counts = {}
        raw_actions = config.get(actions_key, {})
        if not isinstance(raw_actions, dict):
            raw_actions = {}
        raw_offsets = self._safe_offsets(config.get(offsets_key, {}))
        raw_delays = self._safe_delays(config.get(delays_key, {}))
        raw_delay_types = self._safe_delay_types(config.get(delay_types_key, {}))

        path_key = os.path.abspath(self.config_path)
        with self._config_lock:
            live_counts = self._live_counts[path_key][counts_key]
            for filename in order:
                live_counts.setdefault(
                    filename, self._safe_int(raw_counts.get(filename, 0))
                )
            for filename in tuple(live_counts):
                if filename not in order:
                    del live_counts[filename]

            counts = {
                filename: self._safe_int(live_counts.get(filename, 0))
                for filename in order
            }

        actions = {}
        for filename in order:
            action = raw_actions.get(filename, "click")
            actions[filename] = (
                action if action in VALID_TEMPLATE_ACTIONS else "click"
            )
        offsets = {
            filename: raw_offsets[filename]
            for filename in order
            if filename in raw_offsets
        }
        delays = {
            filename: self._safe_number(raw_delays.get(filename, 0.0), 0.0, minimum=0.0)
            for filename in order
        }
        delay_types = {
            filename: raw_delay_types.get(filename, "pre")
            for filename in order
        }
        return order, counts, actions, offsets, delays, delay_types

    def load_config(self):
        """Load validated per-instance settings and synchronize template folders."""
        config = {}
        with self._config_lock:
            try:
                config = self._read_config_unlocked(self.config_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.log(f"설정 로드 중 오류: {error}")

        if not self._device_address_explicit:
            configured_address = config.get("device_address")
            if isinstance(configured_address, str) and configured_address.strip():
                self.device_address = configured_address.strip()

        settings_source = dict(config)
        self._apply_settings(settings_source)

        (
            self.template_order,
            self.template_counts,
            self.template_actions,
            self.template_offsets,
            self.template_delays,
            self.template_delay_types,
        ) = self._sync_template_collection(
            config,
            "template_order",
            "template_counts",
            "template_actions",
            "template_offsets",
            "template_delays",
            "template_delay_types",
            self.template_dir,
        )
        (
            self.fallback_template_order,
            self.fallback_template_counts,
            self.fallback_template_actions,
            self.fallback_template_offsets,
            self.fallback_template_delays,
            self.fallback_template_delay_types,
        ) = self._sync_template_collection(
            config,
            "fallback_template_order",
            "fallback_template_counts",
            "fallback_template_actions",
            "fallback_template_offsets",
            "fallback_template_delays",
            "fallback_template_delay_types",
            self.fallback_template_dir,
        )
        self.template_rois = {
            filename: bounds
            for filename, bounds in self._safe_rois(
                config.get("template_rois", {})
            ).items()
            if filename in self.template_order
        }
        self.fallback_template_rois = {
            filename: bounds
            for filename, bounds in self._safe_rois(
                config.get("fallback_template_rois", {})
            ).items()
            if filename in self.fallback_template_order
        }
        with self._hint_lock:
            self._template_location_hints.clear()
            self._template_hint_hits.clear()
            self._roi_full_scan_times.clear()
            self._roi_full_scan_reservation_times.clear()
            self._roi_full_scan_round_robin.clear()
            self._roi_full_scan_allowed.clear()
            self._roi_full_scan_budget_remaining = 0
            self._roi_full_scan_frame_active = False
            self._roi_screen_shape = None

        if not getattr(self, "_adb_path_explicit", False):
            self.adb_path = resolve_adb_path(
                self.adb_mode, self.custom_adb_path, self.base_dir
            )
        self._remember_template_directory_generations()
        return True

    def _settings_dict(self):
        self.enable_random_click = self.no_match_action == "random_click"
        self.random_click_interval = self.no_match_interval
        return {
            "device_address": self.device_address,
            "adb_mode": getattr(self, "adb_mode", DEFAULT_ADB_MODE),
            "custom_adb_path": getattr(self, "custom_adb_path", DEFAULT_CUSTOM_ADB_PATH),
            "scan_interval": self.scan_interval,
            "no_match_timeout": self.no_match_timeout,
            "similarity_threshold": self.similarity_threshold,
            "match_grayscale": self.match_grayscale,
            "enable_random_click": self.enable_random_click,
            "random_click_interval": self.random_click_interval,
            "double_click_interval": self.double_click_interval,
            "post_action_delay": self.post_action_delay,
            "local_verify": self.local_verify,
            "local_verify_margin": self.local_verify_margin,
            "local_verify_top_k": self.local_verify_top_k,
            "dynamic_roi": self.dynamic_roi,
            "roi_fullscreen_fallback": self.roi_fullscreen_fallback,
            "roi_full_scan_budget": self.roi_full_scan_budget,
            "frame_change_detection": self.frame_change_detection,
            "force_scan_interval": self.force_scan_interval,
            "adaptive_scan_interval": self.adaptive_scan_interval,
            "max_idle_interval": self.max_idle_interval,
            "adaptive_template_order": self.adaptive_template_order,
            "capture_backend": self.capture_backend,
            "performance_metrics": self.performance_metrics,
            "no_match_action": self.no_match_action,
            "no_match_interval": self.no_match_interval,
            "no_match_coords": list(self.no_match_coords),
            "fallback_final_action": self.fallback_final_action,
            "fallback_final_coords": list(self.fallback_final_coords),
            "consecutive_match_threshold": self.consecutive_match_threshold,
            "reset_counts_on_startup": getattr(self, "reset_counts_on_startup", DEFAULT_RESET_COUNTS_ON_STARTUP),
        }

    def save_config(self, include_templates=True):
        """Atomically save settings while preserving other instances and counts."""
        with self._config_lock:
            try:
                config = self._read_config_unlocked(self.config_path)
                snapshots = self._merge_pending_counts_unlocked(
                    self.config_path, config
                )
                settings = self._settings_dict()
                config.update(settings)

                instances = config.get("instances", [])
                if not isinstance(instances, list):
                    instances = []
                cleaned_instances = []
                seen_addrs = set()
                for inst in instances:
                    if isinstance(inst, dict):
                        addr = inst.get("device_address")
                    elif isinstance(inst, str):
                        addr = inst
                    else:
                        addr = None
                    if addr and addr not in seen_addrs:
                        seen_addrs.add(addr)
                        cleaned_instances.append({"device_address": addr})
                if self.device_address and self.device_address not in seen_addrs:
                    cleaned_instances.append({"device_address": self.device_address})
                config["instances"] = cleaned_instances

                if include_templates:
                    template_groups = (
                        (
                            "template_order",
                            "template_counts",
                            "template_actions",
                            "template_offsets",
                            "template_delays",
                            "template_delay_types",
                            self.template_order,
                            self.template_counts,
                            self.template_actions,
                            self.template_offsets,
                            self.template_delays,
                            self.template_delay_types,
                        ),
                        (
                            "fallback_template_order",
                            "fallback_template_counts",
                            "fallback_template_actions",
                            "fallback_template_offsets",
                            "fallback_template_delays",
                            "fallback_template_delay_types",
                            self.fallback_template_order,
                            self.fallback_template_counts,
                            self.fallback_template_actions,
                            self.fallback_template_offsets,
                            self.fallback_template_delays,
                            self.fallback_template_delay_types,
                        ),
                    )
                    path_key = os.path.abspath(self.config_path)
                    for (
                        order_key,
                        counts_key,
                        actions_key,
                        offsets_key,
                        delays_key,
                        delay_types_key,
                        order,
                        counts,
                        actions,
                        offsets,
                        delays,
                        delay_types,
                    ) in template_groups:
                        live_counts = self._live_counts[path_key][counts_key]
                        clean_counts = {
                            filename: self._safe_int(
                                live_counts.get(filename, counts.get(filename, 0))
                            )
                            for filename in order
                        }
                        counts.clear()
                        counts.update(clean_counts)
                        config[order_key] = list(order)
                        config[counts_key] = clean_counts
                        config[actions_key] = {
                            filename: actions.get(filename, "click")
                            for filename in order
                        }
                        config[offsets_key] = {
                            filename: list(offsets[filename])
                            for filename in order
                            if filename in offsets
                        }
                        config[delays_key] = {
                            filename: self._safe_number(delays.get(filename, 0.0), 0.0, minimum=0.0)
                            for filename in order
                            if filename in delays and delays.get(filename, 0.0) > 0
                        }
                        config[delay_types_key] = {
                            filename: delay_types.get(filename, "pre")
                            for filename in order
                            if filename in delays
                            and delays.get(filename, 0.0) > 0
                            and delay_types.get(filename, "pre") == "post"
                        }

                    config["template_rois"] = {
                        filename: list(self.template_rois[filename])
                        for filename in self.template_order
                        if filename in self.template_rois
                    }
                    config["fallback_template_rois"] = {
                        filename: list(self.fallback_template_rois[filename])
                        for filename in self.fallback_template_order
                        if filename in self.fallback_template_rois
                    }
                self._atomic_write_config_unlocked(
                    self.config_path, config, do_fsync=False
                )
                self._ack_pending_counts_unlocked(
                    self.config_path, snapshots
                )
                return True
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.log(f"설정 저장 중 오류: {error}")
                return False

    @classmethod
    def preload_templates(cls, directories=None, grayscales=(False, True)):
        """Preload requested template modes once, including their coarse pyramids."""
        with cls._preload_lock:
            if directories is None:
                directories = [
                    os.path.join(APP_DIR, TEMPLATE_DIR_NAME),
                    os.path.join(APP_DIR, FALLBACK_TEMPLATE_DIR_NAME),
                ]
            elif isinstance(directories, (str, bytes, os.PathLike)):
                directories = [directories]

            valid_dirs = []
            for directory in directories:
                absolute_dir = os.path.abspath(directory)
                if absolute_dir not in valid_dirs:
                    valid_dirs.append(absolute_dir)

            modes = tuple(dict.fromkeys(bool(gray) for gray in grayscales))
            loaded_count = 0
            for directory in valid_dirs:
                manifest = cls._refresh_template_directory(directory)
                for absolute_path in sorted(manifest):
                    for grayscale in modes:
                        cache_key = (absolute_path, grayscale)
                        template, _ = cls._load_template_direct(
                            absolute_path, grayscale=grayscale
                        )
                        if template is None:
                            continue
                        with cls._template_cache_lock:
                            cached_scaled = cls._scaled_template_cache.get(
                                cache_key
                            )
                            if (
                                cached_scaled is not None
                                and cached_scaled[0] is template
                            ):
                                continue
                        cls._get_scaled_template(cache_key, template)
                        loaded_count += 1
            return loaded_count

    @classmethod
    def _load_template_direct(cls, absolute_path, grayscale=False):
        """Decode one manifest-registered template and cache it by generation."""
        absolute_path = os.path.abspath(absolute_path)
        manifest = cls._refresh_template_directory(
            os.path.dirname(absolute_path)
        )
        if absolute_path not in manifest:
            return None, None

        generation = cls._template_file_generations.get(absolute_path)
        if generation is None:
            return None, None

        use_grayscale = bool(grayscale)
        cache_key = (absolute_path, use_grayscale)
        with cls._template_cache_lock:
            cached = cls._template_cache.get(cache_key)
            if cached is not None and cached[0] == generation:
                return cached[1], cached[2]
            if cached is not None:
                cls._scaled_template_cache.pop(cache_key, None)
        try:
            encoded = np.fromfile(absolute_path, dtype=np.uint8)
            template_color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if (
                template_color is None
                or template_color.size == 0
                or template_color.shape[0] <= 0
                or template_color.shape[1] <= 0
            ):
                raise ValueError("이미지 디코딩 결과가 비어 있습니다.")

            template_gray = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)
            method_color = (
                cv2.TM_SQDIFF_NORMED
                if float(template_color.std()) < 1e-6
                else cv2.TM_CCOEFF_NORMED
            )
            method_gray = (
                cv2.TM_SQDIFF_NORMED
                if float(template_gray.std()) < 1e-6
                else cv2.TM_CCOEFF_NORMED
            )

            with cls._template_cache_lock:
                current_generation = cls._template_file_generations.get(
                    absolute_path
                )
                if current_generation != generation:
                    current = cls._template_cache.get(cache_key)
                    if (
                        current is not None
                        and current[0] == current_generation
                    ):
                        return current[1], current[2]
                    return None, None
                cls._scaled_template_cache.pop((absolute_path, False), None)
                cls._scaled_template_cache.pop((absolute_path, True), None)
                cls._template_cache[(absolute_path, False)] = (
                    generation,
                    template_color,
                    method_color,
                )
                cls._template_cache[(absolute_path, True)] = (
                    generation,
                    template_gray,
                    method_gray,
                )
            template = template_gray if use_grayscale else template_color
            method = method_gray if use_grayscale else method_color
        except Exception:
            template, method = None, None
            with cls._template_cache_lock:
                if (
                    cls._template_file_generations.get(absolute_path)
                    == generation
                ):
                    cls._scaled_template_cache.pop(cache_key, None)
                    cls._template_cache[cache_key] = (
                        generation,
                        None,
                        None,
                    )
        return template, method

    @classmethod
    def invalidate_template_cache(cls, template_path=None):
        if template_path is None:
            with cls._template_manifest_lock:
                cls._template_directory_manifests.clear()
                cls._template_directory_last_scan.clear()
                cls._template_directory_generations.clear()
                cls._template_file_generations.clear()
                with cls._template_cache_lock:
                    cls._template_cache.clear()
                    cls._scaled_template_cache.clear()
            return

        absolute_path = os.path.abspath(template_path)
        absolute_dir = os.path.dirname(absolute_path)
        try:
            stat_result = os.stat(absolute_path)
            signature = (stat_result.st_mtime_ns, stat_result.st_size)
        except OSError:
            signature = None

        with cls._template_manifest_lock:
            manifest = dict(
                cls._template_directory_manifests.get(absolute_dir, {})
            )
            if signature is None:
                manifest.pop(absolute_path, None)
            else:
                manifest[absolute_path] = signature
            cls._template_directory_generations[absolute_dir] += 1
            cls._template_file_generations[absolute_path] += 1
            with cls._template_cache_lock:
                cls._drop_template_cache_entries_unlocked(absolute_path)
            cls._template_directory_manifests[absolute_dir] = manifest

    def _remove_count_state(self, filename, counts_key):
        path_key = os.path.abspath(self.config_path)
        with self._config_lock:
            self._pending_count_deltas[path_key][counts_key].pop(filename, None)
            self._live_counts[path_key][counts_key].pop(filename, None)

    def _delete_template(
        self,
        filename,
        directory,
        order,
        counts,
        actions,
        offsets,
        delays,
        delay_types,
        rois,
        counts_key,
    ):
        if not isinstance(filename, str) or os.path.basename(filename) != filename:
            self.log(f"잘못된 템플릿 파일명: {filename}")
            return False

        file_path = os.path.join(directory, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.log(f"파일 삭제됨: {filename}")
            if filename in order:
                order.remove(filename)
            counts.pop(filename, None)
            actions.pop(filename, None)
            offsets.pop(filename, None)
            delays.pop(filename, None)
            delay_types.pop(filename, None)
            rois.pop(filename, None)
            self._discard_location_hint(file_path, clear_full_scan_state=True)
            self._remove_count_state(filename, counts_key)
            self.invalidate_template_cache(file_path)
            return self.save_config(include_templates=True)
        except OSError as error:
            self.log(f"삭제 실패 ({filename}): {error}")
            return False

    def delete_template(self, filename):
        return self._delete_template(
            filename,
            self.template_dir,
            self.template_order,
            self.template_counts,
            self.template_actions,
            self.template_offsets,
            self.template_delays,
            self.template_delay_types,
            self.template_rois,
            "template_counts",
        )

    def delete_fallback_template(self, filename):
        return self._delete_template(
            filename,
            self.fallback_template_dir,
            self.fallback_template_order,
            self.fallback_template_counts,
            self.fallback_template_actions,
            self.fallback_template_offsets,
            self.fallback_template_delays,
            self.fallback_template_delay_types,
            self.fallback_template_rois,
            "fallback_template_counts",
        )

    def _rename_template(
        self,
        old_filename,
        new_filename,
        directory,
        order,
        counts,
        actions,
        offsets,
        delays,
        delay_types,
        rois,
        counts_key,
    ):
        if not isinstance(old_filename, str) or os.path.basename(old_filename) != old_filename:
            self.log(f"잘못된 기존 템플릿 파일명: {old_filename}")
            return False, "잘못된 기존 템플릿 파일명입니다."
        if not isinstance(new_filename, str) or not new_filename.strip():
            self.log("변경할 파일명을 입력해주세요.")
            return False, "변경할 파일명을 입력해주세요."

        new_filename = new_filename.strip()
        if not new_filename.lower().endswith(".png"):
            new_filename += ".png"

        if os.path.basename(new_filename) != new_filename:
            self.log(f"유효하지 않은 파일명 형식입니다: {new_filename}")
            return False, "유효하지 않은 파일명 형식입니다."

        invalid_chars = '<>:"/\\|?*'
        if any(c in new_filename for c in invalid_chars):
            self.log(f"파일명에 사용할 수 없는 문자가 포함되어 있습니다: {new_filename}")
            return False, "파일명에 사용할 수 없는 특수문자(<>:\"/\\|?*)가 포함되어 있습니다."

        if old_filename == new_filename:
            return True, new_filename

        old_path = os.path.join(directory, old_filename)
        new_path = os.path.join(directory, new_filename)

        if os.path.exists(new_path):
            self.log(f"동일한 이름의 템플릿 파일이 이미 존재합니다: {new_filename}")
            return False, f"'{new_filename}' 파일이 이미 존재합니다."

        try:
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                self.log(f"템플릿 파일명 변경됨: {old_filename} -> {new_filename}")
            else:
                self.log(f"기존 템플릿 파일이 존재하지 않습니다: {old_filename}")

            if old_filename in order:
                idx = order.index(old_filename)
                order[idx] = new_filename

            if old_filename in counts:
                counts[new_filename] = counts.pop(old_filename)
            if old_filename in actions:
                actions[new_filename] = actions.pop(old_filename)
            if old_filename in offsets:
                offsets[new_filename] = offsets.pop(old_filename)
            if old_filename in delays:
                delays[new_filename] = delays.pop(old_filename)
            if old_filename in delay_types:
                delay_types[new_filename] = delay_types.pop(old_filename)
            if old_filename in rois:
                rois[new_filename] = rois.pop(old_filename)

            self._discard_location_hint(old_path, clear_full_scan_state=True)
            self._discard_location_hint(new_path, clear_full_scan_state=True)
            self._remove_count_state(old_filename, counts_key)
            self.invalidate_template_cache(old_path)
            self.invalidate_template_cache(new_path)

            self.save_config(include_templates=True)
            return True, new_filename
        except OSError as error:
            self.log(f"이름 변경 실패 ({old_filename} -> {new_filename}): {error}")
            return False, f"이름 변경 실패: {error}"

    def rename_template(self, old_filename, new_filename):
        return self._rename_template(
            old_filename,
            new_filename,
            self.template_dir,
            self.template_order,
            self.template_counts,
            self.template_actions,
            self.template_offsets,
            self.template_delays,
            self.template_delay_types,
            self.template_rois,
            "template_counts",
        )

    def rename_fallback_template(self, old_filename, new_filename):
        return self._rename_template(
            old_filename,
            new_filename,
            self.fallback_template_dir,
            self.fallback_template_order,
            self.fallback_template_counts,
            self.fallback_template_actions,
            self.fallback_template_offsets,
            self.fallback_template_delays,
            self.fallback_template_delay_types,
            self.fallback_template_rois,
            "fallback_template_counts",
        )

    def _reset_counts(self, counts_key, order, counts):
        path_key = os.path.abspath(self.config_path)
        with self._config_lock:
            try:
                self._pending_count_deltas[path_key][counts_key].clear()
                live_counts = self._live_counts[path_key][counts_key]
                live_counts.clear()
                live_counts.update({filename: 0 for filename in order})
                counts.clear()
                counts.update(live_counts)

                config = self._read_config_unlocked(self.config_path)
                config[counts_key] = dict(live_counts)
                self._atomic_write_config_unlocked(self.config_path, config)
                return True
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.log(f"카운트 초기화 실패: {error}")
                return False

    def reset_counts(self):
        if self._reset_counts(
            "template_counts", self.template_order, self.template_counts
        ):
            self.log("모든 템플릿 클릭 카운트가 초기화되었습니다.")

    def reset_fallback_counts(self):
        if self._reset_counts(
            "fallback_template_counts",
            self.fallback_template_order,
            self.fallback_template_counts,
        ):
            self.log("모든 미매칭 복구 템플릿 클릭 카운트가 초기화되었습니다.")

    @staticmethod
    def _next_action(current):
        if current == "click":
            return "double_click"
        if current in ("double_click", "click_click", "double"):
            return "back"
        return "click"

    def toggle_action(self, filename):
        self.template_actions[filename] = self._next_action(
            self.template_actions.get(filename, "click")
        )
        self.save_config(include_templates=True)
        return self.template_actions[filename]

    def toggle_fallback_action(self, filename):
        self.fallback_template_actions[filename] = self._next_action(
            self.fallback_template_actions.get(filename, "click")
        )
        self.save_config(include_templates=True)
        return self.fallback_template_actions[filename]

    def set_template_delay(self, filename, delay, delay_type="pre"):
        safe_delay = self._safe_number(delay, 0.0, minimum=0.0)
        self.template_delays[filename] = safe_delay
        self.template_delay_types[filename] = "post" if delay_type == "post" else "pre"
        self.save_config(include_templates=True)
        return safe_delay, self.template_delay_types[filename]

    def set_fallback_template_delay(self, filename, delay, delay_type="pre"):
        safe_delay = self._safe_number(delay, 0.0, minimum=0.0)
        self.fallback_template_delays[filename] = safe_delay
        self.fallback_template_delay_types[filename] = "post" if delay_type == "post" else "pre"
        self.save_config(include_templates=True)
        return safe_delay, self.fallback_template_delay_types[filename]

    def _record_match(self, filename, is_fallback=False):
        counts_key = (
            "fallback_template_counts" if is_fallback else "template_counts"
        )
        counts = (
            self.fallback_template_counts if is_fallback else self.template_counts
        )
        path_key = os.path.abspath(self.config_path)

        with self._config_lock:
            live_counts = self._live_counts[path_key][counts_key]
            current = self._safe_int(
                live_counts.get(filename, counts.get(filename, 0))
            )
            new_count = current + 1
            live_counts[filename] = new_count
            counts[filename] = new_count
            self._pending_count_deltas[path_key][counts_key][filename] += 1
            self._schedule_count_flush_unlocked(path_key)

        if self.consecutive_match_template == filename:
            self.consecutive_match_count += 1
        else:
            self.consecutive_match_template = filename
            self.consecutive_match_count = 1
            self._consecutive_alert_triggered = False

        current_state = (bool(is_fallback), filename)
        if self.last_matched_template:
            previous_state = (
                bool(self.last_matched_is_fallback),
                self.last_matched_template,
            )
            self._transition_counts[previous_state][current_state] += 1
        self.last_matched_template = filename
        self.last_matched_is_fallback = is_fallback
        self._mark_screen_dirty()

        if self.on_match_callback:
            try:
                self.on_match_callback(filename, new_count, is_fallback)
            except Exception as error:
                self.log(f"매칭 콜백 오류: {error}")

        if (
            self.consecutive_match_threshold > 0
            and self.consecutive_match_count >= self.consecutive_match_threshold
            and not self._consecutive_alert_triggered
        ):
            self._consecutive_alert_triggered = True
            if self.on_consecutive_match_callback:
                try:
                    self.on_consecutive_match_callback(
                        filename, self.consecutive_match_count
                    )
                except Exception as error:
                    self.log(f"연속 매칭 경고 콜백 오류: {error}")
        return new_count

    def get_timers_status(self):
        now = time.monotonic()
        elapsed_match = max(0.0, now - self.last_match_time)
        elapsed_action = max(0.0, now - self.last_random_click_time)
        no_match_elapsed = min(elapsed_match, elapsed_action)
        timeout_elapsed = max(0.0, now - self.last_action_time)
        return {
            "is_running": self.is_running,
            "no_match_action": self.no_match_action,
            "no_match_interval": self.no_match_interval,
            "no_match_elapsed": no_match_elapsed,
            "no_match_remaining": max(0.0, self.no_match_interval - no_match_elapsed),
            "timeout": self.no_match_timeout,
            "timeout_elapsed": timeout_elapsed,
            "timeout_remaining": max(0.0, self.no_match_timeout - timeout_elapsed),
            "last_matched_template": self.last_matched_template,
            "last_matched_is_fallback": self.last_matched_is_fallback,
            "last_match_elapsed": (
                max(0.0, now - self.last_match_time)
                if self.last_matched_template
                else None
            ),
            "consecutive_match_count": self.consecutive_match_count,
            "consecutive_match_template": self.consecutive_match_template,
            "consecutive_match_threshold": self.consecutive_match_threshold,
        }

    def _execute_action(self, action, x=None, y=None):
        if action == "back":
            return self.go_back()
        if action in ("double_click", "click_click", "double"):
            return self.double_click(x, y)
        return self.click(x, y)

    def _operation_cancelled(self):
        active_cancel = self._active_loop_cancel_event
        return (
            self._shutdown_event.is_set()
            or (
                self._automatic_loop_active
                and (
                    self._stop_event.is_set()
                    or (
                        active_cancel is not None
                        and active_cancel.is_set()
                    )
                )
            )
        )

    def _wait_after_action(self, seconds):
        seconds = max(0.0, float(seconds))
        if self._operation_cancelled():
            return False
        wait_event = (
            self._stop_event
            if self._automatic_loop_active
            else self._shutdown_event
        )
        active_cancel = (
            self._active_loop_cancel_event
            if self._automatic_loop_active
            else None
        )
        if active_cancel is None:
            return not wait_event.wait(seconds)

        deadline = time.monotonic() + seconds
        while True:
            if (
                self._shutdown_event.is_set()
                or wait_event.is_set()
                or active_cancel.is_set()
            ):
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if wait_event.wait(min(0.1, remaining)):
                return False

    def _try_fallback_templates(self, screen, now, _prescaled_screen=None):
        self._refresh_template_manifests()
        if not self._roi_full_scan_frame_active:
            self._begin_roi_full_scan_frame(screen.shape)
        scan_order = tuple(self._get_template_scan_order(
            self.fallback_template_order, is_fallback=True
        ))
        self._schedule_roi_full_scan_candidates(
            self.fallback_template_dir,
            scan_order,
            self.fallback_template_rois,
            now=now,
        )
        for filename in scan_order:
            if self._automatic_loop_active and self._stop_event.is_set():
                return False
            template_path = os.path.join(self.fallback_template_dir, filename)

            match = self.find_template(
                screen,
                template_path,
                threshold=self.similarity_threshold,
                filename=filename,
                offsets_dict=self.fallback_template_offsets,
                rois_dict=self.fallback_template_rois,
                _prescaled_screen=_prescaled_screen,
            )
            if self._automatic_loop_active and self._stop_event.is_set():
                return False
            if match is None:
                continue

            x, y, confidence = match
            action = self.fallback_template_actions.get(filename, "click")
            delay = self.fallback_template_delays.get(filename, 0.0)
            delay_type = self.fallback_template_delay_types.get(filename, "pre")
            self.last_match_time = now

            if delay > 0 and delay_type == "pre":
                self.log(
                    f"⚡ [미매칭 복구] 매칭 성공: {filename} (유사도: {confidence:.2f}) "
                    f"-> [동작 전] {delay:g}초 대기 중..."
                )
                if not self._wait_after_action(delay):
                    return False

            if self._automatic_loop_active and self._stop_event.is_set():
                return False
            success = self._execute_action(action, x, y)
            self.log(
                f"⚡ [미매칭 복구] 액션 완료: {filename} "
                f"(유사도: {confidence:.2f}, 액션: {action}, 좌표: {x},{y})"
            )
            if success:
                self.last_action_time = time.monotonic()
                self.last_random_click_time = time.monotonic()
                self._timeout_alert_active = False
                self._record_match(filename, is_fallback=True)

            post_wait = delay if (delay > 0 and delay_type == "post") else self.post_action_delay
            if post_wait > 0:
                if delay > 0 and delay_type == "post":
                    self.log(
                        f"⚡ [미매칭 복구: {filename}] [동작 후] 다음 매칭 전 {post_wait:g}초 대기 중..."
                    )
                self._wait_after_action(post_wait)
            return True
        return False

    def _handle_no_match_action(self, screen, now, _prescaled_screen=None):
        if self._automatic_loop_active and self._stop_event.is_set():
            return

        if self.no_match_action == "none" or self.no_match_interval <= 0:
            return

        elapsed_since_match = now - self.last_match_time
        elapsed_since_action = now - self.last_random_click_time
        if (
            elapsed_since_match < self.no_match_interval
            or elapsed_since_action < self.no_match_interval
        ):
            return

        height, width = screen.shape[:2]
        if self.no_match_action == "fallback_list":
            count = len(self.fallback_template_order)
            matched = False
            if count > 0:
                self.log(
                    f"⚡ [미매칭 복구] 매칭 없음 ({int(elapsed_since_match)}초 경과) "
                    f"-> 미매칭 복구 템플릿({count}개) 검사 시도"
                )
                if _prescaled_screen is None:
                    preprocess_started = time.perf_counter()
                    _prescaled_screen = cv2.resize(
                        screen,
                        None,
                        fx=PRESCALE_FACTOR,
                        fy=PRESCALE_FACTOR,
                        interpolation=cv2.INTER_AREA,
                    )
                    self._record_performance(
                        "preprocess.fallback", preprocess_started
                    )
                matched = self._try_fallback_templates(screen, now, _prescaled_screen=_prescaled_screen)
                if self._automatic_loop_active and self._stop_event.is_set():
                    return
            else:
                self.log(
                    f"⚡ [미매칭 복구] {int(elapsed_since_match)}초 경과 (등록된 미매칭 템플릿 없음)"
                )

            if not matched:
                final_action = getattr(self, "fallback_final_action", "none")
                if final_action != "none":
                    success = self._execute_final_action(screen, final_action)
                    self.last_random_click_time = time.monotonic()
                    if success and self.post_action_delay > 0:
                        self._wait_after_action(self.post_action_delay)
                else:
                    self.log(
                        "⚡ [미매칭 복구] 일치하는 미매칭 템플릿 없음 "
                        f"-> {int(self.no_match_interval)}초 후 재시도"
                    )
                    self.last_random_click_time = time.monotonic()
            return

        if self._automatic_loop_active and self._stop_event.is_set():
            return

        success = False
        if self.no_match_action == "random_click":
            min_x, max_x = int(width * 0.15), max(0, int(width * 0.85))
            min_y, max_y = int(height * 0.20), max(0, int(height * 0.80))
            x = random.randint(min(min_x, max_x), max(min_x, max_x))
            y = random.randint(min(min_y, max_y), max(min_y, max_y))
            self.log(
                f"🎲 매칭 없음 ({int(elapsed_since_match)}초 경과) "
                f"-> 화면 랜덤 클릭 수행: ({x}, {y})"
            )
            success = self.click(x, y)
        elif self.no_match_action in (
            "custom_click",
            "custom_double_click",
            "double_click",
            "click_click",
        ):
            coords = self._safe_coords(self.no_match_coords)
            x = max(0, min(width - 1, coords[0]))
            y = max(0, min(height - 1, coords[1]))
            is_double = self.no_match_action != "custom_click"
            label = "클릭클릭(2회)" if is_double else "클릭"
            self.log(
                f"🎯 매칭 없음 ({int(elapsed_since_match)}초 경과) "
                f"-> 지정 좌표 {label} 수행: ({x}, {y})"
            )
            success = (
                self.double_click(x, y)
                if is_double
                else self.click(x, y)
            )
        elif self.no_match_action == "back":
            self.log(
                f"↩️ 매칭 없음 ({int(elapsed_since_match)}초 경과) "
                "-> 뒤로가기(Back 키) 수행"
            )
            success = self.go_back()

        if success:
            self.last_random_click_time = time.monotonic()
            if self.post_action_delay > 0:
                self._wait_after_action(self.post_action_delay)

    def _execute_final_action(self, screen, final_action):
        if self._operation_cancelled():
            return False

        height, width = screen.shape[:2]
        if final_action == "random_click":
            min_x, max_x = int(width * 0.15), max(0, int(width * 0.85))
            min_y, max_y = int(height * 0.20), max(0, int(height * 0.80))
            x = random.randint(min(min_x, max_x), max(min_x, max_x))
            y = random.randint(min(min_y, max_y), max(min_y, max_y))
            self.log(
                f"⚡ [미매칭 최종 동작] 템플릿 불일치 -> 화면 랜덤 클릭 수행: ({x}, {y})"
            )
            return self.click(x, y)
        elif final_action in (
            "custom_click",
            "custom_double_click",
            "double_click",
            "click_click",
        ):
            coords = self._safe_coords(getattr(self, "fallback_final_coords", [500, 500]))
            x = max(0, min(width - 1, coords[0]))
            y = max(0, min(height - 1, coords[1]))
            is_double = final_action != "custom_click"
            label = "클릭클릭(2회)" if is_double else "클릭"
            self.log(
                f"⚡ [미매칭 최종 동작] 템플릿 불일치 -> 지정 좌표 {label} 수행: ({x}, {y})"
            )
            return (
                self.double_click(x, y)
                if is_double
                else self.click(x, y)
            )
        elif final_action == "back":
            self.log(
                "⚡ [미매칭 최종 동작] 템플릿 불일치 -> 뒤로가기(Back 키) 수행"
            )
            return self.go_back()
        return False

    def run_once(self):
        """Capture once, scan templates, and perform at most one action."""
        self._reset_roi_full_scan_frame()
        cycle_started = time.perf_counter()
        self._refresh_template_manifests()
        screen = self.capture_screen()
        if self._automatic_loop_active and self._stop_event.is_set():
            return False
        if screen is None:
            self._force_full_scan = True
            self.log("화면 캡처 실패")
            self._record_performance("scan.total", cycle_started)
            return False

        preprocess_started = time.perf_counter()
        if self.match_grayscale and screen.ndim == 3:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        if not self._should_scan_frame(screen, now):
            self._record_performance("preprocess", preprocess_started)
            if self._automatic_loop_active and self._stop_event.is_set():
                return False
            self._handle_no_match_action(screen, time.monotonic())
            self._record_performance("scan.skipped", cycle_started)
            return False

        self._begin_roi_full_scan_frame(screen.shape)
        prescaled_cache = [None]
        self._record_performance("preprocess", preprocess_started)

        def get_prescaled_screen():
            if prescaled_cache[0] is None:
                prescale_started = time.perf_counter()
                prescaled_cache[0] = cv2.resize(
                    screen,
                    None,
                    fx=PRESCALE_FACTOR,
                    fy=PRESCALE_FACTOR,
                    interpolation=cv2.INTER_AREA,
                )
                self._record_performance(
                    "preprocess.prescale", prescale_started
                )
            return prescaled_cache[0]

        scan_order = tuple(self._get_template_scan_order(
            self.template_order, is_fallback=False
        ))
        self._schedule_roi_full_scan_candidates(
            self.template_dir,
            scan_order,
            self.template_rois,
            now=now,
        )
        for filename in scan_order:
            if self._automatic_loop_active and self._stop_event.is_set():
                return False
            template_path = os.path.join(self.template_dir, filename)

            match = self.find_template(
                screen,
                template_path,
                threshold=self.similarity_threshold,
                filename=filename,
                rois_dict=self.template_rois,
                _prescaled_screen=get_prescaled_screen,
            )
            if self._automatic_loop_active and self._stop_event.is_set():
                return False
            if match is None:
                continue

            self._last_scan_had_match = True
            x, y, confidence = match
            action = self.template_actions.get(filename, "click")
            delay = self.template_delays.get(filename, 0.0)
            delay_type = self.template_delay_types.get(filename, "pre")
            self.last_match_time = time.monotonic()

            if delay > 0 and delay_type == "pre":
                self.log(
                    f"매칭 성공: {filename} (유사도: {confidence:.2f}) "
                    f"-> [동작 전] {delay:g}초 대기 중..."
                )
                if not self._wait_after_action(delay):
                    return False

            if self._automatic_loop_active and self._stop_event.is_set():
                return False
            action_started = time.perf_counter()
            success = self._execute_action(action, x, y)
            self._record_performance("action", action_started)
            self.log(
                f"액션 완료: {filename} "
                f"(유사도: {confidence:.2f}, 액션: {action}, 좌표: {x},{y})"
            )
            if success:
                self.last_action_time = time.monotonic()
                self.last_random_click_time = time.monotonic()
                self._timeout_alert_active = False
                self._record_match(filename)

            self._record_performance("scan.total", cycle_started)
            post_wait = (
                delay
                if delay > 0 and delay_type == "post"
                else self.post_action_delay
            )
            if post_wait > 0:
                if delay > 0 and delay_type == "post":
                    self.log(
                        f"⏱️ [{filename}] [동작 후] 다음 매칭 전 "
                        f"{post_wait:g}초 대기 중..."
                    )
                self._wait_after_action(post_wait)
            return True

        if self._automatic_loop_active and self._stop_event.is_set():
            return False
        self._last_scan_had_match = False
        self._handle_no_match_action(
            screen,
            time.monotonic(),
            _prescaled_screen=get_prescaled_screen,
        )
        self._record_performance("scan.total", cycle_started)
        return False

    def execute_template(self, filename, is_fallback=False):
        """Find a specific template on screen and execute its configured action once."""
        if self.device is None and not self.start_adb_server():
            self.log("디바이스 연결 대기 중... 수동 실행 불가")
            return False

        screen = self.capture_screen()
        if screen is None:
            self.log(f"[{filename}] 화면 캡처 실패")
            return False

        if self.match_grayscale and screen.ndim == 3:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        target_dir = (
            self.fallback_template_dir if is_fallback else self.template_dir
        )
        template_path = os.path.join(target_dir, filename)
        self._refresh_template_directory(target_dir)
        if not self._template_is_registered(template_path):
            self.log(f"⚠️ 템플릿 파일이 존재하지 않습니다: {filename}")
            return False

        offsets_dict = (
            self.fallback_template_offsets
            if is_fallback
            else self.template_offsets
        )
        actions_dict = (
            self.fallback_template_actions
            if is_fallback
            else self.template_actions
        )
        delays_dict = (
            self.fallback_template_delays
            if is_fallback
            else self.template_delays
        )

        rois_dict = (
            self.fallback_template_rois
            if is_fallback
            else self.template_rois
        )
        match = self.find_template(
            screen,
            template_path,
            threshold=self.similarity_threshold,
            filename=filename,
            offsets_dict=offsets_dict,
            rois_dict=rois_dict,
        )

        prefix = "⚡ [미매칭 더블클릭]" if is_fallback else "🎯 [더블클릭 실행]"
        if match is None:
            self.log(
                f"{prefix} '{filename}' 템플릿을 현재 화면에서 찾을 수 없습니다."
            )
            return False

        x, y, confidence = match
        action = actions_dict.get(filename, "click")
        delay = delays_dict.get(filename, 0.0)
        delay_types_dict = (
            self.fallback_template_delay_types
            if is_fallback
            else self.template_delay_types
        )
        delay_type = delay_types_dict.get(filename, "pre")
        now = time.monotonic()
        self.last_match_time = now

        if delay > 0 and delay_type == "pre":
            self.log(
                f"{prefix} '{filename}' 매칭 성공 -> [동작 전] {delay:g}초 대기 중..."
            )
            if not self._wait_after_action(delay):
                return False
        success = self._execute_action(action, x, y)
        self.log(
            f"{prefix} '{filename}' 액션 완료 "
            f"(유사도: {confidence:.2f}, 액션: {action}, 좌표: {x},{y})"
        )
        if success:
            self.last_action_time = time.monotonic()
            self.last_random_click_time = time.monotonic()
            self._record_match(filename, is_fallback=is_fallback)
            if delay > 0 and delay_type == "post":
                self.log(
                    f"{prefix} '{filename}' [동작 후] {delay:g}초 대기 중..."
                )
                self._wait_after_action(delay)
        return success

    @staticmethod
    def _subprocess_kwargs():
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        return {}

    def _run_adb(self, args, timeout=ADB_COMMAND_TIMEOUT, check=False, text=True):
        cmd = [self.adb_path]
        if (
            "start-server" not in args
            and self.host
            and str(self.host).strip().lower() not in ("127.0.0.1", "localhost")
        ):
            cmd.extend(["-H", str(self.host)])
        if self.port:
            cmd.extend(["-P", str(self.port)])
        cmd.extend(args)
        return subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=text,
            encoding="utf-8" if text else None,
            errors="replace" if text else None,
            timeout=timeout,
            **self._subprocess_kwargs(),
        )

    @classmethod
    def validate_adb_executable(cls, adb_path, timeout=ADB_VERSION_TIMEOUT):
        """Validate an ADB executable without starting or contacting a server."""
        raw_path = os.path.expandvars(os.path.expanduser(str(adb_path or "").strip()))
        if not raw_path:
            return False, raw_path, "ADB 실행 파일 경로가 비어 있습니다."

        resolved_path = raw_path
        if not os.path.isabs(resolved_path) and not os.path.dirname(resolved_path):
            resolved_path = shutil.which(resolved_path) or resolved_path
        resolved_path = os.path.abspath(resolved_path)
        if not os.path.isfile(resolved_path):
            return False, resolved_path, f"ADB 실행 파일을 찾을 수 없습니다: {resolved_path}"

        try:
            stat = os.stat(resolved_path)
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError as error:
            return False, resolved_path, f"ADB 실행 파일을 확인할 수 없습니다: {error}"

        cache_key = os.path.normcase(resolved_path)
        with cls._adb_validation_lock:
            cached = cls._adb_validation_cache.get(cache_key)
            if cached and cached[0] == signature:
                return cached[1], resolved_path, cached[2]

        try:
            result = subprocess.run(
                [resolved_path, "version"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                **cls._subprocess_kwargs(),
            )
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part and part.strip()
            )
            valid = result.returncode == 0 and "Android Debug Bridge" in output
            message = (
                output.splitlines()[0]
                if valid and output
                else f"유효한 adb.exe가 아닙니다 (exit={result.returncode}): "
                f"{output or '출력 없음'}"
            )
        except (OSError, subprocess.SubprocessError) as error:
            valid = False
            message = f"ADB 버전 확인 실패: {error}"

        with cls._adb_validation_lock:
            cls._adb_validation_cache[cache_key] = (signature, valid, message)
        return valid, resolved_path, message

    @classmethod
    def _get_adb_server_lock(cls, host, port):
        normalized_host = str(host or "127.0.0.1").strip().lower()
        if normalized_host in ("", "localhost"):
            normalized_host = "127.0.0.1"
        endpoint = (normalized_host, int(port or ADB_PORT))
        with cls._adb_server_locks_guard:
            return cls._adb_server_locks.setdefault(endpoint, threading.RLock())

    @staticmethod
    def _recv_adb_exact(sock, length):
        chunks = []
        remaining = int(length)
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("ADB 서버가 응답 도중 연결을 종료했습니다.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _probe_adb_server(self, timeout=ADB_SERVER_PROBE_TIMEOUT):
        """Verify that host/port speaks the ADB host protocol, not just TCP."""
        host = str(self.host or "127.0.0.1").strip()
        port = int(self.port or ADB_PORT)
        request = b"host:version"
        framed = f"{len(request):04x}".encode("ascii") + request
        try:
            with socket.create_connection((host, port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(framed)
                if self._recv_adb_exact(sock, 4) != b"OKAY":
                    return False
                length = int(self._recv_adb_exact(sock, 4), 16)
                self._recv_adb_exact(sock, length)
                return True
        except (OSError, ValueError, ConnectionError):
            return False

    @staticmethod
    def _adb_output_text(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()

    def _log_adb_stage_failure(self, stage, result=None, error=None):
        details = []
        if result is not None:
            details.append(f"exit={getattr(result, 'returncode', '?')}")
        if error is not None:
            details.append(str(error))
        source = result if result is not None else error
        for label, value in (
            ("stderr", getattr(source, "stderr", None)),
            ("stdout", getattr(source, "stdout", None)),
        ):
            output = self._adb_output_text(value)
            if output:
                details.append(f"{label}={output[:2000]}")
        self.log(f"ADB {stage} 실패: {' | '.join(details) or '원인 정보 없음'}")

    def _run_adb_stage(self, stage, args, timeout=ADB_COMMAND_TIMEOUT):
        try:
            result = self._run_adb(args, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            self._log_adb_stage_failure(stage, error=error)
            return None
        if result.returncode != 0:
            self._log_adb_stage_failure(stage, result=result)
            return None
        return result

    @staticmethod
    def _adb_connect_output_failed(result):
        output = " ".join(
            str(part or "")
            for part in (getattr(result, "stdout", ""), getattr(result, "stderr", ""))
        ).lower()
        return any(
            marker in output
            for marker in (
                "failed to connect",
                "cannot connect",
                "connection refused",
                "연결할 수",
            )
        )

    def _ensure_adb_server(self):
        """Start the process-wide ADB server once and recover cold-start races."""
        path_requires_validation = (
            self.adb_mode == "custom"
            or os.path.isabs(str(self.adb_path))
            or bool(os.path.dirname(str(self.adb_path)))
        )
        if path_requires_validation:
            valid, resolved_path, message = self.validate_adb_executable(self.adb_path)
            if not valid:
                self.log(f"ADB 실행 파일 검증 실패: {message}")
                return False
            self.adb_path = resolved_path

        host = str(self.host or "127.0.0.1").strip().lower()
        server_lock = self._get_adb_server_lock(host, self.port)
        with server_lock:
            if self._operation_cancelled():
                return False
            if self._probe_adb_server():
                return True
            if host not in ("127.0.0.1", "localhost"):
                self.log(f"원격 ADB 서버에 연결할 수 없습니다: {self.host}:{self.port}")
                return False

            for attempt in range(1, ADB_SERVER_START_ATTEMPTS + 1):
                if self._operation_cancelled():
                    return False
                self.log(
                    f"ADB 서버 시작 시도 ({attempt}/{ADB_SERVER_START_ATTEMPTS}): "
                    f"{self.adb_path}"
                )
                try:
                    result = self._run_adb(["start-server"], check=False)
                except (OSError, subprocess.SubprocessError) as error:
                    result = None
                    self._log_adb_stage_failure("서버 시작", error=error)
                else:
                    if result.returncode != 0:
                        self._log_adb_stage_failure("서버 시작", result=result)

                if self._operation_cancelled():
                    return False
                start_command_succeeded = (
                    result is not None and result.returncode == 0
                )
                if start_command_succeeded and self._probe_adb_server():
                    return True

                delay = ADB_SERVER_RETRY_DELAYS[
                    min(attempt - 1, len(ADB_SERVER_RETRY_DELAYS) - 1)
                ]
                time.sleep(delay)
                if self._probe_adb_server():
                    if start_command_succeeded:
                        self.log("시작한 ADB 서버의 프로토콜 응답을 확인했습니다.")
                    else:
                        self.log("다른 스레드/프로세스가 시작한 ADB 서버를 확인했습니다.")
                    return True

            self.log("ADB 서버 시작 재시도 한도를 초과했습니다.")
            return False

    @staticmethod
    def normalize_device_serials(devices):
        """
        Deduplicate device serials.
        If an IP:Port and an emulator-XXXX mapping to the same port both exist,
        prioritize the IP:Port format and drop the redundant emulator-XXXX.
        If only emulator-XXXX exists, keep it as reported by ADB without guessing.
        """
        ip_devices = []
        emulator_devices = []
        other_devices = []

        for dev in devices:
            dev_str = str(dev).strip()
            if not dev_str:
                continue
            if ":" in dev_str:
                ip_devices.append(dev_str)
            elif dev_str.startswith("emulator-"):
                emulator_devices.append(dev_str)
            else:
                other_devices.append(dev_str)

        # Extract all ports already covered by IP devices
        covered_ports = set()
        for dev in ip_devices:
            try:
                port = int(dev.split(":")[-1])
                covered_ports.add(port)
            except ValueError:
                pass

        final_devices = list(ip_devices)

        # Drop emulator-N only if its corresponding IP port is already covered
        for emu_name in emulator_devices:
            is_covered = False
            try:
                num = int(emu_name.split("-")[1])
                if (num + 1) in covered_ports:
                    is_covered = True
            except (IndexError, ValueError):
                pass

            if not is_covered and emu_name not in final_devices:
                final_devices.append(emu_name)

        for dev in other_devices:
            if dev not in final_devices:
                final_devices.append(dev)

        return list(dict.fromkeys(final_devices))

    def get_connected_devices(self):
        """Return online ADB serials after fast, socket-probed parallel emulator connection."""
        if not self._ensure_adb_server():
            return []

        # Comprehensive emulator ports (LDPlayer, BlueStacks, Nox, MuMu, MEmu, etc.)
        emulator_ports = (
            5555, 5557, 5559, 5561, 5563, 5565, 5567, 5569, 5571, 5573, 5575, 5577, 5579, 5581, 5583, 5585,
            5595, 5605, 5615, 5625, 5635, 5645, 5655,
            62001, 62025, 62026, 62027, 62028, 62029, 62030, 62031, 62032,
            7555, 16384, 16416, 16448, 16480, 16512,
            21503, 21513, 21523, 21533,
        )

        def is_port_open(port):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    return True
            except (OSError, socket.error):
                return False

        # 1. Fast parallel TCP socket probe (< 20ms)
        with ThreadPoolExecutor(max_workers=min(32, len(emulator_ports))) as executor:
            port_status = list(executor.map(lambda p: (p, is_port_open(p)), emulator_ports))

        open_ports = [p for p, is_open in port_status if is_open]

        # 2. Connect only to active listening ports
        def connect_port(port):
            address = f"127.0.0.1:{port}"
            result = self._run_adb_stage(
                f"디바이스 연결 ({address})",
                ["connect", address],
                timeout=1.0,
            )
            if result is not None and self._adb_connect_output_failed(result):
                output = self._adb_output_text(result.stdout or result.stderr)
                self.log(f"ADB 디바이스 연결 실패 ({address}): {output}")

        if open_ports:
            with ThreadPoolExecutor(max_workers=min(16, len(open_ports))) as executor:
                list(executor.map(connect_port, open_ports))

        result = self._run_adb_stage("디바이스 목록 조회", ["devices"])
        if result is None:
            return []

        devices = []
        for line in result.stdout.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith("*") or line_str.startswith("List of devices"):
                continue
            parts = line_str.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
            elif len(parts) >= 1 and "device" in parts[1:]:
                devices.append(parts[0])
        return self.normalize_device_serials(devices)

    def start_adb_server(self):
        """Start ADB and bind only to a device confirmed by adb devices."""
        with self._device_lock:
            self.log(f"ADB 연결 준비: {self.adb_path}")
            try:
                if not self._ensure_adb_server():
                    self.client = None
                    self.device = None
                    return False
                if self._operation_cancelled():
                    return False
                if ":" in self.device_address:
                    connect_result = self._run_adb_stage(
                        f"디바이스 연결 ({self.device_address})",
                        ["connect", self.device_address],
                        timeout=3.0,
                    )
                    if (
                        connect_result is not None
                        and self._adb_connect_output_failed(connect_result)
                    ):
                        output = self._adb_output_text(
                            connect_result.stdout or connect_result.stderr
                        )
                        self.log(
                            f"ADB 디바이스 연결 실패 ({self.device_address}): {output}"
                        )
                    if self._operation_cancelled():
                        return False

                result = self._run_adb_stage("디바이스 목록 조회", ["devices"])
                if result is None:
                    return False
                if self._operation_cancelled():
                    return False
                online_serials = {
                    parts[0]
                    for line in result.stdout.splitlines()
                    for line_str in (line.strip(),)
                    if line_str and not line_str.startswith("*") and not line_str.startswith("List of devices")
                    for parts in (line_str.split(),)
                    if len(parts) >= 2 and parts[1] == "device"
                }
                target_serial = (
                    self.device_address
                    if self.device_address in online_serials
                    else None
                )
                if target_serial is None and ":" in self.device_address:
                    try:
                        port = int(self.device_address.split(":")[-1])
                        emu_serial = f"emulator-{port - 1}"
                        if emu_serial in online_serials:
                            target_serial = emu_serial
                    except ValueError:
                        pass

                client = AdbClient(host=self.host, port=self.port)
                target = (
                    AdbDevice(client, target_serial)
                    if target_serial is not None
                    else None
                )

                self.client = client
                self.device = target
                if target is None:
                    self.log(
                        f"장치를 찾을 수 없습니다: {self.device_address}"
                    )
                    return False

                self._next_reconnect_at = 0.0
                self._reconnect_delay = 1.0
                self._loop_wake_event.set()
                self.log(f"장치 연결 성공: {self.device_address}")
                return True
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                self.client = None
                self.device = None
                self.log(f"ADB 연결 처리 중 예기치 않은 오류: {error}")
                return False
            except Exception as error:
                self.client = None
                self.device = None
                self.log(f"ADB 클라이언트 초기화/장치 확인 오류: {error}")
                return False

    def disconnect(self):
        self.stop_loop()
        with self._device_lock:
            self.device = None
            self.client = None

    @staticmethod
    def _decode_png_screencap(payload, use_grayscale):
        if not payload:
            return None
        read_flag = cv2.IMREAD_GRAYSCALE if use_grayscale else cv2.IMREAD_COLOR
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), read_flag)

    @staticmethod
    def _decode_raw_screencap(payload, use_grayscale):
        """Decode Android screencap raw output with 12- or 16-byte headers."""
        if not payload or len(payload) < 12:
            return None
        try:
            width, height, pixel_format = struct.unpack_from("<III", payload, 0)
        except struct.error:
            return None
        if not (0 < width <= 16384 and 0 < height <= 16384):
            return None
        bytes_per_pixel = {1: 4, 2: 4, 3: 3, 4: 2, 5: 4}.get(pixel_format)
        if bytes_per_pixel is None:
            return None

        minimum_row_bytes = width * bytes_per_pixel
        candidates = []
        for header_size in (12, 16):
            remaining = len(payload) - header_size
            if remaining < minimum_row_bytes * height or remaining % height:
                continue
            row_bytes = remaining // height
            if minimum_row_bytes <= row_bytes <= minimum_row_bytes + 16384:
                candidates.append((row_bytes, header_size))
        if not candidates:
            return None
        row_bytes, header_size = min(candidates)
        flat = np.frombuffer(
            payload,
            dtype=np.uint8,
            count=row_bytes * height,
            offset=header_size,
        ).reshape(height, row_bytes)
        pixels = flat[:, :minimum_row_bytes].reshape(
            height, width, bytes_per_pixel
        )
        if pixel_format in (1, 2):
            code = cv2.COLOR_RGBA2GRAY if use_grayscale else cv2.COLOR_RGBA2BGR
        elif pixel_format == 3:
            code = cv2.COLOR_RGB2GRAY if use_grayscale else cv2.COLOR_RGB2BGR
        elif pixel_format == 4:
            code = cv2.COLOR_BGR5652GRAY if use_grayscale else cv2.COLOR_BGR5652BGR
        else:
            code = cv2.COLOR_BGRA2GRAY if use_grayscale else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(pixels, code)

    @staticmethod
    def _is_local_device_serial(serial):
        serial = str(serial).strip().lower()
        return (
            serial.startswith("emulator-")
            or serial.startswith("127.0.0.1:")
            or serial.startswith("localhost:")
        )

    def _capture_direct_socket(self, serial, backend, use_grayscale):
        """High-performance direct TCP socket transport to ADB server (zero subprocess creation)."""
        host = str(self.host).strip() if self.host else "127.0.0.1"
        port = int(self.port) if self.port else 5037
        started_at = time.perf_counter()
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(ADB_COMMAND_TIMEOUT)
            sock.connect((host, port))

            # 1. Transport to device serial
            transport_cmd = f"host:transport:{serial}"
            sock.sendall(f"{len(transport_cmd):04x}{transport_cmd}".encode("ascii"))
            status = sock.recv(4)
            if status != b"OKAY":
                return None, "transport"

            # 2. Request exec screencap
            exec_cmd = "exec:screencap" if backend == "raw" else "exec:screencap -p"
            sock.sendall(f"{len(exec_cmd):04x}{exec_cmd}".encode("ascii"))
            status = sock.recv(4)
            if status != b"OKAY":
                return None, "transport"

            # 3. Read stream bytes directly
            chunks = []
            while True:
                try:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                except socket.timeout:
                    break
            payload = b"".join(chunks)
        except (OSError, socket.error):
            self._record_performance("capture.socket", started_at)
            return None, "transport"
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        self._record_performance("capture.socket", started_at)
        if not payload:
            return None, "transport"

        decode_started = time.perf_counter()
        if backend == "raw":
            image = self._decode_raw_screencap(payload, use_grayscale)
        else:
            image = self._decode_png_screencap(payload, use_grayscale)
        self._record_performance("capture.decode", decode_started)
        return image, None if image is not None else "decode"

    def _capture_exec_backend(self, serial, backend, use_grayscale):
        command = [self.adb_path]
        if self.host and str(self.host).strip() not in ("127.0.0.1", "localhost"):
            command.extend(["-H", str(self.host)])
        if self.port:
            command.extend(["-P", str(self.port)])
        command.extend(["-s", serial, "exec-out", "screencap"])
        if backend == "png":
            command.append("-p")
        started_at = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                timeout=ADB_COMMAND_TIMEOUT,
                **self._subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            self._record_performance("capture.exec", started_at)
            return None, "transport"
        self._record_performance("capture.exec", started_at)
        if proc.returncode != 0 or not proc.stdout:
            return None, "transport"

        decode_started = time.perf_counter()
        if backend == "raw":
            image = self._decode_raw_screencap(proc.stdout, use_grayscale)
        else:
            image = self._decode_png_screencap(proc.stdout, use_grayscale)
        self._record_performance("capture.decode", decode_started)
        return image, None if image is not None else "decode"

    def _capture_ppadb_backend(self, device, use_grayscale):
        """Use ppadb transport with a real socket timeout for PNG fallback."""
        connection = device.create_connection(timeout=ADB_COMMAND_TIMEOUT)
        with connection:
            connection.send("shell:/system/bin/screencap -p")
            payload = bytes(connection.read_all())
        if payload and len(payload) > 5 and payload[5] == 0x0D:
            payload = payload.replace(b"\r\n", b"\n")
        return self._decode_png_screencap(payload, use_grayscale)

    def capture_screen(self, grayscale=None):
        """Capture with local raw fast path, PNG fallback and backend cooldown."""
        if self._operation_cancelled():
            return None

        total_started = time.perf_counter()
        now = time.monotonic()
        if self.device is None:
            if now < self._next_reconnect_at:
                return None
            if not self.start_adb_server():
                self._next_reconnect_at = time.monotonic() + self._reconnect_delay
                self._reconnect_delay = min(10.0, self._reconnect_delay * 2.0)
                return None
            if self._operation_cancelled():
                return None
            self._next_reconnect_at = 0.0
            self._reconnect_delay = 1.0

        with self._device_lock:
            device = self.device
        if device is None:
            return None

        configured_serial = self.device_address
        device_serial = getattr(device, "serial", None)
        serial = (
            device_serial.strip()
            if isinstance(device_serial, str) and device_serial.strip()
            else configured_serial
        )
        use_grayscale = (
            self.match_grayscale if grayscale is None else bool(grayscale)
        )
        image = None
        failure_kind = None

        exec_attempted = False
        if now >= self._exec_out_disabled_until:
            exec_attempted = True
            backends = []
            wants_raw = self.capture_backend == "raw" or (
                self.capture_backend == "auto"
                and self._is_local_device_serial(serial)
            )
            if wants_raw and now >= self._raw_capture_disabled_until:
                backends.append("raw")
            backends.append("png")
            for backend in dict.fromkeys(backends):
                if self._operation_cancelled():
                    return None
                # 1. High-speed Direct TCP Socket Transport (Zero process creation)
                image, failure_kind = self._capture_direct_socket(
                    serial, backend, use_grayscale
                )
                # 2. Subprocess CLI fallback if direct socket transport failed
                if image is None and failure_kind == "transport":
                    image, failure_kind = self._capture_exec_backend(
                        serial, backend, use_grayscale
                    )
                if self._operation_cancelled():
                    return None
                if image is not None:
                    self._exec_out_failure_count = 0
                    self._exec_out_disabled_until = 0.0
                    break
                if backend == "raw" and failure_kind == "decode":
                    self._raw_capture_disabled_until = time.monotonic() + 60.0
                    continue
                if failure_kind == "transport":
                    break

        if image is None:
            if exec_attempted:
                self._exec_out_failure_count += 1
                cooldown = min(30.0, 2.0 ** min(self._exec_out_failure_count, 4))
                self._exec_out_disabled_until = time.monotonic() + cooldown
            if self._operation_cancelled():
                return None
            fallback_started = time.perf_counter()
            try:
                image = self._capture_ppadb_backend(device, use_grayscale)
            except Exception:
                image = None
            self._record_performance("capture.ppadb", fallback_started)

        if self._operation_cancelled():
            return None
        self._record_performance("capture.total", total_started)
        if image is not None:
            return image

        self.log("화면 캡처 실패: exec-out 및 ADB fallback 모두 실패")
        with self._device_lock:
            if self.device is device:
                self.device = None
        self._next_reconnect_at = time.monotonic() + self._reconnect_delay
        self._reconnect_delay = min(10.0, self._reconnect_delay * 2.0)
        return None
    def _load_template(self, template_path, grayscale=None):
        absolute_path = os.path.abspath(template_path)
        manifest = self._refresh_template_directory(
            os.path.dirname(absolute_path)
        )
        if absolute_path not in manifest:
            return None, None

        use_grayscale = (
            self.match_grayscale if grayscale is None else bool(grayscale)
        )
        cache_key = (absolute_path, use_grayscale)
        generation = self._template_file_generations.get(absolute_path)

        # Lock-free hot path: compare only in-memory generations.
        cached = self._template_cache.get(cache_key)
        if (
            cached is not None
            and generation is not None
            and cached[0] == generation
        ):
            return cached[1], cached[2]

        template, method = self._load_template_direct(
            absolute_path, grayscale=use_grayscale
        )
        if template is None and absolute_path in manifest:
            self.log(f"템플릿 로드 실패 ({template_path})")
        return template, method

    @classmethod
    def _get_scaled_template(cls, cache_key, template):
        """Return a coarse template tied to the exact decoded source object."""
        with cls._template_cache_lock:
            cached = cls._scaled_template_cache.get(cache_key)
            if cached is not None and cached[0] is template:
                return cached[1]

        height, width = template.shape[:2]
        if height < PRESCALE_MIN_TEMPLATE_DIM or width < PRESCALE_MIN_TEMPLATE_DIM:
            scaled = None
        else:
            scaled = cv2.resize(
                template,
                None,
                fx=PRESCALE_FACTOR,
                fy=PRESCALE_FACTOR,
                interpolation=cv2.INTER_AREA,
            )
        with cls._template_cache_lock:
            cached = cls._scaled_template_cache.get(cache_key)
            if cached is not None and cached[0] is template:
                return cached[1]

            current = cls._template_cache.get(cache_key)
            if current is not None and current[1] is not template:
                return scaled

            cls._scaled_template_cache[cache_key] = (template, scaled)
            return scaled

    @staticmethod
    def _score_location(method, min_value, max_value, min_location, max_location):
        if method == cv2.TM_SQDIFF_NORMED:
            return 1.0 - min_value, min_location
        return max_value, max_location

    def _run_match_map(self, image, template, method, stage):
        started_at = time.perf_counter()
        result = cv2.matchTemplate(image, template, method)
        values = cv2.minMaxLoc(result)
        self._record_performance(stage, started_at)
        confidence, location = self._score_location(method, *values)
        return result, float(confidence), location

    @staticmethod
    def _coarse_candidates(result, template_shape, method, threshold, limit):
        """Extract separated coarse peaks in descending confidence order."""
        candidates = []
        template_height, template_width = template_shape[:2]
        suppress_x = max(1, template_width // 2)
        suppress_y = max(1, template_height // 2)
        for _ in range(max(1, int(limit))):
            values = cv2.minMaxLoc(result)
            confidence, location = AutoClicker._score_location(method, *values)
            if not math.isfinite(confidence) or confidence < threshold:
                break
            candidates.append((location, float(confidence)))
            x, y = location
            x0 = max(0, x - suppress_x)
            y0 = max(0, y - suppress_y)
            x1 = min(result.shape[1], x + suppress_x + 1)
            y1 = min(result.shape[0], y + suppress_y + 1)
            result[y0:y1, x0:x1] = (
                1.0 if method == cv2.TM_SQDIFF_NORMED else -1.0
            )
        return candidates

    def _match_region(self, screen, template, method, threshold, bounds, stage):
        left, top, right, bottom = bounds
        region = screen[top:bottom, left:right]
        template_height, template_width = template.shape[:2]
        if (
            region.shape[0] < template_height
            or region.shape[1] < template_width
        ):
            return None
        _, confidence, location = self._run_match_map(
            region, template, method, stage
        )
        if not math.isfinite(confidence) or confidence < threshold:
            return None
        return left + location[0], top + location[1], confidence

    @staticmethod
    def _normalized_roi_bounds(bounds, screen_shape, template_shape):
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            return None
        screen_height, screen_width = screen_shape[:2]
        template_height, template_width = template_shape[:2]
        left = max(0, min(screen_width, int(math.floor(bounds[0] * screen_width))))
        top = max(0, min(screen_height, int(math.floor(bounds[1] * screen_height))))
        right = max(0, min(screen_width, int(math.ceil(bounds[2] * screen_width))))
        bottom = max(0, min(screen_height, int(math.ceil(bounds[3] * screen_height))))
        if right - left < template_width or bottom - top < template_height:
            return None
        return left, top, right, bottom

    def _reset_roi_full_scan_frame(self):
        with self._hint_lock:
            self._roi_full_scan_allowed.clear()
            self._roi_full_scan_budget_remaining = 0
            self._roi_full_scan_frame_active = False

    def _begin_roi_full_scan_frame(self, screen_shape):
        screen_shape = tuple(screen_shape[:2])
        with self._hint_lock:
            if (
                self._roi_screen_shape is not None
                and self._roi_screen_shape != screen_shape
            ):
                self._template_location_hints.clear()
                self._template_hint_hits.clear()
                self._roi_full_scan_times.clear()
                self._roi_full_scan_reservation_times.clear()
                self._roi_full_scan_round_robin.clear()
            self._roi_screen_shape = screen_shape
            self._roi_full_scan_allowed.clear()
            self._roi_full_scan_budget_remaining = max(
                1, int(self.roi_full_scan_budget)
            )
            self._roi_full_scan_frame_active = True

    def _schedule_roi_full_scan_candidates(
        self, directory, scan_order, rois_dict, now=None
    ):
        if (
            not self.roi_fullscreen_fallback
            or not isinstance(rois_dict, dict)
        ):
            return ()

        with self._hint_lock:
            if (
                not self._roi_full_scan_frame_active
                or self._roi_full_scan_budget_remaining <= 0
            ):
                self._roi_full_scan_allowed.clear()
                return ()
            slots = self._roi_full_scan_budget_remaining

        priority_paths = []
        tail_paths = []
        for index, filename in enumerate(tuple(scan_order)):
            if filename not in rois_dict:
                continue
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            if index < ROI_PRIORITY_TEMPLATE_COUNT:
                priority_paths.append(absolute_path)
            else:
                tail_paths.append(absolute_path)

        if not priority_paths and not tail_paths:
            with self._hint_lock:
                self._roi_full_scan_allowed.clear()
            return ()

        now = time.monotonic() if now is None else now
        interval = max(0.0, float(self.force_scan_interval))
        selected = []

        with self._hint_lock:
            self._roi_full_scan_allowed.clear()

            def is_due(path):
                last_scan = self._roi_full_scan_reservation_times.get(path)
                return last_scan is None or now - last_scan >= interval

            for path in priority_paths:
                if len(selected) >= slots:
                    break
                if is_due(path):
                    selected.append(path)

            remaining = slots - len(selected)
            if remaining > 0 and tail_paths:
                cursor_key = os.path.normcase(os.path.abspath(directory))
                cursor = (
                    self._roi_full_scan_round_robin.get(cursor_key, 0)
                    % len(tail_paths)
                )
                last_index = None
                for offset in range(len(tail_paths)):
                    index = (cursor + offset) % len(tail_paths)
                    path = tail_paths[index]
                    if not is_due(path):
                        continue
                    selected.append(path)
                    last_index = index
                    if len(selected) >= slots:
                        break
                if last_index is None:
                    self._roi_full_scan_round_robin[cursor_key] = (
                        cursor + 1
                    ) % len(tail_paths)
                else:
                    self._roi_full_scan_round_robin[cursor_key] = (
                        last_index + 1
                    ) % len(tail_paths)

            for path in selected:
                self._roi_full_scan_reservation_times[path] = now
            self._roi_full_scan_allowed.update(selected)

        return tuple(selected)

    def _discard_location_hint(
        self, template_path, clear_full_scan_state=False
    ):
        absolute_path = os.path.abspath(template_path)
        with self._hint_lock:
            self._template_location_hints.pop(absolute_path, None)
            self._template_hint_hits.pop(absolute_path, None)
            if clear_full_scan_state:
                self._roi_full_scan_times.pop(absolute_path, None)
                self._roi_full_scan_reservation_times.pop(absolute_path, None)
                self._roi_full_scan_allowed.discard(absolute_path)

    def _remember_location_hint(
        self, template_path, screen_shape, left, top, width, height, hint_hit
    ):
        absolute_path = os.path.abspath(template_path)
        with self._hint_lock:
            self._template_location_hints[absolute_path] = (
                tuple(screen_shape[:2]), left, top, width, height
            )
            if hint_hit:
                self._template_hint_hits[absolute_path] += 1
            else:
                self._template_hint_hits[absolute_path] = 0

    def _roi_full_scan_due(self, template_path):
        absolute_path = os.path.abspath(template_path)
        with self._hint_lock:
            if self._roi_full_scan_frame_active:
                if (
                    absolute_path not in self._roi_full_scan_allowed
                    or self._roi_full_scan_budget_remaining <= 0
                ):
                    return False
                self._roi_full_scan_allowed.remove(absolute_path)
                self._roi_full_scan_budget_remaining -= 1
                self._roi_full_scan_times[absolute_path] = time.monotonic()
                return True
        if not self._automatic_loop_active or not self.frame_change_detection:
            return True
        now = time.monotonic()
        with self._hint_lock:
            last_scan = self._roi_full_scan_times.get(absolute_path, 0.0)
            if now - last_scan < self.force_scan_interval:
                return False
            self._roi_full_scan_times[absolute_path] = now
        return True

    def _hint_bounds(self, template_path, screen_shape):
        absolute_path = os.path.abspath(template_path)
        with self._hint_lock:
            if (
                self._template_hint_hits.get(absolute_path, 0)
                >= HINT_FULL_SCAN_INTERVAL - 1
            ):
                return None
            hint = self._template_location_hints.get(absolute_path)
        if not hint:
            return None
        saved_shape, left, top, width, height = hint
        if tuple(saved_shape) != tuple(screen_shape[:2]):
            self._discard_location_hint(absolute_path)
            return None
        screen_height, screen_width = screen_shape[:2]
        margin = max(
            self.local_verify_margin,
            min(96, max(width, height) // 3),
        )
        return (
            max(0, left - margin),
            max(0, top - margin),
            min(screen_width, left + width + margin),
            min(screen_height, top + height + margin),
        )

    def _find_in_region(
        self,
        screen,
        template,
        method,
        threshold,
        cache_key,
        bounds,
        prescaled_screen=None,
        allow_full_fallback=True,
    ):
        left, top, right, bottom = bounds
        template_height, template_width = template.shape[:2]
        region_height = bottom - top
        region_width = right - left
        if region_height < template_height or region_width < template_width:
            return None

        small_template = self._get_scaled_template(cache_key, template)
        if small_template is None:
            return self._match_region(
                screen, template, method, threshold, bounds, "match.full"
            )

        is_full_screen = (
            left == 0
            and top == 0
            and right == screen.shape[1]
            and bottom == screen.shape[0]
        )
        if is_full_screen and prescaled_screen is not None:
            small_screen = (
                prescaled_screen()
                if callable(prescaled_screen)
                else prescaled_screen
            )
        else:
            small_screen = cv2.resize(
                screen[top:bottom, left:right],
                None,
                fx=PRESCALE_FACTOR,
                fy=PRESCALE_FACTOR,
                interpolation=cv2.INTER_AREA,
            )

        if (
            small_screen.shape[0] < small_template.shape[0]
            or small_screen.shape[1] < small_template.shape[1]
        ):
            return None

        coarse_result, coarse_confidence, _ = self._run_match_map(
            small_screen, small_template, method, "match.coarse"
        )
        coarse_threshold = max(0.0, threshold * 0.8)
        if (
            not math.isfinite(coarse_confidence)
            or coarse_confidence < coarse_threshold
        ):
            return None

        if self.local_verify:
            candidates = self._coarse_candidates(
                coarse_result,
                small_template.shape,
                method,
                coarse_threshold,
                self.local_verify_top_k,
            )
            scale_x = region_width / float(small_screen.shape[1])
            scale_y = region_height / float(small_screen.shape[0])
            best_match = None
            margin = self.local_verify_margin
            for location, _ in candidates:
                estimated_left = left + int(round(location[0] * scale_x))
                estimated_top = top + int(round(location[1] * scale_y))
                candidate_bounds = (
                    max(left, estimated_left - margin),
                    max(top, estimated_top - margin),
                    min(right, estimated_left + template_width + margin),
                    min(bottom, estimated_top + template_height + margin),
                )
                match = self._match_region(
                    screen,
                    template,
                    method,
                    threshold,
                    candidate_bounds,
                    "match.local",
                )
                if match is not None and (
                    best_match is None or match[2] > best_match[2]
                ):
                    best_match = match
            if best_match is not None:
                return best_match

        if allow_full_fallback:
            return self._match_region(
                screen, template, method, threshold, bounds, "match.full"
            )
        return None

    def find_template(
        self,
        screen_img,
        template_path,
        threshold=None,
        filename=None,
        offsets_dict=None,
        rois_dict=None,
        _prescaled_screen=None,
    ):
        """Find a cached template using hints, ROI, coarse peaks and local verify."""
        if screen_img is None or not isinstance(screen_img, np.ndarray):
            return None
        use_grayscale = bool(self.match_grayscale)
        if use_grayscale:
            if screen_img.ndim == 3 and screen_img.shape[2] == 3:
                screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
            elif screen_img.ndim != 2:
                return None
        elif screen_img.ndim != 3 or screen_img.shape[2] != 3:
            return None

        threshold = self._safe_number(
            self.similarity_threshold if threshold is None else threshold,
            self.similarity_threshold,
            minimum=0.0,
            maximum=1.0,
        )
        template, method = self._load_template(
            template_path, grayscale=use_grayscale
        )
        if template is None:
            return None
        template_height, template_width = template.shape[:2]
        screen_height, screen_width = screen_img.shape[:2]
        if screen_height < template_height or screen_width < template_width:
            return None

        absolute_path = os.path.abspath(template_path)
        cache_key = (absolute_path, use_grayscale)
        full_bounds = (0, 0, screen_width, screen_height)
        match = None
        hint_hit = False
        target_rois = (
            rois_dict if isinstance(rois_dict, dict) else self.template_rois
        )
        has_configured_roi = bool(
            filename and filename in target_rois
        )
        configured_bounds = None
        if has_configured_roi:
            configured_bounds = self._normalized_roi_bounds(
                target_rois[filename], screen_img.shape, template.shape
            )
            if configured_bounds is None and not self.roi_fullscreen_fallback:
                return None
        try:
            if self.dynamic_roi:
                hint_bounds = self._hint_bounds(absolute_path, screen_img.shape)
                if (
                    hint_bounds is not None
                    and configured_bounds is not None
                    and not self.roi_fullscreen_fallback
                ):
                    hint_bounds = (
                        max(hint_bounds[0], configured_bounds[0]),
                        max(hint_bounds[1], configured_bounds[1]),
                        min(hint_bounds[2], configured_bounds[2]),
                        min(hint_bounds[3], configured_bounds[3]),
                    )
                    if (
                        hint_bounds[2] - hint_bounds[0] < template_width
                        or hint_bounds[3] - hint_bounds[1] < template_height
                    ):
                        hint_bounds = None
                if hint_bounds is not None:
                    match = self._match_region(
                        screen_img,
                        template,
                        method,
                        threshold,
                        hint_bounds,
                        "match.hint",
                    )
                    if match is not None:
                        hint_hit = True
                    if match is None:
                        self._discard_location_hint(absolute_path)

            if match is None and configured_bounds is not None:
                match = self._find_in_region(
                    screen_img,
                    template,
                    method,
                    threshold,
                    cache_key,
                    configured_bounds,
                    prescaled_screen=_prescaled_screen,
                    allow_full_fallback=True,
                )

            needs_full_search = (
                not has_configured_roi
                or configured_bounds != full_bounds
            )
            if (
                match is None
                and needs_full_search
                and (
                    not has_configured_roi
                    or self.roi_fullscreen_fallback
                    and self._roi_full_scan_due(absolute_path)
                )
            ):
                match = self._find_in_region(
                    screen_img,
                    template,
                    method,
                    threshold,
                    cache_key,
                    full_bounds,
                    prescaled_screen=_prescaled_screen,
                    allow_full_fallback=True,
                )
            if match is None:
                return None

            top_left_x, top_left_y, confidence = match
            self._remember_location_hint(
                absolute_path,
                screen_img.shape,
                top_left_x,
                top_left_y,
                template_width,
                template_height,
                hint_hit,
            )

            target_offsets = (
                offsets_dict
                if isinstance(offsets_dict, dict)
                else self.template_offsets
            )
            offsets = target_offsets.get(filename) if filename else None
            if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
                try:
                    click_x = top_left_x + int(offsets[0])
                    click_y = top_left_y + int(offsets[1])
                except (TypeError, ValueError):
                    click_x = top_left_x + template_width // 2
                    click_y = top_left_y + template_height // 2
            else:
                click_x = top_left_x + template_width // 2
                click_y = top_left_y + template_height // 2

            click_x = max(0, min(screen_width - 1, click_x))
            click_y = max(0, min(screen_height - 1, click_y))
            return click_x, click_y, float(confidence)
        except cv2.error as error:
            self.log(f"템플릿 매칭 중 OpenCV 오류 ({filename}): {error}")
            return None

    def click(self, x, y):
        with self._device_lock:
            device = self.device
        if (
            device is None
            or self._operation_cancelled()
        ):
            return False
        try:
            x, y = int(x), int(y)
            device.shell(f"input tap {x} {y}", timeout=ADB_COMMAND_TIMEOUT)
            self._mark_screen_dirty()
            self.log(f"클릭 수행: ({x}, {y})")
            return True
        except Exception as error:
            self.log(f"클릭 중 오류: {error}")
            with self._device_lock:
                if self.device is device:
                    self.device = None
            return False

    def double_click(self, x, y, delay=None):
        if delay is None:
            delay = self.double_click_interval
        if not self.click(x, y):
            return False
        if not self._wait_after_action(max(0.0, float(delay))):
            return False
        return self.click(x, y)

    def go_back(self):
        with self._device_lock:
            device = self.device
        if (
            device is None
            or self._operation_cancelled()
        ):
            return False
        try:
            device.shell("input keyevent 4", timeout=ADB_COMMAND_TIMEOUT)
            self._mark_screen_dirty()
            self.log("뒤로가기(Back 키) 수행")
            return True
        except Exception as error:
            self.log(f"뒤로가기 중 오류: {error}")
            with self._device_lock:
                if self.device is device:
                    self.device = None
            return False

    def start_loop(self, interval=None, cancel_event=None):
        """Run until stop_loop is called, recovering from individual scan errors."""
        if interval is not None:
            self.scan_interval = self._safe_number(
                interval, self.scan_interval, minimum=0.1
            )

        with self._loop_state_lock:
            if (
                self._shutdown_event.is_set()
                or cancel_event is not None and cancel_event.is_set()
                or self._loop_worker_active
            ):
                return
            self._loop_worker_active = True
            self._active_loop_cancel_event = cancel_event
            self.is_running = True
            self._automatic_loop_active = True
            self._stop_event.clear()
            self._loop_wake_event.clear()

        def finish_loop_state():
            with self._loop_state_lock:
                self.is_running = False
                self._automatic_loop_active = False
                self._loop_worker_active = False
                if self._active_loop_cancel_event is cancel_event:
                    self._active_loop_cancel_event = None
                self._stop_event.set()
                self._loop_wake_event.set()
                self._timeout_alert_active = False

        if self._operation_cancelled():
            finish_loop_state()
            self.flush_counts()
            return

        if self.device is None and not self.start_adb_server():
            self.log("디바이스 연결 대기 중... 계속 재시도합니다.")
            self._next_reconnect_at = time.monotonic() + self._reconnect_delay
            self._reconnect_delay = min(10.0, self._reconnect_delay * 2.0)

        if not self.is_running or self._operation_cancelled():
            finish_loop_state()
            self.flush_counts()
            return
        self._last_scan_had_match = False
        self._mark_screen_dirty()
        self._last_full_scan_time = 0.0
        now = time.monotonic()
        self.last_action_time = now
        self.last_match_time = now
        self.last_random_click_time = now
        self._timeout_alert_active = False
        self.preload_templates(
            [self.template_dir, self.fallback_template_dir],
            grayscales=(bool(self.match_grayscale),),
        )
        self.log("자동 클릭커 시작")

        try:
            while (
                self.is_running
                and not self._stop_event.is_set()
                and not (cancel_event is not None and cancel_event.is_set())
            ):
                self._loop_wake_event.clear()
                if (
                    not self.is_running
                    or self._stop_event.is_set()
                    or cancel_event is not None and cancel_event.is_set()
                ):
                    break
                iteration_started = time.monotonic()
                try:
                    self.run_once()
                except Exception as error:
                    self.log(f"⚠️ 스캔 루프 중 예외 발생 (자동 복구): {error}")
                if (
                    not self.is_running
                    or self._stop_event.is_set()
                    or cancel_event is not None and cancel_event.is_set()
                ):
                    break


                try:
                    timeout = self._safe_number(
                        self.no_match_timeout, 0, minimum=0
                    )
                    if (
                        timeout > 0
                        and time.monotonic() - self.last_action_time >= timeout
                    ):
                        should_alert = False
                        timeout_callback = None
                        with self._loop_state_lock:
                            if (
                                not self.is_running
                                or self._operation_cancelled()
                            ):
                                break
                            if not self._timeout_alert_active:
                                self._timeout_alert_active = True
                                should_alert = True
                                timeout_callback = self.on_timeout_callback

                        if should_alert:
                            self.log(
                                f"⚠️ 경고: {timeout}초 동안 템플릿 매칭이 "
                                "발생하지 않았습니다!"
                            )
                            if timeout_callback:
                                try:
                                    timeout_callback(timeout)
                                except Exception as error:
                                    self.log(f"타임아웃 콜백 오류: {error}")
                except Exception as error:
                    self.log(f"타임아웃 검사 중 오류: {error}")

                elapsed = time.monotonic() - iteration_started
                wait_time = self._current_wait_interval(elapsed=elapsed)
                self._wait_for_loop_wake(wait_time, cancel_event)
        except KeyboardInterrupt:
            self.log("사용자에 의해 중단됨")
        except Exception as error:
            self.log(f"치명적 오류 발생 (클릭커 종료): {error}")
        finally:
            finish_loop_state()
            self.flush_counts()
            self.log("자동 클릭커 종료")

    def stop_loop(self):
        with self._loop_state_lock:
            self.is_running = False
            self._stop_event.set()
            self._loop_wake_event.set()
            self._timeout_alert_active = False

    def request_shutdown(self):
        self._shutdown_event.set()
        self.stop_loop()

    def shutdown(self):
        self.request_shutdown()
        self.flush_counts()
        with self._device_lock:
            self.device = None
# Backward compatibility alias
TeraboxClicker = AutoClicker


if __name__ == "__main__":
    clicker = AutoClicker(ADB_PATH, ADB_HOST, ADB_PORT, DEVICE_ADDRESS)
    clicker.start_loop()
