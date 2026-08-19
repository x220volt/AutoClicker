import cv2
import numpy as np
from ppadb.client import Client as AdbClient
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import random
import subprocess
import sys
import threading
import time


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
VALID_TEMPLATE_ACTIONS = {"click", "double_click", "click_click", "double", "back"}
VALID_NO_MATCH_ACTIONS = {
    "none", "fallback_list", "random_click", "custom_click",
    "custom_double_click", "double_click", "click_click", "back",
}
CONFIG_FILENAME = "config.json"
COUNT_FLUSH_INTERVAL = 3.0
ADB_COMMAND_TIMEOUT = 10.0


def get_app_dir():
    """Return the writable directory next to the app/script, independent of cwd."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def get_default_adb_path(base_dir=None):
    """Return the bundled ADB executable when available."""
    if hasattr(sys, "_MEIPASS"):
        bundled_adb = os.path.join(sys._MEIPASS, "ADB", "adb.exe")
        if os.path.exists(bundled_adb):
            return bundled_adb

    local_adb = os.path.join(base_dir or get_app_dir(), "ADB", "adb.exe")
    if os.path.exists(local_adb):
        return local_adb
    return "adb.exe"


ADB_HOST = "127.0.0.1"
ADB_PORT = 5037
DEVICE_ADDRESS = "127.0.0.1:5555"
DEFAULT_SCAN_INTERVAL = 2
DEFAULT_NO_MATCH_TIMEOUT = 120
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_MATCH_GRAYSCALE = True
DEFAULT_ENABLE_RANDOM_CLICK = False
DEFAULT_RANDOM_CLICK_INTERVAL = 30
DEFAULT_DOUBLE_CLICK_INTERVAL = 1.0
APP_DIR = get_app_dir()
CONFIG_PATH = os.path.join(APP_DIR, CONFIG_FILENAME)
ADB_PATH = get_default_adb_path(APP_DIR)


class TeraboxClicker:
    """Thread-safe ADB screen matcher used by both the CLI and GUI."""

    _config_lock = threading.RLock()
    _template_cache_lock = threading.RLock()
    _template_cache = {}
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
    ):
        self.base_dir = os.path.abspath(base_dir or APP_DIR)
        self.adb_path = adb_path or get_default_adb_path(self.base_dir)
        self.host = host
        self.port = port
        self._device_address_explicit = device_address is not None
        self.device_address = str(device_address or DEVICE_ADDRESS).strip()
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
        self.no_match_action = "none"
        self.no_match_interval = self.random_click_interval
        self.no_match_coords = [500, 500]
        self.fallback_final_action = "none"
        self.fallback_final_coords = [500, 500]
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

        self.client = None
        self.device = None
        self.is_running = False
        self._stop_event = threading.Event()
        self._device_lock = threading.RLock()

        self.template_dir = os.path.join(self.base_dir, "templates")
        self.fallback_template_dir = os.path.join(
            self.base_dir, "fallback_templates"
        )
        self.config_path = os.path.join(self.base_dir, CONFIG_FILENAME)
        self.template_order = []
        self.template_counts = {}
        self.template_actions = {}
        self.template_offsets = {}
        self.template_delays = {}
        self.fallback_template_order = []
        self.fallback_template_counts = {}
        self.fallback_template_actions = {}
        self.fallback_template_offsets = {}
        self.fallback_template_delays = {}

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
    def _read_config_unlocked(config_path):
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("config.json 최상위 값은 객체여야 합니다.")
        return config

    @staticmethod
    def _atomic_write_config_unlocked(config_path, config):
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        temp_path = (
            f"{config_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with open(temp_path, "w", encoding="utf-8", newline="\n") as file:
                json.dump(config, file, ensure_ascii=False, indent=4)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, config_path)
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
        pending = cls._pending_count_deltas[config_path]
        if not any(pending[key] for key in pending):
            return True

        config = cls._read_config_unlocked(config_path)
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

        cls._atomic_write_config_unlocked(config_path, config)
        for count_key, deltas in snapshots.items():
            for filename, delta in deltas.items():
                pending[count_key][filename] -= delta
                if pending[count_key][filename] <= 0:
                    del pending[count_key][filename]
        return True

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
    def update_instances_config(cls, instances, config_path=CONFIG_PATH):
        config_path = os.path.abspath(config_path)
        with cls._config_lock:
            try:
                cls._flush_pending_counts_unlocked(config_path)
                config = cls._read_config_unlocked(config_path)
                config["instances"] = [
                    dict(instance)
                    for instance in instances
                    if isinstance(instance, dict)
                ]
                cls._atomic_write_config_unlocked(config_path, config)
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

    @staticmethod
    def _list_template_files(directory):
        try:
            entries = os.scandir(directory)
        except OSError:
            return []
        with entries:
            return sorted(
                entry.name
                for entry in entries
                if entry.is_file()
                and entry.name.lower().endswith(IMAGE_EXTENSIONS)
            )

    def _sync_template_collection(
        self,
        config,
        order_key,
        counts_key,
        actions_key,
        offsets_key,
        delays_key,
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
        return order, counts, actions, offsets, delays

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

        instances = config.get("instances", [])
        target_instance = None
        if isinstance(instances, list):
            target_instance = next(
                (
                    instance
                    for instance in instances
                    if isinstance(instance, dict)
                    and instance.get("device_address") == self.device_address
                ),
                None,
            )
        self._apply_settings(target_instance or config)

        (
            self.template_order,
            self.template_counts,
            self.template_actions,
            self.template_offsets,
            self.template_delays,
        ) = self._sync_template_collection(
            config,
            "template_order",
            "template_counts",
            "template_actions",
            "template_offsets",
            "template_delays",
            self.template_dir,
        )
        (
            self.fallback_template_order,
            self.fallback_template_counts,
            self.fallback_template_actions,
            self.fallback_template_offsets,
            self.fallback_template_delays,
        ) = self._sync_template_collection(
            config,
            "fallback_template_order",
            "fallback_template_counts",
            "fallback_template_actions",
            "fallback_template_offsets",
            "fallback_template_delays",
            self.fallback_template_dir,
        )

        self.adb_path = get_default_adb_path(self.base_dir)
        return True

    def _settings_dict(self):
        self.enable_random_click = self.no_match_action == "random_click"
        self.random_click_interval = self.no_match_interval
        return {
            "device_address": self.device_address,
            "scan_interval": self.scan_interval,
            "no_match_timeout": self.no_match_timeout,
            "similarity_threshold": self.similarity_threshold,
            "match_grayscale": self.match_grayscale,
            "enable_random_click": self.enable_random_click,
            "random_click_interval": self.random_click_interval,
            "double_click_interval": self.double_click_interval,
            "no_match_action": self.no_match_action,
            "no_match_interval": self.no_match_interval,
            "no_match_coords": list(self.no_match_coords),
            "fallback_final_action": self.fallback_final_action,
            "fallback_final_coords": list(self.fallback_final_coords),
        }

    def save_config(self, include_templates=True):
        """Atomically save settings while preserving other instances and counts."""
        with self._config_lock:
            try:
                self._flush_pending_counts_unlocked(self.config_path)
                config = self._read_config_unlocked(self.config_path)
                settings = self._settings_dict()
                config.update(settings)

                instances = config.get("instances", [])
                if not isinstance(instances, list):
                    instances = []
                updated = False
                for index, instance in enumerate(instances):
                    if (
                        isinstance(instance, dict)
                        and instance.get("device_address") == self.device_address
                    ):
                        merged = dict(instance)
                        merged.update(settings)
                        instances[index] = merged
                        updated = True
                        break
                if not updated:
                    instances.append(dict(settings))
                config["instances"] = instances

                if include_templates:
                    template_groups = (
                        (
                            "template_order",
                            "template_counts",
                            "template_actions",
                            "template_offsets",
                            "template_delays",
                            self.template_order,
                            self.template_counts,
                            self.template_actions,
                            self.template_offsets,
                            self.template_delays,
                        ),
                        (
                            "fallback_template_order",
                            "fallback_template_counts",
                            "fallback_template_actions",
                            "fallback_template_offsets",
                            "fallback_template_delays",
                            self.fallback_template_order,
                            self.fallback_template_counts,
                            self.fallback_template_actions,
                            self.fallback_template_offsets,
                            self.fallback_template_delays,
                        ),
                    )
                    path_key = os.path.abspath(self.config_path)
                    for (
                        order_key,
                        counts_key,
                        actions_key,
                        offsets_key,
                        delays_key,
                        order,
                        counts,
                        actions,
                        offsets,
                        delays,
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

                self._atomic_write_config_unlocked(self.config_path, config)
                return True
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.log(f"설정 저장 중 오류: {error}")
                return False

    @classmethod
    def invalidate_template_cache(cls, template_path=None):
        with cls._template_cache_lock:
            if template_path is None:
                cls._template_cache.clear()
            else:
                absolute_path = os.path.abspath(template_path)
                for cache_key in tuple(cls._template_cache):
                    cached_path = (
                        cache_key[0] if isinstance(cache_key, tuple) else cache_key
                    )
                    if cached_path == absolute_path:
                        cls._template_cache.pop(cache_key, None)

    def _remove_count_state(self, filename, counts_key):
        path_key = os.path.abspath(self.config_path)
        with self._config_lock:
            self._pending_count_deltas[path_key][counts_key].pop(filename, None)
            self._live_counts[path_key][counts_key].pop(filename, None)

    def _delete_template(
        self, filename, directory, order, counts, actions, offsets, delays, counts_key
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

    def set_template_delay(self, filename, delay):
        safe_delay = self._safe_number(delay, 0.0, minimum=0.0)
        self.template_delays[filename] = safe_delay
        self.save_config(include_templates=True)
        return safe_delay

    def set_fallback_template_delay(self, filename, delay):
        safe_delay = self._safe_number(delay, 0.0, minimum=0.0)
        self.fallback_template_delays[filename] = safe_delay
        self.save_config(include_templates=True)
        return safe_delay

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

        self.last_matched_template = filename
        self.last_matched_is_fallback = is_fallback

        if self.on_match_callback:
            try:
                self.on_match_callback(filename, new_count, is_fallback)
            except Exception as error:
                self.log(f"매칭 콜백 오류: {error}")
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
        }

    def _execute_action(self, action, x=None, y=None):
        if action == "back":
            return self.go_back()
        if action in ("double_click", "click_click", "double"):
            return self.double_click(x, y)
        return self.click(x, y)

    def _wait_after_action(self, seconds):
        if self.is_running:
            return not self._stop_event.wait(seconds)
        time.sleep(seconds)
        return True

    def _try_fallback_templates(self, screen, now):
        for filename in tuple(self.fallback_template_order):
            if self.is_running and self._stop_event.is_set():
                return False
            template_path = os.path.join(self.fallback_template_dir, filename)
            match = self.find_template(
                screen,
                template_path,
                threshold=self.similarity_threshold,
                filename=filename,
                offsets_dict=self.fallback_template_offsets,
            )
            if match is None:
                continue

            x, y, confidence = match
            action = self.fallback_template_actions.get(filename, "click")
            delay = self.fallback_template_delays.get(filename, 0.0)
            self.last_match_time = now
            if delay > 0:
                self.log(
                    f"⚡ [미매칭 복구] 매칭 성공: {filename} (유사도: {confidence:.2f}) "
                    f"-> 지연 시간 {delay:g}초 대기 중..."
                )
                if not self._wait_after_action(delay):
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
            self._wait_after_action(2.0)
            return True
        return False

    def _handle_no_match_action(self, screen, now):
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
                matched = self._try_fallback_templates(screen, now)
            else:
                self.log(
                    f"⚡ [미매칭 복구] {int(elapsed_since_match)}초 경과 (등록된 미매칭 템플릿 없음)"
                )

            if not matched:
                final_action = getattr(self, "fallback_final_action", "none")
                if final_action != "none":
                    success = self._execute_final_action(screen, final_action)
                    self.last_random_click_time = time.monotonic()
                    if success:
                        self._wait_after_action(1.0)
                else:
                    self.log(
                        "⚡ [미매칭 복구] 일치하는 미매칭 템플릿 없음 "
                        f"-> {int(self.no_match_interval)}초 후 재시도"
                    )
                    self.last_random_click_time = time.monotonic()
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
            self._wait_after_action(1.0)

    def _execute_final_action(self, screen, final_action):
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
        """Capture once, scan templates in priority order, and perform one action."""
        screen = self.capture_screen()
        if screen is None:
            self.log("화면 캡처 실패")
            return False

        if self.match_grayscale and screen.ndim == 3:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        for filename in tuple(self.template_order):
            if self.is_running and self._stop_event.is_set():
                return False
            template_path = os.path.join(self.template_dir, filename)
            match = self.find_template(
                screen,
                template_path,
                threshold=self.similarity_threshold,
                filename=filename,
            )
            if match is None:
                continue

            x, y, confidence = match
            action = self.template_actions.get(filename, "click")
            delay = self.template_delays.get(filename, 0.0)
            self.last_match_time = now
            if delay > 0:
                self.log(
                    f"매칭 성공: {filename} (유사도: {confidence:.2f}) "
                    f"-> 지연 시간 {delay:g}초 대기 중..."
                )
                if not self._wait_after_action(delay):
                    return False
            success = self._execute_action(action, x, y)
            self.log(
                f"액션 완료: {filename} "
                f"(유사도: {confidence:.2f}, 액션: {action}, 좌표: {x},{y})"
            )
            if success:
                self.last_action_time = time.monotonic()
                self.last_random_click_time = time.monotonic()
                self._timeout_alert_active = False
                self._record_match(filename)
            self._wait_after_action(2.0)
            return True

        self._handle_no_match_action(screen, now)
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
        if not os.path.exists(template_path):
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

        match = self.find_template(
            screen,
            template_path,
            threshold=self.similarity_threshold,
            filename=filename,
            offsets_dict=offsets_dict,
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
        now = time.monotonic()
        self.last_match_time = now
        if delay > 0:
            self.log(
                f"{prefix} '{filename}' 매칭 성공 -> 지연 시간 {delay:g}초 대기 중..."
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
        return success

    @staticmethod
    def _subprocess_kwargs():
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        return {}

    def _run_adb(self, args, timeout=ADB_COMMAND_TIMEOUT, check=False, text=True):
        return subprocess.run(
            [self.adb_path, *args],
            check=check,
            capture_output=True,
            text=text,
            timeout=timeout,
            **self._subprocess_kwargs(),
        )

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
        """Return online ADB serials after bounded, parallel emulator probing."""
        try:
            self._run_adb(["start-server"], check=True)
        except (OSError, subprocess.SubprocessError) as error:
            self.log(f"ADB 서버 시작 실패: {error}")
            return []

        emulator_ports = (
            5555, 5557, 5559, 5561, 5563, 5565,
            5575, 5585, 62001, 62025, 7555,
        )

        def connect_port(port):
            try:
                self._run_adb(
                    ["connect", f"127.0.0.1:{port}"], timeout=1.0
                )
            except (OSError, subprocess.SubprocessError):
                pass

        with ThreadPoolExecutor(max_workers=len(emulator_ports)) as executor:
            list(executor.map(connect_port, emulator_ports))

        try:
            result = self._run_adb(["devices"], check=True)
        except (OSError, subprocess.SubprocessError) as error:
            self.log(f"디바이스 목록 가져오기 실패: {error}")
            return []

        devices = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
            elif len(parts) >= 1 and parts[0].endswith(":5555") and "device" in parts:
                devices.append(parts[0])
            elif len(parts) >= 1 and parts[0].startswith("127.0.0.1:") and "device" in parts:
                devices.append(parts[0])
        return self.normalize_device_serials(devices)

    def start_adb_server(self):
        """Start ADB and bind only to a device confirmed by adb devices."""
        with self._device_lock:
            self.log(f"ADB 서버 시작 시도: {self.adb_path}")
            try:
                self._run_adb(["start-server"], check=True)
                if ":" in self.device_address:
                    self._run_adb(
                        ["connect", self.device_address], timeout=3.0
                    )

                client = AdbClient(host=self.host, port=self.port)
                target = next(
                    (
                        device
                        for device in client.devices()
                        if device.serial == self.device_address
                    ),
                    None,
                )
                if target is None and ":" in self.device_address:
                    try:
                        port = int(self.device_address.split(":")[-1])
                        emu_serial = f"emulator-{port - 1}"
                        target = next(
                            (
                                device
                                for device in client.devices()
                                if device.serial == emu_serial
                            ),
                            None,
                        )
                    except ValueError:
                        pass

                self.client = client
                self.device = target
                if target is None:
                    self.log(
                        f"장치를 찾을 수 없습니다: {self.device_address}"
                    )
                    return False

                self.log(f"장치 연결 성공: {self.device_address}")
                return True
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                self.client = None
                self.device = None
                self.log(f"ADB 서버 시작 중 오류 발생: {error}")
                return False
            except Exception as error:
                self.client = None
                self.device = None
                self.log(f"ADB 연결 확인 중 오류 발생: {error}")
                return False

    def disconnect(self):
        self.stop_loop()
        with self._device_lock:
            self.device = None
            self.client = None

    def capture_screen(self, grayscale=None):
        """Capture and decode the current device screen."""
        if self.device is None and not self.start_adb_server():
            return None

        with self._device_lock:
            device = self.device
        if device is None:
            return None

        try:
            result = device.screencap()
            if not result:
                return None
            use_gray = self.match_grayscale if grayscale is None else bool(grayscale)
            read_flag = cv2.IMREAD_GRAYSCALE if use_gray else cv2.IMREAD_COLOR
            image = cv2.imdecode(
                np.frombuffer(result, dtype=np.uint8), read_flag
            )
            if image is None:
                self.log("화면 캡처 이미지 디코딩 실패")
            return image
        except Exception as error:
            self.log(f"화면 캡처 중 오류: {error}")
            with self._device_lock:
                if self.device is device:
                    self.device = None
            return None

    def _load_template(self, template_path, grayscale=None):
        absolute_path = os.path.abspath(template_path)
        use_grayscale = (
            self.match_grayscale if grayscale is None else bool(grayscale)
        )
        cache_key = (absolute_path, use_grayscale)
        try:
            stat = os.stat(absolute_path)
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            self.invalidate_template_cache(absolute_path)
            return None, None

        with self._template_cache_lock:
            cached = self._template_cache.get(cache_key)
            if cached and cached[0] == signature:
                return cached[1], cached[2]

        try:
            encoded = np.fromfile(absolute_path, dtype=np.uint8)
            read_flag = cv2.IMREAD_GRAYSCALE if use_grayscale else cv2.IMREAD_COLOR
            template = cv2.imdecode(encoded, read_flag)
            if template is None or template.size == 0 or template.shape[0] <= 0 or template.shape[1] <= 0:
                raise ValueError("이미지 디코딩 결과가 비어 있습니다.")
            method = (
                cv2.TM_SQDIFF_NORMED
                if float(template.std()) < 1e-6
                else cv2.TM_CCOEFF_NORMED
            )
        except Exception as error:
            self.log(f"템플릿 로드 실패 ({template_path}): {error}")
            template, method = None, None

        with self._template_cache_lock:
            self._template_cache[cache_key] = (
                signature,
                template,
                method,
            )
        return template, method

    def find_template(
        self,
        screen_img,
        template_path,
        threshold=None,
        filename=None,
        offsets_dict=None,
    ):
        """Find a cached template and return a screen-clamped click coordinate."""
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
        template, method = self._load_template(template_path, grayscale=use_grayscale)
        if template is None:
            return None
        if (
            screen_img.shape[0] < template.shape[0]
            or screen_img.shape[1] < template.shape[1]
        ):
            return None

        try:
            result = cv2.matchTemplate(screen_img, template, method)
            min_value, max_value, min_location, max_location = cv2.minMaxLoc(
                result
            )
            if method == cv2.TM_SQDIFF_NORMED:
                confidence = 1.0 - min_value
                top_left_x, top_left_y = min_location
            else:
                confidence = max_value
                top_left_x, top_left_y = max_location

            if not math.isfinite(confidence) or confidence < threshold:
                return None

            height, width = template.shape[:2]
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
                    click_x = top_left_x + width // 2
                    click_y = top_left_y + height // 2
            else:
                click_x = top_left_x + width // 2
                click_y = top_left_y + height // 2

            screen_height, screen_width = screen_img.shape[:2]
            click_x = max(0, min(screen_width - 1, click_x))
            click_y = max(0, min(screen_height - 1, click_y))
            return click_x, click_y, float(confidence)
        except cv2.error as error:
            self.log(f"템플릿 매칭 중 OpenCV 오류 ({filename}): {error}")
            return None

    def click(self, x, y):
        with self._device_lock:
            device = self.device
        if device is None:
            return False
        try:
            x, y = int(x), int(y)
            device.shell(f"input tap {x} {y}")
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
        if device is None:
            return False
        try:
            device.shell("input keyevent 4")
            self.log("뒤로가기(Back 키) 수행")
            return True
        except Exception as error:
            self.log(f"뒤로가기 중 오류: {error}")
            with self._device_lock:
                if self.device is device:
                    self.device = None
            return False

    def start_loop(self, interval=None):
        """Run until stop_loop is called, recovering from individual scan errors."""
        if interval is not None:
            self.scan_interval = self._safe_number(
                interval, self.scan_interval, minimum=0.1
            )

        if self.device is None and not self.start_adb_server():
            self.log("디바이스 연결 대기 중... 계속 재시도합니다.")

        self.is_running = True
        self._stop_event.clear()
        now = time.monotonic()
        self.last_action_time = now
        self.last_match_time = now
        self.last_random_click_time = now
        self._timeout_alert_active = False
        self.log("자동 클릭커 시작")

        try:
            while self.is_running and not self._stop_event.is_set():
                try:
                    self.run_once()
                except Exception as error:
                    self.log(f"⚠️ 스캔 루프 중 예외 발생 (자동 복구): {error}")

                try:
                    timeout = self._safe_number(
                        self.no_match_timeout, 0, minimum=0
                    )
                    if (
                        timeout > 0
                        and time.monotonic() - self.last_action_time >= timeout
                    ):
                        if not self._timeout_alert_active:
                            self._timeout_alert_active = True
                            self.log(
                                f"⚠️ 경고: {timeout}초 동안 템플릿 매칭이 "
                                "발생하지 않았습니다!"
                            )
                            if self.on_timeout_callback:
                                try:
                                    self.on_timeout_callback(timeout)
                                except Exception as error:
                                    self.log(f"타임아웃 콜백 오류: {error}")
                except Exception as error:
                    self.log(f"타임아웃 검사 중 오류: {error}")

                interval_value = self._safe_number(
                    self.scan_interval,
                    DEFAULT_SCAN_INTERVAL,
                    minimum=0.1,
                )
                self._stop_event.wait(float(interval_value))
        except KeyboardInterrupt:
            self.log("사용자에 의해 중단됨")
        except Exception as error:
            self.log(f"치명적 오류 발생 (클릭커 종료): {error}")
        finally:
            self.is_running = False
            self._stop_event.set()
            self._timeout_alert_active = False
            self.flush_counts()
            self.log("자동 클릭커 종료")

    def stop_loop(self):
        self.is_running = False
        self._stop_event.set()
        self._timeout_alert_active = False

    def shutdown(self):
        self.stop_loop()
        self.flush_counts()
        with self._device_lock:
            self.device = None
            self.client = None


if __name__ == "__main__":
    clicker = TeraboxClicker(ADB_PATH, ADB_HOST, ADB_PORT, DEVICE_ADDRESS)
    clicker.start_loop()
