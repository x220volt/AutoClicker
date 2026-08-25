"""AutoClicker - Core Module & CLI Entry Point.

This module provides the AutoClicker core class and maintains 100% backward
compatibility with existing tests, scripts, and imports.
"""

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
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ppadb.client import Client as AdbClient
from ppadb.device import Device as AdbDevice

# Multi-instance CPU/threading optimization
try:
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

import autoclicker.core.clicker as _clicker_mod
import autoclicker.core.constants as _constants_mod
import autoclicker.core.environment as _env_mod

# Synchronize modules so patch("main.xxx") intercepts clicker module calls seamlessly
_clicker_mod.time = time
_clicker_mod.cv2 = cv2
_clicker_mod.np = np
_clicker_mod.os = os
_clicker_mod.subprocess = subprocess
_clicker_mod.json = json
_clicker_mod.random = random
_clicker_mod.shutil = shutil
_clicker_mod.socket = socket
_clicker_mod.struct = struct
_clicker_mod.math = math
_clicker_mod.copy = copy
_clicker_mod.zlib = zlib
_clicker_mod.threading = threading

# Constants
IMAGE_EXTENSIONS = _constants_mod.IMAGE_EXTENSIONS
VALID_TEMPLATE_ACTIONS = _constants_mod.VALID_TEMPLATE_ACTIONS
VALID_DELAY_TYPES = _constants_mod.VALID_DELAY_TYPES
VALID_NO_MATCH_ACTIONS = _constants_mod.VALID_NO_MATCH_ACTIONS
CONFIG_FILENAME = _constants_mod.CONFIG_FILENAME
TEMPLATE_DIR_NAME = _constants_mod.TEMPLATE_DIR_NAME
FALLBACK_TEMPLATE_DIR_NAME = _constants_mod.FALLBACK_TEMPLATE_DIR_NAME
COUNT_FLUSH_INTERVAL = _constants_mod.COUNT_FLUSH_INTERVAL
TEMPLATE_MANIFEST_REFRESH_INTERVAL = _constants_mod.TEMPLATE_MANIFEST_REFRESH_INTERVAL
ADB_COMMAND_TIMEOUT = _constants_mod.ADB_COMMAND_TIMEOUT
ADB_SERVER_PROBE_TIMEOUT = _constants_mod.ADB_SERVER_PROBE_TIMEOUT
ADB_SERVER_START_ATTEMPTS = _constants_mod.ADB_SERVER_START_ATTEMPTS
ADB_SERVER_RETRY_DELAYS = _constants_mod.ADB_SERVER_RETRY_DELAYS
ADB_VERSION_TIMEOUT = _constants_mod.ADB_VERSION_TIMEOUT
PRESCALE_FACTOR = _constants_mod.PRESCALE_FACTOR
PRESCALE_MIN_TEMPLATE_DIM = _constants_mod.PRESCALE_MIN_TEMPLATE_DIM
LOCAL_VERIFY_MARGIN = _constants_mod.LOCAL_VERIFY_MARGIN
LOCAL_VERIFY_TOP_K = _constants_mod.LOCAL_VERIFY_TOP_K
HINT_FULL_SCAN_INTERVAL = _constants_mod.HINT_FULL_SCAN_INTERVAL
PERFORMANCE_SAMPLE_LIMIT = _constants_mod.PERFORMANCE_SAMPLE_LIMIT
DEFAULT_FORCE_SCAN_INTERVAL = _constants_mod.DEFAULT_FORCE_SCAN_INTERVAL
DEFAULT_ROI_FULL_SCAN_BUDGET = _constants_mod.DEFAULT_ROI_FULL_SCAN_BUDGET
ROI_PRIORITY_TEMPLATE_COUNT = _constants_mod.ROI_PRIORITY_TEMPLATE_COUNT
DEFAULT_MAX_IDLE_INTERVAL = _constants_mod.DEFAULT_MAX_IDLE_INTERVAL
DEFAULT_CAPTURE_BACKEND = _constants_mod.DEFAULT_CAPTURE_BACKEND
VALID_CAPTURE_BACKENDS = _constants_mod.VALID_CAPTURE_BACKENDS
DEFAULT_ADB_MODE = _constants_mod.DEFAULT_ADB_MODE
DEFAULT_CUSTOM_ADB_PATH = _constants_mod.DEFAULT_CUSTOM_ADB_PATH
VALID_ADB_MODES = _constants_mod.VALID_ADB_MODES
ADB_HOST = _constants_mod.ADB_HOST
ADB_PORT = _constants_mod.ADB_PORT
DEVICE_ADDRESS = _constants_mod.DEVICE_ADDRESS
DEFAULT_SCAN_INTERVAL = _constants_mod.DEFAULT_SCAN_INTERVAL
DEFAULT_NO_MATCH_TIMEOUT = _constants_mod.DEFAULT_NO_MATCH_TIMEOUT
DEFAULT_SIMILARITY_THRESHOLD = _constants_mod.DEFAULT_SIMILARITY_THRESHOLD
DEFAULT_MATCH_GRAYSCALE = _constants_mod.DEFAULT_MATCH_GRAYSCALE
DEFAULT_ENABLE_RANDOM_CLICK = _constants_mod.DEFAULT_ENABLE_RANDOM_CLICK
DEFAULT_RANDOM_CLICK_INTERVAL = _constants_mod.DEFAULT_RANDOM_CLICK_INTERVAL
DEFAULT_DOUBLE_CLICK_INTERVAL = _constants_mod.DEFAULT_DOUBLE_CLICK_INTERVAL
DEFAULT_POST_ACTION_DELAY = _constants_mod.DEFAULT_POST_ACTION_DELAY
DEFAULT_CONSECUTIVE_MATCH_THRESHOLD = _constants_mod.DEFAULT_CONSECUTIVE_MATCH_THRESHOLD
DEFAULT_RESET_COUNTS_ON_STARTUP = _constants_mod.DEFAULT_RESET_COUNTS_ON_STARTUP

# Environment & Paths
get_app_dir = _env_mod.get_app_dir
get_default_adb_path = _env_mod.get_default_adb_path
resolve_adb_path = _env_mod.resolve_adb_path
APP_DIR = _env_mod.APP_DIR
CONFIG_PATH = _env_mod.CONFIG_PATH
ADB_PATH = _env_mod.ADB_PATH

# Classes
AutoClicker = _clicker_mod.AutoClicker
TeraboxClicker = _clicker_mod.TeraboxClicker

__all__ = [
    "AutoClicker",
    "TeraboxClicker",
    "get_app_dir",
    "get_default_adb_path",
    "resolve_adb_path",
    "APP_DIR",
    "CONFIG_PATH",
    "ADB_PATH",
    "ADB_HOST",
    "ADB_PORT",
    "DEVICE_ADDRESS",
    "DEFAULT_SCAN_INTERVAL",
    "DEFAULT_NO_MATCH_TIMEOUT",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_MATCH_GRAYSCALE",
    "DEFAULT_ENABLE_RANDOM_CLICK",
    "DEFAULT_RANDOM_CLICK_INTERVAL",
    "DEFAULT_DOUBLE_CLICK_INTERVAL",
    "DEFAULT_POST_ACTION_DELAY",
    "DEFAULT_CONSECUTIVE_MATCH_THRESHOLD",
    "DEFAULT_RESET_COUNTS_ON_STARTUP",
    "CONFIG_FILENAME",
    "TEMPLATE_DIR_NAME",
    "FALLBACK_TEMPLATE_DIR_NAME",
    "cv2",
    "np",
    "os",
    "sys",
    "time",
    "subprocess",
    "json",
    "threading",
    "random",
    "shutil",
    "socket",
    "struct",
    "math",
    "copy",
    "zlib",
    "AdbClient",
    "AdbDevice",
]


if __name__ == "__main__":
    clicker = AutoClicker(ADB_PATH, ADB_HOST, ADB_PORT, DEVICE_ADDRESS)
    clicker.start_loop()
