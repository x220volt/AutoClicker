"""AutoClicker - GUI Module & Application Entry Point.

This module acts as the backward-compatible entry point and facade for autoclicker.gui.
"""

import copy
import json
import math
import os
import random
import shutil
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

import autoclicker.gui.tab_view as _tab_view_mod
import autoclicker.gui.main_window as _main_window_mod

# Synchronize modules so patch("gui_app.ctk...") intercepts calls
_tab_view_mod.ctk = ctk
_tab_view_mod.tk = tk
_tab_view_mod.messagebox = messagebox
_tab_view_mod.filedialog = filedialog
_tab_view_mod.cv2 = cv2
_tab_view_mod.np = np
_tab_view_mod.os = os
_tab_view_mod.time = time
_tab_view_mod.threading = threading

_main_window_mod.ctk = ctk
_main_window_mod.tk = tk
_main_window_mod.messagebox = messagebox
_main_window_mod.filedialog = filedialog
_main_window_mod.os = os
_main_window_mod.time = time
_main_window_mod.threading = threading

from autoclicker.core.constants import (
    IMAGE_EXTENSIONS,
    CONFIG_FILENAME,
    TEMPLATE_DIR_NAME,
    FALLBACK_TEMPLATE_DIR_NAME,
    DEFAULT_ADB_MODE,
    DEFAULT_CUSTOM_ADB_PATH,
    VALID_ADB_MODES,
    ADB_HOST,
    ADB_PORT,
    DEVICE_ADDRESS,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_NO_MATCH_TIMEOUT,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_MATCH_GRAYSCALE,
    DEFAULT_ENABLE_RANDOM_CLICK,
    DEFAULT_RANDOM_CLICK_INTERVAL,
    DEFAULT_DOUBLE_CLICK_INTERVAL,
    DEFAULT_POST_ACTION_DELAY,
    DEFAULT_CONSECUTIVE_MATCH_THRESHOLD,
    DEFAULT_RESET_COUNTS_ON_STARTUP,
)
from autoclicker.core.environment import (
    get_app_dir,
    get_default_adb_path,
    resolve_adb_path,
    APP_DIR,
    CONFIG_PATH,
    ADB_PATH,
)
from autoclicker.core.clicker import (
    AutoClicker,
    TeraboxClicker,
)
from autoclicker.gui.theme import (
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SUCCESS,
    COLOR_SUCCESS_HOVER,
    COLOR_DANGER,
    COLOR_DANGER_HOVER,
    COLOR_DANGER_MUTED,
    COLOR_DANGER_MUTED_HOVER,
    COLOR_WARNING,
    COLOR_WARNING_HOVER,
    COLOR_INFO,
    COLOR_INFO_HOVER,
    COLOR_NEUTRAL,
    COLOR_NEUTRAL_HOVER,
    COLOR_CARD_BG,
    COLOR_CARD_INNER,
    COLOR_BORDER,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    RADIUS_SM,
    RADIUS_MD,
    RADIUS_LG,
    NO_MATCH_ACTION_MAP,
    REVERSE_NO_MATCH_ACTION_MAP,
    get_action_button_style,
    get_delay_button_style,
)
from autoclicker.gui.win32_utils import (
    draw_korean_banner,
    set_opencv_window_title,
)
from autoclicker.gui.widgets import (
    TemplateRowWidgets,
    CTKContextMenu,
    TemplatePreviewTooltip,
)
from autoclicker.gui.dialogs import (
    SettingsWindow,
    LicenseNoticeWindow,
    RenameTemplateWindow,
    TemplateDelayWindow,
    AddInstanceWindow,
)
from autoclicker.gui.tab_view import InstanceTabFrame
from autoclicker.gui.main_window import (
    App,
    AutoClickerApp,
    TeraboxClickerApp,
)

__all__ = [
    "App",
    "AutoClickerApp",
    "TeraboxClickerApp",
    "InstanceTabFrame",
    "SettingsWindow",
    "LicenseNoticeWindow",
    "RenameTemplateWindow",
    "TemplateDelayWindow",
    "AddInstanceWindow",
    "TemplateRowWidgets",
    "CTKContextMenu",
    "TemplatePreviewTooltip",
    "AutoClicker",
    "TeraboxClicker",
    "draw_korean_banner",
    "set_opencv_window_title",
    "get_action_button_style",
    "get_delay_button_style",
    "get_app_dir",
    "get_default_adb_path",
    "resolve_adb_path",
    "APP_DIR",
    "CONFIG_PATH",
    "ADB_PATH",
    "ctk",
    "tk",
    "cv2",
    "np",
    "os",
    "sys",
    "time",
    "threading",
]


if __name__ == "__main__":
    app = App()
    app.mainloop()
