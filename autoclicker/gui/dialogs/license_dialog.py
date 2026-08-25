import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np

from autoclicker.core.constants import (
    DEFAULT_ADB_MODE,
    DEFAULT_CUSTOM_ADB_PATH,
    VALID_ADB_MODES,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_DOUBLE_CLICK_INTERVAL,
    DEFAULT_POST_ACTION_DELAY,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_NO_MATCH_TIMEOUT,
    DEFAULT_CONSECUTIVE_MATCH_THRESHOLD,
    DEFAULT_RESET_COUNTS_ON_STARTUP,
)
from autoclicker.core.environment import get_default_adb_path, resolve_adb_path
from autoclicker.gui.theme import (
    COLOR_PRIMARY,
    COLOR_PRIMARY_HOVER,
    COLOR_SUCCESS,
    COLOR_SUCCESS_HOVER,
    COLOR_DANGER,
    COLOR_DANGER_HOVER,
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
from autoclicker.gui.win32_utils import draw_korean_banner, set_opencv_window_title

class LicenseNoticeWindow(ctk.CTkToplevel):
    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.title("📜 오픈소스 라이선스 고지 (Open Source Licenses)")
        self.geometry("620x540")
        self.minsize(500, 400)

        self.transient(parent_window)
        self.grab_set()

        self.header_label = ctk.CTkLabel(
            self,
            text="📜 오픈소스 소프트웨어 라이선스 고지",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        self.header_label.pack(padx=20, pady=(15, 5))

        self.sub_label = ctk.CTkLabel(
            self,
            text="본 프로그램은 아래 오픈소스 컴포넌트를 번들링하거나 활용하고 있습니다.",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.sub_label.pack(padx=20, pady=(0, 10))

        self.textbox = ctk.CTkTextbox(self, corner_radius=RADIUS_SM, font=ctk.CTkFont(size=11))
        self.textbox.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        license_text = (
            "================================================================================\n"
            "1. Android Debug Bridge (ADB)\n"
            "================================================================================\n"
            "Component: ADB executable and libraries (ADB/adb.exe, AdbWinApi.dll, etc.)\n"
            "Copyright (C) 2006-2024 The Android Open Source Project\n"
            "License: Apache License, Version 2.0\n\n"
            "Licensed under the Apache License, Version 2.0 (the \"License\");\n"
            "you may not use this file except in compliance with the License.\n"
            "You may obtain a copy of the License at:\n"
            "    http://www.apache.org/licenses/LICENSE-2.0\n\n"
            "Unless required by applicable law or agreed to in writing, software\n"
            "distributed under the License is distributed on an \"AS IS\" BASIS,\n"
            "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
            "See the License for the specific language governing permissions and\n"
            "limitations under the License.\n\n"
            "[면책 조항 (Disclaimer)]\n"
            "Apache 2.0 라이선스 조건에 따라 본 소프트웨어에 포함된 ADB 바이너리 및 관련\n"
            "라이브러리는 '있는 그대로(AS-IS)' 제공되며, 명시적이거나 묵시적인 어떠한\n"
            "보증(상품성, 특정 목적에의 적합성 등)도 제공하지 않습니다.\n\n"
            "================================================================================\n"
            "2. OpenCV (opencv-python)\n"
            "================================================================================\n"
            "Copyright (C) 2000-2024 OpenCV Foundation / Intel Corporation\n"
            "License: Apache License 2.0 (https://opencv.org/)\n\n"
            "================================================================================\n"
            "3. CustomTkinter\n"
            "================================================================================\n"
            "Copyright (c) 2023 Tom Schimansky\n"
            "License: MIT License (https://github.com/TomSchimansky/CustomTkinter)\n\n"
            "================================================================================\n"
            "4. pure-python-adb\n"
            "================================================================================\n"
            "Copyright (c) 2018 Swind Zheng\n"
            "License: MIT License (https://github.com/Swind/pure-python-adb)\n"
        )
        self.textbox.insert("1.0", license_text)
        self.textbox.configure(state="disabled")

        self.close_btn = ctk.CTkButton(
            self,
            text="확인 (Close)",
            height=32,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_MD,
            command=self.close_window
        )
        self.close_btn.pack(fill="x", padx=20, pady=(0, 15))
        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def close_window(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()
