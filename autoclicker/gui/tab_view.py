import os
import sys
import threading
import time
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from autoclicker.core.constants import (
    CONFIG_FILENAME,
    IMAGE_EXTENSIONS,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_DOUBLE_CLICK_INTERVAL,
    DEFAULT_POST_ACTION_DELAY,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_NO_MATCH_TIMEOUT,
    DEFAULT_CONSECUTIVE_MATCH_THRESHOLD,
    DEFAULT_RESET_COUNTS_ON_STARTUP,
    DEFAULT_ADB_MODE,
    DEFAULT_CUSTOM_ADB_PATH,
    VALID_ADB_MODES,
    ADB_COMMAND_TIMEOUT,
)
from autoclicker.core.environment import get_app_dir, get_default_adb_path, resolve_adb_path
from autoclicker.core.clicker import AutoClicker
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
from autoclicker.gui.win32_utils import draw_korean_banner, set_opencv_window_title
from autoclicker.gui.widgets import (
    TemplateRowWidgets,
    CTKContextMenu,
    TemplatePreviewTooltip,
)
from autoclicker.gui.dialogs import (
    SettingsWindow,
    RenameTemplateWindow,
    TemplateDelayWindow,
)

class InstanceTabFrame(ctk.CTkFrame):
    def __init__(self, master, app_owner, tab_name, device_address="127.0.0.1:5555"):
        super().__init__(master, fg_color="transparent")
        self.app_owner = app_owner
        self.tab_name = tab_name
        self.is_alert_open = False
        self._alert_shown_for_current_timeout = False
        self._destroyed = False
        self._shutdown_started = False
        self._connection_generation = 0
        self._loop_starting = False
        self._loop_generation = 0
        self._loop_cancel_event = threading.Event()
        self._action_threads = set()
        self._action_threads_lock = threading.Lock()
        self._log_line_count = 1
        self._log_buffer = []
        self._log_flush_scheduled = False
        self._drag_positions = []
        self._timer_render_state = {}

        self.clicker = AutoClicker(
            device_address=device_address,
            on_timeout_callback=self.on_no_match_timeout,
            logger=self.log_message,
            on_match_callback=self.on_template_match,
            on_consecutive_match_callback=self.on_consecutive_match_warning,
        )
        self.clicker_thread = None
        self.settings_window = None
        self._last_bg_timer_update = 0.0

        # --- Top Header Bar inside Tab ---
        self.header_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD_BG, corner_radius=RADIUS_MD)
        self.header_frame.pack(fill="x", padx=10, pady=(10, 5))
        self.normal_header_fg = self.header_frame.cget("fg_color")

        # Device Address Selector
        self.device_addr_label = ctk.CTkLabel(
            self.header_frame,
            text="디바이스:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.device_addr_label.pack(side="left", padx=(12, 6), pady=8)

        initial_devices = [self.clicker.device_address]
        self.device_combo = ctk.CTkComboBox(
            self.header_frame,
            values=initial_devices,
            width=165,
            height=32,
            corner_radius=RADIUS_MD,
            command=self.on_device_selected
        )
        self.device_combo.set(self.clicker.device_address)
        self.device_combo.pack(side="left", padx=(0, 6), pady=8)
        self.device_combo.bind("<Return>", self.save_settings)
        self.device_combo.bind("<FocusOut>", self.save_settings)

        # Connect / Disconnect Button
        self.connect_button = ctk.CTkButton(
            self.header_frame,
            text="디바이스 연결",
            width=120,
            height=32,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            corner_radius=RADIUS_MD,
            command=self.toggle_connection
        )
        self.connect_button.pack(side="left", padx=4, pady=8)

        # Start / Stop Clicker Button
        self.start_button = ctk.CTkButton(
            self.header_frame,
            text="클리커 시작",
            width=120,
            height=32,
            state="disabled",
            fg_color=COLOR_SUCCESS,
            hover_color=COLOR_SUCCESS_HOVER,
            corner_radius=RADIUS_MD,
            command=self.toggle_clicker
        )
        self.start_button.pack(side="left", padx=4, pady=8)

        # Status Label
        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="상태: 연결 안 됨",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        self.status_label.pack(side="left", padx=12, pady=8)

        # Right side button in tab header: Delete Tab
        self.delete_tab_btn = ctk.CTkButton(
            self.header_frame,
            text="탭 닫기",
            width=85,
            height=32,
            fg_color=COLOR_DANGER,
            hover_color=COLOR_DANGER_HOVER,
            corner_radius=RADIUS_MD,
            command=lambda: self.app_owner.remove_instance_tab(self.tab_name)
        )
        self.delete_tab_btn.pack(side="right", padx=(4, 12), pady=8)

        # --- Real-time Status & Timer Info Bar ---
        self.timer_bar_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD_BG, corner_radius=RADIUS_MD)
        self.timer_bar_frame.pack(fill="x", padx=10, pady=(0, 5))
        self.timer_bar_frame.grid_columnconfigure(0, weight=1)
        self.timer_bar_frame.grid_columnconfigure(1, weight=1)
        self.timer_bar_frame.grid_columnconfigure(2, weight=1)

        # 1. 미매칭 대기 시간 카드
        self.card_no_match = ctk.CTkFrame(self.timer_bar_frame, fg_color=COLOR_CARD_INNER, corner_radius=RADIUS_SM)
        self.card_no_match.grid(row=0, column=0, padx=5, pady=4, sticky="ew")
        self.no_match_timer_label = ctk.CTkLabel(
            self.card_no_match,
            text="⚡ 미매칭 대기: 정지됨",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        self.no_match_timer_label.pack(padx=10, pady=4)

        # 2. 매칭없음 경고 시간 카드
        self.card_timeout = ctk.CTkFrame(self.timer_bar_frame, fg_color=COLOR_CARD_INNER, corner_radius=RADIUS_SM)
        self.card_timeout.grid(row=0, column=1, padx=5, pady=4, sticky="ew")
        self.timeout_timer_label = ctk.CTkLabel(
            self.card_timeout,
            text="⚠️ 매칭없음 경고: 정지됨",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        self.timeout_timer_label.pack(padx=10, pady=4)

        # 3. 최근 매칭 정보 카드
        self.card_last_match = ctk.CTkFrame(self.timer_bar_frame, fg_color=COLOR_CARD_INNER, corner_radius=RADIUS_SM)
        self.card_last_match.grid(row=0, column=2, padx=5, pady=4, sticky="ew")
        self.last_match_info_label = ctk.CTkLabel(
            self.card_last_match,
            text="🎯 최근 매칭: 대기 중",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT_MUTED
        )
        self.last_match_info_label.pack(padx=10, pady=4)

        # --- Main Body (Split into Log View and Templates View) ---
        self.body_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.body_frame.grid_columnconfigure(0, weight=4)
        self.body_frame.grid_columnconfigure(1, weight=6)
        self.body_frame.grid_rowconfigure(0, weight=1)

        # Left Column: Log Box Frame
        self.log_frame = ctk.CTkFrame(self.body_frame, fg_color=COLOR_CARD_BG, corner_radius=RADIUS_MD)
        self.log_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.log_frame.grid_rowconfigure(1, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)

        self.log_top_header = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        self.log_top_header.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")

        self.log_header_label = ctk.CTkLabel(
            self.log_top_header,
            text="📜 실행 로그 (Activity Log)",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.log_header_label.pack(side="left")

        self.clear_log_btn = ctk.CTkButton(
            self.log_top_header,
            text="🧹 지우기",
            width=70,
            height=28,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_SM,
            font=ctk.CTkFont(size=11),
            command=self.clear_log
        )
        self.clear_log_btn.pack(side="right")

        self.log_textbox = ctk.CTkTextbox(self.log_frame, width=350, corner_radius=RADIUS_SM)
        self.log_textbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.log_textbox.insert("0.0", f"Initialized Tab [{self.tab_name}] for ADB device [{self.clicker.device_address}]\n")
        self.log_textbox.configure(state="disabled")

        # Right Column: Active & Fallback Templates Frame
        self.templates_main_frame = ctk.CTkFrame(self.body_frame, fg_color="transparent")
        self.templates_main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        self.templates_main_frame.grid_rowconfigure(0, weight=1)
        self.templates_main_frame.grid_columnconfigure(0, weight=1)

        # Tabview for Primary and Fallback Templates
        self.template_tabview = ctk.CTkTabview(self.templates_main_frame)
        self.template_tabview.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        self.tab_primary = self.template_tabview.add("📋 기본 템플릿")
        self.tab_fallback = self.template_tabview.add("⚡ 미매칭 템플릿")

        # --- Setup Primary Tab ---
        self.primary_header_frame = ctk.CTkFrame(self.tab_primary, fg_color="transparent")
        self.primary_header_frame.pack(fill="x", padx=5, pady=(5, 5))

        self.primary_header_label = ctk.CTkLabel(
            self.primary_header_frame,
            text="📋 기본 템플릿 목록",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.primary_header_label.pack(side="left")

        self.reset_counts_button = ctk.CTkButton(
            self.primary_header_frame,
            text="🔄 카운트 초기화",
            width=120,
            height=30,
            fg_color=COLOR_DANGER_MUTED,
            hover_color=COLOR_DANGER_MUTED_HOVER,
            corner_radius=RADIUS_MD,
            command=self.reset_counts_event
        )
        self.reset_counts_button.pack(side="right", padx=(5, 0))

        self.crop_button = ctk.CTkButton(
            self.primary_header_frame,
            text="➕ 템플릿 등록",
            width=120,
            height=30,
            fg_color=COLOR_SUCCESS,
            hover_color=COLOR_SUCCESS_HOVER,
            corner_radius=RADIUS_MD,
            command=lambda: self.start_cropping(is_fallback=False),
            state="disabled"
        )
        self.crop_button.pack(side="right", padx=(0, 5))

        # Search Bar for Primary Templates
        self.primary_search_frame = ctk.CTkFrame(self.tab_primary, fg_color="transparent")
        self.primary_search_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.primary_search_var = tk.StringVar()
        self.primary_search_entry = ctk.CTkEntry(
            self.primary_search_frame,
            placeholder_text="🔍 템플릿 검색...",
            textvariable=self.primary_search_var,
            height=28,
            corner_radius=RADIUS_SM,
            font=ctk.CTkFont(size=12),
        )
        self.primary_search_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.primary_search_var.trace_add("write", lambda *args: self.filter_templates(is_fallback=False))
        self.primary_search_entry.bind("<Escape>", lambda e: self.clear_search(is_fallback=False))

        self.primary_clear_search_btn = ctk.CTkButton(
            self.primary_search_frame,
            text="✕",
            width=28,
            height=28,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_SM,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.clear_search(is_fallback=False)
        )
        self.primary_clear_search_btn.pack(side="right")

        self.primary_no_match_label = None

        self.templates_frame = ctk.CTkScrollableFrame(self.tab_primary)
        self.templates_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # --- Setup Fallback Tab ---
        self.fallback_header_frame = ctk.CTkFrame(self.tab_fallback, fg_color="transparent")
        self.fallback_header_frame.pack(fill="x", padx=5, pady=(5, 5))

        self.fallback_header_label = ctk.CTkLabel(
            self.fallback_header_frame,
            text="⚡ 미매칭 복구 템플릿",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.fallback_header_label.pack(side="left")

        self.fallback_timer_sublabel = ctk.CTkLabel(
            self.fallback_header_frame,
            text="대기: --",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_PRIMARY
        )
        self.fallback_timer_sublabel.pack(side="left", padx=(10, 0))

        self.reset_fb_counts_button = ctk.CTkButton(
            self.fallback_header_frame,
            text="🔄 카운트 초기화",
            width=120,
            height=30,
            fg_color=COLOR_DANGER_MUTED,
            hover_color=COLOR_DANGER_MUTED_HOVER,
            corner_radius=RADIUS_MD,
            command=self.reset_fallback_counts_event
        )
        self.reset_fb_counts_button.pack(side="right", padx=(5, 0))

        self.crop_fb_button = ctk.CTkButton(
            self.fallback_header_frame,
            text="➕ 복구 템플릿 등록",
            width=140,
            height=30,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            corner_radius=RADIUS_MD,
            command=lambda: self.start_cropping(is_fallback=True),
            state="disabled"
        )
        self.crop_fb_button.pack(side="right", padx=(0, 5))

        # Search Bar for Fallback Templates
        self.fallback_search_frame = ctk.CTkFrame(self.tab_fallback, fg_color="transparent")
        self.fallback_search_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.fallback_search_var = tk.StringVar()
        self.fallback_search_entry = ctk.CTkEntry(
            self.fallback_search_frame,
            placeholder_text="🔍 복구 템플릿 검색...",
            textvariable=self.fallback_search_var,
            height=28,
            corner_radius=RADIUS_SM,
            font=ctk.CTkFont(size=12),
        )
        self.fallback_search_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.fallback_search_var.trace_add("write", lambda *args: self.filter_templates(is_fallback=True))
        self.fallback_search_entry.bind("<Escape>", lambda e: self.clear_search(is_fallback=True))

        self.fallback_clear_search_btn = ctk.CTkButton(
            self.fallback_search_frame,
            text="✕",
            width=28,
            height=28,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_SM,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self.clear_search(is_fallback=True)
        )
        self.fallback_clear_search_btn.pack(side="right")

        self.fb_no_match_label = None

        self.fallback_templates_frame = ctk.CTkScrollableFrame(self.tab_fallback)
        self.fallback_templates_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # --- Fallback Final Action Control (Bottom Panel) ---
        self.fallback_final_frame = ctk.CTkFrame(self.tab_fallback, fg_color=COLOR_CARD_BG, corner_radius=RADIUS_MD)
        self.fallback_final_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.fb_final_title = ctk.CTkLabel(
            self.fallback_final_frame,
            text="📌 최종 탈출 동작:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.fb_final_title.pack(side="left", padx=(12, 8), pady=8)

        curr_fb_action = getattr(self.clicker, 'fallback_final_action', 'none')
        curr_fb_label = NO_MATCH_ACTION_MAP.get(curr_fb_action, "사용 안 함 (Disabled)")
        final_action_options = [
            NO_MATCH_ACTION_MAP["none"],
            NO_MATCH_ACTION_MAP["custom_click"],
            NO_MATCH_ACTION_MAP["custom_double_click"],
            NO_MATCH_ACTION_MAP["back"],
            NO_MATCH_ACTION_MAP["random_click"],
        ]
        self.fb_final_combo = ctk.CTkOptionMenu(
            self.fallback_final_frame,
            values=final_action_options,
            width=200,
            height=30,
            corner_radius=RADIUS_SM,
            command=self.on_fallback_final_action_changed
        )
        self.fb_final_combo.set(curr_fb_label)
        self.fb_final_combo.pack(side="left", padx=(0, 8), pady=8)

        # Coordinate inputs
        self.fb_final_coord_frame = ctk.CTkFrame(self.fallback_final_frame, fg_color="transparent")
        self.fb_final_coord_frame.pack(side="left", padx=(0, 8), pady=8)

        coords = getattr(self.clicker, 'fallback_final_coords', [500, 500])
        self.fb_final_x_label = ctk.CTkLabel(self.fb_final_coord_frame, text="X:")
        self.fb_final_x_label.pack(side="left", padx=(0, 2))
        self.fb_final_x_entry = ctk.CTkEntry(self.fb_final_coord_frame, width=55, height=30, corner_radius=RADIUS_SM)
        self.fb_final_x_entry.insert(0, str(coords[0]))
        self.fb_final_x_entry.pack(side="left", padx=(0, 6))
        self.fb_final_x_entry.bind("<KeyRelease>", self.save_fallback_final_settings)
        self.fb_final_x_entry.bind("<FocusOut>", self.save_fallback_final_settings)

        self.fb_final_y_label = ctk.CTkLabel(self.fb_final_coord_frame, text="Y:")
        self.fb_final_y_label.pack(side="left", padx=(0, 2))
        self.fb_final_y_entry = ctk.CTkEntry(self.fb_final_coord_frame, width=55, height=30, corner_radius=RADIUS_SM)
        self.fb_final_y_entry.insert(0, str(coords[1]))
        self.fb_final_y_entry.pack(side="left", padx=(0, 6))
        self.fb_final_y_entry.bind("<KeyRelease>", self.save_fallback_final_settings)
        self.fb_final_y_entry.bind("<FocusOut>", self.save_fallback_final_settings)

        self.fb_final_pick_btn = ctk.CTkButton(
            self.fb_final_coord_frame,
            text="🎯 좌표 선택",
            width=95,
            height=30,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_SM,
            command=self.pick_fallback_final_coords
        )
        self.fb_final_pick_btn.pack(side="left", padx=2)

        self.update_fb_final_coord_visibility(curr_fb_action)

        # Test Run Button
        self.fb_final_test_btn = ctk.CTkButton(
            self.fallback_final_frame,
            text="⚡ 즉시 실행",
            width=95,
            height=30,
            fg_color=COLOR_WARNING,
            hover_color=COLOR_WARNING_HOVER,
            corner_radius=RADIUS_SM,
            command=self.test_fallback_final_action
        )
        self.fb_final_test_btn.pack(side="right", padx=(8, 12), pady=8)

        self.refresh_templates()

    def clear_log(self):
        try:
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")
            self._log_line_count = 1
            self.log_textbox.configure(state="disabled")
        except Exception:
            pass

    def update_device_combo_values(self, device_list):
        if not self._destroyed:
            self.device_combo.configure(values=device_list)

    def _set_disconnected_ui(self):
        self.status_label.configure(text="상태: 연결 안 됨", text_color=COLOR_TEXT_MUTED)
        self.start_button.configure(
            state="disabled", text="▶ 시작 (Start)",
            fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER
        )
        self.crop_button.configure(state="disabled")
        self.crop_fb_button.configure(state="disabled")
        self.connect_button.configure(
            state="normal", text="🔗 연결 (Connect)",
            fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER
        )

    def on_device_selected(self, choice):
        if choice:
            self.save_settings()

    def save_settings(self, event=None):
        new_address = self.device_combo.get().strip()
        if not new_address:
            self.device_combo.set(self.clicker.device_address)
            return

        if new_address != self.clicker.device_address:
            self._connection_generation += 1
            self.clicker.disconnect()
            self.clicker.device_address = new_address
            self.clicker.load_config()
            self._set_disconnected_ui()
            self.refresh_templates()

        self.app_owner.save_app_config()

    def log_message(self, message):
        if self._destroyed:
            return
        timestamp = time.strftime("[%H:%M:%S] ")
        self._log_buffer.append(f"{timestamp}{message}\n")
        if not self._log_flush_scheduled:
            self._log_flush_scheduled = True
            self.app_owner.post_to_ui(self._flush_log_buffer)

    def _flush_log_buffer(self):
        self._log_flush_scheduled = False
        if self._destroyed or not self.winfo_exists():
            self._log_buffer.clear()
            return
        if not self._log_buffer:
            return
        combined = "".join(self._log_buffer)
        lines_count = len(self._log_buffer)
        self._log_buffer.clear()

        try:
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", combined)
            self._log_line_count += lines_count
            if self._log_line_count > 150:
                self.log_textbox.delete("1.0", "51.0")
                self._log_line_count -= 50
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")
        except Exception:
            pass

    def on_template_match(self, filename, count, is_fallback):
        self.app_owner.post_to_ui(self.clear_warning_ui)
        self.app_owner.post_to_ui(
            lambda: self.app_owner.update_template_count_for_all(
                filename, count, is_fallback
            )
        )

    def _set_tab_warning_state(self, is_warning):
        if self._destroyed or not hasattr(self.app_owner, "tabview"):
            return
        try:
            segmented_btn = getattr(self.app_owner.tabview, "_segmented_button", None)
            if not segmented_btn or not hasattr(segmented_btn, "_buttons_dict"):
                return
            btn = segmented_btn._buttons_dict.get(self.tab_name)
            if not btn or not btn.winfo_exists():
                return

            if not hasattr(self, "_tab_button_defaults"):
                self._tab_button_defaults = {
                    "text_color": btn.cget("text_color"),
                    "border_color": btn.cget("border_color"),
                    "border_width": btn.cget("border_width"),
                }

            if is_warning:
                current_text = str(btn.cget("text"))
                if not current_text.startswith("⚠️"):
                    btn.configure(
                        text=f"⚠️ {self.tab_name}",
                        text_color="#FF4D4D",
                        border_color="#FF4D4D",
                    )
            else:
                current_text = str(btn.cget("text"))
                defaults = getattr(self, "_tab_button_defaults", None)
                if defaults:
                    btn.configure(
                        text=self.tab_name,
                        text_color=defaults["text_color"],
                        border_color=defaults["border_color"],
                        border_width=defaults["border_width"],
                    )
                else:
                    btn.configure(text=self.tab_name)
        except Exception:
            pass

    def clear_warning_ui(self):
        self._alert_shown_for_current_timeout = False
        self._set_tab_warning_state(False)
        if hasattr(self, "normal_header_fg") and self.header_frame.winfo_exists():
            self.header_frame.configure(fg_color=self.normal_header_fg)
        if self.clicker.is_running and self.clicker.device and self.status_label.winfo_exists():
            self.status_label.configure(
                text=f"Status: Running ({self.clicker.device_address})",
                text_color="green",
            )

    def check_tab_warning_status(self):
        if self._destroyed:
            return
        if not self.clicker.is_running or self.clicker.no_match_timeout <= 0:
            self._set_tab_warning_state(False)
            return
        elapsed = time.monotonic() - self.clicker.last_action_time
        if elapsed >= self.clicker.no_match_timeout:
            self._set_tab_warning_state(True)
        else:
            self._set_tab_warning_state(False)

    def _render_timer_label(self, key, widget, text, color):
        if widget is None or not widget.winfo_exists():
            return
        state = (text, color)
        if self._timer_render_state.get(key) == state:
            return
        self._timer_render_state[key] = state
        widget.configure(text=text, text_color=color)

    def update_timer_display(self):
        if self._destroyed or not self.winfo_exists():
            return

        status = self.clicker.get_timers_status()
        is_running = status["is_running"]
        action = status["no_match_action"]
        interval = status["no_match_interval"]
        no_match_remaining = status["no_match_remaining"]
        no_match_elapsed = status["no_match_elapsed"]
        timeout = status["timeout"]
        timeout_remaining = status["timeout_remaining"]
        timeout_elapsed = status["timeout_elapsed"]
        last_template = status["last_matched_template"]
        last_is_fallback = status["last_matched_is_fallback"]
        last_elapsed = status["last_match_elapsed"]

        action_name = NO_MATCH_ACTION_MAP.get(action, action)
        short_action = action_name.split("(")[0].strip()
        if not is_running:
            no_match_text = (
                f"⚡ 미매칭 대기: 정지됨 (설정: {interval:.0f}s)"
                if interval > 0
                else "⚡ 미매칭 대기: 정지됨"
            )
            no_match_color = "gray"
            fallback_text = f"[설정: {interval:.0f}s]"
            fallback_color = "gray"
        elif action == "none" or interval <= 0:
            no_match_text = "⚡ 미매칭 동작: 사용 안 함 (OFF)"
            no_match_color = "gray"
            fallback_text = "[미매칭 동작 미사용]"
            fallback_color = "gray"
        elif no_match_remaining <= 0.05:
            no_match_text = f"⚡ 미매칭 동작: 실행 대기 중... ({short_action})"
            no_match_color = "#2ECC71"
            fallback_text = "⚡ 검사 실행 대기 중..."
            fallback_color = no_match_color
        else:
            no_match_text = (
                f"⚡ 미매칭 대기: {no_match_remaining:.1f}초 남음 "
                f"({no_match_elapsed:.1f}/{interval:.0f}s)"
            )
            no_match_color = "#5DADE2"
            fallback_text = (
                f"⚡ 검사 대기: {no_match_remaining:.1f}s / {interval:.0f}s"
            )
            fallback_color = no_match_color

        warning_active = bool(
            is_running and timeout > 0 and timeout_elapsed >= timeout
        )
        if not is_running:
            timeout_text = (
                f"⚠️ 매칭없음 경고: 정지됨 (설정: {timeout:.0f}s)"
                if timeout > 0
                else "⚠️ 매칭없음 경고: 비활성화"
            )
            timeout_color = "gray"
        elif timeout <= 0:
            timeout_text = "⚠️ 매칭없음 경고: 비활성화 (OFF)"
            timeout_color = "gray"
        elif warning_active:
            timeout_text = (
                f"⚠️ 매칭없음 경고: 경고 발생 중! "
                f"({timeout_elapsed:.1f}s 경과)"
            )
            timeout_color = "#FF4D4D"
        else:
            timeout_text = (
                f"⚠️ 매칭없음 경고: {timeout_remaining:.1f}초 남음 "
                f"({timeout_elapsed:.1f}/{timeout:.0f}s)"
            )
            timeout_color = "#F39C12" if timeout_remaining < 15 else "#D5D8DC"

        self._set_tab_warning_state(warning_active)
        if not warning_active:
            self._alert_shown_for_current_timeout = False
            if (
                hasattr(self, "normal_header_fg")
                and self.header_frame.cget("fg_color") != self.normal_header_fg
            ):
                self.header_frame.configure(fg_color=self.normal_header_fg)
            if (
                is_running
                and self.clicker.device
                and self.status_label.winfo_exists()
                and str(self.status_label.cget("text_color")).lower() in ("#ff4d4d", "red")
            ):
                self.status_label.configure(
                    text=f"Status: Running ({self.clicker.device_address})",
                    text_color="green",
                )

        consecutive_count = status.get("consecutive_match_count", 0)
        streak_info = f" (연속 {consecutive_count}회)" if consecutive_count > 1 else ""
        if not is_running:
            recent_text = "🎯 최근 매칭: 대기 중"
            recent_color = "gray"
        elif last_template and last_elapsed is not None:
            tag = " [복구]" if last_is_fallback else ""
            recent_text = (
                f"🎯 최근 매칭: {last_template}{tag}{streak_info} "
                f"({last_elapsed:.0f}초 전)"
            )
            recent_color = "#2ECC71" if last_elapsed < 3 else "#BDC3C7"
        else:
            recent_text = "🎯 최근 매칭: 아직 없음"
            recent_color = "#A6ACAF"

        self._render_timer_label(
            "no_match", self.no_match_timer_label, no_match_text, no_match_color
        )
        self._render_timer_label(
            "fallback",
            getattr(self, "fallback_timer_sublabel", None),
            fallback_text,
            fallback_color,
        )
        self._render_timer_label(
            "timeout", self.timeout_timer_label, timeout_text, timeout_color
        )
        self._render_timer_label(
            "recent", self.last_match_info_label, recent_text, recent_color
        )

    def on_consecutive_match_warning(self, filename, count):
        generation = self._loop_generation

        def show_warning_ui():
            if (
                self._destroyed
                or generation != self._loop_generation
                or not self.clicker.is_running
                or not self.winfo_exists()
            ):
                return
            self._set_tab_warning_state(True)
            if not hasattr(self, "normal_header_fg"):
                self.normal_header_fg = self.header_frame.cget("fg_color")
            self.header_frame.configure(fg_color="#5A1A1A")
            self.status_label.configure(
                text=f"⚠️ 경고: '{filename}' 템플릿 {count}회 연속 매칭! ({self.clicker.device_address})",
                text_color="#FF4D4D",
            )
            try:
                self.app_owner.tabview.set(self.tab_name)
            except Exception:
                pass

            if not self.is_alert_open:
                self.is_alert_open = True
                from tkinter import messagebox
                try:
                    messagebox.showwarning(
                        f"⚠️ 연속 매칭 경고 - [{self.tab_name}]",
                        f"⚠️ 경고 발생 탭: {self.tab_name}\n"
                        f"📱 디바이스 주소: {self.clicker.device_address}\n\n"
                        f"⚠️ '{filename}' 템플릿이 연속으로 {count}회 매칭되었습니다!\n"
                        "화면이 멈춰있거나 무한 루프 상태인지 확인해 주세요.",
                    )
                finally:
                    self.is_alert_open = False

        self.app_owner.post_to_ui(show_warning_ui)

    def on_no_match_timeout(self, timeout_sec):
        generation = self._loop_generation

        def show_warning_ui():
            if (
                self._destroyed
                or generation != self._loop_generation
                or not self.clicker.is_running
                or not self.winfo_exists()
            ):
                return
            self._set_tab_warning_state(True)
            if not hasattr(self, "normal_header_fg"):
                self.normal_header_fg = self.header_frame.cget("fg_color")
            self.header_frame.configure(fg_color="#5A1A1A")
            self.status_label.configure(
                text=f"⚠️ 경고: {timeout_sec}초간 매칭 미발생! ({self.clicker.device_address})",
                text_color="#FF4D4D",
            )
            try:
                self.app_owner.tabview.set(self.tab_name)
            except Exception:
                pass

            if not self._alert_shown_for_current_timeout and not self.is_alert_open:
                self._alert_shown_for_current_timeout = True
                self.is_alert_open = True
                from tkinter import messagebox
                try:
                    messagebox.showwarning(
                        f"⚠️ 타임아웃 경고 - [{self.tab_name}]",
                        f"⚠️ 경고 발생 탭: {self.tab_name}\n"
                        f"📱 디바이스 주소: {self.clicker.device_address}\n\n"
                        f"⚠️ 설정된 {timeout_sec}초 동안 어떠한 템플릿도 매칭되지 않았습니다!\n"
                        "해당 에뮬레이터 화면 및 상태를 확인해 주세요.",
                    )
                finally:
                    self.is_alert_open = False

        self.app_owner.post_to_ui(show_warning_ui)

    def toggle_connection(self):
        self.save_settings()
        if self.clicker.device is not None:
            self._connection_generation += 1
            self.clicker.disconnect()
            self._set_disconnected_ui()
            return

        self._connection_generation += 1
        generation = self._connection_generation
        self.connect_button.configure(state="disabled", text="연결 중...")

        def connect_task():
            success = self.clicker.start_adb_server()

            def apply_result():
                if self._destroyed or generation != self._connection_generation:
                    if success:
                        self.clicker.disconnect()
                    return
                if success:
                    self.status_label.configure(
                        text=f"상태: 연결 완료 ({self.clicker.device_address})",
                        text_color=COLOR_SUCCESS,
                    )
                    self.start_button.configure(
                        state="normal", text="클리커 시작",
                        fg_color=COLOR_SUCCESS, hover_color=COLOR_SUCCESS_HOVER
                    )
                    self.crop_button.configure(state="normal")
                    self.crop_fb_button.configure(state="normal")
                    self.connect_button.configure(
                        state="normal", text="연결 해제",
                        fg_color=COLOR_DANGER, hover_color=COLOR_DANGER_HOVER
                    )
                else:
                    self.status_label.configure(
                        text="상태: 연결 실패", text_color=COLOR_DANGER
                    )
                    self.connect_button.configure(
                        state="normal", text="디바이스 연결",
                        fg_color=COLOR_PRIMARY, hover_color=COLOR_PRIMARY_HOVER
                    )

            self.app_owner.post_to_ui(apply_result)

        threading.Thread(target=connect_task, daemon=True).start()

    def toggle_clicker(self):
        self.save_settings()
        if self.clicker.is_running or self._loop_starting:
            self.stop_clicker_loop()
        else:
            self.start_clicker_loop()

    def start_clicker_loop(self):
        if self.clicker.device is None:
            self.log_message("연결된 디바이스가 없습니다. 연결 후 시작해 주세요.")
            return
        if self.clicker.is_running or self._loop_starting:
            return

        self._loop_starting = True
        self._loop_generation += 1
        generation = self._loop_generation
        cancel_event = threading.Event()
        self._loop_cancel_event = cancel_event
        self._alert_shown_for_current_timeout = False
        self.start_button.configure(
            text="클리커 중지", fg_color=COLOR_DANGER, hover_color=COLOR_DANGER_HOVER
        )
        self.status_label.configure(
            text=f"상태: 실행 중 ({self.clicker.device_address})",
            text_color=COLOR_SUCCESS,
        )

        def run_loop():
            try:
                if generation != self._loop_generation or self._destroyed:
                    cancel_event.set()
                    return

                self.clicker.start_loop(cancel_event=cancel_event)
            finally:
                def finish_loop():
                    self._loop_starting = False
                    self._alert_shown_for_current_timeout = False
                    if self._destroyed or not self.winfo_exists():
                        return
                    self.start_button.configure(
                        text="클리커 시작",
                        fg_color=COLOR_SUCCESS,
                        hover_color=COLOR_SUCCESS_HOVER,
                    )
                    self.status_label.configure(
                        text=(
                            f"상태: 연결 완료 ({self.clicker.device_address})"
                            if self.clicker.device
                            else "상태: 연결 안 됨"
                        ),
                        text_color=COLOR_SUCCESS if self.clicker.device else COLOR_TEXT_MUTED,
                    )

                self.app_owner.post_to_ui(finish_loop)

        self.clicker_thread = threading.Thread(target=run_loop, daemon=True)
        self.clicker_thread.start()

    def stop_clicker_loop(self):
        self._loop_cancel_event.set()
        self._loop_generation += 1
        self._alert_shown_for_current_timeout = False
        if self.clicker.is_running or self._loop_starting:
            self.clicker.stop_loop()
            self.start_button.configure(text="Stopping...")

    def begin_shutdown(self):
        if getattr(self, "_shutdown_started", False) is True:
            return
        self._shutdown_started = True
        self._loop_cancel_event.set()
        self._destroyed = True
        self._connection_generation += 1
        self._loop_generation += 1
        self.clicker.request_shutdown()

    def shutdown(self):
        self.begin_shutdown()
        worker = self.clicker_thread
        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(ADB_COMMAND_TIMEOUT + 1.0)
        deadline = time.monotonic() + ADB_COMMAND_TIMEOUT + 1.0
        with self._action_threads_lock:
            action_workers = tuple(self._action_threads)
        for action_worker in action_workers:
            if action_worker is threading.current_thread():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            action_worker.join(remaining)

        self.clicker.shutdown()

    def clear_search(self, is_fallback=False):
        if is_fallback:
            self.fallback_search_var.set("")
            try:
                self.fallback_search_entry.focus_set()
            except Exception:
                pass
        else:
            self.primary_search_var.set("")
            try:
                self.primary_search_entry.focus_set()
            except Exception:
                pass

    def focus_search_box(self):
        try:
            curr_tab = self.template_tabview.get()
            if "미매칭" in curr_tab:
                self.fallback_search_entry.focus_set()
                self.fallback_search_entry.select_range(0, 'end')
            else:
                self.primary_search_entry.focus_set()
                self.primary_search_entry.select_range(0, 'end')
        except Exception:
            pass

    def filter_templates(self, is_fallback=False):
        if self._destroyed or not self.winfo_exists():
            return
        query = (
            self.fallback_search_var.get() if is_fallback else self.primary_search_var.get()
        ).strip().lower()
        order = self.clicker.fallback_template_order if is_fallback else self.clicker.template_order
        row_dict = self.fallback_template_row_widgets if is_fallback else self.template_row_widgets

        visible_count = 0
        for filename in order:
            if filename in row_dict:
                w = row_dict[filename]
                if not query or query in filename.lower():
                    w.frame.pack(fill="x", padx=5, pady=1)
                    visible_count += 1
                else:
                    w.frame.pack_forget()

        parent_frame = self.fallback_templates_frame if is_fallback else self.templates_frame
        attr_name = "fb_no_match_label" if is_fallback else "primary_no_match_label"
        lbl = getattr(self, attr_name, None)

        if visible_count == 0 and len(order) > 0 and query:
            if lbl is None or not lbl.winfo_exists():
                lbl = ctk.CTkLabel(
                    parent_frame,
                    text="🔍 일치하는 템플릿이 없습니다.",
                    font=ctk.CTkFont(size=12),
                    text_color=COLOR_TEXT_MUTED,
                )
                setattr(self, attr_name, lbl)
            lbl.pack(pady=20)
        else:
            if lbl is not None and lbl.winfo_exists():
                lbl.pack_forget()

    def refresh_templates(self):
        if self._destroyed or not self.winfo_exists():
            return

        TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
        self.clicker.load_config()
        self.template_row_widgets = {}
        self.fallback_template_row_widgets = {}
        self.template_count_labels = {}
        self.fallback_template_count_labels = {}
        self.primary_no_match_label = None
        self.fb_no_match_label = None

        # 1. 기본 템플릿 목록 렌더링
        for widget in self.templates_frame.winfo_children():
            widget.destroy()

        for i, filename in enumerate(self.clicker.template_order):
            self.create_template_row(self.templates_frame, filename, i, is_fallback=False)

        # 2. 미매칭 복구 템플릿 목록 렌더링
        for widget in self.fallback_templates_frame.winfo_children():
            widget.destroy()

        for i, filename in enumerate(self.clicker.fallback_template_order):
            self.create_template_row(self.fallback_templates_frame, filename, i, is_fallback=True)

        if hasattr(self, 'fb_final_combo') and self.fb_final_combo.winfo_exists():
            curr_fb_action = getattr(self.clicker, 'fallback_final_action', 'none')
            curr_fb_label = NO_MATCH_ACTION_MAP.get(curr_fb_action, "사용 안 함 (Disabled)")
            self.fb_final_combo.set(curr_fb_label)
            coords = getattr(self.clicker, 'fallback_final_coords', [500, 500])
            self.fb_final_x_entry.delete(0, "end")
            self.fb_final_x_entry.insert(0, str(coords[0]))
            self.fb_final_y_entry.delete(0, "end")
            self.fb_final_y_entry.insert(0, str(coords[1]))
            self.update_fb_final_coord_visibility(curr_fb_action)

        self.filter_templates(is_fallback=False)
        self.filter_templates(is_fallback=True)

    def create_template_row(self, parent_frame, filename, index, is_fallback=False):
        row_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        row_frame.pack(fill="x", padx=5, pady=1)

        # 1. 우측 고정 버튼 영역 (플랫 구조로 레이아웃 부하 최소화)
        del_cmd = (lambda f=filename: self.delete_fallback_template_event(f)) if is_fallback else (lambda f=filename: self.delete_template_event(f))
        del_btn = ctk.CTkButton(
            row_frame, text="✕", width=26, height=26, 
            fg_color=COLOR_DANGER, hover_color=COLOR_DANGER_HOVER,
            corner_radius=RADIUS_SM, font=ctk.CTkFont(size=11, weight="bold"),
            command=del_cmd
        )
        del_btn.pack(side="right", padx=(3, 0))

        action_dict = self.clicker.fallback_template_actions if is_fallback else self.clicker.template_actions
        action = action_dict.get(filename, "click")
        action_text, action_fg, action_hover = get_action_button_style(action)
        
        toggle_cmd = (lambda f=filename: self.toggle_fallback_action_event(f)) if is_fallback else (lambda f=filename: self.toggle_action_event(f))
        action_btn = ctk.CTkButton(
            row_frame, text=action_text, width=120, height=26,
            fg_color=action_fg, hover_color=action_hover,
            corner_radius=RADIUS_SM, font=ctk.CTkFont(size=11, weight="bold"),
            command=toggle_cmd
        )
        action_btn.pack(side="right", padx=(3, 0))

        delays_dict = self.clicker.fallback_template_delays if is_fallback else self.clicker.template_delays
        delay_types_dict = self.clicker.fallback_template_delay_types if is_fallback else self.clicker.template_delay_types
        delay = delays_dict.get(filename, 0.0)
        delay_type = delay_types_dict.get(filename, "pre")
        delay_text, delay_fg, delay_hover = get_delay_button_style(delay, delay_type)

        delay_cmd = (lambda f=filename: self.set_fallback_template_delay_event(f)) if is_fallback else (lambda f=filename: self.set_template_delay_event(f))
        delay_btn = ctk.CTkButton(
            row_frame, text=delay_text, width=66, height=26,
            fg_color=delay_fg, hover_color=delay_hover,
            corner_radius=RADIUS_SM, font=ctk.CTkFont(size=11),
            command=delay_cmd
        )
        delay_btn.pack(side="right", padx=(3, 0))

        counts_dict = self.clicker.fallback_template_counts if is_fallback else self.clicker.template_counts
        count = counts_dict.get(filename, 0)
        count_label = ctk.CTkLabel(
            row_frame, text=f"{count}회", width=48,
            text_color=COLOR_TEXT_MUTED, anchor="e", font=ctk.CTkFont(size=11)
        )
        count_label.pack(side="right", padx=(2, 6))

        # 2. 좌측 컨트롤 영역
        drag_handle = ctk.CTkLabel(
            row_frame, text="☰", width=20, cursor="fleur",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=COLOR_TEXT_MUTED
        )
        drag_handle.pack(side="left", padx=(3, 0))

        priority_label = ctk.CTkLabel(
            row_frame, text=f"{index+1}.", width=26, anchor="w",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=COLOR_TEXT_MUTED
        )
        priority_label.pack(side="left", padx=(2, 4))

        # 3. 중앙 가변 텍스트 라벨
        offset_dict = self.clicker.fallback_template_offsets if is_fallback else self.clicker.template_offsets
        offset_info = ""
        if filename in offset_dict:
            off_x, off_y = offset_dict[filename]
            offset_info = f"  ({off_x:+d},{off_y:+d})"
        
        label = ctk.CTkLabel(
            row_frame, text=f"{filename}{offset_info}", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_TEXT_PRIMARY
        )
        label.pack(side="left", fill="x", expand=True, padx=(0, 4))

        target_dir = self.clicker.fallback_template_dir if is_fallback else self.clicker.template_dir
        template_file_path = os.path.join(target_dir, filename)

        def on_enter(e):
            tooltip = TemplatePreviewTooltip.get_instance(self.winfo_toplevel())
            tooltip.schedule_show(label, template_file_path, filename, offset_info.strip())

        def on_leave(e):
            tooltip = TemplatePreviewTooltip.get_instance(self.winfo_toplevel())
            tooltip.cancel()

        # Right-click Context Menu (우클릭 컨텍스트 메뉴)
        def show_context_menu(event):
            TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
            menu = CTKContextMenu(self)
            menu.add_command(
                "이름 변경 (Rename)",
                lambda f=filename, fb=is_fallback: RenameTemplateWindow(self, f, fb),
            )
            menu.add_command(
                "즉시 실행 (Run Now)",
                lambda f=filename, fb=is_fallback: self.on_template_double_click(f, fb),
            )
            menu.add_command(
                "지연 시간 설정 (Set Delay)",
                lambda f=filename, fb=is_fallback: (self.set_fallback_template_delay_event(f) if fb else self.set_template_delay_event(f)),
            )
            menu.add_separator()
            menu.add_command(
                "템플릿 삭제 (Delete)",
                lambda f=filename, fb=is_fallback: (self.delete_fallback_template_event(f) if fb else self.delete_template_event(f)),
                is_danger=True,
            )
            pop_x = event.x_root if hasattr(event, "x_root") and event.x_root > 0 else self.winfo_pointerx()
            pop_y = event.y_root if hasattr(event, "y_root") and event.y_root > 0 else self.winfo_pointery()
            menu.show(pop_x, pop_y)

        for target in (row_frame, label, priority_label, drag_handle):
            target.bind("<Button-3>", show_context_menu)
            target.bind("<Button-2>", show_context_menu)

        # 이벤트 바인딩
        drag_handle.bind("<ButtonPress-1>", lambda e, f=filename, fb=is_fallback: self.on_drag_start(e, f, fb))
        drag_handle.bind("<B1-Motion>", self.on_drag_motion)
        drag_handle.bind("<ButtonRelease-1>", self.on_drag_end)

        label.bind("<Double-Button-1>", lambda e, f=filename, fb=is_fallback: self.on_template_double_click(f, fb))
        label.bind("<Enter>", on_enter, add="+")
        label.bind("<Leave>", on_leave, add="+")

        row_frame.bind("<Double-Button-1>", lambda e, f=filename, fb=is_fallback: self.on_template_double_click(f, fb))

        widgets = TemplateRowWidgets(
            frame=row_frame,
            priority=priority_label,
            drag=drag_handle,
            action_btn=action_btn,
            delay_btn=delay_btn,
            count_label=count_label,
            label=label,
            del_btn=del_btn,
        )

        if is_fallback:
            self.fallback_template_row_widgets[filename] = widgets
            self.fallback_template_count_labels[filename] = count_label
        else:
            self.template_row_widgets[filename] = widgets
            self.template_count_labels[filename] = count_label

    def _start_action_worker(self, target):
        def run():
            try:
                if not self._destroyed:
                    target()
            finally:
                with self._action_threads_lock:
                    self._action_threads.discard(threading.current_thread())

        worker = threading.Thread(target=run, daemon=True)
        with self._action_threads_lock:
            if self._destroyed:
                return None
            self._action_threads.add(worker)
        try:
            worker.start()
        except Exception:
            with self._action_threads_lock:
                self._action_threads.discard(worker)
            raise
        return worker

    def on_template_double_click(self, filename, is_fallback=False):
        TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
        row_dict = self.fallback_template_row_widgets if is_fallback else self.template_row_widgets
        if filename in row_dict:
            frame = row_dict[filename][0]
            frame.configure(fg_color="#2E7D32")
            self.after(250, lambda: frame.configure(fg_color="transparent") if (not self._destroyed and frame.winfo_exists()) else None)

        if self.clicker.device is None:
            self.log_message(f"⚠️ 디바이스가 연결되어 있지 않아 '{filename}' 동작을 실행할 수 없습니다.")
            return

        def task():
            self.clicker.execute_template(filename, is_fallback=is_fallback)

        self._start_action_worker(task)

    def move_template(self, filename, direction, is_fallback=False):
        TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
        order = self.clicker.fallback_template_order if is_fallback else self.clicker.template_order
        if filename not in order:
            return
        idx = order.index(filename)
        new_idx = idx + direction
        if 0 <= new_idx < len(order):
            item = order.pop(idx)
            order.insert(new_idx, item)
            self.clicker.save_config()
            self.app_owner.update_all_template_order(order, is_fallback=is_fallback)

    def on_drag_start(self, event, filename, is_fallback=False):
        TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
        self.drag_source_filename = filename
        self.drag_target_filename = filename
        self.drag_is_fallback = is_fallback
        row_dict = self.fallback_template_row_widgets if is_fallback else self.template_row_widgets
        order = self.clicker.fallback_template_order if is_fallback else self.clicker.template_order

        # Cache row positions ONCE on drag start to eliminate winfo queries during drag motion
        self._drag_positions = []
        for f in order:
            if f in row_dict:
                frame = row_dict[f][0]
                try:
                    if frame.winfo_exists():
                        y = frame.winfo_rooty()
                        h = frame.winfo_height()
                        self._drag_positions.append((y, y + (h if h > 0 else 30), f))
                except Exception:
                    pass

        if filename in row_dict:
            frame = row_dict[filename][0]
            frame.configure(fg_color="#1F6FE5")

    def on_drag_motion(self, event):
        if not getattr(self, 'drag_source_filename', None):
            return
        
        y = event.y_root
        is_fallback = getattr(self, 'drag_is_fallback', False)
        row_dict = self.fallback_template_row_widgets if is_fallback else self.template_row_widgets
        
        # Fast lookup in cached positions
        target = None
        for y_start, y_end, f in getattr(self, '_drag_positions', []):
            if y_start <= y <= y_end:
                target = f
                break

        prev_target = getattr(self, 'drag_target_filename', None)
        if target and target != prev_target:
            if prev_target and prev_target != self.drag_source_filename and prev_target in row_dict:
                try:
                    row_dict[prev_target][0].configure(fg_color="transparent")
                except Exception:
                    pass
            self.drag_target_filename = target
            if target != self.drag_source_filename and target in row_dict:
                try:
                    row_dict[target][0].configure(fg_color="#3A3A3A")
                except Exception:
                    pass

    def on_drag_end(self, event):
        src = getattr(self, 'drag_source_filename', None)
        tgt = getattr(self, 'drag_target_filename', None)
        is_fallback = getattr(self, 'drag_is_fallback', False)
        row_dict = self.fallback_template_row_widgets if is_fallback else self.template_row_widgets
        
        self.drag_source_filename = None
        self.drag_target_filename = None
        self.drag_is_fallback = False
        self._drag_positions = []

        if src and src in row_dict:
            row_dict[src][0].configure(fg_color="transparent")
        if tgt and tgt in row_dict:
            row_dict[tgt][0].configure(fg_color="transparent")

        order = self.clicker.fallback_template_order if is_fallback else self.clicker.template_order
        if src and tgt and src in order and tgt in order and src != tgt:
            src_idx = order.index(src)
            tgt_idx = order.index(tgt)
            item = order.pop(src_idx)
            order.insert(tgt_idx, item)
            self.clicker.save_config()
            self.app_owner.update_all_template_order(order, is_fallback=is_fallback)

    def toggle_action_event(self, filename):
        new_action = self.clicker.toggle_action(filename)
        self.app_owner.update_all_template_actions(filename, new_action, is_fallback=False)

    def toggle_fallback_action_event(self, filename):
        new_action = self.clicker.toggle_fallback_action(filename)
        self.app_owner.update_all_template_actions(filename, new_action, is_fallback=True)

    def set_template_delay_event(self, filename):
        TemplateDelayWindow(self, filename, is_fallback=False)

    def set_fallback_template_delay_event(self, filename):
        TemplateDelayWindow(self, filename, is_fallback=True)

    def reset_counts_event(self):
        self.clicker.reset_counts()
        self.app_owner.reset_all_template_counts(is_fallback=False)

    def reset_fallback_counts_event(self):
        self.clicker.reset_fallback_counts()
        self.app_owner.reset_all_template_counts(is_fallback=True)

    def delete_template_event(self, filename):
        TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
        if self.clicker.delete_template(filename):
            self.app_owner.remove_template_from_all_tabs(filename, is_fallback=False)

    def delete_fallback_template_event(self, filename):
        TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
        if self.clicker.delete_fallback_template(filename):
            self.app_owner.remove_template_from_all_tabs(filename, is_fallback=True)

    def on_fallback_final_action_changed(self, choice):
        action_key = REVERSE_NO_MATCH_ACTION_MAP.get(choice, "none")
        self.update_fb_final_coord_visibility(action_key)
        self.save_fallback_final_settings()

    def update_fb_final_coord_visibility(self, action_key):
        if action_key in ("custom_click", "custom_double_click"):
            self.fb_final_coord_frame.pack(side="left", padx=(0, 5), pady=6)
        else:
            self.fb_final_coord_frame.pack_forget()

    def save_fallback_final_settings(self, event=None):
        action_key = REVERSE_NO_MATCH_ACTION_MAP.get(self.fb_final_combo.get(), "none")
        self.clicker.fallback_final_action = action_key
        try:
            x = int(self.fb_final_x_entry.get().strip())
            y = int(self.fb_final_y_entry.get().strip())
            self.clicker.fallback_final_coords = [x, y]
        except ValueError:
            pass
        self.app_owner.sync_shared_settings_to_all(self)
        self.app_owner.save_app_config()

    def pick_fallback_final_coords(self):
        try:
            initial_x = int(self.fb_final_x_entry.get().strip())
            initial_y = int(self.fb_final_y_entry.get().strip())
        except ValueError:
            initial_x = initial_y = None

        app = self.app_owner
        if not app.opencv_lock.acquire(blocking=False):
            self.log_message("다른 화면 선택 창이 이미 열려 있습니다.")
            return

        self.fb_final_pick_btn.configure(state="disabled")

        def finish_picker():
            app.opencv_lock.release()
            def enable_button():
                if not self._destroyed and self.winfo_exists():
                    self.fb_final_pick_btn.configure(state="normal")
            app.post_to_ui(enable_button)

        def pick_task():
            window_name = None
            try:
                screen = self.clicker.capture_screen(grayscale=False)
                if screen is None:
                    self.log_message("좌표 선택용 화면 캡처에 실패했습니다.")
                    return

                height, width = screen.shape[:2]
                selected_pt = [
                    width // 2 if initial_x is None else max(0, min(width - 1, initial_x)),
                    height // 2 if initial_y is None else max(0, min(height - 1, initial_y)),
                ]
                window_name = "[Pick Fallback Final Coordinates] Click point -> Enter to confirm"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, width // 2, (height + 60) // 2)

                def update_preview():
                    display = screen.copy()
                    x, y = selected_pt
                    cv2.circle(display, (x, y), 8, (0, 0, 255), 2)
                    cv2.drawMarker(display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                    banner = np.full((60, width, 3), 30, dtype=np.uint8)
                    cv2.putText(banner, f"Selected: ({x}, {y})", (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(banner,
                                "Click screen -> Enter/Space: Confirm, 'c'/ESC: Cancel",
                                (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (220, 220, 220), 1, cv2.LINE_AA)
                    cv2.imshow(window_name, np.vstack([display, banner]))

                def on_mouse(event, mouse_x, mouse_y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN and 0 <= mouse_y < height and 0 <= mouse_x < width:
                        selected_pt[:] = [mouse_x, mouse_y]
                        update_preview()

                cv2.setMouseCallback(window_name, on_mouse)
                update_preview()
                cancelled = False
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (13, 32):
                        break
                    if key in (ord("c"), ord("C"), 27):
                        cancelled = True
                        break
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            cancelled = True
                            break
                    except cv2.error:
                        cancelled = True
                        break

                if not cancelled:
                    def update_entries():
                        if self._destroyed or not self.winfo_exists():
                            return
                        self.fb_final_x_entry.delete(0, "end")
                        self.fb_final_x_entry.insert(0, str(selected_pt[0]))
                        self.fb_final_y_entry.delete(0, "end")
                        self.fb_final_y_entry.insert(0, str(selected_pt[1]))
                        self.save_fallback_final_settings()

                    app.post_to_ui(update_entries)
            except Exception as error:
                self.log_message(f"좌표 선택 중 오류: {error}")
            finally:
                if window_name:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass
                finish_picker()

        threading.Thread(target=pick_task, daemon=True).start()

    def test_fallback_final_action(self):
        if self.clicker.device is None:
            self.log_message("⚠️ 디바이스가 연결되어 있지 않아 실행할 수 없습니다.")
            return

        action_key = REVERSE_NO_MATCH_ACTION_MAP.get(self.fb_final_combo.get(), "none")
        if action_key == "none":
            self.log_message("⚠️ 최종 동작이 '사용 안 함'으로 설정되어 있습니다.")
            return

        def task():
            screen = self.clicker.capture_screen()
            if screen is None:
                self.log_message("화면 캡처 실패")
                return
            self.clicker._execute_final_action(screen, action_key)

        self._start_action_worker(task)

    def open_settings_window(self):
        if self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self)
        else:
            self.settings_window.focus()

    def _prompt_and_save_template(
        self,
        crop_img,
        offset_x,
        offset_y,
        is_fallback,
        search_roi=None,
    ):
        target_type = "미매칭 복구 템플릿" if is_fallback else "기본 템플릿"
        target_dir = (
            self.clicker.fallback_template_dir
            if is_fallback
            else self.clicker.template_dir
        )
        gui_mod = sys.modules.get("gui_app")
        ctk_mod = getattr(gui_mod, "ctk", ctk) if gui_mod else ctk
        dialog_cls = getattr(ctk_mod, "CTkInputDialog", ctk.CTkInputDialog)
        dialog = dialog_cls(
            text=f"저장할 {target_type} 이름을 입력하세요 (예: close_btn):",
            title=f"{target_type} 저장",
        )
        file_name = dialog.get_input()
        if not file_name:
            self.log_message("저장 취소됨")
            return

        file_name = file_name.strip()
        invalid_chars = set('<>:"/|?*') | {chr(92)}
        stem = os.path.splitext(file_name)[0].rstrip(" .")
        reserved = {"CON", "PRN", "AUX", "NUL"} | {
            f"{prefix}{number}"
            for prefix in ("COM", "LPT")
            for number in range(1, 10)
        }
        if (
            not file_name
            or os.path.basename(file_name) != file_name
            or any(char in invalid_chars for char in file_name)
            or not stem
            or stem.upper() in reserved
            or file_name != file_name.rstrip(" .")
        ):
            from tkinter import messagebox
            messagebox.showerror("잘못된 이름", "경로 문자나 Windows 예약 이름은 사용할 수 없습니다.")
            return

        if not file_name.lower().endswith(".png"):
            file_name += ".png"
        save_path = os.path.join(target_dir, file_name)

        if os.path.exists(save_path):
            from tkinter import messagebox
            if not messagebox.askyesno(
                "덮어쓰기 확인", f"{file_name} 파일이 이미 있습니다. 덮어쓸까요?"
            ):
                return

        temp_path = f"{save_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            encoded_ok, encoded = cv2.imencode(".png", crop_img)
            if not encoded_ok:
                raise ValueError("이미지 인코딩 실패")
            encoded.tofile(temp_path)
            os.replace(temp_path, save_path)
            self.clicker._discard_location_hint(save_path)
            self.clicker.invalidate_template_cache(save_path)
            self.clicker.preload_templates(
                os.path.dirname(save_path),
                grayscales=(bool(self.clicker.match_grayscale),),
            )

            if is_fallback:
                order = self.clicker.fallback_template_order
                counts = self.clicker.fallback_template_counts
                actions = self.clicker.fallback_template_actions
                offsets = self.clicker.fallback_template_offsets
                delays = self.clicker.fallback_template_delays
                delay_types = self.clicker.fallback_template_delay_types
                rois = self.clicker.fallback_template_rois
            else:
                order = self.clicker.template_order
                counts = self.clicker.template_counts
                actions = self.clicker.template_actions
                offsets = self.clicker.template_offsets
                delays = self.clicker.template_delays
                delay_types = self.clicker.template_delay_types
                rois = self.clicker.template_rois

            if file_name not in order:
                order.append(file_name)
            counts.setdefault(file_name, 0)
            actions.setdefault(file_name, "click")
            offsets[file_name] = [offset_x, offset_y]
            delays.setdefault(file_name, 0.0)
            delay_types.setdefault(file_name, "pre")
            validated_roi = self.clicker._safe_rois({file_name: search_roi})
            if file_name in validated_roi:
                rois[file_name] = validated_roi[file_name]
            if not self.clicker.save_config(include_templates=True):
                raise OSError(
                    "이미지는 저장됐지만 config.json 갱신에 실패했습니다."
                )
            self.log_message(
                f"[{target_type}] 저장 완료: {save_path} "
                f"(클릭 오프셋: {offset_x:+d}, {offset_y:+d})"
            )
            self.app_owner.add_template_to_all_tabs(
                file_name, offset_x, offset_y, is_fallback=is_fallback
            )
        except Exception as error:
            self.log_message(f"저장 중 오류: {error}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def start_cropping(self, is_fallback=False):
        if getattr(self, "_crop_in_progress", False):
            self.log_message("이미 템플릿 선택 작업이 진행 중입니다.")
            return
        if not self.app_owner.opencv_lock.acquire(blocking=False):
            self.log_message("다른 화면 선택 창이 이미 열려 있습니다.")
            return

        self._crop_in_progress = True
        self.crop_button.configure(state="disabled")
        self.crop_fb_button.configure(state="disabled")
        target_type = "미매칭 복구 템플릿" if is_fallback else "기본 템플릿"

        def finish_crop():
            self.app_owner.opencv_lock.release()

            def update_buttons():
                self._crop_in_progress = False
                if self._destroyed or not self.winfo_exists():
                    return
                state = "normal" if self.clicker.device is not None else "disabled"
                self.crop_button.configure(state=state)
                self.crop_fb_button.configure(state=state)

            self.app_owner.post_to_ui(update_buttons)

        def crop_task():
            open_windows = []
            try:
                self.log_message(f"[{target_type}] 화면을 가져오는 중...")
                screen = self.clicker.capture_screen(grayscale=False)
                if screen is None:
                    self.log_message("캡처 실패")
                    return

                height, width = screen.shape[:2]

                try:
                    top_win = self.winfo_toplevel()
                    win_x = top_win.winfo_x()
                    win_y = top_win.winfo_y()
                    win_w = top_win.winfo_width()
                    win_h = top_win.winfo_height()
                    target_pos_x = max(20, win_x + (win_w - width // 2) // 2)
                    target_pos_y = max(20, win_y + (win_h - height // 2) // 2)
                except Exception:
                    target_pos_x = 100
                    target_pos_y = 100

                banner_height = 65
                step1_key = f"AutoClicker_Step1_{self.clicker.device_address}"
                step1_title = (
                    f"[1단계] 템플릿 인식 영역 드래그 선택 "
                    f"({'복구' if is_fallback else '기본'}) - "
                    f"{self.clicker.device_address}"
                )
                step1_window = step1_key
                open_windows.append(step1_window)
                cv2.namedWindow(step1_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    step1_window, width // 2, (height + banner_height) // 2
                )
                cv2.moveWindow(step1_window, target_pos_x, target_pos_y)
                set_opencv_window_title(step1_window, step1_title)

                roi_drag = {
                    "dragging": False,
                    "start": None,
                    "rect": None,
                }

                def update_step1_preview():
                    display = screen.copy()
                    rect = roi_drag["rect"]
                    if rect is not None:
                        rx, ry, rw, rh = rect
                        cv2.rectangle(
                            display,
                            (rx, ry),
                            (rx + rw, ry + rh),
                            (0, 255, 0),
                            2,
                        )
                    if rect is not None and rect[2] > 0 and rect[3] > 0:
                        rx, ry, rw, rh = rect
                        roi_info = f"선택 영역: (X={rx}, Y={ry}) 크기={rw}x{rh} 픽셀"
                    else:
                        roi_info = "마우스로 화면에서 인식할 템플릿 영역을 드래그하세요"

                    banner = draw_korean_banner(
                        width,
                        banner_height,
                        [
                            (roi_info, (0, 255, 0), 15),
                            (
                                "영역 드래그 ➔ Enter/Space: 다음 단계 (취소: 'c', ESC 또는 [X])",
                                (220, 220, 220),
                                13,
                            ),
                        ],
                    )
                    cv2.imshow(step1_window, np.vstack([display, banner]))
                    set_opencv_window_title(step1_window, step1_title)

                def on_step1_mouse(event, mx, my, flags, param):
                    if (
                        event == cv2.EVENT_LBUTTONDOWN
                        and 0 <= my < height
                        and 0 <= mx < width
                    ):
                        roi_drag["dragging"] = True
                        roi_drag["start"] = (mx, my)
                        roi_drag["rect"] = (mx, my, 0, 0)
                        update_step1_preview()
                    elif event == cv2.EVENT_MOUSEMOVE and roi_drag["dragging"]:
                        cx = min(max(0, mx), width - 1)
                        cy = min(max(0, my), height - 1)
                        sx, sy = roi_drag["start"]
                        rx = min(sx, cx)
                        ry = min(sy, cy)
                        rw = abs(cx - sx)
                        rh = abs(cy - sy)
                        roi_drag["rect"] = (rx, ry, rw, rh)
                        update_step1_preview()
                    elif event == cv2.EVENT_LBUTTONUP and roi_drag["dragging"]:
                        roi_drag["dragging"] = False
                        cx = min(max(0, mx), width - 1)
                        cy = min(max(0, my), height - 1)
                        sx, sy = roi_drag["start"]
                        rx = min(sx, cx)
                        ry = min(sy, cy)
                        rw = abs(cx - sx)
                        rh = abs(cy - sy)
                        roi_drag["rect"] = (
                            (rx, ry, rw, rh) if rw > 2 and rh > 2 else None
                        )
                        update_step1_preview()

                cv2.setMouseCallback(step1_window, on_step1_mouse)
                update_step1_preview()

                cancelled_step1 = False
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (13, 32):
                        if (
                            roi_drag["rect"] is not None
                            and roi_drag["rect"][2] > 0
                            and roi_drag["rect"][3] > 0
                        ):
                            break
                    if key in (ord("c"), ord("C"), 27):
                        cancelled_step1 = True
                        break
                    try:
                        if (
                            cv2.getWindowProperty(
                                step1_window, cv2.WND_PROP_VISIBLE
                            ) < 1
                        ):
                            cancelled_step1 = True
                            break
                    except cv2.error:
                        cancelled_step1 = True
                        break

                def safe_close_window(wname):
                    if wname in open_windows:
                        open_windows.remove(wname)
                    try:
                        cv2.destroyWindow(wname)
                    except Exception:
                        pass

                safe_close_window(step1_window)
                if cancelled_step1 or roi_drag["rect"] is None:
                    self.log_message("영역 선택 취소됨")
                    return

                roi_x, roi_y, roi_width, roi_height = map(
                    int, roi_drag["rect"]
                )
                crop_img = screen[
                    roi_y:roi_y + roi_height,
                    roi_x:roi_x + roi_width,
                ].copy()
                banner_height = 65
                step2_key = f"AutoClicker_Step2_{self.clicker.device_address}"
                step2_title = (
                    f"[2단계] 실제 클릭할 타겟 위치 선택 "
                    f"({'복구' if is_fallback else '기본'}) - "
                    f"{self.clicker.device_address}"
                )
                step2_window = step2_key
                open_windows.append(step2_window)
                cv2.namedWindow(step2_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    step2_window, width // 2, (height + banner_height) // 2
                )
                cv2.moveWindow(step2_window, target_pos_x, target_pos_y)
                set_opencv_window_title(step2_window, step2_title)
                default_point = [
                    roi_x + roi_width // 2,
                    roi_y + roi_height // 2,
                ]
                selected_point = list(default_point)

                def update_preview():
                    display = screen.copy()
                    cv2.rectangle(
                        display,
                        (roi_x, roi_y),
                        (roi_x + roi_width, roi_y + roi_height),
                        (0, 255, 0),
                        2,
                    )
                    x, y = selected_point
                    cv2.circle(display, (x, y), 8, (0, 0, 255), 2)
                    cv2.drawMarker(
                        display, (x, y), (0, 0, 255),
                        cv2.MARKER_CROSS, 22, 2
                    )
                    offset_x, offset_y = x - roi_x, y - roi_y
                    position_type = (
                        " [기본 중앙 클릭]"
                        if selected_point == default_point
                        else " [사용자 지정 위치]"
                    )
                    banner = draw_korean_banner(
                        width,
                        banner_height,
                        [
                            (
                                f"클릭 좌표: ({x}, {y}) | 상대 오프셋: ({offset_x:+d}, {offset_y:+d}){position_type}",
                                (0, 255, 255),
                                15,
                            ),
                            (
                                "화면 클릭으로 위치 지정 ➔ Enter/Space: 저장 완료 (기본 중앙은 바로 Enter)",
                                (220, 220, 220),
                                13,
                            ),
                        ],
                    )
                    cv2.imshow(step2_window, np.vstack([display, banner]))
                    set_opencv_window_title(step2_window, step2_title)

                def on_mouse(event, mouse_x, mouse_y, flags, param):
                    if (
                        event == cv2.EVENT_LBUTTONDOWN
                        and 0 <= mouse_y < height
                        and 0 <= mouse_x < width
                    ):
                        selected_point[:] = [mouse_x, mouse_y]
                        update_preview()

                cv2.setMouseCallback(step2_window, on_mouse)
                update_preview()
                cancelled = False
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (13, 32):
                        break
                    if key in (ord("c"), ord("C"), 27):
                        cancelled = True
                        break
                    try:
                        if cv2.getWindowProperty(
                            step2_window, cv2.WND_PROP_VISIBLE
                        ) < 1:
                            cancelled = True
                            break
                    except cv2.error:
                        cancelled = True
                        break

                safe_close_window(step2_window)
                if cancelled:
                    self.log_message("클릭 위치 선택이 취소되었습니다.")
                    return

                offset_x = selected_point[0] - roi_x
                offset_y = selected_point[1] - roi_y
                search_margin_x = max(48, roi_width)
                search_margin_y = max(48, roi_height)
                search_roi = [
                    max(0, roi_x - search_margin_x) / width,
                    max(0, roi_y - search_margin_y) / height,
                    min(width, roi_x + roi_width + search_margin_x) / width,
                    min(height, roi_y + roi_height + search_margin_y) / height,
                ]
                self.app_owner.post_to_ui(
                    lambda: self._prompt_and_save_template(
                        crop_img, offset_x, offset_y, is_fallback, search_roi
                    )
                )
            except Exception as error:
                self.log_message(f"템플릿 선택 중 오류: {error}")
            finally:
                for window_name in open_windows:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass
                finish_crop()

        threading.Thread(target=crop_task, daemon=True).start()
