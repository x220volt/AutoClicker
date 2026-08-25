import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import threading
import time
import cv2
import numpy as np
import queue
from contextlib import ExitStack

try:
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from main import (
    AutoClicker,
    TeraboxClicker,
    CONFIG_PATH,
    ADB_COMMAND_TIMEOUT,
    resolve_adb_path,
    DEFAULT_ADB_MODE,
)

VERSION = "v0.3.11"

# 앱 전역 테마를 Dark 모드로 고정
ctk.set_appearance_mode("Dark")

# --- Unified Modern Dark Design System Color & Style Tokens ---
# Eye-friendly, soft muted tones designed for long-session dark mode

COLOR_PRIMARY = "#2D5F9E"          # Soft Royal Blue - Primary Actions / Focus / Connect
COLOR_PRIMARY_HOVER = "#3B73BD"

COLOR_SUCCESS = "#257A4E"          # Deep Emerald Green - Start / Active / Single Click
COLOR_SUCCESS_HOVER = "#2FA066"

COLOR_DANGER = "#8A2E3B"           # Soft Crimson Wine - Stop / Disconnect / Delete (Not glaring/neon)
COLOR_DANGER_HOVER = "#A43949"

COLOR_DANGER_MUTED = "#48262D"     # Subdued Dark Wine - Secondary actions like Reset Counts
COLOR_DANGER_MUTED_HOVER = "#5C3039"

COLOR_WARNING = "#915C16"          # Warm Ochre Amber - Back Action / Pre-Delay / Warnings
COLOR_WARNING_HOVER = "#AB6D1B"

COLOR_INFO = "#49528F"             # Slate Indigo - Post-Delay / Special Actions
COLOR_INFO_HOVER = "#5864AD"

COLOR_NEUTRAL = "#2D3442"          # Dark Slate - Settings / Neutral / Utility
COLOR_NEUTRAL_HOVER = "#3C4557"

COLOR_CARD_BG = ("#2B2F3A", "#181B22")
COLOR_CARD_INNER = ("#333945", "#212631")
COLOR_BORDER = "#2E3545"

COLOR_TEXT_MUTED = "#94A3B8"
COLOR_TEXT_PRIMARY = "#F1F5F9"

RADIUS_SM = 5
RADIUS_MD = 6
RADIUS_LG = 8

NO_MATCH_ACTION_MAP = {
    "fallback_list": "미매칭 복구 템플릿 목록 매칭",
    "none": "사용 안 함 (Disabled)",
    "random_click": "화면 랜덤 클릭 (Random Click)",
    "custom_click": "특정 좌표 클릭 (Custom Click)",
    "custom_double_click": "특정 좌표 더블클릭 (Double Click)",
    "back": "뒤로가기 (Back Key)"
}
REVERSE_NO_MATCH_ACTION_MAP = {v: k for k, v in NO_MATCH_ACTION_MAP.items()}


def get_action_button_style(action):
    if action == "back":
        return "뒤로 (Back)", COLOR_WARNING, COLOR_WARNING_HOVER
    elif action in ("double_click", "click_click", "double"):
        return "더블 (Double)", COLOR_PRIMARY, COLOR_PRIMARY_HOVER
    else:
        return "클릭 (Click)", COLOR_SUCCESS, COLOR_SUCCESS_HOVER


def get_delay_button_style(delay, delay_type="pre"):
    if delay > 0:
        if delay_type == "post":
            return f"동작후 {delay:g}s", COLOR_INFO, COLOR_INFO_HOVER
        else:
            return f"동작전 {delay:g}s", COLOR_WARNING, COLOR_WARNING_HOVER
    else:
        return "딜레이 0s", COLOR_NEUTRAL, COLOR_NEUTRAL_HOVER


class TemplateRowWidgets:
    """Container for widgets in a template row, maintaining backward-compatible indexing."""
    def __init__(self, frame, priority, drag, action_btn, delay_btn, count_label, label, del_btn):
        self.frame = frame
        self.priority = priority
        self.drag = drag
        self.action_btn = action_btn
        self.delay_btn = delay_btn
        self.count_label = count_label
        self.label = label
        self.del_btn = del_btn

    def __getitem__(self, index):
        if index == 0:
            return self.frame
        elif index == 1:
            return self.priority
        elif index == 2:
            return self.drag
        raise IndexError(f"Index {index} out of range for TemplateRowWidgets")


class CTKContextMenu:
    """A sleek, modern CustomTkinter-native popup context menu that matches the UI design system."""
    _active_menu = None

    @classmethod
    def close_active(cls):
        if cls._active_menu is not None:
            try:
                if cls._active_menu.winfo_exists():
                    cls._active_menu.destroy()
            except Exception:
                pass
            cls._active_menu = None

    def __init__(self, parent):
        CTKContextMenu.close_active()
        self.parent = parent
        toplevel_parent = parent.winfo_toplevel() if hasattr(parent, "winfo_toplevel") else parent
        self.window = tk.Toplevel(toplevel_parent)
        self.window.wm_overrideredirect(True)
        self.window.wm_attributes("-topmost", True)
        CTKContextMenu._active_menu = self.window

        # Modern Dark Card Frame
        self.frame = ctk.CTkFrame(
            self.window,
            fg_color=COLOR_CARD_BG,
            border_color="#374151",
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        self.frame.pack(fill="both", expand=True)

        self._toplevel = toplevel_parent
        self._bind_id = toplevel_parent.bind("<Button-1>", lambda e: self._check_click_outside(e), add="+")
        self.window.bind("<FocusOut>", lambda e: CTKContextMenu.close_active())

    def _check_click_outside(self, event):
        if not self.window.winfo_exists():
            return
        x, y = event.x_root, event.y_root
        wx = self.window.winfo_rootx()
        wy = self.window.winfo_rooty()
        ww = self.window.winfo_width()
        wh = self.window.winfo_height()
        if not (wx <= x <= wx + ww and wy <= y <= wy + wh):
            CTKContextMenu.close_active()

    def add_command(self, label, command, is_danger=False):
        btn = ctk.CTkButton(
            self.frame,
            text=label,
            anchor="w",
            height=30,
            width=180,
            font=ctk.CTkFont(size=12, weight="bold" if is_danger else "normal"),
            fg_color="transparent",
            text_color="#F87171" if is_danger else COLOR_TEXT_PRIMARY,
            hover_color=COLOR_DANGER if is_danger else COLOR_PRIMARY,
            corner_radius=RADIUS_SM,
            command=lambda: self._execute(command),
        )
        btn.pack(fill="x", padx=6, pady=2)

    def add_separator(self):
        sep = ctk.CTkFrame(self.frame, height=1, fg_color="#374151")
        sep.pack(fill="x", padx=6, pady=4)

    def _execute(self, command):
        CTKContextMenu.close_active()
        if command:
            command()

    def show(self, x, y):
        self.window.update_idletasks()
        w = self.window.winfo_reqwidth()
        h = self.window.winfo_reqheight()
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()

        if x + w > screen_w - 10:
            x = x - w
        if y + h > screen_h - 40:
            y = y - h
        if x < 10:
            x = 10
        if y < 10:
            y = 10

        self.window.geometry(f"+{x}+{y}")
        self.window.focus_set()


class TemplatePreviewTooltip:
    """Hover preview popup showing template image thumbnail, dimensions, and offset."""
    _instance = None
    _image_cache = {}
    _image_cache_order = []
    _IMAGE_CACHE_MAX = 50

    @classmethod
    def get_instance(cls, root_window=None):
        if cls._instance is None:
            cls._instance = cls(root_window)
        elif root_window is not None:
            cls._instance.root = root_window
        return cls._instance

    def __init__(self, root_window=None):
        self.root = root_window
        self.tip_window = None
        self._hover_after_id = None
        self._scheduled_widget = None

    def schedule_show(self, widget, image_path, title_text="", offset_text=""):
        self.cancel()
        self._scheduled_widget = widget
        self._hover_after_id = widget.after(
            150, lambda: self.show(widget, image_path, title_text, offset_text)
        )

    def cancel(self, event=None):
        if self._hover_after_id and self._scheduled_widget:
            try:
                if self._scheduled_widget.winfo_exists():
                    self._scheduled_widget.after_cancel(self._hover_after_id)
            except Exception:
                pass
            self._hover_after_id = None
        self._scheduled_widget = None
        self.hide()

    def hide(self):
        if self.tip_window is not None:
            try:
                if self.tip_window.winfo_exists():
                    self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None

    def show(self, widget, image_path, title_text="", offset_text=""):
        if not os.path.exists(image_path):
            return
        try:
            if not widget.winfo_exists():
                return
        except Exception:
            return

        self.hide()

        try:
            stat = os.stat(image_path)
            sig = (image_path, stat.st_mtime_ns)
            if sig in self._image_cache:
                photo_img, orig_w, orig_h = self._image_cache[sig]
            else:
                pil_img = Image.open(image_path)
                orig_w, orig_h = pil_img.size
                
                max_w, max_h = 200, 140
                scale = min(max_w / max(1, orig_w), max_h / max(1, orig_h), 1.0)
                if scale < 1.0:
                    new_w = max(1, int(orig_w * scale))
                    new_h = max(1, int(orig_h * scale))
                    scaled_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    scaled_img = pil_img

                photo_img = ImageTk.PhotoImage(scaled_img)
                self._image_cache[sig] = (photo_img, orig_w, orig_h)
                self._image_cache_order.append(sig)
                # LRU eviction
                while len(self._image_cache_order) > self._IMAGE_CACHE_MAX:
                    evict_sig = self._image_cache_order.pop(0)
                    self._image_cache.pop(evict_sig, None)

            parent = self.root if (self.root and self.root.winfo_exists()) else widget.winfo_toplevel()
            self.tip_window = tw = tk.Toplevel(parent)
            tw.wm_overrideredirect(True)
            tw.wm_attributes("-topmost", True)
            try:
                tw.wm_attributes("-alpha", 0.97)
            except Exception:
                pass

            container = tk.Frame(
                tw,
                background="#1A1D20",
                highlightbackground="#3B82F6",
                highlightthickness=1,
                padx=8,
                pady=6,
            )
            container.pack(fill="both", expand=True)

            header_text = f"🖼️ {title_text or os.path.basename(image_path)}"
            title_lbl = tk.Label(
                container,
                text=header_text,
                font=("Segoe UI", 9, "bold"),
                foreground="#60A5FA",
                background="#1A1D20",
                anchor="w",
            )
            title_lbl.pack(fill="x", pady=(0, 4))

            img_lbl = tk.Label(
                container,
                image=photo_img,
                background="#0D1117",
                relief="solid",
                borderwidth=1,
            )
            img_lbl.image = photo_img
            img_lbl.pack(pady=(0, 4))

            meta_info = f"크기: {orig_w}x{orig_h}px"
            if offset_text:
                meta_info += f"  |  {offset_text}"
            info_lbl = tk.Label(
                container,
                text=meta_info,
                font=("Segoe UI", 8),
                foreground="#9CA3AF",
                background="#1A1D20",
            )
            info_lbl.pack(fill="x")

            tw.update_idletasks()
            w_width = tw.winfo_reqwidth()
            w_height = tw.winfo_reqheight()

            # 마우스 커서 위치 기준으로 자연스럽게 팝업 배치
            try:
                pointer_x = widget.winfo_pointerx()
                pointer_y = widget.winfo_pointery()
            except Exception:
                pointer_x = widget.winfo_rootx()
                pointer_y = widget.winfo_rooty()

            x = pointer_x + 16
            y = pointer_y + 12

            screen_w = tw.winfo_screenwidth()
            screen_h = tw.winfo_screenheight()

            # 화면 밖으로 벗어남 방지
            if x + w_width > screen_w - 15:
                x = max(10, pointer_x - w_width - 12)
            if y + w_height > screen_h - 40:
                y = max(10, pointer_y - w_height - 10)
            if x < 10:
                x = 10
            if y < 10:
                y = 10

            tw.wm_geometry(f"+{x}+{y}")
        except Exception:
            self.hide()


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent_frame):
        super().__init__(parent_frame)
        self.parent_frame = parent_frame
        self.clicker = parent_frame.clicker
        self._save_after_id = None
        self.title("Settings (공통 환경 설정)")
        self.geometry("500x680")
        self.minsize(460, 520)
        
        # 메인 창에 종속 설정 및 모달 효과
        self.transient(parent_frame.winfo_toplevel())
        self.grab_set()

        # Header (Fixed Top)
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=20, pady=(14, 8))

        self.header_label = ctk.CTkLabel(
            self.header_frame,
            text="⚙️ 공통 환경 설정 (Global Settings)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.header_label.pack(anchor="w")

        self.header_sublabel = ctk.CTkLabel(
            self.header_frame,
            text="모든 인스턴스 탭에 공통으로 적용되는 전역 실행 파라미터입니다.",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.header_sublabel.pack(anchor="w", pady=(2, 0))

        # Main Scrollable Body
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=12, pady=(0, 5))

        def make_card(title):
            card = ctk.CTkFrame(
                self.scroll_frame,
                fg_color=COLOR_CARD_BG,
                border_color=COLOR_BORDER,
                border_width=1,
                corner_radius=RADIUS_MD
            )
            card.pack(fill="x", padx=4, pady=(0, 10))
            
            lbl = ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLOR_TEXT_PRIMARY,
                anchor="w"
            )
            lbl.pack(fill="x", padx=14, pady=(10, 6))
            return card

        def make_entry_row(parent, label_text, default_val, unit="s"):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(0, 6))
            lbl = ctk.CTkLabel(row, text=label_text, anchor="w", font=ctk.CTkFont(size=12))
            lbl.pack(side="left", fill="x", expand=True)
            if unit:
                unit_lbl = ctk.CTkLabel(row, text=unit, text_color=COLOR_TEXT_MUTED, font=ctk.CTkFont(size=11))
                unit_lbl.pack(side="right", padx=(4, 0))
            entry = ctk.CTkEntry(row, width=75, height=28, corner_radius=RADIUS_SM, justify="center")
            entry.insert(0, str(default_val))
            entry.pack(side="right")
            return entry

        # --- Category 1: 클릭 & 타이밍 설정 ---
        self.card_timing = make_card("⏱️  클릭 및 타이밍 (Timing & Delays)")
        self.interval_entry = make_entry_row(self.card_timing, "화면 스캔 검사 간격 (Scan Interval)", self.clicker.scan_interval, "초")
        self.interval_entry.bind("<KeyRelease>", self.schedule_save)
        self.interval_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.double_click_entry = make_entry_row(self.card_timing, "더블클릭 간격 (Double-Click Interval)", getattr(self.clicker, 'double_click_interval', 1.0), "초")
        self.double_click_entry.bind("<KeyRelease>", self.schedule_save)
        self.double_click_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.post_delay_entry = make_entry_row(self.card_timing, "동작 실행 후 기본 대기 (Post-Action Delay)", getattr(self.clicker, 'post_action_delay', 2.0), "초")
        self.post_delay_entry.bind("<KeyRelease>", self.schedule_save)
        self.post_delay_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # --- Category 2: 이미지 매칭 설정 ---
        self.card_matching = make_card("🎯  이미지 매칭 알고리즘 (Image Matching)")
        self.threshold_entry = make_entry_row(self.card_matching, "유사도 임계값 (Similarity Threshold 0.1~1.0)", self.clicker.similarity_threshold, "")
        self.threshold_entry.bind("<KeyRelease>", self.schedule_save)
        self.threshold_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.grayscale_var = ctk.StringVar(value="on" if self.clicker.match_grayscale else "off")
        self.grayscale_switch = ctk.CTkSwitch(
            self.card_matching,
            text="고속 그레이스케일 매칭 (흑백 변환으로 속도 대폭 향상)",
            variable=self.grayscale_var,
            onvalue="on",
            offvalue="off",
            font=ctk.CTkFont(size=12),
            command=self.save_settings,
        )
        self.grayscale_switch.pack(fill="x", padx=14, pady=(2, 10))

        # --- Category 3: 경고 및 모니터링 ---
        self.card_alerts = make_card("⚠️  경고 및 카운터 (Alerts & Monitoring)")
        self.timeout_entry = make_entry_row(self.card_alerts, "매칭 없음 경고 알림 시간 (0: 끄기)", self.clicker.no_match_timeout, "초")
        self.timeout_entry.bind("<KeyRelease>", self.schedule_save)
        self.timeout_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.consecutive_entry = make_entry_row(self.card_alerts, "동일 템플릿 연속 매칭 경고 횟수 (0: 끄기)", getattr(self.clicker, 'consecutive_match_threshold', 0), "회")
        self.consecutive_entry.bind("<KeyRelease>", self.schedule_save)
        self.consecutive_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.reset_counts_var = ctk.StringVar(value="on" if getattr(self.clicker, 'reset_counts_on_startup', False) else "off")
        self.reset_counts_switch = ctk.CTkSwitch(
            self.card_alerts,
            text="앱 실행 시 클릭 카운터 자동 초기화",
            variable=self.reset_counts_var,
            onvalue="on",
            offvalue="off",
            font=ctk.CTkFont(size=12),
            command=self.save_settings,
        )
        self.reset_counts_switch.pack(fill="x", padx=14, pady=(2, 10))

        # --- Category 4: 미매칭 복구 동작 ---
        self.card_nomatch = make_card("⚡  미매칭 자동 복구 동작 (Fallback Action)")
        
        curr_action = getattr(self.clicker, 'no_match_action', 'none')
        curr_label = NO_MATCH_ACTION_MAP.get(curr_action, "사용 안 함 (Disabled)")
        
        action_row = ctk.CTkFrame(self.card_nomatch, fg_color="transparent")
        action_row.pack(fill="x", padx=14, pady=(0, 6))
        action_lbl = ctk.CTkLabel(action_row, text="수행할 복구 동작:", anchor="w", font=ctk.CTkFont(size=12))
        action_lbl.pack(side="left", padx=(0, 10))
        
        self.no_match_combo = ctk.CTkOptionMenu(
            action_row, 
            values=list(NO_MATCH_ACTION_MAP.values()),
            height=28,
            corner_radius=RADIUS_SM,
            command=self.on_action_changed
        )
        self.no_match_combo.set(curr_label)
        self.no_match_combo.pack(side="right", fill="x", expand=True)

        self.no_match_interval_entry = make_entry_row(self.card_nomatch, "동작 실행 대기 시간", getattr(self.clicker, 'no_match_interval', 30), "초")
        self.no_match_interval_entry.bind("<KeyRelease>", self.schedule_save)
        self.no_match_interval_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # Custom Coordinate Frame (For Custom Click & Custom Double Click)
        self.coord_frame = ctk.CTkFrame(self.card_nomatch, fg_color="transparent")
        self.coord_frame.pack(fill="x", padx=14, pady=(0, 10))

        coords = getattr(self.clicker, 'no_match_coords', [500, 500])
        coord_lbl = ctk.CTkLabel(self.coord_frame, text="클릭 대상 좌표:", anchor="w", font=ctk.CTkFont(size=12))
        coord_lbl.pack(side="left", padx=(0, 6))

        self.coord_x_label = ctk.CTkLabel(self.coord_frame, text="X:", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MUTED)
        self.coord_x_label.pack(side="left", padx=(0, 2))
        self.coord_x_entry = ctk.CTkEntry(self.coord_frame, width=55, height=28, corner_radius=RADIUS_SM, justify="center")
        self.coord_x_entry.insert(0, str(coords[0]))
        self.coord_x_entry.pack(side="left", padx=(0, 6))
        self.coord_x_entry.bind("<KeyRelease>", self.schedule_save)
        self.coord_x_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.coord_y_label = ctk.CTkLabel(self.coord_frame, text="Y:", font=ctk.CTkFont(size=11), text_color=COLOR_TEXT_MUTED)
        self.coord_y_label.pack(side="left", padx=(0, 2))
        self.coord_y_entry = ctk.CTkEntry(self.coord_frame, width=55, height=28, corner_radius=RADIUS_SM, justify="center")
        self.coord_y_entry.insert(0, str(coords[1]))
        self.coord_y_entry.pack(side="left", padx=(0, 8))
        self.coord_y_entry.bind("<KeyRelease>", self.schedule_save)
        self.coord_y_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.pick_coord_btn = ctk.CTkButton(
            self.coord_frame, 
            text="🎯 좌표 선택", 
            width=90,
            height=28,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            corner_radius=RADIUS_SM,
            command=self.pick_coords_from_screen
        )
        self.pick_coord_btn.pack(side="right")

        self.update_coord_frame_visibility(curr_action)

        # --- Category 5: ADB 실행 환경 설정 ---
        self.card_adb = make_card("🔌  ADB 실행 환경 설정 (ADB Mode & Path)")

        self.adb_mode_map = {
            "bundled": "내장 ADB 사용 (기본 권장)",
            "custom": "외부 ADB 경로 직접 지정",
        }
        self.reverse_adb_mode_map = {v: k for k, v in self.adb_mode_map.items()}

        mode_row = ctk.CTkFrame(self.card_adb, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=(0, 6))
        mode_lbl = ctk.CTkLabel(mode_row, text="ADB 모드:", anchor="w", font=ctk.CTkFont(size=12))
        mode_lbl.pack(side="left", padx=(0, 10))

        curr_adb_mode = getattr(self.clicker, "adb_mode", "bundled")
        curr_adb_label = self.adb_mode_map.get(curr_adb_mode, "내장 ADB 사용 (기본 권장)")

        self.adb_mode_combo = ctk.CTkOptionMenu(
            mode_row,
            values=list(self.adb_mode_map.values()),
            height=28,
            corner_radius=RADIUS_SM,
            command=self.on_adb_mode_changed,
        )
        self.adb_mode_combo.set(curr_adb_label)
        self.adb_mode_combo.pack(side="right", fill="x", expand=True)

        self.custom_adb_frame = ctk.CTkFrame(self.card_adb, fg_color="transparent")
        self.custom_adb_frame.pack(fill="x", padx=14, pady=(0, 6))

        adb_path_lbl = ctk.CTkLabel(
            self.custom_adb_frame,
            text="경로:",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_TEXT_MUTED,
        )
        adb_path_lbl.pack(side="left", padx=(0, 6))

        self.custom_adb_entry = ctk.CTkEntry(
            self.custom_adb_frame,
            height=28,
            corner_radius=RADIUS_SM,
            placeholder_text="예: C:\\platform-tools\\adb.exe",
        )
        self.custom_adb_entry.insert(0, getattr(self.clicker, "custom_adb_path", ""))
        self.custom_adb_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.custom_adb_entry.bind("<KeyRelease>", self.on_adb_candidate_changed)

        self.browse_adb_btn = ctk.CTkButton(
            self.custom_adb_frame,
            text="📁 찾아보기",
            width=80,
            height=28,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            corner_radius=RADIUS_SM,
            command=self.browse_custom_adb,
        )
        self.browse_adb_btn.pack(side="right")

        self.adb_status_label = ctk.CTkLabel(
            self.card_adb,
            text=f"현재 적용: {self.clicker.adb_path}",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED,
            anchor="w",
        )
        self.adb_status_label.pack(fill="x", padx=14, pady=(0, 10))

        self.update_adb_frame_visibility(curr_adb_mode)

        # Footer Frame (Fixed Bottom)
        self.footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.footer_frame.pack(fill="x", padx=16, pady=(6, 12))

        self.license_btn = ctk.CTkButton(
            self.footer_frame,
            text="📜 오픈소스 라이선스 고지 (Licenses)",
            height=30,
            fg_color="transparent",
            border_width=1,
            border_color="#4B5563",
            text_color=COLOR_TEXT_MUTED,
            hover_color=COLOR_NEUTRAL,
            corner_radius=RADIUS_MD,
            command=lambda: LicenseNoticeWindow(self)
        )
        self.license_btn.pack(fill="x", pady=(0, 6))

        self.close_btn = ctk.CTkButton(
            self.footer_frame,
            text="설정 저장 및 닫기 (Save & Close)",
            height=34,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_MD,
            command=self.close_window
        )
        self.close_btn.pack(fill="x")

        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def on_action_changed(self, choice):
        action_key = REVERSE_NO_MATCH_ACTION_MAP.get(choice, "none")
        self.update_coord_frame_visibility(action_key)
        self.save_settings()

    def update_coord_frame_visibility(self, action_key):
        if action_key in ("custom_click", "custom_double_click"):
            self.coord_frame.pack(fill="x", padx=14, pady=(0, 10))
        else:
            self.coord_frame.pack_forget()

    def on_adb_mode_changed(self, choice):
        mode_key = self.reverse_adb_mode_map.get(choice, "bundled")
        self.update_adb_frame_visibility(mode_key)

    def update_adb_frame_visibility(self, mode_key):
        if mode_key == "custom":
            self.custom_adb_frame.pack(fill="x", padx=14, pady=(0, 6))
        else:
            self.custom_adb_frame.pack_forget()
        self.on_adb_candidate_changed()

    def on_adb_candidate_changed(self, event=None):
        if hasattr(self, "adb_status_label") and self.adb_status_label.winfo_exists():
            mode_key = self.reverse_adb_mode_map.get(
                self.adb_mode_combo.get(), "bundled"
            )
            custom_path = (
                self.custom_adb_entry.get().strip()
                if hasattr(self, "custom_adb_entry")
                else ""
            )
            resolved = resolve_adb_path(
                mode_key,
                custom_path,
                self.clicker.base_dir,
            )
            pending = (
                mode_key != getattr(self.clicker, "adb_mode", DEFAULT_ADB_MODE)
                or custom_path != getattr(self.clicker, "custom_adb_path", "")
            )
            self.adb_status_label.configure(
                text=f"{'저장 대기' if pending else '현재 적용'}: {resolved}",
                text_color=COLOR_WARNING if pending else COLOR_TEXT_MUTED,
            )

    def browse_custom_adb(self):
        initial_dir = None
        curr = self.custom_adb_entry.get().strip()
        if curr and os.path.exists(os.path.dirname(curr)):
            initial_dir = os.path.dirname(curr)
        file_path = filedialog.askopenfilename(
            parent=self,
            title="ADB 실행 파일(adb.exe) 선택",
            filetypes=[("ADB 실행 파일 (*.exe)", "*.exe"), ("모든 파일 (*.*)", "*.*")],
            initialdir=initial_dir,
        )
        if file_path:
            self.custom_adb_entry.delete(0, "end")
            self.custom_adb_entry.insert(0, os.path.normpath(file_path))
            self.on_adb_candidate_changed()

    def pick_coords_from_screen(self):
        try:
            initial_x = int(self.coord_x_entry.get().strip())
            initial_y = int(self.coord_y_entry.get().strip())
        except ValueError:
            initial_x = initial_y = None

        app = self.parent_frame.app_owner
        if not app.opencv_lock.acquire(blocking=False):
            self.parent_frame.log_message("다른 화면 선택 창이 이미 열려 있습니다.")
            return

        self.pick_coord_btn.configure(state="disabled")

        def finish_picker():
            app.opencv_lock.release()

            def enable_button():
                if self.winfo_exists():
                    self.pick_coord_btn.configure(state="normal")

            app.post_to_ui(enable_button)

        def pick_task():
            window_name = None
            try:
                screen = self.clicker.capture_screen(grayscale=False)
                if screen is None:
                    self.parent_frame.log_message("좌표 선택용 화면 캡처에 실패했습니다.")
                    return

                height, width = screen.shape[:2]
                selected_pt = [
                    width // 2 if initial_x is None else max(0, min(width - 1, initial_x)),
                    height // 2 if initial_y is None else max(0, min(height - 1, initial_y)),
                ]
                window_name = "[Pick Target Coordinates] Click point -> Enter to confirm"
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
                        if not self.winfo_exists():
                            return
                        self.coord_x_entry.delete(0, "end")
                        self.coord_x_entry.insert(0, str(selected_pt[0]))
                        self.coord_y_entry.delete(0, "end")
                        self.coord_y_entry.insert(0, str(selected_pt[1]))
                        self.save_settings()

                    app.post_to_ui(update_entries)
            except Exception as error:
                self.parent_frame.log_message(f"좌표 선택 중 오류: {error}")
            finally:
                if window_name:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass
                finish_picker()

        threading.Thread(target=pick_task, daemon=True).start()

    def _apply_form_values(self):
        try:
            value = float(self.interval_entry.get().strip())
            if value >= 0.1:
                self.clicker.scan_interval = int(value) if value.is_integer() else value
        except ValueError:
            pass

        try:
            value = float(self.double_click_entry.get().strip())
            if value >= 0.01:
                self.clicker.double_click_interval = int(value) if value.is_integer() else value
        except ValueError:
            pass

        try:
            value = float(self.post_delay_entry.get().strip())
            if value >= 0.0:
                self.clicker.post_action_delay = int(value) if value.is_integer() else value
        except ValueError:
            pass

        try:
            value = float(self.threshold_entry.get().strip())
            if 0.0 < value <= 1.0:
                self.clicker.similarity_threshold = value
        except ValueError:
            pass
        self.clicker.match_grayscale = self.grayscale_var.get() == "on"


        try:
            value = float(self.timeout_entry.get().strip())
            if value >= 0:
                self.clicker.no_match_timeout = int(value) if value.is_integer() else value
        except ValueError:
            pass

        try:
            value = int(float(self.consecutive_entry.get().strip()))
            if value >= 0:
                self.clicker.consecutive_match_threshold = value
        except ValueError:
            pass

        self.clicker.reset_counts_on_startup = self.reset_counts_var.get() == "on"

        action_key = REVERSE_NO_MATCH_ACTION_MAP.get(self.no_match_combo.get(), "none")
        self.clicker.no_match_action = action_key
        self.clicker.enable_random_click = action_key == "random_click"

        try:
            value = float(self.no_match_interval_entry.get().strip())
            if value >= 0.1:
                value = int(value) if value.is_integer() else value
                self.clicker.no_match_interval = value
                self.clicker.random_click_interval = value
        except ValueError:
            pass

        try:
            self.clicker.no_match_coords = [
                int(self.coord_x_entry.get().strip()),
                int(self.coord_y_entry.get().strip()),
            ]
        except ValueError:
            pass


        self.parent_frame.app_owner.sync_shared_settings_to_all(self.parent_frame)

    def _apply_adb_settings(self):
        """Validate and atomically apply the pending ADB choice to every tab."""
        adb_mode = self.reverse_adb_mode_map.get(
            self.adb_mode_combo.get(), "bundled"
        )
        custom_path = self.custom_adb_entry.get().strip()
        if adb_mode == "custom" and not custom_path:
            validation_message = "외부 ADB 모드에서는 adb.exe 경로를 지정해야 합니다."
            self.adb_status_label.configure(
                text=f"검증 실패: {validation_message}",
                text_color=COLOR_DANGER,
            )
            from tkinter import messagebox

            messagebox.showerror(
                "ADB 실행 파일 검증 실패",
                f"ADB 설정을 적용하지 않았습니다.\n\n{validation_message}",
                parent=self,
            )
            return False
        candidate_path = resolve_adb_path(
            adb_mode,
            custom_path,
            self.clicker.base_dir,
        )
        valid, resolved_path, validation_message = (
            self.clicker.validate_adb_executable(candidate_path)
        )
        if not valid:
            self.adb_status_label.configure(
                text=f"검증 실패: {validation_message}",
                text_color=COLOR_DANGER,
            )
            from tkinter import messagebox

            messagebox.showerror(
                "ADB 실행 파일 검증 실패",
                f"ADB 설정을 적용하지 않았습니다.\n\n{validation_message}",
                parent=self,
            )
            return False

        frames = tuple(self.parent_frame.app_owner.tab_frames.values())
        changed = any(
            frame.clicker.adb_mode != adb_mode
            or frame.clicker.custom_adb_path != custom_path
            or os.path.normcase(os.path.abspath(frame.clicker.adb_path))
            != os.path.normcase(os.path.abspath(resolved_path))
            for frame in frames
        )
        active_frames = []
        if changed:
            with ExitStack() as lock_stack:
                for frame in frames:
                    lock_stack.enter_context(frame.clicker._device_lock)
                active_frames = [
                    frame
                    for frame in frames
                    if frame.clicker.device is not None
                    or frame.clicker.is_running
                    or getattr(frame, "_loop_starting", False)
                ]
                if not active_frames:
                    for frame in frames:
                        frame.clicker.adb_mode = adb_mode
                        frame.clicker.custom_adb_path = custom_path
                        frame.clicker.adb_path = resolved_path
        if changed and active_frames:
            self.adb_status_label.configure(
                text="적용 보류: 모든 디바이스 연결을 먼저 해제하세요.",
                text_color=COLOR_WARNING,
            )
            from tkinter import messagebox

            messagebox.showwarning(
                "ADB 설정 적용 보류",
                "실행 중 ADB 바이너리 교체를 방지하기 위해 설정을 적용하지 않았습니다.\n"
                "모든 디바이스 연결을 해제한 뒤 다시 저장해 주세요.",
                parent=self,
            )
            return False

        self.adb_status_label.configure(
            text=f"현재 적용: {resolved_path} ({validation_message})",
            text_color=COLOR_SUCCESS,
        )
        self.parent_frame.log_message(f"ADB 실행 환경 적용 완료: {resolved_path}")
        return True

    def schedule_save(self, event=None):
        self._apply_form_values()
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
        self._save_after_id = self.after(400, self._persist_settings)

    def _persist_settings(self):
        self._save_after_id = None
        self.parent_frame.app_owner.save_app_config()

    def save_settings(self, event=None):
        self._apply_form_values()
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
            self._save_after_id = None
        self._persist_settings()

    def close_window(self):
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
            self._save_after_id = None
        self._apply_form_values()
        if not self._apply_adb_settings():
            self._persist_settings()
            return
        self._persist_settings()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


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


class RenameTemplateWindow(ctk.CTkToplevel):
    """Dialog for renaming a template filename."""
    def __init__(self, parent_frame, filename, is_fallback=False):
        super().__init__(parent_frame)
        self.parent_frame = parent_frame
        self.old_filename = filename
        self.is_fallback = is_fallback
        self.title("✏️ 템플릿 이름 변경 (Rename)")
        self.geometry("380x210")
        self.resizable(False, False)
        self.transient(parent_frame.winfo_toplevel())
        self.grab_set()

        target_type = "미매칭 복구" if is_fallback else "기본"
        self.header_label = ctk.CTkLabel(
            self,
            text=f"✏️ [{target_type}] 템플릿 이름 변경",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.header_label.pack(pady=(15, 4))

        self.info_label = ctk.CTkLabel(
            self,
            text=f"현재 이름: {filename}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.info_label.pack(pady=(0, 8))

        self.entry_label = ctk.CTkLabel(self, text="새 파일명 (New Name):", anchor="w")
        self.entry_label.pack(fill="x", padx=30, pady=(0, 2))

        base_name = filename[:-4] if filename.lower().endswith(".png") else filename
        self.name_entry = ctk.CTkEntry(self, placeholder_text="새 템플릿 파일명 입력")
        self.name_entry.insert(0, base_name)
        self.name_entry.pack(fill="x", padx=30, pady=(0, 15))
        self.name_entry.select_range(0, "end")
        self.name_entry.focus()

        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=30)

        self.confirm_btn = ctk.CTkButton(
            self.btn_frame,
            text="변경 (Rename)",
            command=self.confirm_rename,
            width=140,
            height=32,
            fg_color=COLOR_SUCCESS,
            hover_color=COLOR_SUCCESS_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            corner_radius=RADIUS_MD,
            font=ctk.CTkFont(weight="bold")
        )
        self.confirm_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.cancel_btn = ctk.CTkButton(
            self.btn_frame,
            text="취소 (Cancel)",
            command=self.close_window,
            width=140,
            height=32,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_MD,
        )
        self.cancel_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        self.bind("<Return>", lambda e: self.confirm_rename())
        self.bind("<Escape>", lambda e: self.close_window())
        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def close_window(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def confirm_rename(self):
        new_name = self.name_entry.get().strip()
        if not new_name:
            from tkinter import messagebox
            messagebox.showwarning("입력 오류", "변경할 템플릿 파일명을 입력해주세요.", parent=self)
            return

        if self.is_fallback:
            success, msg = self.parent_frame.clicker.rename_fallback_template(self.old_filename, new_name)
        else:
            success, msg = self.parent_frame.clicker.rename_template(self.old_filename, new_name)

        if success:
            TemplatePreviewTooltip.get_instance(self.parent_frame.winfo_toplevel()).hide()
            self.parent_frame.app_owner.refresh_all_tabs_templates()
            self.parent_frame.log_message(f"템플릿 이름이 변경되었습니다: '{self.old_filename}' -> '{msg}'")
            self.close_window()
        else:
            from tkinter import messagebox
            messagebox.showerror("이름 변경 실패", msg, parent=self)


class TemplateDelayWindow(ctk.CTkToplevel):
    """Dialog for setting pre-action or post-action delay for a template."""
    def __init__(self, parent_frame, filename, is_fallback=False):
        super().__init__(parent_frame)
        self.parent_frame = parent_frame
        self.filename = filename
        self.is_fallback = is_fallback
        self.clicker = parent_frame.clicker

        delays_dict = (
            self.clicker.fallback_template_delays
            if is_fallback
            else self.clicker.template_delays
        )
        types_dict = (
            self.clicker.fallback_template_delay_types
            if is_fallback
            else self.clicker.template_delay_types
        )
        current_delay = delays_dict.get(filename, 0.0)
        current_type = types_dict.get(filename, "pre")

        self.title("⏱️ 템플릿 지연 시간 설정 (Action Delay)")
        self.geometry("440x370")
        self.resizable(False, False)

        self.transient(parent_frame.winfo_toplevel())
        self.grab_set()

        # Header Title
        target_name = "미매칭 복구 템플릿" if is_fallback else "기본 템플릿"
        self.header_label = ctk.CTkLabel(
            self,
            text=f"⏱️ [{target_name}] 동작 지연 시간 설정",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.header_label.pack(pady=(15, 2))

        self.filename_label = ctk.CTkLabel(
            self,
            text=f"대상: {filename}",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED,
        )
        self.filename_label.pack(pady=(0, 10))

        # Timing selection (Pre vs Post)
        self.timing_label = ctk.CTkLabel(
            self,
            text="지연 타이밍 (Timing):",
            font=ctk.CTkFont(weight="bold"),
            anchor="w",
        )
        self.timing_label.pack(fill="x", padx=30, pady=(0, 4))

        initial_seg = "동작 후 대기 (Post-delay)" if current_type == "post" else "동작 전 대기 (Pre-delay)"
        self.timing_segmented = ctk.CTkSegmentedButton(
            self,
            values=["동작 전 대기 (Pre-delay)", "동작 후 대기 (Post-delay)"],
            command=self._on_timing_changed,
        )
        self.timing_segmented.set(initial_seg)
        self.timing_segmented.pack(fill="x", padx=30, pady=(0, 8))

        # Description box
        self.desc_frame = ctk.CTkFrame(self, fg_color=COLOR_CARD_BG, corner_radius=RADIUS_MD)
        self.desc_frame.pack(fill="x", padx=30, pady=(0, 12))

        self.desc_label = ctk.CTkLabel(
            self.desc_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT_MUTED,
            justify="left",
            wraplength=370,
        )
        self.desc_label.pack(padx=10, pady=8)
        self._update_desc_text()

        # Duration input
        self.duration_label = ctk.CTkLabel(
            self,
            text="대기 시간 (초, Seconds - 0: 즉시 실행):",
            font=ctk.CTkFont(weight="bold"),
            anchor="w",
        )
        self.duration_label.pack(fill="x", padx=30, pady=(0, 5))

        self.duration_entry = ctk.CTkEntry(self, placeholder_text="예: 0, 0.5, 1, 2, 3")
        self.duration_entry.insert(0, str(current_delay))
        self.duration_entry.pack(fill="x", padx=30, pady=(0, 8))

        # Preset buttons
        self.preset_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.preset_frame.pack(fill="x", padx=30, pady=(0, 12))

        presets = [("0초", 0), ("0.5초", 0.5), ("1초", 1.0), ("2초", 2.0), ("3초", 3.0), ("5초", 5.0)]
        for label, val in presets:
            btn = ctk.CTkButton(
                self.preset_frame,
                text=label,
                width=52,
                height=26,
                fg_color=COLOR_NEUTRAL,
                hover_color=COLOR_NEUTRAL_HOVER,
                corner_radius=RADIUS_SM,
                command=lambda v=val: self._set_preset(v),
            )
            btn.pack(side="left", padx=2, expand=True)

        # Action Buttons
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=30, pady=(8, 15))

        self.save_btn = ctk.CTkButton(
            self.btn_frame,
            text="💾 저장 (Save)",
            height=32,
            fg_color=COLOR_SUCCESS,
            hover_color=COLOR_SUCCESS_HOVER,
            corner_radius=RADIUS_MD,
            command=self.save_and_close,
        )
        self.save_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.cancel_btn = ctk.CTkButton(
            self.btn_frame,
            text="취소 (Cancel)",
            height=32,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_MD,
            command=self.close_window,
        )
        self.cancel_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

    def _set_preset(self, val):
        self.duration_entry.delete(0, "end")
        self.duration_entry.insert(0, str(val))

    def _on_timing_changed(self, choice):
        self._update_desc_text()

    def _update_desc_text(self):
        choice = self.timing_segmented.get()
        if "동작 후" in choice:
            self.desc_label.configure(
                text="💡 [동작 후 대기]\n클릭(동작)을 수행한 후, 다음 패턴 인식을 시작하기 전까지 지정된 시간 동안 아무 작업 없이 대기합니다.",
                text_color=COLOR_PRIMARY,
            )
        else:
            self.desc_label.configure(
                text="💡 [동작 전 대기]\n화면에서 템플릿을 인식한 직후, 클릭(동작)을 실행하기 전에 지정된 시간 동안 대기합니다.",
                text_color=COLOR_WARNING,
            )

    def save_and_close(self):
        val_str = self.duration_entry.get().strip()
        try:
            delay_sec = float(val_str)
            if delay_sec < 0:
                raise ValueError
        except ValueError:
            from tkinter import messagebox
            messagebox.showerror("입력 오류", "0 이상의 숫자를 입력해주세요. (예: 0, 0.5, 1, 2.5)")
            return

        choice = self.timing_segmented.get()
        delay_type = "post" if "동작 후" in choice else "pre"

        if self.is_fallback:
            self.clicker.set_fallback_template_delay(self.filename, delay_sec, delay_type)
        else:
            self.clicker.set_template_delay(self.filename, delay_sec, delay_type)

        timing_name = "동작 후" if delay_type == "post" else "동작 전"
        self.parent_frame.log_message(
            f"⏱️ [{self.filename}] 지연 시간: {delay_sec:g}초 ({timing_name} 대기) 설정 완료"
        )
        self.parent_frame.app_owner.update_all_template_delays(
            self.filename, delay_sec, delay_type, is_fallback=self.is_fallback
        )
        self.close_window()

    def close_window(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class AddInstanceWindow(ctk.CTkToplevel):
    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.title("인스턴스 탭 추가 / 선택")
        self.geometry("460x480")
        self.resizable(False, False)

        self.transient(parent_app)
        self.grab_set()

        self.header_label = ctk.CTkLabel(self, text="➕ 새 인스턴스 탭 추가", font=ctk.CTkFont(size=16, weight="bold"))
        self.header_label.pack(pady=(15, 10))

        # 검색된 디바이스 영역
        self.dev_frame = ctk.CTkFrame(self)
        self.dev_frame.pack(fill="both", expand=True, padx=20, pady=5)

        self.dev_title = ctk.CTkLabel(self.dev_frame, text="📱 검색된 ADB 디바이스 목록 (선택):", font=ctk.CTkFont(weight="bold"))
        self.dev_title.pack(anchor="w", padx=15, pady=(10, 5))

        self.scroll_frame = ctk.CTkScrollableFrame(self.dev_frame, height=200)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.checkbox_vars = {} # dev_addr -> BooleanVar

        self.loading_label = ctk.CTkLabel(self.scroll_frame, text="디바이스 검색 중 (ADB scanning)...", text_color=COLOR_TEXT_MUTED)
        self.loading_label.pack(pady=20)

        # 수동 주소 입력 영역
        self.custom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.custom_frame.pack(fill="x", padx=20, pady=5)
        
        self.custom_label = ctk.CTkLabel(self.custom_frame, text="직접 입력 (IP:Port):")
        self.custom_label.pack(side="left", padx=(5, 5))

        self.custom_entry = ctk.CTkEntry(self.custom_frame, placeholder_text="예: 127.0.0.1:5565")
        self.custom_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # 하단 버튼 영역
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=20, pady=(10, 15))

        self.add_selected_btn = ctk.CTkButton(
            self.btn_frame,
            text="➕ 선택 항목 탭 추가",
            height=34,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            corner_radius=RADIUS_MD,
            command=self.add_selected
        )
        self.add_selected_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.add_and_connect_btn = ctk.CTkButton(
            self.btn_frame,
            text="⚡ 전체 추가 & 자동 연결",
            height=34,
            fg_color=COLOR_SUCCESS,
            hover_color=COLOR_SUCCESS_HOVER,
            corner_radius=RADIUS_MD,
            command=self.add_and_connect_all
        )
        self.add_and_connect_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # 스레드로 디바이스 목록 검색
        threading.Thread(target=self.fetch_devices, daemon=True).start()

    def fetch_devices(self):
        dummy = AutoClicker()
        devices = dummy.get_connected_devices()
        
        def update_ui():
            if not self.winfo_exists():
                return
            self.loading_label.destroy()
            
            existing_addrs = [f.clicker.device_address for f in self.parent_app.tab_frames.values()]
            
            if not devices:
                no_dev = ctk.CTkLabel(self.scroll_frame, text="현재 자동 감지된 ADB 디바이스가 없습니다.\n아래 수동 입력을 이용하거나 에뮬레이터를 먼저 실행하세요.", text_color="orange")
                no_dev.pack(pady=20)
                return

            for dev in devices:
                var = ctk.BooleanVar(value=True)
                self.checkbox_vars[dev] = var
                
                is_already = dev in existing_addrs
                label_text = f"{dev} (이미 추가됨)" if is_already else dev
                
                chk = ctk.CTkCheckBox(self.scroll_frame, text=label_text, variable=var)
                if is_already:
                    var.set(False)
                    chk.configure(state="disabled")
                chk.pack(anchor="w", padx=10, pady=5)

        self.parent_app.post_to_ui(update_ui)

    def add_selected(self, auto_connect=False):
        to_add = []
        for dev, var in self.checkbox_vars.items():
            if var.get():
                to_add.append(dev)
        
        custom_val = self.custom_entry.get().strip()
        if custom_val:
            to_add.append(custom_val)
            
        if not to_add:
            from tkinter import messagebox
            messagebox.showinfo("알림", "추가할 디바이스를 하나 이상 선택하거나 주소를 입력하세요.")
            return

        added_frames = []
        for addr in to_add:
            frame = self.parent_app.add_instance_tab(device_address=addr)
            if frame:
                added_frames.append(frame)

        if auto_connect:
            for frame in added_frames:
                frame.toggle_connection()

        self.parent_app.save_app_config()
        self.grab_release()
        self.destroy()

    def add_and_connect_all(self):
        for dev, var in self.checkbox_vars.items():
            var.set(True)
        self.add_selected(auto_connect=True)


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
        dialog = ctk.CTkInputDialog(
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
                step1_window = (
                    f"[Step 1] Select Template Area "
                    f"({'Fallback' if is_fallback else 'Primary'}) - "
                    f"{self.clicker.device_address}"
                )
                open_windows.append(step1_window)
                cv2.namedWindow(step1_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(step1_window, width // 2, height // 2)
                roi = cv2.selectROI(
                    step1_window, screen, showCrosshair=True, fromCenter=False
                )
                cv2.destroyWindow(step1_window)
                open_windows.remove(step1_window)
                if roi[2] <= 0 or roi[3] <= 0:
                    self.log_message("영역 선택 취소됨")
                    return

                roi_x, roi_y, roi_width, roi_height = map(int, roi)
                crop_img = screen[
                    roi_y:roi_y + roi_height,
                    roi_x:roi_x + roi_width,
                ].copy()
                banner_height = 65
                step2_window = (
                    f"[Step 2] Select Click Target "
                    f"({'Fallback' if is_fallback else 'Primary'}) - "
                    f"{self.clicker.device_address}"
                )
                open_windows.append(step2_window)
                cv2.namedWindow(step2_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(
                    step2_window, width // 2, (height + banner_height) // 2
                )
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
                    banner = np.full(
                        (banner_height, width, 3), 30, dtype=np.uint8
                    )
                    offset_x, offset_y = x - roi_x, y - roi_y
                    position_type = (
                        " [Default Center]"
                        if selected_point == default_point
                        else " [Custom Target]"
                    )
                    cv2.putText(
                        banner,
                        f"Target: ({x}, {y}) | Offset: "
                        f"({offset_x:+d}, {offset_y:+d}){position_type}",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        banner,
                        "Click screen -> Enter/Space: Confirm (Cancel: 'c' or ESC)",
                        (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (220, 220, 220), 1, cv2.LINE_AA,
                    )
                    cv2.imshow(step2_window, np.vstack([display, banner]))

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

                cv2.destroyWindow(step2_window)
                open_windows.remove(step2_window)
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


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"Auto Clicker {VERSION}")
        self.geometry("1300x760")
        self.minsize(1050, 600)
        
        self.tab_frames = {}
        self.tab_counter = 0
        self._closing = False
        self._start_all_generation = 0
        self._initializing_tabs = True
        self._ui_queue = queue.SimpleQueue()
        self.opencv_lock = threading.Lock()
        self._ui_pump_id = self.after(50, self._drain_ui_queue)
        self._timer_pump_id = self.after(500, self._pump_timer_updates)

        # Preload templates in background thread
        threading.Thread(
            target=self._preload_configured_templates,
            daemon=True,
        ).start()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Top Global Control Toolbar ---
        self.top_bar = ctk.CTkFrame(self, fg_color=COLOR_CARD_BG, corner_radius=RADIUS_MD, height=52)
        self.top_bar.pack(fill="x", padx=10, pady=(10, 5))

        self.logo_label = ctk.CTkLabel(
            self.top_bar, 
            text="⚡ Auto Clicker", 
            font=ctk.CTkFont(size=17, weight="bold")
        )
        self.logo_label.pack(side="left", padx=(16, 16), pady=8)

        # Top Bar Control Buttons (Unified style & font)
        btn_font = ctk.CTkFont(size=12, weight="bold")

        self.connect_all_btn = ctk.CTkButton(
            self.top_bar,
            text="전체 연결",
            font=btn_font,
            width=105,
            height=34,
            fg_color=COLOR_PRIMARY,
            hover_color=COLOR_PRIMARY_HOVER,
            corner_radius=RADIUS_MD,
            command=self.connect_all_instances
        )
        self.connect_all_btn.pack(side="left", padx=4, pady=8)

        self.start_all_btn = ctk.CTkButton(
            self.top_bar,
            text="전체 시작",
            font=btn_font,
            width=105,
            height=34,
            fg_color=COLOR_SUCCESS,
            hover_color=COLOR_SUCCESS_HOVER,
            corner_radius=RADIUS_MD,
            command=self.start_all_clickers
        )
        self.start_all_btn.pack(side="left", padx=4, pady=8)

        self.stop_all_btn = ctk.CTkButton(
            self.top_bar,
            text="전체 중지",
            font=btn_font,
            width=105,
            height=34,
            fg_color=COLOR_DANGER,
            hover_color=COLOR_DANGER_HOVER,
            corner_radius=RADIUS_MD,
            command=self.stop_all_clickers
        )
        self.stop_all_btn.pack(side="left", padx=4, pady=8)

        self.add_tab_btn = ctk.CTkButton(
            self.top_bar,
            text="인스턴스 추가",
            font=btn_font,
            width=120,
            height=34,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_MD,
            command=self.add_new_instance_dialog
        )
        self.add_tab_btn.pack(side="left", padx=4, pady=8)

        # Settings Button (Top-Right 최상단 우측)
        self.settings_btn = ctk.CTkButton(
            self.top_bar,
            text="설정",
            font=btn_font,
            width=80,
            height=34,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=RADIUS_MD,
            command=self.open_global_settings
        )
        self.settings_btn.pack(side="right", padx=(4, 16), pady=8)


        # --- Main Workspace: Tab View ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        try:
            self.tabview._segmented_button.configure(font=ctk.CTkFont(size=12, weight="bold"))
        except Exception:
            pass

        # Load saved instances or create initial tab
        self.load_and_initialize_tabs()

        # Keyboard shortcuts
        self.bind("<Control-f>", self.on_ctrl_f)
        self.bind("<Control-F>", self.on_ctrl_f)

        # Auto refresh devices after short delay
        self.after(500, self.refresh_all_devices)

    def get_current_tab_frame(self):
        try:
            curr_tab_name = self.tabview.get()
            return self.tab_frames.get(curr_tab_name)
        except Exception:
            return None

    def on_ctrl_f(self, event=None):
        frame = self.get_current_tab_frame()
        if frame and hasattr(frame, "focus_search_box"):
            frame.focus_search_box()
            return "break"

    @staticmethod
    def _preload_configured_templates():
        config = AutoClicker.read_config(CONFIG_PATH)
        default_mode = AutoClicker._safe_bool(
            config.get("global_grayscale_matching", True),
            True,
        )
        modes = {default_mode}
        for inst in config.get("instances", []):
            modes.add(
                AutoClicker._safe_bool(
                    inst.get("grayscale_matching", default_mode),
                    default_mode,
                )
            )
        AutoClicker.preload_templates(grayscales=tuple(modes))

    def format_tab_name(self, index, device_address):
        addr_str = str(device_address).strip()
        if addr_str.startswith("emulator-"):
            short_addr = addr_str.replace("emulator-", "")
        elif addr_str.startswith("127.0.0.1:"):
            short_addr = addr_str.replace("127.0.0.1:", ":")
        elif len(addr_str) > 14:
            short_addr = addr_str[-12:]
        else:
            short_addr = addr_str
        return f"Inst {index} ({short_addr})"

    def post_to_ui(self, callback):
        if not self._closing:
            self._ui_queue.put(callback)

    def _drain_ui_queue(self):
        if self._closing:
            return
        for _ in range(100):
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            except Exception:
                pass
        self._ui_pump_id = self.after(50, self._drain_ui_queue)

    def _pump_timer_updates(self):
        if self._closing:
            return
        try:
            current_tab = self.tabview.get()
        except Exception:
            current_tab = None

        now = time.monotonic()
        for name, frame in tuple(self.tab_frames.items()):
            try:
                if name == current_tab:
                    frame.update_timer_display()
                else:
                    # Lightweight warning status check every 500ms
                    frame.check_tab_warning_status()
                    # Throttle background tab UI text updates to every 2.5 seconds
                    last_update = getattr(frame, "_last_bg_timer_update", 0.0)
                    if not isinstance(last_update, (int, float)):
                        last_update = 0.0
                    if now - last_update >= 2.5:
                        frame._last_bg_timer_update = now
                        frame.update_timer_display()
            except Exception:
                pass
        self._timer_pump_id = self.after(500, self._pump_timer_updates)

    def load_and_initialize_tabs(self):
        config = AutoClicker.read_config(CONFIG_PATH)
        saved_addresses = []
        instances = config.get("instances", [])
        if isinstance(instances, list):
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                address = instance.get("device_address")
                if address and address not in saved_addresses:
                    saved_addresses.append(address)
        if not saved_addresses and config.get("device_address"):
            saved_addresses.append(config["device_address"])
        if not saved_addresses:
            saved_addresses = ["127.0.0.1:5555"]

        first_tab = None
        try:
            for address in saved_addresses:
                frame = self.add_instance_tab(device_address=address)
                if frame is not None and first_tab is None:
                    first_tab = frame.tab_name
        finally:
            self._initializing_tabs = False

        if first_tab:
            self.tabview.set(first_tab)
            self.after(100, lambda: self.tabview.set(first_tab))

        if config.get("reset_counts_on_startup", False):
            self.reset_all_template_counts(is_fallback=False)
            self.reset_all_template_counts(is_fallback=True)

        self.save_app_config()

    def add_new_instance_dialog(self):
        AddInstanceWindow(self)

    def open_global_settings(self):
        try:
            current_tab = self.tabview.get()
            tab_frame = self.tab_frames.get(current_tab)
        except Exception:
            tab_frame = None
        if tab_frame is None and self.tab_frames:
            tab_frame = next(iter(self.tab_frames.values()))
        if tab_frame is not None:
            tab_frame.open_settings_window()

    def add_instance_tab(self, device_address="127.0.0.1:5555"):
        device_address = str(device_address).strip()
        if not device_address:
            return None
        for frame in self.tab_frames.values():
            if frame.clicker.device_address == device_address:
                self.tabview.set(frame.tab_name)
                return None

        self.tab_counter += 1
        tab_name = self.format_tab_name(self.tab_counter, device_address)
        suffix = 1
        base_name = tab_name
        while tab_name in self.tab_frames:
            suffix += 1
            tab_name = f"{base_name}_{suffix}"

        tab_obj = self.tabview.add(tab_name)
        frame = InstanceTabFrame(
            tab_obj,
            app_owner=self,
            tab_name=tab_name,
            device_address=device_address,
        )
        frame.pack(fill="both", expand=True)
        self.tab_frames[tab_name] = frame
        self.tabview.set(tab_name)
        if not self._initializing_tabs:
            self.save_app_config()
        return frame

    def remove_instance_tab(self, tab_name):
        if len(self.tab_frames) <= 1:
            from tkinter import messagebox
            messagebox.showwarning("삭제 불가", "최소 1개의 인스턴스 탭이 존재해야 합니다.")
            return

        frame = self.tab_frames.pop(tab_name, None)
        if frame is None:
            return
        frame.shutdown()
        self.tabview.delete(tab_name)
        self.save_app_config()

    def refresh_all_tabs_templates(self):
        for frame in tuple(self.tab_frames.values()):
            frame.refresh_templates()

    def update_all_template_actions(self, filename, action, is_fallback=False):
        action_text, action_fg, action_hover = get_action_button_style(action)
        for frame in tuple(self.tab_frames.values()):
            if is_fallback:
                frame.clicker.fallback_template_actions[filename] = action
                row_dict = frame.fallback_template_row_widgets
            else:
                frame.clicker.template_actions[filename] = action
                row_dict = frame.template_row_widgets
            if filename in row_dict:
                row_dict[filename].action_btn.configure(
                    text=action_text, fg_color=action_fg, hover_color=action_hover
                )

    def update_all_template_delays(self, filename, delay, delay_type, is_fallback=False):
        delay_text, delay_fg, delay_hover = get_delay_button_style(delay, delay_type)
        for frame in tuple(self.tab_frames.values()):
            if is_fallback:
                frame.clicker.fallback_template_delays[filename] = delay
                frame.clicker.fallback_template_delay_types[filename] = delay_type
                row_dict = frame.fallback_template_row_widgets
            else:
                frame.clicker.template_delays[filename] = delay
                frame.clicker.template_delay_types[filename] = delay_type
                row_dict = frame.template_row_widgets
            if filename in row_dict:
                row_dict[filename].delay_btn.configure(
                    text=delay_text, fg_color=delay_fg, hover_color=delay_hover
                )

    def update_all_template_order(self, order, is_fallback=False):
        for frame in tuple(self.tab_frames.values()):
            if is_fallback:
                frame.clicker.fallback_template_order = list(order)
                row_dict = frame.fallback_template_row_widgets
            else:
                frame.clicker.template_order = list(order)
                row_dict = frame.template_row_widgets
            for w in row_dict.values():
                w.frame.pack_forget()
            for idx, filename in enumerate(order):
                if filename in row_dict:
                    w = row_dict[filename]
                    w.frame.pack(fill="x", padx=5, pady=1)
                    w.priority.configure(text=f"{idx+1}.")
            frame.filter_templates(is_fallback=is_fallback)

    def remove_template_from_all_tabs(self, filename, is_fallback=False):
        for frame in tuple(self.tab_frames.values()):
            if is_fallback:
                order = frame.clicker.fallback_template_order
                if filename in order:
                    order.remove(filename)
                frame.clicker.fallback_template_counts.pop(filename, None)
                frame.clicker.fallback_template_actions.pop(filename, None)
                frame.clicker.fallback_template_offsets.pop(filename, None)
                frame.clicker.fallback_template_delays.pop(filename, None)
                frame.clicker.fallback_template_delay_types.pop(filename, None)
                frame.clicker.fallback_template_rois.pop(filename, None)
                row_dict = frame.fallback_template_row_widgets
                count_dict = frame.fallback_template_count_labels
            else:
                order = frame.clicker.template_order
                if filename in order:
                    order.remove(filename)
                frame.clicker.template_counts.pop(filename, None)
                frame.clicker.template_actions.pop(filename, None)
                frame.clicker.template_offsets.pop(filename, None)
                frame.clicker.template_delays.pop(filename, None)
                frame.clicker.template_delay_types.pop(filename, None)
                frame.clicker.template_rois.pop(filename, None)
                row_dict = frame.template_row_widgets
                count_dict = frame.template_count_labels
            if filename in row_dict:
                w = row_dict.pop(filename)
                w.frame.destroy()
            if filename in count_dict:
                count_dict.pop(filename, None)
            for idx, fn in enumerate(order):
                if fn in row_dict:
                    row_dict[fn].priority.configure(text=f"{idx+1}.")
            frame.filter_templates(is_fallback=is_fallback)

    def add_template_to_all_tabs(self, filename, offset_x=0, offset_y=0, is_fallback=False):
        for frame in tuple(self.tab_frames.values()):
            if is_fallback:
                order = frame.clicker.fallback_template_order
                parent = frame.fallback_templates_frame
                frame.clicker.fallback_template_offsets[filename] = [offset_x, offset_y]
                frame.clicker.fallback_template_counts.setdefault(filename, 0)
                frame.clicker.fallback_template_actions.setdefault(filename, "click")
                frame.clicker.fallback_template_delays.setdefault(filename, 0.0)
                frame.clicker.fallback_template_delay_types.setdefault(filename, "pre")
            else:
                order = frame.clicker.template_order
                parent = frame.templates_frame
                frame.clicker.template_offsets[filename] = [offset_x, offset_y]
                frame.clicker.template_counts.setdefault(filename, 0)
                frame.clicker.template_actions.setdefault(filename, "click")
                frame.clicker.template_delays.setdefault(filename, 0.0)
                frame.clicker.template_delay_types.setdefault(filename, "pre")
            if filename not in order:
                order.append(filename)
            idx = order.index(filename)
            row_dict = frame.fallback_template_row_widgets if is_fallback else frame.template_row_widgets
            if filename not in row_dict:
                frame.create_template_row(parent, filename, idx, is_fallback=is_fallback)
            frame.filter_templates(is_fallback=is_fallback)

    def reset_all_template_counts(self, is_fallback=False):
        for frame in tuple(self.tab_frames.values()):
            if is_fallback:
                frame.clicker.reset_fallback_counts()
                count_dict = frame.fallback_template_count_labels
            else:
                frame.clicker.reset_counts()
                count_dict = frame.template_count_labels
            for label in count_dict.values():
                if label and label.winfo_exists():
                    label.configure(text="Clicks: 0")

    def update_template_count_for_all(self, filename, count, is_fallback):
        for frame in tuple(self.tab_frames.values()):
            labels = (
                frame.fallback_template_count_labels
                if is_fallback
                else frame.template_count_labels
            )
            label = labels.get(filename)
            if label is not None and label.winfo_exists():
                label.configure(text=f"Clicks: {count}")

    def start_all_clickers(self):
        self._start_all_generation += 1
        generation = self._start_all_generation
        def start_frame(frame):
            if (
                generation == self._start_all_generation
                and not frame._destroyed
                and frame.clicker.device
                and not frame.clicker.is_running
            ):
                frame.start_clicker_loop()

        scheduled = 0
        for frame in self.tab_frames.values():
            if frame.clicker.device and not frame.clicker.is_running:
                self.after(scheduled * 150, lambda target=frame: start_frame(target))
                scheduled += 1

    def stop_all_clickers(self):
        self._start_all_generation += 1
        for frame in self.tab_frames.values():
            # Pending staggered starts were invalidated before iteration.
            if frame.clicker.is_running or frame._loop_starting:
                frame.stop_clicker_loop()

    def connect_all_instances(self):
        for frame in self.tab_frames.values():
            if frame.clicker.device is None:
                frame.toggle_connection()

    def refresh_all_devices(self):
        def task():
            dummy_clicker = AutoClicker()
            devices = dummy_clicker.get_connected_devices()

            def update_ui():
                if self._closing:
                    return
                for frame in tuple(self.tab_frames.values()):
                    current_list = list(devices)
                    address = frame.clicker.device_address
                    if address and address not in current_list:
                        current_list.append(address)
                    frame.update_device_combo_values(current_list)

            self.post_to_ui(update_ui)

        threading.Thread(target=task, daemon=True).start()

    def sync_shared_settings_to_all(self, source_frame=None):
        """Synchronize shared settings across all tab frames and their clicker instances."""
        if not self.tab_frames:
            return
        if source_frame is None:
            source_frame = next(iter(self.tab_frames.values()))
        master_settings = source_frame.clicker._settings_dict()

        for frame in tuple(self.tab_frames.values()):
            if frame is not source_frame:
                frame.clicker._apply_settings(master_settings)
                if hasattr(frame, "fb_final_combo") and frame.fb_final_combo.winfo_exists():
                    curr_fb_action = getattr(frame.clicker, "fallback_final_action", "none")
                    curr_fb_label = NO_MATCH_ACTION_MAP.get(curr_fb_action, "사용 안 함 (Disabled)")
                    frame.fb_final_combo.set(curr_fb_label)
                    coords = getattr(frame.clicker, "fallback_final_coords", [500, 500])
                    frame.fb_final_x_entry.delete(0, "end")
                    frame.fb_final_x_entry.insert(0, str(coords[0]))
                    frame.fb_final_y_entry.delete(0, "end")
                    frame.fb_final_y_entry.insert(0, str(coords[1]))
                    frame.update_fb_final_coord_visibility(curr_fb_action)

    def save_app_config(self):
        instances_data = [
            {"device_address": frame.clicker.device_address}
            for frame in self.tab_frames.values()
        ]
        primary_settings = None
        if self.tab_frames:
            primary_settings = next(iter(self.tab_frames.values())).clicker._settings_dict()
        return AutoClicker.update_instances_config(
            instances_data,
            CONFIG_PATH,
            primary_settings=primary_settings,
        )

    def on_closing(self):
        if self._closing:
            return
        self._closing = True
        TemplatePreviewTooltip.get_instance().hide()
        self._start_all_generation += 1
        frames = tuple(self.tab_frames.values())
        for frame in frames:
            frame.begin_shutdown()
        for frame in frames:
            frame.shutdown()
        self.save_app_config()
        try:
            self.after_cancel(self._ui_pump_id)
        except Exception:
            pass
        try:
            self.after_cancel(self._timer_pump_id)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()


