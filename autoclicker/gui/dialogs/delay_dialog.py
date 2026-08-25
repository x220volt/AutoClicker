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
