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
from autoclicker.gui.widgets.tooltip import TemplatePreviewTooltip

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
