"""A sleek, modern CustomTkinter-native popup context menu."""

import tkinter as tk
import customtkinter as ctk
from autoclicker.gui.theme import (
    COLOR_CARD_BG,
    COLOR_PRIMARY,
    COLOR_DANGER,
    COLOR_TEXT_PRIMARY,
    RADIUS_MD,
    RADIUS_SM,
)


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
        toplevel_parent = (
            parent.winfo_toplevel() if hasattr(parent, "winfo_toplevel") else parent
        )
        self.window = tk.Toplevel(toplevel_parent)
        self.window.wm_overrideredirect(True)
        self.window.wm_attributes("-topmost", True)
        CTKContextMenu._active_menu = self.window

        self.frame = ctk.CTkFrame(
            self.window,
            fg_color=COLOR_CARD_BG,
            border_color="#374151",
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        self.frame.pack(fill="both", expand=True)

        self._toplevel = toplevel_parent
        self._bind_id = toplevel_parent.bind(
            "<Button-1>", lambda e: self._check_click_outside(e), add="+"
        )
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
