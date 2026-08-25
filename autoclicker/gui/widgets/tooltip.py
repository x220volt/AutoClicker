"""Hover preview tooltip showing template image thumbnail, dimensions, and offset."""

import os
import tkinter as tk
from PIL import Image, ImageTk


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
                    scaled_img = pil_img.resize(
                        (new_w, new_h), Image.Resampling.LANCZOS
                    )
                else:
                    scaled_img = pil_img

                photo_img = ImageTk.PhotoImage(scaled_img)
                self._image_cache[sig] = (photo_img, orig_w, orig_h)
                self._image_cache_order.append(sig)
                # LRU eviction
                while len(self._image_cache_order) > self._IMAGE_CACHE_MAX:
                    evict_sig = self._image_cache_order.pop(0)
                    self._image_cache.pop(evict_sig, None)

            parent = (
                self.root
                if (self.root and self.root.winfo_exists())
                else widget.winfo_toplevel()
            )
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
