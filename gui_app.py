import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import os
import threading
import time
import cv2
import numpy as np
import queue
from main import TeraboxClicker, CONFIG_PATH, ADB_COMMAND_TIMEOUT

VERSION = "v0.3.6"

# 앱 전역 테마를 Dark 모드로 고정
ctk.set_appearance_mode("Dark")

NO_MATCH_ACTION_MAP = {
    "fallback_list": "📋 미매칭 복구 템플릿 목록 매칭 (Fallback Match List)",
    "none": "사용 안 함 (Disabled)",
    "random_click": "🎲 화면 랜덤 클릭 (Random Click)",
    "custom_click": "👆 특정 좌표 클릭 (Custom Click)",
    "custom_double_click": "✌️ 특정 좌표 클릭클릭 (Double Click)",
    "back": "↩️ 뒤로가기 (Back Key)"
}
REVERSE_NO_MATCH_ACTION_MAP = {v: k for k, v in NO_MATCH_ACTION_MAP.items()}


def get_action_button_style(action):
    if action == "back":
        return "Back (뒤로가기)", "#E67E22", "#D35400"
    elif action in ("double_click", "click_click", "double"):
        return "Double (클릭클릭)", "#2980B9", "#1F618D"
    else:
        return "Click (클릭)", "#27AE60", "#1E8449"


def get_delay_button_style(delay, delay_type="pre"):
    if delay > 0:
        if delay_type == "post":
            return f"⏱️후 {delay:g}s", "#1A5276", "#2471A3"
        else:
            return f"⏱️전 {delay:g}s", "#7D6608", "#9A7D0A"
    else:
        return "⏱️ 0s", "#333333", "#444444"


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

            x = widget.winfo_rootx() + widget.winfo_width() + 8
            y = widget.winfo_rooty() - 10

            screen_w = tw.winfo_screenwidth()
            screen_h = tw.winfo_screenheight()

            if x + w_width > screen_w - 10:
                x = widget.winfo_rootx() - w_width - 8
            if y + w_height > screen_h - 40:
                y = screen_h - w_height - 40
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
        self.title("Settings (공통 설정)")
        self.geometry("440x780")
        self.resizable(False, False)
        
        # 메인 창에 종속 설정 및 모달 효과
        self.transient(parent_frame.winfo_toplevel())
        self.grab_set()

        # Header
        self.header_label = ctk.CTkLabel(self, text="⚙️ 공통 설정 (Global Settings)", font=ctk.CTkFont(size=16, weight="bold"))
        self.header_label.pack(pady=(15, 10))

        # Scan Interval
        self.interval_label = ctk.CTkLabel(self, text="스캔 간격 (Scan Interval, sec):", anchor="w")
        self.interval_label.pack(fill="x", padx=30)
        self.interval_entry = ctk.CTkEntry(self)
        self.interval_entry.insert(0, str(self.clicker.scan_interval))
        self.interval_entry.pack(fill="x", padx=30, pady=(0, 8))
        self.interval_entry.bind("<KeyRelease>", self.schedule_save)
        self.interval_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # Double Click Interval
        self.double_click_label = ctk.CTkLabel(self, text="더블클릭(클릭클릭) 간격 (Double-Click Interval, sec):", anchor="w")
        self.double_click_label.pack(fill="x", padx=30)
        self.double_click_entry = ctk.CTkEntry(self)
        self.double_click_entry.insert(0, str(getattr(self.clicker, 'double_click_interval', 1.0)))
        self.double_click_entry.pack(fill="x", padx=30, pady=(0, 8))
        self.double_click_entry.bind("<KeyRelease>", self.schedule_save)
        self.double_click_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # Post-Action Delay (동작 후 대기 시간)
        self.post_delay_label = ctk.CTkLabel(self, text="동작 후 대기 시간 (Post-Action Delay, sec):", anchor="w")
        self.post_delay_label.pack(fill="x", padx=30)
        self.post_delay_entry = ctk.CTkEntry(self)
        self.post_delay_entry.insert(0, str(getattr(self.clicker, 'post_action_delay', 2.0)))
        self.post_delay_entry.pack(fill="x", padx=30, pady=(0, 8))
        self.post_delay_entry.bind("<KeyRelease>", self.schedule_save)
        self.post_delay_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # Similarity Threshold
        self.threshold_label = ctk.CTkLabel(self, text="이미지 유사도 임계값 (Similarity Threshold 0.1~1.0):", anchor="w")
        self.threshold_label.pack(fill="x", padx=30)
        self.threshold_entry = ctk.CTkEntry(self)
        self.threshold_entry.insert(0, str(self.clicker.similarity_threshold))
        self.threshold_entry.pack(fill="x", padx=30, pady=(0, 8))
        self.threshold_entry.bind("<KeyRelease>", self.schedule_save)
        self.threshold_entry.bind("<FocusOut>", lambda e: self.save_settings())
        self.grayscale_var = ctk.StringVar(
            value="on" if self.clicker.match_grayscale else "off"
        )
        self.grayscale_switch = ctk.CTkSwitch(
            self,
            text="고속 그레이스케일 매칭 (색상 구분 필요 시 끄기)",
            variable=self.grayscale_var,
            onvalue="on",
            offvalue="off",
            command=self.save_settings,
        )
        self.grayscale_switch.pack(fill="x", padx=30, pady=(0, 8))


        # Timeout Alert
        self.timeout_label = ctk.CTkLabel(self, text="매칭 없음 경고 알림 시간 (No-match Alert sec, 0: 끄기):", anchor="w")
        self.timeout_label.pack(fill="x", padx=30)
        self.timeout_entry = ctk.CTkEntry(self)
        self.timeout_entry.insert(0, str(self.clicker.no_match_timeout))
        self.timeout_entry.pack(fill="x", padx=30, pady=(0, 8))
        self.timeout_entry.bind("<KeyRelease>", self.schedule_save)
        self.timeout_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # Consecutive Match Alert
        self.consecutive_label = ctk.CTkLabel(self, text="동일 템플릿 연속 매칭 경고 횟수 (0: 끄기):", anchor="w")
        self.consecutive_label.pack(fill="x", padx=30)
        self.consecutive_entry = ctk.CTkEntry(self)
        self.consecutive_entry.insert(0, str(getattr(self.clicker, 'consecutive_match_threshold', 0)))
        self.consecutive_entry.pack(fill="x", padx=30, pady=(0, 10))
        self.consecutive_entry.bind("<KeyRelease>", self.schedule_save)
        self.consecutive_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # --- No-Match Action Section ---
        self.no_match_section_label = ctk.CTkLabel(self, text="⚡ 매칭 미발생 시 자동 동작 (No-Match Action):", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.no_match_section_label.pack(fill="x", padx=30, pady=(5, 2))

        curr_action = getattr(self.clicker, 'no_match_action', 'none')
        curr_label = NO_MATCH_ACTION_MAP.get(curr_action, "사용 안 함 (Disabled)")
        self.no_match_combo = ctk.CTkOptionMenu(
            self, 
            values=list(NO_MATCH_ACTION_MAP.values()),
            command=self.on_action_changed
        )
        self.no_match_combo.set(curr_label)
        self.no_match_combo.pack(fill="x", padx=30, pady=(0, 8))

        self.no_match_interval_label = ctk.CTkLabel(self, text="동작 대기 시간 (Action Interval, sec):", anchor="w")
        self.no_match_interval_label.pack(fill="x", padx=30)
        self.no_match_interval_entry = ctk.CTkEntry(self)
        self.no_match_interval_entry.insert(0, str(getattr(self.clicker, 'no_match_interval', 30)))
        self.no_match_interval_entry.pack(fill="x", padx=30, pady=(0, 8))
        self.no_match_interval_entry.bind("<KeyRelease>", self.schedule_save)
        self.no_match_interval_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # Custom Coordinate Frame (For Custom Click & Custom Double Click)
        self.coord_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.coord_frame.pack(fill="x", padx=30, pady=(0, 10))

        coords = getattr(self.clicker, 'no_match_coords', [500, 500])
        self.coord_x_label = ctk.CTkLabel(self.coord_frame, text="X:")
        self.coord_x_label.pack(side="left", padx=(0, 3))
        self.coord_x_entry = ctk.CTkEntry(self.coord_frame, width=70)
        self.coord_x_entry.insert(0, str(coords[0]))
        self.coord_x_entry.pack(side="left", padx=(0, 10))
        self.coord_x_entry.bind("<KeyRelease>", self.schedule_save)
        self.coord_x_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.coord_y_label = ctk.CTkLabel(self.coord_frame, text="Y:")
        self.coord_y_label.pack(side="left", padx=(0, 3))
        self.coord_y_entry = ctk.CTkEntry(self.coord_frame, width=70)
        self.coord_y_entry.insert(0, str(coords[1]))
        self.coord_y_entry.pack(side="left", padx=(0, 10))
        self.coord_y_entry.bind("<KeyRelease>", self.schedule_save)
        self.coord_y_entry.bind("<FocusOut>", lambda e: self.save_settings())

        self.pick_coord_btn = ctk.CTkButton(
            self.coord_frame, 
            text="🎯 화면에서 좌표 선택", 
            width=140,
            command=self.pick_coords_from_screen
        )
        self.pick_coord_btn.pack(side="right")

        self.update_coord_frame_visibility(curr_action)

        # Close Button
        self.close_btn = ctk.CTkButton(self, text="Close (닫기)", command=self.close_window)
        self.close_btn.pack(fill="x", padx=30, pady=(10, 15))

        self.protocol("WM_DELETE_WINDOW", self.close_window)

    def on_action_changed(self, choice):
        action_key = REVERSE_NO_MATCH_ACTION_MAP.get(choice, "none")
        self.update_coord_frame_visibility(action_key)
        self.save_settings()

    def update_coord_frame_visibility(self, action_key):
        if action_key in ("custom_click", "custom_double_click"):
            self.coord_frame.pack(fill="x", padx=30, pady=(0, 10))
        else:
            self.coord_frame.pack_forget()

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
        self._persist_settings()
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()
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

        type_title = "미매칭 복구 템플릿" if is_fallback else "기본 템플릿"
        self.title(f"지연 시간 설정 - {filename}")
        self.geometry("450x430")
        self.resizable(False, False)

        self.transient(parent_frame.winfo_toplevel())
        self.grab_set()

        # Header
        self.header_label = ctk.CTkLabel(
            self,
            text=f"⏱️ 지연 시간 설정 ({type_title})",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.header_label.pack(pady=(15, 5))

        self.filename_label = ctk.CTkLabel(
            self,
            text=f"파일명: {filename}",
            font=ctk.CTkFont(size=13),
            text_color="#60A5FA",
        )
        self.filename_label.pack(pady=(0, 15))

        # Timing Section (적용 시점)
        self.timing_label = ctk.CTkLabel(
            self,
            text="지연 시간 적용 시점 (Delay Timing):",
            font=ctk.CTkFont(weight="bold"),
            anchor="w",
        )
        self.timing_label.pack(fill="x", padx=30, pady=(0, 5))

        self.timing_segmented = ctk.CTkSegmentedButton(
            self,
            values=["동작 전 대기 (Pre-Action)", "동작 후 대기 (Post-Action)"],
            command=self._on_timing_changed,
            font=ctk.CTkFont(weight="bold"),
        )
        self.timing_segmented.set(
            "동작 후 대기 (Post-Action)"
            if current_type == "post"
            else "동작 전 대기 (Pre-Action)"
        )
        self.timing_segmented.pack(fill="x", padx=30, pady=(0, 6))

        # Description box
        self.desc_frame = ctk.CTkFrame(self, fg_color=("#2B2B2B", "#1C1D1F"), corner_radius=6)
        self.desc_frame.pack(fill="x", padx=30, pady=(0, 12))

        self.desc_label = ctk.CTkLabel(
            self.desc_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#A6ACAF",
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
                height=24,
                fg_color="#333333",
                hover_color="#555555",
                command=lambda v=val: self._set_preset(v),
            )
            btn.pack(side="left", padx=2, expand=True)

        # Action Buttons
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(fill="x", padx=30, pady=(8, 15))

        self.save_btn = ctk.CTkButton(
            self.btn_frame,
            text="💾 저장 (Save)",
            fg_color="#27AE60",
            hover_color="#1E8449",
            command=self.save_and_close,
        )
        self.save_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.cancel_btn = ctk.CTkButton(
            self.btn_frame,
            text="취소 (Cancel)",
            fg_color="#4A4A4A",
            hover_color="#5A5A5A",
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
                text_color="#5DADE2",
            )
        else:
            self.desc_label.configure(
                text="💡 [동작 전 대기]\n화면에서 템플릿을 인식한 직후, 클릭(동작)을 실행하기 전에 지정된 시간 동안 대기합니다.",
                text_color="#F5B041",
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

        self.loading_label = ctk.CTkLabel(self.scroll_frame, text="디바이스 검색 중 (ADB scanning)...", text_color="gray")
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
            fg_color="#2980B9",
            hover_color="#1F618D",
            command=self.add_selected
        )
        self.add_selected_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.add_and_connect_btn = ctk.CTkButton(
            self.btn_frame,
            text="⚡ 전체 추가 & 자동 연결",
            fg_color="green",
            hover_color="#006400",
            command=self.add_and_connect_all
        )
        self.add_and_connect_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # 스레드로 디바이스 목록 검색
        threading.Thread(target=self.fetch_devices, daemon=True).start()

    def fetch_devices(self):
        dummy = TeraboxClicker()
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

        self.clicker = TeraboxClicker(
            device_address=device_address,
            on_timeout_callback=self.on_no_match_timeout,
            logger=self.log_message,
            on_match_callback=self.on_template_match,
            on_consecutive_match_callback=self.on_consecutive_match_warning,
        )
        self.clicker_thread = None
        self.settings_window = None

        # --- Top Header Bar inside Tab ---
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.pack(fill="x", padx=10, pady=(10, 5))
        self.normal_header_fg = self.header_frame.cget("fg_color")

        # Device Address Selector
        self.device_addr_label = ctk.CTkLabel(self.header_frame, text="Device:", font=ctk.CTkFont(weight="bold"))
        self.device_addr_label.pack(side="left", padx=(10, 5))

        initial_devices = [self.clicker.device_address]
        self.device_combo = ctk.CTkComboBox(
            self.header_frame,
            values=initial_devices,
            width=160,
            command=self.on_device_selected
        )
        self.device_combo.set(self.clicker.device_address)
        self.device_combo.pack(side="left", padx=(0, 5))
        self.device_combo.bind("<Return>", self.save_settings)
        self.device_combo.bind("<FocusOut>", self.save_settings)

        # Connect / Disconnect Button
        self.connect_button = ctk.CTkButton(
            self.header_frame,
            text="Connect Device",
            width=120,
            command=self.toggle_connection
        )
        self.connect_button.pack(side="left", padx=5)

        # Start / Stop Clicker Button
        self.start_button = ctk.CTkButton(
            self.header_frame,
            text="Start Clicker",
            width=120,
            state="disabled",
            fg_color="green",
            hover_color="#006400",
            command=self.toggle_clicker
        )
        self.start_button.pack(side="left", padx=5)

        # Status Label
        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="Status: Disconnected",
            font=ctk.CTkFont(weight="bold"),
            text_color="gray"
        )
        self.status_label.pack(side="left", padx=15)

        # Right side buttons in tab header: Settings & Delete Tab
        self.delete_tab_btn = ctk.CTkButton(
            self.header_frame,
            text="❌ Delete Tab",
            width=90,
            fg_color="#8B0000",
            hover_color="#FF0000",
            command=lambda: self.app_owner.remove_instance_tab(self.tab_name)
        )
        self.delete_tab_btn.pack(side="right", padx=(5, 10))

        self.settings_button = ctk.CTkButton(
            self.header_frame,
            text="⚙️ Settings",
            width=90,
            fg_color="#4A4A4A",
            hover_color="#5A5A5A",
            command=self.open_settings_window
        )
        self.settings_button.pack(side="right", padx=5)

        # --- Real-time Status & Timer Info Bar ---
        self.timer_bar_frame = ctk.CTkFrame(self, fg_color=("#2B2B2B", "#1C1D1F"), corner_radius=6)
        self.timer_bar_frame.pack(fill="x", padx=10, pady=(0, 5))
        self.timer_bar_frame.grid_columnconfigure(0, weight=1)
        self.timer_bar_frame.grid_columnconfigure(1, weight=1)
        self.timer_bar_frame.grid_columnconfigure(2, weight=1)

        # 1. 미매칭 대기 시간 카드
        self.card_no_match = ctk.CTkFrame(self.timer_bar_frame, fg_color=("#333333", "#24252A"), corner_radius=5)
        self.card_no_match.grid(row=0, column=0, padx=5, pady=4, sticky="ew")
        self.no_match_timer_label = ctk.CTkLabel(
            self.card_no_match,
            text="⚡ 미매칭 대기: 정지됨",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        self.no_match_timer_label.pack(padx=10, pady=3)

        # 2. 매칭없음 경고 시간 카드
        self.card_timeout = ctk.CTkFrame(self.timer_bar_frame, fg_color=("#333333", "#24252A"), corner_radius=5)
        self.card_timeout.grid(row=0, column=1, padx=5, pady=4, sticky="ew")
        self.timeout_timer_label = ctk.CTkLabel(
            self.card_timeout,
            text="⚠️ 매칭없음 경고: 정지됨",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        self.timeout_timer_label.pack(padx=10, pady=3)

        # 3. 최근 매칭 정보 카드
        self.card_last_match = ctk.CTkFrame(self.timer_bar_frame, fg_color=("#333333", "#24252A"), corner_radius=5)
        self.card_last_match.grid(row=0, column=2, padx=5, pady=4, sticky="ew")
        self.last_match_info_label = ctk.CTkLabel(
            self.card_last_match,
            text="🎯 최근 매칭: 대기 중",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        self.last_match_info_label.pack(padx=10, pady=3)

        # --- Main Body (Split into Log View and Templates View) ---
        self.body_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.body_frame.grid_columnconfigure(0, weight=4)
        self.body_frame.grid_columnconfigure(1, weight=6)
        self.body_frame.grid_rowconfigure(0, weight=1)

        # Left Column: Log Box Frame
        self.log_frame = ctk.CTkFrame(self.body_frame)
        self.log_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.log_frame.grid_rowconfigure(1, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)

        self.log_header_label = ctk.CTkLabel(self.log_frame, text="📜 Activity Log", font=ctk.CTkFont(weight="bold"))
        self.log_header_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        self.log_textbox = ctk.CTkTextbox(self.log_frame, width=350)
        self.log_textbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.log_textbox.insert("0.0", f"Initialized Tab [{self.tab_name}] for ADB device [{self.clicker.device_address}]\n")
        self.log_textbox.configure(state="disabled")

        # Right Column: Active & Fallback Templates Frame
        self.templates_main_frame = ctk.CTkFrame(self.body_frame)
        self.templates_main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        self.templates_main_frame.grid_rowconfigure(0, weight=1)
        self.templates_main_frame.grid_columnconfigure(0, weight=1)

        # Tabview for Primary and Fallback Templates
        self.template_tabview = ctk.CTkTabview(self.templates_main_frame)
        self.template_tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.tab_primary = self.template_tabview.add("📋 기본 템플릿")
        self.tab_fallback = self.template_tabview.add("⚡ 미매칭 템플릿")

        # --- Setup Primary Tab ---
        self.primary_header_frame = ctk.CTkFrame(self.tab_primary, fg_color="transparent")
        self.primary_header_frame.pack(fill="x", padx=5, pady=(5, 5))

        self.primary_header_label = ctk.CTkLabel(self.primary_header_frame, text="Primary Templates", font=ctk.CTkFont(weight="bold"))
        self.primary_header_label.pack(side="left")

        self.reset_counts_button = ctk.CTkButton(
            self.primary_header_frame,
            text="Reset Counts",
            width=100,
            height=26,
            fg_color="#D9534F",
            hover_color="#C9302C",
            command=self.reset_counts_event
        )
        self.reset_counts_button.pack(side="right", padx=(5, 0))

        self.crop_button = ctk.CTkButton(
            self.primary_header_frame,
            text="+ Create Template",
            width=125,
            height=26,
            command=lambda: self.start_cropping(is_fallback=False),
            state="disabled"
        )
        self.crop_button.pack(side="right", padx=(0, 5))

        self.templates_frame = ctk.CTkScrollableFrame(self.tab_primary)
        self.templates_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # --- Setup Fallback Tab ---
        self.fallback_header_frame = ctk.CTkFrame(self.tab_fallback, fg_color="transparent")
        self.fallback_header_frame.pack(fill="x", padx=5, pady=(5, 5))

        self.fallback_header_label = ctk.CTkLabel(self.fallback_header_frame, text="No-Match Fallback Templates", font=ctk.CTkFont(weight="bold"))
        self.fallback_header_label.pack(side="left")

        self.fallback_timer_sublabel = ctk.CTkLabel(
            self.fallback_header_frame,
            text="⚡ 대기: --",
            font=ctk.CTkFont(size=12),
            text_color="#5DADE2"
        )
        self.fallback_timer_sublabel.pack(side="left", padx=(10, 0))

        self.reset_fb_counts_button = ctk.CTkButton(
            self.fallback_header_frame,
            text="Reset Counts",
            width=100,
            height=26,
            fg_color="#D9534F",
            hover_color="#C9302C",
            command=self.reset_fallback_counts_event
        )
        self.reset_fb_counts_button.pack(side="right", padx=(5, 0))

        self.crop_fb_button = ctk.CTkButton(
            self.fallback_header_frame,
            text="+ Create Fallback",
            width=125,
            height=26,
            fg_color="#3498DB",
            hover_color="#2980B9",
            command=lambda: self.start_cropping(is_fallback=True),
            state="disabled"
        )
        self.crop_fb_button.pack(side="right", padx=(0, 5))

        self.fallback_templates_frame = ctk.CTkScrollableFrame(self.tab_fallback)
        self.fallback_templates_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        # --- Fallback Final Action Control (Bottom Panel) ---
        self.fallback_final_frame = ctk.CTkFrame(self.tab_fallback, fg_color=("#2B2B2B", "#1E1E22"), corner_radius=6)
        self.fallback_final_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.fb_final_title = ctk.CTkLabel(
            self.fallback_final_frame,
            text="📌 모든 템플릿 불일치 시 최종 동작:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.fb_final_title.pack(side="left", padx=(10, 8), pady=6)

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
            width=180,
            command=self.on_fallback_final_action_changed
        )
        self.fb_final_combo.set(curr_fb_label)
        self.fb_final_combo.pack(side="left", padx=(0, 6), pady=6)

        # Coordinate inputs
        self.fb_final_coord_frame = ctk.CTkFrame(self.fallback_final_frame, fg_color="transparent")
        self.fb_final_coord_frame.pack(side="left", padx=(0, 5), pady=6)

        coords = getattr(self.clicker, 'fallback_final_coords', [500, 500])
        self.fb_final_x_label = ctk.CTkLabel(self.fb_final_coord_frame, text="X:")
        self.fb_final_x_label.pack(side="left", padx=(0, 2))
        self.fb_final_x_entry = ctk.CTkEntry(self.fb_final_coord_frame, width=50)
        self.fb_final_x_entry.insert(0, str(coords[0]))
        self.fb_final_x_entry.pack(side="left", padx=(0, 6))
        self.fb_final_x_entry.bind("<KeyRelease>", self.save_fallback_final_settings)
        self.fb_final_x_entry.bind("<FocusOut>", self.save_fallback_final_settings)

        self.fb_final_y_label = ctk.CTkLabel(self.fb_final_coord_frame, text="Y:")
        self.fb_final_y_label.pack(side="left", padx=(0, 2))
        self.fb_final_y_entry = ctk.CTkEntry(self.fb_final_coord_frame, width=50)
        self.fb_final_y_entry.insert(0, str(coords[1]))
        self.fb_final_y_entry.pack(side="left", padx=(0, 6))
        self.fb_final_y_entry.bind("<KeyRelease>", self.save_fallback_final_settings)
        self.fb_final_y_entry.bind("<FocusOut>", self.save_fallback_final_settings)

        self.fb_final_pick_btn = ctk.CTkButton(
            self.fb_final_coord_frame,
            text="🎯 좌표 선택",
            width=90,
            command=self.pick_fallback_final_coords
        )
        self.fb_final_pick_btn.pack(side="left", padx=2)

        self.update_fb_final_coord_visibility(curr_fb_action)

        # Test Run Button
        self.fb_final_test_btn = ctk.CTkButton(
            self.fallback_final_frame,
            text="⚡ 즉시 실행",
            width=90,
            fg_color="#D35400",
            hover_color="#A04000",
            command=self.test_fallback_final_action
        )
        self.fb_final_test_btn.pack(side="right", padx=(5, 10), pady=6)

        self.refresh_templates()

    def update_device_combo_values(self, device_list):
        if not self._destroyed:
            self.device_combo.configure(values=device_list)

    def _set_disconnected_ui(self):
        self.status_label.configure(text="Status: Disconnected", text_color="gray")
        self.start_button.configure(
            state="disabled", text="Start Clicker",
            fg_color="green", hover_color="#006400"
        )
        self.crop_button.configure(state="disabled")
        self.crop_fb_button.configure(state="disabled")
        self.connect_button.configure(state="normal", text="Connect Device")

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
        self.connect_button.configure(state="disabled", text="Connecting...")

        def connect_task():
            success = self.clicker.start_adb_server()

            def apply_result():
                if self._destroyed or generation != self._connection_generation:
                    if success:
                        self.clicker.disconnect()
                    return
                if success:
                    self.status_label.configure(
                        text=f"Status: Connected to {self.clicker.device_address}",
                        text_color="green",
                    )
                    self.start_button.configure(state="normal")
                    self.crop_button.configure(state="normal")
                    self.crop_fb_button.configure(state="normal")
                    self.connect_button.configure(state="normal", text="Disconnect")
                else:
                    self.status_label.configure(
                        text="Status: Connection Failed", text_color="red"
                    )
                    self.connect_button.configure(
                        state="normal", text="Connect Device"
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
            text="Stop Clicker", fg_color="red", hover_color="#8B0000"
        )
        self.status_label.configure(
            text=f"Status: Running ({self.clicker.device_address})",
            text_color="green",
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
                        text="Start Clicker",
                        fg_color="green",
                        hover_color="#006400",
                    )
                    self.status_label.configure(
                        text=(
                            f"Status: Connected to {self.clicker.device_address}"
                            if self.clicker.device
                            else "Status: Disconnected"
                        ),
                        text_color="green" if self.clicker.device else "gray",
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

    def refresh_templates(self):
        if self._destroyed or not self.winfo_exists():
            return

        TemplatePreviewTooltip.get_instance(self.winfo_toplevel()).hide()
        self.clicker.load_config()
        self.template_row_widgets = {}
        self.fallback_template_row_widgets = {}
        self.template_count_labels = {}
        self.fallback_template_count_labels = {}

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

    def create_template_row(self, parent_frame, filename, index, is_fallback=False):
        row_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        row_frame.pack(fill="x", padx=5, pady=1)

        # 1. 우측 고정 버튼 영역 (플랫 구조로 레이아웃 부하 최소화)
        del_cmd = (lambda f=filename: self.delete_fallback_template_event(f)) if is_fallback else (lambda f=filename: self.delete_template_event(f))
        del_btn = ctk.CTkButton(row_frame, text="X", width=28, height=25, 
                              fg_color="#8B0000", hover_color="#FF0000",
                              command=del_cmd)
        del_btn.pack(side="right", padx=(3, 0))

        action_dict = self.clicker.fallback_template_actions if is_fallback else self.clicker.template_actions
        action = action_dict.get(filename, "click")
        action_text, action_fg, action_hover = get_action_button_style(action)
        
        toggle_cmd = (lambda f=filename: self.toggle_fallback_action_event(f)) if is_fallback else (lambda f=filename: self.toggle_action_event(f))
        action_btn = ctk.CTkButton(row_frame, text=action_text, width=115, height=25,
                                   fg_color=action_fg, hover_color=action_hover,
                                   command=toggle_cmd)
        action_btn.pack(side="right", padx=(3, 0))

        delays_dict = self.clicker.fallback_template_delays if is_fallback else self.clicker.template_delays
        delay_types_dict = self.clicker.fallback_template_delay_types if is_fallback else self.clicker.template_delay_types
        delay = delays_dict.get(filename, 0.0)
        delay_type = delay_types_dict.get(filename, "pre")
        delay_text, delay_fg, delay_hover = get_delay_button_style(delay, delay_type)

        delay_cmd = (lambda f=filename: self.set_fallback_template_delay_event(f)) if is_fallback else (lambda f=filename: self.set_template_delay_event(f))
        delay_btn = ctk.CTkButton(row_frame, text=delay_text, width=62, height=25,
                                  fg_color=delay_fg, hover_color=delay_hover,
                                  command=delay_cmd)
        delay_btn.pack(side="right", padx=(3, 0))

        counts_dict = self.clicker.fallback_template_counts if is_fallback else self.clicker.template_counts
        count = counts_dict.get(filename, 0)
        count_label = ctk.CTkLabel(row_frame, text=f"Clicks: {count}", width=68, text_color="gray", anchor="e", font=ctk.CTkFont(size=11))
        count_label.pack(side="right", padx=(2, 6))

        # 2. 좌측 컨트롤 영역
        drag_handle = ctk.CTkLabel(row_frame, text="☰", width=22, cursor="fleur", font=ctk.CTkFont(size=13, weight="bold"), text_color="gray")
        drag_handle.pack(side="left", padx=(3, 0))

        priority_label = ctk.CTkLabel(row_frame, text=f"{index+1}.", width=28, anchor="w", font=ctk.CTkFont(size=12))
        priority_label.pack(side="left", padx=(2, 4))

        # 3. 중앙 가변 텍스트 라벨
        offset_dict = self.clicker.fallback_template_offsets if is_fallback else self.clicker.template_offsets
        offset_info = ""
        if filename in offset_dict:
            off_x, off_y = offset_dict[filename]
            offset_info = f"  ({off_x:+d},{off_y:+d})"
        
        label = ctk.CTkLabel(row_frame, text=f"{filename}{offset_info}", anchor="w", font=ctk.CTkFont(size=12))
        label.pack(side="left", fill="x", expand=True, padx=(0, 4))

        target_dir = self.clicker.fallback_template_dir if is_fallback else self.clicker.template_dir
        template_file_path = os.path.join(target_dir, filename)

        def on_enter(e):
            tooltip = TemplatePreviewTooltip.get_instance(self.winfo_toplevel())
            tooltip.schedule_show(label, template_file_path, filename, offset_info.strip())

        def on_leave(e):
            tooltip = TemplatePreviewTooltip.get_instance(self.winfo_toplevel())
            tooltip.cancel()

        # 이벤트 바인딩 최소화 (필요한 요소에만 정밀 바인딩)
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

        self.title(f"Terabox Auto Clicker Multi-Instance {VERSION}")
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
        self.top_bar = ctk.CTkFrame(self, height=50)
        self.top_bar.pack(fill="x", padx=10, pady=(10, 5))

        self.logo_label = ctk.CTkLabel(
            self.top_bar, 
            text="⚡ Terabox Clicker Multi", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.logo_label.pack(side="left", padx=(15, 15))

        # Top Bar Control Buttons (Unified style & font)
        btn_font = ctk.CTkFont(size=13, weight="bold")

        self.connect_all_btn = ctk.CTkButton(
            self.top_bar,
            text="🔗  전체 연결",
            font=btn_font,
            width=115,
            fg_color="#27AE60",
            hover_color="#1E8449",
            command=self.connect_all_instances
        )
        self.connect_all_btn.pack(side="left", padx=4)

        self.start_all_btn = ctk.CTkButton(
            self.top_bar,
            text="▶  전체 시작",
            font=btn_font,
            width=115,
            fg_color="#2EA043",
            hover_color="#238636",
            command=self.start_all_clickers
        )
        self.start_all_btn.pack(side="left", padx=4)

        self.stop_all_btn = ctk.CTkButton(
            self.top_bar,
            text="⏹  전체 중지",
            font=btn_font,
            width=115,
            fg_color="#DA3633",
            hover_color="#B62324",
            command=self.stop_all_clickers
        )
        self.stop_all_btn.pack(side="left", padx=4)

        self.add_tab_btn = ctk.CTkButton(
            self.top_bar,
            text="➕  인스턴스 탭 추가",
            font=btn_font,
            width=145,
            fg_color="#1F6FE5",
            hover_color="#1859B8",
            command=self.add_new_instance_dialog
        )
        self.add_tab_btn.pack(side="left", padx=4)


        # --- Main Workspace: Tab View ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        try:
            self.tabview._segmented_button.configure(font=ctk.CTkFont(size=12, weight="bold"))
        except Exception:
            pass

        # Load saved instances or create initial tab
        self.load_and_initialize_tabs()

        # Auto refresh devices after short delay
        self.after(500, self.refresh_all_devices)

    @staticmethod
    def _preload_configured_templates():
        config = TeraboxClicker.read_config(CONFIG_PATH)
        default_mode = TeraboxClicker._safe_bool(
            config.get("match_grayscale"), True
        )
        instances = config.get("instances", [])
        if not isinstance(instances, list):
            instances = []
        modes = {
            TeraboxClicker._safe_bool(
                instance.get("match_grayscale"), default_mode
            )
            for instance in instances
            if isinstance(instance, dict)
        }
        if not modes:
            modes.add(default_mode)
        TeraboxClicker.preload_templates(grayscales=tuple(modes))
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

        for name, frame in tuple(self.tab_frames.items()):
            try:
                if name == current_tab:
                    frame.update_timer_display()
                else:
                    frame.check_tab_warning_status()
            except Exception:
                pass
        self._timer_pump_id = self.after(500, self._pump_timer_updates)

    def load_and_initialize_tabs(self):
        config = TeraboxClicker.read_config(CONFIG_PATH)
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
        self.save_app_config()

    def add_new_instance_dialog(self):
        AddInstanceWindow(self)

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
            for idx, filename in enumerate(order):
                if filename in row_dict:
                    w = row_dict[filename]
                    w.frame.pack(fill="x", padx=5, pady=1)
                    w.priority.configure(text=f"{idx+1}.")

    def remove_template_from_all_tabs(self, filename, is_fallback=False):
        for frame in tuple(self.tab_frames.values()):
            if is_fallback:
                order = frame.clicker.fallback_template_order
                row_dict = frame.fallback_template_row_widgets
                count_dict = frame.fallback_template_count_labels
            else:
                order = frame.clicker.template_order
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
            dummy_clicker = TeraboxClicker()
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
        return TeraboxClicker.update_instances_config(
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


