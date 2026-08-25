import os
import sys
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from autoclicker import __version__

VERSION = f"v{__version__}"

from autoclicker.core.constants import (
    CONFIG_FILENAME,
    DEVICE_ADDRESS,
    DEFAULT_ADB_MODE,
    DEFAULT_CUSTOM_ADB_PATH,
    VALID_ADB_MODES,
)
from autoclicker.core.environment import (
    get_app_dir,
    get_default_adb_path,
    resolve_adb_path,
    CONFIG_PATH,
)
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
    LicenseNoticeWindow,
    AddInstanceWindow,
)
from autoclicker.gui.tab_view import InstanceTabFrame

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
                    label.configure(text="0회")

    def update_template_count_for_all(self, filename, count, is_fallback):
        for frame in tuple(self.tab_frames.values()):
            labels = (
                frame.fallback_template_count_labels
                if is_fallback
                else frame.template_count_labels
            )
            label = labels.get(filename)
            if label is not None and label.winfo_exists():
                label.configure(text=f"{count}회")

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


# Backward compatibility aliases
AutoClickerApp = App
TeraboxClickerApp = App


if __name__ == "__main__":
    app = App()
    app.mainloop()

