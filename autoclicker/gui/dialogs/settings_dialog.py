import os
import sys
import threading
from contextlib import ExitStack
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
from autoclicker.gui.dialogs.license_dialog import LicenseNoticeWindow

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
                banner_h = 65
                window_key = f"AutoClicker_PickCoord_{getattr(self.clicker, 'device_address', 'dev')}"
                window_title = f"[좌표 선택] 화면 클릭 후 Enter로 확정 - {getattr(self.clicker, 'device_address', '')}"
                window_name = window_key
                cv2.namedWindow(window_key, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_key, width // 2, (height + banner_h) // 2)
                set_opencv_window_title(window_key, window_title)

                def update_preview():
                    display = screen.copy()
                    x, y = selected_pt
                    cv2.circle(display, (x, y), 8, (0, 0, 255), 2)
                    cv2.drawMarker(display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                    banner = draw_korean_banner(
                        width,
                        banner_h,
                        [
                            (f"선택 좌표: (X={x}, Y={y})", (0, 255, 255), 15),
                            ("화면 클릭으로 좌표 지정 ➔ Enter/Space: 적용 (취소: 'c', ESC 또는 [X])", (220, 220, 220), 13),
                        ],
                    )
                    cv2.imshow(window_key, np.vstack([display, banner]))
                    set_opencv_window_title(window_key, window_title)

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
