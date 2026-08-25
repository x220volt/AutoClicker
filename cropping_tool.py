import cv2
import numpy as np
import os
import threading

from main import ADB_HOST, ADB_PATH, ADB_PORT, DEVICE_ADDRESS, AutoClicker


def normalize_filename(value):
    filename = value.strip()
    invalid_chars = set('<>:"/|?*') | {chr(92)}
    stem = os.path.splitext(filename)[0].rstrip(" .")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in range(1, 10)
    }
    if (
        not filename
        or os.path.basename(filename) != filename
        or any(char in invalid_chars for char in filename)
        or not stem
        or stem.upper() in reserved
        or filename != filename.rstrip(" .")
    ):
        raise ValueError("경로 문자나 Windows 예약 이름은 사용할 수 없습니다.")
    if not filename.lower().endswith(".png"):
        filename += ".png"
    return filename


def save_template(clicker, crop_img, filename, offset):
    filename = normalize_filename(filename)
    save_path = os.path.join(clicker.template_dir, filename)
    if os.path.exists(save_path):
        answer = input(f"'{filename}' 파일이 이미 있습니다. 덮어쓸까요? (y/N): ")
        if answer.strip().lower() != "y":
            return None

    temp_path = f"{save_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        success, encoded = cv2.imencode(".png", crop_img)
        if not success:
            raise ValueError("이미지 인코딩 실패")
        encoded.tofile(temp_path)
        os.replace(temp_path, save_path)
        clicker.invalidate_template_cache(save_path)

        if filename not in clicker.template_order:
            clicker.template_order.append(filename)
        clicker.template_counts.setdefault(filename, 0)
        clicker.template_actions.setdefault(filename, "click")
        clicker.template_offsets[filename] = [int(offset[0]), int(offset[1])]
        clicker.template_delays.setdefault(filename, 0.0)
        if not clicker.save_config(include_templates=True):
            raise OSError("이미지는 저장됐지만 config.json 갱신에 실패했습니다.")
        return save_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def select_click_point(screen, roi):
    roi_x, roi_y, roi_width, roi_height = map(int, roi)
    height, width = screen.shape[:2]
    banner_height = 65
    window_name = (
        "2. Select Click Point "
        "(Click target -> Enter / Default Center: Enter)"
    )
    default_point = [
        roi_x + roi_width // 2,
        roi_y + roi_height // 2,
    ]
    selected_point = list(default_point)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width // 2, (height + banner_height) // 2)
    cv2.moveWindow(window_name, 100, 100)

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
            display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 22, 2
        )
        banner = np.full((banner_height, width, 3), 30, dtype=np.uint8)
        offset_x, offset_y = x - roi_x, y - roi_y
        position_type = (
            " [Default Center]"
            if selected_point == default_point
            else " [Custom Target]"
        )
        cv2.putText(
            banner,
            f"Click Target: ({x}, {y}) | Offset: "
            f"({offset_x:+d}, {offset_y:+d}){position_type}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            banner,
            "Click screen -> Enter/Space: Confirm (Cancel: 'c' or ESC)",
            (15, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, np.vstack([display, banner]))

    def on_mouse(event, mouse_x, mouse_y, flags, param):
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and 0 <= mouse_y < height
            and 0 <= mouse_x < width
        ):
            selected_point[:] = [mouse_x, mouse_y]
            update_preview()

    cv2.setMouseCallback(window_name, on_mouse)
    update_preview()
    cancelled = False
    try:
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
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    if cancelled:
        return None
    return selected_point[0] - roi_x, selected_point[1] - roi_y


def main():
    print("=== 템플릿 추출 도구 실행 ===")
    clicker = AutoClicker(ADB_PATH, ADB_HOST, ADB_PORT, DEVICE_ADDRESS)
    try:
        if not clicker.start_adb_server():
            print("에뮬레이터 연결 실패. 설정과 포트를 확인하세요.")
            return

        while True:
            print("\n1. 현재 화면 캡처 및 영역 선택")
            print("2. 종료 (q)")
            choice = input("선택: ").strip().lower()
            if choice in {"q", "2"}:
                break
            if choice not in {"", "1"}:
                print("1, 2 또는 q를 입력하세요.")
                continue

            print("화면을 가져오는 중...")
            screen = clicker.capture_screen()
            if screen is None:
                print("캡처 실패")
                continue

            height, width = screen.shape[:2]
            window_name = (
                "1. Select Template Area "
                "(Drag & Press Enter / Cancel: c)"
            )
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width // 2, height // 2)
            cv2.moveWindow(window_name, 100, 100)
            try:
                roi = cv2.selectROI(
                    window_name,
                    screen,
                    showCrosshair=True,
                    fromCenter=False,
                )
            finally:
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass

            if roi[2] <= 0 or roi[3] <= 0:
                print("선택 취소됨")
                continue

            roi_x, roi_y, roi_width, roi_height = map(int, roi)
            crop_img = screen[
                roi_y:roi_y + roi_height,
                roi_x:roi_x + roi_width,
            ].copy()
            print(
                "[2단계] 실제 클릭 위치를 선택한 뒤 Enter를 누르세요. "
                "(기본 중앙: 바로 Enter)"
            )
            offset = select_click_point(screen, roi)
            if offset is None:
                print("클릭 위치 선택 취소됨")
                continue

            raw_name = input(
                "저장할 파일 이름을 입력하세요 "
                "(확장자 제외, 예: close): "
            )
            try:
                save_path = save_template(clicker, crop_img, raw_name, offset)
                if save_path:
                    print(
                        f"저장 완료: {save_path} "
                        f"(클릭 오프셋: {offset[0]:+d}, {offset[1]:+d})"
                    )
                else:
                    print("저장 취소됨")
            except (OSError, ValueError) as error:
                print(f"저장 중 오류: {error}")
    finally:
        cv2.destroyAllWindows()
        clicker.shutdown()
        print("도구를 종료합니다.")


if __name__ == "__main__":
    main()
