from main import AutoClicker, ADB_PATH, ADB_HOST, ADB_PORT, DEVICE_ADDRESS
import cv2

def test():
    print("=== 연결 및 캡처 테스트 시작 ===")
    clicker = AutoClicker(ADB_PATH, ADB_HOST, ADB_PORT, DEVICE_ADDRESS)
    
    if clicker.start_adb_server():
        print("1. ADB 서버 및 장치 연결 성공")
        
        print("2. 화면 캡처 시도 중...")
        screen = clicker.capture_screen()
        
        if screen is not None:
            save_path = "debug_screen.png"
            cv2.imwrite(save_path, screen)
            print(f"3. 화면 캡처 성공! '{save_path}'로 저장되었습니다.")
            print("현재 화면을 확인하여 에뮬레이터 화면이 맞는지 확인하세요.")
        else:
            print("3. 화면 캡처 실패")
    else:
        print("1. 연결 실패. MSI App Player가 켜져 있는지, ADB가 활성화되었는지 확인하세요.")
    clicker.shutdown()

if __name__ == "__main__":
    test()
