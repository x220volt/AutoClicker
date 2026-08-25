"""Windows and OpenCV UI rendering helpers for Korean text and Unicode titlebars."""

import os
import sys
import cv2
import numpy as np


def draw_korean_banner(width, banner_height, lines):
    """Render crisp Korean and English text lines on an OpenCV banner using PIL."""
    banner = np.full((banner_height, width, 3), 30, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont

        banner_rgb = cv2.cvtColor(banner, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(banner_rgb)
        draw = ImageDraw.Draw(pil_img)

        font_candidates = [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/malgunbd.ttf",
            "C:/Windows/Fonts/gulim.ttc",
            "malgun.ttf",
        ]
        chosen_font_path = None
        for fp in font_candidates:
            if os.path.exists(fp):
                chosen_font_path = fp
                break

        y = 10
        for text, color_bgr, font_size in lines:
            try:
                if chosen_font_path:
                    font = ImageFont.truetype(chosen_font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            color_rgb = (
                (color_bgr[2], color_bgr[1], color_bgr[0])
                if len(color_bgr) == 3
                else (255, 255, 255)
            )
            draw.text((15, y), text, font=font, fill=color_rgb)
            y += font_size + 8

        result_rgb = np.array(pil_img)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return banner


def set_opencv_window_title(window_key, unicode_title):
    """Set Unicode window title on Windows via SetWindowTextW to prevent mojibake."""
    if sys.platform == "win32":
        try:
            import ctypes

            hwnd = ctypes.windll.user32.FindWindowW(None, window_key)
            if hwnd:
                ctypes.windll.user32.SetWindowTextW(hwnd, str(unicode_title))
        except Exception:
            pass
