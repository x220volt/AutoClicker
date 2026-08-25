"""GUI theme colors, styles, and action presentation helpers."""

import customtkinter as ctk

# Ensure dark mode appearance
ctk.set_appearance_mode("Dark")

# Unified Modern Dark Design System Color & Style Tokens
COLOR_PRIMARY = "#2D5F9E"          # Soft Royal Blue - Primary Actions / Focus / Connect
COLOR_PRIMARY_HOVER = "#3B73BD"

COLOR_SUCCESS = "#257A4E"          # Deep Emerald Green - Start / Active / Single Click
COLOR_SUCCESS_HOVER = "#2FA066"

COLOR_DANGER = "#8A2E3B"           # Soft Crimson Wine - Stop / Disconnect / Delete
COLOR_DANGER_HOVER = "#A43949"

COLOR_DANGER_MUTED = "#48262D"     # Subdued Dark Wine - Secondary actions like Reset Counts
COLOR_DANGER_MUTED_HOVER = "#5C3039"

COLOR_WARNING = "#915C16"          # Warm Ochre Amber - Back Action / Pre-Delay / Warnings
COLOR_WARNING_HOVER = "#AB6D1B"

COLOR_INFO = "#49528F"             # Slate Indigo - Post-Delay / Special Actions
COLOR_INFO_HOVER = "#5864AD"

COLOR_NEUTRAL = "#2D3442"          # Dark Slate - Settings / Neutral / Utility
COLOR_NEUTRAL_HOVER = "#3C4557"

COLOR_CARD_BG = ("#2B2F3A", "#181B22")
COLOR_CARD_INNER = ("#333945", "#212631")
COLOR_BORDER = "#2E3545"

COLOR_TEXT_MUTED = "#94A3B8"
COLOR_TEXT_PRIMARY = "#F1F5F9"

RADIUS_SM = 5
RADIUS_MD = 6
RADIUS_LG = 8

NO_MATCH_ACTION_MAP = {
    "fallback_list": "미매칭 복구 템플릿 목록 매칭",
    "none": "사용 안 함 (Disabled)",
    "random_click": "화면 랜덤 클릭 (Random Click)",
    "custom_click": "특정 좌표 클릭 (Custom Click)",
    "custom_double_click": "특정 좌표 더블클릭 (Double Click)",
    "back": "뒤로가기 (Back Key)",
}
REVERSE_NO_MATCH_ACTION_MAP = {v: k for k, v in NO_MATCH_ACTION_MAP.items()}


def get_action_button_style(action):
    """Return display text, base color, and hover color for template action buttons."""
    if action == "back":
        return "뒤로 (Back)", COLOR_WARNING, COLOR_WARNING_HOVER
    elif action in ("double_click", "click_click", "double"):
        return "더블 (Double)", COLOR_PRIMARY, COLOR_PRIMARY_HOVER
    else:
        return "클릭 (Click)", COLOR_SUCCESS, COLOR_SUCCESS_HOVER


def get_delay_button_style(delay, delay_type="pre"):
    """Return display text, base color, and hover color for delay configuration buttons."""
    if delay > 0:
        if delay_type == "post":
            return f"동작후 {delay:g}s", COLOR_INFO, COLOR_INFO_HOVER
        else:
            return f"동작전 {delay:g}s", COLOR_WARNING, COLOR_WARNING_HOVER
    else:
        return "딜레이 0s", COLOR_NEUTRAL, COLOR_NEUTRAL_HOVER
