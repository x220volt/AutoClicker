"""GUI package for AutoClicker."""

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
from autoclicker.gui.win32_utils import (
    draw_korean_banner,
    set_opencv_window_title,
)
from autoclicker.gui.widgets import (
    TemplateRowWidgets,
    CTKContextMenu,
    TemplatePreviewTooltip,
)
from autoclicker.gui.dialogs import (
    SettingsWindow,
    LicenseNoticeWindow,
    RenameTemplateWindow,
    TemplateDelayWindow,
    AddInstanceWindow,
)
from autoclicker.gui.tab_view import InstanceTabFrame
from autoclicker.gui.main_window import App, AutoClickerApp, TeraboxClickerApp

__all__ = [
    "App",
    "AutoClickerApp",
    "TeraboxClickerApp",
    "InstanceTabFrame",
    "SettingsWindow",
    "LicenseNoticeWindow",
    "RenameTemplateWindow",
    "TemplateDelayWindow",
    "AddInstanceWindow",
    "TemplateRowWidgets",
    "CTKContextMenu",
    "TemplatePreviewTooltip",
    "draw_korean_banner",
    "set_opencv_window_title",
    "get_action_button_style",
    "get_delay_button_style",
]
