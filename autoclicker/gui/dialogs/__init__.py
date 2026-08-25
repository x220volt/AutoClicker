"""GUI dialogs subpackage."""

from autoclicker.gui.dialogs.settings_dialog import SettingsWindow
from autoclicker.gui.dialogs.license_dialog import LicenseNoticeWindow
from autoclicker.gui.dialogs.rename_dialog import RenameTemplateWindow
from autoclicker.gui.dialogs.delay_dialog import TemplateDelayWindow
from autoclicker.gui.dialogs.instance_dialog import AddInstanceWindow

__all__ = [
    "SettingsWindow",
    "LicenseNoticeWindow",
    "RenameTemplateWindow",
    "TemplateDelayWindow",
    "AddInstanceWindow",
]
