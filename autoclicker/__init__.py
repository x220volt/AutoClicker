"""AutoClicker - Modular Android Automation & Image Matching Clicker."""

from autoclicker.core.clicker import AutoClicker, TeraboxClicker
from autoclicker.core.environment import (
    get_app_dir,
    get_default_adb_path,
    resolve_adb_path,
    APP_DIR,
    CONFIG_PATH,
    ADB_PATH,
)

__version__ = "0.4.0"

__all__ = [
    "AutoClicker",
    "TeraboxClicker",
    "get_app_dir",
    "get_default_adb_path",
    "resolve_adb_path",
    "APP_DIR",
    "CONFIG_PATH",
    "ADB_PATH",
]
