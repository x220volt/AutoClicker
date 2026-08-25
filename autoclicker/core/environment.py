"""Environment and executable path resolution helpers for AutoClicker."""

import os
import shutil
import sys
from autoclicker.core.constants import (
    DEFAULT_ADB_MODE,
    CONFIG_FILENAME,
)


def get_app_dir():
    """Return the writable directory next to the app/script, independent of cwd."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_default_adb_path(base_dir=None):
    """Return the bundled ADB executable when available, extracted to a persistent path in onefile mode."""
    if base_dir:
        custom_base_adb = os.path.join(base_dir, "ADB", "adb.exe")
        if os.path.exists(custom_base_adb):
            return custom_base_adb

    app_dir_adb = os.path.join(get_app_dir(), "ADB", "adb.exe")
    if os.path.exists(app_dir_adb):
        return app_dir_adb

    if hasattr(sys, "_MEIPASS"):
        meipass_adb_dir = os.path.join(sys._MEIPASS, "ADB")
        meipass_adb_exe = os.path.join(meipass_adb_dir, "adb.exe")
        if os.path.exists(meipass_adb_exe):
            persistent_adb_dir = os.path.join(
                os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
                "AutoClicker",
                "ADB",
            )
            persistent_adb_exe = os.path.join(persistent_adb_dir, "adb.exe")
            try:
                os.makedirs(persistent_adb_dir, exist_ok=True)
                for item in os.listdir(meipass_adb_dir):
                    src_file = os.path.join(meipass_adb_dir, item)
                    dst_file = os.path.join(persistent_adb_dir, item)
                    if os.path.isfile(src_file):
                        if (
                            not os.path.exists(dst_file)
                            or os.path.getsize(src_file) != os.path.getsize(dst_file)
                        ):
                            try:
                                shutil.copy2(src_file, dst_file)
                            except Exception:
                                pass
                if os.path.exists(persistent_adb_exe):
                    return persistent_adb_exe
            except Exception:
                pass
            return meipass_adb_exe

    return "adb.exe"


def resolve_adb_path(adb_mode=DEFAULT_ADB_MODE, custom_adb_path="", base_dir=None):
    """Return the executable path based on ADB mode ('bundled' vs 'custom')."""
    if adb_mode == "custom" and custom_adb_path and str(custom_adb_path).strip():
        return os.path.expandvars(os.path.expanduser(str(custom_adb_path).strip()))
    return get_default_adb_path(base_dir)


APP_DIR = get_app_dir()
CONFIG_PATH = os.path.join(APP_DIR, CONFIG_FILENAME)
ADB_PATH = get_default_adb_path(APP_DIR)
