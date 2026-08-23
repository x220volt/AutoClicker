# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import zipfile
from PyInstaller.utils.hooks import collect_data_files

customtkinter_datas = collect_data_files("customtkinter")

# Base extra datas
extra_datas = [
    ("ADB", "ADB"),
    ("LICENSE-adb.txt", "."),
    ("LICENSE", "."),
]

# Tcl/Tk support for Python 3.14+ (where Tcl 9 zipfs is used) and standard Python versions
base_prefix = getattr(sys, "base_prefix", sys.prefix)
tcl_base_dir = os.path.join(base_prefix, "tcl")

if os.path.exists(tcl_base_dir):
    tcl_zip = None
    tk_zip = None
    for item in os.listdir(tcl_base_dir):
        if item.startswith("libtcl") and item.endswith(".zip"):
            tcl_zip = os.path.join(tcl_base_dir, item)
        elif item.startswith("libtk") and item.endswith(".zip"):
            tk_zip = os.path.join(tcl_base_dir, item)

    if tcl_zip and tk_zip:
        cache_dir = os.path.abspath(os.path.join("build", "_tcl_tk_cache"))
        tcl_extract = os.path.join(cache_dir, "_tcl_data")
        tk_extract = os.path.join(cache_dir, "_tk_data")

        os.makedirs(tcl_extract, exist_ok=True)
        with zipfile.ZipFile(tcl_zip) as z:
            for member in z.infolist():
                name = member.filename
                if name.startswith("tcl_library/"):
                    rel = name[len("tcl_library/"):]
                    if not rel:
                        continue
                    dest = os.path.join(tcl_extract, rel)
                    if member.is_dir():
                        os.makedirs(dest, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with open(dest, "wb") as f:
                            f.write(z.read(name))

        os.makedirs(tk_extract, exist_ok=True)
        with zipfile.ZipFile(tk_zip) as z:
            for member in z.infolist():
                name = member.filename
                if name.startswith("tk_library/"):
                    rel = name[len("tk_library/"):]
                    if not rel:
                        continue
                    dest = os.path.join(tk_extract, rel)
                    if member.is_dir():
                        os.makedirs(dest, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with open(dest, "wb") as f:
                            f.write(z.read(name))

        # Add extracted directories as 2-tuples (src, dst)
        extra_datas.append((tcl_extract, "_tcl_data"))
        extra_datas.append((tk_extract, "_tk_data"))

        # Add additional folders and zip files from tcl_base_dir
        for item in os.listdir(tcl_base_dir):
            item_path = os.path.join(tcl_base_dir, item)
            if os.path.isdir(item_path):
                extra_datas.append((item_path, os.path.join("_tcl_data", item)))
            elif item.endswith(".zip"):
                extra_datas.append((item_path, "."))
                extra_datas.append((item_path, "tcl"))
    else:
        extra_datas.append((tcl_base_dir, "_tcl_data"))

a = Analysis(
    ["gui_app.py"],
    pathex=[],
    binaries=[],
    datas=customtkinter_datas + extra_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="TeraboxClicker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
