# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, get_package_paths

block_cipher = None

# Get llama_cpp package path
_, llama_cpp_pkg_path = get_package_paths('llama_cpp')

# Collect all llama_cpp files
llama_datas, llama_binaries, llama_hiddenimports = collect_all('llama_cpp')

# Collect art library data
art_datas, art_binaries, art_hiddenimports = collect_all('art')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=llama_binaries + art_binaries,
    datas=llama_datas + art_datas,
    hiddenimports=[
        'llama_cpp',
        'art',
        'pyperclip',
        'tqdm',
        'requests',
        'urllib3',
        'certifi',
        'charset_normalizer',
        'idna',
    ] + llama_hiddenimports + art_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='NexusAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
        icon=r'C:\Users\0x3EF8\Desktop\nexus\Nexus.ico',
)
