"""Reproduce a default-Windows ``pip install litellm`` to catch the 260-char
MAX_PATH regression that content-filter fixtures keep reintroducing
(#21941, #22039, #29536, #43851). Run after ``uv build --wheel --out-dir dist``.

pip writes every wheel entry verbatim under ``site-packages``, so an entry
busts the limit when ``site-packages`` prefix + entry reaches MAX_PATH (260,
which counts the terminating NUL, so 259 visible characters), and its parent
directory busts ``CreateDirectoryW`` at 248. Microsoft Store Python has the
deepest common ``site-packages``: 134 characters plus the profile folder name
(learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
and the Store install layout, checked 2026-09-30).

The install must go through pip, not uv: uv writes files from Rust, which
switches to extended-length paths on its own and never hits MAX_PATH.
"""

import glob
import os
import subprocess
import sys
import zipfile

MAX_PATH = 260
MAX_DIRECTORY_PATH = 248
STORE_PYTHON_SITE_PACKAGES = (
    "C:\\Users\\{profile}\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0"
    "\\LocalCache\\local-packages\\Python312\\site-packages\\"
)
WORST_CASE_PREFIX = len(STORE_PYTHON_SITE_PACKAGES.format(profile="x" * 15))


def busts_windows_limits(entry, prefix_len=WORST_CASE_PREFIX):
    return (
        prefix_len + len(entry) >= MAX_PATH
        or prefix_len + len(os.path.dirname(entry)) >= MAX_DIRECTORY_PATH
    )


def overlong_install_paths(wheel, prefix_len=WORST_CASE_PREFIX):
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    return sorted(
        (n for n in names if busts_windows_limits(n, prefix_len)), key=len, reverse=True
    )


def _deep_venv_dir(target_prefix=WORST_CASE_PREFIX):
    drive = os.path.splitdrive(os.getcwd())[0] or "C:"
    root = drive + os.sep + "lmwin" + os.sep
    # +2: the sep joining the venv root to "Lib", plus the trailing sep before the entry
    suffix = len(os.path.join("Lib", "site-packages")) + 2
    return root + "x" * (target_prefix - suffix - len(root))


def _run(cmd):
    print("+ " + subprocess.list2cmdline(cmd), flush=True)
    return subprocess.call(cmd)


def main(argv):
    wheels = glob.glob(os.path.join("dist", "*.whl"))
    if not wheels:
        print("::error::no wheel in dist/; run `uv build --wheel --out-dir dist` first")
        return 1
    wheel = max(wheels, key=os.path.getmtime)

    offenders = overlong_install_paths(wheel)
    if offenders:
        print(
            f"::error::{len(offenders)} packaged path(s) bust the Windows MAX_PATH limit "
            f"at a {WORST_CASE_PREFIX}-char install prefix (Store Python, 15-char profile name):"
        )
        for n in offenders[:15]:
            print(f"  on-disk {WORST_CASE_PREFIX + len(n):4}  {n}")
        return 1
    if "--lengths-only" in argv:
        print(f"ok: every path in {os.path.basename(wheel)} fits MAX_PATH at a {WORST_CASE_PREFIX}-char prefix")
        return 0

    venv = _deep_venv_dir()
    os.makedirs(os.path.dirname(venv), exist_ok=True)
    if _run([sys.executable, "-m", "venv", venv]) != 0:
        return 1
    python = os.path.join(venv, "Scripts", "python.exe")
    if _run([python, "-m", "pip", "install", wheel]) != 0:
        print(
            f"::error::installing {os.path.basename(wheel)} into a deep prefix failed"
        )
        return 1
    if _run([python, "-c", "import litellm; import litellm.types.utils"]) != 0:
        print("::error::litellm did not import after install (half-unpacked package)")
        return 1

    print(
        f"ok: {os.path.basename(wheel)} installs into a worst-case prefix and imports"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
