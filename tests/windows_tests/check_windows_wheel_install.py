"""Reproduce a default-Windows ``pip install litellm`` to catch the 260-char
MAX_PATH regression that content-filter benchmark fixtures keep reintroducing
(#21941, #22039, #29536). Run after ``uv build --wheel --out-dir dist``.
"""

import glob
import os
import subprocess
import sys
import zipfile

MAX_PATH = 260
# Worst-case Windows site-packages prefix: long profile name + roaming AppData venv.
WORST_CASE_PREFIX = 100


def overlong_install_paths(wheel, prefix_len=WORST_CASE_PREFIX, max_path=MAX_PATH):
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    return sorted(
        (n for n in names if prefix_len + len(n) > max_path), key=len, reverse=True
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


def main():
    wheels = glob.glob(os.path.join("dist", "*.whl"))
    if not wheels:
        print("::error::no wheel in dist/; run `uv build --wheel --out-dir dist` first")
        return 1
    wheel = max(wheels, key=os.path.getmtime)

    offenders = overlong_install_paths(wheel)
    if offenders:
        print(
            f"::error::{len(offenders)} packaged path(s) bust the Windows MAX_PATH limit "
            f"at a {WORST_CASE_PREFIX}-char install prefix:"
        )
        for n in offenders[:15]:
            print(f"  on-disk {WORST_CASE_PREFIX + len(n):4}  {n}")
        return 1

    venv = _deep_venv_dir()
    os.makedirs(os.path.dirname(venv), exist_ok=True)
    if _run(["uv", "venv", venv]) != 0:
        return 1
    python = os.path.join(venv, "Scripts", "python.exe")
    if _run(["uv", "pip", "install", "--python", python, wheel]) != 0:
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
    sys.exit(main())
