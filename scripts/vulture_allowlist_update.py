"""Regenerate the vulture allowlist baseline (.vulture_allowlist.py).

Mirrors the ruff/basedpyright budget-update ratchet: re-capture the set of
currently accepted dead-code findings so `make lint-deadcode` only surfaces new
ones. The hand-written header is preserved; everything below it is regenerated
from `vulture --make-whitelist`.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = REPO_ROOT / ".vulture_allowlist.py"
MARKER = "# === generated below; do not edit by hand ==="


def _read_header() -> str:
    text = ALLOWLIST.read_text()
    end = text.find(MARKER)
    if end == -1:
        raise SystemExit(f"Could not find marker {MARKER!r} in {ALLOWLIST}")
    return text[: end + len(MARKER)] + "\n\n"


def _vulture_config() -> tuple[tuple[str, ...], tuple[str, ...]]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"][
        "vulture"
    ]
    paths = tuple(p for p in config["paths"] if p != ALLOWLIST.name)
    exclude = tuple(config.get("exclude", ()))
    return paths, exclude


def main() -> int:
    header = _read_header()
    paths, exclude = _vulture_config()
    cmd = [
        "vulture",
        *paths,
        "--min-confidence",
        "100",
        "--make-whitelist",
    ]
    if exclude:
        cmd += ["--exclude", ",".join(exclude)]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 3):
        sys.stderr.write(result.stderr)
        return result.returncode
    ALLOWLIST.write_text(header + result.stdout)
    print(f"Updated {ALLOWLIST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
