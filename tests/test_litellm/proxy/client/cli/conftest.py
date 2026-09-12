import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.client.cli.commands import claude_settings

REAL_CLAUDE_SETTINGS: Final = Path(os.path.expanduser("~")) / ".claude" / "settings.json"


def _current_bytes() -> bytes | None:
    return REAL_CLAUDE_SETTINGS.read_bytes() if REAL_CLAUDE_SETTINGS.exists() else None


@pytest.fixture(autouse=True)
def _statusline_script_under_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(claude_settings, "STATUSLINE_SCRIPT_PATH", tmp_path / "litellm-home" / "statusline.py")


@pytest.fixture(autouse=True)
def isolated_claude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    before: Final = _current_bytes()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    yield tmp_path
    after: Final = _current_bytes()
    if after == before:
        return
    if before is None:
        REAL_CLAUDE_SETTINGS.unlink()
    else:
        REAL_CLAUDE_SETTINGS.write_bytes(before)
    pytest.fail(
        f"this test wrote the developer's real {REAL_CLAUDE_SETTINGS}; the original bytes were restored. "
        "Resolve the Claude settings path at call time (never Path.home() at import) and point the test at tmp_path"
    )
