import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final

PRIVATE_DIR_MODE: Final = 0o700


def ensure_private_dir(directory: Path) -> None:
    """Create directory (and parents) owner-only, tightening it if it already exists group/world readable"""
    directory.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    if stat.S_IMODE(directory.stat().st_mode) & 0o077:
        directory.chmod(PRIVATE_DIR_MODE)


def stage_private_json(path: str, data: Mapping[str, object]) -> str:
    """Write JSON to a private temp file beside `path`, ready for `commit_staged_json`.

    Staging is the half that can fail on a read-only or full directory, so callers with something
    to lose can find that out before they act on the assumption that the rewrite will land.
    """
    parent: Final = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return tmp_path


def commit_staged_json(staged: str, path: str) -> None:
    """Move a staged file into place, replacing whatever is there in one step"""
    try:
        os.replace(staged, path)
    except OSError:
        Path(staged).unlink(missing_ok=True)
        raise


def discard_staged_json(staged: str) -> None:
    """Throw a staged file away when the change it was part of is abandoned"""
    Path(staged).unlink(missing_ok=True)


def write_private_json(path: str, data: Mapping[str, object]) -> None:
    """Atomically write JSON to path with owner-only permissions (0600)"""
    commit_staged_json(stage_private_json(path, data), path)
