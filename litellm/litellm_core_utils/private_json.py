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


def write_private_json(path: str, data: Mapping[str, object]) -> None:
    """Atomically write JSON to path with owner-only permissions (0600)"""
    parent: Final = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
