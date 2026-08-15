import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final


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
