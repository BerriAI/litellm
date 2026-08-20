"""Snapshot machinery shared by all characterization tests.

A snapshot is canonical JSON: pretty-printed, sorted keys, trailing newline.
v2's differential gate re-runs the same corpus through v2 and asserts the
emitted JSON is byte-identical to these files.
"""

import json
from pathlib import Path
from typing import Any

PKG_ROOT = Path(__file__).parent
CASES_DIR = PKG_ROOT / "cases"
FIXTURES_DIR = PKG_ROOT / "fixtures"
SNAPSHOTS_DIR = PKG_ROOT / "snapshots"


def jsonable(obj: Any) -> Any:
    """Best-effort conversion of v1 output objects into plain JSON data."""
    if hasattr(obj, "model_dump"):
        return jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def canonical_json(obj: Any) -> str:
    return json.dumps(jsonable(obj), indent=2, sort_keys=True) + "\n"


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def assert_snapshot(rel_path: str, obj: Any, update: bool) -> None:
    """Compare ``obj`` against ``snapshots/<rel_path>``; rewrite when updating."""
    path = SNAPSHOTS_DIR / rel_path
    rendered = canonical_json(obj)
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        return
    assert path.exists(), (
        f"Missing snapshot {path}. Run with --snapshot-update (or SNAPSHOT_UPDATE=1) "
        "to record it, then commit the file."
    )
    assert rendered == path.read_text(), (
        f"Characterization drift for {rel_path}. If this change is intentional, "
        "regenerate with --snapshot-update and ship the snapshot diff as its own PR."
    )
