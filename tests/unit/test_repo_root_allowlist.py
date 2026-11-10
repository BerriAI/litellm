import json
import subprocess
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
ALLOWLIST: Final = ROOT / ".github/root-entries-allowlist.json"


def test_tracked_root_entries_have_reviewed_reasons() -> None:
    allowed: Final[dict[str, str]] = json.loads(ALLOWLIST.read_text())
    tracked: Final = frozenset(
        path.split("/", 1)[0]
        for path in subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split("\0")
        if path
    )
    unlisted: Final = tuple(sorted(tracked - allowed.keys()))
    stale: Final = tuple(sorted(allowed.keys() - tracked))
    invalid_reasons: Final = tuple(
        sorted(
            entry
            for entry, reason in allowed.items()
            if not isinstance(reason, str) or not reason.strip() or len(reason.splitlines()) != 1
        )
    )
    assert not (unlisted or stale or invalid_reasons), (
        f"Unlisted tracked root entries: {unlisted}\n"
        f"Stale allowlist entries: {stale}\n"
        f"Entries without a one-line reason: {invalid_reasons}\n"
        "Edit .github/root-entries-allowlist.json to add each allowed root entry with a one-line reason, "
        "or remove stale entries. Code owners must review allowlist changes."
    )
