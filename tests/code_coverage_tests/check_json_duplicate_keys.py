"""
Fail if a hand-maintained JSON config file declares the same key twice inside one object.

Why this needs its own check:

    json.load() keeps the LAST of a repeated key and reports no error. So when two
    copies of the same object exist and they are not identical, the earlier copy is
    dropped without a trace - the file parses, CI passes, and whatever fields only
    the earlier copy had are simply not there. Nobody is told.

    Every existing check reads the PARSED file, and the duplicate is already gone by
    then. This check re-reads the raw text with object_pairs_hook, which is the only
    place the evidence still exists.

Add a file to FILES_TO_CHECK when it is edited by hand and read by code at runtime.
Generated files and lockfiles do not belong here.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Repo-root-relative paths of hand-maintained JSON that code reads at runtime.
FILES_TO_CHECK = [
    "provider_endpoints_support.json",
]


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.parent


def find_duplicate_keys(file_path: Path) -> List[Tuple[str, int]]:
    """Return [(key, occurrences), ...] for keys repeated inside the same object."""
    duplicates: List[Tuple[str, int]] = []

    def collect(pairs):
        counts: Dict[str, int] = {}
        for key, _ in pairs:
            counts[key] = counts.get(key, 0) + 1
        duplicates.extend((key, n) for key, n in counts.items() if n > 1)
        return dict(pairs)

    with open(file_path, "r") as f:
        json.loads(f.read(), object_pairs_hook=collect)

    return duplicates


def main() -> None:
    repo_root = get_repo_root()
    failures: List[str] = []
    checked = 0

    print("🔑 Checking hand-maintained JSON config for duplicate keys...\n")

    for rel_path in FILES_TO_CHECK:
        file_path = repo_root / rel_path

        if not file_path.exists():
            # A path that no longer exists means this check is silently protecting
            # nothing, so say so instead of passing.
            failures.append(f"{rel_path}: file not found at {file_path}")
            continue

        checked += 1
        duplicates = find_duplicate_keys(file_path)

        if duplicates:
            for key, occurrences in duplicates:
                failures.append(
                    f"{rel_path}: '{key}' is declared {occurrences} times inside the "
                    "same object; only the last one survives json.load()"
                )
        else:
            print(f"  ✅ {rel_path}")

    if not checked:
        # Scanning zero files must never look like success.
        print("\n❌ No files were checked - FILES_TO_CHECK is empty or all paths are stale")
        sys.exit(1)

    if failures:
        print("\n❌ Duplicate keys found:\n")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nMerge the duplicate objects into one, keeping every field that appears "
            "in either copy."
        )
        sys.exit(1)

    print(f"\n✅ {checked} file(s) checked, no duplicate keys found")


if __name__ == "__main__":
    main()
