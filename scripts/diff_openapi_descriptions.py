"""Print GitHub warning annotations for OpenAPI description changes.

oasdiff has no description-level checks, but a description change can hide a
behavioral contract change (e.g. a filter param switching from exact to
substring matching), so CI waves a non-blocking flag at reviewers.
"""

import json
import sys
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Final

MAX_ANNOTATIONS: Final = 30


def _clip(text: str) -> str:
    flat: Final = " ".join(text.split())
    return flat if len(flat) <= 100 else flat[:97] + "..."


def _diffs(base: object, head: object, path: str) -> Iterator[str]:
    if isinstance(base, dict) and isinstance(head, dict):
        b_desc = base.get("description")
        h_desc = head.get("description")
        if isinstance(b_desc, str) and isinstance(h_desc, str) and b_desc != h_desc:
            yield f"{path}: {_clip(b_desc)!r} -> {_clip(h_desc)!r}"
        elif isinstance(b_desc, str) != isinstance(h_desc, str):
            yield f"{path}: description {'removed' if isinstance(b_desc, str) else 'added'}"
        for key in sorted(set(base) | set(head)):
            if key != "description":
                yield from _diffs(base.get(key), head.get(key), f"{path}.{key}" if path else key)
    elif isinstance(base, list) and isinstance(head, list):
        for i, (b, h) in enumerate(zip_longest(base, head)):
            yield from _diffs(b, h, f"{path}[{i}]")


def main() -> None:
    base: Final = json.loads(Path(sys.argv[1]).read_text())
    head: Final = json.loads(Path(sys.argv[2]).read_text())
    findings: Final = tuple(_diffs(base, head, ""))
    for line in findings[:MAX_ANNOTATIONS]:
        print(f"::warning title=openapi-description-changed::{line}")
    if len(findings) > MAX_ANNOTATIONS:
        print(f"::warning title=openapi-description-changed::...and {len(findings) - MAX_ANNOTATIONS} more")
    print(f"{len(findings)} description change(s); review them for hidden behavior changes.")


if __name__ == "__main__":
    main()
