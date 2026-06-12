"""No-monster-files gate for litellm/translation (06-quality-team.md tenet:
no monster files or god objects).

Hard line cap per .py file. 720 = 1.3x today's largest file (ir.py, 555
lines, the deliberate home of every IR union) so the current tree passes
with ~30% headroom; everything else is under 360. Raising the cap is a
reviewed decision, not a drive-by edit.

Run: python scripts/check_translation_file_sizes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES = 720
PACKAGE = Path(__file__).resolve().parent.parent / "litellm" / "translation"


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    counts = sorted(
        ((line_count(p), p) for p in PACKAGE.rglob("*.py")), reverse=True
    )
    offenders = [(n, p) for n, p in counts if n > MAX_LINES]
    if not offenders:
        biggest, where = counts[0]
        print(
            f"translation file sizes OK: {len(counts)} files, "
            f"largest {where.relative_to(PACKAGE.parent.parent)} at {biggest} "
            f"lines (cap {MAX_LINES})"
        )
        return 0
    print(
        f"monster file: {len(offenders)} file(s) over the {MAX_LINES}-line cap "
        "(tenets: no monster files or god objects). Split along the package "
        "map in litellm/translation/CLAUDE.md:"
    )
    for n, p in offenders:
        print(f"  {p.relative_to(PACKAGE.parent.parent)}: {n} lines")
    return 1


if __name__ == "__main__":
    sys.exit(main())
