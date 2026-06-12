"""CLAUDE.md structural-freshness gate for litellm/translation
(06-quality-team.md tenet: deliberate file/folder structure with a
hand-written CLAUDE.md mapping it).

Structural only, no prose judgment:
- litellm/translation/CLAUDE.md must exist
- every real directory under the package (any depth) must be mentioned as
  ``<name>/`` somewhere in CLAUDE.md
- every ``<name>/`` token in CLAUDE.md's tree code block must be a real
  directory (no phantom dirs)

Run: python scripts/check_translation_claude_md.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "litellm" / "translation"
CLAUDE_MD = PACKAGE / "CLAUDE.md"
DIR_TOKEN = re.compile(r"([A-Za-z_][\w.]*)/")


def real_dirs() -> frozenset[str]:
    return frozenset(
        p.name
        for p in PACKAGE.rglob("*")
        if p.is_dir() and p.name != "__pycache__"
    )


def tree_block_lines(text: str) -> tuple[str, ...]:
    blocks = re.findall(r"^```\n(.*?)^```$", text, re.DOTALL | re.MULTILINE)
    if not blocks:
        return ()
    return tuple(line.split("#", 1)[0] for line in blocks[0].splitlines())


def main() -> int:
    if not CLAUDE_MD.is_file():
        print(
            "CLAUDE.md freshness: litellm/translation/CLAUDE.md is missing "
            "(tenets: deliberate structure, hand-written CLAUDE.md at the root)"
        )
        return 1
    text = CLAUDE_MD.read_text(encoding="utf-8")
    dirs = real_dirs()

    unmentioned = sorted(d for d in dirs if f"{d}/" not in text)
    claimed = frozenset(
        m for line in tree_block_lines(text) for m in DIR_TOKEN.findall(line)
    )
    phantoms = sorted(claimed - dirs - {"translation"})

    if not unmentioned and not phantoms:
        print(
            f"CLAUDE.md freshness OK: {len(dirs)} directories all mapped, "
            "no phantom entries"
        )
        return 0
    for d in unmentioned:
        print(
            f"CLAUDE.md freshness: directory '{d}/' exists but is not in "
            "litellm/translation/CLAUDE.md; map it (tenets: deliberate "
            "structure, CLAUDE.md kept true)"
        )
    for d in phantoms:
        print(
            f"CLAUDE.md freshness: CLAUDE.md's tree names '{d}/' but no such "
            "directory exists under litellm/translation; prune it (tenets: "
            "deliberate structure, CLAUDE.md kept true)"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
