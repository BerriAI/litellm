from __future__ import annotations

import dataclasses
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
FILES_JSON: Final = PurePosixPath(".greptile/files.json")
WRITE_FLAG: Final = "--write"


@dataclasses.dataclass(frozen=True, slots=True)
class _ContextFile:
    path: str
    description: str
    scope: tuple[str, ...]


def _nested_agents_md_paths(repo_root: Path) -> tuple[PurePosixPath, ...]:
    result: Final = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "ls-files", "-z", "--", "*/AGENTS.md"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(sorted(PurePosixPath(os.fsdecode(path)) for path in result.stdout.split(b"\0") if path))


def _context_file(agents_md: PurePosixPath) -> _ContextFile:
    return _ContextFile(
        path=str(agents_md),
        description=f"Conventions for code under {agents_md.parent}/",
        scope=(f"{agents_md.parent}/**",),
    )


def render(repo_root: Path) -> str:
    files: Final = tuple(dataclasses.asdict(_context_file(path)) for path in _nested_agents_md_paths(repo_root))
    return json.dumps({"files": files}, indent=2) + "\n"


def main(argv: tuple[str, ...]) -> int:
    expected: Final = render(REPO_ROOT)
    target: Final = REPO_ROOT / FILES_JSON
    if WRITE_FLAG in argv:
        target.parent.mkdir(exist_ok=True)
        target.write_text(expected)
        print(f"Wrote {FILES_JSON}")
        return 0
    actual: Final = target.read_text() if target.is_file() else ""
    if actual == expected:
        print(f"{FILES_JSON} lists every nested AGENTS.md")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(
            actual.splitlines(keepends=True), expected.splitlines(keepends=True), str(FILES_JSON), "expected"
        )
    )
    print(
        f"\n{FILES_JSON} is out of sync with the nested AGENTS.md files. Regenerate it with:\n"
        f"  python {Path(__file__).resolve().relative_to(REPO_ROOT)} {WRITE_FLAG}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv[1:])))
