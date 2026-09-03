from __future__ import annotations

import re
from pathlib import Path

_RUST_TEST_PATTERN = re.compile(
    r"#\[(?:test|tokio::test)\][^\n]*\n(?:[^\n]*\n)*?\s*(?:async\s+)?fn\s+(\w+)\s*\("
)


def enumerate_rust_tests(repo_root: Path, relative_path: str) -> frozenset[str]:
    source = (repo_root / relative_path).read_text(encoding="utf-8")
    return frozenset(match.group(1) for match in _RUST_TEST_PATTERN.finditer(source))
