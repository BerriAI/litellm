#!/usr/bin/env python3
"""Rewrite ``[tool.mutmut].tests_dir`` in pyproject.toml to a single suite.

Mutation testing of ``litellm/proxy/management_endpoints/`` is exercised by
two test suites that cannot share one pytest session: the legacy mock suite
(``tests/test_litellm/proxy/management_endpoints/``) patches ``prisma_client``
globally, while the behavior suite (``tests/proxy_behavior/management/``)
drives a real proxy against a live Postgres database.

mutmut reads ``tests_dir`` only from pyproject.toml and offers no CLI/env
override, so .github/workflows/mutation-test.yml runs mutmut once per suite
(a 2-leg matrix) and calls this script before each ``mutmut run`` to scope
``tests_dir`` to that leg's suite.

Usage: scope_mutmut_tests_dir.py <tests-dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Matches the whole `tests_dir = [ ... ]` array in [tool.mutmut] — including a
# multi-line body — up to the first `]` that begins a line.
TESTS_DIR_BLOCK = re.compile(r"(?ms)^tests_dir = \[.*?^\]")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    suite = argv[1].strip()
    if not suite:
        print("error: <tests-dir> argument is empty", file=sys.stderr)
        return 2

    text = PYPROJECT.read_text()
    new_text, count = TESTS_DIR_BLOCK.subn(
        f'tests_dir = [\n    "{suite}",\n]', text, count=1
    )
    if count != 1:
        print(
            f"error: could not find [tool.mutmut] tests_dir block in {PYPROJECT}",
            file=sys.stderr,
        )
        return 1

    PYPROJECT.write_text(new_text)
    print(f"mutmut tests_dir scoped to: {suite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
