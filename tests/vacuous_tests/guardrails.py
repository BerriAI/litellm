"""Guardrails on what the vacuous-test audit is allowed to change.

The failure mode of an automated "make the vacuous-test count go down" loop is
gaming the metric: deleting tests, weakening assertions, or editing production
code until the suite agrees. This runs against a diff and rejects it unless the
change is confined to test bodies.

Rules:
  1. Only files under tests/ may change, and never conftest.py, CI config, or
     the audit tooling itself.
  2. The number of test functions may not drop, unless --allow-removals is
     passed with a citations file naming the test that now covers each removal.
  3. Assertions may not be net removed from a test file.

Usage:
    python tests/vacuous_tests/guardrails.py --base origin/litellm_internal_staging
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from typing import Dict, FrozenSet, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORBIDDEN_NAMES = ("conftest.py", "pytest.ini", "pyproject.toml", "ruff.toml", "Makefile")
FORBIDDEN_PREFIXES = (".github/",)
# The audit's own logic is off limits; its JSON ledgers are exactly what the
# daily run has to update.
TOOLING_DIR = "tests/vacuous_tests/"


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def changed_files(base: str) -> List[str]:
    output = _git("diff", "--name-only", f"{base}...HEAD")
    return [line for line in output.splitlines() if line.strip()]


def _test_names(source: str) -> FrozenSet[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    return frozenset(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    )


def _assert_count(source: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))


def _blob(ref: str, path: str) -> str:
    try:
        return _git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return ""


def check(base: str, allow_removals: Optional[str]) -> List[str]:
    violations: List[str] = []
    files = changed_files(base)
    if not files:
        return ["diff is empty"]

    for path in files:
        if not path.startswith("tests/"):
            violations.append(f"{path}: outside tests/; this automation may only change tests")
        if os.path.basename(path) in FORBIDDEN_NAMES or path.startswith(FORBIDDEN_PREFIXES):
            violations.append(f"{path}: shared test or CI configuration is off limits")
        if path.startswith(TOOLING_DIR) and path.endswith(".py"):
            violations.append(f"{path}: the audit may not edit its own detection logic")

    citations: Dict[str, str] = {}
    if allow_removals:
        with open(allow_removals, "r", encoding="utf-8") as handle:
            citations = json.load(handle)

    for path in files:
        if not path.endswith(".py"):
            continue
        before_source = _blob(base, path)
        after_source = _blob("HEAD", path)
        before_names = _test_names(before_source)
        after_names = _test_names(after_source)
        violations.extend(uncited_removals(path, before_names - after_names, citations))
        before_asserts = _assert_count(before_source)
        after_asserts = _assert_count(after_source)
        if after_asserts < before_asserts and len(after_names) >= len(before_names):
            violations.append(
                f"{path}: assertion count dropped from {before_asserts} to {after_asserts} "
                "without removing tests, which looks like weakening"
            )
    return violations


def uncited_removals(path: str, removed: FrozenSet[str], citations: Dict[str, str]) -> List[str]:
    """Every removed test needs its own citation naming the test that now covers it."""
    return [
        f'{path}::{name} was removed without a citation; add "{path}::{name}": "<test id that now covers this>"'
        for name in sorted(removed)
        if not citations.get(f"{path}::{name}", "").strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base ref to diff against")
    parser.add_argument("--allow-removals", metavar="PATH", help="JSON map of removed test id -> covering test id")
    args = parser.parse_args()

    violations = check(args.base, args.allow_removals)
    if violations:
        print("vacuous-test guardrails FAILED")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("vacuous-test guardrails OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
