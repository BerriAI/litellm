#!/usr/bin/env python3
"""Pin-list gate for the proxy/utils.py PR3 (bottom helpers).

For each identifier in the pin list, asserts that this directory contains:
  1. At least one test that references the identifier.
  2. At least one happy-path test with a strong assertion (normalize() call,
     Model.model_validate(), or dict-equality with >= 3 keys).
  3. At least one error-path test (name hints at error/invalid/raises/etc.,
     OR uses pytest.raises / a with-raises context manager).

Status-only and "did not raise"-only tests do not satisfy the happy-path
requirement. ``_harness_smoke_test.py`` is ignored.

Exits 0 on PASS, non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent

PIN_LINE_RE = re.compile(r"^- `([^`]+)`\s*$")
ERROR_NAME_HINTS = (
    "_error_path_",
    "_raises_",
    "_raises",
    "_fails_",
    "_fail_",
    "_invalid_",
    "_missing_",
    "_denied_",
    "_rejected_",
    "_unauthorized_",
    "_forbidden_",
    "_too_short_",
    "_too_long_",
    "_too_many_",
    "_too_few_",
    "_not_set_",
    "_empty_",
    "_returns_none",
    "_disabled_",
)


@dataclass
class TestFunction:
    name: str
    file: Path
    source: str
    has_strong_assertion: bool = False
    has_only_did_not_raise: bool = True
    raises_calls: int = 0


def parse_pin_list(path: Path) -> List[str]:
    items: List[str] = []
    for line in path.read_text().splitlines():
        m = PIN_LINE_RE.match(line)
        if m:
            items.append(m.group(1).strip())
    return items


def _has_strong_assertion(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id == "normalize":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "model_validate":
                return True
        if (
            isinstance(sub, ast.Compare)
            and len(sub.ops) == 1
            and isinstance(sub.ops[0], ast.Eq)
        ):
            rhs = sub.comparators[0]
            if isinstance(rhs, ast.Dict) and len(rhs.keys) >= 3:
                return True
    return False


def collect_test_functions(test_dir: Path) -> List[TestFunction]:
    funcs: List[TestFunction] = []
    for path in sorted(test_dir.glob("test_*.py")):
        if path.name == "_harness_smoke_test.py":
            continue
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            tf = TestFunction(name=node.name, file=path, source=source)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assert):
                    tf.has_only_did_not_raise = False
                    if _has_strong_assertion(sub):
                        tf.has_strong_assertion = True
                if isinstance(sub, ast.With):
                    for item in sub.items:
                        ctx = item.context_expr
                        if isinstance(ctx, ast.Call) and isinstance(
                            ctx.func, ast.Attribute
                        ):
                            if ctx.func.attr == "raises":
                                tf.raises_calls += 1
            funcs.append(tf)
    return funcs


def _looks_like_error_test(tf: TestFunction, pin: str = "") -> bool:
    name_lower = tf.name.lower()
    if pin:
        name_lower = name_lower.replace(pin.lower(), "")
    if any(hint in name_lower for hint in ERROR_NAME_HINTS):
        return True
    if tf.raises_calls > 0:
        return True
    return False


def _references_pin(tf: TestFunction, pin: str) -> bool:
    return pin in tf.source


def check(pin_list: List[str], funcs: List[TestFunction]) -> Tuple[bool, List[str]]:
    failures: List[str] = []

    by_pin: Dict[str, List[TestFunction]] = {pin: [] for pin in pin_list}
    for tf in funcs:
        for pin in pin_list:
            if _references_pin(tf, pin):
                by_pin[pin].append(tf)

    for pin, matches in by_pin.items():
        if not matches:
            failures.append(f"no tests reference pin: {pin}")
            continue
        has_happy = any(
            tf.has_strong_assertion
            and not _looks_like_error_test(tf, pin=pin)
            and not tf.has_only_did_not_raise
            for tf in matches
        )
        has_error = any(_looks_like_error_test(tf, pin=pin) for tf in matches)
        if not has_happy:
            failures.append(
                f"no happy-path test with strong assertion "
                f"(normalize/model_validate/dict-eq>=3) for pin: {pin}"
            )
        if not has_error:
            failures.append(f"no error-path test for pin: {pin}")

    return (not failures), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        required=True,
        help="Path to pin list file (markdown bullets in `- ` + backtick form)",
    )
    parser.add_argument(
        "--test-dir",
        default=str(HERE),
        help="Test directory to scan (default: this directory)",
    )
    args = parser.parse_args()

    pin_path = Path(args.list)
    if not pin_path.is_file():
        print(f"FAIL: pin list not found at {pin_path}", file=sys.stderr)
        return 2

    pin_list = parse_pin_list(pin_path)
    if not pin_list:
        print(f"FAIL: pin list at {pin_path} contained zero items", file=sys.stderr)
        return 2

    test_dir = Path(args.test_dir)
    funcs = collect_test_functions(test_dir)
    ok, failures = check(pin_list, funcs)
    if failures:
        for f in failures:
            print(f)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
