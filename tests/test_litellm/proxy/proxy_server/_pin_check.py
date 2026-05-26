#!/usr/bin/env python3
"""Pin-list gate for the proxy_server.py behavior-pinning project.

For each identifier in a pin list, asserts that the test directory contains:
  1. At least one happy-path test that references the identifier and uses
     a real assertion (normalize(response.json()) == {...}, .model_validate,
     or a dict-equality with >= 3 keys).
  2. At least one error-path test (name hints at error OR asserts a 4xx/5xx
     status OR uses pytest.raises).
  3. No test that is "status-only" (its sole assert is on response.status_code).

``test_harness_smoke.py`` is ignored (harness self-tests don't count toward
behavior pinning).

Exits 0 on PASS, non-zero on FAIL.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent

PIN_LINE_RE = re.compile(r"^- `([^`]+)`\s*$")
ERROR_NAME_HINTS = (
    "error",
    "fail",
    "invalid",
    "unauthorized",
    "forbidden",
    "missing",
    "denied",
    "rejected",
    "bad",
    "raises",
    "exception",
    "404",
    "401",
    "403",
    "422",
    "500",
)
ERROR_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 405, 409, 422, 500, 502, 503})


@dataclass
class TestFunction:
    name: str
    file: Path
    source: str
    asserts: List[ast.Assert] = field(default_factory=list)
    raises_calls: int = 0
    status_code_asserts: List[int] = field(default_factory=list)
    has_strong_assertion: bool = (
        False  # normalize() or .model_validate() or large dict-eq
    )


def parse_pin_list(path: Path) -> List[str]:
    items: List[str] = []
    for line in path.read_text().splitlines():
        m = PIN_LINE_RE.match(line)
        if m:
            items.append(m.group(1).strip())
    return items


def _has_strong_assertion(node: ast.AST) -> bool:
    """True if an assert subtree contains normalize(), .model_validate(), or dict-eq with >=3 keys."""
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
            # response.json() == {<dict literal with >= 3 keys>}
            rhs = sub.comparators[0]
            if isinstance(rhs, ast.Dict) and len(rhs.keys) >= 3:
                return True
    return False


def _extract_status_code(node: ast.Assert) -> Optional[int]:
    """If this assert is exactly ``X.status_code == <int>``, return the int."""
    test = node.test
    if not isinstance(test, ast.Compare):
        return None
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return None
    left = test.left
    if not (isinstance(left, ast.Attribute) and left.attr == "status_code"):
        return None
    right = test.comparators[0]
    if isinstance(right, ast.Constant) and isinstance(right.value, int):
        return right.value
    return None


def collect_test_functions(test_dir: Path) -> List[TestFunction]:
    funcs: List[TestFunction] = []
    for path in sorted(test_dir.glob("test_*.py")):
        # Skip the harness's own smoke tests — they don't count toward
        # behavior pinning.
        if path.name == "test_harness_smoke.py":
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
                    tf.asserts.append(sub)
                    sc = _extract_status_code(sub)
                    if sc is not None:
                        tf.status_code_asserts.append(sc)
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


def _is_status_only(tf: TestFunction) -> bool:
    """A test that has >=1 status_code assert and ALL its asserts are status_code."""
    return len(tf.asserts) >= 1 and len(tf.status_code_asserts) == len(tf.asserts)


def _looks_like_error_test(tf: TestFunction) -> bool:
    name_lower = tf.name.lower()
    if any(hint in name_lower for hint in ERROR_NAME_HINTS):
        return True
    if tf.raises_calls > 0:
        return True
    if any(sc in ERROR_STATUS_CODES for sc in tf.status_code_asserts):
        return True
    return False


def _references_pin(tf: TestFunction, pin: str) -> bool:
    """Cheap string-contains check against the test function's source.

    This is intentionally permissive — if the pin identifier (e.g.
    ``update_cache`` or ``POST /chat/completions``) appears anywhere in
    the test file we count it. Aliased route paths or parametrize
    cases trigger the same reference.
    """
    return pin in tf.source


def check(pin_list: List[str], funcs: List[TestFunction]) -> Tuple[bool, List[str]]:
    failures: List[str] = []

    status_only = [tf for tf in funcs if _is_status_only(tf)]
    for tf in status_only:
        failures.append(
            f"status-only test (only asserts response.status_code): "
            f"{tf.file.name}::{tf.name}"
        )

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
            tf.has_strong_assertion and not _looks_like_error_test(tf) for tf in matches
        )
        has_error = any(_looks_like_error_test(tf) for tf in matches)
        if not has_happy:
            failures.append(
                f"no happy-path test with strong assertion (normalize/model_validate/dict-eq>=3) "
                f"for pin: {pin}"
            )
        if not has_error:
            failures.append(f"no error-path test for pin: {pin}")

    return (not failures), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        required=True,
        help="Path to pin list file (markdown bullets in `- ` + backtick + symbol + backtick format)",
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
    print(f"pins:  {len(pin_list)}")
    print(f"tests: {len(funcs)}")
    if failures:
        for f in failures:
            print(f"  - {f}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
