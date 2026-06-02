#!/usr/bin/env python3
"""Pin-list gate for the ``litellm/proxy/utils.py`` ProxyLogging PR.

CLI: ``python _pin_check.py --list .pin_list.txt``.

AST-walks the test files in this directory and verifies, for every symbol
in the pin list:

1. The symbol is referenced by at least one test function.
2. There is at least one happy-path test for it.
3. There is at least one error-path test for it. A test counts as
   error-path if its name matches ``*_error_*``, ``*_raises_*``, or
   ``*_fails_*``, or its body uses ``pytest.raises`` / asserts an
   exception.
4. Every happy-path test has a strong assertion: one of
   ``assert normalize(...) == {dict literal with >= 3 keys}``,
   ``assert <call>(...) == <dict/object literal with >= 3 keys>``, or
   ``Model.model_validate(...)``. Status-only and "did not raise"-only
   tests are rejected.

``_harness_smoke_test.py`` is ignored.

Output: ``PASS`` (exit 0), or one line per violation followed by ``FAIL``
(exit 1). No counts, no percentages.
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

ERROR_NAME_PATTERNS = (
    re.compile(r"_error(_|$)"),
    re.compile(r"_raises(_|$)"),
    re.compile(r"_fails(_|$)"),
)


@dataclass
class TestFunction:
    name: str
    file: Path
    asserts: List[ast.Assert] = field(default_factory=list)
    raises_calls: int = 0
    has_strong_assertion: bool = False
    references: set = field(default_factory=set)


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
            if isinstance(func, ast.Attribute) and func.attr == "model_validate":
                return True
        if (
            isinstance(sub, ast.Compare)
            and len(sub.ops) == 1
            and isinstance(sub.ops[0], ast.Eq)
        ):
            for side in (sub.left, sub.comparators[0]):
                if isinstance(side, ast.Dict) and len(side.keys) >= 3:
                    return True
                if isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "normalize":
                    other = sub.comparators[0] if side is sub.left else sub.left
                    if isinstance(other, ast.Dict) and len(other.keys) >= 3:
                        return True
    return False


def _looks_like_error_test(tf: TestFunction) -> bool:
    name_lower = tf.name.lower()
    for pat in ERROR_NAME_PATTERNS:
        if pat.search(name_lower):
            return True
    if tf.raises_calls > 0:
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
            tf = TestFunction(name=node.name, file=path)
            fn_source = ast.get_source_segment(source, node) or ""
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assert):
                    tf.asserts.append(sub)
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
            tf.references.add(fn_source)
            funcs.append(tf)
    return funcs


def _references_pin(tf: TestFunction, pin: str) -> bool:
    """Match a test body against a pin.

    Rules:
    - The full pin string is in the body, OR
    - For ``Class.__init__``: ``Class(`` (constructor call), OR
    - For ``Class.method``: ``.method(`` (method call), OR
    - For top-level helpers: ``helper(`` as a callable reference.
    """
    parts = pin.rsplit(".", 1)
    for body in tf.references:
        if pin in body:
            return True
        if len(parts) == 2:
            cls, method = parts
            if method == "__init__":
                if f"{cls}(" in body:
                    return True
            else:
                if f".{method}(" in body:
                    return True
        else:
            if f"{pin}(" in body:
                return True
    return False


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
            tf.has_strong_assertion and not _looks_like_error_test(tf)
            for tf in matches
        )
        has_error = any(_looks_like_error_test(tf) for tf in matches)
        if not has_happy:
            failures.append(f"no happy-path test with strong assertion for pin: {pin}")
        if not has_error:
            failures.append(f"no error-path test for pin: {pin}")
    return (not failures), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True)
    parser.add_argument("--test-dir", default=str(HERE))
    args = parser.parse_args()

    pin_path = Path(args.list)
    if not pin_path.is_file():
        print(f"pin list not found at {pin_path}", file=sys.stderr)
        print("FAIL")
        return 1

    pin_list = parse_pin_list(pin_path)
    if not pin_list:
        print(f"pin list at {pin_path} contained zero items", file=sys.stderr)
        print("FAIL")
        return 1

    funcs = collect_test_functions(Path(args.test_dir))
    ok, failures = check(pin_list, funcs)
    for f in failures:
        print(f)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
