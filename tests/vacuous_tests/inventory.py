"""Stage A of the vacuous-test audit: a static inventory of *candidate* vacuous tests.

A vacuous test is one that cannot fail when the code it claims to test is
broken. That property is not statically decidable, so this script only
produces candidates; `mutation_probe.py` (Stage B) is what actually decides,
by mutating the lines a single test covers and checking whether the test
notices.

Two jobs:

1. Ratchet (CI). `--check` compares the candidates found now against the ones
   named in `inventory_baseline.json` and fails on any candidate the baseline
   does not already name, so a new vacuous test cannot land by taking the slot
   of one that was fixed. Regenerate with `--update-baseline` after a cleanup
   or a rename.
2. Queue (automation). `--queue N` prints the next N candidates for the daily
   run, skipping anything Stage B has already cleared in
   `verified_not_vacuous.json`. `--todays-area` keeps a run inside one area,
   rotating by date so each PR stays reviewable by one owner and no state file
   is needed.

Usage:
    python tests/vacuous_tests/inventory.py --report
    python tests/vacuous_tests/inventory.py --check
    python tests/vacuous_tests/inventory.py --update-baseline
    python tests/vacuous_tests/inventory.py --areas
    python tests/vacuous_tests/inventory.py --queue 15 --todays-area
    python tests/vacuous_tests/inventory.py --queue 15 --area tests/litellm_utils_tests
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL_DIR = os.path.join(REPO_ROOT, "tests", "vacuous_tests")
BASELINE_PATH = os.path.join(TOOL_DIR, "inventory_baseline.json")
CLEARED_PATH = os.path.join(TOOL_DIR, "verified_not_vacuous.json")
TESTS_ROOT = os.path.join(REPO_ROOT, "tests")

# Buckets, most to least specific. A test is reported under exactly one bucket
# (the first that matches) so counts stay stable when a test has several
# problems.
BUCKETS = (
    "dead_skip",
    "swallowed_failure",
    "trivial_assert",
    "mock_tautology",
    "no_assert",
)

# Directories whose tests are out of scope: they talk to live providers, take
# minutes, or measure performance rather than behaviour, so "does a mutant kill
# it" is either unaffordable or meaningless.
EXCLUDED_DIRS = (
    "tests/load_tests",
    "tests/benchmarks",
    "tests/e2e",
    "tests/multi_instance_e2e_tests",
    "tests/old_proxy_tests",
    "tests/pass_through_tests",
    "tests/vacuous_tests",
)

MOCK_FACTORIES = {"Mock", "MagicMock", "AsyncMock", "NonCallableMock", "patch"}
ASSERT_CALL_PREFIXES = ("assert_", "assert", "check_", "verify_", "expect_")
PYTEST_ASSERT_FUNCS = {"raises", "fail", "approx", "warns", "deprecated_call"}
# Handler bodies made up only of these are swallowing the failure.
SWALLOWING_CALLS = {"skip", "xfail", "print", "warn", "debug", "info", "warning"}
ASSERTION_CATCHERS = frozenset({"AssertionError", "Exception", "BaseException"})
# Attributes that only ever hold what the test itself configured or recorded on a mock.
MOCK_CONFIG_ATTRS = ("return_value", "side_effect", "call_args", "await_args", "call_args_list")


@dataclass(frozen=True)
class Candidate:
    path: str  # repo-relative
    lineno: int
    name: str  # dotted: Class.test_method or test_func
    bucket: str
    evidence: str

    @property
    def test_id(self) -> str:
        return f"{self.path}::{'::'.join(self.name.split('.'))}"

    def to_json(self) -> Dict[str, object]:
        return {
            "test_id": self.test_id,
            "path": self.path,
            "lineno": self.lineno,
            "name": self.name,
            "bucket": self.bucket,
            "evidence": self.evidence,
        }


TestFunction = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def _decorator_names(node: TestFunction) -> List[str]:
    names = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        names.append(ast.unparse(target))
    return names


def _decorators(node: TestFunction) -> List[ast.expr]:
    return list(node.decorator_list)


def _call_name(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _mock_expression(value: ast.expr, mock_names: Set[str]) -> bool:
    """True for a mock factory call or an attribute chain rooted at a known mock.

    Deliberately narrow: an expression that merely *passes* a mock into real code
    (`result = handler(mock_client)`) is not mock-derived, because production code
    still decides the outcome.
    """
    node = value.value if isinstance(value, ast.Await) else value
    if isinstance(node, ast.Call):
        return _call_name(node) in MOCK_FACTORIES
    if isinstance(node, ast.Name):
        return node.id in mock_names
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in mock_names


def _mock_bound_names(fn: TestFunction) -> Set[str]:
    """Names in `fn` bound to a Mock/patch result, transitively."""
    mock_names: Set[str] = set()

    def rhs_is_mock(value: ast.expr) -> bool:
        return _mock_expression(value, mock_names)

    # Two passes so `b = a.child` picks up a mock bound later in source order.
    for _ in range(2):
        for node in ast.walk(fn):
            targets: List[ast.expr] = []
            value: Optional[ast.expr] = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None and rhs_is_mock(item.context_expr):
                        targets.append(item.optional_vars)
                        value = item.context_expr
            if value is None or not rhs_is_mock(value):
                continue
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        mock_names.add(sub.id)
    # Fixture/decorator-injected mocks: `@patch(...)` passes the mock in as an
    # argument, conventionally named mock_*.
    for arg in fn.args.args:
        if arg.arg.startswith("mock") or arg.arg.endswith("_mock"):
            mock_names.add(arg.arg)
    return mock_names


def _local_names(node: ast.AST) -> Set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _is_assertive_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Assert):
        return True
    name = _call_name(node)
    if name is None:
        return False
    if name in PYTEST_ASSERT_FUNCS:
        return True
    if name.startswith(ASSERT_CALL_PREFIXES):
        return True
    return False


def _has_assertion(fn: TestFunction) -> bool:
    return any(_is_assertive_node(node) for node in ast.walk(fn))


def _is_truthy_constant(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    return False


def _constant_locals(fn: TestFunction) -> Set[str]:
    """Names assigned a truthy literal exactly once and never rebound."""
    assigned: Dict[str, int] = {}
    literal: Set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assigned[target.id] = assigned.get(target.id, 0) + 1
        if _is_truthy_constant(node.value):
            literal.add(target.id)
    return {name for name in literal if assigned.get(name) == 1}


def _unconditional_skip(fn: TestFunction, decorators: Iterable[ast.expr]) -> Optional[str]:
    for dec in decorators:
        target = dec.func if isinstance(dec, ast.Call) else dec
        rendered = ast.unparse(target)
        if rendered.endswith("mark.skip"):
            return f"@{rendered}"
        if rendered.endswith("mark.skipif") and isinstance(dec, ast.Call) and dec.args:
            if _is_truthy_constant(dec.args[0]):
                return f"@{rendered}(<always true>)"
    body = [s for s in fn.body if not _is_docstring(s)]
    if body and isinstance(body[0], ast.Expr):
        name = _call_name(body[0].value)
        if name in {"skip", "xfail"} and isinstance(body[0].value, ast.Call):
            func = body[0].value.func
            if isinstance(func, ast.Attribute) and ast.unparse(func).startswith("pytest."):
                return f"unconditional pytest.{name}() as first statement"
    return None


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _handler_swallows(handler: ast.ExceptHandler) -> bool:
    """True when the handler cannot surface the failure it caught."""
    body = [s for s in handler.body if not _is_docstring(s)]
    if not body:
        return True
    for stmt in body:
        if isinstance(stmt, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if isinstance(stmt, ast.Return) and stmt.value is None:
            continue
        if isinstance(stmt, ast.Expr):
            name = _call_name(stmt.value)
            if name in SWALLOWING_CALLS:
                continue
        return False
    return True


def _swallowed_assertion(fn: TestFunction) -> Optional[str]:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Try):
            continue
        if not any(_is_assertive_node(sub) for stmt in node.body for sub in ast.walk(stmt)):
            continue
        for handler in node.handlers:
            # `except KeyError: pass` around an assert is deliberate setup
            # tolerance; only a handler that can eat the AssertionError counts.
            if not _swallows_assertion_errors(handler):
                continue
            if _handler_swallows(handler):
                return f"assert inside try/{ast.unparse(handler.type) if handler.type else 'except'} whose handler swallows the failure (line {handler.lineno})"
    return None


def _caught_names(handler: ast.ExceptHandler) -> FrozenSet[str]:
    """The exception names a handler catches, unqualified and tuples flattened.

    Matching the unparsed text instead reads `HTTPException` as broad.
    """
    if handler.type is None:
        return frozenset()
    caught = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return frozenset(
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in caught
        if isinstance(node, (ast.Attribute, ast.Name))
    )


def _swallows_assertion_errors(handler: ast.ExceptHandler) -> bool:
    return handler.type is None or bool(_caught_names(handler) & ASSERTION_CATCHERS)


def _trivial_assert(fn: TestFunction, constant_names: Set[str]) -> Optional[str]:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if _is_truthy_constant(test):
            return f"`assert {ast.unparse(test)}` (line {node.lineno})"
        if isinstance(test, ast.Name) and test.id in constant_names:
            return f"`assert {test.id}` where {test.id} is a literal (line {node.lineno})"
        if isinstance(test, ast.Compare) and len(test.comparators) == 1:
            if ast.unparse(test.left) == ast.unparse(test.comparators[0]):
                return f"`assert {ast.unparse(test)}` compares a value with itself (line {node.lineno})"
    return None


def _mock_tautology(fn: TestFunction, mock_names: Set[str]) -> Optional[str]:
    if not mock_names:
        return None
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
            continue
        sides = (test.left, test.comparators[0])
        if not all(_mock_expression(side, mock_names) for side in sides):
            continue
        if not all(any(attr in ast.unparse(side) for attr in MOCK_CONFIG_ATTRS) for side in sides):
            continue
        if _local_names(test) - mock_names:
            continue
        return f"`assert {ast.unparse(test)}` compares two mock-derived values (line {node.lineno})"
    return None


def _module_level_skip(tree: ast.Module) -> Optional[str]:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
            continue
        rendered = ast.unparse(node.value)
        if "mark.skip" in rendered and "skipif" not in rendered:
            return "module-level pytestmark skip"
    return None


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _nested_scopes(child: ast.stmt) -> Tuple[Sequence[ast.stmt], ...]:
    if isinstance(child, ast.If):
        # Nothing under `if __name__ == "__main__"` runs on import, so pytest never sees it
        return (child.orelse,) if _is_main_guard(child) else (child.body, child.orelse)
    if isinstance(child, ast.Try):
        return (child.body, child.orelse, child.finalbody, *(handler.body for handler in child.handlers))
    if isinstance(child, (ast.With, ast.AsyncWith)):
        return (child.body,)
    if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
        return (child.body, child.orelse)
    return ()


def _scope_statements(body: Sequence[ast.stmt]) -> List[ast.stmt]:
    """Every statement pytest sees at this scope.

    A `def test_x` guarded by `if` or `try` still binds on the module or class,
    so pytest collects it; only another function's body is a different scope.
    """
    nested = [stmt for child in body for group in _nested_scopes(child) for stmt in group]
    return list(body) + (_scope_statements(nested) if nested else [])


def classify_file(path: str, source: str) -> List[Candidate]:
    rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return []
    module_skip = _module_level_skip(tree)
    candidates: List[Candidate] = []

    def visit(node: Union[ast.Module, ast.ClassDef], prefix: str) -> None:
        for child in _scope_statements(node.body):
            if isinstance(child, ast.ClassDef):
                if child.name.startswith("Test"):
                    visit(child, f"{prefix}{child.name}.")
                continue
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # pytest's default python_functions is `test*`, so `testFoo` counts too
            if not child.name.startswith("test"):
                continue
            found = classify_test(child, module_skip)
            if found is not None:
                bucket, evidence = found
                candidates.append(
                    Candidate(
                        path=rel,
                        lineno=child.lineno,
                        name=f"{prefix}{child.name}",
                        bucket=bucket,
                        evidence=evidence,
                    )
                )

    visit(tree, "")
    return candidates


def classify_test(fn: TestFunction, module_skip: Optional[str]) -> Optional[Tuple[str, str]]:
    decorators = _decorators(fn)
    skip = module_skip or _unconditional_skip(fn, decorators)
    if skip:
        return "dead_skip", skip
    swallowed = _swallowed_assertion(fn)
    if swallowed:
        return "swallowed_failure", swallowed
    trivial = _trivial_assert(fn, _constant_locals(fn))
    if trivial:
        return "trivial_assert", trivial
    mock_names = _mock_bound_names(fn)
    tautology = _mock_tautology(fn, mock_names)
    if tautology:
        return "mock_tautology", tautology
    if not _has_assertion(fn):
        return "no_assert", "no assert, pytest.raises/fail, or assert_* call in the test body"
    return None


def iter_test_files(root: str = TESTS_ROOT) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, REPO_ROOT).replace(os.sep, "/")
        if any(rel_dir == ex or rel_dir.startswith(ex + "/") for ex in EXCLUDED_DIRS):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in {"__pycache__", ".pytest_cache"})
        for filename in sorted(filenames):
            if filename.endswith(".py") and (filename.startswith("test_") or filename.endswith("_test.py")):
                yield os.path.join(dirpath, filename)


def collect(root: str = TESTS_ROOT) -> List[Candidate]:
    candidates: List[Candidate] = []
    for path in iter_test_files(root):
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            candidates.extend(classify_file(path, handle.read()))
    return sorted(candidates, key=lambda c: (c.path, c.lineno))


def to_identities(candidates: Iterable[Candidate]) -> Dict[str, Dict[str, List[str]]]:
    """Which tests are candidates, per file and bucket.

    The baseline records names, not counts, so that a fixed test being replaced
    by a newly vacuous one in the same file cannot ride through on an unchanged
    count.
    """
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.path, {}).setdefault(candidate.bucket, []).append(candidate.name)
    return {
        path: {bucket: sorted(names) for bucket, names in sorted(buckets.items())}
        for path, buckets in sorted(grouped.items())
    }


def load_json(path: str, default: object) -> object:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cleared_ids() -> Set[str]:
    data = load_json(CLEARED_PATH, {})
    if isinstance(data, dict):
        return set(data.get("test_ids", []))
    return set()


def write_baseline(identities: Dict[str, Dict[str, List[str]]]) -> None:
    totals: Dict[str, int] = {}
    for buckets in identities.values():
        for bucket, names in buckets.items():
            totals[bucket] = totals.get(bucket, 0) + len(names)
    payload = {
        "_comment": (
            "Ratchet baseline for tests/vacuous_tests/inventory.py. It names every known "
            "candidate per file and bucket, and any candidate missing from it fails the "
            "check. Regenerate with --update-baseline after a cleanup or a rename."
        ),
        "totals": dict(sorted(totals.items())),
        "files": identities,
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def regressions(
    identities: Dict[str, Dict[str, List[str]]],
    baseline_files: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    return sorted(
        f"{path}::{name} is a new {bucket} candidate"
        for path, buckets in identities.items()
        for bucket, names in buckets.items()
        for name in names
        if name not in baseline_files.get(path, {}).get(bucket, [])
    )


def check_against_baseline(identities: Dict[str, Dict[str, List[str]]]) -> int:
    baseline = load_json(BASELINE_PATH, None)
    if baseline is None:
        print(
            f"ERROR: no baseline at {os.path.relpath(BASELINE_PATH, REPO_ROOT)}; run with --update-baseline",
            file=sys.stderr,
        )
        return 1
    base_files: Dict[str, Dict[str, List[str]]] = baseline["files"]
    failures = regressions(identities, base_files)
    if failures:
        print("Vacuous-test ratchet failed. New candidate vacuous tests:\n", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nEach bucket is explained in tests/vacuous_tests/README.md. Make the new "
            "test assert something a mutant can break. A renamed or moved candidate "
            "lands here too; if this is a rename, or a deliberate assert-by-not-raising "
            "test with a docstring saying so, regenerate the baseline with:\n"
            "  python tests/vacuous_tests/inventory.py --update-baseline",
            file=sys.stderr,
        )
        return 1
    improvements = sum(
        1
        for path, buckets in base_files.items()
        for bucket, names in buckets.items()
        for name in names
        if name not in identities.get(path, {}).get(bucket, [])
    )
    print(f"Vacuous-test ratchet OK ({improvements} candidate(s) below baseline).")
    return 0


def print_report(candidates: List[Candidate]) -> None:
    totals: Dict[str, int] = {bucket: 0 for bucket in BUCKETS}
    for candidate in candidates:
        totals[candidate.bucket] += 1
    print(f"candidate vacuous tests: {len(candidates)} across {len({c.path for c in candidates})} files")
    for bucket in BUCKETS:
        print(f"  {bucket:<20} {totals[bucket]}")


def area_of(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:3]) if len(parts) > 3 else os.path.dirname(path)


def areas(candidates: Sequence[Candidate]) -> Tuple[Tuple[str, int], ...]:
    cleared = cleared_ids()
    open_candidates = tuple(c for c in candidates if c.test_id not in cleared)
    return tuple(
        sorted(
            Counter(area_of(c.path) for c in open_candidates).items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def rotated_area(candidates: Sequence[Candidate], day: date) -> Optional[str]:
    """Pick one area per day without storing state, so reviewers get one area per PR."""
    ranked = areas(candidates)
    return ranked[day.toordinal() % len(ranked)][0] if ranked else None


def print_queue(candidates: List[Candidate], limit: int, area: Optional[str]) -> None:
    cleared = cleared_ids()
    queue = [c for c in candidates if c.test_id not in cleared and (area is None or c.path.startswith(area))]
    # Most-specific buckets first: they are the highest-confidence candidates,
    # so the daily run spends its mutation budget where it pays off.
    order = {bucket: index for index, bucket in enumerate(BUCKETS)}
    queue.sort(key=lambda c: (order[c.bucket], c.path, c.lineno))
    print(json.dumps([c.to_json() for c in queue[:limit]], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if counts grew over baseline")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--report", action="store_true", help="print per-bucket totals")
    parser.add_argument("--json", metavar="PATH", help="write the full candidate list")
    parser.add_argument("--queue", type=int, metavar="N", help="print the next N candidates")
    parser.add_argument("--area", help="restrict --queue to a path prefix")
    parser.add_argument("--areas", action="store_true", help="print candidate counts per area")
    parser.add_argument(
        "--todays-area",
        action="store_true",
        help="print the area this day's run should take, rotating by date",
    )
    parser.add_argument("--root", default=TESTS_ROOT, help="tests root to scan")
    args = parser.parse_args()

    candidates = collect(args.root)
    identities = to_identities(candidates)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump([c.to_json() for c in candidates], handle, indent=2)
            handle.write("\n")
    if args.update_baseline:
        write_baseline(identities)
        print(f"wrote {os.path.relpath(BASELINE_PATH, REPO_ROOT)}")
    today = rotated_area(candidates, date.today())
    if args.areas:
        for area, count in areas(candidates):
            print(f"  {area:<50} {count}")
    if args.todays_area and not args.queue:
        print(today or "")
    if args.queue:
        print_queue(candidates, args.queue, args.area or (today if args.todays_area else None))
    if args.report or not (
        args.check or args.update_baseline or args.queue or args.json or args.areas or args.todays_area
    ):
        print_report(candidates)
    if args.check:
        return check_against_baseline(identities)
    return 0


if __name__ == "__main__":
    sys.exit(main())
