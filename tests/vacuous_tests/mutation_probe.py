"""Stage B of the vacuous-test audit: decide whether one test is actually vacuous.

Runs a single test under coverage to find the production lines it executes,
mutates only those lines, and re-runs the same test against each mutant. A test
that survives every mutant cannot fail when the code it covers is broken, which
is the working definition of vacuous. A single kill proves the test has teeth,
and is recorded in verified_not_vacuous.json so the daily run stops re-flagging
it.

Unlike a whole-folder mutmut run (see .github/workflows/mutation-test.yml) this
is scoped to one test and a handful of mutants, so it finishes in minutes.

Usage:
    python tests/vacuous_tests/mutation_probe.py \
        "tests/test_litellm/foo/test_bar.py::test_baz" --json report.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL_DIR = os.path.join(REPO_ROOT, "tests", "vacuous_tests")
CLEARED_PATH = os.path.join(TOOL_DIR, "verified_not_vacuous.json")

# Files that are configuration or generated data rather than behaviour: mutating
# them says nothing about whether a test has teeth.
SKIPPED_SOURCES = ("litellm/types/", "litellm/proxy/_types.py", "litellm/litellm_core_utils/model_param_helper.py")

# Below this, "coverage" of a file is incidental (a lazy import, a decorator) rather
# than the test exercising it.
MIN_LINES_PER_FILE = 3

COMPARE_SWAPS: Dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
BINOP_SWAPS: Dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.Div: ast.Mult,
}


@dataclass(frozen=True)
class Mutant:
    path: str
    lineno: int
    description: str
    source: str
    # A test that asserts by not raising can only be killed by a mutant that
    # raises, so these go first in the budget.
    swallow: bool = False


@dataclass
class MutantResult:
    path: str
    lineno: int
    description: str
    outcome: str  # killed | survived | timeout | broken


@dataclass
class ProbeReport:
    test_id: str
    verdict: str
    detail: str
    covered_files: Dict[str, int] = field(default_factory=dict)
    mutants: List[MutantResult] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {
            "test_id": self.test_id,
            "verdict": self.verdict,
            "detail": self.detail,
            "covered_files": self.covered_files,
            "mutants": [vars(m) for m in self.mutants],
        }


def _pytest_env() -> Dict[str, str]:
    return {
        **os.environ,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def run_test(test_id: str, timeout: int, overlay: Optional[str] = None) -> Tuple[int, str]:
    command: Sequence[str] = (
        sys.executable,
        "-m",
        "pytest",
        test_id,
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        f"--timeout={timeout}",
    )
    try:
        completed = subprocess.run(
            command,
            cwd=overlay or REPO_ROOT,
            env=_pytest_env(),
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )
    except subprocess.TimeoutExpired:
        return 124, "pytest wall-clock timeout"
    return completed.returncode, (completed.stdout + completed.stderr)[-4000:]


def _coverage_of(test_id: str, timeout: int) -> Dict[str, Set[int]]:
    import coverage

    with tempfile.TemporaryDirectory() as tmp:
        data_file = os.path.join(tmp, ".coverage")
        command: Sequence[str] = (
            sys.executable,
            "-m",
            "coverage",
            "run",
            f"--data-file={data_file}",
            "--source=litellm",
            "-m",
            "pytest",
            test_id,
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            f"--timeout={timeout}",
        )
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=_pytest_env(),
            capture_output=True,
            text=True,
            timeout=timeout + 120,
        )
        data = coverage.CoverageData(basename=data_file)
        data.read()
        result: Dict[str, Set[int]] = {}
        for measured in data.measured_files():
            rel = os.path.relpath(measured, REPO_ROOT).replace(os.sep, "/")
            if not rel.startswith("litellm/") or rel.startswith(SKIPPED_SOURCES):
                continue
            lines = data.lines(measured) or []
            if lines:
                result[rel] = set(lines)
        return result


def covered_lines(test_id: str, timeout: int) -> Dict[str, List[int]]:
    """Lines the test itself exercises, with import-time coverage subtracted.

    Collecting any test in a directory imports litellm and that directory's
    conftest, which lights up thousands of module-level lines. Those lines are
    covered no matter what the test does, so mutating them measures the import,
    not the test. A no-op test in the same directory gives the floor to subtract.
    """
    test_path = test_id.split("::", 1)[0]
    with _noop_test(os.path.dirname(os.path.join(REPO_ROOT, test_path))) as noop_id:
        floor = _coverage_of(noop_id, timeout)
    actual = _coverage_of(test_id, timeout)
    result: Dict[str, List[int]] = {}
    for path, lines in actual.items():
        own = sorted(lines - floor.get(path, set()))
        if own:
            result[path] = own
    return result


@contextmanager
def _noop_test(directory: str) -> Iterator[str]:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="test_vacuous_probe_floor_", dir=directory, delete=False
    )
    try:
        handle.write("def test_noop() -> None:\n    assert True\n")
        handle.close()
        rel = os.path.relpath(handle.name, REPO_ROOT).replace(os.sep, "/")
        yield f"{rel}::test_noop"
    finally:
        os.unlink(handle.name)


NodeKey = Tuple[str, int, int]


def _position(node: ast.AST) -> Tuple[int, int]:
    if isinstance(node, (ast.expr, ast.stmt, ast.ExceptHandler)):
        return (node.lineno, node.col_offset)
    return (-1, -1)


def _key(node: ast.AST) -> NodeKey:
    lineno, col_offset = _position(node)
    return (type(node).__name__, lineno, col_offset)


Mutation = Callable[[ast.AST], ast.AST]


def _mutant_source(original: str, key: NodeKey, mutate: Mutation) -> str:
    class Transformer(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if _key(node) == key:
                return mutate(node)
            return super().visit(node)

    mutated = Transformer().visit(ast.parse(original))
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated)


def function_body_lines(tree: ast.Module) -> Set[int]:
    """Line numbers inside a function body.

    Module and class level lines run at import, so a mutant there is killed (or
    not) by whether the module still imports, which says nothing about the test.
    """
    lines: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for statement in node.body:
            end = statement.end_lineno or statement.lineno
            lines.update(range(statement.lineno, end + 1))
    return lines


def behavioural_lines(path: str, own_lines: Iterable[int]) -> List[int]:
    with open(os.path.join(REPO_ROOT, path), "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    inside = function_body_lines(tree)
    return sorted(set(own_lines) & inside)


def imported_modules(test_path: str) -> Set[str]:
    with open(os.path.join(REPO_ROOT, test_path), "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {module for module in modules if module.startswith("litellm")}


def _module_name(path: str) -> str:
    trimmed = path[: -len(".py")] if path.endswith(".py") else path
    return trimmed.replace("/", ".").removesuffix(".__init__")


def _is_under_test(path: str, imports: Set[str]) -> bool:
    """True when the test file imports this module, directly or as a parent package."""
    module = _module_name(path)
    return any(imported == module or imported.startswith(module + ".") for imported in imports)


def generate_mutants(path: str, lines: Iterable[int]) -> List[Mutant]:
    absolute = os.path.join(REPO_ROOT, path)
    with open(absolute, "r", encoding="utf-8") as handle:
        original = handle.read()
    tree = ast.parse(original)
    covered = set(lines)
    mutants: List[Mutant] = []

    def emit(node: ast.AST, description: str, mutate: Mutation, swallow: bool = False) -> None:
        lineno, _ = _position(node)
        if lineno not in covered:
            return
        try:
            source = _mutant_source(original, _key(node), mutate)
        except Exception:
            return
        mutants.append(
            Mutant(path=path, lineno=lineno, description=description, source=source, swallow=swallow)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and not _bare_reraise(node):
            emit(
                node,
                f"stop swallowing `{_snippet(node.type) if node.type else 'except'}` and re-raise",
                lambda n: ast.ExceptHandler(type=n.type, name=n.name, body=[ast.Raise()]),
                swallow=True,
            )
            continue
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARE_SWAPS:
            swap = COMPARE_SWAPS[type(node.ops[0])]
            emit(
                node,
                f"flip `{type(node.ops[0]).__name__}` to `{swap.__name__}` in `{_snippet(node)}`",
                lambda n, swap=swap: ast.Compare(left=n.left, ops=[swap()], comparators=n.comparators),
            )
        elif isinstance(node, ast.BoolOp):
            swap = ast.Or if isinstance(node.op, ast.And) else ast.And
            emit(
                node,
                f"swap `{'and' if isinstance(node.op, ast.And) else 'or'}` in `{_snippet(node)}`",
                lambda n, swap=swap: ast.BoolOp(op=swap(), values=n.values),
            )
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAPS:
            swap = BINOP_SWAPS[type(node.op)]
            emit(
                node,
                f"swap `{type(node.op).__name__}` for `{swap.__name__}` in `{_snippet(node)}`",
                lambda n, swap=swap: ast.BinOp(left=n.left, op=swap(), right=n.right),
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            emit(
                node,
                f"replace `{node.value}` with `{not node.value}`",
                lambda n: ast.Constant(value=not n.value),
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            emit(
                node,
                f"replace `{node.value}` with `{node.value + 1}`",
                lambda n: ast.Constant(value=n.value + 1),
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
            emit(
                node,
                f"replace string `{node.value[:30]}` with a different value",
                lambda n: ast.Constant(value=f"litellm_mutant_{n.value}"),
            )
        elif isinstance(node, ast.Return) and node.value is not None and not _is_none(node.value):
            emit(
                node,
                f"replace `return {_snippet(node.value)}` with `return None`",
                lambda n: ast.Return(value=ast.Constant(value=None)),
            )
    return mutants


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _bare_reraise(handler: ast.ExceptHandler) -> bool:
    only = handler.body[0] if len(handler.body) == 1 else None
    return isinstance(only, ast.Raise) and only.exc is None


def _snippet(node: ast.AST, limit: int = 60) -> str:
    try:
        rendered = ast.unparse(node)
    except Exception:
        return "<expr>"
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def select_mutants(mutants: Sequence[Mutant], limit: int) -> List[Mutant]:
    """Round-robin over distinct lines so the budget spreads across the code path.

    Swallowed-exception mutants come first: a test whose only claim is that the
    call does not raise cannot be killed by anything else, and getting that
    wrong would call a real regression test vacuous.
    """
    swallows = [mutant for mutant in mutants if mutant.swallow][:limit]
    if swallows:
        return swallows + select_mutants([m for m in mutants if not m.swallow], limit - len(swallows))
    by_line: Dict[Tuple[str, int], List[Mutant]] = {}
    for mutant in mutants:
        by_line.setdefault((mutant.path, mutant.lineno), []).append(mutant)
    ordered: List[Mutant] = []
    depth = 0
    while len(ordered) < limit:
        added = False
        for key in sorted(by_line):
            bucket = by_line[key]
            if depth < len(bucket):
                ordered.append(bucket[depth])
                added = True
                if len(ordered) == limit:
                    break
        if not added:
            break
        depth += 1
    return ordered


def interleave(groups: Sequence[Sequence[Mutant]], limit: int) -> List[Mutant]:
    """Take from each file in turn, so the module under test always gets mutants.

    Files are already ranked with the module under test first, and a busy shared
    module like a cache can otherwise eat the whole budget.
    """
    return [
        group[index]
        for index in range(max((len(group) for group in groups), default=0))
        for group in groups
        if index < len(group)
    ][:limit]


def probe(test_id: str, max_mutants: int, max_files: int, timeout: int) -> ProbeReport:
    code, output = run_test(test_id, timeout)
    if code == 5:
        return ProbeReport(test_id, "inconclusive", "test not collected (renamed or parametrized away)")
    if code != 0:
        return ProbeReport(test_id, "already_failing", f"baseline run failed:\n{output[-800:]}")
    if " skipped" in output and " passed" not in output:
        return ProbeReport(test_id, "dead", "test is skipped in this environment, so it can never fail")

    coverage_map = covered_lines(test_id, timeout)
    behavioural = {
        path: lines
        for path, lines in ((path, behavioural_lines(path, own)) for path, own in coverage_map.items())
        if len(lines) >= MIN_LINES_PER_FILE
    }
    if not behavioural:
        return ProbeReport(
            test_id,
            "vacuous",
            "test executes no litellm function body of its own; it only imports modules",
        )

    imports = imported_modules(test_id.partition("::")[0])
    ranked = sorted(
        behavioural.items(),
        key=lambda item: (not _is_under_test(item[0], imports), -len(item[1]), item[0]),
    )[:max_files]
    if not any(_is_under_test(path, imports) for path, _ in ranked):
        return ProbeReport(
            test_id,
            "inconclusive",
            "the test runs no function body of the modules it imports, only shared "
            f"infrastructure ({', '.join(path for path, _ in ranked)}); needs a human",
            covered_files={path: len(lines) for path, lines in ranked},
        )
    report = ProbeReport(
        test_id,
        "inconclusive",
        "",
        covered_files={path: len(lines) for path, lines in ranked},
    )
    candidates = interleave(
        [select_mutants(generate_mutants(path, lines), max_mutants) for path, lines in ranked],
        max_mutants,
    )
    if not candidates:
        report.detail = "no mutable statements on the covered lines"
        return report

    kills = 0
    for mutant in candidates:
        outcome = _run_mutant(mutant, test_id, timeout)
        report.mutants.append(MutantResult(mutant.path, mutant.lineno, mutant.description, outcome))
        if outcome in {"killed", "timeout"}:
            kills += 1
            break

    if kills:
        report.verdict = "not_vacuous"
        report.detail = f"killed by mutant: {report.mutants[-1].description} ({report.mutants[-1].path}:{report.mutants[-1].lineno})"
        return report
    tested = [m for m in report.mutants if m.outcome != "broken"]
    if len(tested) < 3:
        report.detail = f"only {len(tested)} usable mutant(s); not enough signal"
        return report
    report.verdict = "vacuous"
    report.detail = f"survived all {len(tested)} mutants on the code it covers"
    return report


def _mirror(source: str, destination: str, remaining: Sequence[str]) -> None:
    os.makedirs(destination, exist_ok=True)
    head = remaining[0]
    for entry in os.scandir(source):
        if entry.name != head:
            os.symlink(entry.path, os.path.join(destination, entry.name))
    if len(remaining) > 1:
        _mirror(os.path.join(source, head), os.path.join(destination, head), remaining[1:])


@contextmanager
def mutant_overlay(mutant: Mutant) -> Iterator[str]:
    """A working directory where only the mutated file differs from the real tree.

    Everything except the mutated file's own directory chain is symlinked, so the
    real source is never written to: a crashed or killed probe cannot leave a
    mutant behind in the repo. Running pytest with this as cwd puts it at the
    front of sys.path, so the mutated module wins over the installed one.
    """
    with tempfile.TemporaryDirectory(prefix="vacuous_mutant_") as overlay:
        parts = mutant.path.split("/")
        _mirror(REPO_ROOT, overlay, parts)
        with open(os.path.join(overlay, mutant.path), "w", encoding="utf-8") as handle:
            handle.write(mutant.source)
        yield overlay


def _run_mutant(mutant: Mutant, test_id: str, timeout: int) -> str:
    with mutant_overlay(mutant) as overlay:
        code, output = run_test(test_id, timeout, overlay=overlay)
    if code == 124:
        return "timeout"
    if code == 0:
        return "survived"
    if "ImportError" in output or "SyntaxError" in output or code == 5:
        return "broken"
    return "killed"


def record_cleared(test_id: str) -> None:
    payload = {"test_ids": []}
    if os.path.exists(CLEARED_PATH):
        with open(CLEARED_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    ids = sorted(set(payload.get("test_ids", [])) | {test_id})
    payload["test_ids"] = ids
    payload.setdefault(
        "_comment",
        "Tests a mutation probe proved have teeth. inventory.py --queue skips these.",
    )
    with open(CLEARED_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_id", help="pytest node id, e.g. tests/x/test_y.py::test_z")
    parser.add_argument("--max-mutants", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180, help="per-test-run timeout in seconds")
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--record", action="store_true", help="record a not_vacuous verdict")
    args = parser.parse_args()

    report = probe(args.test_id, args.max_mutants, args.max_files, args.timeout)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report.to_json(), handle, indent=2)
            handle.write("\n")
    print(f"{report.verdict}: {report.test_id}")
    print(f"  {report.detail}")
    for mutant in report.mutants:
        print(f"  [{mutant.outcome}] {mutant.path}:{mutant.lineno} {mutant.description}")
    if args.record and report.verdict == "not_vacuous":
        record_cleared(report.test_id)
        print(f"  recorded in {os.path.relpath(CLEARED_PATH, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
