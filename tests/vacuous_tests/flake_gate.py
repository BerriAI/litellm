"""Anti-flake gate for tests the vacuous-test audit rewrote.

A test that is vacuous today must not become flaky tomorrow, so every test the
daily run touches has to clear this before its PR opens: repeated runs under
different hash seeds, a full run of the owning file, and a static scan for the
usual sources of nondeterminism.

Usage:
    python tests/vacuous_tests/flake_gate.py tests/x/test_y.py::test_z [more ...]
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (dotted call or attribute, why it is a flake risk)
FLAKY_CALLS: Tuple[Tuple[str, str], ...] = (
    ("time.sleep", "wall-clock sleep: slow and racy under load"),
    ("asyncio.sleep", "wall-clock sleep: racy under load, prefer awaiting the real signal"),
    ("datetime.now", "current time in a test makes it depend on when it runs"),
    ("datetime.utcnow", "current time in a test makes it depend on when it runs"),
    ("time.time", "current time in a test makes it depend on when it runs"),
    ("random.random", "unseeded randomness"),
    ("random.choice", "unseeded randomness"),
    ("uuid.uuid4", "unseeded randomness in an assertion is unpredictable"),
    ("requests.get", "real network call"),
    ("requests.post", "real network call"),
    ("httpx.get", "real network call"),
    ("httpx.post", "real network call"),
    ("litellm.completion", "hits a live provider unless mocked or replayed"),
    ("litellm.acompletion", "hits a live provider unless mocked or replayed"),
)
MOCK_MARKERS = ("mock", "patch", "respx", "vcr", "cassette", "monkeypatch", "AsyncMock", "MagicMock")
# Safe when the transport is patched for the call, unlike time and randomness,
# which stay nondeterministic however the test is written.
NETWORK_CALLS = frozenset(
    {
        "requests.get",
        "requests.post",
        "httpx.get",
        "httpx.post",
        "litellm.completion",
        "litellm.acompletion",
    }
)


@dataclass(frozen=True)
class Finding:
    test_id: str
    problem: str


def _env(seed: str) -> Dict[str, str]:
    return {
        **os.environ,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PYTHONHASHSEED": seed,
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _run(target: str, seed: str, timeout: int) -> Tuple[int, str]:
    command: Sequence[str] = (
        sys.executable,
        "-m",
        "pytest",
        target,
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        f"--timeout={timeout}",
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, env=_env(seed), capture_output=True, text=True)
    return completed.returncode, (completed.stdout + completed.stderr)[-2000:]


TestFunction = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def _find_test(path: str, name: str) -> Optional[TestFunction]:
    with open(os.path.join(REPO_ROOT, path), "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    wanted = name.split("::")[-1]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == wanted:
            return node
    return None


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _mocked(marker_source: str) -> bool:
    return any(marker in marker_source for marker in MOCK_MARKERS)


def mocked_lines(node: TestFunction) -> FrozenSet[int]:
    """Lines where a patch is in force, either from a decorator or an enclosing with.

    A mock anywhere in the test body is not enough: a test can patch one client
    and still call a live provider two lines later.
    """
    if any(_mocked(ast.unparse(decorator)) for decorator in node.decorator_list):
        return frozenset(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return frozenset(
        line
        for statement in ast.walk(node)
        if isinstance(statement, (ast.With, ast.AsyncWith))
        and any(_mocked(ast.unparse(item.context_expr)) for item in statement.items)
        for line in range(statement.lineno, (statement.end_lineno or statement.lineno) + 1)
    )


def static_findings(test_id: str) -> List[Finding]:
    path, _, name = test_id.partition("::")
    node = _find_test(path, name)
    if node is None:
        return [Finding(test_id, "test not found in file")]
    patched = mocked_lines(node)
    reasons: Dict[str, str] = dict(FLAKY_CALLS)
    findings: List[Finding] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        dotted = _dotted(call.func)
        matched = next((known for known in reasons if dotted == known or dotted.endswith(f".{known}")), None)
        if matched is None or (matched in NETWORK_CALLS and call.lineno in patched):
            continue
        findings.append(Finding(test_id, f"uses `{matched}`: {reasons[matched]}"))
    return findings


def dynamic_findings(test_id: str, repeat: int, timeout: int) -> List[Finding]:
    findings: List[Finding] = []
    for index in range(repeat):
        seed = str(index * 7919 + 1)
        code, output = _run(test_id, seed, timeout)
        if code != 0:
            findings.append(
                Finding(test_id, f"failed on repeat {index + 1}/{repeat} with PYTHONHASHSEED={seed}:\n{output[-600:]}")
            )
            break
    path = test_id.partition("::")[0]
    code, output = _run(path, "0", timeout)
    if code != 0:
        findings.append(Finding(test_id, f"owning file {path} does not pass as a whole:\n{output[-600:]}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_ids", nargs="+")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    findings: List[Finding] = []
    for test_id in args.test_ids:
        findings.extend(static_findings(test_id))
        if not args.static_only:
            findings.extend(dynamic_findings(test_id, args.repeat, args.timeout))

    if findings:
        print("flake gate FAILED")
        for finding in findings:
            print(f"  - {finding.test_id}: {finding.problem}")
        return 1
    print(f"flake gate OK for {len(args.test_ids)} test(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
