"""Tests for scripts/openapi_docs_gate.py.

The blame rule lives in scripts/lint_base_counts.py and is tested there; what is tested
here is the checker output parse, `introduced`, the diff scan that names the new
endpoint behind a breach, and the base scan's import of the Rust bridge extension that
git does not carry.
"""

import subprocess
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Final

import openapi_docs_gate as gate

_NATIVE_RESOLVING_CHECKER: Final = (
    "import importlib.util, pathlib\n"
    "spec = importlib.util.find_spec('litellm.rust_bridge._native')\n"
    "if spec is None:\n"
    "    raise ImportError(\"cannot import name '_native' from 'litellm.rust_bridge'\")\n"
    "print('POST /native ' + pathlib.Path(spec.origin).read_text().strip())\n"
)


def test_editing_the_checker_rekeys_the_base_counts(tmp_path: Path) -> None:
    checker: Final = tmp_path / "check.py"
    checker.write_text("print('v1')\n")
    before: Final = gate.checker_identity(checker).artifact_name("abc123")
    checker.write_text("print('v2')\n")
    assert gate.checker_identity(checker).artifact_name("abc123") != before


def test_parse_violations_skips_anything_that_is_not_a_finding_line() -> None:
    out: Final = "Some warning from an import\nPOST /v1/decisions OA001\nPOST /v1/decisions OA003\nnot a line\n"
    assert gate.parse_violations(out) == (
        gate.Violation("POST", "/v1/decisions", "OA001"),
        gate.Violation("POST", "/v1/decisions", "OA003"),
    )


def test_count_by_rule_sums_findings_per_code() -> None:
    violations: Final = (
        gate.Violation("POST", "/a", "OA001"),
        gate.Violation("POST", "/b", "OA001"),
        gate.Violation("POST", "/b", "OA003"),
    )
    assert dict(gate.count_by_rule(violations)) == {"OA001": 2, "OA003": 1}


def test_introduced_keeps_only_routes_registered_on_added_lines() -> None:
    diff: Final = (
        "diff --git a/litellm/proxy/x.py b/litellm/proxy/x.py\n"
        "--- a/litellm/proxy/x.py\n"
        "+++ b/litellm/proxy/x.py\n"
        "@@ -0,0 +3,2 @@\n"
        '+@router.post("/v1/new")\n'
        "+async def new(request: Request): ...\n"
        "@@ -10,1 +20,1 @@\n"
        '-@router.post("/v1/removed")\n'
    )
    violations: Final = (
        gate.Violation("POST", "/v1/new", "OA001"),
        gate.Violation("POST", "/v1/old", "OA001"),
        gate.Violation("POST", "/v1/removed", "OA001"),
    )
    assert gate.introduced(violations, gate.added_text(diff)) == (gate.Violation("POST", "/v1/new", "OA001"),)


def test_introduced_does_not_match_a_path_that_is_only_a_prefix_of_an_added_one() -> None:
    added: Final = gate.added_text('+@router.post("/v1/decisions/batch")\n')
    assert gate.introduced((gate.Violation("POST", "/v1/decisions", "OA002"),), added) == ()


def _git(cwd: Path, *args: str) -> None:
    proc: Final = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def _repo_committing(tmp_path: Path, tracked: tuple[str, ...]) -> Path:
    repo: Final = tmp_path / "repo"
    for relative in tracked:
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text("")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "gate@example.com")
    _git(repo, "config", "user.name", "gate")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _checker(tmp_path: Path, source: str) -> Path:
    checker: Final = tmp_path / "check.py"
    checker.write_text(source)
    return checker


def test_base_scan_imports_the_rust_extension_the_head_tree_built(tmp_path: Path) -> None:
    repo: Final = _repo_committing(tmp_path, ("litellm/__init__.py", "litellm/rust_bridge/__init__.py"))
    (repo / "litellm" / "rust_bridge" / f"_native{EXTENSION_SUFFIXES[0]}").write_text("OA101")
    (repo / "litellm" / "rust_bridge" / f"_native{EXTENSION_SUFFIXES[1]}").write_text("OA102")
    counts: Final = gate.base_counts("HEAD", repo_root=repo, checker=_checker(tmp_path, _NATIVE_RESOLVING_CHECKER))
    assert dict(counts) == {"OA101": 1}


def test_base_scan_of_a_tree_without_the_rust_bridge_still_counts(tmp_path: Path) -> None:
    repo: Final = _repo_committing(tmp_path, ("litellm/__init__.py",))
    (repo / "litellm" / "rust_bridge").mkdir()
    (repo / "litellm" / "rust_bridge" / f"_native{EXTENSION_SUFFIXES[0]}").write_text("OA101")
    checker: Final = _checker(tmp_path, "print('POST /v1/old OA001')\n")
    assert dict(gate.base_counts("HEAD", repo_root=repo, checker=checker)) == {"OA001": 1}
