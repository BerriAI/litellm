"""Tests for litellm/scripts/openapi_docs_gate.py.

The blame rule lives in litellm/scripts/lint_base_counts.py and is tested there; what is tested
here is the checker output parse and `introduced`, the diff scan that names the new
endpoint behind a breach.
"""

import subprocess
from pathlib import Path
from typing import Final
from unittest.mock import patch

import openapi_docs_gate as gate
import pytest


def test_base_counts_stops_when_the_native_build_fails(tmp_path: Path) -> None:
    parent: Final = tmp_path / "oa_base"
    parent.mkdir()
    checker: Final = tmp_path / "check_openapi_docs.py"
    checker.write_text("print('POST /v1/test OA001')\n")
    with (
        patch("openapi_docs_gate.tempfile.mkdtemp", return_value=str(parent)),
        patch(
            "subprocess.run",
            side_effect=(
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                subprocess.CompletedProcess([], 1, stdout="", stderr="build failed"),
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            ),
        ) as run,
        patch("signal.signal"),
    ):
        with pytest.raises(SystemExit, match="uv exited 1"):
            gate.base_counts("HEAD", repo_root=tmp_path, checker=checker)
        assert run.call_count == 3


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


_CHECKER_THAT_NEEDS_THE_NATIVE_MODULE: Final = (
    "from pathlib import Path\n"
    "native = Path('litellm/rust_bridge/_native.abi3.so')\n"
    "print('POST /v1/test OA001' if native.read_bytes() == b'head build' else '')\n"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ("git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args),
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repo_with_a_built_native_module(tmp_path: Path) -> tuple[Path, Path]:
    repo: Final = tmp_path / "repo"
    (repo / "litellm/rust/crates").mkdir(parents=True)
    (repo / "litellm/rust_bridge").mkdir(parents=True)
    (repo / "litellm/rust/crates/lib.rs").write_text("pub fn bridge() {}\n")
    (repo / "litellm/rust_bridge/__init__.py").write_text("")
    (repo / "pyproject.toml").write_text("not a valid pyproject\n")
    (repo / ".gitignore").write_text("*.so\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "litellm/rust_bridge/_native.abi3.so").write_bytes(b"head build")
    checker: Final = tmp_path / "check_openapi_docs.py"
    checker.write_text(_CHECKER_THAT_NEEDS_THE_NATIVE_MODULE)
    return repo, checker


def test_base_counts_reuses_the_head_native_module_when_the_rust_build_inputs_match(tmp_path: Path) -> None:
    repo, checker = _repo_with_a_built_native_module(tmp_path)
    with patch("signal.signal"):
        assert gate.base_counts("HEAD", repo_root=repo, checker=checker) == {"OA001": 1}


@pytest.mark.parametrize(
    "change",
    (
        lambda repo: (repo / "litellm/rust/crates/lib.rs").write_text("pub fn bridge_v2() {}\n"),
        lambda repo: (repo / "pyproject.toml").write_text("still not a valid pyproject\n"),
        lambda repo: (repo / "litellm/rust_bridge/_native.abi3.so").unlink(),
    ),
    ids=("rust source changed", "pyproject changed", "no head build"),
)
def test_base_counts_builds_the_base_native_module_when_head_cannot_stand_in(tmp_path: Path, change) -> None:
    repo, checker = _repo_with_a_built_native_module(tmp_path)
    change(repo)
    with patch("signal.signal"), pytest.raises(SystemExit, match="uv exited"):
        gate.base_counts("HEAD", repo_root=repo, checker=checker)
