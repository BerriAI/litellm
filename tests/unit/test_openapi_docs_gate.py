"""Tests for scripts/openapi_docs_gate.py.

The blame rule lives in scripts/lint_base_counts.py and is tested there; what is tested
here is the checker output parse and `introduced`, the diff scan that names the new
endpoint behind a breach.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Final
from unittest.mock import call, patch

import openapi_docs_gate as gate
import pytest


def test_base_counts_installs_the_native_extension_outside_the_active_environment(tmp_path: Path) -> None:
    parent: Final = tmp_path / "oa_base"
    parent.mkdir()
    worktree: Final = parent / "wt"
    checker: Final = tmp_path / "check_openapi_docs.py"
    checker.write_text("print('POST /v1/test OA001')\n")
    cargo_target_dir: Final = str((tmp_path / "cargo").resolve())
    command: Final = [
        "uv", "pip", "install", "--python", sys.executable, "--no-deps",
        "--target", str(worktree.parent / "native-install"), "--editable", str(worktree),
    ]
    completed: Final = subprocess.CompletedProcess([], 0, stdout="POST /v1/test OA001\n", stderr="")
    with (
        patch.dict("os.environ", {"CARGO_TARGET_DIR": cargo_target_dir}),
        patch("openapi_docs_gate.tempfile.mkdtemp", return_value=str(parent)),
        patch("subprocess.run", return_value=completed) as run,
        patch("signal.signal"),
    ):
        counts: Final = gate.base_counts("HEAD", repo_root=tmp_path, checker=checker)
        calls: Final = run.call_args_list
        install_env: Final = {**os.environ, "CARGO_TARGET_DIR": cargo_target_dir}
        check_env: Final = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(worktree.resolve()), str(worktree.resolve() / "enterprise"), os.environ.get("PYTHONPATH", ""))
            ),
        }
        install_index: Final = calls.index(
            call(command, cwd=worktree, capture_output=True, text=True, env=install_env)
        )
        check_index: Final = next(
            index for index, recorded_call in enumerate(calls) if recorded_call.kwargs.get("env") == check_env
        )
        assert dict(counts) == {"OA001": 1}
        assert install_index < check_index
        assert run.call_count == 4


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
