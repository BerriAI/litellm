"""Regression tests for the UI unit-test scope decision.

`.github/workflows/test-litellm-ui-unit.yml` narrows the dashboard's Vitest run to
`vitest related <changed files>` so a pull request only pays for the tests it can
affect. `related` resolves a file to the tests that import it, so a file no test
imports resolves to nothing, and with `--passWithNoTests` the job then goes green
without running a single test. `package.json`, `package-lock.json`, the Vitest and
TypeScript configs and `tests/setupTests.ts` are all such files, so a dependency
bump used to merge untested.

`.github/scripts/select_ui_test_scope.sh` is the decision function that closes
that hole: it prints `related` only when every changed file lives under `src/`,
and `full` otherwise. These tests lock both the decision and the workflow step
that consumes it, running the step's real shell against stubbed `gh`, `git` and
`npm` so a regression shows up as the wrong Vitest command.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "select_ui_test_scope.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test-litellm-ui-unit.yml"
STEP_NAME = "Run UI unit tests (Vitest)"

FULL_SUITE_ARGV = ["run", "test", "--", "--run", "--pool", "forks", "--poolOptions.forks.maxForks=14"]

NON_SRC_FILES = [
    "package.json",
    "package-lock.json",
    "vitest.config.ts",
    "tsconfig.json",
    "tests/setupTests.ts",
    "next.config.mjs",
]


def scope(changed: list[str]) -> str:
    result = subprocess.run(
        ["bash", str(SCOPE_SCRIPT)],
        input="\n".join(changed),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("changed_file", NON_SRC_FILES)
def test_a_file_no_test_imports_selects_the_full_suite(changed_file: str) -> None:
    assert scope([changed_file]) == "full"


def test_src_only_changes_stay_on_related() -> None:
    assert scope(["src/app/page.tsx", "src/lib/http/client.ts"]) == "related"


def test_one_non_src_file_pulls_a_src_only_set_up_to_full() -> None:
    assert scope(["src/app/page.tsx", "package-lock.json"]) == "full"


def test_a_path_merely_prefixed_with_src_is_not_under_src() -> None:
    assert scope(["srcipts/build.mjs"]) == "full"


def test_an_empty_change_set_fails_open_to_the_full_suite() -> None:
    assert scope([]) == "full"


def _step_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["ui-unit-tests"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == STEP_NAME)
    resolved = script.replace("${{ github.repository }}", "BerriAI/litellm")
    assert "${{" not in resolved, "the step uses an Actions expression this harness does not resolve"
    return resolved


def _stub(bin_dir: Path, name: str, body: str) -> None:
    stub = bin_dir / name
    stub.write_text(f"#!/usr/bin/env bash\n{body}\n")
    stub.chmod(0o755)


def _run_step(tmp_path: Path, changed: list[str], base_sha: str = "basesha") -> tuple[int, str, list[list[str]]]:
    """Run the workflow step's real shell; return its status, stdout and every npm argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text("".join(f"{name}\n" for name in changed))
    npm_log = tmp_path / "npm.log"

    _stub(bin_dir, "gh", 'echo "mergebasesha"')
    _stub(bin_dir, "git", 'if [ "$1" = diff ]; then cat "$CHANGED_FILES"; fi')
    _stub(bin_dir, "npm", 'printf "%s\\n" "$@" >>"$NPM_LOG"; printf "\\0" >>"$NPM_LOG"')

    step = tmp_path / "step.sh"
    step.write_text(_step_script())

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BASE_SHA"] = base_sha
    env["HEAD_SHA"] = "headsha"
    env["GITHUB_WORKSPACE"] = str(REPO_ROOT)
    env["GITHUB_REF_NAME"] = "litellm_internal_staging"
    env["CHANGED_FILES"] = str(changed_file)
    env["NPM_LOG"] = str(npm_log)

    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(step)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )
    raw = npm_log.read_text() if npm_log.exists() else ""
    invocations = [call.splitlines() for call in raw.split("\0") if call]
    return result.returncode, result.stdout + result.stderr, invocations


def _related_argv(changed: list[str]) -> list[str]:
    return [
        "run",
        "test",
        "--",
        "related",
        *changed,
        "--run",
        "--passWithNoTests",
        "--pool",
        "forks",
        "--poolOptions.forks.maxForks=14",
    ]


def test_step_runs_related_for_a_src_only_pull_request(tmp_path: Path) -> None:
    changed = ["src/app/page.tsx", "src/lib/http/client.ts"]
    returncode, output, invocations = _run_step(tmp_path, changed)
    assert returncode == 0, output
    assert invocations == [_related_argv(changed)]


@pytest.mark.parametrize("changed_file", NON_SRC_FILES)
def test_step_runs_the_full_suite_when_a_changed_file_is_outside_src(tmp_path: Path, changed_file: str) -> None:
    returncode, output, invocations = _run_step(tmp_path, ["src/app/page.tsx", changed_file])
    assert returncode == 0, output
    assert invocations == [FULL_SUITE_ARGV]
    assert "related" not in " ".join(invocations[0])


def test_step_runs_the_full_suite_on_a_push(tmp_path: Path) -> None:
    returncode, output, invocations = _run_step(tmp_path, ["src/app/page.tsx"], base_sha="")
    assert returncode == 0, output
    assert invocations == [FULL_SUITE_ARGV]


def test_step_runs_nothing_when_the_pull_request_touches_no_dashboard_file(tmp_path: Path) -> None:
    returncode, output, invocations = _run_step(tmp_path, [])
    assert returncode == 0, output
    assert invocations == []
