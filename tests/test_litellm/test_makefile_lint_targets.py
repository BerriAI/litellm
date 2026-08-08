import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

BASEDPYRIGHT_COMMANDS = (
    "uv run --no-sync python scripts/type_check_gate.py --base origin/litellm_internal_staging",
    "uv run --no-sync basedpyright tests/e2e",
)


def _dry_run(target: str) -> str:
    proc = subprocess.run(
        ["make", "--dry-run", target],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.mark.parametrize("command", BASEDPYRIGHT_COMMANDS)
def test_typecheck_runs_every_basedpyright_pass(command: str) -> None:
    assert command in _dry_run("typecheck")


def test_typecheck_provisions_the_environment_basedpyright_resolves_against() -> None:
    output = _dry_run("typecheck")
    assert "uv sync --inexact --frozen --group proxy-dev --group e2e-dev" in output
    assert "scripts/prisma_generate_if_needed.py" in output
    assert "git fetch origin litellm_internal_staging" in output


@pytest.mark.parametrize("command", BASEDPYRIGHT_COMMANDS)
def test_lint_leaves_basedpyright_to_typecheck(command: str) -> None:
    assert command not in _dry_run("lint")


@pytest.mark.parametrize(
    "command",
    [
        "ruff format --check",
        "ruff check .",
        "scripts/ruff_strict_gate.py --base origin/litellm_internal_staging",
        "scripts/type_discipline_gate.py --base origin/litellm_internal_staging",
        "tests/documentation_tests/test_circular_imports.py",
        "from litellm import *",
    ],
)
def test_lint_still_runs_the_other_gating_ci_checks(command: str) -> None:
    assert command in _dry_run("lint")
