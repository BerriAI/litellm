from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Final

import pytest

WORKFLOW: Final = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "test-linting.yml"
DIFF_GATE: Final = re.compile(r'git diff --name-only --diff-filter=\w+ "\$GATE_BASE_SHA" HEAD -- (.+?) \|')
GATES: Final = tuple(tuple(shlex.split(gate.group(1))) for gate in DIFF_GATE.finditer(WORKFLOW.read_text()))
PYTHON_GATES: Final = tuple(gate for gate in GATES if gate[0].startswith(":(glob)"))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _scoped_root(pathspec: str) -> str:
    return re.sub(r"^:\([^)]*\)", "", pathspec).split("*", 1)[0]


def _gate_rooted_at(root: str) -> tuple[str, ...]:
    return next(gate for gate in GATES if _scoped_root(gate[0]) == root)


def _changed_files_selected_by(tmp_path: Path, pathspecs: tuple[str, ...], files: tuple[str, ...]) -> frozenset[str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "change")
    return frozenset(
        _git(tmp_path, "diff", "--name-only", "--diff-filter=ACMRD", "HEAD~1", "HEAD", "--", *pathspecs).split()
    )


def test_workflow_still_carries_the_ruff_format_e2e_basedpyright_and_e2e_harness_diff_gates() -> None:
    assert len(GATES) == 3
    assert frozenset(gate[0] for gate in GATES) == frozenset(
        {":(glob)litellm/**/*.py", ":(glob)tests/e2e/**/*.py", "tests/e2e"}
    )


@pytest.mark.parametrize("pathspecs", PYTHON_GATES, ids=" ".join)
def test_diff_gate_selects_top_level_and_nested_python_files_only(tmp_path: Path, pathspecs: tuple[str, ...]) -> None:
    root = _scoped_root(pathspecs[0])
    top_level = f"{root}top_level_module.py"
    nested = f"{root}pkg/sub/nested_module.py"
    selected = _changed_files_selected_by(
        tmp_path,
        pathspecs,
        (top_level, nested, f"{root}notes.md", "elsewhere/top_level_module.py", "elsewhere/pkg/nested_module.py"),
    )
    assert selected == frozenset({top_level, nested})


@pytest.mark.parametrize(
    "trigger",
    (
        "tests/e2e_harness/top_level_module.py",
        "tests/e2e_harness/pkg/sub/nested_module.py",
        "pyrightconfig.json",
    ),
)
def test_e2e_basedpyright_gate_also_fires_on_harness_python_and_pyrightconfig(tmp_path: Path, trigger: str) -> None:
    selected = _changed_files_selected_by(
        tmp_path,
        _gate_rooted_at("tests/e2e/"),
        (trigger, "tests/e2e_harness/notes.md", "elsewhere/pyrightconfig.json", "tests/e2e_harnessish/module.py"),
    )
    assert selected == frozenset({trigger})


@pytest.mark.parametrize(
    "trigger",
    (
        "tests/e2e/claude_code/cron_vm/install_claude_code.sh",
        "tests/e2e/notes.md",
        "tests/e2e/pkg/sub/nested_module.py",
        "tests/e2e_harness/claude_code/test_driver.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/test-linting.yml",
    ),
)
def test_e2e_harness_gate_fires_on_any_e2e_or_harness_file_its_dependency_manifests_and_workflow(
    tmp_path: Path, trigger: str
) -> None:
    selected = _changed_files_selected_by(
        tmp_path,
        _gate_rooted_at("tests/e2e"),
        (trigger, "tests/e2e/ui/spec.ts", "tests/e2e/ui/pkg/page.py", "elsewhere/pyproject.toml", "tests/e2e_other/module.py"),
    )
    assert selected == frozenset({trigger})
