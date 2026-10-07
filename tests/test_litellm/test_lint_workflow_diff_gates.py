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


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _scoped_root(pathspec: str) -> str:
    return re.sub(r"^:\([^)]*\)", "", pathspec).split("*", 1)[0]


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


def test_workflow_still_carries_the_ruff_format_and_e2e_basedpyright_diff_gates() -> None:
    assert frozenset(_scoped_root(gate[0]) for gate in GATES) == frozenset({"litellm/", "tests/e2e/"})


@pytest.mark.parametrize("pathspecs", GATES, ids=" ".join)
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
