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


def _harness_gate() -> tuple[str, ...]:
    return next(gate for gate in GATES if gate[0] == "tests/e2e")


def _changed_files_selected_by(tmp_path: Path, pathspecs: tuple[str, ...], files: tuple[str, ...]) -> frozenset[str]:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    for name in files:
        target: Final = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "change")
    return frozenset(
        _git(tmp_path, "diff", "--name-only", "--diff-filter=ACMRD", "HEAD~1", "HEAD", "--", *pathspecs).split()
    )


@pytest.mark.parametrize("root", ("litellm/", "tests/e2e/", "tests/e2e_harness/"))
def test_python_diff_gate_selects_top_level_and_nested_python_files_only(tmp_path: Path, root: str) -> None:
    pathspecs: Final = next(gate for gate in GATES if f":(glob){root}**/*.py" in gate)
    top_level: Final = f"{root}top_level_module.py"
    nested: Final = f"{root}pkg/sub/nested_module.py"
    selected: Final = _changed_files_selected_by(
        tmp_path,
        pathspecs,
        (top_level, nested, f"{root}notes.md", "elsewhere/top_level_module.py", "elsewhere/pkg/nested_module.py"),
    )
    assert selected == frozenset({top_level, nested})


@pytest.mark.parametrize(
    "trigger",
    (
        "tests/e2e/claude_code/cron_vm/install_claude_code.sh",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/test-linting.yml",
    ),
)
def test_harness_gate_fires_on_its_installer_dependency_manifests_and_workflow(tmp_path: Path, trigger: str) -> None:
    selected: Final = _changed_files_selected_by(
        tmp_path, _harness_gate(), (trigger, "elsewhere/pyproject.toml", "tests/e2e/ui/notes.md")
    )
    assert selected == frozenset({trigger})


@pytest.mark.parametrize(
    "changed",
    (
        "tests/e2e/test_endpoint.py",
        "tests/e2e/claude_code/cron_vm/settings.json",
        "tests/e2e_harness/test_cli.py",
        "tests/e2e_harness/nested/fixture.txt",
    ),
)
def test_harness_gate_selects_harness_files_and_excludes_ui(tmp_path: Path, changed: str) -> None:
    selected: Final = _changed_files_selected_by(
        tmp_path, _harness_gate(), (changed, "tests/e2e/ui/test_ui.py", "elsewhere/test_endpoint.py")
    )
    assert selected == frozenset({changed})
