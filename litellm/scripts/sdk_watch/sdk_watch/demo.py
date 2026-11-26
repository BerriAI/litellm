import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final

from sdk_watch.implement import pytest_command, test_env
from sdk_watch.report import DemoVerdict, PrDraft


def run_demo(
    repo: Path, patch: Path, draft: PrDraft, python: Path, echo: Callable[[str], None], timeout_seconds: float
) -> DemoVerdict:
    command: Final = (*pytest_command(python, draft.test_files), "--tb=line")
    shown: Final = shlex.join(("python", *command[1:]))
    echo(f"# {draft.title}")
    echo(f"# {draft.sdk} {draft.version}: {draft.sdk_change}")
    echo("")
    echo("# Before: the PR's tests on main, without the fix")
    _apply(repo, "--include=tests/*", str(patch))
    echo(f"$ {shown}")
    passed_before: Final = _passes(repo, command, echo, timeout_seconds)
    _apply(repo, "--reverse", "--include=tests/*", str(patch))
    echo("")
    echo("# After: the same tests with the fix")
    _apply(repo, str(patch))
    echo(f"$ {shown}")
    passed_after: Final = _passes(repo, command, echo, timeout_seconds)
    _apply(repo, "--reverse", str(patch))
    verdict: Final = DemoVerdict(fails_before=not passed_before, passes_after=passed_after)
    echo("")
    echo(
        f"# Before: {'failed' if verdict.fails_before else 'passed'}. After: {'passed' if verdict.passes_after else 'failed'}"
    )
    return verdict


def _passes(repo: Path, command: tuple[str, ...], echo: Callable[[str], None], timeout_seconds: float) -> bool:
    try:
        completed: Final = subprocess.run(command, cwd=repo, env=test_env(), check=False, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        echo(f"# Timed out after {timeout_seconds:g} s")
        return False
    return completed.returncode == 0


def _apply(repo: Path, *args: str) -> None:
    subprocess.run(("git", "-C", str(repo), "apply", *args), check=True)
