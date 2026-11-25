import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

RUNNER: Final = Path(__file__).parents[2] / ".github" / "scripts" / "ci" / "run_parallel_checks.py"
WAIT_FOR_PARTNER: Final = "touch {mine}; for _ in $(seq 40); do [ -e {theirs} ] && exit 0; sleep 0.05; done; exit 1"


def _run(root: Path, commands: tuple[str, ...], jobs: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(RUNNER), "--jobs", str(jobs)],
        input="\n".join(commands) + "\n",
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_every_check_passing_passes_the_run(tmp_path: Path) -> None:
    proc: Final = _run(tmp_path, ("true", "true"), jobs=2)

    assert proc.returncode == 0, proc.stdout
    assert "2 of 2 checks passed." in proc.stdout


def test_a_failing_check_fails_the_run_and_the_checks_after_it_still_run(tmp_path: Path) -> None:
    proc: Final = _run(tmp_path, ("exit 3", "touch ran-after-the-failure"), jobs=1)

    assert proc.returncode == 1
    assert (tmp_path / "ran-after-the-failure").exists()
    assert "Failed: exit 3" in proc.stdout


def test_a_failing_checks_stderr_is_shown_in_the_log(tmp_path: Path) -> None:
    proc: Final = _run(tmp_path, ("printf 'the-%s\\n' reason-it-failed >&2; exit 1",), jobs=1)

    assert proc.returncode == 1
    assert "the-reason-it-failed" in proc.stdout


def test_no_checks_given_fails_the_run(tmp_path: Path) -> None:
    proc: Final = _run(tmp_path, ("", "   "), jobs=4)

    assert proc.returncode == 1
    assert "No checks were given" in proc.stdout


@pytest.mark.parametrize(("jobs", "returncode"), ((2, 0), (1, 1)))
def test_checks_run_at_the_same_time_up_to_the_job_limit(tmp_path: Path, jobs: int, returncode: int) -> None:
    proc: Final = _run(
        tmp_path,
        (WAIT_FOR_PARTNER.format(mine="a", theirs="b"), WAIT_FOR_PARTNER.format(mine="b", theirs="a")),
        jobs=jobs,
    )

    assert proc.returncode == returncode, proc.stdout
