from typing import Final

import pytest

import code_quality_checks as runner
from code_quality_checks import Check, Outcome

PASSES: Final = Check(name="passes", run="true")
BREAKS: Final = Check(name="breaks", run="false")
LINUX_ONLY: Final = Check(name="linux_only", run="true", ci_only="reads /proc/net/tcp")


def _scripted(check: Check) -> Outcome:
    failed: Final = check is BREAKS
    return Outcome(check=check, returncode=3 if failed else 0, output="boom\n" if failed else "fine\n", seconds=0.0)


def test_a_local_run_skips_ci_only_checks_and_fails_naming_the_broken_one(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code: Final = runner.run_checks((PASSES, BREAKS, LINUX_ONLY), local=True, annotate=False, run_one=_scripted)
    out: Final = capsys.readouterr().out
    assert exit_code == 1
    assert "SKIP linux_only (CI-only: reads /proc/net/tcp)" in out
    assert "linux_only (0.0s)" not in out
    assert "PASS passes (0.0s)" in out
    assert "fine" not in out
    assert "FAIL breaks (0.0s, exit 3)\nboom" in out
    assert "code-quality: 1 passed, 1 failed, 1 skipped as CI-only" in out
    assert out.rstrip().endswith("code-quality: FAIL breaks")


def test_a_ci_run_executes_ci_only_checks_and_passes_when_every_check_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code: Final = runner.run_checks((PASSES, LINUX_ONLY), local=False, annotate=False, run_one=_scripted)
    out: Final = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS linux_only (0.0s)" in out
    assert "SKIP" not in out
    assert out.rstrip().endswith("code-quality: PASS")


def test_github_annotations_fold_passing_output_and_flag_each_failure(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code: Final = runner.run_checks((PASSES, BREAKS), local=False, annotate=True, run_one=_scripted)
    out: Final = capsys.readouterr().out
    assert exit_code == 1
    assert "::group::PASS passes (0.0s)\nfine\n::endgroup::" in out
    assert "::error title=code-quality::breaks failed with exit code 3" in out


def test_execute_runs_the_step_like_github_actions_with_its_env_and_errexit() -> None:
    passing: Final = runner.execute(Check(name="env", run='test "$PROBE" = set\necho seen', env={"PROBE": "set"}))
    failing: Final = runner.execute(Check(name="errexit", run="false\necho unreachable"))
    assert (passing.returncode, passing.output) == (0, "seen\n")
    assert failing.returncode == 1
    assert "unreachable" not in failing.output
