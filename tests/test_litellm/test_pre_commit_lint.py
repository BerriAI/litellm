import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pre_commit_lint.sh"

BARRIER_HELPER = """barrier_sync() {
    touch "$STUB_BARRIER_DIR/$1.started"
    for other in $2; do
        tries=0
        while [ ! -f "$STUB_BARRIER_DIR/$other.started" ]; do
            tries=$((tries + 1))
            if [ "$tries" -gt 100 ]; then
                echo "barrier timeout: $1 never saw $other start" >&2
                exit 1
            fi
            sleep 0.1
        done
    done
}
"""

MAKE_STUB = """#!/bin/sh
. "$STUB_BIN/barrier.sh"
case "$*" in
    lint)
        [ "${STUB_FAIL:-}" = "make-lint" ] && exit 1
        [ -n "${STUB_BARRIER_DIR:-}" ] && barrier_sync python "dashboard genapi"
        if [ -n "${STUB_HANG_DIR:-}" ]; then
            echo "$$" > "$STUB_HANG_DIR/make.pid"
            touch "$STUB_HANG_DIR/make.started"
            sleep 60
        fi
        ;;
esac
exit 0
"""

NPX_STUB = """#!/bin/sh
. "$STUB_BIN/barrier.sh"
case "$*" in
    prettier*)
        [ -n "${STUB_BARRIER_DIR:-}" ] && barrier_sync dashboard "python genapi"
        ;;
    "eslint --no-warn-ignored"*)
        [ "${STUB_FAIL:-}" = "eslint" ] && exit 1
        ;;
esac
exit 0
"""

UV_STUB = """#!/bin/sh
. "$STUB_BIN/barrier.sh"
case "$*" in
    *orjson*)
        [ -n "${STUB_BARRIER_DIR:-}" ] && barrier_sync genapi "python dashboard"
        ;;
esac
exit 0
"""

NPM_STUB = """#!/bin/sh
case "$*" in
    "run gen:api")
        [ "${STUB_FAIL:-}" = "gen-api" ] && exit 1
        ;;
esac
exit 0
"""

NODE_STUB = """#!/bin/sh
exit 0
"""


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _sandbox(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "litellm" / "proxy").mkdir(parents=True)
    (repo / "litellm" / "foo.py").write_text("x = 1\n")
    (repo / "litellm" / "proxy" / "spec.py").write_text("y = 2\n")
    dashboard = repo / "ui" / "litellm-dashboard"
    (dashboard / "src").mkdir(parents=True)
    (dashboard / "node_modules").mkdir()
    (dashboard / "src" / "app.ts").write_text("export {}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "barrier.sh").write_text(BARRIER_HELPER)
    _write_executable(bin_dir / "make", MAKE_STUB)
    _write_executable(bin_dir / "npx", NPX_STUB)
    _write_executable(bin_dir / "uv", UV_STUB)
    _write_executable(bin_dir / "npm", NPM_STUB)
    _write_executable(bin_dir / "node", NODE_STUB)
    return repo, bin_dir


def _env(repo: Path, bin_dir: Path, extra_env: dict[str, str]) -> dict[str, str]:
    return {
        "PATH": os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]),
        "HOME": str(repo.parent),
        "STUB_BIN": str(bin_dir),
        **extra_env,
    }


def _run(repo: Path, bin_dir: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = _env(repo, bin_dir, extra_env)
    return subprocess.run(
        [str(SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _commit_all(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message],
        cwd=repo,
        check=True,
    )


def _set_base_ref(repo: Path) -> None:
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/litellm_internal_staging", "HEAD"],
        cwd=repo,
        check=True,
    )


def test_nothing_staged_scopes_to_working_tree_diff_and_runs_checks(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    _set_base_ref(repo)
    (repo / "litellm" / "foo.py").write_text("x = 2\n")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nothing staged; scoping to the working tree's diff" in proc.stdout
    assert "litellm/foo.py" in proc.stdout
    assert "linting Python" in proc.stdout


def test_nothing_staged_checks_committed_branch_changes(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    _set_base_ref(repo)
    (repo / "litellm" / "foo.py").write_text("x = 2\n")
    _commit_all(repo, "branch change")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nothing staged; scoping to the working tree's diff" in proc.stdout
    assert "linting Python" in proc.stdout


def test_nothing_staged_includes_untracked_files_in_scope(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    _set_base_ref(repo)
    (repo / "litellm" / "brand_new.py").write_text("z = 3\n")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "litellm/brand_new.py" in proc.stdout
    assert "linting Python" in proc.stdout


def test_nothing_staged_deletion_only_branch_triggers_checks(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    _set_base_ref(repo)
    (repo / "litellm" / "foo.py").unlink()
    _commit_all(repo, "delete module")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nothing to check" not in proc.stdout
    assert "litellm/foo.py" in proc.stdout
    assert "linting Python" in proc.stdout
    assert "ruff format --check" not in proc.stdout


def test_staged_deletion_triggers_checks_without_feeding_missing_files_to_tools(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    subprocess.run(["git", "rm", "-q", "litellm/foo.py"], cwd=repo, check=True)
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nothing staged" not in proc.stdout
    assert "linting Python" in proc.stdout
    assert "ruff format --check" not in proc.stdout


def test_deleted_dashboard_file_still_triggers_dashboard_lint(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    _set_base_ref(repo)
    (repo / "ui" / "litellm-dashboard" / "src" / "app.ts").unlink()
    _commit_all(repo, "delete dashboard file")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "linting dashboard" in proc.stdout


def test_nothing_staged_and_no_changes_is_an_explicit_no_op(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    _set_base_ref(repo)
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "nothing to check" in proc.stdout
    assert "check: PASS" in proc.stdout
    assert "linting Python" not in proc.stdout


def test_nothing_staged_without_a_base_ref_fails_with_a_fetch_hint(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 1
    assert "cannot resolve the merge base" in proc.stdout
    assert "git fetch origin litellm_internal_staging" in proc.stdout
    assert "check: FAIL" in proc.stdout


def test_partial_staging_warns_which_checks_were_skipped(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    (repo / "notes.md").write_text("hi\n")
    subprocess.run(["git", "add", "notes.md"], cwd=repo, check=True)
    (repo / "litellm" / "foo.py").write_text("x = 4\n")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SKIPPED Python lint (make lint)" in proc.stdout
    assert "litellm/foo.py" in proc.stdout
    assert "linting Python" not in proc.stdout


def test_python_dashboard_and_gen_api_blocks_run_concurrently_with_grouped_output(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    proc = _run(repo, bin_dir, {"STUB_BARRIER_DIR": str(barrier_dir)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "barrier timeout" not in proc.stdout + proc.stderr
    python_at = proc.stdout.index("linting Python")
    dashboard_at = proc.stdout.index("linting dashboard")
    gen_api_at = proc.stdout.index("API types")
    assert python_at < dashboard_at < gen_api_at


def test_all_blocks_passing_exits_zero(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_full_output_is_saved_to_a_log_file_in_the_git_dir(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    (repo / "scratch.txt").write_text("")
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log_file = repo / ".git" / "pre_commit_lint.log"
    log = log_file.read_text()
    assert "linting Python" in log
    assert "linting dashboard" in log
    assert "API types" in log
    assert "unstaged/untracked changes" in log
    assert f"check: full log: {log_file}" in proc.stdout
    assert "check: full log:" not in log


def test_unwritable_log_warns_and_falls_back_to_running_without_one(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    (repo / ".git" / "pre_commit_lint.log").mkdir()
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "linting Python" in proc.stdout
    assert "output will not be saved" in proc.stderr
    assert "check: full log:" not in proc.stdout
    failing = _run(repo, bin_dir, {"STUB_FAIL": "make-lint"})
    assert failing.returncode == 1


def test_failing_run_exit_code_survives_the_log_pipeline(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    proc = _run(repo, bin_dir, {"STUB_FAIL": "make-lint"})
    assert proc.returncode == 1
    log = (repo / ".git" / "pre_commit_lint.log").read_text()
    assert "Python lint failed" in log


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_interrupt_kills_background_jobs_and_removes_logs(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    hang_dir = tmp_path / "hang"
    hang_dir.mkdir()
    tmp_dir = tmp_path / "tmpdir"
    tmp_dir.mkdir()
    extra = {"STUB_HANG_DIR": str(hang_dir), "TMPDIR": str(tmp_dir)}
    proc = subprocess.Popen(
        [str(SCRIPT)],
        cwd=repo,
        env=_env(repo, bin_dir, extra),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert _wait_until((hang_dir / "make.started").exists, 10)
        os.killpg(proc.pid, signal.SIGINT)
        assert proc.wait(timeout=10) != 0
        make_pid = int((hang_dir / "make.pid").read_text())
        assert _wait_until(lambda: _pid_gone(make_pid), 5)
        assert list(tmp_dir.iterdir()) == []
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)


def test_interrupt_spares_the_invoking_process(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    hang_dir = tmp_path / "hang"
    hang_dir.mkdir()
    marker = tmp_path / "invoker_survived"
    proc = subprocess.Popen(
        ["bash", "-c", 'trap : INT; "$1"; echo "$?" > "$2"', "bash", str(SCRIPT), str(marker)],
        cwd=repo,
        env=_env(repo, bin_dir, {"STUB_HANG_DIR": str(hang_dir)}),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert _wait_until((hang_dir / "make.started").exists, 10)
        os.killpg(proc.pid, signal.SIGINT)
        assert proc.wait(timeout=10) == 0
        assert _wait_until(marker.exists, 5)
        assert marker.read_text().strip() == "130"
    finally:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)


@pytest.mark.parametrize(
    ("fail", "message"),
    [
        ("make-lint", "Python lint failed"),
        ("eslint", "Dashboard lint failed"),
        ("gen-api", "npm run gen:api failed"),
    ],
)
def test_a_failing_block_fails_the_whole_run(tmp_path: Path, fail: str, message: str) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    proc = _run(repo, bin_dir, {"STUB_FAIL": fail})
    assert proc.returncode == 1
    assert message in proc.stdout + proc.stderr


def test_run_ends_with_a_summary_of_ran_and_skipped_blocks(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "check: summary" in proc.stdout
    assert "ran:     Python lint (make lint)" in proc.stdout
    assert "ran:     dashboard lint (prettier + eslint + lint budgets)" in proc.stdout
    assert "ran:     dashboard API-type sync (npm run gen:api)" in proc.stdout
    assert "skipped: tests/e2e checks (basedpyright + raw HTTP client ban) (no tests/e2e Python files in scope)" in proc.stdout
    assert "check: PASS" in proc.stdout
    assert "check: FAIL" not in proc.stdout


def test_staged_files_matching_no_check_print_an_explicit_noop_note_and_nonempty_log(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    _commit_all(repo, "base")
    tests_dir = repo / "tests" / "test_litellm"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_x.py").write_text("def test_x() -> None: ...\n")
    subprocess.run(["git", "add", "tests"], cwd=repo, check=True)
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no gating lint check matches the files in scope, so nothing ran" in proc.stdout
    assert "tests/test_litellm/test_x.py" in proc.stdout
    assert "a no-op, not a lint verdict" in proc.stdout
    assert "check: PASS" in proc.stdout
    assert "linting Python" not in proc.stdout
    log = (repo / ".git" / "pre_commit_lint.log").read_text()
    assert "check: summary" in log
    assert "skipped: Python lint (make lint) (no litellm/ Python files in scope)" in log


def test_run_queues_through_the_machine_wide_gate_slot_lock(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    lock_dir = tmp_path / "gate-locks"
    proc = _run(repo, bin_dir, {"LITELLM_GATE_SLOT_DIR": str(lock_dir)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (lock_dir / "slot-0.lock").exists()


def test_run_under_a_held_slot_skips_reacquiring_the_gate_lock(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    lock_dir = tmp_path / "gate-locks"
    proc = _run(
        repo,
        bin_dir,
        {"LITELLM_GATE_SLOT_DIR": str(lock_dir), "LITELLM_GATE_SLOT_HELD": "1"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not lock_dir.exists()


def test_hook_symlink_install_still_resolves_the_slot_lock_helper(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir()
    shutil.copy(SCRIPT, scripts_dir / "pre_commit_lint.sh")
    shutil.copy(SCRIPT.parent / "gate_slot_lock.py", scripts_dir / "gate_slot_lock.py")
    (repo / ".git" / "hooks" / "pre-commit").symlink_to(Path("../../scripts/pre_commit_lint.sh"))
    lock_dir = tmp_path / "gate-locks"
    proc = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "hooked"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_env(repo, bin_dir, {"LITELLM_GATE_SLOT_DIR": str(lock_dir)}),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (lock_dir / "slot-0.lock").exists()


def test_failing_run_ends_with_a_fail_verdict(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    proc = _run(repo, bin_dir, {"STUB_FAIL": "make-lint"})
    assert proc.returncode == 1
    assert "check: FAIL" in proc.stdout
    assert "check: PASS" not in proc.stdout
