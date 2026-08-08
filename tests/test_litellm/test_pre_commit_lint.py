import os
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
[ -n "${STUB_MAKE_LOG:-}" ] && echo "$*" >> "$STUB_MAKE_LOG"
case "$*" in
    bootstrap-dashboard)
        [ "${STUB_FAIL:-}" = "bootstrap-dashboard" ] && exit 1
        ;;
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


def _sandbox(tmp_path: Path, stage: tuple[str, ...] = (".",)) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "litellm" / "proxy").mkdir(parents=True)
    (repo / "litellm" / "foo.py").write_text("x = 1\n")
    (repo / "litellm" / "proxy" / "spec.py").write_text("y = 2\n")
    dashboard = repo / "ui" / "litellm-dashboard"
    (dashboard / "src").mkdir(parents=True)
    (dashboard / "node_modules").mkdir()
    (dashboard / "src" / "app.ts").write_text("export {}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", *stage], cwd=repo, check=True)

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


def _make_invocations(tmp_path: Path, repo: Path, bin_dir: Path) -> list[str]:
    log = tmp_path / "make.log"
    proc = _run(repo, bin_dir, {"STUB_MAKE_LOG": str(log)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return log.read_text().split()


def test_a_python_only_commit_never_provisions_the_dashboard_toolchain(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path, stage=("litellm/foo.py",))
    invocations = _make_invocations(tmp_path, repo, bin_dir)
    assert "lint" in invocations
    assert "bootstrap-dashboard" not in invocations


@pytest.mark.parametrize("staged", ["ui/litellm-dashboard/src/app.ts", "litellm/proxy/spec.py"])
def test_a_commit_that_reaches_the_dashboard_provisions_it(tmp_path: Path, staged: str) -> None:
    repo, bin_dir = _sandbox(tmp_path, stage=(staged,))
    assert "bootstrap-dashboard" in _make_invocations(tmp_path, repo, bin_dir)


def test_the_dashboard_is_provisioned_once_when_both_node_blocks_run(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    invocations = _make_invocations(tmp_path, repo, bin_dir)
    assert invocations.count("bootstrap-dashboard") == 1


def test_failed_dashboard_provisioning_skips_the_node_blocks_rather_than_running_them(tmp_path: Path) -> None:
    """A failed `make bootstrap-dashboard` must not fall through into the node blocks.

    They would run against a stale or absent node_modules and report a second,
    misleading failure ("format with npm run format") on top of the real cause,
    which is the one a reader needs. The Python block shares nothing with the node
    toolchain, so it still has to run."""
    repo, bin_dir = _sandbox(tmp_path)
    proc = _run(repo, bin_dir, {"STUB_FAIL": "bootstrap-dashboard"})
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 1
    assert "make bootstrap-dashboard failed" in combined
    assert "linting dashboard" not in combined
    assert "API types" not in combined
    assert "Dashboard lint failed" not in combined
    assert "linting Python" in combined


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
    assert f"pre-commit: full log: {log_file}" in proc.stdout
    assert "pre-commit: full log:" not in log


def test_unwritable_log_warns_and_falls_back_to_running_without_one(tmp_path: Path) -> None:
    repo, bin_dir = _sandbox(tmp_path)
    (repo / ".git" / "pre_commit_lint.log").mkdir()
    proc = _run(repo, bin_dir, {})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "linting Python" in proc.stdout
    assert "output will not be saved" in proc.stderr
    assert "pre-commit: full log:" not in proc.stdout
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
