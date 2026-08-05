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
