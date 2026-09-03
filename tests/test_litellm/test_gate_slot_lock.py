import fcntl
import importlib.util
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "gate_slot_lock.py"

_spec = importlib.util.spec_from_file_location("gate_slot_lock", HELPER)
assert _spec is not None and _spec.loader is not None
gate_slot_lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate_slot_lock)

START_THEN_WAIT_FOR = (
    "import pathlib, sys, time\n"
    "pathlib.Path(sys.argv[1]).touch()\n"
    "deadline = time.monotonic() + 20\n"
    "while not pathlib.Path(sys.argv[2]).exists():\n"
    "    if time.monotonic() > deadline:\n"
    "        sys.exit(3)\n"
    "    time.sleep(0.05)\n"
)

TOUCH_TARGET = "import pathlib, sys\npathlib.Path(sys.argv[1]).touch()\n"

RECORD_INTERVAL = (
    "import sys, time\n"
    "with open(sys.argv[1], 'a') as events:\n"
    "    events.write(f'start {time.monotonic()}\\n')\n"
    "    events.flush()\n"
    "    time.sleep(0.6)\n"
    "    events.write(f'end {time.monotonic()}\\n')\n"
    "    events.flush()\n"
)


def _env(lock_dir: Path, slots: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(lock_dir.parent),
        "LITELLM_GATE_SLOT_DIR": str(lock_dir),
        "LITELLM_GATE_SLOTS": slots,
    }


def _wrapped(payload: Sequence[str]) -> list[str]:
    return [sys.executable, str(HELPER), sys.executable, "-c", *payload]


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def _reap(process: subprocess.Popen[bytes]) -> None:
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)


def test_six_contenders_never_exceed_two_slots_and_all_complete(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    events_file = tmp_path / "events.log"
    env = _env(lock_dir, "2")
    procs = [
        subprocess.Popen(
            _wrapped([RECORD_INTERVAL, str(events_file)]),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(6)
    ]
    try:
        assert [proc.wait(timeout=60) for proc in procs] == [0] * 6
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    events = sorted(
        (float(stamp), 1 if kind == "start" else -1)
        for kind, stamp in (line.split() for line in events_file.read_text().splitlines())
    )
    assert len(events) == 12
    concurrency_peaks = []
    running = 0
    for _, delta in events:
        running += delta
        concurrency_peaks.append(running)
    assert max(concurrency_peaks) <= 2


def test_two_slots_admit_two_holders_at_once(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    first_started = tmp_path / "first.started"
    second_started = tmp_path / "second.started"
    env = _env(lock_dir, "2")
    first = subprocess.Popen(_wrapped([START_THEN_WAIT_FOR, str(first_started), str(second_started)]), env=env)
    second = subprocess.Popen(_wrapped([START_THEN_WAIT_FOR, str(second_started), str(first_started)]), env=env)
    assert first.wait(timeout=30) == 0
    assert second.wait(timeout=30) == 0


def test_contender_beyond_capacity_queues_until_the_slot_frees(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    holder_started = tmp_path / "holder.started"
    release = tmp_path / "release"
    done = tmp_path / "done"
    env = _env(lock_dir, "1")
    holder = subprocess.Popen(_wrapped([START_THEN_WAIT_FOR, str(holder_started), str(release)]), env=env)
    try:
        assert _wait_until(holder_started.exists, 10)
        contender = subprocess.Popen(
            _wrapped([TOUCH_TARGET, str(done)]),
            env=env,
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(1.5)
            assert not done.exists()
            release.touch()
            assert holder.wait(timeout=10) == 0
            assert contender.wait(timeout=30) == 0
            assert done.exists()
            assert contender.stderr is not None
            assert b"queueing" in contender.stderr.read()
        finally:
            release.touch()
            _reap(contender)
    finally:
        release.touch()
        _reap(holder)


def test_nested_wrapping_reenters_instead_of_deadlocking(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    nested = [
        sys.executable,
        str(HELPER),
        sys.executable,
        str(HELPER),
        sys.executable,
        "-c",
        "print('nested ok')",
    ]
    proc = subprocess.Popen(
        nested,
        env=_env(lock_dir, "1"),
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        _terminate_group(proc)
        pytest.fail("nested gate_slot_lock invocations deadlocked")
    assert proc.returncode == 0
    assert b"nested ok" in stdout


def test_wrapped_command_exit_code_is_propagated(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(HELPER), sys.executable, "-c", "raise SystemExit(7)"],
        env=_env(tmp_path / "locks", "2"),
    )
    assert proc.returncode == 7


def test_missing_command_exits_127_and_no_command_exits_2(tmp_path: Path) -> None:
    env = _env(tmp_path / "locks", "2")
    missing = subprocess.run(
        [sys.executable, str(HELPER), str(tmp_path / "no-such-binary")],
        env=env,
        capture_output=True,
    )
    assert missing.returncode == 127
    bare = subprocess.run([sys.executable, str(HELPER)], env=env, capture_output=True)
    assert bare.returncode == 2


def test_wrapped_command_killed_by_signal_maps_to_128_plus_signal(tmp_path: Path) -> None:
    proc = subprocess.run(
        _wrapped(["import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n"]),
        env=_env(tmp_path / "locks", "2"),
    )
    assert proc.returncode == 128 + signal.SIGTERM


def test_unusable_lock_dir_fails_open_and_still_runs_the_command(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    done = tmp_path / "done"
    proc = subprocess.run(
        _wrapped([TOUCH_TARGET, str(done)]),
        env=_env(blocker / "locks", "2"),
        capture_output=True,
    )
    assert proc.returncode == 0
    assert done.exists()
    assert b"running unlocked" in proc.stderr


def test_zero_slots_disables_locking_entirely(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    done = tmp_path / "done"
    proc = subprocess.run(
        _wrapped([TOUCH_TARGET, str(done)]),
        env=_env(lock_dir, "0"),
    )
    assert proc.returncode == 0
    assert done.exists()
    assert not lock_dir.exists()


def test_non_integer_slot_count_warns_and_falls_back_to_default(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(HELPER), sys.executable, "-c", "print('ran')"],
        env=_env(tmp_path / "locks", "lots"),
        capture_output=True,
    )
    assert proc.returncode == 0
    assert b"ran" in proc.stdout
    assert b"LITELLM_GATE_SLOTS" in proc.stderr


def test_killed_holder_releases_its_slot_for_the_next_contender(tmp_path: Path) -> None:
    lock_dir = tmp_path / "locks"
    holder_started = tmp_path / "holder.started"
    never = tmp_path / "never"
    env = _env(lock_dir, "1")
    holder = subprocess.Popen(
        _wrapped([START_THEN_WAIT_FOR, str(holder_started), str(never)]),
        env=env,
        start_new_session=True,
    )
    try:
        assert _wait_until(holder_started.exists, 10)
    finally:
        _terminate_group(holder)
    holder.wait(timeout=10)
    after = subprocess.run(
        [sys.executable, str(HELPER), sys.executable, "-c", "print('freed')"],
        env=env,
        capture_output=True,
        timeout=20,
    )
    assert after.returncode == 0
    assert b"freed" in after.stdout


def test_acquire_slot_holds_marks_and_releases_in_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("LITELLM_GATE_SLOT_HELD", "")
    monkeypatch.setenv("LITELLM_GATE_SLOT_DIR", str(lock_dir))
    monkeypatch.setenv("LITELLM_GATE_SLOTS", "1")
    handle = gate_slot_lock.acquire_slot()
    assert handle is not None
    assert os.environ["LITELLM_GATE_SLOT_HELD"] == "1"
    assert gate_slot_lock.acquire_slot() is None
    with (lock_dir / "slot-0.lock").open("wb") as probe:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.close()
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)


def test_held_slot_context_manager_releases_on_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("LITELLM_GATE_SLOT_HELD", "")
    monkeypatch.setenv("LITELLM_GATE_SLOT_DIR", str(lock_dir))
    monkeypatch.setenv("LITELLM_GATE_SLOTS", "1")
    with gate_slot_lock.held_slot():
        assert os.environ["LITELLM_GATE_SLOT_HELD"] == "1"
        with (lock_dir / "slot-0.lock").open("wb") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not os.environ.get("LITELLM_GATE_SLOT_HELD")
    with (lock_dir / "slot-0.lock").open("wb") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)


def _make_rule(target: str) -> tuple[list[str], list[str]]:
    database = subprocess.run(
        ["make", "--dry-run", "--print-data-base", "info"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = database.splitlines()
    for index, line in enumerate(lines):
        if line != f"{target}:" and not line.startswith(f"{target}: "):
            continue
        recipe: list[str] = []
        for follower in lines[index + 1 :]:
            if follower.startswith("#"):
                continue
            if not follower.startswith("\t"):
                break
            recipe.append(follower.strip())
        return line.split(":", 1)[1].split(), recipe
    raise AssertionError(f"target {target} not found in make database")


def test_direct_make_lint_takes_a_slot_before_any_setup() -> None:
    lint_prerequisites, lint_recipe = _make_rule("lint")
    assert lint_prerequisites == []
    assert any("$(GATE_SLOT_LOCK)" in line for line in lint_recipe)
    inner_prerequisites, _ = _make_rule("lint-inner")
    assert "lint-install" in inner_prerequisites
    assert "lint-fetch-base" in inner_prerequisites
