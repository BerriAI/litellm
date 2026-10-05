"""Cross-process behavior of the stack lock: readers share it, an exclusive holder waits for
every reader and keeps them out, and a reader arriving behind a waiting exclusive holder
queues behind it instead of starving it."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest

from stack_lock import STACK_DIGEST

HARNESS_DIR: Final = Path(__file__).resolve().parent
DEADLINE_SECONDS: Final = 30.0
SETTLE_SECONDS: Final = 0.5
HOLDER_SCRIPT: Final = """
import sys, time
from pathlib import Path
from stack_lock import stack_lock
name, mode, release_path, log_path = sys.argv[1:]


def record(event):
    with Path(log_path).open("a") as log:
        log.write(f"{name} {event}\\n")


record("waiting")
with stack_lock(exclusive=mode == "exclusive"):
    record("enter")
    while not Path(release_path).exists():
        time.sleep(0.02)
    record("exit")
"""


def _events(log_path: Path) -> tuple[str, ...]:
    return tuple(log_path.read_text().splitlines()) if log_path.exists() else ()


def _wait_for_event(log_path: Path, event: str) -> None:
    deadline: Final = time.monotonic() + DEADLINE_SECONDS
    while event not in _events(log_path):
        if time.monotonic() > deadline:
            pytest.fail(f"{event!r} never appeared; events so far: {_events(log_path)}")
        time.sleep(0.02)


def _wait_until_gate_is_held_exclusively(gate_path: Path) -> None:
    deadline: Final = time.monotonic() + DEADLINE_SECONDS
    with gate_path.open("a") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            fcntl.flock(handle, fcntl.LOCK_UN)
            if time.monotonic() > deadline:
                pytest.fail("no exclusive holder ever took the gate")
            time.sleep(0.02)


def _start_holder(held: ExitStack, tmp_path: Path, name: str, mode: str) -> subprocess.Popen[bytes]:
    holder: Final = held.enter_context(
        subprocess.Popen(
            (
                sys.executable,
                "-P",
                "-c",
                HOLDER_SCRIPT,
                name,
                mode,
                str(tmp_path / f"release-{name}"),
                str(tmp_path / "events"),
            ),
            cwd=HARNESS_DIR,
            env={**os.environ, "TMPDIR": str(tmp_path), "PYTHONPATH": str(HARNESS_DIR)},
        )
    )
    held.callback(holder.kill)
    return holder


def test_readers_share_exclusive_waits_and_a_waiting_exclusive_beats_later_readers(tmp_path: Path) -> None:
    lock_dir: Final = tmp_path / f"litellm-e2e-stack-{STACK_DIGEST}"
    lock_dir.mkdir()
    log_path: Final = tmp_path / "events"
    with ExitStack() as held:
        first_reader: Final = _start_holder(held, tmp_path, "A", "shared")
        _wait_for_event(log_path, "A enter")
        second_reader: Final = _start_holder(held, tmp_path, "R", "shared")
        _wait_for_event(log_path, "R enter")
        (tmp_path / "release-R").touch()
        _wait_for_event(log_path, "R exit")
        writer: Final = _start_holder(held, tmp_path, "W", "exclusive")
        _wait_until_gate_is_held_exclusively(lock_dir / "gate")
        late_reader: Final = _start_holder(held, tmp_path, "B", "shared")
        _wait_for_event(log_path, "B waiting")
        time.sleep(SETTLE_SECONDS)
        (tmp_path / "release-A").touch()
        _wait_for_event(log_path, "W enter")
        (tmp_path / "release-W").touch()
        _wait_for_event(log_path, "B enter")
        (tmp_path / "release-B").touch()
        for holder in (first_reader, second_reader, writer, late_reader):
            assert holder.wait(timeout=DEADLINE_SECONDS) == 0
    events: Final = _events(log_path)
    assert events.index("R enter") < events.index("A exit")
    assert events.index("W enter") > events.index("A exit")
    assert events.index("B enter") > events.index("W exit")
