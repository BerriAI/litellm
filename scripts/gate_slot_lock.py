#!/usr/bin/env python3
"""Machine-wide slot lock for this repo's heavy entrypoints.

`make check`, `make lint`, and the standalone budget gates
(scripts/ruff_strict_gate.py, scripts/type_discipline_gate.py,
scripts/type_check_gate.py) each hold one of N machine-wide slots while they
run, so however many sessions and worktrees share one machine, at most N of
them execute a basedpyright/pytest/prettier storm at a time instead of all
thrashing it at once. Slots are fcntl.flock files (macOS ships no flock(1)
binary, hence python3 + stdlib only, runnable before any venv exists) under a
per-user cache directory shared by every worktree and session:
~/.cache/litellm/gate-slots by default, $LITELLM_GATE_SLOT_DIR to override.
A holder's lock dies with its process, so a crash leaves nothing to clean up.

$LITELLM_GATE_SLOTS sets the slot count (default 2); 0 disables locking.
Waiting is a blocking flock on a turnstile file plus a slow poll of the slots,
so contenders queue roughly first-come-first-served without busy-spinning.
A process that acquired (or deliberately skipped) a slot exports
LITELLM_GATE_SLOT_HELD, and nested acquisitions under that marker are no-ops,
so `make check` invoking the gates internally can never deadlock against
itself. Any filesystem error fails open and the command runs unlocked: the
lock is a courtesy to the machine, never a gate that may break a build (CI
runs one job per machine, so there it only ever takes the instant path).

CLI: python3 scripts/gate_slot_lock.py <command> [args...]
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator

HELD_MARKER_ENV: Final = "LITELLM_GATE_SLOT_HELD"
SLOT_COUNT_ENV: Final = "LITELLM_GATE_SLOTS"
SLOT_DIR_ENV: Final = "LITELLM_GATE_SLOT_DIR"
DEFAULT_SLOT_COUNT: Final = 2
POLL_SECONDS: Final = 2.0


def _slot_dir() -> Path:
    override: Final = os.environ.get(SLOT_DIR_ENV)
    return Path(override) if override else Path.home() / ".cache" / "litellm" / "gate-slots"


def _slot_count() -> int:
    raw: Final = os.environ.get(SLOT_COUNT_ENV)
    if not raw:
        return DEFAULT_SLOT_COUNT
    try:
        return int(raw)
    except ValueError:
        print(
            f"gate_slot_lock: ignoring non-integer {SLOT_COUNT_ENV}={raw!r}; "
            f"using {DEFAULT_SLOT_COUNT} slots",
            file=sys.stderr,
        )
        return DEFAULT_SLOT_COUNT


def _try_slot(directory: Path, index: int) -> IO[bytes] | None:
    handle: Final = (directory / f"slot-{index}.lock").open("wb")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except OSError:
        handle.close()
        raise
    return handle


def _wait_for_slot(directory: Path, count: int) -> IO[bytes]:
    print(
        f"gate_slot_lock: all {count} machine-wide slots are busy; queueing "
        f"(set {SLOT_COUNT_ENV}=0 to disable)",
        file=sys.stderr,
        flush=True,
    )
    with (directory / "turnstile.lock").open("wb") as turnstile:
        fcntl.flock(turnstile, fcntl.LOCK_EX)
        while True:
            for index in range(count):
                held = _try_slot(directory, index)
                if held is not None:
                    return held
            time.sleep(POLL_SECONDS)


def _locked_handle(count: int) -> IO[bytes]:
    directory: Final = _slot_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        immediate = _try_slot(directory, index)
        if immediate is not None:
            return immediate
    return _wait_for_slot(directory, count)


def acquire_slot() -> IO[bytes] | None:
    """Hold a machine-wide slot for the life of the returned handle.

    The caller must keep the handle referenced until the process exits;
    dropping it closes the file and releases the slot. Returns None without
    locking when this process already runs under a held slot, when locking is
    disabled, or when the filesystem refuses to cooperate."""
    if os.environ.get(HELD_MARKER_ENV):
        return None
    count: Final = _slot_count()
    if count <= 0:
        os.environ[HELD_MARKER_ENV] = "1"
        return None
    try:
        handle: Final = _locked_handle(count)
    except (OSError, RuntimeError) as error:
        print(f"gate_slot_lock: locking unavailable ({error}); running unlocked", file=sys.stderr)
        os.environ[HELD_MARKER_ENV] = "1"
        return None
    os.environ[HELD_MARKER_ENV] = "1"
    return handle


@contextlib.contextmanager
def held_slot() -> Iterator[None]:
    """Run the with-block while holding a machine-wide slot (or its no-op forms)."""
    prior_marker: Final = os.environ.get(HELD_MARKER_ENV)
    handle: Final = acquire_slot()
    try:
        yield
    finally:
        if handle is not None:
            handle.close()
        if not prior_marker:
            os.environ.pop(HELD_MARKER_ENV, None)


def _wait_ignoring_interrupts(process: subprocess.Popen[bytes]) -> int:
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: gate_slot_lock.py <command> [args...]", file=sys.stderr)
        return 2
    try:
        held: Final = acquire_slot()
    except KeyboardInterrupt:
        return 130
    try:
        code: Final = _wait_ignoring_interrupts(subprocess.Popen(sys.argv[1:]))
    except FileNotFoundError as error:
        print(f"gate_slot_lock: {error}", file=sys.stderr)
        return 127
    if held is not None:
        held.close()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
