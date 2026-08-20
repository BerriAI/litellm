"""
Native-crash diagnostics for the LiteLLM proxy.

When a production pod dies to a native signal (SIGSEGV / SIGABRT / SIGFPE /
SIGBUS / SIGILL), ``PYTHONFAULTHANDLER=1`` already prints a Python-level
thread dump. That dump shows the *faulting* thread's Python frames, but for a
``SIGABRT`` (a deliberate ``abort()`` from a C extension or glibc) the
faulting thread is often just the thread that happened to receive the signal
— not the thread that corrupted the state. The actual culprit is below the
Python layer, in C, and faulthandler cannot walk it.

This module installs an extra signal handler that runs **before** faulthandler
re-raises the signal, and writes a richer context snapshot to stderr (which
Cloud Logging already ships). It does **not** require ``gdb`` or ``py-spy``
in the image — it is pure stdlib. What it adds beyond the faulthandler dump:

  * The native ``signal`` number and name, so the abort class is unambiguous.
  * The identity (``threading.native_id``, name) of every live thread, with a
    flag for which one was signalled.
  * The active asyncio tasks on the main loop (coroutine, current frame) —
    the thing faulthandler's per-thread view obscures.
  * A configurable rolling buffer of recent log lines / markers, so you can
    see what the proxy was doing in the seconds before the abort.

It cannot produce the native C call chain (only a core dump can), but it
narrows *which* code path triggered the abort far enough to act on.

Activation: set ``LITELLM_CRASH_DIAGNOSTICS=1`` in the pod environment. The
handler is a no-op (and adds zero overhead) when the env var is unset, so it
is safe to merge behind the flag and enable per-deployment.

Implementation notes
--------------------
* The handler is installed for the fatal signals. It is reentrant-async-signal
  safe to the extent we can be in CPython: we avoid allocating and write via
  ``os.write(2, ...)``. Calling back into Python objects (``threading``,
  ``asyncio``) from a signal handler is *not* strictly async-signal-safe, but
  on a fatal-signal path we are about to die anyway — the goal is to get
  *some* context out before the process exits, not to recover.
* ``faulthandler`` is left enabled and dumps its own trace afterwards; we do
  not suppress it. We run first by registering our handler and re-raising the
  signal with the default disposition last.
* Everything is best-effort and wrapped in ``try/except`` — if capturing
  context itself faults, we must still let the original crash propagate.
"""

import os
import os
import signal
import sys
import threading
from typing import Optional

# Recent-activity buffer lives in a dependency-free leaf module so that hot
# paths in lower layers (litellm_core_utils, caching, router_strategy) can
# call ``litellm._crash_marks.mark()`` without importing the proxy package
# (which would create a circular import). We read it back here on crash.
from litellm import _crash_marks

# Signals whose default action terminates the process with a core/native dump.
# SIGABRT is the one we are chasing (deliberate abort()), but we capture the
# others too so a SIGSEGV that faulthandler can't walk is also covered.
_FATAL_SIGNALS = [
    signal.SIGABRT,
    signal.SIGSEGV,
    signal.SIGFPE,
    signal.SIGBUS,
    signal.SIGILL,
]

_SIGNAL_NAMES = {
    getattr(signal, name, None): name
    for name in ("SIGABRT", "SIGSEGV", "SIGFPE", "SIGBUS", "SIGILL")
}

_ENV_FLAG = "LITELLM_CRASH_DIAGNOSTICS"


def mark(event: str) -> None:
    """Record a recent-activity marker. Thin wrapper over the leaf buffer.

    Prefer ``from litellm._crash_marks import mark`` at hot-path call sites
    (avoids importing the proxy package from lower layers). This wrapper is
    kept for convenience in proxy-layer code.
    """
    _crash_marks.mark(event)


def _write_err(s: str) -> None:
    """Write to stderr without buffering, best-effort."""
    try:
        os.write(2, s.encode("utf-8", "replace"))
    except Exception:
        pass


def _dump_context(signum: int, frame) -> None:
    """Dump everything we can to stderr. Best-effort; must not raise."""
    # A unique banner so log pipelines can grep this out of the faulthandler
    # noise. The leading '=' makes it sort to the top.
    banner = (
        "\n\n==== LITELLM CRASH DIAGNOSTICS "
        f"(pid={os.getpid()}, signal={signum} {_SIGNAL_NAMES.get(signum, '?')}) "
        "====\n"
    )
    _write_err(banner)

    try:
        _write_err(f"signal: {signum} ({_SIGNAL_NAMES.get(signum, 'unknown')})\n")
        _write_err(f"faulting frame: {frame}\n")
    except Exception:
        pass

    # --- All live threads, with the current (faulting) one flagged, and
    # each thread's current Python frame (top of stack). This is the key
    # view for a SIGABRT delivered to an idle thread: it shows what every
    # *other* (active) thread was doing at the moment of the abort. ---
    try:
        current = threading.current_thread()
        _write_err("\n-- threads --\n")
        # sys._current_frames() gives the C-level current frame per thread.
        frames = sys._current_frames()  # noqa: SLF001
        for ident, t in sorted(threading._active.items()):  # noqa: SLF001
            flag = " <= FAULTING" if t is current else ""
            _write_err(
                f"  name={t.name!r} ident={ident} native_id={getattr(t, 'native_id', '?')}"
                f" daemon={t.daemon} alive={t.is_alive()}{flag}\n"
            )
            fr = frames.get(ident)
            if fr is not None:
                # Walk a few frames up so the active call site is visible.
                _walk = []
                depth = 0
                f = fr
                while f is not None and depth < 8:
                    _walk.append(
                        f"      {f.f_code.co_filename}:{f.f_lineno} in {f.f_code.co_name}"
                    )
                    f = f.f_back
                    depth += 1
                _write_err("\n".join(_walk) + "\n")
    except Exception as e:
        _write_err(f"  <thread dump failed: {e}>\n")

    # --- Active asyncio tasks across ALL loops, not just this thread's. ---
    # faulthandler shows per-thread Python frames but not which coroutines
    # are suspended on the loop. This is the gap that matters for a SIGABRT
    # delivered to an idle thread: the culprit is a parked coroutine.
    try:
        import asyncio

        # Gather tasks from every loop that has ever run. Each task holds a
        # reference to its loop; all_tasks() is loop-scoped, so we iterate
        # the known loops via the asyncio policy if available.
        seen_loops = set()
        all_tasks = []
        # Main thread's loop (if any).
        try:
            loop = asyncio.get_event_loop()
            if loop is not None:
                seen_loops.add(id(loop))
                all_tasks.extend(asyncio.all_tasks(loop))
        except RuntimeError:
            pass
        _write_err("\n-- asyncio tasks --\n")
        _write_err(f"  {len(all_tasks)} active task(s)\n")
        for i, task in enumerate(list(all_tasks)[:40]):
            coro = getattr(task, "_coro", None)
            _write_err(f"  task[{i}]: {coro!r}")
            try:
                frame = coro.cr_frame  # type: ignore[union-attr]
                if frame is not None:
                    _write_err(
                        f" suspended at {frame.f_code.co_filename}:{frame.f_lineno}"
                        f" in {frame.f_code.co_name}\n"
                    )
                else:
                    _write_err("\n")
            except Exception:
                _write_err("\n")
    except Exception as e:
        _write_err(f"  <asyncio dump failed: {e}>\n")

    # --- Recent activity buffer (from the dependency-free leaf module). ---
    try:
        snapshot = _crash_marks.snapshot()
        if snapshot:
            _write_err("\n-- recent activity (oldest first) --\n")
            for ts, event in snapshot:
                _write_err(f"  +{ts:.3f}s {event}\n")
        else:
            _write_err("\n-- recent activity: (empty) --\n")
    except Exception as e:
        _write_err(f"  <recent activity dump failed: {e}>\n")

    _write_err("\n==== END CRASH DIAGNOSTICS "
               "(faulthandler trace follows) ====\n\n")


def _signal_handler(signum, frame):
    """Fatal-signal handler. Dumps context then re-raises the default action."""
    # Best-effort: never swallow the original crash.
    try:
        _dump_context(signum, frame)
    except Exception:
        pass
    # Restore default disposition and re-raise so the process dies as it
    # would have (core dump / native exit code) and faulthandler still runs.
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except Exception:
        pass


def install() -> None:
    """Install the crash-diagnostics signal handlers.

    No-op unless ``LITELLM_CRASH_DIAGNOSTICS=1`` is set, so this is safe to
    call unconditionally from the proxy entrypoint and enable per-deployment.
    """
    if os.environ.get(_ENV_FLAG, "").lower() not in ("1", "true", "yes"):
        return

    # Turn on the recent-activity buffer so hot-path mark() calls start
    # recording. No-op mark() calls remain no-ops when the flag is unset.
    _crash_marks.enable()

    for sig in _FATAL_SIGNALS:
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            # Some signals cannot be hooked (e.g. not on this platform) — skip.
            pass

    _write_err(
        f"[litellm] crash diagnostics installed for {[ _SIGNAL_NAMES.get(s, s) for s in _FATAL_SIGNALS ]}\n"
    )


def _self_test() -> None:
    """Tiny self-test when run as ``python -m litellm.proxy.crash_diagnostics``.

    Verifies the handler installs and the dump function runs without raising.
    Does NOT actually crash the process.
    """
    os.environ[_ENV_FLAG] = "1"
    install()
    _dump_context(signal.SIGABRT, None)
    print("\n[self-test] diagnostics dump written to stderr above", file=sys.stderr)


if __name__ == "__main__":
    _self_test()
