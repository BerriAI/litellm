"""
Session and Run state machines.

This module is pure — no I/O. Endpoints call into these helpers to validate
state transitions before persisting them.
"""

from typing import Optional

from litellm.proxy.agent_session_endpoints.constants import (
    RUN_ACTIVE_STATUSES,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_ERROR,
    RUN_STATUS_FINISHED,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_TERMINAL_STATUSES,
    SESSION_ACCEPTING_RUN_STATUSES,
    SESSION_STATUS_BUSY,
    SESSION_STATUS_ERROR,
    SESSION_STATUS_PROVISIONING,
    SESSION_STATUS_READY,
    SESSION_STATUS_TERMINATED,
    SESSION_TERMINAL_STATUSES,
)


# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

# Valid forward transitions. ``terminated`` is a sink — nothing transitions out.
_SESSION_TRANSITIONS = {
    SESSION_STATUS_PROVISIONING: {
        SESSION_STATUS_READY,
        SESSION_STATUS_ERROR,
        SESSION_STATUS_TERMINATED,
    },
    SESSION_STATUS_READY: {
        SESSION_STATUS_BUSY,
        SESSION_STATUS_ERROR,
        SESSION_STATUS_TERMINATED,
    },
    SESSION_STATUS_BUSY: {
        SESSION_STATUS_READY,
        SESSION_STATUS_ERROR,
        SESSION_STATUS_TERMINATED,
    },
    SESSION_STATUS_ERROR: {SESSION_STATUS_TERMINATED},
    SESSION_STATUS_TERMINATED: set(),
}


def is_valid_session_transition(current: str, target: str) -> bool:
    """Return True if a session can move from ``current`` to ``target``."""
    if current == target:
        return True
    return target in _SESSION_TRANSITIONS.get(current, set())


def session_can_accept_runs(status: str) -> bool:
    """Sessions in `ready` or `busy` accept new run inserts.

    `busy` returns True here because the lock-then-check flow in
    `POST /runs` lets `busy` through to a more specific 409 reason
    (`run_busy`) inside the transaction. Sessions that are
    `provisioning`, `error`, or `terminated` are rejected up-front.
    """
    return status in SESSION_ACCEPTING_RUN_STATUSES


def session_is_terminal(status: str) -> bool:
    return status in SESSION_TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# Run state machine
# ---------------------------------------------------------------------------

_RUN_TRANSITIONS = {
    RUN_STATUS_QUEUED: {
        RUN_STATUS_RUNNING,
        RUN_STATUS_CANCELLED,
        RUN_STATUS_ERROR,
    },
    RUN_STATUS_RUNNING: {
        RUN_STATUS_FINISHED,
        RUN_STATUS_CANCELLED,
        RUN_STATUS_ERROR,
    },
    RUN_STATUS_FINISHED: set(),
    RUN_STATUS_CANCELLED: set(),
    RUN_STATUS_ERROR: set(),
}


def is_valid_run_transition(current: str, target: str) -> bool:
    if current == target:
        return True
    return target in _RUN_TRANSITIONS.get(current, set())


def run_is_active(status: str) -> bool:
    return status in RUN_ACTIVE_STATUSES


def run_is_terminal(status: str) -> bool:
    return status in RUN_TERMINAL_STATUSES


def derive_session_status_from_runs(
    current_session_status: str,
    has_active_run: bool,
) -> Optional[str]:
    """Return the new session status or None if no transition is needed.

    This is the single source of truth for the busy<->ready oscillation
    triggered by run start/finish events. It is intentionally a pure
    function — callers persist the result inside their own transaction.
    """
    if current_session_status in SESSION_TERMINAL_STATUSES:
        return None
    if current_session_status == SESSION_STATUS_PROVISIONING:
        # Session must transition to ready (via daemon registration)
        # before run-driven busy/ready toggling kicks in.
        return None
    target = SESSION_STATUS_BUSY if has_active_run else SESSION_STATUS_READY
    if target == current_session_status:
        return None
    return target
