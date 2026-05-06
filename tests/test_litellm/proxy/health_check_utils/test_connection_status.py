"""
Unit tests for ConnectionStatusTracker (LIT-2607).
"""

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm._service_logger import (  # noqa: E402
    ServiceLogging,
    ServiceTypes,
    _record_dependency_status,
)
from litellm.proxy.health_check_utils.connection_status import (  # noqa: E402
    ConnectionStatusTracker,
    connection_status_tracker,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    connection_status_tracker.reset()
    yield
    connection_status_tracker.reset()


def test_initial_state_is_unknown():
    tracker = ConnectionStatusTracker()
    assert tracker.get("db").state == "unknown"
    assert tracker.get("redis").state == "unknown"
    assert tracker.get("db").last_updated is None
    assert tracker.get("redis").last_error is None


def test_mark_up_then_down_then_up_transitions():
    tracker = ConnectionStatusTracker()

    tracker.mark_up("db")
    assert tracker.get("db").state == "up"
    assert tracker.get("db").last_updated is not None
    assert tracker.get("db").last_error is None

    tracker.mark_down("db", error="ECONNREFUSED")
    assert tracker.get("db").state == "down"
    assert tracker.get("db").last_error == "ECONNREFUSED"

    tracker.mark_up("db")
    assert tracker.get("db").state == "up"
    assert tracker.get("db").last_error is None


def test_unknown_state_is_not_treated_as_down():
    """
    A dependency that's never been observed (just-started pod, or
    dependency not configured for this deployment) stays in
    ``unknown`` — and ``unknown`` must never fail liveness.
    """
    tracker = ConnectionStatusTracker()
    assert tracker.get("db").state == "unknown"
    assert tracker.get("redis").state == "unknown"
    assert tracker.is_any_down() is False


def test_is_any_down_only_true_when_actually_down():
    tracker = ConnectionStatusTracker()
    assert tracker.is_any_down() is False  # both unknown

    tracker.mark_up("db")
    tracker.mark_up("redis")
    assert tracker.is_any_down() is False  # both up

    tracker.mark_down("redis", error="x")
    assert tracker.is_any_down() is True

    tracker.mark_up("redis")
    tracker.mark_down("db", error="y")
    assert tracker.is_any_down() is True


def test_long_error_message_is_truncated():
    tracker = ConnectionStatusTracker()
    long_msg = "x" * 5000
    tracker.mark_down("db", error=long_msg)
    err = tracker.get("db").last_error
    assert err is not None
    assert len(err) == ConnectionStatusTracker._MAX_ERROR_LEN


def test_snapshot_is_json_friendly_and_isolated_from_internal_state():
    tracker = ConnectionStatusTracker()
    tracker.mark_up("db")
    tracker.mark_down("redis", error="boom")

    snap = tracker.snapshot()
    assert snap["db"]["status"] == "up"
    assert snap["redis"]["status"] == "down"
    assert snap["redis"]["last_error"] == "boom"
    assert isinstance(snap["db"]["last_updated"], str)

    # Mutating the snapshot must not affect internal state.
    snap["db"]["status"] = "tampered"
    assert tracker.get("db").state == "up"


def test_concurrent_mark_calls_are_thread_safe():
    """
    The tracker uses threading.Lock — make sure parallel mark_up/mark_down
    don't corrupt internal state. We run a swarm of writers; the final
    state must be a coherent up or down (never a half-written object).
    """
    tracker = ConnectionStatusTracker()

    def writer(state: str):
        for _ in range(500):
            if state == "up":
                tracker.mark_up("db")
            else:
                tracker.mark_down("db", error="e")

    threads = [threading.Thread(target=writer, args=(s,)) for s in ["up", "down"] * 8]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = tracker.get("db")
    assert final.state in ("up", "down")
    assert final.last_updated is not None


@pytest.mark.asyncio
async def test_service_logger_success_marks_db_up():
    sl = ServiceLogging()
    await sl.async_service_success_hook(
        service=ServiceTypes.DB, call_type="x", duration=0.1
    )
    assert connection_status_tracker.get("db").state == "up"


@pytest.mark.asyncio
async def test_service_logger_failure_marks_db_down_with_error():
    sl = ServiceLogging()
    await sl.async_service_failure_hook(
        service=ServiceTypes.DB,
        duration=0.1,
        error=Exception("connection refused"),
        call_type="x",
    )
    db = connection_status_tracker.get("db")
    assert db.state == "down"
    assert db.last_error is not None
    assert "connection refused" in db.last_error


@pytest.mark.asyncio
async def test_service_logger_success_marks_redis_up():
    sl = ServiceLogging()
    await sl.async_service_success_hook(
        service=ServiceTypes.REDIS, call_type="ping", duration=0.0
    )
    assert connection_status_tracker.get("redis").state == "up"


@pytest.mark.asyncio
async def test_service_logger_failure_marks_redis_down():
    sl = ServiceLogging()
    await sl.async_service_failure_hook(
        service=ServiceTypes.REDIS,
        duration=0.0,
        error="redis timeout",
        call_type="ping",
    )
    assert connection_status_tracker.get("redis").state == "down"


def test_record_dependency_status_ignores_non_db_redis_services():
    """
    Other ServiceTypes (PROXY_PRE_CALL, AUTH, ROUTER, ...) must not flip
    DB or Redis status — those services have nothing to do with the
    liveness probe's dependency view.
    """
    _record_dependency_status(
        ServiceTypes.PROXY_PRE_CALL, is_error=True, error_message="nope"
    )
    assert connection_status_tracker.get("db").state == "unknown"
    assert connection_status_tracker.get("redis").state == "unknown"


def test_record_dependency_status_batch_write_to_db_treated_as_db():
    _record_dependency_status(ServiceTypes.BATCH_WRITE_TO_DB, is_error=False)
    assert connection_status_tracker.get("db").state == "up"
