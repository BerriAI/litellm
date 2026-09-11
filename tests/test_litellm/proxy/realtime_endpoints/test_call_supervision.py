import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.realtime_streaming import REALTIME_SESSION_SUCCESS_LOGGED_KEY
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.realtime_endpoints.call_supervision import CallSupervisor, CallSupervisors


class Socket:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.messages.get()
        if message is None:
            raise StopAsyncIteration
        if isinstance(message, Exception):
            raise message
        return json.dumps(message)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_lost", [False, True])
async def test_supervisor_holds_call_lease_until_terminal_accounting(lease_lost):
    from litellm.proxy.hooks.realtime_call_lease import RealtimeCallLease

    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)
    lost = asyncio.Event()
    lease = MagicMock(spec=RealtimeCallLease)
    lease.wait_failed = lost.wait

    async def release():
        assert socket.closed
        assert sink.logs == 1

    lease.close = AsyncMock(side_effect=release)

    async def close():
        await socket.messages.put({"type": "session.closed", "usage": {"audio_duration_ms": 1000}})

    terminate = AsyncMock(side_effect=close)
    supervisor = CallSupervisor(socket, sink, logger, UserAPIKeyAuth(), terminate, lease=lease)
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    lease.close.assert_not_awaited()
    if lease_lost:
        lost.set()
    else:
        await close()
    await asyncio.wait_for(supervisor.wait(), 1)
    assert terminate.await_count == int(lease_lost)
    lease.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled_step", ["close", "drain"])
async def test_live_initial_close_reserves_time_for_independent_hangup(stalled_step):
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    close_cancelled = asyncio.Event()

    async def close():
        if stalled_step == "close":
            try:
                await asyncio.Event().wait()
            finally:
                close_cancelled.set()

    async def force_close():
        await socket.messages.put({"type": "session.closed", "usage": {"audio_duration_ms": 1000}})

    force = AsyncMock(side_effect=force_close)
    sink = Sink(logger)
    supervisor = CallSupervisor(
        socket,
        sink,
        logger,
        UserAPIKeyAuth(),
        close,
        force_close_call=force,
        drain_timeout=1,
        termination_timeout=0.08,
    )
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await asyncio.wait_for(supervisor.close(), timeout=0.5)
    force.assert_awaited_once()
    assert close_cancelled.is_set() == (stalled_step == "close")
    assert any(event["type"] == "session.closed" for event in sink.events)
    assert not logger.model_call_details.get("realtime_usage_incomplete")
    assert sink.logs == 1
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", ["terminal", "no_terminal", "timeout"])
async def test_live_unacknowledged_close_uses_bounded_independent_hangup(monkeypatch, fallback):
    from litellm.proxy.realtime_endpoints import call_supervision

    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)
    invalidate = AsyncMock()
    monkeypatch.setattr(call_supervision, "invalidate_budget_reservation_counters", invalidate)

    async def force_close():
        if fallback == "terminal":
            await socket.messages.put({"type": "session.closed", "usage": {"audio_duration_ms": 1000}})
        elif fallback == "timeout":
            await asyncio.Event().wait()

    force = AsyncMock(side_effect=force_close)
    close = AsyncMock()
    supervisor = CallSupervisor(
        socket,
        sink,
        logger,
        UserAPIKeyAuth(),
        close,
        force_close_call=force,
        drain_timeout=0.01,
        termination_timeout=0.08,
    )
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await asyncio.wait_for(supervisor.close(), timeout=0.5)
    close.assert_awaited_once()
    force.assert_awaited_once()
    assert socket.closed
    if fallback == "terminal":
        invalidate.assert_not_awaited()
        assert not logger.model_call_details.get("realtime_usage_incomplete")
    else:
        invalidate.assert_awaited_once()
        assert logger.model_call_details["realtime_usage_incomplete"] is True


@pytest.mark.asyncio
async def test_live_confirmed_terminal_does_not_force_hangup():
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}

    async def close():
        await socket.messages.put({"type": "session.closed"})

    force = AsyncMock()
    supervisor = CallSupervisor(socket, Sink(logger), logger, UserAPIKeyAuth(), close, force_close_call=force)
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await supervisor.close()
    force.assert_not_awaited()


class Sink:
    def __init__(self, logger):
        self.logger = logger
        self.events = []
        self.logs = 0

    def store_message(self, message):
        self.events.append(json.loads(message))

    async def log_messages(self, *, wait_for_dispatch=False):
        assert wait_for_dispatch
        self.logs += 1
        self.logger.model_call_details[REALTIME_SESSION_SUCCESS_LOGGED_KEY] = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duration,valid", [(0, True), (1000, True), (None, False), (-1, False), (True, False), ("1000", False)]
)
async def test_live_terminal_requires_valid_duration_for_accounting(monkeypatch, duration, valid):
    from litellm.proxy.realtime_endpoints import call_supervision

    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    invalidate = AsyncMock()
    monkeypatch.setattr(call_supervision, "invalidate_budget_reservation_counters", invalidate)
    close = AsyncMock()
    force = AsyncMock()
    supervisor = CallSupervisor(socket, Sink(logger), logger, UserAPIKeyAuth(), close, force_close_call=force)
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await socket.messages.put(
        {"type": "session.closed", **({"usage": {"audio_duration_ms": duration}} if duration is not None else {})}
    )
    await supervisor.wait()
    close.assert_not_awaited()
    force.assert_not_awaited()
    assert socket.closed
    assert bool(logger.model_call_details.get("realtime_usage_incomplete")) is not valid
    assert invalidate.await_count == (0 if valid else 1)


def fixture(*, ready_timeout=1, lifetime=1):
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)

    async def hangup():
        assert not socket.closed
        await socket.messages.put({"type": "session.closed", "usage": {"total_tokens": 42}})

    close_call = AsyncMock(side_effect=hangup)
    supervisor = CallSupervisor(
        socket,
        sink,
        logger,
        UserAPIKeyAuth(),
        close_call,
        ready_timeout=ready_timeout,
        lifetime=lifetime,
        drain_timeout=0.05,
    )
    return socket, sink, close_call, supervisor


@pytest.mark.asyncio
async def test_observer_logs_webrtc_usage_without_client_sideband():
    socket, sink, close_call, supervisor = fixture()
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await socket.messages.put({"type": "response.done", "response": {"usage": {"total_tokens": 15}}})
    await socket.messages.put({"type": "session.closed", "usage": {"total_tokens": 19}})
    await supervisor.wait()
    await supervisor.close()
    assert sink.logs == 1
    assert sink.events[-1]["usage"]["total_tokens"] == 19
    assert sink.events[1]["response"]["usage"]["total_tokens"] == 15
    assert socket.closed
    close_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_early_upstream_eof_rejects_start():
    socket, sink, close_call, supervisor = fixture()
    await socket.messages.put(None)
    with pytest.raises(RuntimeError, match="ended before"):
        await supervisor.start()
    assert socket.closed
    assert sink.logs == 1
    close_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_start_hangs_up_and_drains_terminal_usage():
    socket, sink, close_call, supervisor = fixture()
    started = asyncio.create_task(supervisor.start())
    await asyncio.sleep(0)
    started.cancel()
    with pytest.raises(asyncio.CancelledError):
        await started
    close_call.assert_awaited_once()
    assert socket.closed
    assert sink.logs == 1
    assert sink.events[-1]["usage"]["total_tokens"] == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [1, 2, 3])
async def test_repeated_start_cancellation_keeps_lease_until_shutdown_finishes(cancel_count):
    from litellm.proxy.hooks.realtime_call_lease import RealtimeCallLease

    reading = asyncio.Event()
    close_entered = asyncio.Event()
    allow_close = asyncio.Event()
    released = asyncio.Event()

    class ObservedSocket(Socket):
        async def __anext__(self):
            reading.set()
            return await super().__anext__()

    socket = ObservedSocket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)

    async def close_call():
        close_entered.set()
        await allow_close.wait()
        await socket.messages.put({"type": "session.closed", "usage": {"audio_duration_ms": 1000}})

    async def release():
        released.set()

    lease = RealtimeCallLease(renew=AsyncMock(return_value=True), release=release)
    lease.start()
    supervisor = CallSupervisor(
        socket, sink, logger, UserAPIKeyAuth(), close_call, lease=lease, ready_timeout=10, termination_timeout=10
    )
    registry = CallSupervisors()

    async def signaling():
        transferred = False
        try:
            await registry.start(supervisor)
            transferred = True
        finally:
            # The signaling endpoint retains lease ownership until registry startup succeeds.
            if not transferred:
                await lease.close()

    started = asyncio.create_task(signaling())
    shutdown = None
    try:
        await asyncio.wait_for(reading.wait(), timeout=1)
        started.cancel()
        await asyncio.wait_for(close_entered.wait(), timeout=1)
        for _ in range(cancel_count - 1):
            started.cancel()
            done, _ = await asyncio.wait({started}, timeout=0.02)
            assert not done
            assert not released.is_set()
        shutdown = asyncio.create_task(registry.shutdown())
        done, _ = await asyncio.wait({started, shutdown}, timeout=0.02)
        assert not done
        assert not released.is_set()
        assert not socket.closed
        assert sink.logs == 0
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(started, timeout=1)
        await asyncio.wait_for(shutdown, timeout=1)
        assert socket.closed
        assert sink.logs == 1
        assert released.is_set()
        assert sink.events[-1]["usage"]["audio_duration_ms"] == 1000
    finally:
        allow_close.set()
        await asyncio.wait_for(supervisor.wait(), timeout=1)
        await asyncio.gather(started, return_exceptions=True)
        if shutdown is not None:
            await shutdown
        await registry.shutdown()
        await lease.close()


@pytest.mark.asyncio
async def test_worker_shutdown_drains_all_calls():
    registry = CallSupervisors()
    socket, sink, close_call, supervisor = fixture()
    await socket.messages.put({"type": "session.created"})
    await registry.start(supervisor)
    await registry.shutdown()
    await registry.shutdown()
    close_call.assert_awaited_once()
    assert socket.closed
    assert sink.logs == 1
    assert sink.events[-1]["usage"]["total_tokens"] == 42


@pytest.mark.asyncio
async def test_ready_timeout_hangs_up_before_returning_error():
    socket, sink, close_call, supervisor = fixture(ready_timeout=0.01)
    with pytest.raises(asyncio.TimeoutError):
        await supervisor.start()
    close_call.assert_awaited_once()
    assert socket.closed
    assert sink.logs == 1


@pytest.mark.asyncio
async def test_lifetime_limit_closes_call_and_collects_final_usage():
    socket, sink, close_call, supervisor = fixture(lifetime=0.01)
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await supervisor.wait()
    close_call.assert_awaited_once()
    assert socket.closed
    assert sink.events[-1]["usage"]["total_tokens"] == 42


@pytest.mark.asyncio
async def test_socket_eof_after_ready_still_hangs_up_provider_call(monkeypatch):
    from litellm.proxy.realtime_endpoints import call_supervision

    invalidate = AsyncMock()
    release = AsyncMock()
    monkeypatch.setattr(call_supervision, "invalidate_budget_reservation_counters", invalidate)
    monkeypatch.setattr(call_supervision, "release_or_invalidate_budget_reservation", release)
    socket, sink, close_call, supervisor = fixture()
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await socket.messages.put(None)
    await supervisor.wait()
    close_call.assert_awaited_once()
    assert socket.closed
    assert sink.logs == 1
    assert sink.logger.model_call_details["realtime_usage_incomplete"] is True
    invalidate.assert_awaited_once()
    release.assert_not_awaited()


@pytest.mark.asyncio
async def test_observer_error_rejects_start(caplog):
    socket, sink, close_call, supervisor = fixture()
    await socket.messages.put(RuntimeError("private-provider-credential"))
    with pytest.raises(RuntimeError, match="ended before"):
        await supervisor.start()
    assert socket.closed
    close_call.assert_awaited_once()
    assert "private-provider-credential" not in caplog.text


@pytest.mark.asyncio
async def test_failed_logging_invalidates_reservation_without_zeroing_spend(monkeypatch):
    from litellm.proxy.realtime_endpoints import call_supervision

    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = MagicMock()
    sink.log_messages = AsyncMock(side_effect=RuntimeError("logging unavailable"))
    release = AsyncMock()
    invalidate = AsyncMock()
    monkeypatch.setattr(call_supervision, "release_or_invalidate_budget_reservation", release)
    monkeypatch.setattr(call_supervision, "invalidate_budget_reservation_counters", invalidate)
    supervisor = CallSupervisor(socket, sink, logger, UserAPIKeyAuth(), AsyncMock())
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await socket.messages.put({"type": "session.closed"})
    with pytest.raises(RuntimeError, match="logging unavailable"):
        await supervisor.wait()
    release.assert_not_awaited()
    invalidate.assert_awaited_once_with(budget_reservation=None)
    assert logger.model_call_details["realtime_accounting_incomplete"] is True
    assert socket.closed
    sink.log_messages.assert_awaited_once_with(wait_for_dispatch=True)


@pytest.mark.asyncio
async def test_start_rejects_terminal_session_while_accounting_is_pending():
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    dispatch_started = asyncio.Event()
    allow_dispatch = asyncio.Event()

    async def log_messages(*, wait_for_dispatch=False):
        dispatch_started.set()
        await allow_dispatch.wait()
        logger.model_call_details[REALTIME_SESSION_SUCCESS_LOGGED_KEY] = True

    sink = MagicMock()
    sink.log_messages = AsyncMock(side_effect=log_messages)
    supervisor = CallSupervisor(socket, sink, logger, UserAPIKeyAuth(), AsyncMock())
    await socket.messages.put({"type": "session.created"})
    await socket.messages.put({"type": "session.closed"})
    startup = asyncio.create_task(supervisor.start())
    try:
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
    finally:
        allow_dispatch.set()
    with pytest.raises(RuntimeError, match="ended before"):
        await asyncio.wait_for(startup, timeout=1)
    assert socket.closed
    sink.log_messages.assert_awaited_once_with(wait_for_dispatch=True)


@pytest.mark.asyncio
async def test_shutdown_bounds_accounting_and_invalidates_partial_dispatch(monkeypatch):
    from litellm.proxy.realtime_endpoints import call_supervision

    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    dispatch_cancelled = asyncio.Event()
    invalidate = AsyncMock()
    release = AsyncMock()
    monkeypatch.setattr(call_supervision, "invalidate_budget_reservation_counters", invalidate)
    monkeypatch.setattr(call_supervision, "release_or_invalidate_budget_reservation", release)

    async def log_messages(*, wait_for_dispatch=False):
        try:
            await asyncio.Event().wait()
        finally:
            dispatch_cancelled.set()

    async def hangup():
        await socket.messages.put({"type": "session.closed", "usage": {"total_tokens": 42}})

    sink = MagicMock()
    sink.log_messages = AsyncMock(side_effect=log_messages)
    supervisor = CallSupervisor(socket, sink, logger, UserAPIKeyAuth(), hangup, logging_timeout=0.01)
    registry = CallSupervisors()
    await socket.messages.put({"type": "session.created"})
    await registry.start(supervisor)
    await asyncio.wait_for(registry.shutdown(), timeout=1)
    assert dispatch_cancelled.is_set()
    assert socket.closed
    assert logger.model_call_details["realtime_accounting_incomplete"] is True
    assert not logger.model_call_details.get(REALTIME_SESSION_SUCCESS_LOGGED_KEY)
    invalidate.assert_awaited_once_with(budget_reservation=None)
    release.assert_not_awaited()
    sink.log_messages.assert_awaited_once_with(wait_for_dispatch=True)


@pytest.mark.asyncio
async def test_shutdown_waits_for_usage_dispatch_completion():
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    dispatch_started = asyncio.Event()
    dispatch_complete = asyncio.Event()
    dispatch_finished = asyncio.Event()

    async def log_messages(*, wait_for_dispatch=False):
        assert wait_for_dispatch
        dispatch_started.set()
        await dispatch_complete.wait()
        logger.model_call_details[REALTIME_SESSION_SUCCESS_LOGGED_KEY] = True
        dispatch_finished.set()

    async def hangup():
        await socket.messages.put({"type": "session.closed", "usage": {"total_tokens": 42}})

    sink = MagicMock()
    sink.log_messages = AsyncMock(side_effect=log_messages)
    registry = CallSupervisors()
    supervisor = CallSupervisor(socket, sink, logger, UserAPIKeyAuth(), hangup)
    await socket.messages.put({"type": "session.created"})
    await registry.start(supervisor)
    shutdown = asyncio.create_task(registry.shutdown())
    try:
        await asyncio.wait_for(dispatch_started.wait(), timeout=1)
        assert not shutdown.done()
        assert not dispatch_finished.is_set()
    finally:
        dispatch_complete.set()
        await asyncio.wait_for(shutdown, timeout=1)
    assert dispatch_finished.is_set()
    sink.log_messages.assert_awaited_once_with(wait_for_dispatch=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_usage_required", [True, False])
async def test_confirmed_hangup_without_terminal_usage_matches_protocol(terminal_usage_required):
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)
    close_call = AsyncMock()
    supervisor = CallSupervisor(
        socket,
        sink,
        logger,
        UserAPIKeyAuth(),
        close_call,
        terminal_usage_required=terminal_usage_required,
        drain_timeout=0.01,
    )
    await socket.messages.put({"type": "session.created"})
    await supervisor.start()
    await socket.messages.put({"type": "response.done", "response": {"usage": {"total_tokens": 17}}})
    await supervisor.close()
    assert bool(logger.model_call_details.get("realtime_usage_incomplete")) == terminal_usage_required
    assert sink.events[-1]["response"]["usage"]["total_tokens"] == 17
    assert sink.logs == 1
    assert socket.closed
    close_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_allows_hangup_longer_than_usage_drain_timeout():
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)
    hangup_started = asyncio.Event()
    allow_hangup = asyncio.Event()
    hangup_finished = asyncio.Event()

    async def hangup():
        hangup_started.set()
        await allow_hangup.wait()
        await socket.messages.put({"type": "session.closed", "usage": {"total_tokens": 42}})
        hangup_finished.set()

    supervisor = CallSupervisor(
        socket,
        sink,
        logger,
        UserAPIKeyAuth(),
        hangup,
        drain_timeout=0.01,
        termination_timeout=1,
        terminal_usage_required=False,
    )
    registry = CallSupervisors()
    await socket.messages.put({"type": "session.created"})
    await registry.start(supervisor)
    shutdown = asyncio.create_task(registry.shutdown())
    try:
        await asyncio.wait_for(hangup_started.wait(), timeout=1)
        await asyncio.sleep(0.04)
        assert not shutdown.done()
        assert not socket.closed
        assert not hangup_finished.is_set()
    finally:
        allow_hangup.set()
        await asyncio.wait_for(shutdown, timeout=1)
    assert hangup_finished.is_set()
    assert socket.closed
    assert sink.logs == 1
    assert sink.events[-1]["usage"]["total_tokens"] == 42
    assert not logger.model_call_details.get("realtime_usage_incomplete")


@pytest.mark.asyncio
async def test_termination_timeout_cancels_hangup_and_finishes_cleanup():
    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)
    hangup_cancelled = asyncio.Event()

    async def hangup():
        try:
            await asyncio.Event().wait()
        finally:
            hangup_cancelled.set()

    supervisor = CallSupervisor(
        socket,
        sink,
        logger,
        UserAPIKeyAuth(),
        hangup,
        drain_timeout=0.01,
        termination_timeout=0.02,
        terminal_usage_required=False,
    )
    await socket.messages.put({"type": "session.created"})
    await supervisor.start()
    await asyncio.wait_for(supervisor.close(), timeout=1)
    assert hangup_cancelled.is_set()
    assert socket.closed
    assert sink.logs == 1
    assert logger.model_call_details["realtime_usage_incomplete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("closure", ["eof", "normal_close", "error"])
@pytest.mark.parametrize("hangup_succeeds", [True, False])
async def test_ga_observer_disconnect_requires_confirmed_hangup(closure, hangup_succeeds):
    from websockets.exceptions import ConnectionClosedOK
    from websockets.frames import Close

    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = Sink(logger)
    close_call = AsyncMock(side_effect=None if hangup_succeeds else RuntimeError("unconfirmed hangup"))
    supervisor = CallSupervisor(
        socket,
        sink,
        logger,
        UserAPIKeyAuth(),
        close_call,
        terminal_usage_required=False,
        drain_timeout=0.01,
    )
    await socket.messages.put({"type": "session.created"})
    await supervisor.start()
    await socket.messages.put(
        None
        if closure == "eof"
        else ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
        if closure == "normal_close"
        else RuntimeError("observer failed")
    )
    await supervisor.wait()
    close_call.assert_awaited_once()
    assert bool(logger.model_call_details.get("realtime_usage_incomplete")) == (not hangup_succeeds)
    assert sink.logs == 1
    assert socket.closed
