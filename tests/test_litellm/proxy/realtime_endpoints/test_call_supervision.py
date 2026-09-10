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
async def test_failed_logging_releases_reservation(monkeypatch):
    from litellm.proxy.realtime_endpoints import call_supervision

    socket = Socket()
    logger = MagicMock(spec=Logging)
    logger.model_call_details = {}
    sink = MagicMock()
    sink.log_messages = AsyncMock(side_effect=RuntimeError("logging unavailable"))
    release = AsyncMock()
    monkeypatch.setattr(call_supervision, "release_or_invalidate_budget_reservation", release)
    supervisor = CallSupervisor(socket, sink, logger, UserAPIKeyAuth(), AsyncMock())
    await socket.messages.put({"type": "session.started"})
    await supervisor.start()
    await socket.messages.put({"type": "session.closed"})
    with pytest.raises(RuntimeError, match="logging unavailable"):
        await supervisor.wait()
    release.assert_awaited_once_with(budget_reservation=None)
    assert socket.closed
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
