import asyncio
import os
import signal
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


@pytest.fixture
def mock_proxy_logging():
    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    return proxy_logging


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_succeed(mock_proxy_logging):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.recreate_prisma_client = AsyncMock(return_value=True)
    # Probe fails (connection genuinely broken) so the direct path proceeds to
    # recreate; the post-recreate smoke test then succeeds. A healthy probe
    # would instead skip the recreate (covered in test_prisma_client_reconnect).
    client.db.query_raw = AsyncMock(
        side_effect=[ConnectionError("probe failed"), [{"result": 1}]]
    )
    client._start_engine_watcher = AsyncMock()

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        result = await client.attempt_db_reconnect(
            reason="unit_test_reconnect_success",
            force=True,
        )

    assert result is True
    client.db.recreate_prisma_client.assert_awaited_once_with(
        "postgresql://test", expected_generation=0
    )
    assert client.db.query_raw.await_count == 2


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_skip_when_in_cooldown(mock_proxy_logging):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])
    client._db_reconnect_cooldown_seconds = 120
    client._db_last_reconnect_attempt_ts = time.time()

    result = await client.attempt_db_reconnect(
        reason="unit_test_reconnect_cooldown",
        force=False,
    )

    assert result is False
    client.db.disconnect.assert_not_called()
    client.db.connect.assert_not_called()
    client.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_skip_when_lock_timeout_expires(
    mock_proxy_logging,
):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])

    await client._db_reconnect_lock.acquire()
    try:
        result = await client.attempt_db_reconnect(
            reason="unit_test_reconnect_lock_timeout",
            force=True,
            timeout_seconds=0.1,
            lock_timeout_seconds=0.01,
        )
    finally:
        client._db_reconnect_lock.release()

    assert result is False
    client.db.disconnect.assert_not_called()
    client.db.connect.assert_not_called()
    client.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_not_leak_lock_on_timeout_race(
    mock_proxy_logging,
):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])

    async def _fake_wait(tasks, timeout=None, return_when=None):
        # Let the acquire task run first, then emulate a timeout response
        # from asyncio.wait to exercise timeout-race cleanup.
        await asyncio.sleep(0)
        return set(), set(tasks)

    with patch("litellm.proxy.utils.asyncio.wait", side_effect=_fake_wait):
        result = await client.attempt_db_reconnect(
            reason="unit_test_reconnect_lock_timeout_race",
            force=True,
            timeout_seconds=0.1,
            lock_timeout_seconds=0.01,
        )

    assert result is False
    assert client._db_reconnect_lock.locked() is False
    client.db.disconnect.assert_not_called()
    client.db.connect.assert_not_called()
    client.db.query_raw.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_db_reconnect_should_set_cooldown_after_attempt(
    mock_proxy_logging,
):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client._db_last_reconnect_attempt_ts = 0.0
    client._db_reconnect_cooldown_seconds = 10
    client.db.recreate_prisma_client = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])
    client._start_engine_watcher = AsyncMock()

    # Use a counter-based mock to avoid StopIteration when time.time() is called
    # more times than expected (varies by Python version / internal code paths).
    fake_clock = iter(range(100, 10000))
    with (
        patch(
            "litellm.proxy.utils.time.time",
            side_effect=lambda: float(next(fake_clock)),
        ),
        patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
    ):
        result = await client.attempt_db_reconnect(
            reason="unit_test_cooldown_timestamp_after_attempt",
            timeout_seconds=0.1,
        )

    assert result is True
    # The last time.time() call sets _db_last_reconnect_attempt_ts in the finally block.
    # Just verify it was updated to a value greater than the initial 0.0.
    assert client._db_last_reconnect_attempt_ts > 0.0


@pytest.mark.asyncio
async def test_run_reconnect_cycle_watchdog_should_use_recreate_prisma_client(
    mock_proxy_logging,
):
    """Direct reconnect goes through recreate_prisma_client (which non-blockingly
    kills the old engine) instead of calling disconnect() — see issue #26191.
    """
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.disconnect = AsyncMock(
        side_effect=AssertionError("disconnect must not be called")
    )
    client.db.recreate_prisma_client = AsyncMock(return_value=True)
    # Probe fails so we proceed to recreate (and verify disconnect is never
    # used — issue #26191); the post-recreate smoke test then succeeds.
    client.db.query_raw = AsyncMock(
        side_effect=[ConnectionError("probe failed"), [{"result": 1}]]
    )
    client._start_engine_watcher = AsyncMock()

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        await client._run_reconnect_cycle(timeout_seconds=None)

    client.db.recreate_prisma_client.assert_awaited_once_with(
        "postgresql://test", expected_generation=0
    )
    assert client.db.query_raw.await_count == 2
    client.db.disconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_reconnect_cycle_watchdog_should_use_default_timeout_budget(
    mock_proxy_logging,
):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client._db_watchdog_reconnect_timeout_seconds = 0.1
    client._start_engine_watcher = AsyncMock()

    async def _slow_recreate(_db_url, **_kwargs):
        await asyncio.sleep(0.08)

    probe_calls = {"n": 0}

    async def _probe_fails_then_slow_smoke(_query: str):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:
            # Probe fails fast so the cycle proceeds to the slow recreate +
            # smoke test, whose combined time must exceed the overall budget.
            raise ConnectionError("probe failed")
        await asyncio.sleep(0.08)
        return [{"result": 1}]

    client.db.recreate_prisma_client = AsyncMock(side_effect=_slow_recreate)
    client.db.query_raw = AsyncMock(side_effect=_probe_fails_then_slow_smoke)

    with (
        pytest.raises(asyncio.TimeoutError),
        patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
    ):
        await client._run_reconnect_cycle(timeout_seconds=None)


@pytest.mark.asyncio
async def test_run_reconnect_cycle_timeout_should_use_single_overall_budget(
    mock_proxy_logging,
):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client._start_engine_watcher = AsyncMock()

    async def _slow_recreate(_db_url, **_kwargs):
        await asyncio.sleep(0.08)

    probe_calls = {"n": 0}

    async def _probe_fails_then_slow_smoke(_query: str):
        probe_calls["n"] += 1
        if probe_calls["n"] == 1:
            # Probe fails fast so the cycle proceeds to the slow recreate +
            # smoke test, whose combined time must exceed the overall budget.
            raise ConnectionError("probe failed")
        await asyncio.sleep(0.08)
        return [{"result": 1}]

    client.db.recreate_prisma_client = AsyncMock(side_effect=_slow_recreate)
    client.db.query_raw = AsyncMock(side_effect=_probe_fails_then_slow_smoke)

    with (
        pytest.raises(asyncio.TimeoutError),
        patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
    ):
        await client._run_reconnect_cycle(timeout_seconds=0.1)


@pytest.mark.asyncio
async def test_db_health_watchdog_should_trigger_reconnect_on_db_error(
    mock_proxy_logging,
):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.query_raw = AsyncMock(side_effect=Exception("db connection dropped"))
    client.attempt_db_reconnect = AsyncMock(return_value=True)
    client._db_health_watchdog_interval_seconds = 1
    client._db_watchdog_reconnect_timeout_seconds = 7.0
    client._db_health_watchdog_probe_timeout_seconds = 0.2

    with (
        patch(
            "litellm.proxy.utils.asyncio.sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError()]),
        ),
        patch(
            "litellm.proxy.db.exception_handler.PrismaDBExceptionHandler.is_database_connection_error",
            return_value=True,
        ),
    ):
        await client._db_health_watchdog_loop()

    client.attempt_db_reconnect.assert_awaited_once_with(
        reason="db_health_watchdog_connection_error",
        timeout_seconds=7.0,
    )


@pytest.mark.asyncio
async def test_db_health_watchdog_should_trigger_reconnect_on_probe_timeout(
    mock_proxy_logging,
):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.query_raw = AsyncMock(side_effect=asyncio.TimeoutError())
    client.attempt_db_reconnect = AsyncMock(return_value=True)
    client._db_health_watchdog_interval_seconds = 1
    client._db_watchdog_reconnect_timeout_seconds = 9.0
    client._db_health_watchdog_probe_timeout_seconds = 0.2

    with (
        patch(
            "litellm.proxy.utils.asyncio.sleep",
            AsyncMock(side_effect=[None, asyncio.CancelledError()]),
        ),
        patch(
            "litellm.proxy.db.exception_handler.PrismaDBExceptionHandler.is_database_connection_error",
            return_value=False,
        ),
    ):
        await client._db_health_watchdog_loop()

    client.attempt_db_reconnect.assert_awaited_once_with(
        reason="db_health_watchdog_connection_error",
        timeout_seconds=9.0,
    )


@pytest.mark.asyncio
async def test_db_health_watchdog_start_stop_lifecycle(mock_proxy_logging):
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client._db_health_watchdog_enabled = True
    client._db_health_watchdog_interval_seconds = 3600

    loop = asyncio.get_running_loop()
    dummy_task = loop.create_task(asyncio.sleep(3600))

    def _fake_create_task(coro):
        # create_task is patched in this test, so explicitly close the incoming coroutine
        # to avoid "coroutine was never awaited" warnings.
        coro.close()
        return dummy_task

    with patch(
        "litellm.proxy.utils.asyncio.create_task", side_effect=_fake_create_task
    ):
        await client.start_db_health_watchdog_task()
        assert client._db_health_watchdog_task is dummy_task

        await client.stop_db_health_watchdog_task()
        assert client._db_health_watchdog_task is None
        assert dummy_task.cancelled() is True


@pytest.mark.asyncio
async def test_recreate_prisma_client_kills_old_engine_without_disconnect(
    mock_proxy_logging,
):
    """recreate_prisma_client SIGTERMs the old engine PID directly rather than
    calling `disconnect()`, which blocks the asyncio event loop on the sync
    `subprocess.Popen.wait()` inside prisma-client-py — see issue #26191.
    """
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    disconnect_mock = AsyncMock(
        side_effect=AssertionError("disconnect must not be called on reconnect path")
    )
    client.db._original_prisma.disconnect = disconnect_mock

    with (
        patch.object(client.db, "_get_engine_pid", return_value=9999),
        patch("litellm.proxy.db.prisma_client.os.kill") as mock_kill,
        patch("litellm.proxy.db.prisma_client.asyncio.sleep", new_callable=AsyncMock),
    ):
        # Return a Prisma instance whose connect() is awaitable.
        fake_new_prisma = MagicMock()
        fake_new_prisma.connect = AsyncMock(return_value=None)
        with patch("prisma.Prisma", return_value=fake_new_prisma):
            await client.db.recreate_prisma_client("postgresql://test")

    mock_kill.assert_any_call(9999, signal.SIGTERM)
    disconnect_mock.assert_not_awaited()
    fake_new_prisma.connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_generic_data: transport-reconnect-and-retry coverage (issue #25143)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_generic_data_retries_on_transport_error_for_config_table(
    mock_proxy_logging,
):
    """`get_generic_data(table_name="config")` self-heals on a transient
    `httpx.ReadError`: reconnect once, retry once, return the row.

    Regression for issue #25143 — the 1.83.x line lost the reconnect-and-retry
    branch that 1.82.6 had on this method. `_update_config_from_db` fans out
    four concurrent `get_generic_data` calls, so a single transport flap used
    to surface as four `db_exceptions` alerts and a stale config window.
    """
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )

    expected_row = {"param_name": "general_settings", "param_value": {"foo": "bar"}}
    invocations: list[None] = []

    async def _flaky_find_first(**kwargs):
        invocations.append(None)
        if len(invocations) == 1:
            raise httpx.ReadError("simulated transport blip")
        return expected_row

    client.db.litellm_config.find_first = AsyncMock(side_effect=_flaky_find_first)
    client.attempt_db_reconnect = AsyncMock(return_value=True)

    result = await client.get_generic_data(
        key="param_name",
        value="general_settings",
        table_name="config",
    )

    assert result == expected_row
    assert len(invocations) == 2
    client.attempt_db_reconnect.assert_awaited_once()
    reconnect_kwargs = client.attempt_db_reconnect.await_args.kwargs
    assert reconnect_kwargs["reason"] == "prisma_get_generic_data_config_lookup_failure"

    # The failure_handler telemetry side-effect must NOT fire on the first
    # transport blip — only if the post-retry call also fails. Drain the
    # event loop so any spuriously-spawned task would have run by now.
    await asyncio.sleep(0)
    mock_proxy_logging.failure_handler.assert_not_called()


@pytest.mark.asyncio
async def test_get_generic_data_propagates_when_reconnect_fails(mock_proxy_logging):
    """If reconnect itself does not succeed, propagate the original transport
    error and let the existing failure_handler / db_exceptions telemetry fire."""
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )

    client.db.litellm_config.find_first = AsyncMock(
        side_effect=httpx.ReadError("simulated transport blip")
    )
    client.attempt_db_reconnect = AsyncMock(return_value=False)

    with pytest.raises(httpx.ReadError):
        await client.get_generic_data(
            key="param_name",
            value="general_settings",
            table_name="config",
        )

    client.attempt_db_reconnect.assert_awaited_once()
    # Failure telemetry IS expected here — the read genuinely failed.
    await asyncio.sleep(0)
    mock_proxy_logging.failure_handler.assert_called_once()


# ---------------------------------------------------------------------------
# _engine_confirmed_dead flag-reset bug (B2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_confirmed_dead_persists_across_failed_heavy_reconnect(
    mock_proxy_logging,
):
    """Regression test for the flag-reset bug.

    Before the fix, `_run_reconnect_cycle` cleared
    `self._engine_confirmed_dead = False` *before* awaiting
    `_do_heavy_reconnect()`. If the heavy reconnect raised (e.g. timeout,
    missing DATABASE_URL, recreate failure), the flag was left cleared and the
    next attempt could demote to the lightweight path even though the engine
    was genuinely dead.

    The fix moves the reset into the success branch — the flag must stay True
    when heavy reconnect raises.
    """
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client._engine_confirmed_dead = True
    client._engine_pid = 0  # so `_is_engine_alive` is not consulted

    # Make the heavy reconnect path raise.
    client.db.recreate_prisma_client = AsyncMock(
        side_effect=RuntimeError("simulated heavy reconnect failure")
    )
    client._start_engine_watcher = AsyncMock()
    client._cleanup_engine_watcher = MagicMock()
    client._reap_all_zombies = MagicMock()

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        with pytest.raises(Exception):
            await client._run_reconnect_cycle(timeout_seconds=5.0)

    # The flag must STILL be True so the next attempt re-enters the heavy
    # branch instead of silently demoting to the lightweight path.
    assert client._engine_confirmed_dead is True


@pytest.mark.asyncio
async def test_heavy_reconnect_recovers_from_disconnected_prisma_client(
    mock_proxy_logging, mock_prisma_binary, disconnected_prisma
):
    """Once the active Prisma client is in the disconnected state, every DB
    call raises ClientNotConnectedError. The heavy reconnect path is the only
    way out, so it must not re-raise that same error while inspecting the
    broken client; otherwise `recreate_prisma_client` fails before it can
    build a replacement and the proxy loops on failed reconnects forever.

    The full real reconnect path (attempt_db_reconnect -> _run_reconnect_cycle
    -> recreate_prisma_client) must succeed from that wedged state.
    """
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db._original_prisma = disconnected_prisma
    client._engine_confirmed_dead = True
    client._start_engine_watcher = AsyncMock()

    replacement = MagicMock()
    replacement.connect = AsyncMock()
    mock_prisma_binary.Prisma.return_value = replacement

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        result = await client.attempt_db_reconnect(
            reason="unit_test_disconnected_client",
            force=True,
        )

    assert result is True
    assert client.db._original_prisma is replacement
    replacement.connect.assert_awaited_once()
    assert client._consecutive_reconnect_failures == 0
    assert client._engine_confirmed_dead is False


@pytest.mark.asyncio
async def test_db_health_watchdog_should_reconnect_degraded_writer(
    mock_proxy_logging,
):
    """LIT-3792: when the proxy booted during a primary outage (reads served
    by the replica, writer never connected), a healthy reader probe must not
    mask the degraded writer — the watchdog drives the writer reconnect."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    writer = MagicMock()
    reader = MagicMock()
    reader.query_raw = AsyncMock(return_value=[{"result": 1}])
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    routing._writer_unavailable = True
    client.db = routing
    client.attempt_db_reconnect = AsyncMock(return_value=True)
    client._db_health_watchdog_interval_seconds = 1
    client._db_watchdog_reconnect_timeout_seconds = 7.0
    client._db_health_watchdog_probe_timeout_seconds = 0.2

    with patch(
        "litellm.proxy.utils.asyncio.sleep",
        AsyncMock(side_effect=[None, asyncio.CancelledError()]),
    ):
        await client._db_health_watchdog_loop()

    client.attempt_db_reconnect.assert_awaited_once_with(
        reason="db_health_watchdog_writer_unavailable",
        timeout_seconds=7.0,
    )


@pytest.mark.asyncio
async def test_db_health_watchdog_should_not_reconnect_healthy_writer(
    mock_proxy_logging,
):
    """A healthy probe with no degraded writer must not trigger reconnects."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    writer = MagicMock()
    reader = MagicMock()
    reader.query_raw = AsyncMock(return_value=[{"result": 1}])
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    client.db = routing
    client.attempt_db_reconnect = AsyncMock(return_value=True)
    client._db_health_watchdog_interval_seconds = 1
    client._db_health_watchdog_probe_timeout_seconds = 0.2

    with patch(
        "litellm.proxy.utils.asyncio.sleep",
        AsyncMock(side_effect=[None, asyncio.CancelledError()]),
    ):
        await client._db_health_watchdog_loop()

    client.attempt_db_reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_reconnect_probe_success_clears_writer_unavailable(
    mock_proxy_logging,
):
    """If the writer probe inside _do_direct_reconnect succeeds (engine already
    reconnected by another path, e.g. an IAM token refresh), the early return
    skips recreate_prisma_client — the degraded-writer flag must still be
    cleared there or the watchdog fires reconnect attempts forever."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    writer = MagicMock()
    writer.query_raw = AsyncMock(return_value=[{"result": 1}])
    reader = MagicMock()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    routing._writer_unavailable = True
    client.db = routing
    client._start_engine_watcher = AsyncMock()

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        await client._run_reconnect_cycle(timeout_seconds=5.0)

    writer.query_raw.assert_awaited_once_with("SELECT 1")
    assert routing.writer_unavailable is False


@pytest.fixture
def no_backoff_sleep():
    """Make health_check's @backoff.on_exception retries run instantly."""
    with patch("backoff._async.asyncio.sleep", new=AsyncMock()):
        yield


@pytest.mark.asyncio
async def test_health_check_not_alerted_while_recreate_lock_held(
    mock_proxy_logging, no_backoff_sleep
):
    """A SELECT 1 that races a planned engine recreate (wrapper reconnection
    lock held) must not be reported to failure_handler as a DB exception."""
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.query_raw = AsyncMock(
        side_effect=httpx.ConnectError("All connection attempts failed")
    )

    await client.db._reconnection_lock.acquire()
    try:
        with pytest.raises(httpx.ConnectError):
            await client.health_check()
    finally:
        client.db._reconnection_lock.release()

    await asyncio.sleep(0)
    mock_proxy_logging.failure_handler.assert_not_called()


@pytest.mark.asyncio
async def test_health_check_not_alerted_when_engine_generation_moved(
    mock_proxy_logging, no_backoff_sleep
):
    """A recreate that completes (engine generation bumped) while the probe is
    in flight must not be reported, even though the lock is already free."""
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )

    def _bump_then_fail(*args, **kwargs):
        client.db._engine_generation += 1
        raise httpx.ConnectError("All connection attempts failed")

    client.db.query_raw = AsyncMock(side_effect=_bump_then_fail)

    with pytest.raises(httpx.ConnectError):
        await client.health_check()

    await asyncio.sleep(0)
    mock_proxy_logging.failure_handler.assert_not_called()


@pytest.mark.asyncio
async def test_health_check_reports_real_outage_with_correct_label(
    mock_proxy_logging, no_backoff_sleep
):
    """A persistent connection error with no recreate in flight is a real
    outage: it must still be reported, and the traceback must be labeled
    health_check() rather than the mislabeled disconnect()."""
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db.query_raw = AsyncMock(
        side_effect=httpx.ConnectError("All connection attempts failed")
    )

    with pytest.raises(httpx.ConnectError):
        await client.health_check()

    await asyncio.sleep(0.05)
    mock_proxy_logging.failure_handler.assert_called()
    _, kwargs = mock_proxy_logging.failure_handler.call_args
    assert kwargs["call_type"] == "health_check"
    assert "health_check()" in kwargs["traceback_str"]
    assert "disconnect()" not in kwargs["traceback_str"]
