"""Pin ``PrismaClient`` lifecycle methods.

Symbols pinned here:
  - ``PrismaClient.__init__``
  - ``PrismaClient.writer_db``
  - ``PrismaClient.connect``
  - ``PrismaClient.disconnect``
  - ``PrismaClient.start_view_setup_task``
  - ``PrismaClient.stop_view_setup_task``
  - ``PrismaClient._run_view_setup``
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import PrismaClient

_PROBE_SQL = "SELECT to_regclass($1) IS NOT NULL AS present"


def _absent() -> list[dict[str, bool]]:
    return [{"present": False}]


def _present() -> list[dict[str, bool]]:
    return [{"present": True}]


def _wire_view_setup(prisma_client: PrismaClient, probe: AsyncMock) -> MagicMock:
    prisma_client.db.query_raw = probe
    prisma_client.check_view_exists = AsyncMock()
    prisma_client._set_spend_logs_row_count_in_proxy_state = AsyncMock()
    call_order = MagicMock()
    call_order.attach_mock(probe, "probe")
    call_order.attach_mock(prisma_client.check_view_exists, "views")
    call_order.attach_mock(prisma_client._set_spend_logs_row_count_in_proxy_state, "row_count")
    return call_order


@pytest.mark.asyncio
async def test_prismaclient_init_wires_default_config(
    patched_prisma_import: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    monkeypatch.delenv("IAM_TOKEN_DB_AUTH", raising=False)
    monkeypatch.delenv("PRISMA_RECONNECT_COOLDOWN_SECONDS", raising=False)
    monkeypatch.delenv("PRISMA_HEALTH_WATCHDOG_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("PRISMA_HEALTH_WATCHDOG_ENABLED", raising=False)
    monkeypatch.delenv("PRISMA_RECONNECT_ESCALATION_THRESHOLD", raising=False)

    proxy_logging = MagicMock()
    pc = PrismaClient(
        database_url="postgres://x:y@h:5432/db",
        proxy_logging_obj=proxy_logging,
    )
    pinned = {
        "token_auth": pc.token_auth,
        "db_reconnect_cooldown_seconds": pc._db_reconnect_cooldown_seconds,
        "db_health_watchdog_interval_seconds": pc._db_health_watchdog_interval_seconds,
        "db_health_watchdog_enabled": pc._db_health_watchdog_enabled,
        "reconnect_escalation_threshold": pc._reconnect_escalation_threshold,
        "consecutive_reconnect_failures": pc._consecutive_reconnect_failures,
        "engine_pid": pc._engine_pid,
        "watching_engine": pc._watching_engine,
        "proxy_logging_obj_set": pc.proxy_logging_obj is proxy_logging,
        "db_reconnect_lock_is_lock": isinstance(pc._db_reconnect_lock, asyncio.Lock),
    }
    assert pinned == {
        "token_auth": None,
        "db_reconnect_cooldown_seconds": 15,
        "db_health_watchdog_interval_seconds": 30,
        "db_health_watchdog_enabled": True,
        "reconnect_escalation_threshold": 3,
        "consecutive_reconnect_failures": 0,
        "engine_pid": 0,
        "watching_engine": False,
        "proxy_logging_obj_set": True,
        "db_reconnect_lock_is_lock": True,
    }


def test_prismaclient_init_honors_env_overrides(
    patched_prisma_import: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRISMA_RECONNECT_COOLDOWN_SECONDS", "42")
    monkeypatch.setenv("PRISMA_HEALTH_WATCHDOG_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("PRISMA_HEALTH_WATCHDOG_ENABLED", "false")
    monkeypatch.setenv("PRISMA_RECONNECT_ESCALATION_THRESHOLD", "7")
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    monkeypatch.delenv("IAM_TOKEN_DB_AUTH", raising=False)

    pc = PrismaClient(
        database_url="postgres://x:y@h:5432/db",
        proxy_logging_obj=MagicMock(),
    )
    pinned = {
        "db_reconnect_cooldown_seconds": pc._db_reconnect_cooldown_seconds,
        "db_health_watchdog_interval_seconds": pc._db_health_watchdog_interval_seconds,
        "db_health_watchdog_enabled": pc._db_health_watchdog_enabled,
        "reconnect_escalation_threshold": pc._reconnect_escalation_threshold,
    }
    assert pinned == {
        "db_reconnect_cooldown_seconds": 42,
        "db_health_watchdog_interval_seconds": 60,
        "db_health_watchdog_enabled": False,
        "reconnect_escalation_threshold": 7,
    }


def test_prismaclient_init_raises_when_prisma_not_generated() -> None:
    """If ``from prisma import Prisma`` fails, the init re-raises with the
    'prisma generate' guidance message.
    """
    import prisma as _prisma_pkg

    had_prisma_attr = "Prisma" in _prisma_pkg.__dict__
    previous_prisma_attr = _prisma_pkg.__dict__.get("Prisma")
    if had_prisma_attr:
        del _prisma_pkg.Prisma  # type: ignore[attr-defined]
    try:
        with pytest.raises(Exception, match="prisma generate"):
            PrismaClient(
                database_url="postgres://x:y@h:5432/db",
                proxy_logging_obj=MagicMock(),
            )
    finally:
        if had_prisma_attr:
            _prisma_pkg.Prisma = previous_prisma_attr  # type: ignore[attr-defined]


def test_writer_db_returns_db_when_no_routing(prisma_client: PrismaClient) -> None:
    actual = {
        "writer_is_db": prisma_client.writer_db is prisma_client.db,
        "type_consistency": type(prisma_client.writer_db) is type(prisma_client.db),
        "callable_query_raw": callable(prisma_client.writer_db.query_raw),
    }
    assert actual == {
        "writer_is_db": True,
        "type_consistency": True,
        "callable_query_raw": True,
    }


def test_writer_db_unwraps_routing_wrapper(prisma_client: PrismaClient) -> None:
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    inner_writer = MagicMock(name="WriterInsideRouter")

    class _FakeRouting(RoutingPrismaWrapper):  # type: ignore[misc]
        def __init__(self) -> None:
            self._writer = inner_writer

    prisma_client.db = _FakeRouting()
    assert prisma_client.writer_db is inner_writer


def test_writer_db_error_when_db_attribute_missing(prisma_client: PrismaClient) -> None:
    del prisma_client.db
    with pytest.raises(AttributeError):
        _ = prisma_client.writer_db


@pytest.mark.asyncio
async def test_connect_invokes_underlying_when_disconnected(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.is_connected = MagicMock(return_value=False)
    prisma_client.db.connect = AsyncMock()
    await prisma_client.connect()
    actual = {
        "connect_called": prisma_client.db.connect.await_count,
        "is_connected_called": prisma_client.db.is_connected.call_count,
        "no_failure_handler": prisma_client.proxy_logging_obj.failure_handler.await_count,
    }
    assert actual == {
        "connect_called": 1,
        "is_connected_called": 1,
        "no_failure_handler": 0,
    }


@pytest.mark.asyncio
async def test_connect_is_noop_when_already_connected(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.is_connected = MagicMock(return_value=True)
    prisma_client.db.connect = AsyncMock()
    await prisma_client.connect()
    assert prisma_client.db.connect.await_count == 0


@pytest.mark.asyncio
async def test_connect_invokes_failure_handler_and_raises_on_error(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.is_connected = MagicMock(return_value=False)
    prisma_client.db.connect = AsyncMock(side_effect=RuntimeError("network down"))
    with pytest.raises(RuntimeError, match="network down"):
        await prisma_client.connect()


@pytest.mark.asyncio
async def test_disconnect_calls_underlying(prisma_client: PrismaClient) -> None:
    prisma_client.db.disconnect = AsyncMock()
    await prisma_client.disconnect()
    actual = {
        "disconnect_called": prisma_client.db.disconnect.await_count,
        "failure_handler_called": prisma_client.proxy_logging_obj.failure_handler.await_count,
        "type": type(prisma_client.db.disconnect).__name__,
    }
    assert actual == {
        "disconnect_called": 1,
        "failure_handler_called": 0,
        "type": "AsyncMock",
    }


@pytest.mark.asyncio
async def test_disconnect_raises_when_underlying_fails(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.disconnect = AsyncMock(side_effect=RuntimeError("disconnect boom"))
    with pytest.raises(RuntimeError, match="disconnect boom"):
        await prisma_client.disconnect()


@pytest.mark.asyncio
async def test_view_setup_waits_for_the_spend_logs_table_before_creating_views(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a fresh database the migrations Job can still be running when the proxy
    boots. The views reference ``LiteLLM_SpendLogs``, so creating them before the
    table exists raised inside a fire-and-forget task and the views never appeared."""
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
    probe = AsyncMock(side_effect=[_absent(), _absent(), _present()])
    call_order = _wire_view_setup(prisma_client, probe)

    outcome = await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=5)

    actual = {
        "outcome": outcome,
        "calls": [call[0] for call in call_order.mock_calls],
        "probe_args": probe.await_args.args,
    }
    assert actual == {
        "outcome": "ready",
        "calls": ["probe", "probe", "probe", "views", "row_count"],
        "probe_args": (_PROBE_SQL, '"public"."LiteLLM_SpendLogs"'),
    }


@pytest.mark.asyncio
async def test_view_setup_probes_the_configured_database_schema(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_SCHEMA", "litellm_tenant")
    probe = AsyncMock(return_value=_present())
    _wire_view_setup(prisma_client, probe)

    await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=5)

    assert probe.await_args.args == (_PROBE_SQL, '"litellm_tenant"."LiteLLM_SpendLogs"')


@pytest.mark.asyncio
async def test_view_setup_gives_up_when_the_table_never_appears(prisma_client: PrismaClient) -> None:
    probe = AsyncMock(return_value=_absent())
    _wire_view_setup(prisma_client, probe)

    outcome = await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=0.02)

    actual = {
        "outcome": outcome,
        "kept_polling": probe.await_count > 1,
        "views_attempted": prisma_client.check_view_exists.await_count,
        "row_count_attempted": prisma_client._set_spend_logs_row_count_in_proxy_state.await_count,
    }
    assert actual == {
        "outcome": "timed_out",
        "kept_polling": True,
        "views_attempted": 0,
        "row_count_attempted": 0,
    }


@pytest.mark.asyncio
async def test_view_setup_retries_when_view_creation_fails_mid_migration(prisma_client: PrismaClient) -> None:
    """``LiteLLM_SpendLogs`` lands early in the migration set while
    ``LiteLLM_VerificationTokenView`` references columns the newest migrations add,
    so the first attempt after the table appears can still fail."""
    probe = AsyncMock(return_value=_present())
    call_order = _wire_view_setup(prisma_client, probe)
    prisma_client.check_view_exists.side_effect = [RuntimeError('column "tpd_limit" does not exist'), None]

    outcome = await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=5)

    actual = {
        "outcome": outcome,
        "calls": [call[0] for call in call_order.mock_calls],
    }
    assert actual == {
        "outcome": "ready",
        "calls": ["probe", "views", "probe", "views", "row_count"],
    }


@pytest.mark.asyncio
async def test_view_setup_retries_when_the_table_probe_itself_fails(prisma_client: PrismaClient) -> None:
    probe = AsyncMock(side_effect=[RuntimeError("connection reset"), _present()])
    call_order = _wire_view_setup(prisma_client, probe)

    outcome = await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=5)

    actual = {
        "outcome": outcome,
        "calls": [call[0] for call in call_order.mock_calls],
    }
    assert actual == {
        "outcome": "ready",
        "calls": ["probe", "probe", "views", "row_count"],
    }


@pytest.mark.asyncio
async def test_run_view_setup_logs_an_error_naming_the_table_on_timeout(
    prisma_client: PrismaClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
    _wire_view_setup(prisma_client, AsyncMock(return_value=_absent()))

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        outcome = await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=0.01)

    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    actual = {
        "outcome": outcome,
        "error_count": len(errors),
        "names_table": '"public"."LiteLLM_SpendLogs"' in errors[0],
        "tells_operator_to_migrate": "migrations" in errors[0] and "restart" in errors[0],
    }
    assert actual == {
        "outcome": "timed_out",
        "error_count": 1,
        "names_table": True,
        "tells_operator_to_migrate": True,
    }


@pytest.mark.asyncio
async def test_run_view_setup_reports_the_last_error_when_views_keep_failing_on_a_present_table(
    prisma_client: PrismaClient, caplog: pytest.LogCaptureFixture
) -> None:
    """A database role without CREATE on the schema fails every attempt even though
    the table is there, so the timeout must blame that error, not missing migrations."""
    _wire_view_setup(prisma_client, AsyncMock(return_value=_present()))
    prisma_client.check_view_exists.side_effect = RuntimeError("permission denied for schema public")

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        outcome = await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=0.01)

    errors = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    actual = {
        "outcome": outcome,
        "error_count": len(errors),
        "names_the_error": "permission denied for schema public" in errors[0],
        "blames_missing_migrations": "did not appear" in errors[0],
        "tells_operator_to_restart": "restart" in errors[0],
    }
    assert actual == {
        "outcome": "timed_out",
        "error_count": 1,
        "names_the_error": True,
        "blames_missing_migrations": False,
        "tells_operator_to_restart": True,
    }


@pytest.mark.asyncio
async def test_run_view_setup_stays_quiet_when_views_are_ready(
    prisma_client: PrismaClient, caplog: pytest.LogCaptureFixture
) -> None:
    _wire_view_setup(prisma_client, AsyncMock(return_value=_present()))

    with caplog.at_level(logging.ERROR, logger="LiteLLM Proxy"):
        outcome = await prisma_client._run_view_setup(poll_interval_seconds=0.001, deadline_seconds=0.01)

    actual = {
        "outcome": outcome,
        "errors": [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR],
    }
    assert actual == {"outcome": "ready", "errors": []}


@pytest.mark.asyncio
async def test_stop_view_setup_task_cancels_a_task_parked_between_polls(prisma_client: PrismaClient) -> None:
    probe = AsyncMock(return_value=_absent())
    _wire_view_setup(prisma_client, probe)

    prisma_client.start_view_setup_task()
    task = prisma_client._view_setup_task
    await asyncio.sleep(0)
    await asyncio.wait_for(prisma_client.stop_view_setup_task(), timeout=1)

    actual = {
        "probed_before_parking": probe.await_count,
        "task_cancelled": task is not None and task.cancelled(),
        "reference_cleared": prisma_client._view_setup_task,
        "views_attempted": prisma_client.check_view_exists.await_count,
    }
    assert actual == {
        "probed_before_parking": 1,
        "task_cancelled": True,
        "reference_cleared": None,
        "views_attempted": 0,
    }


@pytest.mark.asyncio
async def test_stop_view_setup_task_is_a_noop_without_a_task(prisma_client: PrismaClient) -> None:
    await asyncio.wait_for(prisma_client.stop_view_setup_task(), timeout=1)
    assert prisma_client._view_setup_task is None


@pytest.mark.asyncio
async def test_start_view_setup_task_twice_keeps_the_first_task(prisma_client: PrismaClient) -> None:
    _wire_view_setup(prisma_client, AsyncMock(return_value=_absent()))

    prisma_client.start_view_setup_task()
    first = prisma_client._view_setup_task
    prisma_client.start_view_setup_task()
    second = prisma_client._view_setup_task
    await asyncio.wait_for(prisma_client.stop_view_setup_task(), timeout=1)

    actual = {
        "first_is_task": isinstance(first, asyncio.Task),
        "second_is_first": second is first,
    }
    assert actual == {"first_is_task": True, "second_is_first": True}
