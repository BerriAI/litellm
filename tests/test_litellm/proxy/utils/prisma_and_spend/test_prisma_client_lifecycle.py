"""Pin ``PrismaClient`` lifecycle methods.

Symbols pinned here:
  - ``PrismaClient.__init__``
  - ``PrismaClient.writer_db``
  - ``PrismaClient.connect``
  - ``PrismaClient.disconnect``
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import PrismaClient


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
        "iam_token_db_auth": pc.iam_token_db_auth,
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
        "iam_token_db_auth": None,
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
