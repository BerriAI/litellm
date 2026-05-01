"""
Unit tests for `call_with_db_reconnect_retry` — the canonical "try DB read,
on transport error reconnect once and retry once" helper.

Covers the regression in issue #25143 where read paths (e.g.
`PrismaClient.get_generic_data`) lost their reconnect-and-retry-once branch in
LiteLLM 1.83.x and started emitting `db_exceptions` alerts on transient
`httpx.ReadError` flaps that used to self-heal in 1.82.6.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from prisma.errors import UniqueViolationError

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.db.exception_handler import call_with_db_reconnect_retry


def _make_client(
    *,
    attempt_db_reconnect_return: bool = True,
    has_attempt_db_reconnect: bool = True,
):
    """Build a minimal stand-in for PrismaClient that exposes only the surface
    `call_with_db_reconnect_retry` actually pokes at."""
    client = MagicMock()
    if has_attempt_db_reconnect:
        client.attempt_db_reconnect = AsyncMock(
            return_value=attempt_db_reconnect_return
        )
    else:
        # `hasattr(client, "attempt_db_reconnect")` must return False — MagicMock
        # auto-creates attributes, so we wipe it out via `spec`.
        client = MagicMock(spec=[])
    client._db_auth_reconnect_timeout_seconds = 2.0
    client._db_auth_reconnect_lock_timeout_seconds = 0.1
    return client


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_returns_value_on_first_success():
    """Happy path: factory succeeds first call, no reconnect attempted."""
    client = _make_client()

    async def _factory():
        return {"id": 1}

    result = await call_with_db_reconnect_retry(client, _factory, reason="happy_path")

    assert result == {"id": 1}
    client.attempt_db_reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_retries_after_transport_error():
    """Transport error on first call → reconnect → second call succeeds."""
    client = _make_client(attempt_db_reconnect_return=True)

    invocations = []

    async def _factory():
        invocations.append(None)
        if len(invocations) == 1:
            raise httpx.ReadError("transport blip")
        return {"id": 1}

    result = await call_with_db_reconnect_retry(
        client, _factory, reason="prisma_get_generic_data_config_lookup_failure"
    )

    assert result == {"id": 1}
    assert len(invocations) == 2
    client.attempt_db_reconnect.assert_awaited_once()
    call_kwargs = client.attempt_db_reconnect.await_args.kwargs
    assert call_kwargs["reason"] == "prisma_get_generic_data_config_lookup_failure"


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_does_not_retry_on_data_layer_error():
    """Data-layer errors (e.g. UniqueViolationError) are NOT transport errors —
    propagate immediately, do not reconnect."""
    client = _make_client()

    async def _factory():
        raise UniqueViolationError(
            data={"user_facing_error": {"meta": {}}},
            message="Unique constraint failed",
        )

    with pytest.raises(UniqueViolationError):
        await call_with_db_reconnect_retry(client, _factory, reason="data_layer_test")

    client.attempt_db_reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_propagates_when_reconnect_fails():
    """Transport error, but reconnect returns False → propagate the original
    exception. Do not call factory a second time."""
    client = _make_client(attempt_db_reconnect_return=False)

    invocations = []

    async def _factory():
        invocations.append(None)
        raise httpx.ReadError("transport blip")

    with pytest.raises(httpx.ReadError):
        await call_with_db_reconnect_retry(client, _factory, reason="reconnect_fails")

    assert len(invocations) == 1
    client.attempt_db_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_propagates_after_second_transport_error():
    """Transport error, reconnect succeeds, retry also raises transport error →
    propagate. At most one retry by construction (no infinite loop)."""
    client = _make_client(attempt_db_reconnect_return=True)

    invocations = []

    async def _factory():
        invocations.append(None)
        raise httpx.ReadError("still failing")

    with pytest.raises(httpx.ReadError):
        await call_with_db_reconnect_retry(
            client, _factory, reason="second_transport_error"
        )

    assert len(invocations) == 2
    client.attempt_db_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_skips_when_no_attempt_db_reconnect_attr():
    """Older PrismaClient stand-ins / partial mocks may not expose
    `attempt_db_reconnect`. The helper must not crash — just propagate the
    original exception. Mirrors the `hasattr` guard from
    `auth_checks._fetch_key_object_from_db_with_reconnect`."""
    client = _make_client(has_attempt_db_reconnect=False)

    async def _factory():
        raise httpx.ReadError("transport blip")

    with pytest.raises(httpx.ReadError):
        await call_with_db_reconnect_retry(client, _factory, reason="no_reconnect_attr")


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_invokes_factory_twice_not_same_coro():
    """Guard against the obvious bug of awaiting the same coroutine twice
    (`RuntimeError: cannot reuse already awaited coroutine`). The helper must
    call the factory a fresh time on retry, not cache an awaitable."""
    client = _make_client(attempt_db_reconnect_return=True)

    factory_call_count = 0

    async def _factory():
        nonlocal factory_call_count
        factory_call_count += 1
        if factory_call_count == 1:
            raise httpx.ReadError("transport blip")
        return "ok"

    result = await call_with_db_reconnect_retry(
        client, _factory, reason="fresh_coro_on_retry"
    )

    assert result == "ok"
    assert factory_call_count == 2


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_passes_explicit_timeouts():
    """Explicit timeout_seconds / lock_timeout_seconds override the auth
    defaults read off the prisma_client object."""
    client = _make_client(attempt_db_reconnect_return=True)

    async def _factory():
        if not hasattr(_factory, "_called"):
            _factory._called = True  # type: ignore[attr-defined]
            raise httpx.ReadError("transport blip")
        return "ok"

    result = await call_with_db_reconnect_retry(
        client,
        _factory,
        reason="explicit_timeouts",
        timeout_seconds=5.5,
        lock_timeout_seconds=0.25,
    )

    assert result == "ok"
    call_kwargs = client.attempt_db_reconnect.await_args.kwargs
    assert call_kwargs["timeout_seconds"] == 5.5
    assert call_kwargs["lock_timeout_seconds"] == 0.25


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_uses_auth_defaults_when_unset():
    """When timeouts are not provided, helper reads
    `_db_auth_reconnect_timeout_seconds` / `_db_auth_reconnect_lock_timeout_seconds`
    off the prisma_client (matching the auth path's existing convention)."""
    client = _make_client(attempt_db_reconnect_return=True)
    client._db_auth_reconnect_timeout_seconds = 3.0
    client._db_auth_reconnect_lock_timeout_seconds = 0.5

    async def _factory():
        if not hasattr(_factory, "_called"):
            _factory._called = True  # type: ignore[attr-defined]
            raise httpx.ReadError("transport blip")
        return "ok"

    await call_with_db_reconnect_retry(client, _factory, reason="defaults")

    call_kwargs = client.attempt_db_reconnect.await_args.kwargs
    assert call_kwargs["timeout_seconds"] == 3.0
    assert call_kwargs["lock_timeout_seconds"] == 0.5


@pytest.mark.asyncio
async def test_call_with_db_reconnect_retry_preserves_original_error_when_reconnect_raises():
    """If `attempt_db_reconnect` itself raises (lock cancellation, timer
    error, unexpected internal failure), the helper must surface the
    *original* transport error to telemetry — not the reconnect exception.
    Otherwise `failure_handler` / `db_exceptions` alerts log the wrong
    error string and the actual DB transport problem becomes invisible.

    The reconnect error is chained as the `__cause__` for debuggability."""
    client = MagicMock()
    reconnect_exc = RuntimeError("simulated reconnect lock cancellation")
    client.attempt_db_reconnect = AsyncMock(side_effect=reconnect_exc)
    client._db_auth_reconnect_timeout_seconds = 2.0
    client._db_auth_reconnect_lock_timeout_seconds = 0.1

    original_exc = httpx.ReadError("transport blip")

    async def _factory():
        raise original_exc

    with pytest.raises(httpx.ReadError) as exc_info:
        await call_with_db_reconnect_retry(
            client, _factory, reason="reconnect_itself_raises"
        )

    assert exc_info.value is original_exc
    assert exc_info.value.__cause__ is reconnect_exc
    client.attempt_db_reconnect.assert_awaited_once()
