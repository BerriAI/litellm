"""Pin ``PrismaClient`` read-side data operations.

Symbols pinned here:
  - ``PrismaClient.hash_token``
  - ``PrismaClient.jsonify_object``
  - ``PrismaClient.jsonify_team_object``
  - ``PrismaClient.check_view_exists``
  - ``PrismaClient.get_request_status``
  - ``PrismaClient.get_generic_data``
  - ``PrismaClient._query_first_with_cached_plan_fallback``
  - ``PrismaClient.get_data``
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_VerificationTokenView
from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_happy_returns_row(
    prisma_client: PrismaClient,
) -> None:
    expected = {"token": "abc", "team_spend": 1.0, "team_max_budget": 5.0}
    prisma_client.db.query_first = AsyncMock(return_value=expected)
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    result = await prisma_client._query_first_with_cached_plan_fallback(
        "SELECT * FROM x WHERE token = $1", "abc"
    )
    actual = {
        "result": result,
        "call_count": prisma_client.db.query_first.await_count,
        "args": prisma_client.db.query_first.await_args.args,
        "matches": result == expected,
    }
    assert actual == {
        "result": expected,
        "call_count": 1,
        "args": ("SELECT * FROM x WHERE token = $1", "abc"),
        "matches": True,
    }
    prisma_client.attempt_db_reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_reconnects_then_retries_identical_query(
    prisma_client: PrismaClient,
) -> None:
    original_query = 'SELECT * FROM "LiteLLM_VerificationToken" WHERE v.token = $1'
    expected = {"token": "abc", "team_spend": 1.0, "team_max_budget": 5.0}
    manager = MagicMock()
    query_first = AsyncMock(
        side_effect=[
            RuntimeError("cached plan must not change result type"),
            expected,
        ]
    )
    reconnect = AsyncMock(return_value=True)
    manager.attach_mock(query_first, "query_first")
    manager.attach_mock(reconnect, "attempt_db_reconnect")
    prisma_client.db.query_first = query_first
    prisma_client.attempt_db_reconnect = reconnect

    result = await prisma_client._query_first_with_cached_plan_fallback(
        original_query, "abc"
    )

    assert result == expected
    assert query_first.await_count == 2
    first_call, retry_call = query_first.await_args_list
    assert retry_call.args == first_call.args == (original_query, "abc")
    reconnect.assert_awaited_once()
    assert reconnect.await_args.kwargs.get("force", False) is False
    assert [name for name, *_ in manager.mock_calls] == [
        "query_first",
        "attempt_db_reconnect",
        "query_first",
    ]


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_never_deallocates(
    prisma_client: PrismaClient,
) -> None:
    expected = {"token": "abc"}
    prisma_client.db.query_first = AsyncMock(
        side_effect=[
            RuntimeError("cached plan must not change result type"),
            expected,
        ]
    )
    prisma_client.db.execute_raw = AsyncMock(return_value=0)
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)

    await prisma_client._query_first_with_cached_plan_fallback("SELECT 1")

    prisma_client.db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_propagates_when_retry_also_fails(
    prisma_client: PrismaClient,
) -> None:
    plan_error = RuntimeError("cached plan must not change result type")
    prisma_client.db.query_first = AsyncMock(side_effect=[plan_error, plan_error])
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)

    with pytest.raises(RuntimeError, match="cached plan must not change result type"):
        await prisma_client._query_first_with_cached_plan_fallback("SELECT 1")

    assert prisma_client.db.query_first.await_count == 2
    prisma_client.attempt_db_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_retries_when_reconnect_returns_false(
    prisma_client: PrismaClient,
) -> None:
    expected = {"token": "abc"}
    prisma_client.db.query_first = AsyncMock(
        side_effect=[
            RuntimeError("cached plan must not change result type"),
            expected,
        ]
    )
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=False)

    result = await prisma_client._query_first_with_cached_plan_fallback("SELECT 1")

    assert result == expected
    assert prisma_client.db.query_first.await_count == 2


@pytest.mark.asyncio
async def test_query_first_with_cached_plan_fallback_reraises_non_plan_errors(
    prisma_client: PrismaClient,
) -> None:
    prisma_client.db.query_first = AsyncMock(
        side_effect=RuntimeError("totally unrelated")
    )
    prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    with pytest.raises(RuntimeError, match="totally unrelated"):
        await prisma_client._query_first_with_cached_plan_fallback("SELECT 1")
    assert prisma_client.db.query_first.await_count == 1
    prisma_client.attempt_db_reconnect.assert_not_awaited()



@pytest.mark.asyncio
async def test_get_data_combined_view_returns_view_for_deprecated_key(
    prisma_client: PrismaClient,
) -> None:
    """Grace-period rotation, full get_data flow: the old hash misses the
    combined view, the deprecated-key table resolves it to the active token,
    and get_data must return the recursive lookup's finished view instead of
    re-running dict normalization on it (which raised TypeError and turned
    every grace-period request into a 401)."""
    old_hash = "hashed-old-token-grace-e2e"
    active_hash = "hashed-active-token-grace-e2e"
    active_row = {
        "token": active_hash,
        "team_models": None,
        "team_blocked": None,
        "team_members_with_roles": None,
        "user_id": None,
        "expires": None,
    }
    prisma_client.db.query_first = AsyncMock(side_effect=[None, active_row])
    prisma_client.db.litellm_deprecatedverificationtoken = MagicMock()
    prisma_client.db.litellm_deprecatedverificationtoken.find_first = AsyncMock(
        return_value=SimpleNamespace(
            active_token_id=active_hash,
            revoke_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    )

    response = await prisma_client.get_data(
        token=old_hash, table_name="combined_view", query_type="find_unique"
    )

    assert isinstance(response, LiteLLM_VerificationTokenView)
    assert response.token == active_hash
