"""/spend/logs treats end_date as a whole calendar day (#40216)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response

import litellm.proxy.spend_tracking.spend_management_endpoints as spend_endpoints
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_management_endpoints import (
    _spend_logs_daily_summary_sql,
    view_spend_logs,
)


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin")


def test_daily_summary_sql_uses_exclusive_upper_bound():
    sql, _ = _spend_logs_daily_summary_sql(
        start_date_iso="2026-09-07T00:00:00+00:00",
        end_date_iso="2026-09-08T00:00:00+00:00",
        api_key=None,
        request_id=None,
        user_id=None,
    )
    assert "\"startTime\" < ($2::timestamptz AT TIME ZONE 'UTC')" in sql
    assert "<= ($2" not in sql


@pytest.mark.asyncio
async def test_summary_queries_through_end_of_end_date(monkeypatch):
    prisma = MagicMock()
    prisma.db.query_raw = AsyncMock(
        return_value=[{"day": "2026-09-08", "api_key": "k", "user": "u", "model": "m", "spend": 1.5}]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    result = await view_spend_logs(
        fastapi_response=Response(),
        api_key=None,
        user_id=None,
        request_id=None,
        start_date="2026-09-08",
        end_date="2026-09-08",
        summarize=True,
        user_api_key_dict=_admin(),
    )

    params = prisma.db.query_raw.call_args[0][1:]
    assert params[0] == "2026-09-08T00:00:00+00:00"
    assert params[1] == "2026-09-09T00:00:00+00:00"
    # The real end-day row is returned and no synthetic zero row is appended.
    assert len(result) == 1
    assert result[0]["spend"] == 1.5


@pytest.mark.asyncio
async def test_unsummarized_filter_includes_end_date(monkeypatch):
    find_spend_logs = AsyncMock(return_value=[])
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(spend_endpoints, "_find_spend_logs", find_spend_logs)

    await view_spend_logs(
        fastapi_response=Response(),
        api_key=None,
        user_id=None,
        request_id=None,
        start_date="2026-09-07",
        end_date="2026-09-07",
        summarize=False,
        user_api_key_dict=_admin(),
    )

    where = find_spend_logs.call_args.kwargs["where"]
    assert where["startTime"] == {
        "gte": "2026-09-07T00:00:00+00:00",
        "lt": "2026-09-08T00:00:00+00:00",
    }
