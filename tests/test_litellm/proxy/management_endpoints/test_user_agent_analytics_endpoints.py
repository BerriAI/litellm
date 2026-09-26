from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy.management_endpoints import user_agent_analytics_endpoints as endpoints


@pytest.mark.asyncio
@pytest.mark.parametrize("period", ["daily", "weekly", "monthly"])
async def test_active_user_windows_include_the_reporting_day(monkeypatch: pytest.MonkeyPatch, period: str) -> None:
    instant: Final = datetime(2026, 9, 25, 17, tzinfo=timezone.utc)
    monkeypatch.setattr(litellm, "daily_usage_timezone", "Asia/Singapore")
    monkeypatch.setattr(endpoints, "datetime", SimpleNamespace(now=lambda zone: instant.astimezone(zone)))
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    handlers: Final = {
        "daily": endpoints.get_daily_active_users,
        "weekly": endpoints.get_weekly_active_users,
        "monthly": endpoints.get_monthly_active_users,
    }
    response: Final = await handlers[period](tag_filter=None, tag_filters=None)
    assert response.results == []
    assert prisma.db.query_raw.await_args.args[2] == "2026-09-27"


@pytest.mark.asyncio
async def test_per_user_analytics_uses_the_reporting_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    instant: Final = datetime(2026, 9, 25, 17, tzinfo=timezone.utc)
    monkeypatch.setattr(litellm, "daily_usage_timezone", "Asia/Singapore")
    monkeypatch.setattr(endpoints, "datetime", SimpleNamespace(now=lambda zone: instant.astimezone(zone)))
    prisma: Final = MagicMock()
    prisma.db.litellm_dailytagspend.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    response: Final = await endpoints.get_per_user_analytics(tag_filter=None, tag_filters=None, page=1, page_size=10)
    assert response.total_count == 0
    assert prisma.db.litellm_dailytagspend.find_many.await_args.kwargs["where"]["date"] == {
        "gte": "2026-08-28", "lte": "2026-09-27"
    }
