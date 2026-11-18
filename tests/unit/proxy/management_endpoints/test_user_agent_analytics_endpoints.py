from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.user_agent_analytics_endpoints import (
    ActiveUsersAnalyticsResponse,
    DistinctTagResponse,
    DistinctTagsResponse,
    TagActiveUsersResponse,
    get_daily_active_users,
    get_distinct_user_agent_tags,
)


def _prisma_returning(rows: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(db=SimpleNamespace(query_raw=AsyncMock(return_value=rows)))


@pytest.mark.asyncio
async def test_distinct_user_agent_tags_are_read_from_the_raw_query_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    prisma: Final = _prisma_returning([{"tag": "User-Agent:curl", "usage_count": 7}, {"tag": "prod", "usage_count": 2}])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    response: Final = await get_distinct_user_agent_tags(user_api_key_dict=UserAPIKeyAuth())

    assert response == DistinctTagsResponse(
        results=[DistinctTagResponse(tag="User-Agent:curl"), DistinctTagResponse(tag="prod")]
    )
    assert len(prisma.db.query_raw.await_args.args) == 1


@pytest.mark.asyncio
async def test_daily_active_users_forwards_each_tag_filter_as_a_query_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma: Final = _prisma_returning([{"tag": "User-Agent:curl", "active_users": 3, "date": "2026-08-01"}])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)

    response: Final = await get_daily_active_users(
        tag_filter=None,
        tag_filters=["User-Agent:curl", "prod"],
        user_api_key_dict=UserAPIKeyAuth(),
    )

    assert response == ActiveUsersAnalyticsResponse(
        results=[TagActiveUsersResponse(tag="User-Agent:curl", active_users=3, date="2026-08-01")]
    )
    sql_query, *params = prisma.db.query_raw.await_args.args
    assert params[2:] == ["User-Agent:curl", "prod"]
    assert "dts.tag = $3 OR dts.tag = $4" in sql_query
