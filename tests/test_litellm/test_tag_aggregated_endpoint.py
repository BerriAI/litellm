import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath("../../"))

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.management_endpoints.common_daily_activity import (
    get_daily_activity_aggregated,
)

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )
    return prisma_client


@pytest.mark.asyncio
async def test_tag_daily_activity_aggregated_smoke(prisma_client):
    await prisma_client.connect()

    # use the last 7 days
    end = datetime.utcnow().date()
    start = end - timedelta(days=7)

    # call the shared aggregated function directly (as the endpoint does)
    resp = await get_daily_activity_aggregated(
        prisma_client=prisma_client,
        table_name="litellm_dailytagspend",
        entity_id_field="tag",
        entity_id=None,
        entity_metadata_field=None,
        start_date=str(start),
        end_date=str(end),
        model=None,
        api_key=None,
    )

    assert resp is not None
    assert hasattr(resp, "results")
    assert hasattr(resp, "metadata")
    # results can be empty in a fresh DB, but the shape must be correct
    assert resp.metadata.page == 1
    assert resp.metadata.total_pages == 1
    assert resp.metadata.has_more is False


