"""
The cache dashboard buckets spend-log rows whose call_type is empty as "Unknown";
those rows are failed requests. The activity SQL must therefore report a
failed_rows count per group so the UI can chart failures as their own series.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.analytics_endpoints.analytics_endpoints import get_global_activity


@pytest.fixture
def mock_prisma(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    prisma = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    return prisma


@pytest.mark.asyncio
async def test_cache_hits_query_counts_failed_rows_per_group(mock_prisma: MagicMock):
    rows = [
        {
            "api_key": "my-key-alias",
            "call_type": "acompletion",
            "model": "gpt-5.1",
            "total_rows": 10,
            "cache_hit_true_rows": 3,
            "failed_rows": 2,
            "cached_completion_tokens": 100,
            "generated_completion_tokens": 900,
        }
    ]
    mock_prisma.db.query_raw = AsyncMock(return_value=rows)

    response = await get_global_activity(start_date="2026-07-01", end_date="2026-07-27")

    assert response == rows
    sql_query = mock_prisma.db.query_raw.call_args.args[0]
    assert 'SUM(CASE WHEN sl."status" = \'failure\' THEN 1 ELSE 0 END) AS failed_rows' in sql_query
    assert 'SUM(CASE WHEN sl."cache_hit" = \'True\' THEN 1 ELSE 0 END) AS cache_hit_true_rows' in sql_query


@pytest.mark.asyncio
async def test_cache_hits_requires_date_range(mock_prisma: MagicMock):
    with pytest.raises(HTTPException) as exc_info:
        await get_global_activity(start_date=None, end_date=None)

    assert exc_info.value.status_code == 400
    mock_prisma.db.query_raw.assert_not_called()
