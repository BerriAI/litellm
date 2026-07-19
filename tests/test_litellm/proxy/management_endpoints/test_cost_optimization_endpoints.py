from collections.abc import Mapping
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.management_endpoints.cost_optimization_endpoints import (
    build_optimized_request_log,
    cost_optimization_usage_logs,
)


def _row(metadata: Mapping[str, object], **overrides: object) -> dict[str, object]:
    return {
        "request_id": "req-1",
        "startTime": datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc),
        "model": "test-model",
        "custom_llm_provider": "openai",
        "total_tokens": 150,
        "spend": 0.02,
        "metadata": metadata,
        **overrides,
    }


def test_build_optimized_request_log_computes_both_savings():
    row = _row(
        {
            "compression_savings": {"tokens_saved": 100},
            "usage_object": {"cache_read_input_tokens": 50},
        }
    )
    with patch(
        "litellm.proxy.spend_tracking.savings._input_and_cache_read_cost",
        return_value=(0.001, 0.0002),
    ):
        result = build_optimized_request_log(row)

    assert result is not None
    assert result.optimization_type == "both"
    assert result.tokens_saved == 100
    assert result.cache_read_tokens == 50
    assert result.compression_savings_spend == 0.1
    assert result.prompt_caching_savings_spend == 0.04
    assert result.savings == 0.14
    assert result.original_cost == 0.16


def test_build_optimized_request_log_ignores_unoptimized_row():
    assert build_optimized_request_log(_row({"usage_object": {"prompt_tokens": 10}})) is None


@pytest.mark.asyncio
async def test_cost_optimization_usage_logs_returns_paginated_entries():
    db = MagicMock()
    query_raw_mock = AsyncMock(
        side_effect=[
            [{"total": 1}],
            [_row({"compression_savings": {"tokens_saved": 25}})],
        ]
    )
    db.query_raw = query_raw_mock
    prisma = MagicMock()
    prisma.db = db
    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma),
        patch(
            "litellm.proxy.spend_tracking.savings._input_and_cache_read_cost",
            return_value=(0.001, 0.001),
        ),
    ):
        result = await cost_optimization_usage_logs(
            page=1,
            page_size=50,
            start_date="2026-07-01",
            end_date="2026-07-31",
        )

    assert result.total == 1
    assert result.total_pages == 1
    assert result.logs[0].optimization_type == "compression"
    assert result.logs[0].tokens_saved == 25
    assert query_raw_mock.await_count == 2
