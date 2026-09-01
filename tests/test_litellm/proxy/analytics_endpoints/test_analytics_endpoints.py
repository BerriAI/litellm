"""
The cache dashboard chart is fed by /global/activity/cache_hits. Aggregation
lives server-side: the SQL groups per call_type (splitting cache hits vs
successful vs failed requests; failed spend logs have call_type '' today and
must surface as 'Unknown'), and the endpoint returns chart-ready groups,
totals for the stat cards, and the filter options for the UI dropdowns.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.analytics_endpoints.analytics_endpoints import get_global_activity
from litellm.proxy.analytics_endpoints.cache_activity import (
    ERROR_BREAKDOWN_SQL,
    GROUPS_SQL,
    CacheActivityGroup,
    compute_totals,
)

GROUP_ROWS = [
    {
        "call_type": "acompletion",
        "api_requests": 1000,
        "cache_hits": 300,
        "failed_requests": 200,
        "cached_completion_tokens": 12000,
        "generated_completion_tokens": 48000,
    },
    {
        "call_type": "Unknown",
        "api_requests": 0,
        "cache_hits": 0,
        "failed_requests": 110,
        "cached_completion_tokens": 0,
        "generated_completion_tokens": 0,
    },
]
ERROR_ROWS = [
    {"call_type": "acompletion", "error_code": "429", "error_class": "RateLimitError", "count": 150},
    {"call_type": "acompletion", "error_code": "401", "error_class": "AuthenticationError", "count": 50},
    {"call_type": "Unknown", "error_code": "Unknown", "error_class": "Unknown", "count": 110},
]
KEY_ALIAS_ROWS = [{"key_alias": "Unnamed Key"}, {"key_alias": "my-key"}]
MODEL_ROWS = [{"model": "gpt-5.1"}]


def build_prisma(query_raw: AsyncMock) -> MagicMock:
    prisma = MagicMock()
    prisma.db.query_raw = query_raw
    return prisma


def dispatching_query_raw() -> AsyncMock:
    async def dispatch(sql: str, *params: object) -> list[dict[str, object]]:
        if "error_code" in sql:
            return ERROR_ROWS
        if "GROUP BY" in sql:
            return GROUP_ROWS
        if "key_alias" in sql:
            return KEY_ALIAS_ROWS
        return MODEL_ROWS

    return AsyncMock(side_effect=dispatch)


@pytest.fixture
def mock_prisma(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    prisma = build_prisma(dispatching_query_raw())
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    return prisma


@pytest.mark.asyncio
async def test_returns_groups_totals_and_filter_options(mock_prisma: MagicMock):
    response = await get_global_activity(start_date="2026-07-01", end_date="2026-07-27", key_aliases=[], models=[])

    assert [group.call_type for group in response.groups] == ["acompletion", "Unknown"]
    assert response.groups[0].api_requests == 1000
    assert response.groups[0].failed_requests == 200
    assert response.totals.api_requests == 1000
    assert response.totals.cache_hits == 300
    assert response.totals.failed_requests == 310
    assert response.totals.cached_completion_tokens == 12000
    assert response.totals.cache_hit_ratio == pytest.approx((300 / 1610) * 100)
    assert response.filter_options.key_aliases == ["Unnamed Key", "my-key"]
    assert response.filter_options.models == ["gpt-5.1"]
    assert [(bucket.error_code, bucket.error_class, bucket.count) for bucket in response.error_breakdown] == [
        ("429", "RateLimitError", 150),
        ("401", "AuthenticationError", 50),
        ("Unknown", "Unknown", 110),
    ]


@pytest.mark.asyncio
async def test_filters_are_passed_to_sql_as_json_arrays(mock_prisma: MagicMock):
    await get_global_activity(
        start_date="2026-07-01",
        end_date="2026-07-27",
        key_aliases=["my-key"],
        models=["gpt-5.1", "claude-opus-4-8"],
    )

    filtered_calls = [
        call for call in mock_prisma.db.query_raw.call_args_list if call.args[0] in (GROUPS_SQL, ERROR_BREAKDOWN_SQL)
    ]
    assert len(filtered_calls) == 2
    for call in filtered_calls:
        assert call.args[3] == json.dumps(["my-key"])
        assert call.args[4] == json.dumps(["gpt-5.1", "claude-opus-4-8"])


@pytest.mark.asyncio
async def test_rejects_malformed_dates_with_400(mock_prisma: MagicMock):
    with pytest.raises(HTTPException) as exc_info:
        await get_global_activity(start_date="07/01/2026", end_date="2026-07-27", key_aliases=[], models=[])

    assert exc_info.value.status_code == 400
    mock_prisma.db.query_raw.assert_not_called()


def test_totals_ratio_is_zero_without_requests():
    totals = compute_totals([])

    assert totals.cache_hit_ratio == 0.0
    assert totals.api_requests == 0


def test_totals_denominator_includes_failed_requests():
    group = CacheActivityGroup(
        call_type="acompletion",
        api_requests=60,
        cache_hits=20,
        failed_requests=20,
        cached_completion_tokens=0,
        generated_completion_tokens=0,
    )

    assert compute_totals([group]).cache_hit_ratio == pytest.approx(20.0)


def test_groups_sql_splits_failures_and_labels_empty_call_type_unknown():
    assert "SUM(CASE WHEN sl.\"status\" = 'failure' THEN 1 ELSE 0 END)" in GROUPS_SQL
    assert "CASE WHEN sl.\"call_type\" = '' THEN 'Unknown' ELSE sl.\"call_type\" END" in GROUPS_SQL


def test_error_breakdown_sql_counts_only_failures_bucketed_by_code_and_class():
    assert "sl.\"status\" = 'failure'" in ERROR_BREAKDOWN_SQL
    assert "sl.\"metadata\"->'error_information'->>'error_code'" in ERROR_BREAKDOWN_SQL
    assert "sl.\"metadata\"->'error_information'->>'error_class'" in ERROR_BREAKDOWN_SQL
    assert "GROUP BY 1, 2, 3" in ERROR_BREAKDOWN_SQL
