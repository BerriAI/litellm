from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.spend_tracking.ptu_feature_flag import PTU_COST_ATTRIBUTION_ENV_VAR


from litellm.proxy.management_endpoints.common_daily_activity import (
    _adjust_dates_for_timezone,
    _build_aggregated_sql_query,
    _build_entity_rollup_sql_query,
    _is_user_agent_tag,
    _record_to_spend_metrics,
    get_api_key_metadata,
    get_daily_activity,
    get_daily_activity_aggregated,
    update_metrics,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailySpendMetadata,
    SpendMetrics,
)


@pytest.mark.asyncio
async def test_get_daily_activity_empty_entity_id_list():
    # Mock PrismaClient
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    # Mock the table methods
    mock_table = MagicMock()
    mock_table.count = AsyncMock(return_value=0)
    mock_table.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    # Set the table name dynamically
    mock_prisma.db.litellm_dailyspend = mock_table

    # Call the function with empty entity_id list
    result = await get_daily_activity(
        prisma_client=mock_prisma,
        table_name="litellm_dailyspend",
        entity_id_field="team_id",
        entity_id=[],
        entity_metadata_field=None,
        start_date="2024-01-01",
        end_date="2024-01-02",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
    )

    # Verify the where conditions were set correctly
    mock_table.find_many.assert_called_once()
    call_args = mock_table.find_many.call_args[1]
    where_conditions = call_args["where"]

    # Check that team_id is set to empty list
    assert "team_id" in where_conditions
    assert where_conditions["team_id"] == {"in": []}


@pytest.mark.asyncio
async def test_get_daily_activity_order_has_id_tiebreaker():
    """Regression for #30164.

    ``date`` alone is not a unique sort key for either
    ``LiteLLM_DailyUserSpend`` or ``LiteLLM_DailyTeamSpend`` -- a busy
    tenant has many rows per date (one per api_key, model, model_group,
    provider, endpoint, ...).  Offset pagination over a non-unique sort
    landed on arbitrary page boundaries between queries, so summing
    per-page totals across pages produced non-deterministic results
    (sometimes inflated, sometimes deflated).  The tiebreaker on the
    UUID primary key pins the row order so a client paging through all
    results gets the correct total.
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_table = MagicMock()
    mock_table.count = AsyncMock(return_value=0)
    mock_table.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_dailyspend = mock_table

    await get_daily_activity(
        prisma_client=mock_prisma,
        table_name="litellm_dailyspend",
        entity_id_field="team_id",
        entity_id="team-1",
        entity_metadata_field=None,
        start_date="2024-01-01",
        end_date="2024-01-02",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
    )

    mock_table.find_many.assert_called_once()
    order = mock_table.find_many.call_args[1]["order"]
    assert order == [{"date": "desc"}, {"id": "asc"}], (
        f"order must include the id tiebreaker after date for stable offset pagination (see #30164); got {order!r}"
    )


def test_is_user_agent_tag():
    """Test _is_user_agent_tag function."""
    # Test None and empty string
    assert _is_user_agent_tag(None) is False
    assert _is_user_agent_tag("") is False

    # Test user-agent variations (should return True)
    assert _is_user_agent_tag("user-agent:chrome") is True
    assert _is_user_agent_tag("user agent:firefox") is True
    assert _is_user_agent_tag("USER-AGENT:safari") is True
    assert _is_user_agent_tag("User Agent:edge") is True
    assert _is_user_agent_tag("  user-agent:opera  ") is True  # with whitespace

    # Test regular tags (should return False)
    assert _is_user_agent_tag("production") is False
    assert _is_user_agent_tag("tag:value") is False
    assert _is_user_agent_tag("user-agent-tag") is False  # no colon


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_with_endpoint_breakdown():
    """Test that endpoint breakdown is included in aggregated daily activity."""
    # Mock PrismaClient
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    # query_raw now returns rollup rows produced by GROUPING SETS, each
    # tagged with its grouping level via GROUPING_ID(). The dispatcher
    # places each row directly in its bucket without Python-side summing.
    # GROUPING_ID values for relevant levels (date, api_key, model,
    # model_group, custom_llm_provider, mcp, endpoint):
    #   () grand total                  = 127
    #   (date)                          =  63
    #   (date, endpoint)                =  62
    #   (date, endpoint, api_key)       =  30
    base = {
        "model": None,
        "model_group": None,
        "custom_llm_provider": None,
        "mcp_namespaced_tool_name": None,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 0,
        "compression_savings_spend": 0.0,
        "prompt_caching_savings_spend": 0.0,
        "autorouter_savings_spend": 0.0,
        "failed_requests": 0,
    }
    mock_rows = [
        # (date, endpoint) — rolls up across api_keys and models
        {
            **base,
            "date": "2024-01-01",
            "endpoint": "/v1/chat/completions",
            "api_key": None,
            "group_level": 62,
            "spend": 15.0,
            "prompt_tokens": 150,
            "completion_tokens": 75,
            "api_requests": 2,
            "successful_requests": 2,
        },
        {
            **base,
            "date": "2024-01-01",
            "endpoint": "/v1/embeddings",
            "api_key": None,
            "group_level": 62,
            "spend": 3.0,
            "prompt_tokens": 30,
            "completion_tokens": 0,
            "api_requests": 1,
            "successful_requests": 1,
        },
        # (date, endpoint, api_key) — populates the per-key sub-bucket
        {
            **base,
            "date": "2024-01-01",
            "endpoint": "/v1/chat/completions",
            "api_key": "key-1",
            "group_level": 30,
            "spend": 15.0,
            "prompt_tokens": 150,
            "completion_tokens": 75,
            "api_requests": 2,
            "successful_requests": 2,
        },
        {
            **base,
            "date": "2024-01-01",
            "endpoint": "/v1/embeddings",
            "api_key": "key-2",
            "group_level": 30,
            "spend": 3.0,
            "prompt_tokens": 30,
            "completion_tokens": 0,
            "api_requests": 1,
            "successful_requests": 1,
        },
        # (date) — per-date totals
        {
            **base,
            "date": "2024-01-01",
            "endpoint": None,
            "api_key": None,
            "group_level": 63,
            "spend": 18.0,
            "prompt_tokens": 180,
            "completion_tokens": 75,
            "api_requests": 3,
            "successful_requests": 3,
        },
        # () — grand total
        {
            **base,
            "date": None,
            "endpoint": None,
            "api_key": None,
            "group_level": 127,
            "spend": 18.0,
            "prompt_tokens": 180,
            "completion_tokens": 75,
            "api_requests": 3,
            "successful_requests": 3,
        },
    ]

    mock_prisma.db.query_raw = AsyncMock(return_value=mock_rows)
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    # Call the function
    result = await get_daily_activity_aggregated(
        prisma_client=mock_prisma,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None,
        entity_metadata_field=None,
        start_date="2024-01-01",
        end_date="2024-01-01",
        model=None,
        api_key=None,
    )

    # Verify the results
    assert len(result.results) == 1
    daily_data = result.results[0]
    assert daily_data.date.strftime("%Y-%m-%d") == "2024-01-01"

    # Verify endpoint breakdown exists
    assert "endpoints" in daily_data.breakdown.model_fields
    assert len(daily_data.breakdown.endpoints) == 2

    # Verify /v1/chat/completions endpoint breakdown
    assert "/v1/chat/completions" in daily_data.breakdown.endpoints
    chat_endpoint = daily_data.breakdown.endpoints["/v1/chat/completions"]
    assert chat_endpoint.metrics.spend == 15.0  # 10.0 + 5.0
    assert chat_endpoint.metrics.prompt_tokens == 150  # 100 + 50
    assert chat_endpoint.metrics.completion_tokens == 75  # 50 + 25

    # Verify /v1/embeddings endpoint breakdown
    assert "/v1/embeddings" in daily_data.breakdown.endpoints
    embeddings_endpoint = daily_data.breakdown.endpoints["/v1/embeddings"]
    assert embeddings_endpoint.metrics.spend == 3.0
    assert embeddings_endpoint.metrics.prompt_tokens == 30
    assert embeddings_endpoint.metrics.completion_tokens == 0

    # Verify API key breakdowns within endpoints
    assert "key-1" in chat_endpoint.api_key_breakdown
    assert chat_endpoint.api_key_breakdown["key-1"].metrics.spend == 15.0
    assert "key-2" in embeddings_endpoint.api_key_breakdown
    assert embeddings_endpoint.api_key_breakdown["key-2"].metrics.spend == 3.0

    # Verify query_raw was called (not find_many)
    mock_prisma.db.query_raw.assert_called_once()


@pytest.mark.asyncio
async def test_get_api_key_metadata_returns_active_key_metadata():
    """Test that get_api_key_metadata should return metadata for active keys."""
    mock_prisma = MagicMock()

    # Mock active key record
    mock_active_key = MagicMock()
    mock_active_key.token = "active-key-hash-123"
    mock_active_key.key_alias = "my-active-key"
    mock_active_key.team_id = "team-abc"

    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[mock_active_key])

    result = await get_api_key_metadata(
        prisma_client=mock_prisma,
        api_keys={"active-key-hash-123"},
    )

    assert "active-key-hash-123" in result
    assert result["active-key-hash-123"]["key_alias"] == "my-active-key"
    assert result["active-key-hash-123"]["team_id"] == "team-abc"


@pytest.mark.asyncio
async def test_get_api_key_metadata_falls_back_to_deleted_keys():
    """Test that get_api_key_metadata should fall back to deleted keys table for missing keys."""
    mock_prisma = MagicMock()

    # No active keys found
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    # Deleted key record exists
    mock_deleted_key = MagicMock()
    mock_deleted_key.token = "deleted-key-hash-456"
    mock_deleted_key.key_alias = "toto-test-2"
    mock_deleted_key.team_id = "team-xyz"

    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[mock_deleted_key])

    result = await get_api_key_metadata(
        prisma_client=mock_prisma,
        api_keys={"deleted-key-hash-456"},
    )

    assert "deleted-key-hash-456" in result
    assert result["deleted-key-hash-456"]["key_alias"] == "toto-test-2"
    assert result["deleted-key-hash-456"]["team_id"] == "team-xyz"

    # Verify deleted table was queried with the missing key
    mock_prisma.db.litellm_deletedverificationtoken.find_many.assert_called_once_with(
        where={"token": {"in": ["deleted-key-hash-456"]}},
        order={"deleted_at": "desc"},
    )


@pytest.mark.asyncio
async def test_get_api_key_metadata_mixed_active_and_deleted_keys():
    """Test that get_api_key_metadata should return metadata for both active and deleted keys."""
    mock_prisma = MagicMock()

    # One active key found
    mock_active_key = MagicMock()
    mock_active_key.token = "active-key-hash"
    mock_active_key.key_alias = "active-alias"
    mock_active_key.team_id = "team-active"

    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[mock_active_key])

    # One deleted key found
    mock_deleted_key = MagicMock()
    mock_deleted_key.token = "deleted-key-hash"
    mock_deleted_key.key_alias = "deleted-alias"
    mock_deleted_key.team_id = "team-deleted"

    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[mock_deleted_key])

    result = await get_api_key_metadata(
        prisma_client=mock_prisma,
        api_keys={"active-key-hash", "deleted-key-hash"},
    )

    # Both keys should have metadata
    assert len(result) == 2
    assert result["active-key-hash"]["key_alias"] == "active-alias"
    assert result["active-key-hash"]["team_id"] == "team-active"
    assert result["deleted-key-hash"]["key_alias"] == "deleted-alias"
    assert result["deleted-key-hash"]["team_id"] == "team-deleted"


@pytest.mark.asyncio
async def test_get_api_key_metadata_deleted_table_not_queried_when_all_keys_found():
    """Test that get_api_key_metadata should not query deleted table when all keys are active."""
    mock_prisma = MagicMock()

    mock_active_key = MagicMock()
    mock_active_key.token = "key-hash-1"
    mock_active_key.key_alias = "alias-1"
    mock_active_key.team_id = "team-1"

    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[mock_active_key])
    mock_prisma.db.litellm_deletedverificationtoken = MagicMock()
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])

    result = await get_api_key_metadata(
        prisma_client=mock_prisma,
        api_keys={"key-hash-1"},
    )

    assert len(result) == 1
    assert result["key-hash-1"]["key_alias"] == "alias-1"
    # Deleted table should NOT have been queried
    mock_prisma.db.litellm_deletedverificationtoken.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_get_api_key_metadata_deleted_table_error_handled_gracefully():
    """Test that get_api_key_metadata should handle errors from deleted table gracefully."""
    mock_prisma = MagicMock()

    # No active keys found
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    # Deleted table raises an error (e.g., table doesn't exist in older schema)
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(side_effect=Exception("Table not found"))

    result = await get_api_key_metadata(
        prisma_client=mock_prisma,
        api_keys={"missing-key-hash"},
    )

    # Should return empty dict without raising
    assert result == {}


@pytest.mark.asyncio
async def test_get_api_key_metadata_regenerated_key_uses_most_recent_deleted_record():
    """Test that get_api_key_metadata should use the most recent deleted record for regenerated keys."""
    mock_prisma = MagicMock()

    # No active keys found (old hash no longer in active table after regeneration)
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    # Multiple deleted records for same token (e.g., regenerated multiple times)
    mock_deleted_1 = MagicMock()
    mock_deleted_1.token = "old-key-hash"
    mock_deleted_1.key_alias = "latest-alias"
    mock_deleted_1.team_id = "latest-team"

    mock_deleted_2 = MagicMock()
    mock_deleted_2.token = "old-key-hash"
    mock_deleted_2.key_alias = "older-alias"
    mock_deleted_2.team_id = "older-team"

    # Ordered by deleted_at desc, so first record is the most recent
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[mock_deleted_1, mock_deleted_2])

    result = await get_api_key_metadata(
        prisma_client=mock_prisma,
        api_keys={"old-key-hash"},
    )

    # Should use the first (most recent) record
    assert result["old-key-hash"]["key_alias"] == "latest-alias"
    assert result["old-key-hash"]["team_id"] == "latest-team"


@pytest.mark.asyncio
async def test_tag_daily_activity_metadata_totals_not_zero():
    """Test that tag daily activity returns correct metadata totals.

    Regression test: the tag endpoint previously passed metadata_metrics_func=
    compute_tag_metadata_totals, which skipped every row whose request_id is
    NULL.  Rows in litellm_dailytagspend are pre-aggregated and always have
    NULL request_id, so the totals panel showed $0.  The fix is to pass
    metadata_metrics_func=None so the fallback aggregation path is used instead.
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    # Create mock tag spend records (request_id is NULL for aggregated rows)
    mock_record_1 = MagicMock()
    mock_record_1.request_id = None  # NULL in aggregated daily rows
    mock_record_1.tag = "production"
    mock_record_1.date = "2024-01-01"
    mock_record_1.api_key = "key-1"
    mock_record_1.model = "gpt-4"
    mock_record_1.model_group = "gpt-4"
    mock_record_1.custom_llm_provider = "openai"
    mock_record_1.mcp_namespaced_tool_name = None
    mock_record_1.endpoint = "/chat/completions"
    mock_record_1.spend = 25.0
    mock_record_1.prompt_tokens = 500
    mock_record_1.completion_tokens = 200
    mock_record_1.cache_read_input_tokens = 0
    mock_record_1.cache_creation_input_tokens = 0
    mock_record_1.compression_saved_tokens = 0
    mock_record_1.compression_savings_spend = 0.0
    mock_record_1.prompt_caching_savings_spend = 0.0
    mock_record_1.autorouter_savings_spend = 0.0
    mock_record_1.api_requests = 10
    mock_record_1.successful_requests = 9
    mock_record_1.failed_requests = 1

    mock_record_2 = MagicMock()
    mock_record_2.request_id = None
    mock_record_2.tag = "staging"
    mock_record_2.date = "2024-01-01"
    mock_record_2.api_key = "key-2"
    mock_record_2.model = "gpt-3.5-turbo"
    mock_record_2.model_group = "gpt-3.5-turbo"
    mock_record_2.custom_llm_provider = "openai"
    mock_record_2.mcp_namespaced_tool_name = None
    mock_record_2.endpoint = "/chat/completions"
    mock_record_2.spend = 5.0
    mock_record_2.prompt_tokens = 300
    mock_record_2.completion_tokens = 100
    mock_record_2.cache_read_input_tokens = 0
    mock_record_2.cache_creation_input_tokens = 0
    mock_record_2.compression_saved_tokens = 0
    mock_record_2.compression_savings_spend = 0.0
    mock_record_2.prompt_caching_savings_spend = 0.0
    mock_record_2.autorouter_savings_spend = 0.0
    mock_record_2.api_requests = 5
    mock_record_2.successful_requests = 5
    mock_record_2.failed_requests = 0

    mock_table = MagicMock()
    mock_table.count = AsyncMock(return_value=2)
    mock_table.find_many = AsyncMock(return_value=[mock_record_1, mock_record_2])
    mock_prisma.db.litellm_dailytagspend = mock_table
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    result = await get_daily_activity(
        prisma_client=mock_prisma,
        table_name="litellm_dailytagspend",
        entity_id_field="tag",
        entity_id=None,
        entity_metadata_field=None,
        start_date="2024-01-01",
        end_date="2024-01-01",
        model=None,
        api_key=None,
        page=1,
        page_size=1000,
        metadata_metrics_func=None,  # No custom func — matches the fix
    )

    # Metadata totals must reflect actual spend, NOT be zero
    assert result.metadata.total_spend == 30.0  # 25.0 + 5.0
    assert result.metadata.total_api_requests == 15  # 10 + 5
    assert result.metadata.total_successful_requests == 14  # 9 + 5
    assert result.metadata.total_failed_requests == 1
    assert result.metadata.total_tokens == 1100  # (500+200) + (300+100)

    # Verify breakdown still works
    assert len(result.results) == 1
    daily = result.results[0]
    assert "production" in daily.breakdown.entities
    assert "staging" in daily.breakdown.entities
    assert daily.breakdown.entities["production"].metrics.spend == 25.0
    assert daily.breakdown.entities["staging"].metrics.spend == 5.0


@pytest.mark.asyncio
async def test_aggregated_activity_preserves_metadata_for_deleted_keys():
    """Test that the full aggregation pipeline should preserve metadata for deleted keys."""
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    # GROUPING SETS rollup rows. The api_key metadata lookup is driven
    # by any non-NULL api_key in the result set, so the (date, endpoint,
    # api_key) row at level 30 is what ensures get_api_key_metadata is
    # called for "deleted-key-hash".
    base = {
        "model": None,
        "model_group": None,
        "custom_llm_provider": None,
        "mcp_namespaced_tool_name": None,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 0,
        "compression_savings_spend": 0.0,
        "prompt_caching_savings_spend": 0.0,
        "autorouter_savings_spend": 0.0,
        "failed_requests": 0,
    }
    mock_rows = [
        {
            **base,
            "date": "2024-01-01",
            "endpoint": "/v1/chat/completions",
            "api_key": None,
            "group_level": 62,
            "spend": 10.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "api_requests": 1,
            "successful_requests": 1,
        },
        {
            **base,
            "date": "2024-01-01",
            "endpoint": "/v1/chat/completions",
            "api_key": "deleted-key-hash",
            "group_level": 30,
            "spend": 10.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "api_requests": 1,
            "successful_requests": 1,
        },
    ]

    mock_prisma.db.query_raw = AsyncMock(return_value=mock_rows)

    # Active table returns nothing for this key
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    # Deleted table returns the metadata
    mock_deleted_key = MagicMock()
    mock_deleted_key.token = "deleted-key-hash"
    mock_deleted_key.key_alias = "toto-test-2"
    mock_deleted_key.team_id = "69cd4b77-b095-4489-8c46-4f2f31d840a2"

    mock_prisma.db.litellm_deletedverificationtoken = MagicMock()
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[mock_deleted_key])

    result = await get_daily_activity_aggregated(
        prisma_client=mock_prisma,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None,
        entity_metadata_field=None,
        start_date="2024-01-01",
        end_date="2024-01-01",
        model=None,
        api_key=None,
    )

    # Verify the deleted key's metadata is preserved
    daily_data = result.results[0]
    chat_endpoint = daily_data.breakdown.endpoints["/v1/chat/completions"]
    assert "deleted-key-hash" in chat_endpoint.api_key_breakdown
    key_data = chat_endpoint.api_key_breakdown["deleted-key-hash"]
    assert key_data.metadata.key_alias == "toto-test-2"
    assert key_data.metadata.team_id == "69cd4b77-b095-4489-8c46-4f2f31d840a2"
    assert key_data.metrics.spend == 10.0


def _daily_user_spend_record(*, user_id, api_key, spend, model="gpt-4", model_group="gpt-4"):
    """A LiteLLM_DailyUserSpend row as the per-user breakdown reads it."""
    return SimpleNamespace(
        date="2024-01-01",
        user_id=user_id,
        api_key=api_key,
        model=model,
        model_group=model_group,
        custom_llm_provider="openai",
        mcp_namespaced_tool_name=None,
        endpoint="/chat/completions",
        spend=spend,
        prompt_tokens=10,
        completion_tokens=5,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        compression_saved_tokens=0,
        compression_savings_spend=0.0,
        prompt_caching_savings_spend=0.0,
        autorouter_savings_spend=0.0,
        api_requests=1,
        successful_requests=1,
        failed_requests=0,
    )


@pytest.mark.asyncio
async def test_get_daily_activity_applies_resolve_entity_metadata_to_breakdown():
    """Regression for LIT-3889: the Spend Per User chart showed raw UUIDs.

    /user/daily/activity used to pass entity_metadata_field=None, so every
    user entity in the breakdown carried empty metadata and the dashboard had
    nothing to render but the user_id UUID. The page-scoped resolver must put
    the resolved email/alias onto the entity metadata so the UI can label it,
    while a spender with no email on file still falls back to the raw UUID.
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    records = [
        _daily_user_spend_record(user_id="user-with-email", api_key="key-1", spend=7.0),
        _daily_user_spend_record(user_id="user-no-email", api_key="key-2", spend=3.0),
    ]

    mock_table = MagicMock()
    mock_table.count = AsyncMock(return_value=len(records))
    mock_table.find_many = AsyncMock(return_value=records)
    mock_prisma.db.litellm_dailyuserspend = mock_table
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    seen_user_ids = {}

    async def resolver(page_records):
        seen_user_ids["ids"] = {r.user_id for r in page_records}
        return {"user-with-email": {"user_email": "spender@example.com"}}

    result = await get_daily_activity(
        prisma_client=mock_prisma,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None,
        entity_metadata_field=None,
        start_date="2024-01-01",
        end_date="2024-01-01",
        model=None,
        api_key=None,
        page=1,
        page_size=1000,
        resolve_entity_metadata=resolver,
    )

    # Resolver is driven by the user_ids actually on the page
    assert seen_user_ids["ids"] == {"user-with-email", "user-no-email"}

    entities = result.results[0].breakdown.entities
    # Email is on the entity metadata so the UI labels the chart with it
    assert entities["user-with-email"].metadata["user_email"] == "spender@example.com"
    # No email on file -> empty metadata -> UI falls back to the UUID
    assert entities["user-no-email"].metadata == {}


@pytest.mark.asyncio
async def test_model_groups_breakdown_keys_by_public_name_with_model_fallback():
    """The usage UI labels model traffic with the model_groups breakdown.

    Keys must be the requested public model name (model_group), and rows with a
    NULL or empty model_group (pre-routing failures, rows written before the
    column existed) must fall back to their model name instead of being dropped
    from the breakdown. The models breakdown keeps the upstream litellm names.
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    records = [
        _daily_user_spend_record(user_id="u1", api_key="key-1", spend=7.0, model="gpt-5.2", model_group="gpt-5.2-eu"),
        _daily_user_spend_record(user_id="u1", api_key="key-1", spend=3.0, model="gpt-5.2", model_group=None),
        _daily_user_spend_record(user_id="u1", api_key="key-1", spend=2.0, model="claude-x", model_group=""),
    ]

    mock_table = MagicMock()
    mock_table.count = AsyncMock(return_value=len(records))
    mock_table.find_many = AsyncMock(return_value=records)
    mock_prisma.db.litellm_dailyuserspend = mock_table
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    result = await get_daily_activity(
        prisma_client=mock_prisma,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None,
        entity_metadata_field=None,
        start_date="2024-01-01",
        end_date="2024-01-01",
        model=None,
        api_key=None,
        page=1,
        page_size=1000,
    )

    breakdown = result.results[0].breakdown

    assert set(breakdown.model_groups.keys()) == {"gpt-5.2-eu", "gpt-5.2", "claude-x"}
    assert breakdown.model_groups["gpt-5.2-eu"].metrics.spend == 7.0
    assert breakdown.model_groups["gpt-5.2"].metrics.spend == 3.0
    assert breakdown.model_groups["claude-x"].metrics.spend == 2.0
    assert breakdown.model_groups["gpt-5.2"].api_key_breakdown["key-1"].metrics.spend == 3.0

    assert set(breakdown.models.keys()) == {"gpt-5.2", "claude-x"}
    assert breakdown.models["gpt-5.2"].metrics.spend == 10.0
    assert breakdown.models["claude-x"].metrics.spend == 2.0


class TestAdjustDatesForTimezone:
    """
    Regression tests for the timezone double-counting bug.

    Background: the previous implementation expanded the SQL date range by a full
    UTC day on whichever side a non-UTC timezone offset pointed. Because spend is
    bucketed in whole UTC days in the aggregation table, that expansion caused
    single-day queries from non-UTC timezones to include a second full UTC day's
    worth of data, producing approximately 2x over-counting. The sum of single-day
    spends across a window then exceeded the equivalent multi-day aggregate, which
    is mathematically impossible.

    These tests pin the function to a pass-through and assert the additivity
    invariant that any future implementation must preserve.
    """

    @pytest.mark.parametrize(
        "offset_minutes",
        [
            None,
            0,
            -330,  # IST UTC+5:30
            -540,  # JST UTC+9
            -60,  # CET UTC+1
            240,  # AST UTC-4
            300,  # EST UTC-5
            480,  # PST UTC-8
        ],
    )
    def test_returns_input_dates_unchanged_for_any_offset(self, offset_minutes):
        start, end = _adjust_dates_for_timezone("2026-05-29", "2026-05-29", offset_minutes)
        assert start == "2026-05-29"
        assert end == "2026-05-29"

    def test_single_day_query_does_not_widen_to_two_utc_days(self):
        """
        Pins the boundary that caused the original 2x bug: a single IST day must
        not be translated into a SQL filter covering two UTC days.
        """
        start, end = _adjust_dates_for_timezone("2026-05-29", "2026-05-29", -330)
        assert start == end == "2026-05-29", (
            "Single-day IST query expanded to a multi-day UTC range; this is "
            "the regression that produced approximately 2x over-counting."
        )

    def test_multi_day_range_endpoints_are_preserved(self):
        start, end = _adjust_dates_for_timezone("2026-05-29", "2026-06-02", -330)
        assert (start, end) == ("2026-05-29", "2026-06-02")

    @pytest.mark.parametrize("offset_minutes", [-330, 480])
    def test_single_day_sums_match_multi_day_window(self, offset_minutes):
        """
        Additivity invariant: querying each day in a window separately and summing
        the resulting SQL ranges must cover exactly the same range as querying the
        whole window at once. The bug broke this; without it, single-day sums
        exceeded the multi-day total by ~50% over a 5-day IST window.
        """
        days = ["2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01", "2026-06-02"]
        single_day_ranges = [_adjust_dates_for_timezone(d, d, offset_minutes) for d in days]
        multi_day_range = _adjust_dates_for_timezone(days[0], days[-1], offset_minutes)

        per_day_starts = [r[0] for r in single_day_ranges]
        per_day_ends = [r[1] for r in single_day_ranges]
        assert min(per_day_starts) == multi_day_range[0]
        assert max(per_day_ends) == multi_day_range[1]
        assert per_day_starts == days
        assert per_day_ends == days


class TestAdjustDatesForTimezoneLiveEnd:
    """
    Regression tests for the stale-evening bug: a caller west of UTC whose range
    ends on their local "today" was capped at that local date's UTC bucket, so
    once UTC rolled past their local midnight (5pm PT), everything sent that
    evening sat in the next UTC bucket and the dashboard reported $0 for it
    until local midnight. A range that reaches the caller's current day and
    opts in via include_current_utc_day must extend to today's UTC bucket; the
    only part of that bucket outside the range is the future, which is empty,
    so the extension cannot over-count. Callers that do not opt in keep the
    pass-through byte for byte.
    """

    PT_EVENING_UTC: Final = datetime(2026, 8, 6, 4, 30, tzinfo=timezone.utc)

    def test_pt_evening_range_ending_today_extends_to_utc_today(self):
        start, end = _adjust_dates_for_timezone(
            "2026-07-06", "2026-08-05", 420, include_current_utc_day=True, utc_now=self.PT_EVENING_UTC
        )
        assert (start, end) == ("2026-07-06", "2026-08-06")

    def test_without_opt_in_live_range_keeps_pass_through(self):
        start, end = _adjust_dates_for_timezone("2026-07-06", "2026-08-05", 420, utc_now=self.PT_EVENING_UTC)
        assert (start, end) == ("2026-07-06", "2026-08-05")

    def test_pt_historical_range_is_untouched(self):
        start, end = _adjust_dates_for_timezone(
            "2026-07-01", "2026-08-04", 420, include_current_utc_day=True, utc_now=self.PT_EVENING_UTC
        )
        assert (start, end) == ("2026-07-01", "2026-08-04")

    def test_east_of_utc_local_today_already_covers_utc_today(self):
        ist_evening_utc: Final = datetime(2026, 8, 5, 17, 0, tzinfo=timezone.utc)
        start, end = _adjust_dates_for_timezone(
            "2026-07-07", "2026-08-06", -330, include_current_utc_day=True, utc_now=ist_evening_utc
        )
        assert (start, end) == ("2026-07-07", "2026-08-06")

    def test_missing_offset_stays_pass_through_even_for_live_range(self):
        start, end = _adjust_dates_for_timezone(
            "2026-07-06", "2026-08-05", None, include_current_utc_day=True, utc_now=self.PT_EVENING_UTC
        )
        assert (start, end) == ("2026-07-06", "2026-08-05")

    def test_utc_caller_range_ending_today_is_unchanged(self):
        utc_noon: Final = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
        start, end = _adjust_dates_for_timezone(
            "2026-07-06", "2026-08-05", 0, include_current_utc_day=True, utc_now=utc_noon
        )
        assert (start, end) == ("2026-07-06", "2026-08-05")

    def test_future_end_date_extends_no_further_than_requested(self):
        start, end = _adjust_dates_for_timezone(
            "2026-07-06", "2026-08-09", 420, include_current_utc_day=True, utc_now=self.PT_EVENING_UTC
        )
        assert (start, end) == ("2026-07-06", "2026-08-09")


class TestBuildAggregatedSqlQuery:
    """
    Asserts the SQL emitted by the aggregated query path stays anchored to the
    user-supplied date range. The original bug shipped a function that returned
    expanded dates from _adjust_dates_for_timezone, so the regression surface is
    not just the helper but the SQL it feeds into.
    """

    @pytest.mark.parametrize("offset_minutes", [None, 0, -330, 480])
    def test_sql_date_bounds_are_user_supplied_dates(self, offset_minutes):
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id="user-1",
            start_date="2026-05-29",
            end_date="2026-05-29",
            model=None,
            api_key=None,
            timezone_offset_minutes=offset_minutes,
        )

        assert params[0] == "2026-05-29"
        assert params[1] == "2026-05-29"
        assert "date >= $1" in sql
        assert "date <= $2" in sql

    @pytest.mark.parametrize("build", [_build_aggregated_sql_query, _build_entity_rollup_sql_query])
    def test_include_current_utc_day_extends_live_end_bound(self, build):
        """
        An offset larger than 24h keeps the caller's local date behind UTC at any
        wall-clock hour, so the live-end extension is deterministic: a range ending
        on the caller's local today must reach today's UTC bucket (LIT-5818, guards
        the #36051 behavior on the aggregated path).
        """
        offset_minutes: Final = 1500
        caller_local_today: Final = (datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)).date().isoformat()
        utc_today: Final = datetime.now(timezone.utc).date().isoformat()

        _sql, params = build(
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id="user-1",
            start_date="2026-05-01",
            end_date=caller_local_today,
            model=None,
            api_key=None,
            timezone_offset_minutes=offset_minutes,
            include_current_utc_day=True,
        )

        assert params[0] == "2026-05-01"
        assert params[1] == utc_today

    def test_optional_filters_appear_in_params_in_order(self):
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id="user-1",
            start_date="2026-05-29",
            end_date="2026-06-02",
            model="bedrock/global.anthropic.claude-opus-4-8",
            api_key="sk-test",
            timezone_offset_minutes=-330,
        )

        assert params == [
            "2026-05-29",
            "2026-06-02",
            "user-1",
            "bedrock/global.anthropic.claude-opus-4-8",
            "sk-test",
        ]
        assert "model = $4" in sql
        assert "api_key = $5" in sql

    def test_model_group_rollups_fall_back_to_model_name(self):
        """Aggregated model_groups rollups must fall back to model for group-less rows.

        The (date, model_group) grouping level cannot recover the model column
        after the fact (it is rolled up), so the fallback has to happen in SQL;
        without it, group-less rows silently vanish from the model_groups
        breakdown that the usage UI now renders by default. Group-less rows are
        stored as empty strings, not NULL (spend_tracking_utils defaults
        model_group to ""), so a plain COALESCE is not enough: the fallback must
        be NULLIF-wrapped to catch both
        """
        sql, _ = _build_aggregated_sql_query(
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id=None,
            start_date="2026-07-01",
            end_date="2026-07-01",
            model=None,
            api_key=None,
        )

        normalized = " ".join(sql.split())
        fallback = "COALESCE(NULLIF(model_group, ''), model)"
        assert f"{fallback} AS model_group" in normalized
        assert (
            f"GROUPING(date, api_key, model, {fallback}, "
            "custom_llm_provider, mcp_namespaced_tool_name, endpoint) AS group_level" in normalized
        )
        assert f"(date, {fallback}), (date, {fallback}, api_key)," in normalized
        assert "(date, model_group)" not in normalized
        assert "COALESCE(model_group, model)" not in normalized


class TestAggregatedEmptyEntityFilter:
    _BUILDERS: Final = (_build_aggregated_sql_query, _build_entity_rollup_sql_query)

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_empty_entity_list_emits_no_degenerate_in_clause(self, build):
        sql, params = build(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=[],
            start_date="2026-08-01",
            end_date="2026-08-19",
            model=None,
            api_key=None,
        )

        normalized = " ".join(sql.split())
        assert "IN ()" not in normalized
        assert '"team_id" IN' not in normalized
        assert params == ["2026-08-01", "2026-08-19"]

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_empty_entity_list_matches_nothing_rather_than_everything(self, build):
        sql, _ = build(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=[],
            start_date="2026-08-01",
            end_date="2026-08-19",
            model=None,
            api_key=None,
        )

        assert "FALSE" in " ".join(sql.split())

    @pytest.mark.parametrize("build", _BUILDERS)
    def test_populated_entity_list_still_filters_on_its_ids(self, build):
        sql, params = build(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=["team-alpha", "team-beta"],
            start_date="2026-08-01",
            end_date="2026-08-19",
            model=None,
            api_key=None,
        )

        normalized = " ".join(sql.split())
        assert '"team_id" IN ($3, $4)' in normalized
        assert "FALSE" not in normalized
        assert params == ["2026-08-01", "2026-08-19", "team-alpha", "team-beta"]


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_empty_result_set():
    """Regression test for the empty-range 500.

    When the date filter matches zero rows, Postgres still emits the
    grand-total () grouping-set row with every SUM column NULL. The
    endpoint must return an empty result set with zeroed totals, not
    crash on None + None.
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    mock_rows = [
        {
            "date": None,
            "api_key": None,
            "model": None,
            "model_group": None,
            "custom_llm_provider": None,
            "mcp_namespaced_tool_name": None,
            "endpoint": None,
            "group_level": 127,
            "spend": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "compression_saved_tokens": None,
            "compression_savings_spend": None,
            "prompt_caching_savings_spend": None,
            "autorouter_savings_spend": None,
            "api_requests": None,
            "successful_requests": None,
            "failed_requests": None,
        }
    ]
    mock_prisma.db.query_raw = AsyncMock(return_value=mock_rows)

    result = await get_daily_activity_aggregated(
        prisma_client=mock_prisma,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None,
        entity_metadata_field=None,
        start_date="2026-06-16",
        end_date="2026-06-16",
        model=None,
        api_key=None,
    )

    assert result.results == []
    assert result.metadata.total_spend == 0.0
    assert result.metadata.total_prompt_tokens == 0
    assert result.metadata.total_completion_tokens == 0
    assert result.metadata.total_tokens == 0
    assert result.metadata.total_api_requests == 0
    assert result.metadata.total_successful_requests == 0
    assert result.metadata.total_failed_requests == 0
    assert result.metadata.total_cache_read_input_tokens == 0
    assert result.metadata.total_cache_creation_input_tokens == 0
    assert result.metadata.total_compression_saved_tokens == 0


def _no_spend_record():
    """A rollup row for a key with no spend, where SUM() returns NULL (None)."""
    return SimpleNamespace(
        spend=None,
        prompt_tokens=None,
        completion_tokens=None,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        compression_saved_tokens=None,
        compression_savings_spend=None,
        prompt_caching_savings_spend=None,
        autorouter_savings_spend=None,
        api_requests=None,
        successful_requests=None,
        failed_requests=None,
    )


def test_record_to_spend_metrics_handles_none_values():
    """Keys with no spend produce NULL aggregates; treat them as zero, not a crash."""
    metrics = _record_to_spend_metrics(_no_spend_record())
    assert metrics.spend == 0
    assert metrics.prompt_tokens == 0
    assert metrics.completion_tokens == 0
    assert metrics.total_tokens == 0
    assert metrics.api_requests == 0
    assert metrics.successful_requests == 0
    assert metrics.failed_requests == 0
    assert metrics.cache_read_input_tokens == 0
    assert metrics.cache_creation_input_tokens == 0
    assert metrics.compression_saved_tokens == 0


def test_update_metrics_handles_none_values():
    """update_metrics should coalesce NULL aggregates instead of raising TypeError."""
    metrics = update_metrics(SpendMetrics(), _no_spend_record())
    assert metrics.spend == 0
    assert metrics.prompt_tokens == 0
    assert metrics.completion_tokens == 0
    assert metrics.total_tokens == 0
    assert metrics.api_requests == 0
    assert metrics.successful_requests == 0
    assert metrics.failed_requests == 0
    assert metrics.cache_read_input_tokens == 0
    assert metrics.cache_creation_input_tokens == 0
    assert metrics.compression_saved_tokens == 0


class TestEverySavingsDriverSurvivesTheReadPath:
    """A savings driver is only real if it survives the whole read path.

    The write path can price a driver correctly and persist it to all six rollup
    tables, and the dashboard can still render a permanent $0.00 because the
    aggregation query never summed the column or the response model never
    declared it. That failure is silent: the card renders, the number is just
    always zero, which is indistinguishable from having saved nothing. These
    tests enumerate the drivers from the response model itself, so a driver added
    later cannot be half-wired.
    """

    def _drivers(self) -> list[str]:
        drivers = [field for field in SpendMetrics.model_fields if field.endswith("_savings_spend")]
        assert drivers, "expected the dashboard response to expose at least one savings driver"
        return drivers

    def test_every_driver_is_summed_by_the_rollup_query(self):
        sql, _ = _build_aggregated_sql_query(
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id="user-1",
            start_date="2026-07-01",
            end_date="2026-07-31",
            model=None,
            api_key=None,
            timezone_offset_minutes=None,
        )
        for driver in self._drivers():
            assert f"SUM({driver})" in sql, f"{driver} is never summed, so it reads as zero"

    def test_every_driver_is_accumulated_across_rows(self):
        for driver in self._drivers():
            record = _no_spend_record()
            setattr(record, driver, 1.25)
            metrics = update_metrics(SpendMetrics(), record)
            assert getattr(metrics, driver) == pytest.approx(1.25), f"{driver} is dropped when accumulating rows"

    def test_every_driver_is_carried_by_a_single_row_conversion(self):
        for driver in self._drivers():
            record = _no_spend_record()
            setattr(record, driver, 2.5)
            assert getattr(_record_to_spend_metrics(record), driver) == pytest.approx(2.5)

    def test_every_driver_has_a_range_total(self):
        for driver in self._drivers():
            assert f"total_{driver}" in DailySpendMetadata.model_fields, (
                f"total_{driver} is missing, so the range summary omits the driver"
            )


@pytest.fixture
def ptu_cost_attribution_enabled(monkeypatch):
    monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")


def _spend_record(api_key, *, model="gpt-4o-mini-ptu", spend=0.0, ptu_flat_cost=0.0):
    return SimpleNamespace(
        api_key=api_key,
        model=model,
        model_group=None,
        mcp_namespaced_tool_name=None,
        custom_llm_provider="openai",
        endpoint=None,
        spend=spend,
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        compression_saved_tokens=0,
        compression_savings_spend=0,
        prompt_caching_savings_spend=0,
        autorouter_savings_spend=0,
        total_tokens=0,
        api_requests=0,
        successful_requests=0,
        failed_requests=0,
        ptu_flat_cost=ptu_flat_cost,
    )


def test_update_metrics_accumulates_ptu_flat_cost(ptu_cost_attribution_enabled):
    metrics = update_metrics(SpendMetrics(), _spend_record("real-key", spend=1.0, ptu_flat_cost=240.0))
    assert metrics.flat_cost == 240.0
    assert metrics.spend == 1.0


def test_ptu_sentinel_excluded_from_key_breakdown_but_flat_cost_aggregates(ptu_cost_attribution_enabled):
    from litellm.constants import PTU_SENTINEL_API_KEY
    from litellm.proxy.management_endpoints.common_daily_activity import update_breakdown_metrics
    from litellm.types.proxy.management_endpoints.common_daily_activity import BreakdownMetrics

    breakdown = BreakdownMetrics()
    update_breakdown_metrics(breakdown, _spend_record("real-key", spend=5.0, ptu_flat_cost=0.0), {}, {}, {})
    update_breakdown_metrics(breakdown, _spend_record(PTU_SENTINEL_API_KEY, spend=0.0, ptu_flat_cost=240.0), {}, {}, {})

    model_bucket = breakdown.models["gpt-4o-mini-ptu"]
    # flat cost aggregates into the parent model metrics
    assert model_bucket.metrics.flat_cost == 240.0
    assert model_bucket.metrics.spend == 5.0
    # the sentinel never appears as an api_key row; only the real key does
    assert PTU_SENTINEL_API_KEY not in model_bucket.api_key_breakdown
    assert "real-key" in model_bucket.api_key_breakdown


def _grouping_row(
    group_level,
    *,
    api_key=None,
    model=None,
    model_group=None,
    custom_llm_provider="openai",
    mcp_namespaced_tool_name=None,
    endpoint=None,
    spend=0.0,
    ptu_flat_cost=0.0,
):
    from litellm.proxy.management_endpoints.common_daily_activity import _GroupingSetsRow

    return _GroupingSetsRow(
        date="2024-01-01",
        api_key=api_key,
        model=model,
        model_group=model_group,
        custom_llm_provider=custom_llm_provider,
        mcp_namespaced_tool_name=mcp_namespaced_tool_name,
        endpoint=endpoint,
        group_level=group_level,
        spend=spend,
        ptu_flat_cost=ptu_flat_cost,
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        compression_saved_tokens=0,
        compression_savings_spend=0.0,
        prompt_caching_savings_spend=0.0,
        autorouter_savings_spend=0.0,
        api_requests=0,
        successful_requests=0,
        failed_requests=0,
    )


def test_grouping_sets_dispatcher_excludes_ptu_sentinel_from_key_breakdowns(ptu_cost_attribution_enabled):
    """The GROUPING SETS path must mirror the per-row path: the flat-cost sentinel
    aggregates into the date/model/total metrics but never surfaces as an api_key."""
    from litellm.constants import PTU_SENTINEL_API_KEY
    from litellm.proxy.management_endpoints.common_daily_activity import (
        _GROUP_DATE_API_KEY,
        _GROUP_DATE_MODEL,
        _GROUP_DATE_MODEL_API_KEY,
        _GROUP_GRAND_TOTAL,
        _aggregate_grouping_sets_records_sync,
    )

    records = [
        _grouping_row(_GROUP_DATE_API_KEY, api_key="real-key", spend=5.0),
        _grouping_row(_GROUP_DATE_API_KEY, api_key=PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0),
        _grouping_row(_GROUP_DATE_MODEL, model="gpt-4o-mini-ptu", spend=5.0, ptu_flat_cost=240.0),
        _grouping_row(_GROUP_DATE_MODEL_API_KEY, model="gpt-4o-mini-ptu", api_key="real-key", spend=5.0),
        _grouping_row(
            _GROUP_DATE_MODEL_API_KEY, model="gpt-4o-mini-ptu", api_key=PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0
        ),
        _grouping_row(_GROUP_GRAND_TOTAL, spend=5.0, ptu_flat_cost=240.0),
    ]

    aggregated = _aggregate_grouping_sets_records_sync(records=records, api_key_metadata={})

    assert aggregated["totals"].flat_cost == 240.0
    day = aggregated["results"][0]
    assert PTU_SENTINEL_API_KEY not in day.breakdown.api_keys
    assert "real-key" in day.breakdown.api_keys

    model_bucket = day.breakdown.models["gpt-4o-mini-ptu"]
    assert model_bucket.metrics.flat_cost == 240.0
    assert model_bucket.metrics.spend == 5.0
    assert PTU_SENTINEL_API_KEY not in model_bucket.api_key_breakdown
    assert "real-key" in model_bucket.api_key_breakdown


def test_grouping_sets_dispatcher_populates_every_breakdown_level(ptu_cost_attribution_enabled):
    """Every GROUPING SETS level lands in its bucket, and the flat-cost sentinel
    is kept out of the model_group and provider api_key sub-breakdowns too."""
    from litellm.constants import PTU_SENTINEL_API_KEY
    from litellm.proxy.management_endpoints.common_daily_activity import (
        _GROUP_DATE_ENDPOINT,
        _GROUP_DATE_ENDPOINT_API_KEY,
        _GROUP_DATE_MCP,
        _GROUP_DATE_MCP_API_KEY,
        _GROUP_DATE_MODEL_GROUP,
        _GROUP_DATE_MODEL_GROUP_API_KEY,
        _GROUP_DATE_PROVIDER,
        _GROUP_DATE_PROVIDER_API_KEY,
        _aggregate_grouping_sets_records_sync,
    )

    records = [
        _grouping_row(_GROUP_DATE_MODEL_GROUP, model_group="grp", spend=4.0, ptu_flat_cost=240.0),
        _grouping_row(_GROUP_DATE_MODEL_GROUP_API_KEY, model_group="grp", api_key="real-key", spend=4.0),
        _grouping_row(
            _GROUP_DATE_MODEL_GROUP_API_KEY, model_group="grp", api_key=PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0
        ),
        _grouping_row(_GROUP_DATE_PROVIDER, custom_llm_provider="azure", spend=4.0),
        _grouping_row(_GROUP_DATE_PROVIDER_API_KEY, custom_llm_provider="azure", api_key="real-key", spend=4.0),
        _grouping_row(
            _GROUP_DATE_PROVIDER_API_KEY,
            custom_llm_provider="azure",
            api_key=PTU_SENTINEL_API_KEY,
            ptu_flat_cost=240.0,
        ),
        _grouping_row(_GROUP_DATE_MCP, mcp_namespaced_tool_name="srv/tool", spend=2.0),
        _grouping_row(_GROUP_DATE_MCP_API_KEY, mcp_namespaced_tool_name="srv/tool", api_key="real-key", spend=2.0),
        _grouping_row(_GROUP_DATE_ENDPOINT, endpoint="/v1/chat/completions", spend=3.0),
        _grouping_row(_GROUP_DATE_ENDPOINT_API_KEY, endpoint="/v1/chat/completions", api_key="real-key", spend=3.0),
    ]

    aggregated = _aggregate_grouping_sets_records_sync(records=records, api_key_metadata={})
    day = aggregated["results"][0]

    group_bucket = day.breakdown.model_groups["grp"]
    assert group_bucket.metrics.flat_cost == 240.0
    assert PTU_SENTINEL_API_KEY not in group_bucket.api_key_breakdown
    assert "real-key" in group_bucket.api_key_breakdown

    provider_bucket = day.breakdown.providers["azure"]
    assert PTU_SENTINEL_API_KEY not in provider_bucket.api_key_breakdown
    assert "real-key" in provider_bucket.api_key_breakdown

    assert "real-key" in day.breakdown.mcp_servers["srv/tool"].api_key_breakdown
    assert "real-key" in day.breakdown.endpoints["/v1/chat/completions"].api_key_breakdown


def test_grouping_sets_dispatcher_keeps_ptu_flat_cost_out_of_the_provider_breakdown():
    """Sentinel rows carry no provider, so their flat cost must not surface under the
    "unknown" provider - the per-row path skips them for exactly the same reason."""
    from litellm.proxy.management_endpoints.common_daily_activity import (
        _GROUP_DATE_PROVIDER,
        _aggregate_grouping_sets_records_sync,
    )

    records = [
        _grouping_row(_GROUP_DATE_PROVIDER, custom_llm_provider="azure", spend=4.0),
        # the sentinel's own provider-level row: empty provider, flat cost only
        _grouping_row(_GROUP_DATE_PROVIDER, custom_llm_provider="", ptu_flat_cost=240.0),
    ]

    aggregated = _aggregate_grouping_sets_records_sync(records=records, api_key_metadata={})
    providers = aggregated["results"][0].breakdown.providers

    # the bucket is still reported (a legacy all-zero row must not vanish); only the
    # flat cost is withheld, so no provider is credited with PTU capacity cost
    assert providers["azure"].metrics.spend == 4.0
    assert sum(bucket.metrics.flat_cost for bucket in providers.values()) == 0.0


def test_grouping_sets_dispatcher_keeps_a_real_provider_row_that_shares_the_sentinel_shape():
    """A request row whose provider is empty still gets its "unknown" bucket - only the
    flat cost is withheld, so provider attribution of real spend is unchanged."""
    from litellm.proxy.management_endpoints.common_daily_activity import (
        _GROUP_DATE_PROVIDER,
        _aggregate_grouping_sets_records_sync,
    )

    records = [_grouping_row(_GROUP_DATE_PROVIDER, custom_llm_provider="", spend=4.0, ptu_flat_cost=240.0)]

    aggregated = _aggregate_grouping_sets_records_sync(records=records, api_key_metadata={})
    unknown = aggregated["results"][0].breakdown.providers["unknown"]

    assert unknown.metrics.spend == 4.0
    assert unknown.metrics.flat_cost == 0.0


def test_update_breakdown_metrics_covers_mcp_endpoint_and_entity(ptu_cost_attribution_enabled):
    """A full request record fans out into the mcp, endpoint, provider and entity
    breakdowns, while the flat-cost sentinel stays out of the entity api_key sub-map."""
    from litellm.constants import PTU_SENTINEL_API_KEY
    from litellm.proxy.management_endpoints.common_daily_activity import update_breakdown_metrics
    from litellm.types.proxy.management_endpoints.common_daily_activity import BreakdownMetrics

    breakdown = BreakdownMetrics()
    record = SimpleNamespace(
        api_key="real-key",
        model="gpt-4o-mini-ptu",
        model_group="grp",
        mcp_namespaced_tool_name="srv/tool",
        custom_llm_provider="azure",
        endpoint="/v1/chat/completions",
        spend=5.0,
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        compression_saved_tokens=0,
        compression_savings_spend=0,
        prompt_caching_savings_spend=0,
        autorouter_savings_spend=0,
        total_tokens=0,
        api_requests=0,
        successful_requests=0,
        failed_requests=0,
        ptu_flat_cost=0.0,
        team_id="team-1",
    )
    update_breakdown_metrics(breakdown, record, {}, {}, {}, entity_id_field="team_id")

    assert "srv/tool" in breakdown.mcp_servers
    assert "real-key" in breakdown.mcp_servers["srv/tool"].api_key_breakdown
    assert "/v1/chat/completions" in breakdown.endpoints
    assert "azure" in breakdown.providers
    assert "team-1" in breakdown.entities
    assert "real-key" in breakdown.entities["team-1"].api_key_breakdown

    sentinel = SimpleNamespace(**{**record.__dict__, "api_key": PTU_SENTINEL_API_KEY, "ptu_flat_cost": 240.0})
    update_breakdown_metrics(breakdown, sentinel, {}, {}, {}, entity_id_field="team_id")
    assert PTU_SENTINEL_API_KEY not in breakdown.entities["team-1"].api_key_breakdown
    assert breakdown.entities["team-1"].metrics.flat_cost == 240.0


def test_grouping_sets_dispatcher_keeps_an_all_zero_legacy_provider_bucket():
    """LiteLLM_DailyTeamSpend predates its api_requests column; the migration that added it
    backfilled NOT NULL DEFAULT 0, so a legacy keyless row is all zeroes. Dropping those
    would silently remove a provider the base build reported."""
    from litellm.proxy.management_endpoints.common_daily_activity import (
        _GROUP_DATE_PROVIDER,
        _aggregate_grouping_sets_records_sync,
    )

    records = [
        _grouping_row(_GROUP_DATE_PROVIDER, custom_llm_provider="ollama"),  # spend/tokens/requests all 0
        _grouping_row(_GROUP_DATE_PROVIDER, custom_llm_provider="openai", spend=0.25),
    ]

    providers = _aggregate_grouping_sets_records_sync(records=records, api_key_metadata={})["results"][
        0
    ].breakdown.providers

    assert set(providers) == {"ollama", "openai"}
    assert providers["ollama"].metrics.spend == 0.0
    assert providers["ollama"].metrics.flat_cost == 0.0


class TestSentinelRowsDisplayTheirModelName:
    """A sentinel row keys on the deployment id so a rename cannot move it. The usage views
    render the breakdown key directly as a label, so the read path has to show the name."""

    @pytest.fixture(autouse=True)
    def _enabled(self, ptu_cost_attribution_enabled):
        """Flat cost is gated off by default, and these assert on the amounts."""

    @staticmethod
    def _breakdown(records):
        from litellm.proxy.management_endpoints.common_daily_activity import update_breakdown_metrics
        from litellm.types.proxy.management_endpoints.common_daily_activity import BreakdownMetrics

        breakdown = BreakdownMetrics()
        for record in records:
            update_breakdown_metrics(breakdown, record, {}, {}, {})
        return breakdown

    @staticmethod
    def _sentinel(*, model_id, model_group, flat_cost=480.0):
        from litellm.constants import PTU_SENTINEL_API_KEY

        record = _spend_record(PTU_SENTINEL_API_KEY, model=model_id, spend=0.0, ptu_flat_cost=flat_cost)
        record.model_group = model_group
        return record

    def test_models_breakdown_keys_a_sentinel_row_on_its_public_name(self):
        models = self._breakdown([self._sentinel(model_id="dep-1", model_group="gpt-4o-ptu")]).models

        assert "gpt-4o-ptu" in models, f"the UI would label this row a UUID: {list(models)}"
        assert "dep-1" not in models
        assert models["gpt-4o-ptu"].metrics.flat_cost == pytest.approx(480.0)

    def test_two_deployments_sharing_a_name_merge_under_it(self):
        """The write path stopped collapsing them, so the read path has to."""
        models = self._breakdown(
            [
                self._sentinel(model_id="dep-a", model_group="gpt-4o-ptu", flat_cost=240.0),
                self._sentinel(model_id="dep-b", model_group="gpt-4o-ptu", flat_cost=120.0),
            ]
        ).models

        assert list(models) == ["gpt-4o-ptu"]
        assert models["gpt-4o-ptu"].metrics.flat_cost == pytest.approx(360.0)

    def test_a_request_row_still_keys_on_its_model(self):
        """Scoped to sentinel rows: a request row keys on model as it always has, even
        though it also carries a model_group."""
        record = _spend_record("real-key", model="gemini/gemini-2.5-flash", spend=1.25)
        record.model_group = "gemini-live"

        models = self._breakdown([record]).models

        assert "gemini/gemini-2.5-flash" in models
        assert "gemini-live" not in models

    def test_a_sentinel_row_without_a_model_group_falls_back_to_the_id(self):
        """Never drop the charge: an unexpected row with no display name still reports."""
        models = self._breakdown([self._sentinel(model_id="dep-1", model_group=None)]).models

        assert models["dep-1"].metrics.flat_cost == pytest.approx(480.0)


def _daily_team_row(api_key, *, spend=0.0, ptu_flat_cost=0.0):
    """A LiteLLM_DailyTeamSpend row as the paginated read path receives it from find_many."""
    base: Final = _spend_record(api_key, spend=spend, ptu_flat_cost=ptu_flat_cost)
    return SimpleNamespace(**{**base.__dict__, "date": "2026-07-01", "team_id": "team-1"})


class TestPtuCostAttributionDisabled:
    """With LITELLM_ENABLE_PTU_COST_ATTRIBUTION unset, both read paths report zero flat
    cost, while the sentinel filtering that keeps ``__ptu_flat_cost__`` out of the
    breakdowns keeps running.

    Filtering is deliberately not gated: an operator can enable the flag, accrue
    sentinel rows, then disable it, and those rows stay in LiteLLM_DailyTeamSpend
    forever. Gating the filter too would surface the sentinel as a bogus api_key and
    mint a provider bucket for its empty provider.
    """

    @pytest.fixture(autouse=True)
    def _flag_off(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)

    def test_paginated_path_reports_zero_flat_cost(self):
        metrics = update_metrics(SpendMetrics(), _spend_record("real-key", spend=1.0, ptu_flat_cost=240.0))

        assert metrics.flat_cost == 0.0
        assert metrics.spend == 1.0

    def test_aggregated_path_reports_zero_flat_cost(self):
        from litellm.proxy.management_endpoints.common_daily_activity import _GROUP_GRAND_TOTAL

        metrics = _record_to_spend_metrics(_grouping_row(_GROUP_GRAND_TOTAL, spend=5.0, ptu_flat_cost=240.0))

        assert metrics.flat_cost == 0.0
        assert metrics.spend == 5.0

    def test_aggregated_totals_and_buckets_report_zero_flat_cost(self):
        from litellm.constants import PTU_SENTINEL_API_KEY
        from litellm.proxy.management_endpoints.common_daily_activity import (
            _GROUP_DATE_API_KEY,
            _GROUP_DATE_MODEL,
            _GROUP_GRAND_TOTAL,
            _aggregate_grouping_sets_records_sync,
        )

        records = [
            _grouping_row(_GROUP_DATE_API_KEY, api_key=PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0),
            _grouping_row(_GROUP_DATE_MODEL, model="gpt-4o-mini-ptu", spend=5.0, ptu_flat_cost=240.0),
            _grouping_row(_GROUP_GRAND_TOTAL, spend=5.0, ptu_flat_cost=240.0),
        ]

        aggregated = _aggregate_grouping_sets_records_sync(records=records, api_key_metadata={})

        assert aggregated["totals"].flat_cost == 0.0
        assert aggregated["totals"].spend == 5.0
        assert aggregated["results"][0].breakdown.models["gpt-4o-mini-ptu"].metrics.flat_cost == 0.0

    def test_sentinel_still_excluded_from_the_api_key_breakdown(self):
        from litellm.constants import PTU_SENTINEL_API_KEY
        from litellm.proxy.management_endpoints.common_daily_activity import update_breakdown_metrics
        from litellm.types.proxy.management_endpoints.common_daily_activity import BreakdownMetrics

        breakdown = BreakdownMetrics()
        update_breakdown_metrics(breakdown, _spend_record("real-key", spend=5.0), {}, {}, {})
        update_breakdown_metrics(
            breakdown, _spend_record(PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0), {}, {}, {}, entity_id_field="team_id"
        )

        assert PTU_SENTINEL_API_KEY not in breakdown.api_keys
        assert PTU_SENTINEL_API_KEY not in breakdown.models["gpt-4o-mini-ptu"].api_key_breakdown
        assert "real-key" in breakdown.models["gpt-4o-mini-ptu"].api_key_breakdown

    def test_sentinel_still_excluded_from_the_provider_breakdown(self):
        from litellm.constants import PTU_SENTINEL_API_KEY
        from litellm.proxy.management_endpoints.common_daily_activity import update_breakdown_metrics
        from litellm.types.proxy.management_endpoints.common_daily_activity import BreakdownMetrics

        breakdown = BreakdownMetrics()
        update_breakdown_metrics(breakdown, _spend_record(PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0), {}, {}, {})

        assert breakdown.providers == {}

    def test_grouping_sets_sentinel_still_excluded_from_breakdowns(self):
        from litellm.constants import PTU_SENTINEL_API_KEY
        from litellm.proxy.management_endpoints.common_daily_activity import (
            _GROUP_DATE_API_KEY,
            _GROUP_DATE_MODEL,
            _GROUP_DATE_MODEL_API_KEY,
            _GROUP_DATE_PROVIDER,
            _aggregate_grouping_sets_records_sync,
        )

        records = [
            _grouping_row(_GROUP_DATE_API_KEY, api_key=PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0),
            _grouping_row(_GROUP_DATE_MODEL, model="gpt-4o-mini-ptu", spend=5.0, ptu_flat_cost=240.0),
            _grouping_row(
                _GROUP_DATE_MODEL_API_KEY, model="gpt-4o-mini-ptu", api_key=PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0
            ),
            _grouping_row(_GROUP_DATE_PROVIDER, custom_llm_provider="", ptu_flat_cost=240.0),
        ]

        day = _aggregate_grouping_sets_records_sync(records=records, api_key_metadata={})["results"][0]

        assert PTU_SENTINEL_API_KEY not in day.breakdown.api_keys
        assert PTU_SENTINEL_API_KEY not in day.breakdown.models["gpt-4o-mini-ptu"].api_key_breakdown
        assert sum(bucket.metrics.flat_cost for bucket in day.breakdown.providers.values()) == 0.0

    @pytest.mark.asyncio
    async def test_team_daily_activity_endpoint_reports_zero_flat_cost(self):
        """/team/daily/activity reads rows with find_many rather than the aggregated SQL, so
        forcing the SQL select to a constant zero would leave this path reporting flat cost."""
        from litellm.constants import PTU_SENTINEL_API_KEY

        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_table = MagicMock()
        mock_table.count = AsyncMock(return_value=2)
        mock_table.find_many = AsyncMock(
            return_value=[
                _daily_team_row("real-key", spend=5.0),
                _daily_team_row(PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0),
            ]
        )
        mock_prisma.db.litellm_verificationtoken = MagicMock()
        mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
        mock_prisma.db.litellm_dailyteamspend = mock_table

        result = await get_daily_activity(
            prisma_client=mock_prisma,
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id="team-1",
            entity_metadata_field=None,
            start_date="2026-07-01",
            end_date="2026-07-01",
            model=None,
            api_key=None,
            page=1,
            page_size=50,
        )

        assert result.metadata.total_flat_cost == 0.0
        assert result.metadata.total_spend == 5.0
        assert PTU_SENTINEL_API_KEY not in result.results[0].breakdown.api_keys

    @pytest.mark.asyncio
    async def test_team_daily_activity_endpoint_reports_flat_cost_once_enabled(self, monkeypatch):
        from litellm.constants import PTU_SENTINEL_API_KEY

        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")

        mock_prisma = MagicMock()
        mock_prisma.db = MagicMock()
        mock_table = MagicMock()
        mock_table.count = AsyncMock(return_value=2)
        mock_table.find_many = AsyncMock(
            return_value=[
                _daily_team_row("real-key", spend=5.0),
                _daily_team_row(PTU_SENTINEL_API_KEY, ptu_flat_cost=240.0),
            ]
        )
        mock_prisma.db.litellm_verificationtoken = MagicMock()
        mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
        mock_prisma.db.litellm_dailyteamspend = mock_table

        result = await get_daily_activity(
            prisma_client=mock_prisma,
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id="team-1",
            entity_metadata_field=None,
            start_date="2026-07-01",
            end_date="2026-07-01",
            model=None,
            api_key=None,
            page=1,
            page_size=50,
        )

        assert result.metadata.total_flat_cost == 240.0
        assert PTU_SENTINEL_API_KEY not in result.results[0].breakdown.api_keys


class TestFlagIsNotReadOnTheHotPath:
    """update_metrics runs once per accumulation and a record fans out across roughly a
    dozen breakdowns, so a flag that reads through the secret manager must not be consulted
    for rows that carry no flat cost at all."""

    @staticmethod
    def _count_flag_reads(records):
        import litellm.proxy.management_endpoints.common_daily_activity as cda
        from litellm.types.proxy.management_endpoints.common_daily_activity import BreakdownMetrics

        reads = []
        real = cda.is_ptu_cost_attribution_enabled

        def counted():
            reads.append(1)
            return real()

        cda.is_ptu_cost_attribution_enabled = counted
        try:
            breakdown = BreakdownMetrics()
            for record in records:
                cda.update_breakdown_metrics(breakdown, record, {}, {}, {})
        finally:
            cda.is_ptu_cost_attribution_enabled = real
        return len(reads)

    def test_a_request_row_never_reads_the_flag(self):
        reads = self._count_flag_reads([_spend_record("real-key", spend=5.0, ptu_flat_cost=0.0)])
        assert reads == 0, f"{reads} secret-manager lookups for a row with no flat cost"

    def test_a_page_of_request_rows_never_reads_the_flag(self):
        rows = [_spend_record(f"key-{i}", spend=1.0, ptu_flat_cost=0.0) for i in range(50)]
        assert self._count_flag_reads(rows) == 0

    def test_a_sentinel_row_still_consults_the_flag(self):
        from litellm.constants import PTU_SENTINEL_API_KEY

        reads = self._count_flag_reads([_spend_record(PTU_SENTINEL_API_KEY, spend=0.0, ptu_flat_cost=240.0)])
        assert reads > 0


def test_entity_rollup_sql_query_and_api_key_list_filter():
    """The entity rollup companion query keeps its own two grouping sets keyed
    by GROUPING(api_key), shares the WHERE builder (list api_key becomes a
    parameterized IN, an empty list must match nothing), and the main
    aggregated query stays entity-free."""
    from litellm.proxy.management_endpoints.common_daily_activity import (
        _build_entity_rollup_sql_query,
    )

    sql, params = _build_entity_rollup_sql_query(
        table_name="litellm_dailyteamspend",
        entity_id_field="team_id",
        entity_id=None,
        start_date="2024-01-01",
        end_date="2024-01-31",
        model=None,
        api_key=["key-1", "key-2"],
    )
    assert '"team_id" AS entity_id' in sql
    assert "GROUPING(api_key) AS api_key_rolled" in sql
    assert '(date, "team_id"),' in sql
    assert '(date, "team_id", api_key)' in sql
    assert "api_key IN ($3, $4)" in sql
    assert "SUM(ptu_flat_cost)::float" in sql
    assert params == ["2024-01-01", "2024-01-31", "key-1", "key-2"]

    plain_sql, _ = _build_aggregated_sql_query(
        table_name="litellm_dailyteamspend",
        entity_id_field="team_id",
        entity_id=None,
        start_date="2024-01-01",
        end_date="2024-01-31",
        model=None,
        api_key=None,
    )
    assert "entity_id" not in plain_sql
    assert "GROUPING(date" in plain_sql

    empty_sql, empty_params = _build_aggregated_sql_query(
        table_name="litellm_dailyteamspend",
        entity_id_field="team_id",
        entity_id=None,
        start_date="2024-01-01",
        end_date="2024-01-31",
        model=None,
        api_key=[],
    )
    assert "FALSE" in empty_sql
    assert empty_params == ["2024-01-01", "2024-01-31"]


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_with_entity_breakdown():
    """include_entity_breakdown must run the companion entity rollup query and
    fold breakdown.entities onto the response, without disturbing the main
    query's rollup dispatch."""
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    base = {
        "model": None,
        "model_group": None,
        "custom_llm_provider": None,
        "mcp_namespaced_tool_name": None,
        "endpoint": None,
        "api_key": None,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 0,
        "compression_savings_spend": 0.0,
        "prompt_caching_savings_spend": 0.0,
        "autorouter_savings_spend": 0.0,
        "failed_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "api_requests": 0,
        "successful_requests": 0,
    }
    main_rows = [
        {**base, "date": None, "group_level": 127, "spend": 18.0},
        {**base, "date": "2024-01-01", "group_level": 63, "spend": 18.0},
        {**base, "date": "2024-01-01", "model": "gpt-4o", "group_level": 47, "spend": 18.0},
        {**base, "date": "2024-01-01", "api_key": "key-1", "group_level": 31, "spend": 12.0},
    ]
    entity_base = {
        key: value
        for key, value in base.items()
        if key not in ("model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint")
    }
    entity_rows = [
        {**entity_base, "date": "2024-01-01", "entity_id": "team-a", "api_key_rolled": 1, "spend": 12.0},
        {**entity_base, "date": "2024-01-01", "entity_id": "team-b", "api_key_rolled": 1, "spend": 6.0},
        {
            **entity_base,
            "date": "2024-01-01",
            "entity_id": "team-a",
            "api_key": "key-1",
            "api_key_rolled": 0,
            "spend": 12.0,
        },
        {
            **entity_base,
            "date": "2024-01-01",
            "entity_id": "team-b",
            "api_key": "key-2",
            "api_key_rolled": 0,
            "spend": 6.0,
        },
    ]

    mock_prisma.db.query_raw = AsyncMock(side_effect=[main_rows, entity_rows])
    mock_prisma.db.litellm_verificationtoken = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    result = await get_daily_activity_aggregated(
        prisma_client=mock_prisma,
        table_name="litellm_dailyteamspend",
        entity_id_field="team_id",
        entity_id=None,
        entity_metadata_field={"team-a": {"team_alias": "Alpha"}},
        start_date="2024-01-01",
        end_date="2024-01-01",
        model=None,
        api_key=None,
        include_entity_breakdown=True,
    )

    assert mock_prisma.db.query_raw.call_count == 2
    main_sql = mock_prisma.db.query_raw.call_args_list[0][0][0]
    entity_sql = mock_prisma.db.query_raw.call_args_list[1][0][0]
    assert "entity_id" not in main_sql
    assert '"team_id" AS entity_id' in entity_sql
    assert '(date, "team_id"),' in entity_sql

    assert result.metadata.total_spend == 18.0
    assert len(result.results) == 1
    daily = result.results[0]
    assert daily.metrics.spend == 18.0

    entities = daily.breakdown.entities
    assert set(entities) == {"team-a", "team-b"}
    assert entities["team-a"].metrics.spend == 12.0
    assert entities["team-a"].metadata == {"team_alias": "Alpha"}
    assert entities["team-a"].api_key_breakdown["key-1"].metrics.spend == 12.0
    assert entities["team-b"].metrics.spend == 6.0
    assert entities["team-b"].metadata == {}
    assert entities["team-b"].api_key_breakdown["key-2"].metrics.spend == 6.0

    # Rollups with the entity bit set must still land in their usual buckets
    assert daily.breakdown.models["gpt-4o"].metrics.spend == 18.0
    assert daily.breakdown.api_keys["key-1"].metrics.spend == 12.0
