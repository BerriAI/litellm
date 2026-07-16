import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.management_endpoints.common_daily_activity import (
    _adjust_dates_for_timezone,
    _aggregate_grouping_sets_records_sync,
    _build_aggregated_sql_query,
    _is_user_agent_tag,
    _record_to_spend_metrics,
    get_api_key_metadata,
    get_daily_activity,
    get_daily_activity_aggregated,
    update_breakdown_metrics,
    update_metrics,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    BreakdownMetrics,
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
        f"order must include the id tiebreaker after date for stable offset "
        f"pagination (see #30164); got {order!r}"
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

    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[mock_active_key]
    )

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

    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(
        return_value=[mock_deleted_key]
    )

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

    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[mock_active_key]
    )

    # One deleted key found
    mock_deleted_key = MagicMock()
    mock_deleted_key.token = "deleted-key-hash"
    mock_deleted_key.key_alias = "deleted-alias"
    mock_deleted_key.team_id = "team-deleted"

    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(
        return_value=[mock_deleted_key]
    )

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

    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[mock_active_key]
    )
    mock_prisma.db.litellm_deletedverificationtoken = MagicMock()
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(
        return_value=[]
    )

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
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(
        side_effect=Exception("Table not found")
    )

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
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(
        return_value=[mock_deleted_1, mock_deleted_2]
    )

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
    mock_prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(
        return_value=[mock_deleted_key]
    )

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


def _daily_user_spend_record(*, user_id, api_key, spend):
    """A LiteLLM_DailyUserSpend row as the per-user breakdown reads it."""
    return SimpleNamespace(
        date="2024-01-01",
        user_id=user_id,
        api_key=api_key,
        model="gpt-4",
        model_group="gpt-4",
        custom_llm_provider="openai",
        mcp_namespaced_tool_name=None,
        endpoint="/chat/completions",
        spend=spend,
        prompt_tokens=10,
        completion_tokens=5,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
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
        start, end = _adjust_dates_for_timezone(
            "2026-05-29", "2026-05-29", offset_minutes
        )
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
        single_day_ranges = [
            _adjust_dates_for_timezone(d, d, offset_minutes) for d in days
        ]
        multi_day_range = _adjust_dates_for_timezone(days[0], days[-1], offset_minutes)

        per_day_starts = [r[0] for r in single_day_ranges]
        per_day_ends = [r[1] for r in single_day_ranges]
        assert min(per_day_starts) == multi_day_range[0]
        assert max(per_day_ends) == multi_day_range[1]
        assert per_day_starts == days
        assert per_day_ends == days


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

    def test_includes_model_breakdown_for_model_groups(self):
        sql, _ = _build_aggregated_sql_query(
            table_name="litellm_dailyuserspend",
            entity_id_field="user_id",
            entity_id=None,
            start_date="2026-05-29",
            end_date="2026-05-29",
            model=None,
            api_key=None,
        )

        assert "(date, model, model_group)" in sql
        assert "(date, api_key, model, model_group)" in sql


def test_aggregate_grouping_sets_populates_model_group_model_breakdown():
    record = SimpleNamespace(
        group_level=39,
        date="2026-05-29",
        api_key=None,
        model="bedrock/claude-opus-4-8",
        model_group="smart-router",
        custom_llm_provider=None,
        mcp_namespaced_tool_name=None,
        endpoint=None,
        spend=12.5,
        prompt_tokens=8000,
        completion_tokens=2000,
        cache_read_input_tokens=4000,
        cache_creation_input_tokens=1000,
        api_requests=8,
        successful_requests=8,
        failed_requests=0,
    )

    result = _aggregate_grouping_sets_records_sync(records=[record], api_key_metadata={})

    metrics = result["results"][0].breakdown.model_groups["smart-router"].model_breakdown[
        "bedrock/claude-opus-4-8"
    ]
    assert metrics.api_requests == 8
    assert metrics.total_tokens == 10000
    assert metrics.cache_read_input_tokens == 4000
    assert metrics.cache_creation_input_tokens == 1000


def test_aggregate_grouping_sets_populates_key_model_group_model_breakdown():
    model_record = SimpleNamespace(
        group_level=7,
        date="2026-05-29",
        api_key="key-hash",
        model="bedrock/claude-opus-4-8",
        model_group="smart-router",
        custom_llm_provider=None,
        mcp_namespaced_tool_name=None,
        endpoint=None,
        spend=12.5,
        prompt_tokens=8000,
        completion_tokens=2000,
        cache_read_input_tokens=4000,
        cache_creation_input_tokens=1000,
        api_requests=8,
        successful_requests=8,
        failed_requests=0,
    )
    model_group_key_record = SimpleNamespace(
        **{
            **model_record.__dict__,
            "group_level": 23,
            "model": None,
        }
    )

    result = _aggregate_grouping_sets_records_sync(
        records=[model_record, model_group_key_record],
        api_key_metadata={},
    )

    key_metrics = (
        result["results"][0]
        .breakdown.model_groups["smart-router"]
        .api_key_breakdown["key-hash"]
    )
    metrics = key_metrics.model_breakdown["bedrock/claude-opus-4-8"]
    assert key_metrics.metrics.api_requests == 8
    assert metrics.api_requests == 8
    assert metrics.total_tokens == 10000
    assert metrics.cache_read_input_tokens == 4000
    assert metrics.cache_creation_input_tokens == 1000


def test_update_breakdown_metrics_populates_model_group_model_breakdown():
    record = SimpleNamespace(
        model="bedrock/claude-sonnet-4-5",
        model_group="smart-router",
        api_key="key-hash",
        custom_llm_provider="bedrock",
        mcp_namespaced_tool_name=None,
        endpoint=None,
        spend=4.5,
        prompt_tokens=3000,
        completion_tokens=1000,
        cache_read_input_tokens=1500,
        cache_creation_input_tokens=500,
        api_requests=2,
        successful_requests=2,
        failed_requests=0,
    )

    result = update_breakdown_metrics(
        breakdown=BreakdownMetrics(),
        record=record,
        model_metadata={},
        provider_metadata={},
        api_key_metadata={},
    )

    metrics = result.model_groups["smart-router"].model_breakdown[
        "bedrock/claude-sonnet-4-5"
    ]
    assert metrics.api_requests == 2
    assert metrics.total_tokens == 4000
    assert metrics.cache_read_input_tokens == 1500
    assert metrics.cache_creation_input_tokens == 500
    key_metrics = result.model_groups["smart-router"].api_key_breakdown["key-hash"].model_breakdown[
        "bedrock/claude-sonnet-4-5"
    ]
    assert key_metrics.api_requests == 2
    assert key_metrics.total_tokens == 4000
    assert key_metrics.cache_read_input_tokens == 1500
    assert key_metrics.cache_creation_input_tokens == 500


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


def _no_spend_record():
    """A rollup row for a key with no spend, where SUM() returns NULL (None)."""
    return SimpleNamespace(
        spend=None,
        prompt_tokens=None,
        completion_tokens=None,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
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
