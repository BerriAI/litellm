import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.management_endpoints.common_daily_activity import (
    _build_aggregated_sql_query,
    _is_user_agent_tag,
    compute_tag_metadata_totals,
    get_api_key_metadata,
    get_daily_activity,
    get_daily_activity_aggregated,
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


def test_compute_tag_metadata_totals():
    """Test compute_tag_metadata_totals function."""
    # Create mock records
    class MockRecord:
        def __init__(self, request_id, tag, spend, prompt_tokens=10, completion_tokens=5):
            self.request_id = request_id
            self.tag = tag
            self.spend = spend
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens
            self.cache_read_input_tokens = 0
            self.cache_creation_input_tokens = 0
            self.api_requests = 1
            self.successful_requests = 1
            self.failed_requests = 0

    # Test deduplication by request_id (keeps max spend)
    records = [
        MockRecord("req-1", "production", spend=10.0),
        MockRecord("req-1", "staging", spend=20.0),  # Higher spend, should be kept
        MockRecord("req-2", "production", spend=15.0),
    ]
    result = compute_tag_metadata_totals(records)
    assert result.spend == 35.0  # 20.0 + 15.0 (deduplicated req-1)
    assert result.prompt_tokens == 20  # 10 + 10 (only deduplicated records)
    assert result.completion_tokens == 10  # 5 + 5 (only deduplicated records)

    # Test ignoring user-agent tags
    records_with_ua = [
        MockRecord("req-1", "production", spend=10.0),
        MockRecord("req-1", "user-agent:chrome", spend=50.0),  # Should be ignored
        MockRecord("req-2", "staging", spend=15.0),
    ]
    result = compute_tag_metadata_totals(records_with_ua)
    assert result.spend == 25.0  # 10.0 + 15.0 (user-agent ignored)

    # Test ignoring records without request_id
    records_no_req_id = [
        MockRecord("req-1", "production", spend=10.0),
        MockRecord(None, "staging", spend=20.0),  # Should be ignored
    ]
    result = compute_tag_metadata_totals(records_no_req_id)
    assert result.spend == 10.0

    # Test empty records
    result = compute_tag_metadata_totals([])
    assert result.spend == 0.0
    assert result.prompt_tokens == 0


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_with_endpoint_breakdown():
    """Test that endpoint breakdown is included in aggregated daily activity."""
    # Mock PrismaClient
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    # query_raw returns list of dicts (pre-aggregated by GROUP BY)
    mock_rows = [
        {
            "date": "2024-01-01",
            "endpoint": "/v1/chat/completions",
            "api_key": "key-1",
            "model": "gpt-4",
            "model_group": None,
            "custom_llm_provider": "openai",
            "mcp_namespaced_tool_name": None,
            "spend": 15.0,
            "prompt_tokens": 150,
            "completion_tokens": 75,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "api_requests": 2,
            "successful_requests": 2,
            "failed_requests": 0,
        },
        {
            "date": "2024-01-01",
            "endpoint": "/v1/embeddings",
            "api_key": "key-2",
            "model": "text-embedding-ada-002",
            "model_group": None,
            "custom_llm_provider": "openai",
            "mcp_namespaced_tool_name": None,
            "spend": 3.0,
            "prompt_tokens": 30,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
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
async def test_aggregated_activity_preserves_metadata_for_deleted_keys():
    """Test that the full aggregation pipeline should preserve metadata for deleted keys."""
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()

    # query_raw returns list of dicts (pre-aggregated by GROUP BY)
    mock_rows = [
        {
            "date": "2024-01-01",
            "endpoint": "/v1/chat/completions",
            "api_key": "deleted-key-hash",
            "model": "gpt-4",
            "model_group": None,
            "custom_llm_provider": "openai",
            "mcp_namespaced_tool_name": None,
            "spend": 10.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
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


class TestBuildAggregatedSqlQuery:
    """Tests for _build_aggregated_sql_query conditional GROUP BY behavior."""

    def test_include_entity_id_true_no_group_by(self):
        """When include_entity_id=True, skip GROUP BY (it's a no-op on unique constraint)."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
            include_entity_id=True,
        )
        assert "GROUP BY" not in sql
        assert "SUM(" not in sql
        assert '"team_id",' in sql
        assert "spend::float AS spend" in sql
        assert "ORDER BY date DESC" in sql
        assert '"LiteLLM_DailyTeamSpend"' in sql
        assert params == ["2024-01-01", "2024-01-31"]

    def test_include_entity_id_false_uses_group_by(self):
        """When include_entity_id=False (default), use GROUP BY to collapse across entities."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
            include_entity_id=False,
        )
        assert "GROUP BY" in sql
        assert "SUM(spend)" in sql
        assert '"team_id",' not in sql
        assert "ORDER BY date DESC" in sql
        assert params == ["2024-01-01", "2024-01-31"]

    def test_group_by_excludes_model_group(self):
        """GROUP BY should not include model_group (uses MAX instead)."""
        sql, _ = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
            include_entity_id=False,
        )
        assert "MAX(model_group)" in sql
        # model_group should not appear in GROUP BY clause
        group_by_clause = sql.split("GROUP BY")[1].split("ORDER BY")[0]
        assert "model_group" not in group_by_clause

    def test_include_entity_id_true_selects_columns_directly(self):
        """With include_entity_id=True, metric columns are selected directly (no SUM)."""
        sql, _ = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
            include_entity_id=True,
        )
        assert "spend::float AS spend" in sql
        assert "prompt_tokens::bigint AS prompt_tokens" in sql
        assert "completion_tokens::bigint AS completion_tokens" in sql
        assert "api_requests::bigint AS api_requests" in sql

    def test_entity_id_single_value_filter(self):
        """Single entity_id value should produce an = condition."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id="team-abc",
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
        )
        assert '"team_id" = $3' in sql
        assert params == ["2024-01-01", "2024-01-31", "team-abc"]

    def test_entity_id_list_filter(self):
        """List of entity_ids should produce an IN condition."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=["team-a", "team-b"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
        )
        assert '"team_id" IN ($3, $4)' in sql
        assert params == ["2024-01-01", "2024-01-31", "team-a", "team-b"]

    def test_exclude_entity_ids(self):
        """exclude_entity_ids should produce a NOT IN condition."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
            exclude_entity_ids=["team-x", "team-y"],
        )
        assert '"team_id" NOT IN ($3, $4)' in sql
        assert params == ["2024-01-01", "2024-01-31", "team-x", "team-y"]

    def test_model_filter(self):
        """model filter should add a model = condition."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model="gpt-4",
            api_key=None,
        )
        assert "model = $3" in sql
        assert params == ["2024-01-01", "2024-01-31", "gpt-4"]

    def test_api_key_single_filter(self):
        """Single api_key should produce an = condition."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key="key-123",
        )
        assert "api_key = $3" in sql
        assert params == ["2024-01-01", "2024-01-31", "key-123"]

    def test_api_key_list_filter(self):
        """List of api_keys should produce an IN condition."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=["key-1", "key-2"],
        )
        assert "api_key IN ($3, $4)" in sql
        assert params == ["2024-01-01", "2024-01-31", "key-1", "key-2"]

    def test_all_filters_combined(self):
        """All filters together should produce correct parameter ordering."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id="team-abc",
            start_date="2024-01-01",
            end_date="2024-01-31",
            model="gpt-4",
            api_key="key-1",
            exclude_entity_ids=["team-x"],
        )
        assert params == [
            "2024-01-01",
            "2024-01-31",
            "team-abc",
            "team-x",
            "gpt-4",
            "key-1",
        ]
        assert '"team_id" = $3' in sql
        assert '"team_id" NOT IN ($4)' in sql
        assert "model = $5" in sql
        assert "api_key = $6" in sql

    def test_timezone_offset_positive(self):
        """Positive timezone offset (west of UTC) should extend end_date by 1 day."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
            timezone_offset_minutes=480,  # PST
        )
        assert params == ["2024-01-01", "2024-02-01"]

    def test_timezone_offset_negative(self):
        """Negative timezone offset (east of UTC) should extend start_date by -1 day."""
        sql, params = _build_aggregated_sql_query(
            table_name="litellm_dailyteamspend",
            entity_id_field="team_id",
            entity_id=None,
            start_date="2024-01-01",
            end_date="2024-01-31",
            model=None,
            api_key=None,
            timezone_offset_minutes=-330,  # IST
        )
        assert params == ["2023-12-31", "2024-01-31"]

    def test_unknown_table_name_raises(self):
        """Unknown table name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown table name"):
            _build_aggregated_sql_query(
                table_name="nonexistent_table",
                entity_id_field="team_id",
                entity_id=None,
                start_date="2024-01-01",
                end_date="2024-01-31",
                model=None,
                api_key=None,
            )

    def test_all_table_names_supported(self):
        """All known table names should produce valid queries in both modes."""
        tables = [
            ("litellm_dailyuserspend", "user_id", "LiteLLM_DailyUserSpend"),
            ("litellm_dailyteamspend", "team_id", "LiteLLM_DailyTeamSpend"),
            (
                "litellm_dailyorganizationspend",
                "organization_id",
                "LiteLLM_DailyOrganizationSpend",
            ),
            ("litellm_dailyenduserspend", "end_user_id", "LiteLLM_DailyEndUserSpend"),
            ("litellm_dailyagentspend", "agent_id", "LiteLLM_DailyAgentSpend"),
            ("litellm_dailytagspend", "tag", "LiteLLM_DailyTagSpend"),
        ]
        for table_name, entity_field, pg_table in tables:
            # include_entity_id=True: no GROUP BY
            sql, _ = _build_aggregated_sql_query(
                table_name=table_name,
                entity_id_field=entity_field,
                entity_id=None,
                start_date="2024-01-01",
                end_date="2024-01-31",
                model=None,
                api_key=None,
                include_entity_id=True,
            )
            assert f'FROM "{pg_table}"' in sql
            assert "GROUP BY" not in sql

            # include_entity_id=False: uses GROUP BY
            sql, _ = _build_aggregated_sql_query(
                table_name=table_name,
                entity_id_field=entity_field,
                entity_id=None,
                start_date="2024-01-01",
                end_date="2024-01-31",
                model=None,
                api_key=None,
                include_entity_id=False,
            )
            assert f'FROM "{pg_table}"' in sql
            assert "GROUP BY" in sql
