import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.management_endpoints.common_daily_activity import (
    _is_user_agent_tag,
    compute_tag_metadata_totals,
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

    # Create mock records with endpoint fields
    class MockRecord:
        def __init__(self, date, endpoint, api_key, model, spend, prompt_tokens, completion_tokens):
            self.date = date
            self.endpoint = endpoint
            self.api_key = api_key
            self.model = model
            self.model_group = None
            self.custom_llm_provider = "openai"
            self.mcp_namespaced_tool_name = None
            self.spend = spend
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = prompt_tokens + completion_tokens
            self.cache_read_input_tokens = 0
            self.cache_creation_input_tokens = 0
            self.api_requests = 1
            self.successful_requests = 1
            self.failed_requests = 0

    mock_records = [
        MockRecord("2024-01-01", "/v1/chat/completions", "key-1", "gpt-4", 10.0, 100, 50),
        MockRecord("2024-01-01", "/v1/chat/completions", "key-1", "gpt-4", 5.0, 50, 25),
        MockRecord("2024-01-01", "/v1/embeddings", "key-2", "text-embedding-ada-002", 3.0, 30, 0),
    ]

    # Mock the table methods
    mock_table = MagicMock()
    mock_table.find_many = AsyncMock(return_value=mock_records)
    mock_prisma.db.litellm_dailyuserspend = mock_table
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
