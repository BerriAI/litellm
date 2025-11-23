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
