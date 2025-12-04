"""
Test that spend queries use timestamp parameters instead of date parameters.

This prevents the performance issue where date casting prevents index usage.
GitHub Issue: #17487
"""

import datetime
import os
import sys
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.spend_tracking.spend_tracking_utils import (
    get_spend_by_team_and_customer,
)


@pytest.mark.asyncio
async def test_spend_query_accepts_timestamp_parameters():
    """
    Test that spend queries accept datetime (timestamp) parameters.

    The fix changes queries from using date casting (slow, no index)
    to timestamp filtering (fast, uses indexes).
    """
    # Mock prisma client
    mock_prisma = MagicMock()
    mock_db = MagicMock()
    mock_query_raw = AsyncMock(return_value=[])
    mock_db.query_raw = mock_query_raw
    mock_prisma.db = mock_db

    # Use datetime objects (timestamps)
    start_date = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.datetime(2024, 1, 31, tzinfo=timezone.utc)

    # Call function - should work with timestamp parameters
    await get_spend_by_team_and_customer(
        start_date=start_date,
        end_date=end_date,
        team_id="test_team",
        customer_id="test_customer",
        prisma_client=mock_prisma,
    )

    # Verify it was called
    assert mock_query_raw.called

    # Get the parameters
    call_args = mock_query_raw.call_args[0]

    # Verify parameters are timestamps (not dates)
    assert isinstance(call_args[1], datetime.datetime), "start_date should be datetime"
    assert isinstance(call_args[2], datetime.datetime), "end_date should be datetime"
