"""
Test that spend queries use timestamp filtering instead of date casting.

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
async def test_spend_query_uses_timestamp_filtering():
    """
    Test that spend queries use timestamp filtering for index optimization.

    Verifies:
    1. SQL does NOT cast the startTime column to DATE (which prevents index usage)
    2. SQL uses >= and < operators with INTERVAL for timestamp range filtering
    3. Parameters passed are datetime objects (not date objects)
    """
    # Mock prisma client
    mock_prisma = MagicMock()
    mock_db = MagicMock()
    mock_query_raw = AsyncMock(return_value=[])
    mock_db.query_raw = mock_query_raw
    mock_prisma.db = mock_db

    # Use timezone-aware datetime objects
    start_date = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.datetime(2024, 1, 31, tzinfo=timezone.utc)

    # Call the function
    await get_spend_by_team_and_customer(
        start_date=start_date,
        end_date=end_date,
        team_id="test_team",
        customer_id="test_customer",
        prisma_client=mock_prisma,
    )

    # Verify the query was called
    assert mock_query_raw.called, "query_raw should have been called"

    # Extract SQL and parameters
    # Prisma query_raw is called like: query_raw(sql, param1, param2, ...)
    call_args = mock_query_raw.call_args[0]
    sql = call_args[0]
    params = call_args[1:]

    # 1) SQL should NOT cast the startTime column to DATE (prevents index usage)
    assert "::date" not in sql.lower(), \
        "SQL should not use '::date' casting which prevents index usage"
    assert "date(" not in sql.lower(), \
        "SQL should not use DATE() function which prevents index usage"

    # 2) SQL should use timestamp-range filtering pattern for index optimization
    assert '"startTime" >=' in sql or '"startTime">=' in sql, \
        "SQL should use >= operator for lower bound"
    assert '"startTime" <' in sql or '"startTime"<' in sql, \
        "SQL should use < operator for upper bound"
    assert "interval '1 day'" in sql.lower(), \
        "SQL should use INTERVAL for date arithmetic"

    # 3) Parameters should be datetime objects (not date objects)
    assert isinstance(params[0], datetime.datetime), \
        "First parameter (start_date) should be datetime object"
    assert isinstance(params[1], datetime.datetime), \
        "Second parameter (end_date) should be datetime object"
    assert params[0].tzinfo is not None, \
        "start_date should be timezone-aware"
    assert params[1].tzinfo is not None, \
        "end_date should be timezone-aware"
