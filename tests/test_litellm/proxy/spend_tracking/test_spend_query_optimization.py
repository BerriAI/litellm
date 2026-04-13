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
    assert (
        "::date" not in sql.lower()
    ), "SQL should not use '::date' casting which prevents index usage"
    assert (
        "date(" not in sql.lower()
    ), "SQL should not use DATE() function which prevents index usage"

    # 2) SQL should use timestamp-range filtering pattern for index optimization
    assert (
        '"startTime" >=' in sql or '"startTime">=' in sql
    ), "SQL should use >= operator for lower bound"
    assert (
        '"startTime" <' in sql or '"startTime"<' in sql
    ), "SQL should use < operator for upper bound"
    assert (
        "interval '1 day'" in sql.lower()
    ), "SQL should use INTERVAL for date arithmetic"

    # 3) Parameters should be datetime objects (not date objects)
    assert isinstance(
        params[0], datetime.datetime
    ), "First parameter (start_date) should be datetime object"
    assert isinstance(
        params[1], datetime.datetime
    ), "Second parameter (end_date) should be datetime object"
    assert params[0].tzinfo is not None, "start_date should be timezone-aware"
    assert params[1].tzinfo is not None, "end_date should be timezone-aware"


@pytest.mark.asyncio
async def test_global_activity_wraps_params_in_at_time_zone_utc(monkeypatch):
    """
    /global/activity must emit `AT TIME ZONE 'UTC'` around its date params
    so the date window and `date_trunc` bucketing do not depend on the DB
    session timezone. Regression guard for Issue 1.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        get_global_activity,
    )

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")

    await get_global_activity(
        start_date="2026-02-16",
        end_date="2026-02-16",
        user_api_key_dict=auth,
    )

    assert mock_prisma.db.query_raw.called, "query_raw should have been called"
    call_args = mock_prisma.db.query_raw.call_args[0]
    sql = call_args[0]
    params = call_args[1:]

    # 1) SQL must wrap both bounds in `AT TIME ZONE 'UTC'`.
    assert sql.count("AT TIME ZONE 'UTC'") >= 2, (
        "Both date bounds must be wrapped with `AT TIME ZONE 'UTC'` so that "
        "comparison against the plain-timestamp column is session-TZ-independent. "
        f"SQL was:\n{sql}"
    )

    # 2) Params must still be tz-aware UTC datetimes (preserves existing contract).
    assert isinstance(params[0], datetime.datetime)
    assert isinstance(params[1], datetime.datetime)
    assert params[0].tzinfo is not None and params[0].utcoffset() == datetime.timedelta(
        0
    )
    assert params[1].tzinfo is not None and params[1].utcoffset() == datetime.timedelta(
        0
    )


@pytest.mark.asyncio
async def test_global_activity_internal_user_wraps_params_in_at_time_zone_utc(
    monkeypatch,
):
    """
    The internal-user branch of /global/activity goes through a different
    helper (`get_global_activity_internal_user`) and has its own SQL string.
    Both branches must carry the fix. Regression guard for Issue 1.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        get_global_activity,
    )

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="internal_user_1"
    )

    await get_global_activity(
        start_date="2026-02-16",
        end_date="2026-02-16",
        user_api_key_dict=auth,
    )

    assert mock_prisma.db.query_raw.called
    sql = mock_prisma.db.query_raw.call_args[0][0]
    assert sql.count("AT TIME ZONE 'UTC'") >= 2, (
        "Internal-user branch must also wrap date bounds with "
        f"`AT TIME ZONE 'UTC'`. SQL was:\n{sql}"
    )


@pytest.mark.asyncio
async def test_spend_logs_ui_wraps_params_in_at_time_zone_utc(monkeypatch):
    """
    /spend/logs/ui builds its WHERE clause dynamically. The date-range
    conditions must wrap the param side with `AT TIME ZONE 'UTC'` so the
    log filter window doesn't drift with the DB session TZ. Regression
    guard for GH #22529.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        ui_view_spend_logs,
    )

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])
    mock_prisma.db.litellm_spendlogs = MagicMock()
    mock_prisma.db.litellm_spendlogs.count = AsyncMock(return_value=0)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")

    mock_request = MagicMock()
    mock_request.url.path = "/spend/logs/ui"

    await ui_view_spend_logs(
        request=mock_request,
        api_key=None,
        user_id=None,
        request_id=None,
        start_date="2026-02-16 00:00:00",
        end_date="2026-02-16 23:59:59",
        page=1,
        page_size=50,
        sort_by="startTime",
        sort_order="desc",
        user_api_key_dict=auth,
    )

    assert mock_prisma.db.query_raw.called, "query_raw should have been called"
    sql = mock_prisma.db.query_raw.call_args[0][0]
    assert sql.count("AT TIME ZONE 'UTC'") >= 2, (
        "/spend/logs/ui must wrap both `startTime` bounds with "
        f"`AT TIME ZONE 'UTC'`. SQL was:\n{sql}"
    )
