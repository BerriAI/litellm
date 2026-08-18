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
    get_spend_by_team,
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
    assert "::date" not in sql.lower(), "SQL should not use '::date' casting which prevents index usage"
    assert "date(" not in sql.lower(), "SQL should not use DATE() function which prevents index usage"

    # 2) SQL should use timestamp-range filtering pattern for index optimization
    assert '"startTime" >=' in sql or '"startTime">=' in sql, "SQL should use >= operator for lower bound"
    assert '"startTime" <' in sql or '"startTime"<' in sql, "SQL should use < operator for upper bound"
    assert "interval '1 day'" in sql.lower(), "SQL should use INTERVAL for date arithmetic"

    # 3) Parameters should be datetime objects (not date objects)
    assert isinstance(params[0], datetime.datetime), "First parameter (start_date) should be datetime object"
    assert isinstance(params[1], datetime.datetime), "Second parameter (end_date) should be datetime object"
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
    assert params[0].tzinfo is not None and params[0].utcoffset() == datetime.timedelta(0)
    assert params[1].tzinfo is not None and params[1].utcoffset() == datetime.timedelta(0)


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

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="internal_user_1")

    await get_global_activity(
        start_date="2026-02-16",
        end_date="2026-02-16",
        user_api_key_dict=auth,
    )

    assert mock_prisma.db.query_raw.called
    sql = mock_prisma.db.query_raw.call_args[0][0]
    assert sql.count("AT TIME ZONE 'UTC'") >= 2, (
        f"Internal-user branch must also wrap date bounds with `AT TIME ZONE 'UTC'`. SQL was:\n{sql}"
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
        f"/spend/logs/ui must wrap both `startTime` bounds with `AT TIME ZONE 'UTC'`. SQL was:\n{sql}"
    )


def _make_ui_spend_logs_mock(count_total, page_rows):
    """
    Build a prisma mock whose first `query_raw` (the bounded count) returns
    `count_total` and whose second `query_raw` (the page data) returns
    `page_rows`.
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(
        side_effect=[[{"total_count": count_total}], page_rows]
    )
    mock_prisma.db.litellm_spendlogs = MagicMock()
    mock_prisma.db.litellm_spendlogs.count = AsyncMock(return_value=0)
    return mock_prisma


@pytest.mark.asyncio
async def test_spend_logs_ui_uses_bounded_count_not_full_scan(monkeypatch):
    """
    /spend/logs/ui must compute its pagination total with a bounded
    `SELECT COUNT(*) FROM (SELECT 1 ... LIMIT $cap+1)` so it never scans the
    whole time window of a huge LiteLLM_SpendLogs table (Aurora ACU spike,
    LIT-4119). It must also avoid the unbounded prisma `.count()` /
    `COUNT(*) OVER ()` full-window count that reads every matching row.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        SPEND_LOGS_PAGINATION_COUNT_CAP,
        ui_view_spend_logs,
    )

    page_rows = [
        {"request_id": "req-1", "metadata": "{}", "session_id": None},
        {"request_id": "req-2", "metadata": "{}", "session_id": None},
    ]
    mock_prisma = _make_ui_spend_logs_mock(count_total=137, page_rows=page_rows)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
    mock_request = MagicMock()
    mock_request.url.path = "/spend/logs/ui"

    response = await ui_view_spend_logs(
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

    mock_prisma.db.litellm_spendlogs.count.assert_not_called()

    count_call = mock_prisma.db.query_raw.call_args_list[0]
    count_sql = count_call[0][0]
    assert "COUNT(*) OVER ()" not in count_sql
    assert "LIMIT" in count_sql and "FROM (" in count_sql, (
        "the total must come from a bounded subquery count, not a full-window "
        f"scan. SQL was:\n{count_sql}"
    )
    assert count_call[0][-1] == SPEND_LOGS_PAGINATION_COUNT_CAP + 1, (
        "the bounded count must probe at most cap+1 rows"
    )

    page_sql = mock_prisma.db.query_raw.call_args_list[1][0][0]
    assert "COUNT(*) OVER ()" not in page_sql, (
        "the page query must not carry a window count that forces a full-window "
        f"scan. SQL was:\n{page_sql}"
    )

    assert response["total"] == 137
    assert response["total_is_capped"] is False
    assert response["total_pages"] == (137 + 50 - 1) // 50

    for row in response["data"]:
        assert "total_count" not in row, "the window-function helper column must be stripped before serialising rows"


@pytest.mark.asyncio
async def test_spend_logs_ui_caps_total_for_large_result_sets(monkeypatch):
    """
    When more than the cap match, /spend/logs/ui reports the cap and flags
    `total_is_capped` so the UI can render `<cap>+` instead of an exact total
    that would require scanning the whole window (LIT-4119).
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        SPEND_LOGS_PAGINATION_COUNT_CAP,
        ui_view_spend_logs,
    )

    page_rows = [{"request_id": "req-1", "metadata": "{}", "session_id": None}]
    mock_prisma = _make_ui_spend_logs_mock(
        count_total=SPEND_LOGS_PAGINATION_COUNT_CAP + 1, page_rows=page_rows
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
    mock_request = MagicMock()
    mock_request.url.path = "/spend/logs/ui"

    response = await ui_view_spend_logs(
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

    assert response["total"] == SPEND_LOGS_PAGINATION_COUNT_CAP
    assert response["total_is_capped"] is True
    assert response["total_pages"] == (SPEND_LOGS_PAGINATION_COUNT_CAP + 50 - 1) // 50


@pytest.mark.asyncio
async def test_spend_logs_ui_empty_page_reports_zero_total(monkeypatch):
    """
    When nothing matches, the bounded count query returns a single row with a
    zero count (real `COUNT(*)` always returns one row) and the page query
    returns no rows, so the total is zero without an unbounded prisma `.count()`.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        ui_view_spend_logs,
    )

    # First query_raw call is the bounded count (0 matches), second is the empty
    # page.
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(side_effect=[[{"total_count": 0}], []])
    mock_prisma.db.litellm_spendlogs = MagicMock()
    mock_prisma.db.litellm_spendlogs.count = AsyncMock(return_value=0)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
    mock_request = MagicMock()
    mock_request.url.path = "/spend/logs/ui"

    response = await ui_view_spend_logs(
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

    mock_prisma.db.litellm_spendlogs.count.assert_not_called()
    assert response["total"] == 0
    assert response["total_pages"] == 0
    assert response["data"] == []


@pytest.mark.asyncio
async def test_spend_logs_ui_out_of_range_page_keeps_total(monkeypatch):
    """
    An out-of-range page (offset past the last matching row) returns no rows,
    but the bounded count query runs independently of the page query, so the
    total must not collapse to zero and no unbounded prisma `.count()` is
    needed. total/total_pages stay accurate off the hot path too.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        ui_view_spend_logs,
    )

    # First query_raw call is the bounded count (7 matches), second is the
    # out-of-range page (empty).
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(side_effect=[[{"total_count": 7}], []])
    mock_prisma.db.litellm_spendlogs = MagicMock()
    mock_prisma.db.litellm_spendlogs.count = AsyncMock(return_value=0)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin")
    mock_request = MagicMock()
    mock_request.url.path = "/spend/logs/ui"

    response = await ui_view_spend_logs(
        request=mock_request,
        api_key=None,
        user_id=None,
        request_id=None,
        start_date="2026-02-16 00:00:00",
        end_date="2026-02-16 23:59:59",
        page=99,
        page_size=2,
        sort_by="startTime",
        sort_order="desc",
        user_api_key_dict=auth,
    )

    mock_prisma.db.litellm_spendlogs.count.assert_not_called()
    assert response["total"] == 7
    assert response["total_pages"] == (7 + 2 - 1) // 2
    assert response["data"] == []


@pytest.mark.asyncio
async def test_get_spend_by_team_binds_optional_team_filter():
    """
    get_spend_by_team must bind team_id as query parameter $3 behind an
    `IS NULL OR` predicate: a provided team_id narrows the result to that team,
    a None team_id short-circuits the filter and returns every team. Regression
    guard for LIT-4125 (the team query previously carried no team_id predicate).
    """
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_query_raw = AsyncMock(return_value=[])
    mock_prisma.db.query_raw = mock_query_raw

    start_date = datetime.datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.datetime(2024, 1, 31, tzinfo=timezone.utc)

    await get_spend_by_team(
        start_date=start_date,
        end_date=end_date,
        team_id="test_team",
        prisma_client=mock_prisma,
    )

    assert mock_query_raw.called, "query_raw should have been called"
    sql = mock_query_raw.call_args[0][0]
    params = mock_query_raw.call_args[0][1:]

    # team_id is bound as parameter $3 (not string-interpolated) and referenced in WHERE
    assert "sl.team_id = $3" in sql, f"WHERE must filter on team_id. SQL was:\n{sql}"
    assert "$3::text IS NULL OR" in sql, (
        f"the team filter must be optional via an IS NULL short-circuit. SQL was:\n{sql}"
    )
    assert params[2] == "test_team", "team_id must be forwarded as the third query param"

    # None team_id still forwards param $3 (as None) so the predicate no-ops
    mock_query_raw.reset_mock()
    await get_spend_by_team(
        start_date=start_date,
        end_date=end_date,
        team_id=None,
        prisma_client=mock_prisma,
    )
    assert mock_query_raw.call_args[0][1:][2] is None


@pytest.mark.asyncio
async def test_global_spend_report_team_group_forwards_team_id(monkeypatch):
    """
    GET /global/spend/report?group_by=team&team_id=X must filter to team X.

    Before LIT-4125 the group_by=team branch ran a query with no team_id
    predicate, so spend for every team in the range was returned regardless of
    team_id (team_id was only honored when a customer_id was also supplied).
    This asserts the endpoint forwards team_id into the DB query.
    """
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        get_global_spend_report,
    )

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.query_raw = AsyncMock(return_value=[])

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)

    await get_global_spend_report(
        start_date="2026-07-01",
        end_date="2026-07-03",
        group_by="team",
        api_key=None,
        internal_user_id=None,
        team_id="team_x",
        customer_id=None,
    )

    assert mock_prisma.db.query_raw.called, "query_raw should have been called"
    sql = mock_prisma.db.query_raw.call_args[0][0]
    params = mock_prisma.db.query_raw.call_args[0][1:]
    assert "team_x" in params, "team_id must be forwarded into the DB query params"
    assert "sl.team_id = $3" in sql, f"team query must filter on team_id. SQL was:\n{sql}"
