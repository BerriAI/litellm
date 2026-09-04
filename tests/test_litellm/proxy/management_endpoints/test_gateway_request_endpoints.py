import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# Patching ``litellm.proxy.proxy_server.prisma_client`` imports that module, whose
# module-level setup reads DATABASE_URL and LITELLM_MASTER_KEY. Tier-zero runners
# set neither, so pin throwaways first, as test_component_allowlists.py does. The
# prior values are restored below so a non-postgres URL cannot leak into sibling
# tests sharing the xdist worker and make them treat a phantom database as live.
_THROWAWAY_ENV = {
    "DATABASE_URL": "sqlite:///:memory:",
    "LITELLM_MASTER_KEY": "sk-test-gateway-request-endpoints",
}
_PRE_EXISTING_ENV = {key: os.environ.get(key) for key in _THROWAWAY_ENV}
for _key, _value in _THROWAWAY_ENV.items():
    os.environ.setdefault(_key, _value)

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.gateway_request_endpoints import (
    _AggregateRow,
    _default_range,
    _fold_by_date,
    _fold_by_route,
    get_gateway_daily_activity,
    router,
)

for _key, _previous in _PRE_EXISTING_ENV.items():
    if _previous is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _previous

# The handler stamps "today" from the wall clock, so any assertion that names a
# date has to pin it. Recomputing the expected range in the assertion instead
# would disagree with the request's own range whenever a run crosses UTC
# midnight between the two evaluations.
# A date in the past on purpose. Pinning "today" would let these assertions pass
# on a day the fixture silently failed to patch, which is the same vacuous pass a
# mutation check exists to catch.
_FROZEN_NOW = datetime(2023, 3, 15, 12, 0, tzinfo=timezone.utc)
_FROZEN_RANGE = ("2023-02-13", "2023-03-15")


@pytest.fixture
def frozen_clock():
    with patch("litellm.proxy.management_endpoints.gateway_request_endpoints.datetime") as clock:
        clock.now.return_value = _FROZEN_NOW
        yield


def _row(
    date: str = "2026-08-04",
    category: str = "llm",
    route: str = "/chat/completions",
    successful: int = 0,
    failed: int = 0,
) -> _AggregateRow:
    return _AggregateRow(
        date=date,
        category=category,
        route=route,
        successful_requests=successful,
        failed_requests=failed,
    )


def _admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.PROXY_ADMIN)


def _prisma_returning(rows: list) -> MagicMock:
    client = MagicMock()
    client.db = MagicMock()
    client.db.query_raw = AsyncMock(return_value=rows)
    return client


class TestDefaultRange:
    def test_spans_the_documented_lookback(self):
        start, end = _default_range()
        span = datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")
        assert span == timedelta(days=30)

    def test_ends_today_in_utc(self, frozen_clock):
        assert _default_range() == _FROZEN_RANGE


class TestFoldByDate:
    def test_sums_every_route_into_one_entry_per_date(self):
        folded = _fold_by_date(
            (
                _row(date="2026-08-03", route="/chat/completions", successful=5, failed=1),
                _row(date="2026-08-03", route="/embeddings", successful=2, failed=0),
                _row(date="2026-08-04", route="/chat/completions", successful=7, failed=3),
            )
        )
        assert [(entry.date, entry.successful_requests, entry.failed_requests) for entry in folded] == [
            ("2026-08-03", 7, 1),
            ("2026-08-04", 7, 3),
        ]

    def test_orders_oldest_first_regardless_of_row_order(self):
        rows = (_row(date="2026-08-09"), _row(date="2026-08-01"), _row(date="2026-08-05"))
        assert [entry.date for entry in _fold_by_date(rows)] == ["2026-08-01", "2026-08-05", "2026-08-09"]
        assert [entry.date for entry in _fold_by_date(tuple(reversed(rows)))] == [
            "2026-08-01",
            "2026-08-05",
            "2026-08-09",
        ]

    def test_no_rows_yields_no_entries(self):
        assert _fold_by_date(()) == ()


class TestFoldByRoute:
    def test_sums_across_dates_for_one_route(self):
        folded = _fold_by_route(
            (
                _row(date="2026-08-03", route="/chat/completions", successful=5, failed=1),
                _row(date="2026-08-04", route="/chat/completions", successful=7, failed=3),
            )
        )
        assert len(folded) == 1
        assert (folded[0].route, folded[0].successful_requests, folded[0].failed_requests) == (
            "/chat/completions",
            12,
            4,
        )

    def test_keeps_same_route_under_different_categories_apart(self):
        folded = _fold_by_route(
            (
                _row(category="mcp", route="/tools/call", successful=2),
                _row(category="a2a", route="/tools/call", successful=1),
            )
        )
        assert {(entry.category, entry.successful_requests) for entry in folded} == {("mcp", 2), ("a2a", 1)}

    def test_orders_busiest_route_first_whatever_the_row_order(self):
        rows = (
            _row(route="/embeddings", successful=4),
            _row(route="/chat/completions", successful=11),
            _row(route="/rerank", successful=7),
        )
        expected = ["/chat/completions", "/rerank", "/embeddings"]
        assert [entry.route for entry in _fold_by_route(rows)] == expected
        assert [entry.route for entry in _fold_by_route(tuple(reversed(rows)))] == expected


class TestGatewayDailyActivityEndpoint:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "role",
        [
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            LitellmUserRoles.TEAM,
            LitellmUserRoles.ORG_ADMIN,
        ],
    )
    async def test_refuses_every_non_admin_role(self, role):
        with patch("litellm.proxy.proxy_server.prisma_client", _prisma_returning([])):
            with pytest.raises(HTTPException) as exc:
                await get_gateway_daily_activity(
                    user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_role=role),
                )
        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "role",
        [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
    )
    async def test_serves_both_admin_roles(self, role):
        with patch("litellm.proxy.proxy_server.prisma_client", _prisma_returning([])):
            response = await get_gateway_daily_activity(
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_role=role),
            )
        assert response.total_successful_requests == 0

    @pytest.mark.asyncio
    async def test_reports_db_not_connected_rather_than_crashing(self):
        with patch("litellm.proxy.proxy_server.prisma_client", None):
            with pytest.raises(HTTPException) as exc:
                await get_gateway_daily_activity(user_api_key_dict=_admin())
        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_totals_and_breakdowns_come_from_the_same_rows(self):
        rows = [
            {
                "date": "2026-08-03",
                "category": "llm",
                "route": "/chat/completions",
                "successful_requests": 5,
                "failed_requests": 1,
            },
            {
                "date": "2026-08-04",
                "category": "llm",
                "route": "/chat/completions",
                "successful_requests": 7,
                "failed_requests": 3,
            },
            {
                "date": "2026-08-04",
                "category": "llm",
                "route": "/embeddings",
                "successful_requests": 4,
                "failed_requests": 0,
            },
        ]
        with patch("litellm.proxy.proxy_server.prisma_client", _prisma_returning(rows)):
            response = await get_gateway_daily_activity(user_api_key_dict=_admin())

        assert response.total_successful_requests == 16
        assert response.total_failed_requests == 4
        assert sum(entry.successful_requests for entry in response.by_date) == 16
        assert sum(entry.successful_requests for entry in response.by_route) == 16
        assert [entry.date for entry in response.by_date] == ["2026-08-03", "2026-08-04"]
        assert [entry.route for entry in response.by_route] == ["/chat/completions", "/embeddings"]

    @pytest.mark.asyncio
    async def test_a_null_result_set_is_not_an_error(self):
        client = _prisma_returning(None)
        with patch("litellm.proxy.proxy_server.prisma_client", client):
            response = await get_gateway_daily_activity(user_api_key_dict=_admin())
        assert response.total_successful_requests == 0
        assert response.by_date == ()
        assert response.by_route == ()

class TestGatewayDailyActivityRoute:
    """
    Driven through the mounted route rather than by calling the handler.

    The date parameters carry FastAPI ``Query`` defaults, which only resolve to
    None when the framework builds the call; invoking the handler directly hands
    it the Query object instead, so a direct call cannot check what an omitted
    date does.
    """

    def test_caller_dates_are_passed_through_verbatim(self):
        prisma = _prisma_returning([])
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = _admin
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            response = TestClient(app).get(
                "/gateway/daily/activity",
                params={"start_date": "2026-01-01", "end_date": "2026-01-31"},
            )
        assert response.status_code == 200
        _, start, end = prisma.db.query_raw.call_args.args
        assert (start, end) == ("2026-01-01", "2026-01-31")

    def test_omitted_dates_fall_back_to_the_default_window(self, frozen_clock):
        prisma = _prisma_returning([])
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = _admin
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            response = TestClient(app).get("/gateway/daily/activity")
        assert response.status_code == 200
        _, start, end = prisma.db.query_raw.call_args.args
        assert (start, end) == _FROZEN_RANGE

    def test_serialized_response_carries_the_documented_shape(self):
        prisma = _prisma_returning(
            [
                {
                    "date": "2026-08-04",
                    "category": "llm",
                    "route": "/chat/completions",
                    "successful_requests": 7,
                    "failed_requests": 3,
                }
            ]
        )
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = _admin
        with patch("litellm.proxy.proxy_server.prisma_client", prisma):
            body = TestClient(app).get("/gateway/daily/activity").json()

        assert body == {
            "total_successful_requests": 7,
            "total_failed_requests": 3,
            "by_date": [{"date": "2026-08-04", "successful_requests": 7, "failed_requests": 3}],
            "by_route": [
                {
                    "category": "llm",
                    "route": "/chat/completions",
                    "successful_requests": 7,
                    "failed_requests": 3,
                }
            ],
        }
