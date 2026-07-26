from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLMRoutes, LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    problem_response,
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail

app = FastAPI()


@app.exception_handler(ManagementProblem)
async def management_problem_exception_handler(request: Request, exc: ManagementProblem):
    return problem_response(exc.problem)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return problem_response(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}invalid-query-parameter",
            title="Invalid query parameter",
            status=400,
            detail="; ".join(
                f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}" for error in exc.errors()
            )
            or "The request query parameters are invalid.",
        )
    )


app.include_router(router)
client = TestClient(app)

END_USERS_PATH = f"{MANAGEMENT_V1_PREFIX}/spend_logs/end_users"
WINDOW = "filter[startTime][gte]=2026-07-23T00:00:00Z&filter[startTime][lte]=2026-07-24T00:00:00Z"


@pytest.fixture
def mock_prisma_client(monkeypatch):
    prisma_client = MagicMock()
    prisma_client.db.query_raw = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    return prisma_client


@pytest.fixture
def as_proxy_admin():
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    yield
    app.dependency_overrides.clear()


def _mock_rows(mock_prisma_client, end_users: List[str]) -> AsyncMock:
    query_raw = AsyncMock(return_value=[{"end_user": eu} for eu in end_users])
    mock_prisma_client.db.query_raw = query_raw
    return query_raw


def _as_role(role: LitellmUserRoles, user_id):
    original = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id=user_id, user_role=role)
    return original


def _get(query: str = WINDOW):
    suffix = f"?{query}" if query else ""
    return client.get(f"{END_USERS_PATH}{suffix}", headers={"Authorization": "Bearer k"})


def test_returns_the_control_plane_envelope(mock_prisma_client, as_proxy_admin):
    """`{data, meta, links}` is the contract; a bare list or a legacy `aliases` key is not."""
    _mock_rows(mock_prisma_client, ["a", "b"])

    response = _get()

    assert response.status_code == 200
    body = response.json()
    assert body["data"] == ["a", "b"]
    assert body["meta"] == {"page": 1, "page_size": 50, "has_more": False}
    assert set(body) == {"data", "meta", "links"}
    assert "aliases" not in body
    assert "total_count" not in body["meta"]


def test_links_let_a_client_page_without_building_urls(mock_prisma_client, as_proxy_admin):
    """The UI follows links.next; if it is absent the client has to recompute page params,
    which is what makes a later switch to cursor pagination a breaking change."""
    _mock_rows(mock_prisma_client, [f"u{i}" for i in range(4)])

    links = _get(f"{WINDOW}&page=2&page_size=3").json()["links"]

    assert links["self"].startswith(f"{END_USERS_PATH}?")
    assert "page=2" in links["self"]
    assert "page=1" in links["prev"] and "page_size=3" in links["prev"]
    assert "page=3" in links["next"] and "page_size=3" in links["next"]


def test_next_link_is_absent_on_the_last_page(mock_prisma_client, as_proxy_admin):
    _mock_rows(mock_prisma_client, ["u0", "u1"])

    body = _get(f"{WINDOW}&page_size=3").json()

    assert body["meta"]["has_more"] is False
    assert body["links"]["next"] is None
    assert body["links"]["prev"] is None


def test_reads_spend_logs_not_the_end_user_table(mock_prisma_client, as_proxy_admin):
    """Team scoping only exists in spend logs, so that is the source of truth."""
    query_raw = _mock_rows(mock_prisma_client, ["a"])

    _get()

    sql = query_raw.call_args.args[0]
    assert '"LiteLLM_SpendLogs"' in sql
    assert "LiteLLM_EndUserTable" not in sql


def test_caps_the_rows_it_scans(mock_prisma_client, as_proxy_admin):
    """The inner LIMIT is the crash guard: DISTINCT must never see an unbounded set."""
    from litellm.proxy.management_endpoints.management_v1.spend_logs import (
        SPEND_LOGS_FACET_SCAN_CAP,
    )

    query_raw = _mock_rows(mock_prisma_client, [])

    _get()

    sql = query_raw.call_args.args[0]
    inner = sql[sql.index("FROM (") : sql.index(") recent")]
    assert "LIMIT $3" in inner
    assert query_raw.call_args.args[3] == SPEND_LOGS_FACET_SCAN_CAP
    assert 'ORDER BY "startTime" DESC' in inner


def test_scan_cap_matches_the_logs_page_bound():
    """Pin the cap's value, not just that it is passed through.

    Asserting the param equals the constant is tautological: raising the constant
    to a billion keeps that assertion green while removing the bound entirely.
    """
    from litellm.proxy.management_endpoints.management_v1.spend_logs import (
        SPEND_LOGS_FACET_SCAN_CAP,
    )
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        SPEND_LOGS_PAGINATION_COUNT_CAP,
    )

    assert SPEND_LOGS_FACET_SCAN_CAP == SPEND_LOGS_PAGINATION_COUNT_CAP


def test_breaks_start_time_ties_deterministically(mock_prisma_client, as_proxy_admin):
    query_raw = _mock_rows(mock_prisma_client, [])

    _get()

    assert 'ORDER BY "startTime" DESC, request_id DESC' in query_raw.call_args.args[0]


def test_bounds_the_window_on_the_indexed_start_time(mock_prisma_client, as_proxy_admin):
    query_raw = _mock_rows(mock_prisma_client, [])

    _get()

    sql = query_raw.call_args.args[0]
    assert "\"startTime\" >= ($1::timestamptz AT TIME ZONE 'UTC')" in sql
    assert "\"startTime\" <= ($2::timestamptz AT TIME ZONE 'UTC')" in sql
    assert query_raw.call_args.args[1] == datetime(2026, 7, 23, tzinfo=timezone.utc)
    assert query_raw.call_args.args[2] == datetime(2026, 7, 24, tzinfo=timezone.utc)


def test_a_naive_window_bound_is_read_as_utc(mock_prisma_client, as_proxy_admin):
    """The dashboard sends 'YYYY-MM-DD HH:MM:SS' with no offset; reading it as
    server-local time would shift the window off what the logs table is showing."""
    query_raw = _mock_rows(mock_prisma_client, [])

    _get("filter[startTime][gte]=2026-07-23 00:00:00&filter[startTime][lte]=2026-07-24 00:00:00")

    assert query_raw.call_args.args[1] == datetime(2026, 7, 23, tzinfo=timezone.utc)
    assert query_raw.call_args.args[2] == datetime(2026, 7, 24, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "query",
    ["", "filter[startTime][gte]=2026-07-23T00:00:00Z"],
    ids=["no-window", "half-window"],
)
def test_requires_a_time_window(mock_prisma_client, as_proxy_admin, query):
    """No window means no index bound, which is the unbounded scan we must not allow."""
    _mock_rows(mock_prisma_client, [])

    response = _get(query)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")


def test_rejects_a_malformed_window_as_a_problem_document(mock_prisma_client, as_proxy_admin):
    _mock_rows(mock_prisma_client, [])

    response = _get(f"filter[startTime][gte]=yesterday&filter[startTime][lte]=2026-07-24T00:00:00Z")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["type"].startswith(PROBLEM_TYPE_BASE)
    assert body["status"] == 400
    assert body["title"] and body["detail"]
    assert "error" not in body


def test_rejects_an_unknown_query_parameter(mock_prisma_client, as_proxy_admin):
    """A silently ignored filter over-returns data, which is worse than a rejected request."""
    query_raw = _mock_rows(mock_prisma_client, [])

    response = _get(f"{WINDOW}&q_typo=acme")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert "q_typo" in body["detail"]
    assert "q" in body["allowed"]
    query_raw.assert_not_called()


def test_accepts_every_declared_parameter(mock_prisma_client, as_proxy_admin):
    """Guards the unknown-param check against rejecting the endpoint's own contract."""
    _mock_rows(mock_prisma_client, [])

    assert _get(f"{WINDOW}&q=acme&page=2&page_size=10").status_code == 200


def test_caps_page_size(mock_prisma_client, as_proxy_admin):
    _mock_rows(mock_prisma_client, [])

    assert _get(f"{WINDOW}&page_size=100000").status_code == 400


def test_applies_no_scope_for_a_proxy_admin(mock_prisma_client, as_proxy_admin):
    query_raw = _mock_rows(mock_prisma_client, [])

    _get()

    sql = query_raw.call_args.args[0]
    assert '"user" =' not in sql
    assert "team_id" not in sql


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_scopes_a_team_admin_to_their_own_rows_and_teams(mock_prisma_client, role):
    """A team admin must not see end users belonging to teams they cannot read."""
    query_raw = _mock_rows(mock_prisma_client, ["cust-a"])
    original = _as_role(role, user_id="team-admin-1")
    try:
        with patch(
            "litellm.proxy.spend_tracking.spend_management_endpoints._get_permitted_team_ids_for_spend_logs",
            new=AsyncMock(return_value=["team-a", "team-b"]),
        ):
            response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 200
    # Same clause shape ui_view_spend_logs builds, so the two cannot diverge.
    assert '("user" = $3 OR team_id = ANY($4::text[]))' in query_raw.call_args.args[0]
    assert query_raw.call_args.args[3] == "team-admin-1"
    assert query_raw.call_args.args[4] == ["team-a", "team-b"]


def test_scopes_a_teamless_user_to_their_own_rows(mock_prisma_client):
    query_raw = _mock_rows(mock_prisma_client, [])
    original = _as_role(LitellmUserRoles.INTERNAL_USER, user_id="solo")
    try:
        with patch(
            "litellm.proxy.spend_tracking.spend_management_endpoints._get_permitted_team_ids_for_spend_logs",
            new=AsyncMock(return_value=[]),
        ):
            response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 200
    sql = query_raw.call_args.args[0]
    assert '("user" = $3)' in sql
    assert "team_id" not in sql
    assert query_raw.call_args.args[3] == "solo"


def test_returns_nothing_when_the_caller_owns_no_scope(mock_prisma_client):
    """Unidentifiable caller must match no rows, never fall through to unscoped."""
    query_raw = _mock_rows(mock_prisma_client, [])
    original = _as_role(LitellmUserRoles.INTERNAL_USER, user_id=None)
    try:
        with patch(
            "litellm.proxy.spend_tracking.spend_management_endpoints._get_permitted_team_ids_for_spend_logs",
            new=AsyncMock(return_value=[]),
        ):
            response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 200
    assert "FALSE" in query_raw.call_args.args[0]


def test_scopes_when_the_permitted_team_lookup_fails(mock_prisma_client):
    """A failed team lookup must degrade to own-rows-only, never to unscoped."""
    query_raw = _mock_rows(mock_prisma_client, [])
    original = _as_role(LitellmUserRoles.INTERNAL_USER, user_id="solo")
    try:
        with patch(
            "litellm.proxy.spend_tracking.spend_management_endpoints._get_permitted_team_ids_for_spend_logs",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 200
    sql = query_raw.call_args.args[0]
    assert '("user" = $3)' in sql
    assert "team_id" not in sql


def test_fetches_one_extra_row_and_trims_it(mock_prisma_client, as_proxy_admin):
    query_raw = _mock_rows(mock_prisma_client, [f"u{i}" for i in range(4)])

    body = _get(f"{WINDOW}&page_size=3").json()

    assert body["data"] == ["u0", "u1", "u2"]
    assert body["meta"]["has_more"] is True
    assert query_raw.call_args.args[4:] == (4, 0)


def test_reports_no_more_pages_on_an_exactly_full_page(mock_prisma_client, as_proxy_admin):
    _mock_rows(mock_prisma_client, ["u0", "u1", "u2"])

    body = _get(f"{WINDOW}&page_size=3").json()

    assert body["data"] == ["u0", "u1", "u2"]
    assert body["meta"]["has_more"] is False


def test_offsets_by_page(mock_prisma_client, as_proxy_admin):
    query_raw = _mock_rows(mock_prisma_client, [])

    body = _get(f"{WINDOW}&page=3&page_size=25").json()

    assert body["meta"]["page"] == 3
    assert query_raw.call_args.args[4:] == (26, 50)


def test_q_escapes_like_metacharacters(mock_prisma_client, as_proxy_admin):
    """End-user ids routinely contain '_'; unescaped it is a wildcard."""
    query_raw = _mock_rows(mock_prisma_client, [])

    _get(f"{WINDOW}&q=device_id%25")

    assert "end_user ILIKE $3 ESCAPE" in query_raw.call_args.args[0]
    assert query_raw.call_args.args[3] == r"%device\_id\%%"


def test_q_placeholder_precedes_the_scan_limit_and_offset(mock_prisma_client, as_proxy_admin):
    query_raw = _mock_rows(mock_prisma_client, [])

    _get(f"{WINDOW}&q=acme&page_size=10")

    sql = query_raw.call_args.args[0]
    assert "LIMIT $4" in sql
    assert "LIMIT $5 OFFSET $6" in sql
    assert query_raw.call_args.args[3] == "%acme%"
    assert query_raw.call_args.args[5:] == (11, 0)


@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    ],
)
def test_is_reachable_by_every_role_that_can_open_the_logs_page(role):
    """Route-level auth gate, which the dependency_overrides in the other tests bypass.

    Handler-side team scoping is dead code if RouteChecks rejects the role first.
    """
    from litellm.proxy.auth.route_checks import RouteChecks

    for allowed in (
        LiteLLMRoutes.internal_user_routes.value,
        LiteLLMRoutes.internal_user_view_only_routes.value,
    ):
        assert ("/spend/logs/ui" in allowed) == (END_USERS_PATH in allowed)

    if role in (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY):
        allowed_routes = (
            LiteLLMRoutes.internal_user_routes.value
            if role == LitellmUserRoles.INTERNAL_USER
            else LiteLLMRoutes.internal_user_view_only_routes.value
        )
        assert RouteChecks.check_route_access(route=END_USERS_PATH, allowed_routes=allowed_routes)
    else:
        assert END_USERS_PATH in LiteLLMRoutes.admin_viewer_routes.value
