from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLMRoutes, LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.budgets import BUDGETS_LIST_SPEC
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    problem_response,
)
from litellm.proxy.management_endpoints.management_v1.list_framework import (
    ScopeWhere,
    build_query_plan,
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
            detail="The request query parameters are invalid.",
        )
    )


app.include_router(router)
client = TestClient(app)

BUDGETS_PATH = f"{MANAGEMENT_V1_PREFIX}/budgets"
SORTABLE = ["budget_id", "created_at", "max_budget", "rpm_limit", "tpm_limit"]


def _row(budget_id: str, **overrides: Any) -> dict[str, Any]:
    return {
        "budget_id": budget_id,
        "max_budget": 10.0,
        "soft_budget": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "budget_duration": "30d",
        "budget_reset_at": None,
        "created_at": datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
        **overrides,
    }


@pytest.fixture
def budget_table(monkeypatch):
    table = MagicMock()
    table.count = AsyncMock(return_value=0)
    table.find_many = AsyncMock(return_value=[])
    prisma_client = MagicMock()
    prisma_client.db.litellm_budgettable = table
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    return table


@pytest.fixture
def as_proxy_admin():
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    yield
    app.dependency_overrides.clear()


def _serve(budget_table, rows: list[dict[str, Any]], total: int | None = None) -> None:
    budget_table.find_many = AsyncMock(return_value=rows)
    budget_table.count = AsyncMock(return_value=len(rows) if total is None else total)


def _as_role(role: LitellmUserRoles):
    original = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="u", user_role=role)
    return original


def _get(query: str = ""):
    suffix = f"?{query}" if query else ""
    return client.get(f"{BUDGETS_PATH}{suffix}", headers={"Authorization": "Bearer k"})


def test_returns_flat_rows_in_the_control_plane_envelope(budget_table, as_proxy_admin):
    """`{data, meta, links}` with flat rows; no JSON:API `{type, id, attributes}` wrapper."""
    _serve(budget_table, [_row("b-1")])

    response = _get()

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data", "meta", "links"}
    assert body["data"][0]["budget_id"] == "b-1"
    assert "attributes" not in body["data"][0]


def test_serves_the_columns_the_budgets_page_renders(budget_table, as_proxy_admin):
    _serve(budget_table, [_row("b-1", soft_budget=5.0, budget_reset_at=datetime(2026, 8, 1, tzinfo=timezone.utc))])

    row = _get().json()["data"][0]

    assert set(row) == {
        "budget_id",
        "max_budget",
        "soft_budget",
        "tpm_limit",
        "rpm_limit",
        "budget_duration",
        "budget_reset_at",
        "created_at",
        "updated_at",
    }
    assert row["soft_budget"] == 5.0
    assert row["budget_reset_at"].startswith("2026-08-01T00:00:00")


def test_defaults_to_newest_first_with_budget_id_breaking_ties(budget_table, as_proxy_admin):
    """Two budgets created in the same transaction share a created_at; without the
    tiebreaker their relative order is undefined and pages can repeat or drop rows."""
    _serve(budget_table, [])

    _get()

    assert budget_table.find_many.call_args.kwargs["order"] == [
        {"created_at": "desc"},
        {"budget_id": "asc"},
    ]


def test_appends_the_tiebreaker_to_an_explicit_sort(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    _get("sort=-max_budget")

    assert budget_table.find_many.call_args.kwargs["order"] == [
        {"max_budget": "desc"},
        {"budget_id": "asc"},
    ]


def test_does_not_duplicate_the_tiebreaker_when_it_is_sorted_on(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    _get("sort=-budget_id")

    assert budget_table.find_many.call_args.kwargs["order"] == [{"budget_id": "desc"}]


def test_refuses_to_sort_on_budget_duration(budget_table, as_proxy_admin):
    """The column holds "7d"/"30d", so a lexicographic ORDER BY would put "30d"
    before "7d" and silently mis-order the page."""
    _serve(budget_table, [])

    response = _get("sort=budget_duration")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert "budget_duration" in body["detail"]
    assert body["allowed"] == SORTABLE
    budget_table.find_many.assert_not_called()


def test_the_advertised_sort_fields_are_the_ones_that_work(budget_table, as_proxy_admin):
    """Guards the rejection above against drifting from what the spec actually accepts."""
    _serve(budget_table, [])

    for field in SORTABLE:
        assert _get(f"sort={field}").status_code == 200, field
    assert sorted(BUDGETS_LIST_SPEC.sortable) == SORTABLE


def test_rejects_an_unknown_query_parameter(budget_table, as_proxy_admin):
    """A silently ignored filter over-returns budgets, which is worse than a rejected request."""
    _serve(budget_table, [])

    response = _get("filtre[max_budget][gte]=5")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert "filtre[max_budget][gte]" in body["detail"]
    assert "filter[max_budget][gte]" in body["allowed"]
    budget_table.count.assert_not_called()
    budget_table.find_many.assert_not_called()


def test_rejects_an_operator_the_filter_does_not_declare(budget_table, as_proxy_admin):
    """`max_budget` takes ranges, not `in`; accepting an undeclared operator is how a
    filter starts meaning something the query planner never checked."""
    _serve(budget_table, [])

    assert _get("filter[max_budget][in]=5,10").status_code == 400
    assert _get("filter[created_at][is_null]=true").status_code == 400


def test_omitted_page_size_serves_fifty(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    body = _get().json()

    assert body["meta"]["page_size"] == 50
    assert budget_table.find_many.call_args.kwargs["take"] == 50


def test_clamps_an_oversized_page_size_to_a_hundred(budget_table, as_proxy_admin):
    """Unclamped, one request can ask the proxy to serialize the whole budget table."""
    _serve(budget_table, [])

    body = _get("page_size=500").json()

    assert body["meta"]["page_size"] == 100
    assert budget_table.find_many.call_args.kwargs["take"] == 100


def test_offsets_by_page(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    _get("page=3&page_size=25")

    assert budget_table.find_many.call_args.kwargs["skip"] == 50
    assert budget_table.find_many.call_args.kwargs["take"] == 25


@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.TEAM,
    ],
)
def test_refuses_a_caller_without_admin_view(budget_table, role):
    """Budgets are proxy-wide, so a caller who cannot read all of them must be told
    so. Answering 200 with an empty list would read as "there are no budgets"."""
    _serve(budget_table, [_row("b-1")])
    original = _as_role(role)
    try:
        response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 403
    budget_table.count.assert_not_called()
    budget_table.find_many.assert_not_called()


@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_admins_and_admin_viewers_may_read_every_budget(budget_table, role):
    _serve(budget_table, [_row("b-1")])
    original = _as_role(role)
    try:
        response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 200
    assert [row["budget_id"] for row in response.json()["data"]] == ["b-1"]


def test_a_denied_caller_stays_denied_whatever_they_filter_on(budget_table):
    """The scope decision reads the caller, never the query string."""
    _serve(budget_table, [_row("b-1")])
    original = _as_role(LitellmUserRoles.INTERNAL_USER)
    try:
        response = _get("filter[max_budget][gte]=0&q=b-")
    finally:
        app.dependency_overrides = original

    assert response.status_code == 403


def test_a_filter_narrows_the_scope_predicate_instead_of_replacing_it(budget_table, as_proxy_admin):
    """A filter is ANDed in. Assigning it over the scope clause is what would let a
    caller widen their own read."""
    _serve(budget_table, [])

    _get("filter[max_budget][gte]=5")

    where = budget_table.find_many.call_args.kwargs["where"]
    assert {"max_budget": {"gte": 5.0}} in where["AND"]


def test_a_scoped_caller_keeps_their_scope_clause_alongside_their_filter():
    """Same spec, driven through the planner with a row-scoped caller: the scope
    clause has to survive next to whatever the caller filtered on."""
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": BUDGETS_PATH,
            "headers": [],
            "query_string": b"filter[max_budget][gte]=5",
        }
    )

    plan = build_query_plan(request, BUDGETS_LIST_SPEC, ScopeWhere(where={"budget_id": {"in": ("b-1",)}}))

    assert {"budget_id": {"in": ("b-1",)}} in plan.where["AND"]
    assert {"max_budget": {"gte": 5.0}} in plan.where["AND"]


def test_q_matches_budget_id_case_insensitively(budget_table, as_proxy_admin):
    """budget_id is the only text identity on the row; matching anything else would
    return budgets whose ids do not contain what the user typed."""
    _serve(budget_table, [])

    _get("q=Prod")

    where = budget_table.find_many.call_args.kwargs["where"]
    assert {"OR": ({"budget_id": {"contains": "Prod", "mode": "insensitive"}},)} in where["AND"]


def test_q_does_not_search_any_other_column(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    _get("q=30d")

    searched = budget_table.find_many.call_args.kwargs["where"]["AND"][0]["OR"]
    assert [next(iter(clause)) for clause in searched] == ["budget_id"]
    assert BUDGETS_LIST_SPEC.searchable == frozenset({"budget_id"})


def test_is_null_selects_the_unlimited_budgets(budget_table, as_proxy_admin):
    """"Unlimited" is max_budget IS NULL; `max_budget = 0` would be a hard zero cap."""
    _serve(budget_table, [_row("b-unlimited", max_budget=None)])

    body = _get("filter[max_budget][is_null]=true").json()

    assert {"max_budget": None} in budget_table.find_many.call_args.kwargs["where"]["AND"]
    assert body["data"][0]["max_budget"] is None


def test_is_null_false_selects_the_capped_budgets(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    _get("filter[max_budget][is_null]=false")

    assert {"max_budget": {"not": None}} in budget_table.find_many.call_args.kwargs["where"]["AND"]


def test_in_filter_splits_the_requested_durations(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    _get("filter[budget_duration][in]=7d,30d")

    assert {"budget_duration": {"in": ("7d", "30d")}} in budget_table.find_many.call_args.kwargs["where"]["AND"]


def test_created_at_range_is_read_as_a_timestamp(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    _get("filter[created_at][gte]=2026-07-01T00:00:00%2B00:00")

    assert {
        "created_at": {"gte": datetime(2026, 7, 1, tzinfo=timezone.utc)}
    } in budget_table.find_many.call_args.kwargs["where"]["AND"]


def test_rejects_a_filter_value_that_is_not_of_the_declared_type(budget_table, as_proxy_admin):
    _serve(budget_table, [])

    assert _get("filter[max_budget][gte]=lots").status_code == 400
    assert _get("filter[created_at][gte]=yesterday").status_code == 400


def test_reports_the_total_and_links_every_page_on_a_middle_page(budget_table, as_proxy_admin):
    """The Budgets page renders a page count, so the total has to be the match total,
    not the length of the page it just received."""
    _serve(budget_table, [_row("b-3"), _row("b-4")], total=7)

    body = _get("page=2&page_size=2").json()

    assert body["meta"] == {"page": 2, "page_size": 2, "total_count": 7, "total_pages": 4}
    links = body["links"]
    assert "page=1" in links["first"] and "page_size=2" in links["first"]
    assert "page=1" in links["prev"]
    assert "page=3" in links["next"]
    assert "page=4" in links["last"]
    assert "page=2" in links["self"]


def test_the_last_page_has_no_next(budget_table, as_proxy_admin):
    _serve(budget_table, [_row("b-5")], total=5)

    links = _get("page=3&page_size=2").json()["links"]

    assert links["next"] is None
    assert "page=2" in links["prev"]


def test_the_first_page_has_no_prev(budget_table, as_proxy_admin):
    _serve(budget_table, [_row("b-1")], total=5)

    links = _get("page_size=2").json()["links"]

    assert links["prev"] is None
    assert "page=2" in links["next"]


def test_an_empty_table_still_links_a_first_and_last_page(budget_table, as_proxy_admin):
    _serve(budget_table, [], total=0)

    body = _get().json()

    assert body["meta"]["total_count"] == 0
    assert body["meta"]["total_pages"] == 0
    assert "page=1" in body["links"]["first"] and "page=1" in body["links"]["last"]


def test_counts_over_the_same_predicate_it_pages(budget_table, as_proxy_admin):
    """A total counted without the caller's filter would page through rows the
    filter excluded."""
    _serve(budget_table, [], total=0)

    _get("filter[budget_duration][in]=30d")

    assert budget_table.count.call_args.kwargs["where"] == budget_table.find_many.call_args.kwargs["where"]


def test_bigint_limits_serialize_as_json_numbers(budget_table, as_proxy_admin):
    """tpm_limit/rpm_limit are BigInt? in Prisma; the query engine can hand them back
    as decimal strings, and a quoted "60000" breaks arithmetic in the dashboard."""
    _serve(budget_table, [_row("b-1", tpm_limit="60000", rpm_limit=1200)])

    row = _get().json()["data"][0]

    assert row["tpm_limit"] == 60000
    assert row["rpm_limit"] == 1200
    assert isinstance(row["tpm_limit"], int) and not isinstance(row["tpm_limit"], bool)
    assert '"tpm_limit": "60000"' not in _get().text


def test_reports_a_missing_database_as_a_problem_document(monkeypatch, as_proxy_admin):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    response = _get()

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")


def test_is_reachable_by_the_roles_that_can_open_the_budgets_page():
    """Route-level auth gate, which the dependency_overrides above bypass. The handler's
    admin-view check is dead code if RouteChecks rejects the role first."""
    assert BUDGETS_PATH in LiteLLMRoutes.admin_viewer_routes.value
    assert ("/budget/list" in LiteLLMRoutes.admin_viewer_routes.value) == (
        BUDGETS_PATH in LiteLLMRoutes.admin_viewer_routes.value
    )
