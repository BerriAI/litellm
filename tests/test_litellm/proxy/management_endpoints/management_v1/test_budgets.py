from dataclasses import replace
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
from litellm.proxy.management_endpoints.management_v1.budgets import (
    BUDGETS_LIST_SPEC,
    BudgetListItem,
)
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    problem_response,
)
from litellm.proxy.management_endpoints.management_v1.list_framework import (
    Compare,
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
        "created_at": "2026-07-20T12:00:00+00:00",
        "updated_at": "2026-07-21T12:00:00+00:00",
        **overrides,
    }


@pytest.fixture
def query_raw(monkeypatch):
    """Mocks the one call the executor makes. `count` reads the first result, `find_many` the second."""
    mock = AsyncMock(side_effect=[[{"count": 0}], []])
    prisma_client = MagicMock()
    prisma_client.db.query_raw = mock
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    return mock


def _serve(query_raw, rows: list[dict[str, Any]], total: int | None = None) -> None:
    query_raw.side_effect = [[{"count": len(rows) if total is None else total}], rows]


@pytest.fixture
def as_proxy_admin():
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    yield
    app.dependency_overrides.clear()


def _as_role(role: LitellmUserRoles):
    original = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="u", user_role=role)
    return original


def _get(query: str = ""):
    suffix = f"?{query}" if query else ""
    return client.get(f"{BUDGETS_PATH}{suffix}", headers={"Authorization": "Bearer k"})


def _select_call(query_raw):
    """The find_many call: (sql, *params). The count call comes first."""
    return query_raw.call_args_list[1].args


def test_returns_flat_rows_in_the_control_plane_envelope(query_raw, as_proxy_admin):
    """`{data, meta, links}` with flat rows; no JSON:API `{type, id, attributes}` wrapper."""
    _serve(query_raw, [_row("b-1")])

    response = _get()

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"data", "meta", "links"}
    assert body["data"][0]["budget_id"] == "b-1"
    assert "attributes" not in body["data"][0]


def test_serves_the_columns_the_budgets_page_renders(query_raw, as_proxy_admin):
    _serve(query_raw, [_row("b-1", soft_budget=5.0, budget_reset_at="2026-08-01T00:00:00+00:00")])

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


def test_selects_only_the_columns_it_serves(query_raw, as_proxy_admin):
    """A `SELECT *` would ship created_by/updated_by and model_max_budget to the browser."""
    _serve(query_raw, [])

    _get()

    sql = _select_call(query_raw)[0]
    assert "SELECT *" not in sql
    assert '"budget_id"' in sql and '"max_budget"' in sql
    assert "created_by" not in sql and "model_max_budget" not in sql


def test_defaults_to_newest_first_with_budget_id_breaking_ties(query_raw, as_proxy_admin):
    """Two budgets created in the same transaction share a created_at; without the
    tiebreaker their relative order is undefined and pages can repeat or drop rows."""
    _serve(query_raw, [])

    _get()

    assert 'ORDER BY "created_at" DESC NULLS LAST, "budget_id" ASC NULLS LAST' in _select_call(query_raw)[0]


def test_appends_the_tiebreaker_to_an_explicit_sort(query_raw, as_proxy_admin):
    _serve(query_raw, [])

    _get("sort=-max_budget")

    assert 'ORDER BY "max_budget" DESC NULLS LAST, "budget_id" ASC NULLS LAST' in _select_call(query_raw)[0]


def test_refuses_to_sort_on_budget_duration(query_raw, as_proxy_admin):
    """The column holds "7d"/"30d", so a lexicographic ORDER BY would put "30d"
    before "7d" and silently mis-order the page."""
    _serve(query_raw, [])

    response = _get("sort=budget_duration")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert "budget_duration" in body["detail"]
    assert body["allowed"] == SORTABLE
    query_raw.assert_not_called()


def test_the_advertised_sort_fields_are_the_ones_that_work(query_raw, as_proxy_admin):
    """Guards the rejection above against drifting from what the spec actually accepts."""
    for field in SORTABLE:
        _serve(query_raw, [])
        assert _get(f"sort={field}").status_code == 200, field
    assert sorted(BUDGETS_LIST_SPEC.sortable) == SORTABLE


def test_rejects_an_unknown_query_parameter(query_raw, as_proxy_admin):
    """A silently ignored filter over-returns budgets, which is worse than a rejected request."""
    _serve(query_raw, [])

    response = _get("filtre[max_budget][gte]=5")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert "filtre[max_budget][gte]" in body["detail"]
    assert "filter[max_budget][gte]" in body["allowed"]
    query_raw.assert_not_called()


def test_rejects_an_operator_the_filter_does_not_declare(query_raw, as_proxy_admin):
    """`max_budget` takes ranges, not `in`; accepting an undeclared operator is how a
    filter starts meaning something the query planner never checked."""
    for query in ("filter[max_budget][in]=5,10", "filter[created_at][is_null]=true"):
        _serve(query_raw, [])
        assert _get(query).status_code == 400, query


def test_omitted_page_size_serves_fifty(query_raw, as_proxy_admin):
    _serve(query_raw, [])

    body = _get().json()

    assert body["meta"]["page_size"] == 50
    assert _select_call(query_raw)[-2] == 50


def test_clamps_an_oversized_page_size_to_a_hundred(query_raw, as_proxy_admin):
    """Unclamped, one request can ask the proxy to serialize the whole budget table."""
    _serve(query_raw, [])

    body = _get("page_size=500").json()

    assert body["meta"]["page_size"] == 100
    assert _select_call(query_raw)[-2] == 100


def test_offsets_by_page(query_raw, as_proxy_admin):
    _serve(query_raw, [])

    _get("page=3&page_size=25")

    assert _select_call(query_raw)[-2:] == (25, 50)


@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.TEAM,
    ],
)
def test_refuses_a_caller_without_admin_view(query_raw, role):
    """Budgets are proxy-wide, so a caller who cannot read all of them must be told
    so. Answering 200 with an empty list would read as "there are no budgets"."""
    _serve(query_raw, [_row("b-1")])
    original = _as_role(role)
    try:
        response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 403
    query_raw.assert_not_called()


@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_admins_and_admin_viewers_may_read_every_budget(query_raw, role):
    _serve(query_raw, [_row("b-1")])
    original = _as_role(role)
    try:
        response = _get()
    finally:
        app.dependency_overrides = original

    assert response.status_code == 200
    assert [row["budget_id"] for row in response.json()["data"]] == ["b-1"]
    assert "WHERE" not in _select_call(query_raw)[0]


def test_a_denied_caller_stays_denied_whatever_they_filter_on(query_raw):
    """The scope decision reads the caller, never the query string."""
    _serve(query_raw, [_row("b-1")])
    original = _as_role(LitellmUserRoles.INTERNAL_USER)
    try:
        response = _get("filter[max_budget][gte]=0&q=b-")
    finally:
        app.dependency_overrides = original

    assert response.status_code == 403


def test_a_filter_sits_behind_the_scope_predicate_instead_of_replacing_it():
    """Same spec, planned for a row-scoped caller: the scope clause has to survive, and
    lead, whatever the caller filtered on. Replacing it would let a filter widen a read."""
    scoped = replace(
        BUDGETS_LIST_SPEC,
        scope=lambda _caller: ScopeWhere(where=(Compare(field="budget_id", op="eq", value="b-1"),)),
    )

    plan = build_query_plan(
        spec=scoped,
        params={"filter[max_budget][gte]": "5"},
        caller=UserAPIKeyAuth(user_id="u", user_role=LitellmUserRoles.INTERNAL_USER),
    )

    assert plan.where[0] == Compare(field="budget_id", op="eq", value="b-1")
    assert Compare(field="max_budget", op="gte", value=5.0) in plan.where


def test_q_matches_budget_id_case_insensitively(query_raw, as_proxy_admin):
    """budget_id is the only text identity on the row; matching anything else would
    return budgets whose ids do not contain what the user typed."""
    _serve(query_raw, [])

    _get("q=Prod")

    sql, *params = _select_call(query_raw)
    assert '"budget_id" ILIKE $1' in sql
    assert params[0] == "%Prod%"


def test_q_escapes_like_metacharacters(query_raw, as_proxy_admin):
    """Budget ids routinely contain '_'; unescaped it is a single-character wildcard."""
    _serve(query_raw, [])

    _get("q=team_a%25")

    assert _select_call(query_raw)[1] == r"%team\_a\%%"


def test_q_does_not_search_any_other_column(query_raw, as_proxy_admin):
    _serve(query_raw, [])

    _get("q=30d")

    assert "budget_duration" not in _select_call(query_raw)[0].split("WHERE")[1]
    assert BUDGETS_LIST_SPEC.searchable == frozenset({"budget_id"})


def test_is_null_selects_the_unlimited_budgets(query_raw, as_proxy_admin):
    """"Unlimited" is max_budget IS NULL; `max_budget = 0` would be a hard zero cap."""
    _serve(query_raw, [_row("b-unlimited", max_budget=None)])

    body = _get("filter[max_budget][is_null]=true").json()

    assert '"max_budget" IS NULL' in _select_call(query_raw)[0]
    assert body["data"][0]["max_budget"] is None


def test_is_null_false_selects_the_capped_budgets(query_raw, as_proxy_admin):
    _serve(query_raw, [])

    _get("filter[max_budget][is_null]=false")

    assert '"max_budget" IS NOT NULL' in _select_call(query_raw)[0]


def test_in_filter_binds_each_requested_duration(query_raw, as_proxy_admin):
    _serve(query_raw, [])

    _get("filter[budget_duration][in]=7d,30d")

    sql, *params = _select_call(query_raw)
    assert '"budget_duration" IN ($1, $2)' in sql
    assert params[:2] == ["7d", "30d"]


def test_created_at_range_is_bound_as_a_timestamp(query_raw, as_proxy_admin):
    """The bind crosses into the query engine as JSON, so an uncast placeholder reaches
    Postgres as text and `timestamp >= text` is a hard error, not a wrong answer."""
    _serve(query_raw, [])

    _get("filter[created_at][gte]=2026-07-01T00:00:00Z")

    sql, *params = _select_call(query_raw)
    assert "\"created_at\" >= $1::timestamptz AT TIME ZONE 'UTC'" in sql
    assert params[0] == datetime(2026, 7, 1, tzinfo=timezone.utc)


def test_numbers_placeholders_continuously_across_predicates(query_raw, as_proxy_admin):
    """Each predicate is numbered after the binds the ones before it consumed. Restart
    the count and `$1` gets read as the duration while the search string goes unbound."""
    _serve(query_raw, [])

    _get("filter[budget_duration][in]=7d,30d&filter[max_budget][gte]=5&q=prod")

    sql, *params = _select_call(query_raw)
    assert '"budget_duration" IN ($1, $2)' in sql
    assert '"max_budget" >= $3' in sql
    assert '"budget_id" ILIKE $4' in sql
    assert params[:4] == ["7d", "30d", 5.0, "%prod%"]


def test_a_non_datetime_bind_is_not_cast(query_raw, as_proxy_admin):
    """Guards the cast above from being applied to every placeholder."""
    _serve(query_raw, [])

    _get("filter[max_budget][gte]=5")

    sql = _select_call(query_raw)[0]
    assert '"max_budget" >= $1' in sql
    assert "timestamptz" not in sql


def test_an_offsetless_created_at_bound_is_read_as_utc(query_raw, as_proxy_admin):
    """The dashboard sends 'YYYY-MM-DDTHH:MM:SS' with no offset. Left naive, Postgres
    would compare it in the session timezone and shift the window off the rows shown."""
    _serve(query_raw, [])

    _get("filter[created_at][gte]=2026-07-01T00:00:00")

    bound = _select_call(query_raw)[1]
    assert bound == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert bound.tzinfo is not None


def test_rejects_a_filter_value_that_is_not_of_the_declared_type(query_raw, as_proxy_admin):
    for query in ("filter[max_budget][gte]=lots", "filter[created_at][gte]=yesterday"):
        _serve(query_raw, [])
        assert _get(query).status_code == 400, query


def test_reports_the_total_and_links_every_page_on_a_middle_page(query_raw, as_proxy_admin):
    """The Budgets page renders a page count, so the total has to be the match total,
    not the length of the page it just received."""
    _serve(query_raw, [_row("b-3"), _row("b-4")], total=7)

    body = _get("page=2&page_size=2").json()

    assert body["meta"] == {"page": 2, "page_size": 2, "total_count": 7, "total_pages": 4}
    links = body["links"]
    assert "page=1" in links["first"] and "page_size=2" in links["first"]
    assert "page=1" in links["prev"]
    assert "page=3" in links["next"]
    assert "page=4" in links["last"]
    assert "page=2" in links["self"]


def test_the_last_page_has_no_next(query_raw, as_proxy_admin):
    _serve(query_raw, [_row("b-5")], total=5)

    links = _get("page=3&page_size=2").json()["links"]

    assert links["next"] is None
    assert "page=2" in links["prev"]


def test_the_first_page_has_no_prev(query_raw, as_proxy_admin):
    _serve(query_raw, [_row("b-1")], total=5)

    links = _get("page_size=2").json()["links"]

    assert links["prev"] is None
    assert "page=2" in links["next"]


def test_an_empty_table_still_links_a_first_and_last_page(query_raw, as_proxy_admin):
    _serve(query_raw, [], total=0)

    body = _get().json()

    assert body["meta"]["total_count"] == 0
    assert body["meta"]["total_pages"] == 0
    assert "page=1" in body["links"]["first"] and "page=1" in body["links"]["last"]


def test_counts_over_the_same_predicate_it_pages(query_raw, as_proxy_admin):
    """A total counted without the caller's filter would page through rows the
    filter excluded."""
    _serve(query_raw, [], total=0)

    _get("filter[budget_duration][in]=30d")

    count_sql, *count_params = query_raw.call_args_list[0].args
    assert '"budget_duration" IN ($1)' in count_sql
    assert count_params == ["30d"]
    assert "COUNT(*)" in count_sql


def test_bigint_limits_serialize_as_json_numbers(query_raw, as_proxy_admin):
    """tpm_limit/rpm_limit are BigInt? in Prisma; the query engine can hand them back
    as decimal strings, and a quoted "60000" breaks arithmetic in the dashboard."""
    _serve(query_raw, [_row("b-1", tpm_limit="60000", rpm_limit=1200)])

    response = _get()
    row = response.json()["data"][0]

    assert row["tpm_limit"] == 60000
    assert row["rpm_limit"] == 1200
    assert isinstance(row["tpm_limit"], int) and not isinstance(row["tpm_limit"], bool)
    assert '"tpm_limit": "60000"' not in response.text


def test_reports_a_missing_database_as_a_problem_document(monkeypatch, as_proxy_admin):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    response = _get()

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")


def test_the_spec_serves_what_the_row_model_declares():
    """SELECTED_COLUMNS is built off the model, so a field added to one cannot go
    missing from the other and produce a row the validator rejects."""
    from litellm.proxy.management_endpoints.management_v1.budgets import SELECTED_COLUMNS

    assert SELECTED_COLUMNS == ", ".join(f'"{name}"' for name in BudgetListItem.model_fields)
    assert BUDGETS_LIST_SPEC.tiebreaker in BudgetListItem.model_fields
    assert BUDGETS_LIST_SPEC.sortable <= frozenset(BudgetListItem.model_fields)
    assert frozenset(BUDGETS_LIST_SPEC.filters) <= frozenset(BudgetListItem.model_fields)


def test_is_reachable_by_the_roles_that_can_open_the_budgets_page():
    """Route-level auth gate, which the dependency_overrides above bypass. The handler's
    admin-view check is dead code if RouteChecks rejects the role first."""
    assert BUDGETS_PATH in LiteLLMRoutes.admin_viewer_routes.value
    assert ("/budget/list" in LiteLLMRoutes.admin_viewer_routes.value) == (
        BUDGETS_PATH in LiteLLMRoutes.admin_viewer_routes.value
    )
