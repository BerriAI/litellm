from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import pytest
from fastapi import Request
from pydantic import BaseModel

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    build_page_links,
)
from litellm.proxy.management_endpoints.management_v1.list_framework import (
    AnyOf,
    Compare,
    FilterSpec,
    IsNull,
    ListSpec,
    QueryPlan,
    ScopeAll,
    ScopeDenied,
    ScopeWhere,
    SortKey,
    Within,
    build_query_plan,
    handle_list,
    order_by_sql,
    where_sql,
)
from litellm.types.proxy.management_endpoints.management_v1 import (
    PageLinks,
    PageMeta,
    ProblemDetail,
)

BUDGETS_PATH = f"{MANAGEMENT_V1_PREFIX}/budgets"
CALLER = UserAPIKeyAuth(user_id="caller-1")


@dataclass(frozen=True, slots=True)
class BudgetRow:
    budget_id: str
    max_budget: float | None
    created_by: str


class BudgetOut(BaseModel):
    budget_id: str
    max_budget: float | None


def _serialize(row: BudgetRow) -> BudgetOut:
    return BudgetOut(budget_id=row.budget_id, max_budget=row.max_budget)


def _spec(
    scope=lambda caller: ScopeAll(),
    searchable=frozenset({"budget_id", "created_by"}),
    sortable=frozenset({"max_budget", "created_at", "budget_id"}),
) -> ListSpec[BudgetRow, BudgetOut]:
    return ListSpec(
        resource="budgets",
        sortable=sortable,
        searchable=searchable,
        filters={
            "max_budget": FilterSpec(type=float, ops=frozenset({"eq", "gte", "lte", "is_null"})),
            "created_at": FilterSpec(type=datetime, ops=frozenset({"gte", "lte"})),
            "created_by": FilterSpec(type=str, ops=frozenset({"eq", "in", "contains"})),
            "tpm_limit": FilterSpec(type=int, ops=frozenset({"eq"})),
        },
        default_sort=(SortKey(field="created_at", descending=True),),
        default_page_size=25,
        max_page_size=100,
        scope=scope,
        serialize=_serialize,
        tiebreaker="budget_id",
    )


def _spec_with(**overrides) -> ListSpec[BudgetRow, BudgetOut]:
    """`replace` re-runs `__init__`, so the spec's own validation applies to the override."""
    return replace(_spec(), **overrides)


class RecordingExecutor:
    """In-memory stand-in for the Prisma-backed executor PR 2 supplies."""

    def __init__(self, rows: tuple[BudgetRow, ...], total_count: int | None = None) -> None:
        self.rows = rows
        self.total_count = len(rows) if total_count is None else total_count
        self.plan: QueryPlan | None = None
        self.count_where: tuple[object, ...] | None = None

    async def count(self, where: tuple[object, ...]) -> int:
        self.count_where = where
        return self.total_count

    async def find_many(self, plan: QueryPlan) -> Sequence[BudgetRow]:
        self.plan = plan
        return self.rows[plan.skip : plan.skip + plan.take]


def _request(query: str = "") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "root_path": "",
            "path": BUDGETS_PATH,
            "query_string": query.encode(),
            "headers": [(b"host", b"testserver")],
        }
    )


def _plan(query_params: Mapping[str, str], spec: ListSpec[BudgetRow, BudgetOut] | None = None) -> QueryPlan:
    result = build_query_plan(spec=spec or _spec(), params=query_params, caller=CALLER)
    assert isinstance(result, QueryPlan), result
    return result


def _problem(query_params: Mapping[str, str], spec: ListSpec[BudgetRow, BudgetOut] | None = None) -> ProblemDetail:
    result = build_query_plan(spec=spec or _spec(), params=query_params, caller=CALLER)
    assert isinstance(result, ProblemDetail), result
    return result


def _conjuncts(plan: QueryPlan) -> tuple[object, ...]:
    return plan.where


# ---------------------------------------------------------------- invariant 1


def test_appends_the_tiebreaker_to_the_default_sort():
    """Without a unique final key, ordering by an all-null column lets Postgres hand the
    same row back on two different pages."""
    assert _plan({}).order == (SortKey(field="created_at", descending=True), SortKey(field="budget_id", descending=False))


def test_appends_the_tiebreaker_to_an_explicit_multi_key_sort():
    order = _plan({"sort": "-max_budget,created_at"}).order

    assert len(order) == 3
    assert order[-1] == SortKey(field="budget_id", descending=False)


def test_appends_the_tiebreaker_even_when_the_caller_already_sorts_by_it():
    """Deduplicating it away is the tempting simplification, and it is the one that
    reintroduces a non-total order the moment the leading key stops being unique."""
    assert _plan({"sort": "-budget_id"}).order == (
        SortKey(field="budget_id", descending=True),
        SortKey(field="budget_id", descending=False),
    )


# ---------------------------------------------------------------- invariant 2


def test_orders_nulls_last_in_both_directions():
    """Postgres sorts nulls last ascending but first descending, so flipping the sort
    direction on max_budget would otherwise float every "Unlimited" row to the top."""
    sql = order_by_sql((SortKey(field="max_budget", descending=True), SortKey(field="budget_id", descending=False)))

    assert sql == '"max_budget" DESC NULLS LAST, "budget_id" ASC NULLS LAST'


def test_order_sql_covers_every_key_in_the_plan():
    sql = order_by_sql(_plan({"sort": "-max_budget,created_at"}).order)

    assert sql.count("NULLS LAST") == 3
    assert sql == '"max_budget" DESC NULLS LAST, "created_at" ASC NULLS LAST, "budget_id" ASC NULLS LAST'


# ------------------------------------------------------------ where rendering


def test_every_caller_value_is_bound_not_interpolated():
    """The one property that keeps a filter value from reaching the SQL text. A value that
    looks like SQL has to come back as a parameter, never as part of the statement."""
    sql, params = where_sql((Compare(field="budget_id", op="eq", value="'; DROP TABLE x --"),))

    assert sql == '"budget_id" = $1'
    assert params == ("'; DROP TABLE x --",)
    assert "DROP" not in sql


def test_placeholders_are_numbered_across_the_whole_plan():
    """A predicate that binds several values has to advance the counter by that many, or
    every later predicate reads the wrong parameter."""
    sql, params = where_sql(
        (
            Compare(field="created_by", op="eq", value="alice"),
            Within(field="budget_id", values=("a", "b", "c")),
            Compare(field="max_budget", op="gte", value=5.0),
        )
    )

    assert sql == '"created_by" = $1 AND "budget_id" IN ($2, $3, $4) AND "max_budget" >= $5'
    assert params == ("alice", "a", "b", "c", 5.0)


def test_placeholder_numbering_can_start_past_earlier_parameters():
    sql, params = where_sql((Compare(field="created_by", op="eq", value="alice"),), first_index=4)

    assert sql == '"created_by" = $4'
    assert params == ("alice",)


def test_is_null_binds_no_parameter_and_does_not_consume_a_placeholder():
    sql, params = where_sql(
        (IsNull(field="max_budget", negated=False), Compare(field="created_by", op="eq", value="alice"))
    )

    assert sql == '"max_budget" IS NULL AND "created_by" = $1'
    assert params == ("alice",)


def test_is_null_negated_renders_is_not_null():
    assert where_sql((IsNull(field="max_budget", negated=True),))[0] == '"max_budget" IS NOT NULL'


def test_a_search_renders_as_a_parenthesised_or():
    """Without the parentheses the OR would bind looser than the surrounding ANDs and the
    scope predicate would stop constraining the search branch."""
    sql, params = where_sql(
        (
            Compare(field="created_by", op="eq", value="alice"),
            AnyOf(
                clauses=(
                    Compare(field="budget_id", op="contains", value="prod"),
                    Compare(field="created_by", op="contains", value="prod"),
                )
            ),
        )
    )

    assert sql == (
        '"created_by" = $1 AND ('
        "\"budget_id\" ILIKE $2 ESCAPE '\\'"
        " OR "
        "\"created_by\" ILIKE $3 ESCAPE '\\'"
        ")"
    )
    assert params == ("alice", "%prod%", "%prod%")


def test_contains_escapes_like_metacharacters():
    """Budget ids routinely contain '_', which is a single-character wildcard unescaped."""
    _, params = where_sql((Compare(field="budget_id", op="contains", value="device_id%"),))

    assert params == (r"%device\_id\%%",)


@pytest.mark.parametrize(
    ("op", "operator"),
    [("eq", "="), ("not", "<>"), ("gte", ">="), ("lte", "<="), ("gt", ">"), ("lt", "<")],
)
def test_each_comparison_operator_renders_its_sql_spelling(op, operator):
    assert where_sql((Compare(field="max_budget", op=op, value=1),))[0] == f'"max_budget" {operator} $1'


def test_an_empty_plan_renders_no_where_body():
    assert where_sql(()) == ("", ())


def test_a_planned_filter_renders_end_to_end():
    """Ties the parser to the renderer: what build_query_plan produces is what executes."""
    sql, params = where_sql(_plan({"filter[max_budget][is_null]": "true", "q": "prod"}).where)

    assert sql == (
        '"max_budget" IS NULL AND ('
        "\"budget_id\" ILIKE $1 ESCAPE '\\'"
        " OR "
        "\"created_by\" ILIKE $2 ESCAPE '\\'"
        ")"
    )
    assert params == ("%prod%", "%prod%")


# ---------------------------------------------------------------- invariant 3


def test_the_scope_predicate_is_the_first_conjunct():
    spec = _spec(scope=lambda caller: ScopeWhere(where=(Compare(field="created_by", op="eq", value=caller.user_id),)))

    conjuncts = _conjuncts(_plan({"filter[max_budget][gte]": "5"}, spec=spec))

    assert conjuncts[0] == Compare(field="created_by", op="eq", value="caller-1")


def test_a_caller_filter_cannot_replace_the_scope_predicate():
    """The failure this guards is a `{**scope, **filters}` merge: a caller filtering on
    the scoped column would silently overwrite the scope and read another user's rows."""
    spec = _spec(scope=lambda caller: ScopeWhere(where=(Compare(field="created_by", op="eq", value=caller.user_id),)))

    conjuncts = _conjuncts(_plan({"filter[created_by][eq]": "someone-else"}, spec=spec))

    assert conjuncts[0] == Compare(field="created_by", op="eq", value="caller-1")
    assert Compare(field="created_by", op="eq", value="someone-else") in conjuncts
    assert len(conjuncts) == 2


def test_the_scope_predicate_survives_a_search():
    spec = _spec(scope=lambda caller: ScopeWhere(where=(Compare(field="created_by", op="eq", value=caller.user_id),)))

    conjuncts = _conjuncts(_plan({"q": "prod"}, spec=spec))

    assert conjuncts[0] == Compare(field="created_by", op="eq", value="caller-1")
    assert any(isinstance(conjunct, AnyOf) for conjunct in conjuncts)


def test_an_unscoped_caller_gets_no_scope_conjunct():
    assert _plan({"filter[max_budget][gte]": "5"}).where == (Compare(field="max_budget", op="gte", value=5.0),)


def test_an_unfiltered_unscoped_list_has_an_empty_where():
    assert _plan({}).where == ()


# ---------------------------------------------------------------- invariant 4


def test_a_denied_scope_is_a_403_problem():
    spec = _spec(scope=lambda caller: ScopeDenied(reason="Only a proxy admin can list budgets."))

    problem = _problem({}, spec=spec)

    assert problem.status == 403
    assert problem.type == f"{PROBLEM_TYPE_BASE}forbidden"
    assert problem.detail == "Only a proxy admin can list budgets."


@pytest.mark.asyncio
async def test_a_denied_scope_never_reaches_the_database():
    """A 200 with an empty list would tell the caller the resource is empty rather than
    that they cannot read it, and would still pay for the query."""
    spec = _spec(scope=lambda caller: ScopeDenied(reason="nope"))
    executor = RecordingExecutor(rows=(BudgetRow(budget_id="b1", max_budget=None, created_by="x"),))

    with pytest.raises(ManagementProblem) as raised:
        await handle_list(spec=spec, executor=executor, request=_request(), caller=CALLER)

    assert raised.value.problem.status == 403
    assert executor.plan is None
    assert executor.count_where is None


# ---------------------------------------------------------------- invariant 5


def test_page_size_falls_back_to_the_spec_default():
    assert _plan({}).take == 25


def test_page_size_is_clamped_to_the_spec_maximum():
    """Clamped rather than rejected: an over-large page is a UI bug, not a caller error,
    but serving it would let one request read the whole table."""
    assert _plan({"page_size": "100000"}).take == 100


def test_page_offsets_by_page_size():
    plan = _plan({"page": "3", "page_size": "10"})

    assert (plan.skip, plan.take) == (20, 10)


@pytest.mark.parametrize("page", ["0", "-1"], ids=["zero", "negative"])
def test_page_below_one_is_rejected(page):
    problem = _problem({"page": page})

    assert problem.status == 400
    assert problem.type == f"{PROBLEM_TYPE_BASE}invalid-query-parameter"


@pytest.mark.parametrize(
    "params",
    [{"page": "one"}, {"page_size": "many"}, {"page_size": "0"}],
    ids=["page-not-an-int", "page-size-not-an-int", "page-size-zero"],
)
def test_unusable_paging_values_are_rejected(params):
    assert _problem(params).status == 400


# ---------------------------------------------------------------- invariant 6


def test_an_unknown_query_parameter_is_rejected_with_the_allowed_set():
    problem = _problem({"page_sizee": "10"})

    assert problem.status == 400
    assert problem.type == f"{PROBLEM_TYPE_BASE}unknown-query-parameter"
    assert "page_sizee" in problem.detail
    assert problem.allowed is not None
    assert "page_size" in problem.allowed
    assert "filter[max_budget][gte]" in problem.allowed


def test_the_allowed_set_enumerates_only_operators_the_field_declares():
    problem = _problem({"nope": "1"})

    assert problem.allowed is not None
    assert "filter[max_budget][is_null]" in problem.allowed
    assert "filter[tpm_limit][gte]" not in problem.allowed
    assert "filter[created_by][in]" in problem.allowed


def test_a_filter_on_an_undeclared_field_is_an_unknown_parameter():
    problem = _problem({"filter[secret_column][eq]": "x"})

    assert problem.type == f"{PROBLEM_TYPE_BASE}unknown-query-parameter"
    assert "filter[secret_column][eq]" in problem.detail


def test_every_declared_parameter_is_accepted():
    """Guards the unknown-param check against rejecting the spec's own contract."""
    plan = _plan(
        {
            "page": "2",
            "page_size": "10",
            "sort": "-max_budget",
            "q": "prod",
            "filter[max_budget][gte]": "5",
            "filter[created_by][in]": "a,b",
        }
    )

    assert plan.take == 10


# ---------------------------------------------------------------- invariant 7


def test_sorting_by_an_undeclared_field_is_rejected():
    problem = _problem({"sort": "api_key"})

    assert problem.status == 400
    assert problem.type == f"{PROBLEM_TYPE_BASE}invalid-sort-field"
    assert problem.allowed == ["budget_id", "created_at", "max_budget"]
    assert "api_key" in problem.detail


def test_one_bad_key_rejects_the_whole_multi_key_sort():
    """Dropping the unknown key and sorting by the rest would silently return a
    differently-ordered page than the one asked for."""
    assert _problem({"sort": "-created_at,api_key"}).type == f"{PROBLEM_TYPE_BASE}invalid-sort-field"


def test_a_double_dash_prefix_is_not_a_descending_sort():
    assert _problem({"sort": "--created_at"}).type == f"{PROBLEM_TYPE_BASE}invalid-sort-field"


# ---------------------------------------------------------------- invariant 8


def test_an_operator_the_field_does_not_declare_is_rejected():
    problem = _problem({"filter[max_budget][contains]": "5"})

    assert problem.status == 400
    assert problem.type == f"{PROBLEM_TYPE_BASE}unsupported-filter-operator"
    assert problem.allowed == ["eq", "gte", "is_null", "lte"]
    assert "contains" in problem.detail


def test_the_same_operator_is_accepted_on_a_field_that_declares_it():
    """Pins the rejection to the field's own operator set rather than a global denylist."""
    conjuncts = _conjuncts(_plan({"filter[created_by][contains]": "ops"}))

    assert conjuncts == (Compare(field="created_by", op="contains", value="ops"),)


def test_a_string_that_is_not_an_operator_at_all_is_an_unknown_parameter():
    """`gt3` is a typo, not an operator the field withheld, so the useful reply is the
    parameter list rather than this field's operator set."""
    problem = _problem({"filter[max_budget][gt3]": "5"})

    assert problem.type == f"{PROBLEM_TYPE_BASE}unknown-query-parameter"
    assert problem.allowed is not None
    assert "filter[max_budget][gte]" in problem.allowed


# ---------------------------------------------------------------- invariant 9


def test_search_against_a_spec_with_nothing_searchable_is_rejected():
    """A silently-empty search filter returns the unfiltered table, which reads as
    "no results were filtered out" rather than "this resource cannot be searched"."""
    problem = _problem({"q": "prod"}, spec=_spec(searchable=frozenset()))

    assert problem.status == 400
    assert problem.type == f"{PROBLEM_TYPE_BASE}unknown-query-parameter"
    assert problem.allowed is not None
    assert "q" not in problem.allowed


def test_search_is_a_case_insensitive_or_across_every_searchable_field():
    conjuncts = _conjuncts(_plan({"q": "Prod"}))

    assert conjuncts == (
        AnyOf(
            clauses=(
                Compare(field="budget_id", op="contains", value="Prod"),
                Compare(field="created_by", op="contains", value="Prod"),
            )
        ),
    )


def test_an_empty_search_string_adds_no_filter():
    assert _plan({"q": ""}).where == ()


# --------------------------------------------------------------- invariant 10


def test_multi_key_sort_parses_the_json_api_grammar():
    order = _plan({"sort": "-created_at,budget_id,-max_budget"}).order

    assert order[:3] == (
        SortKey(field="created_at", descending=True),
        SortKey(field="budget_id", descending=False),
        SortKey(field="max_budget", descending=True),
    )


def test_sort_segments_tolerate_surrounding_whitespace():
    assert _plan({"sort": "-created_at, budget_id"}).order[:2] == (
        SortKey(field="created_at", descending=True),
        SortKey(field="budget_id", descending=False),
    )


# ------------------------------------------------------------- filter parsing


def test_comparison_operators_become_prisma_range_fragments():
    conjuncts = _conjuncts(_plan({"filter[max_budget][gte]": "5", "filter[max_budget][lte]": "50"}))

    assert conjuncts == (
        Compare(field="max_budget", op="gte", value=5.0),
        Compare(field="max_budget", op="lte", value=50.0),
    )


def test_eq_is_a_bare_value_not_a_wrapped_one():
    assert _conjuncts(_plan({"filter[tpm_limit][eq]": "100"})) == (Compare(field="tpm_limit", op="eq", value=100),)


def test_a_filter_with_no_operator_bracket_means_eq():
    """`filter[status]=active` is the design doc's canonical spelling for equality;
    only the non-eq operators carry a second bracket."""
    assert _conjuncts(_plan({"filter[tpm_limit]": "100"})) == (Compare(field="tpm_limit", op="eq", value=100),)


def test_the_bare_form_and_the_explicit_eq_form_agree():
    assert _plan({"filter[created_by]": "alice"}) == _plan({"filter[created_by][eq]": "alice"})


def test_the_bare_form_still_coerces_to_the_declared_type():
    assert _problem({"filter[tpm_limit]": "1.5"}).status == 400


def test_the_bare_form_is_rejected_on_a_field_that_does_not_declare_eq():
    """The shorthand is sugar for the eq operator, not a bypass around the operator set."""
    problem = _problem({"filter[created_at]": "2026-07-23T00:00:00Z"})

    assert problem.type == f"{PROBLEM_TYPE_BASE}unsupported-filter-operator"
    assert problem.allowed == ["gte", "lte"]


def test_the_allowed_set_advertises_the_bare_spelling_for_eq():
    allowed = _problem({"nope": "1"}).allowed

    assert allowed is not None
    assert "filter[max_budget]" in allowed
    assert "filter[max_budget][eq]" not in allowed
    assert "filter[created_at][gte]" in allowed
    assert "filter[created_at]" not in allowed


@pytest.mark.parametrize(
    "name",
    ["filter[]", "filter[a][b][c]", "filter[a][", "filter", "filter[a][gte", "filter[max_budget]]["],
    ids=["empty", "triple", "unbalanced", "bare-word", "unterminated", "bracketed-field"],
)
def test_malformed_filter_keys_are_unknown_parameters_not_eq_filters(name):
    """A malformed key must not fall through to the bare-eq branch and silently filter
    on a field nobody declared. `field in spec.filters` is the gate that makes this hold,
    which is also why the parser needs no separate well-formedness guard."""
    assert _problem({name: "x"}).type == f"{PROBLEM_TYPE_BASE}unknown-query-parameter"


def test_in_splits_on_commas_and_coerces_every_member():
    assert _conjuncts(_plan({"filter[created_by][in]": "alice, bob"})) == (
        Within(field="created_by", values=("alice", "bob")),
    )


def test_is_null_true_matches_rows_with_no_budget():
    """Budgets renders a null max_budget as "Unlimited"; without is_null there is no way
    to ask for those rows."""
    assert _conjuncts(_plan({"filter[max_budget][is_null]": "true"})) == (
        IsNull(field="max_budget", negated=False),
    )


def test_is_null_false_matches_rows_that_have_one():
    assert _conjuncts(_plan({"filter[max_budget][is_null]": "false"})) == (
        IsNull(field="max_budget", negated=True),
    )


def test_is_null_rejects_a_non_boolean():
    assert _problem({"filter[max_budget][is_null]": "maybe"}).status == 400


@pytest.mark.parametrize(
    "params",
    [
        {"filter[max_budget][gte]": "lots"},
        {"filter[tpm_limit][eq]": "1.5"},
        {"filter[created_at][gte]": "yesterday"},
        {"filter[created_by][in]": "alice,"},
    ],
    ids=["float", "int", "datetime", "in-member"],
)
def test_a_value_that_does_not_match_the_declared_type_is_rejected(params):
    numeric_in = _spec_with(filters={**_spec().filters, "created_by": FilterSpec(type=int, ops=frozenset({"in"}))})
    target = numeric_in if "filter[created_by][in]" in params else _spec()

    assert _problem(params, spec=target).status == 400


def test_a_datetime_filter_is_normalised_to_utc():
    """The dashboard sends both offset-bearing and naive timestamps; reading a naive one
    as server-local time would shift the window off what the table is showing."""
    with_offset = _conjuncts(_plan({"filter[created_at][gte]": "2026-07-23T02:00:00+02:00"}))
    naive = _conjuncts(_plan({"filter[created_at][gte]": "2026-07-23 00:00:00"}))

    assert with_offset == (Compare(field="created_at", op="gte", value=datetime(2026, 7, 23, tzinfo=timezone.utc)),)
    assert naive == with_offset


def test_filters_are_ordered_deterministically():
    """Two requests differing only in query-string order must plan identically, or the
    plan stops being a comparable value."""
    forwards = _plan({"filter[created_by][eq]": "a", "filter[max_budget][gte]": "5"})
    backwards = _plan({"filter[max_budget][gte]": "5", "filter[created_by][eq]": "a"})

    assert forwards == backwards


# ------------------------------------------------------------------- envelope


@pytest.mark.asyncio
async def test_returns_the_page_mode_envelope():
    executor = RecordingExecutor(
        rows=tuple(BudgetRow(budget_id=f"b{i}", max_budget=float(i), created_by="u") for i in range(10)),
        total_count=42,
    )

    response = await handle_list(
        spec=_spec(), executor=executor, request=_request("page=2&page_size=5"), caller=CALLER
    )
    body = response.model_dump(by_alias=True)

    assert body["meta"] == {"total_count": 42, "page": 2, "page_size": 5, "total_pages": 9}
    assert set(body) == {"data", "meta", "links"}
    assert "has_more" not in body["meta"]


@pytest.mark.asyncio
async def test_serializes_rows_flat_without_a_json_api_resource_wrapper():
    executor = RecordingExecutor(rows=(BudgetRow(budget_id="b1", max_budget=None, created_by="u"),))

    response = await handle_list(spec=_spec(), executor=executor, request=_request(), caller=CALLER)
    body = response.model_dump(by_alias=True)

    assert body["data"] == [{"budget_id": "b1", "max_budget": None}]
    assert "attributes" not in body["data"][0]
    assert "created_by" not in body["data"][0]


@pytest.mark.asyncio
async def test_links_let_a_client_page_without_building_urls():
    executor = RecordingExecutor(rows=(), total_count=42)

    response = await handle_list(
        spec=_spec(), executor=executor, request=_request("page=2&page_size=5"), caller=CALLER
    )
    links = response.model_dump(by_alias=True)["links"]

    assert links["self"] == f"{BUDGETS_PATH}?page_size=5&page=2"
    assert links["first"] == f"{BUDGETS_PATH}?page_size=5&page=1"
    assert links["prev"] == f"{BUDGETS_PATH}?page_size=5&page=1"
    assert links["next"] == f"{BUDGETS_PATH}?page_size=5&page=3"
    assert links["last"] == f"{BUDGETS_PATH}?page_size=5&page=9"


@pytest.mark.asyncio
async def test_the_last_page_has_no_next_link():
    executor = RecordingExecutor(rows=(), total_count=10)

    response = await handle_list(
        spec=_spec(), executor=executor, request=_request("page=2&page_size=5"), caller=CALLER
    )
    links = response.model_dump(by_alias=True)["links"]

    assert links["next"] is None
    assert links["prev"] == f"{BUDGETS_PATH}?page_size=5&page=1"


@pytest.mark.asyncio
async def test_an_empty_result_set_still_resolves_every_link():
    executor = RecordingExecutor(rows=(), total_count=0)

    response = await handle_list(spec=_spec(), executor=executor, request=_request(), caller=CALLER)
    body = response.model_dump(by_alias=True)

    assert body["data"] == []
    assert body["meta"]["total_pages"] == 0
    assert body["links"]["first"] == body["links"]["last"] == f"{BUDGETS_PATH}?page=1"
    assert body["links"]["next"] is None
    assert body["links"]["prev"] is None


@pytest.mark.asyncio
async def test_the_executor_counts_the_same_predicate_it_reads():
    """Counting a wider predicate than the read inflates total_pages and hands the UI
    pages that are always empty."""
    executor = RecordingExecutor(rows=(), total_count=3)
    spec = _spec(scope=lambda caller: ScopeWhere(where=(Compare(field="created_by", op="eq", value=caller.user_id),)))

    await handle_list(spec=spec, executor=executor, request=_request("filter[max_budget][gte]=5"), caller=CALLER)

    assert executor.plan is not None
    assert executor.count_where == executor.plan.where


@pytest.mark.asyncio
async def test_a_rejected_request_is_raised_as_a_problem_before_any_query():
    executor = RecordingExecutor(rows=())

    with pytest.raises(ManagementProblem) as raised:
        await handle_list(spec=_spec(), executor=executor, request=_request("sort=api_key"), caller=CALLER)

    assert raised.value.problem.status == 400
    assert executor.count_where is None


# ------------------------------------------------------- spec construction


def test_a_default_page_size_above_the_cap_is_rejected_at_construction():
    """The cap is only enforced on a supplied page_size, so a default above it would serve
    more rows than the resource allows on exactly the request that omits page_size."""
    with pytest.raises(ValueError, match="default_page_size"):
        _spec_with(default_page_size=200, max_page_size=100)


@pytest.mark.parametrize(
    "overrides",
    [{"default_page_size": 0}, {"default_page_size": -5}, {"max_page_size": 0}],
    ids=["zero-default", "negative-default", "zero-cap"],
)
def test_a_non_positive_page_size_is_rejected_at_construction(overrides):
    """take=0 divides by zero when handle_list computes total_pages, so the resource would
    500 on every request instead of failing when it is registered."""
    with pytest.raises(ValueError, match="default_page_size"):
        _spec_with(**overrides)


def test_a_default_sort_on_a_non_sortable_field_is_rejected_at_construction():
    """Caller-supplied sort is validated against `sortable`; default_sort is not read from
    the request, so without this it reaches order_by_sql and yields invalid SQL."""
    with pytest.raises(ValueError, match="default_sort"):
        _spec_with(default_sort=(SortKey(field="not_a_column", descending=True),))


def test_an_empty_tiebreaker_is_rejected_at_construction():
    with pytest.raises(ValueError, match="tiebreaker"):
        _spec_with(tiebreaker="")


def test_a_page_size_equal_to_the_cap_is_a_valid_spec():
    """Guards the bound against being tightened into an off-by-one that bans max==default."""
    assert _spec_with(default_page_size=100, max_page_size=100).default_page_size == 100


# ------------------------------------------------------ repeated parameters


@pytest.mark.asyncio
async def test_a_repeated_query_parameter_is_rejected():
    """Starlette keeps the last value, so ?page=1&page=999 would page from 999 with nothing
    telling the caller which one won. The doc rejects silently-altered params for this reason."""
    executor = RecordingExecutor(rows=())

    with pytest.raises(ManagementProblem) as raised:
        await handle_list(spec=_spec(), executor=executor, request=_request("page=1&page=999"), caller=CALLER)

    assert raised.value.problem.status == 400
    assert raised.value.problem.type == f"{PROBLEM_TYPE_BASE}duplicate-query-parameter"
    assert "page" in raised.value.problem.detail
    assert executor.count_where is None


@pytest.mark.asyncio
async def test_a_repeated_filter_parameter_is_rejected():
    executor = RecordingExecutor(rows=())

    with pytest.raises(ManagementProblem) as raised:
        await handle_list(
            spec=_spec(),
            executor=executor,
            request=_request("filter[created_by][eq]=alice&filter[created_by][eq]=bob"),
            caller=CALLER,
        )

    assert raised.value.problem.type == f"{PROBLEM_TYPE_BASE}duplicate-query-parameter"
    assert "filter[created_by][eq]" in raised.value.problem.detail


@pytest.mark.asyncio
async def test_distinct_parameters_are_not_treated_as_duplicates():
    """Guards the check against rejecting two different operators on one field, which is
    how a range filter is expressed."""
    executor = RecordingExecutor(rows=(), total_count=0)

    response = await handle_list(
        spec=_spec(),
        executor=executor,
        request=_request("filter[max_budget][gte]=5&filter[max_budget][lte]=50&page=2"),
        caller=CALLER,
    )

    assert response.meta.page == 2
    assert executor.count_where is not None


@pytest.mark.asyncio
async def test_a_denied_scope_outranks_a_duplicate_parameter():
    """Permission is the stronger statement about the caller, so it is answered first."""
    spec = _spec(scope=lambda caller: ScopeDenied(reason="nope"))
    executor = RecordingExecutor(rows=())

    with pytest.raises(ManagementProblem) as raised:
        await handle_list(spec=spec, executor=executor, request=_request("page=1&page=2"), caller=CALLER)

    assert raised.value.problem.status == 403


# --------------------------------------------------- facet-mode regression


def test_the_facet_page_shapes_are_untouched_by_page_mode():
    """The live facet endpoint reports `has_more` and has no first/last, because it
    deliberately skips the COUNT(*). Folding it into the page-mode shapes would either
    break its response or make every keystroke pay for a full-table count."""
    assert set(PageMeta.model_fields) == {"page", "page_size", "has_more"}
    assert set(PageLinks.model_fields) == {"self_link", "prev", "next"}

    links = build_page_links(request=_request("q=ac&page=2"), page=2, has_more=True).model_dump(by_alias=True)

    assert set(links) == {"self", "prev", "next"}
    assert links["next"] == "/management/v1/budgets?q=ac&page=3"
