from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import TypeAdapter

import litellm
import litellm.proxy.proxy_server as ps
from litellm.caching.dual_cache import DualCache
from litellm.models.budget import BudgetLimit
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.customer_hierarchy_auth import (
    CUSTOMER_MEMBER_ID_HEADER,
    CUSTOMER_ORG_ALIAS_HEADER,
    CUSTOMER_TEAM_ID_HEADER,
    AmbiguousOrganizationAlias,
    check_customer_hierarchy_budget,
    check_customer_hierarchy_key_scope,
    check_customer_hierarchy_model_access,
    check_customer_hierarchy_windows,
    customer_hierarchy_from_member_row,
    customer_identity_from_headers,
    customer_member_id_from_headers,
    forget_customer_organization_aliases,
    get_customer_hierarchy,
    get_customer_organization_hierarchy,
    resolve_customer_hierarchy,
    select_customer_team,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure
from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerRequestIdentity,
    CustomerScopeAuth,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import CustomerMember, Organization
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def customer_spend_store(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "spend_counter_store", SpendCounterStore(connect(None).spend_counters()))


@pytest.mark.asyncio
@pytest.mark.parametrize("alias_only", (False, True))
async def test_a_missing_shared_counter_reseeds_spend_without_reloading_a_stale_cached_snapshot(
    alias_only: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    cache: Final = UserApiKeyCache()
    old: Final = _Prisma(
        _MemberTable({"member-1": _row(org_budget={"max_budget": 1})}),
        _OrganizationTable((_organization("tenant", max_budget=1),)),
    )
    persisted: Final = _Prisma(
        _MemberTable({"member-1": _row(org_budget={"max_budget": 1}, spend=(2, 0, 0))}),
        _OrganizationTable((_organization("tenant", max_budget=1, spend=2),)),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", persisted)
    if alias_only:
        _ = await get_customer_organization_hierarchy(
            "tenant",
            old,  # pyright: ignore[reportArgumentType]  # database boundary fixture
            cache,
        )
    else:
        _ = await get_customer_hierarchy(
            "member-1",
            old,  # pyright: ignore[reportArgumentType]  # database boundary fixture
            cache,
        )
    identity: Final = CustomerRequestIdentity(
        customer_org_alias="tenant" if alias_only else None,
        customer_member_id=None if alias_only else "member-1",
    )
    with pytest.raises(litellm.BudgetExceededError) as rejected:
        await resolve_customer_hierarchy(
            identity,
            persisted,  # pyright: ignore[reportArgumentType]  # database boundary fixture
            cache,
            None,
            "model",
            "/responses",
        )
    assert rejected.value.current_cost == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ("org", "team", "member"))
@pytest.mark.parametrize("alias_only", (False, True))
async def test_settled_spend_reaches_another_workers_cached_hierarchy(
    monkeypatch: pytest.MonkeyPatch, level: str, alias_only: bool
) -> None:
    from litellm.proxy import proxy_server

    shared: Final[dict[str, object]] = {}
    settling: Final = FakeSpendCounters(redis=shared).store()
    reading: Final = FakeSpendCounters(redis=shared).store()
    monkeypatch.setattr(proxy_server, "spend_counter_store", reading)
    cache: Final = UserApiKeyCache()
    row: Final = _row(template={"max_budget": 1.0}, team_budget={"max_budget": 1.0}, org_budget={"max_budget": 1.0})
    prisma: Final = _Prisma(
        _MemberTable({"member-1": row}), _OrganizationTable((_organization("tenant", max_budget=1),))
    )
    identity: Final = CustomerRequestIdentity(
        customer_org_alias="tenant" if alias_only else None,
        customer_member_id=None if alias_only else "member-1",
    )
    await resolve_customer_hierarchy(
        identity,
        prisma,  # pyright: ignore[reportArgumentType]  # database boundary fixture
        cache,
        None,
        "model",
        "/responses",
    )
    counter: Final = f"spend:customer_{level}:{level}-1:lifetime"
    await settling.increment(counter, 1.25)
    if alias_only and level != "org":
        allowed: Final = await resolve_customer_hierarchy(
            identity,
            prisma,  # pyright: ignore[reportArgumentType]  # database boundary fixture
            cache,
            None,
            "model",
            "/responses",
        )
        assert allowed.customer_member_id is None
        return
    with pytest.raises(litellm.BudgetExceededError) as rejected:
        await resolve_customer_hierarchy(
            identity,
            prisma,  # pyright: ignore[reportArgumentType]  # database boundary fixture
            cache,
            None,
            "model",
            "/responses",
        )
    assert (rejected.value.entity_type, rejected.value.current_cost) == (f"customer_{level}", 1.25)


def _row(
    *,
    template: Mapping[str, object] | None = None,
    override: Mapping[str, object] | None = None,
    override_expires_at: datetime | None = None,
    team_budget: Mapping[str, object] | None = None,
    org_budget: Mapping[str, object] | None = None,
    spend: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mapping[str, object]:
    org_spend, team_spend, member_spend = spend
    return {
        "customer_member_id": "member-1",
        "parent_org_id": "org-1",
        "parent_team_ids": ("team-1",),
        "spend": member_spend,
        "budget": template,
        "budget_override": override,
        "budget_override_expires_at": override_expires_at,
        "parent_org": {"customer_org_id": "org-1", "spend": org_spend, "budget": org_budget},
        "teams": (
            {"customer_team_id": "team-1", "parent_org_id": "org-1", "spend": team_spend, "budget": team_budget},
        ),
    }


def test_customer_display_names_come_from_the_resolved_rows() -> None:
    hierarchy: Final = customer_hierarchy_from_member_row(
        {
            "customer_member_id": "customer-member-1",
            "alias": "Customer member display",
            "parent_team_ids": ["customer-team-1"],
            "teams": [{"customer_team_id": "customer-team-1", "alias": "Customer team display"}],
            "parent_org": {"customer_org_id": "customer-org-1", "alias": "Customer org display"},
        },
        NOW,
    )
    assert (hierarchy.customer_org_alias, hierarchy.customer_team_alias, hierarchy.customer_member_alias) == (
        "Customer org display",
        "Customer team display",
        "Customer member display",
    )


def _scopes(hierarchy: CustomerHierarchyAuth) -> Mapping[str, CustomerScopeAuth]:
    return {scope.key: scope for scope in hierarchy.scopes}


def test_member_row_yields_org_team_and_member_scopes_with_the_effective_member_budget() -> None:
    hierarchy: Final = customer_hierarchy_from_member_row(
        _row(
            template={"max_budget": 10.0, "tpm_limit": 1000, "model_tpm_limit": {"gpt-4o": 400}},
            override={"max_budget": 25.0, "allowed_models": ["gpt-4o"]},
            override_expires_at=NOW + timedelta(days=1),
            team_budget={"rpm_limit": 50, "model_rpm_limit": {"gpt-4o": 5}},
            org_budget={"max_budget": 100.0},
            spend=(1.0, 2.0, 3.0),
        ),
        NOW,
    )
    scopes: Final = _scopes(hierarchy)

    assert (hierarchy.customer_org_id, hierarchy.customer_team_id, hierarchy.customer_member_id) == (
        "org-1",
        "team-1",
        "member-1",
    )
    assert tuple(scope.key for scope in hierarchy.scopes) == ("customer_org", "customer_team", "customer_member")
    assert (scopes["customer_org"].max_budget, scopes["customer_org"].spend) == (100.0, 1.0)
    assert (scopes["customer_team"].rpm_limit, scopes["customer_team"].model_rpm_limit) == (50, {"gpt-4o": 5})
    assert scopes["customer_team"].max_budget is None
    member: Final = scopes["customer_member"]
    assert (member.max_budget, member.tpm_limit, member.allowed_models, member.spend) == (
        25.0,
        1000,
        ("gpt-4o",),
        3.0,
    )
    assert member.model_tpm_limit == {"gpt-4o": 400}


def test_a_team_override_caps_the_team_scope_until_it_expires() -> None:
    base: Final = _row(template={"max_budget": 10.0}, team_budget={"max_budget": 50.0, "rpm_limit": 5})
    team: Final = {**_team_of(base), "budget_override": {"max_budget": 500.0}}

    active: Final = _scopes(
        customer_hierarchy_from_member_row(
            {**base, "teams": ({**team, "budget_override_expires_at": NOW + timedelta(days=1)},)}, NOW
        )
    )["customer_team"]
    expired: Final = _scopes(
        customer_hierarchy_from_member_row(
            {**base, "teams": ({**team, "budget_override_expires_at": NOW - timedelta(seconds=1)},)}, NOW
        )
    )["customer_team"]

    assert (active.max_budget, active.rpm_limit) == (500.0, 5)
    assert (expired.max_budget, expired.rpm_limit) == (50.0, 5)


def _team_of(row: Mapping[str, object]) -> Mapping[str, object]:
    return TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(row["teams"])[0]


def test_a_member_in_two_teams_is_charged_to_the_first_unless_the_request_names_another() -> None:
    base: Final = _row(template={"max_budget": 10.0}, team_budget={"max_budget": 50.0})
    second: Final = {
        "customer_team_id": "team-2",
        "parent_org_id": "org-1",
        "spend": 7.0,
        "budget": {"max_budget": 5.0},
    }
    hierarchy: Final = customer_hierarchy_from_member_row(
        {**base, "parent_team_ids": ("team-1", "team-2"), "teams": (*_teams_of(base), second)}, NOW
    )

    assert hierarchy.customer_team_id == "team-1"
    assert tuple(scope.entity_id for scope in hierarchy.scopes) == ("org-1", "team-1", "member-1")
    assert tuple(scope.entity_id for scope in hierarchy.team_scopes) == ("team-1", "team-2")
    selected: Final = select_customer_team(hierarchy, "team-2")
    assert selected is not None
    assert selected.customer_team_id == "team-2"
    assert tuple((scope.entity_id, scope.max_budget) for scope in selected.scopes) == (
        ("org-1", None),
        ("team-2", 5.0),
        ("member-1", 10.0),
    )
    with pytest.raises(litellm.BudgetExceededError):
        check_customer_hierarchy_budget(selected)
    assert select_customer_team(hierarchy, "team-9") is None


@pytest.mark.asyncio
async def test_a_team_header_must_name_one_of_the_members_teams() -> None:
    prisma: Final = _Prisma(_MemberTable({"member-1": _row(template={"max_budget": 10.0})}))

    selected: Final = await _resolve(
        prisma, CustomerRequestIdentity(customer_member_id="member-1", customer_team_id="team-1")
    )
    assert selected.customer_team_id == "team-1"
    with pytest.raises(ProxyException) as exc:
        await _resolve(prisma, CustomerRequestIdentity(customer_member_id="member-1", customer_team_id="team-9"))
    assert (exc.value.code, exc.value.param) == ("403", CUSTOMER_TEAM_ID_HEADER)


def _teams_of(row: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(row["teams"])


def test_an_expired_override_leaves_the_member_on_its_template() -> None:
    hierarchy: Final = customer_hierarchy_from_member_row(
        _row(
            template={"max_budget": 10.0},
            override={"max_budget": 25.0},
            override_expires_at=NOW - timedelta(seconds=1),
        ),
        NOW,
    )
    assert _scopes(hierarchy)["customer_member"].max_budget == 10.0


def test_levels_without_a_budget_row_carry_no_limits() -> None:
    hierarchy: Final = customer_hierarchy_from_member_row(_row(template={"rpm_limit": 1}), NOW)
    org: Final = _scopes(hierarchy)["customer_org"]
    assert (org.max_budget, org.tpm_limit, org.rpm_limit, org.allowed_models, org.model_tpm_limit) == (
        None,
        None,
        None,
        None,
        None,
    )


def test_budget_check_fails_the_outermost_exhausted_level_first() -> None:
    hierarchy: Final = customer_hierarchy_from_member_row(
        _row(
            template={"max_budget": 1.0},
            team_budget={"max_budget": 5.0},
            org_budget={"max_budget": 50.0},
            spend=(50.0, 5.0, 0.5),
        ),
        NOW,
    )
    with pytest.raises(litellm.BudgetExceededError) as exc:
        check_customer_hierarchy_budget(hierarchy)
    assert (exc.value.entity_type, exc.value.entity_id) == ("customer_org", "org-1")

    within: Final = customer_hierarchy_from_member_row(
        _row(template={"max_budget": 1.0}, team_budget={"max_budget": 5.0}, spend=(0.0, 4.99, 0.99)), NOW
    )
    check_customer_hierarchy_budget(within)


def test_model_access_requires_every_level_with_an_allowlist_to_permit_the_model() -> None:
    hierarchy: Final = customer_hierarchy_from_member_row(
        _row(template={"allowed_models": ["luna", "opus"]}, team_budget={"allowed_models": ["luna"]}), NOW
    )
    check_customer_hierarchy_model_access(hierarchy, "luna", llm_router=None)
    with pytest.raises(ProxyException) as exc:
        check_customer_hierarchy_model_access(hierarchy, "opus", llm_router=None)
    assert exc.value.code == "403"

    unrestricted: Final = customer_hierarchy_from_member_row(_row(template={"max_budget": 1.0}), NOW)
    check_customer_hierarchy_model_access(unrestricted, "anything", llm_router=None)


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"X-LiteLLM-Customer-Member-Id": " member-9 "}, "member-9"),
        ({CUSTOMER_MEMBER_ID_HEADER: ""}, None),
        ({"x-litellm-customer-id": "end-user"}, None),
        (None, None),
    ],
)
def test_member_header_is_read_case_insensitively_and_blank_values_are_ignored(
    headers: Mapping[str, object] | None, expected: str | None
) -> None:
    assert customer_member_id_from_headers(headers) == expected


class _MemberTable:
    def __init__(self, rows: Mapping[str, Mapping[str, object]]) -> None:
        self.rows = rows
        self.lookups = 0

    async def find_unique(self, where: Mapping[str, object], include: Mapping[str, object] | None = None):
        self.lookups += 1
        row: Final = self.rows.get(str(where["customer_member_id"]))
        return _Record(row) if row is not None else None


class _Record:
    def __init__(self, values: Mapping[str, object]) -> None:
        self.values = values

    def model_dump(self) -> Mapping[str, object]:
        return self.values

    @property
    def spend(self):
        return self.values.get("spend", 0.0)


class _OrganizationTable:
    def __init__(self, rows: Sequence[Mapping[str, object]] = ()) -> None:
        self.rows = rows
        self.lookups = 0

    async def find_many(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None, take: int | None = None
    ):
        self.lookups += 1
        matching: Final = [_Record(row) for row in self.rows if row.get("alias") == where["alias"]]
        return matching[:take] if take is not None else matching

    async def find_unique(self, where: Mapping[str, object], include: Mapping[str, object] | None = None):
        return next((_Record(row) for row in self.rows if row.get("customer_org_id") == where["customer_org_id"]), None)


class _TeamTable:
    """Serves the team rows the member fixtures embed under ``teams``, as the proxy's second lookup would."""

    def __init__(self, members: _MemberTable) -> None:
        self.members = members

    async def find_many(self, where: Mapping[str, object], include: Mapping[str, object] | None = None):
        wanted: Final = TypeAdapter(Mapping[str, Sequence[str]]).validate_python(where["customer_team_id"])["in"]
        teams: Final = TypeAdapter(tuple[Mapping[str, object], ...])
        return [
            _Record(team)
            for row in self.members.rows.values()
            for team in teams.validate_python(row.get("teams") or ())
            if team["customer_team_id"] in wanted
        ]  # comprehension-ok: flattening the fixture rows' embedded teams

    async def find_unique(self, where: Mapping[str, object], include: Mapping[str, object] | None = None):
        rows: Final = await self.find_many({"customer_team_id": {"in": (where["customer_team_id"],)}}, include)
        return rows[0] if rows else None


class _Db:
    def __init__(self, table: _MemberTable, organizations: _OrganizationTable) -> None:
        self.litellm_customermembertable = table
        self.litellm_customerteamtable = _TeamTable(table)
        self.litellm_customerorganizationtable = organizations


class _Prisma:
    def __init__(self, table: _MemberTable, organizations: _OrganizationTable | None = None) -> None:
        self.db = _Db(table, organizations if organizations is not None else _OrganizationTable())


def _organization(
    alias: str, org_id: str = "org-1", spend: float = 0.0, max_budget: float | None = None
) -> Mapping[str, object]:
    budget: Final = {"max_budget": max_budget} if max_budget is not None else None
    return {"customer_org_id": org_id, "alias": alias, "spend": spend, "budget": budget}


class _Registrar:
    """Stands in for ensure_customer_member: records the ask and adds the member row, or fails."""

    def __init__(self, members: _MemberTable, failure: Failure | None = None) -> None:
        self.members = members
        self.failure = failure
        self.calls: tuple[tuple[str, str], ...] = ()

    async def __call__(self, org_id: str, member_id: str, prisma_client: object) -> CustomerMember | Failure:
        self.calls = (*self.calls, (org_id, member_id))
        if self.failure is not None:
            return self.failure
        self.members.rows = {
            **self.members.rows,
            member_id: {**_row(template={"max_budget": 10.0}), "customer_member_id": member_id},
        }
        return CustomerMember.model_validate(
            {
                "customer_member_id": member_id,
                "parent_org_id": "org-1",
                "parent_team_ids": ("team-1",),
                "effective_budget": {"max_budget": 10.0},
                "created_at": NOW,
                "created_by": "x",
                "updated_at": NOW,
                "updated_by": "x",
            }
        )


class _OrganizationRegistrar:
    """Stands in for ensure_customer_organization: records the alias and adds the organization row, or fails."""

    def __init__(self, organizations: _OrganizationTable, failure: Failure | None = None) -> None:
        self.organizations = organizations
        self.failure = failure
        self.calls: tuple[str, ...] = ()

    async def __call__(self, alias: str, prisma_client: object) -> Organization | Failure:
        self.calls = (*self.calls, alias)
        if self.failure is not None:
            return self.failure
        self.organizations.rows = (*self.organizations.rows, _organization(alias, org_id=alias))
        return Organization.model_validate(
            {
                "customer_org_id": alias,
                "alias": alias,
                "member_count": 0,
                "created_at": NOW,
                "created_by": "x",
                "updated_at": NOW,
                "updated_by": "x",
            }
        )


def _resolve(
    prisma: _Prisma,
    identity: CustomerRequestIdentity,
    route: str = "/chat/completions",
    register: _Registrar | None = None,
    key_metadata: Mapping[str, object] | None = None,
    register_organization: _OrganizationRegistrar | None = None,
):
    return resolve_customer_hierarchy(
        identity=identity,
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # test double
        user_api_key_cache=UserApiKeyCache(),
        llm_router=None,
        request_model="gpt-4o",
        route=route,
        key_metadata=key_metadata,
        register=register if register is not None else _Registrar(_MemberTable({}), Failure(ErrorCode.NOT_FOUND)),
        register_organization=register_organization
        if register_organization is not None
        else _OrganizationRegistrar(_OrganizationTable(), Failure(ErrorCode.NOT_FOUND)),
    )


@pytest.mark.asyncio
async def test_hierarchy_is_loaded_once_and_then_served_from_the_cache() -> None:
    table: Final = _MemberTable({"member-1": _row(template={"max_budget": 10.0})})
    cache: Final = UserApiKeyCache()
    prisma: Final = _Prisma(table)

    first: Final = await get_customer_hierarchy("member-1", prisma, cache, now=NOW)  # pyright: ignore[reportArgumentType]  # test double
    second: Final = await get_customer_hierarchy("member-1", prisma, cache, now=NOW)  # pyright: ignore[reportArgumentType]  # test double

    assert first is not None and second == first
    assert table.lookups == 1
    assert await get_customer_hierarchy("missing", prisma, cache, now=NOW) is None  # pyright: ignore[reportArgumentType]  # test double


@pytest.mark.asyncio
async def test_an_unknown_member_or_a_proxy_without_a_database_runs_the_request_untracked() -> None:
    cache: Final = UserApiKeyCache()
    untracked: Final = await resolve_customer_hierarchy(
        identity=CustomerRequestIdentity(customer_member_id="ghost"),
        prisma_client=_Prisma(_MemberTable({})),  # pyright: ignore[reportArgumentType]  # test double
        user_api_key_cache=cache,
        llm_router=None,
        request_model="gpt-4o",
        route="/chat/completions",
    )
    assert untracked is None
    without_database: Final = await resolve_customer_hierarchy(
        identity=CustomerRequestIdentity(customer_member_id="member-1"),
        prisma_client=None,
        user_api_key_cache=cache,
        llm_router=None,
        request_model="gpt-4o",
        route="/chat/completions",
    )
    assert without_database is None


@pytest.mark.asyncio
async def test_resolution_enforces_budget_on_model_routes_but_not_on_info_routes() -> None:
    cache: Final = UserApiKeyCache()
    prisma: Final = _Prisma(_MemberTable({"member-1": _row(template={"max_budget": 1.0}, spend=(0.0, 0.0, 1.0))}))
    with pytest.raises(litellm.BudgetExceededError):
        await resolve_customer_hierarchy(
            identity=CustomerRequestIdentity(customer_member_id="member-1"),
            prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # test double
            user_api_key_cache=cache,
            llm_router=None,
            request_model="gpt-4o",
            route="/chat/completions",
        )
    hierarchy: Final = await resolve_customer_hierarchy(
        identity=CustomerRequestIdentity(customer_member_id="member-1"),
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # test double
        user_api_key_cache=cache,
        llm_router=None,
        request_model=None,
        route="/key/info",
    )
    assert hierarchy is not None and hierarchy.customer_member_id == "member-1"


def test_a_key_pinned_to_another_organization_cannot_act_for_the_member() -> None:
    hierarchy: Final = customer_hierarchy_from_member_row(_row(template={"max_budget": 10.0}), now=NOW)
    with pytest.raises(ProxyException) as denied:
        check_customer_hierarchy_key_scope(hierarchy, {"customer_org_id": "some-other-org"})
    assert denied.value.code == "403"
    check_customer_hierarchy_key_scope(hierarchy, {"customer_org_id": "org-1"})
    check_customer_hierarchy_key_scope(hierarchy, {})


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"X-LiteLLM-Customer-Org-Alias": " acme "}, CustomerRequestIdentity(customer_org_alias="acme")),
        (
            {CUSTOMER_ORG_ALIAS_HEADER: "acme", CUSTOMER_MEMBER_ID_HEADER: "member-1"},
            CustomerRequestIdentity(customer_member_id="member-1", customer_org_alias="acme"),
        ),
        ({CUSTOMER_MEMBER_ID_HEADER: "member-1"}, CustomerRequestIdentity(customer_member_id="member-1")),
        ({CUSTOMER_ORG_ALIAS_HEADER: "  ", "x-litellm-end-user-id": "end-user"}, None),
        (None, None),
    ],
)
def test_the_request_identity_is_read_from_both_customer_headers(
    headers: Mapping[str, object] | None, expected: CustomerRequestIdentity | None
) -> None:
    assert customer_identity_from_headers(headers) == expected


@pytest.mark.asyncio
async def test_an_alias_alone_charges_only_the_organization_and_enforces_its_budget() -> None:
    prisma: Final = _Prisma(_MemberTable({}), _OrganizationTable((_organization("acme", max_budget=5.0, spend=1.0),)))
    hierarchy: Final = await _resolve(prisma, CustomerRequestIdentity(customer_org_alias="acme"))

    assert hierarchy is not None
    assert (hierarchy.customer_org_id, hierarchy.customer_team_id, hierarchy.customer_member_id) == (
        "org-1",
        None,
        None,
    )
    assert tuple((scope.key, scope.entity_id, scope.max_budget, scope.spend) for scope in hierarchy.scopes) == (
        ("customer_org", "org-1", 5.0, 1.0),
    )

    exhausted: Final = _Prisma(
        _MemberTable({}), _OrganizationTable((_organization("acme", max_budget=5.0, spend=5.0),))
    )
    with pytest.raises(litellm.BudgetExceededError) as exc:
        await _resolve(exhausted, CustomerRequestIdentity(customer_org_alias="acme"))
    assert (exc.value.entity_type, exc.value.entity_id) == ("customer_org", "org-1")


@pytest.mark.asyncio
async def test_an_alias_next_to_a_member_must_name_that_members_organization() -> None:
    members: Final = _MemberTable({"member-1": _row(template={"max_budget": 10.0})})
    organizations: Final = _OrganizationTable((_organization("acme"), _organization("other", org_id="org-2")))
    prisma: Final = _Prisma(members, organizations)

    matching: Final = await _resolve(
        prisma, CustomerRequestIdentity(customer_member_id="member-1", customer_org_alias="acme")
    )
    assert matching is not None
    assert (matching.customer_org_id, matching.customer_member_id) == ("org-1", "member-1")
    assert tuple(scope.key for scope in matching.scopes) == ("customer_org", "customer_team", "customer_member")

    with pytest.raises(ProxyException) as denied:
        await _resolve(prisma, CustomerRequestIdentity(customer_member_id="member-1", customer_org_alias="other"))
    assert (denied.value.code, denied.value.param) == ("403", CUSTOMER_ORG_ALIAS_HEADER)


@pytest.mark.asyncio
@pytest.mark.parametrize("member_id", (None, "new-member"))
async def test_a_scoped_key_cannot_create_another_organization_before_being_rejected(member_id: str | None) -> None:
    organizations: Final = _OrganizationTable((_organization("existing"),))
    registrar: Final = _OrganizationRegistrar(organizations)
    prisma: Final = _Prisma(_MemberTable({}), organizations)

    with pytest.raises(ProxyException) as denied:
        await _resolve(
            prisma,
            CustomerRequestIdentity(customer_org_alias="new-organization", customer_member_id=member_id),
            key_metadata={"customer_org_id": "org-1"},
            register_organization=registrar,
        )

    assert (denied.value.code, denied.value.param) == ("403", CUSTOMER_MEMBER_ID_HEADER)
    assert registrar.calls == ()
    assert tuple(row["customer_org_id"] for row in organizations.rows) == ("org-1",)


@pytest.mark.asyncio
async def test_an_unknown_alias_becomes_a_new_organization_charged_for_the_request() -> None:
    organizations: Final = _OrganizationTable(())
    registrar: Final = _OrganizationRegistrar(organizations)
    prisma: Final = _Prisma(_MemberTable({}), organizations)

    hierarchy: Final = await _resolve(
        prisma, CustomerRequestIdentity(customer_org_alias="ghost"), register_organization=registrar
    )

    assert registrar.calls == ("ghost",)
    assert hierarchy is not None
    assert (hierarchy.customer_org_id, hierarchy.customer_team_id, hierarchy.customer_member_id) == (
        "ghost",
        None,
        None,
    )
    assert tuple((scope.key, scope.entity_id) for scope in hierarchy.scopes) == (("customer_org", "ghost"),)

    again: Final = await _resolve(
        prisma, CustomerRequestIdentity(customer_org_alias="ghost"), register_organization=registrar
    )
    assert again is not None and registrar.calls == ("ghost",)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (ErrorCode.ALREADY_EXISTS, ErrorCode.FREE_TIER_LIMIT))
async def test_an_alias_that_cannot_be_registered_runs_untracked_and_a_shared_alias_is_rejected_as_ambiguous(
    error: ErrorCode,
) -> None:
    failing: Final = _OrganizationRegistrar(_OrganizationTable(()), Failure(error))
    untracked: Final = await _resolve(
        _Prisma(_MemberTable({})), CustomerRequestIdentity(customer_org_alias="ghost"), register_organization=failing
    )
    assert (untracked, failing.calls) == (None, ("ghost",))

    oversized: Final = _OrganizationRegistrar(_OrganizationTable(()))
    assert (
        await _resolve(
            _Prisma(_MemberTable({})),
            CustomerRequestIdentity(customer_org_alias="g" * 257),
            register_organization=oversized,
        )
        is None
    )
    assert oversized.calls == ()

    shared: Final = _Prisma(
        _MemberTable({}), _OrganizationTable((_organization("acme"), _organization("acme", org_id="org-2")))
    )
    with pytest.raises(ProxyException) as ambiguous:
        await _resolve(shared, CustomerRequestIdentity(customer_org_alias="acme"))
    assert ambiguous.value.code == "400"


@pytest.mark.asyncio
async def test_an_organization_alias_is_loaded_once_then_served_from_the_cache_until_forgotten() -> None:
    organizations: Final = _OrganizationTable((_organization("acme"),))
    prisma: Final = _Prisma(_MemberTable({}), organizations)
    cache: Final = UserApiKeyCache()

    first: Final = await get_customer_organization_hierarchy("acme", prisma, cache, now=NOW)  # pyright: ignore[reportArgumentType]  # test double
    second: Final = await get_customer_organization_hierarchy("acme", prisma, cache, now=NOW)  # pyright: ignore[reportArgumentType]  # test double
    assert isinstance(first, CustomerHierarchyAuth) and second == first
    assert organizations.lookups == 1

    await forget_customer_organization_aliases(("acme", None), cache)
    await get_customer_organization_hierarchy("acme", prisma, cache, now=NOW)  # pyright: ignore[reportArgumentType]  # test double
    assert organizations.lookups == 2
    assert await get_customer_organization_hierarchy("ghost", prisma, cache, now=NOW) is None  # pyright: ignore[reportArgumentType]  # test double
    shared: Final = _Prisma(
        _MemberTable({}), _OrganizationTable((_organization("acme"), _organization("acme", org_id="org-2")))
    )
    assert isinstance(
        await get_customer_organization_hierarchy("acme", shared, UserApiKeyCache(), now=NOW),  # pyright: ignore[reportArgumentType]  # test double
        AmbiguousOrganizationAlias,
    )


@pytest.mark.asyncio
async def test_a_member_the_aliased_organization_has_never_seen_is_registered_there_and_served() -> None:
    members: Final = _MemberTable({})
    registrar: Final = _Registrar(members)
    prisma: Final = _Prisma(members, _OrganizationTable((_organization("acme"),)))

    hierarchy: Final = await _resolve(
        prisma, CustomerRequestIdentity(customer_member_id="user-7", customer_org_alias="acme"), register=registrar
    )

    assert registrar.calls == (("org-1", "user-7"),)
    assert hierarchy is not None
    assert (hierarchy.customer_org_id, hierarchy.customer_member_id) == ("org-1", "user-7")
    assert _scopes(hierarchy)["customer_member"].max_budget == 10.0


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (ErrorCode.BUDGET_REQUIRED, ErrorCode.FREE_TIER_LIMIT))
@pytest.mark.parametrize("team_id", (None, "team-1"))
async def test_an_unknown_member_runs_untracked_without_an_alias_and_on_the_organization_when_registration_fails(
    error: ErrorCode,
    team_id: str | None,
) -> None:
    members: Final = _MemberTable({})
    registrar: Final = _Registrar(members)
    no_alias: Final = await _resolve(
        _Prisma(members), CustomerRequestIdentity(customer_member_id="user-7"), register=registrar
    )
    assert (no_alias, registrar.calls) == (None, ())

    failing: Final = _Registrar(members, Failure(error))
    prisma: Final = _Prisma(members, _OrganizationTable((_organization("acme", max_budget=5.0),)))
    organization_only: Final = await _resolve(
        prisma,
        CustomerRequestIdentity(customer_member_id="user-7", customer_org_alias="acme", customer_team_id=team_id),
        register=failing,
    )
    assert failing.calls == (("org-1", "user-7"),)
    assert organization_only is not None
    assert (organization_only.customer_org_id, organization_only.customer_member_id) == ("org-1", None)
    assert tuple(scope.key for scope in organization_only.scopes) == ("customer_org",)
    assert "user-7" not in members.rows


@pytest.mark.asyncio
async def test_a_key_pinned_to_another_organization_is_refused_before_any_member_is_registered() -> None:
    members: Final = _MemberTable({})
    registrar: Final = _Registrar(members)
    prisma: Final = _Prisma(members, _OrganizationTable((_organization("acme"),)))

    with pytest.raises(ProxyException) as denied:
        await _resolve(
            prisma,
            CustomerRequestIdentity(customer_member_id="user-7", customer_org_alias="acme"),
            register=registrar,
            key_metadata={"customer_org_id": "some-other-org"},
        )

    assert (denied.value.code, registrar.calls, members.rows) == ("403", (), {})


@pytest.mark.asyncio
async def test_a_key_pinned_to_another_organization_is_refused_before_an_unknown_alias_is_registered() -> None:
    organizations: Final = _OrganizationTable((_organization("acme"),))
    registrar: Final = _OrganizationRegistrar(organizations)
    prisma: Final = _Prisma(_MemberTable({}), organizations)

    with pytest.raises(ProxyException) as denied:
        await _resolve(
            prisma,
            CustomerRequestIdentity(customer_org_alias="ghost"),
            key_metadata={"customer_org_id": "org-1"},
            register_organization=registrar,
        )
    assert (denied.value.code, registrar.calls, len(organizations.rows)) == ("403", (), 1)

    own: Final = await _resolve(
        prisma,
        CustomerRequestIdentity(customer_org_alias="ghost"),
        key_metadata={"customer_org_id": "ghost"},
        register_organization=registrar,
    )
    assert own is not None and (own.customer_org_id, registrar.calls) == ("ghost", ("ghost",))


@pytest.mark.asyncio
async def test_a_member_id_past_the_identifier_bound_is_not_registered_and_only_the_organization_is_charged() -> None:
    members: Final = _MemberTable({})
    registrar: Final = _Registrar(members)
    prisma: Final = _Prisma(members, _OrganizationTable((_organization("acme"),)))

    oversized: Final = await _resolve(
        prisma, CustomerRequestIdentity(customer_member_id="u" * 257, customer_org_alias="acme"), register=registrar
    )
    assert oversized is not None
    assert (oversized.customer_org_id, oversized.customer_member_id, registrar.calls) == ("org-1", None, ())

    longest: Final = await _resolve(
        prisma, CustomerRequestIdentity(customer_member_id="u" * 256, customer_org_alias="acme"), register=registrar
    )
    assert longest is not None
    assert (longest.customer_member_id, registrar.calls) == ("u" * 256, (("org-1", "u" * 256),))


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted_duration", ("1mo", "12mo"))
async def test_either_customer_cap_blocks_while_the_other_has_capacity(exhausted_duration: str, monkeypatch) -> None:
    hierarchy = customer_hierarchy_from_member_row(
        _row(
            template={
                "budget_limits": [
                    {"budget_duration": "1mo", "max_budget": 10},
                    {"budget_duration": "12mo", "max_budget": 100},
                ]
            }
        ),
        NOW,
    )
    backend = FakeSpendCounters()
    store = backend.store()
    monkeypatch.setattr(ps, "spend_counter_store", store)
    monkeypatch.setattr(ps, "prisma_client", None)
    member = hierarchy.scopes[-1]
    counters = customer_window_counters(member.key, member.entity_id, member.budget_limits, now=NOW)
    for counter in counters:
        await store.set(counter.counter_key, 1.0)
    await check_customer_hierarchy_windows(hierarchy, NOW)
    exhausted = next(counter for counter in counters if counter.window_duration == exhausted_duration)
    await store.set(exhausted.counter_key, exhausted.max_budget)
    with pytest.raises(litellm.BudgetExceededError) as error:
        await check_customer_hierarchy_windows(hierarchy, NOW)
    assert error.value.entity_id == member.entity_id
    assert error.value.current_cost == exhausted.max_budget
    assert exhausted_duration in str(error.value)


@pytest.mark.asyncio
async def test_missing_customer_window_balance_rejects_request(monkeypatch) -> None:
    hierarchy = customer_hierarchy_from_member_row(
        _row(template={"budget_limits": [{"budget_duration": "1mo", "max_budget": 10}]}), NOW
    )
    monkeypatch.setattr(ps, "spend_counter_store", FakeSpendCounters().store())
    monkeypatch.setattr(ps, "prisma_client", None)
    with pytest.raises(ProxyException) as error:
        await check_customer_hierarchy_windows(hierarchy, NOW)
    assert error.value.code == "503"
    assert error.value.param == "budget_limits"


@pytest.mark.asyncio
async def test_customer_windows_are_independent_for_members_sharing_a_budget(monkeypatch) -> None:
    first = customer_hierarchy_from_member_row(
        _row(template={"budget_limits": [{"budget_duration": "1mo", "max_budget": 10}]}), NOW
    )
    first_scope = first.scopes[-1]
    second_scope = first_scope.model_copy(update={"entity_id": "member-2"})
    second = first.model_copy(update={"customer_member_id": "member-2", "scopes": (second_scope,)})
    backend = FakeSpendCounters()
    store = backend.store()
    monkeypatch.setattr(ps, "spend_counter_store", store)
    monkeypatch.setattr(ps, "prisma_client", None)
    first_counter = customer_window_counters(
        first_scope.key, first_scope.entity_id, first_scope.budget_limits, now=NOW
    )[0]
    second_counter = customer_window_counters(
        second_scope.key, second_scope.entity_id, second_scope.budget_limits, now=NOW
    )[0]
    await store.set(first_counter.counter_key, 10)
    await store.set(second_counter.counter_key, 0)
    with pytest.raises(litellm.BudgetExceededError):
        await check_customer_hierarchy_windows(first, NOW)
    await check_customer_hierarchy_windows(second, NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_key", ("customer_org", "customer_team", "customer_member"))
@pytest.mark.parametrize("duration", ("1mo", "12mo"))
async def test_customer_window_repairs_stale_redis_and_rejects_from_durable_spend(monkeypatch, scope_key, duration):
    scope: Final = CustomerScopeAuth(
        key=scope_key,
        entity_id="subject",
        budget_limits=(BudgetLimit(budget_duration=duration, max_budget=10),),
    )
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=(scope,))
    counter: Final = customer_window_counters(scope.key, scope.entity_id, scope.budget_limits, now=NOW)[0]
    backend: Final = FakeSpendCounters()
    store: Final = backend.store()
    await store.set(counter.counter_key, 2.0)
    prisma: Final = MagicMock()
    prisma.db.litellm_budgetwindowspend.find_unique = AsyncMock(
        return_value=SimpleNamespace(window_start=counter.window_start, spend=12.0)
    )
    monkeypatch.setattr(ps, "prisma_client", prisma)
    monkeypatch.setattr(ps, "spend_counter_store", store)
    monkeypatch.setattr(ps, "spend_counter_cache", DualCache())
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": False})

    with pytest.raises(litellm.BudgetExceededError) as error:
        await check_customer_hierarchy_windows(hierarchy, NOW)

    assert error.value.current_cost == 12.0
    assert error.value.entity_id == scope.entity_id
    assert backend.redis[counter.counter_key] == 12.0
    prisma.db.litellm_budgetwindowspend.find_unique.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("outage", ("redis_read", "redis_seed", "database"))
async def test_customer_window_unavailable_stays_closed_without_legacy_flag(monkeypatch, outage):
    scope: Final = CustomerScopeAuth(
        key="customer_member",
        entity_id="outage-member",
        budget_limits=(BudgetLimit(budget_duration="1mo", max_budget=10),),
    )
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=(scope,))
    counter: Final = customer_window_counters(scope.key, scope.entity_id, scope.budget_limits, now=NOW)[0]
    backend: Final = FakeSpendCounters(fail=outage == "redis_read", memory={counter.counter_key: 2.0})
    prisma: Final = MagicMock()

    async def find_window(**_):
        if outage == "database":
            raise ConnectionError("database unavailable")
        if outage == "redis_seed":
            backend.fail = True
        return SimpleNamespace(window_start=counter.window_start, spend=0.0)

    prisma.db.litellm_budgetwindowspend.find_unique = AsyncMock(side_effect=find_window)
    monkeypatch.setattr(ps, "prisma_client", prisma)
    monkeypatch.setattr(ps, "spend_counter_store", backend.store())
    monkeypatch.setattr(ps, "spend_counter_cache", DualCache())
    monkeypatch.setattr(ps, "general_settings", {"fail_closed_budget_enforcement": False})

    with pytest.raises(ProxyException) as error:
        await check_customer_hierarchy_windows(hierarchy, NOW)

    assert error.value.code == "503"
    assert error.value.param == "budget_limits"


@pytest.mark.asyncio
async def test_customer_window_floor_checks_reuse_main_cache_and_one_mget_per_request(monkeypatch):
    scope: Final = CustomerScopeAuth(
        key="customer_member",
        entity_id="cached-floor-member",
        budget_limits=tuple(BudgetLimit(budget_duration=duration, max_budget=10) for duration in ("1mo", "12mo")),
    )
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=(scope,))
    counters: Final = customer_window_counters(scope.key, scope.entity_id, scope.budget_limits, now=NOW)
    backend: Final = FakeSpendCounters(redis={counter.counter_key: 4.0 for counter in counters})
    starts: Final = {counter.window_duration: counter.window_start for counter in counters}
    prisma: Final = MagicMock()
    prisma.db.litellm_budgetwindowspend.find_unique = AsyncMock(
        side_effect=lambda where: SimpleNamespace(
            window_start=starts[where["entity_type_entity_id_window_duration"]["window_duration"]], spend=2.0
        )
    )
    monkeypatch.setattr(ps, "prisma_client", prisma)
    monkeypatch.setattr(ps, "spend_counter_store", backend.store())
    monkeypatch.setattr(ps, "spend_counter_cache", DualCache())

    await check_customer_hierarchy_windows(hierarchy, NOW)
    await check_customer_hierarchy_windows(hierarchy, NOW)

    assert prisma.db.litellm_budgetwindowspend.find_unique.await_count == 2
    assert backend.redis == {counter.counter_key: 4.0 for counter in counters}
    assert len(backend.commands) == 2
    assert all(command.startswith("MGET ") for command in backend.commands)
