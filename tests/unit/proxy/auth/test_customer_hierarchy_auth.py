from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

import litellm
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.customer_hierarchy_auth import (
    CUSTOMER_MEMBER_ID_HEADER,
    CUSTOMER_ORG_ALIAS_HEADER,
    AmbiguousOrganizationAlias,
    check_customer_hierarchy_budget,
    check_customer_hierarchy_key_scope,
    check_customer_hierarchy_model_access,
    customer_hierarchy_from_member_row,
    customer_identity_from_headers,
    customer_member_id_from_headers,
    forget_customer_organization_aliases,
    get_customer_hierarchy,
    get_customer_organization_hierarchy,
    resolve_customer_hierarchy,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerRequestIdentity,
    CustomerScopeAuth,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import CustomerMember

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


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
        "parent_team_id": "team-1",
        "spend": member_spend,
        "budget": template,
        "budget_override": override,
        "budget_override_expires_at": override_expires_at,
        "parent_team": {
            "customer_team_id": "team-1",
            "parent_org_id": "org-1",
            "spend": team_spend,
            "budget": team_budget,
            "parent_org": {"customer_org_id": "org-1", "spend": org_spend, "budget": org_budget},
        },
    }


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


class _Db:
    def __init__(self, table: _MemberTable, organizations: _OrganizationTable) -> None:
        self.litellm_customermembertable = table
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
                "parent_team_id": "team-1",
                "effective_budget": {"max_budget": 10.0},
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
):
    return resolve_customer_hierarchy(
        identity=identity,
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # test double
        user_api_key_cache=UserApiKeyCache(),
        llm_router=None,
        request_model="gpt-4o",
        route=route,
        register=register if register is not None else _Registrar(_MemberTable({}), Failure(ErrorCode.NOT_FOUND)),
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
async def test_resolution_refuses_an_unknown_member_and_a_proxy_without_a_database() -> None:
    cache: Final = UserApiKeyCache()
    with pytest.raises(ProxyException) as missing:
        await resolve_customer_hierarchy(
            identity=CustomerRequestIdentity(customer_member_id="ghost"),
            prisma_client=_Prisma(_MemberTable({})),  # pyright: ignore[reportArgumentType]  # test double
            user_api_key_cache=cache,
            llm_router=None,
            request_model="gpt-4o",
            route="/chat/completions",
        )
    assert missing.value.code == "401"
    with pytest.raises(ProxyException):
        await resolve_customer_hierarchy(
            identity=CustomerRequestIdentity(customer_member_id="member-1"),
            prisma_client=None,
            user_api_key_cache=cache,
            llm_router=None,
            request_model="gpt-4o",
            route="/chat/completions",
        )


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
    assert hierarchy.customer_member_id == "member-1"


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
    assert (matching.customer_org_id, matching.customer_member_id) == ("org-1", "member-1")
    assert tuple(scope.key for scope in matching.scopes) == ("customer_org", "customer_team", "customer_member")

    with pytest.raises(ProxyException) as denied:
        await _resolve(prisma, CustomerRequestIdentity(customer_member_id="member-1", customer_org_alias="other"))
    assert (denied.value.code, denied.value.param) == ("403", CUSTOMER_ORG_ALIAS_HEADER)


@pytest.mark.asyncio
async def test_an_unknown_alias_is_refused_and_a_shared_alias_is_rejected_as_ambiguous() -> None:
    with pytest.raises(ProxyException) as missing:
        await _resolve(_Prisma(_MemberTable({})), CustomerRequestIdentity(customer_org_alias="ghost"))
    assert (missing.value.code, missing.value.param) == ("401", CUSTOMER_ORG_ALIAS_HEADER)

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
    assert (hierarchy.customer_org_id, hierarchy.customer_member_id) == ("org-1", "user-7")
    assert _scopes(hierarchy)["customer_member"].max_budget == 10.0


@pytest.mark.asyncio
async def test_an_unknown_member_stays_refused_without_an_alias_or_when_registration_fails() -> None:
    members: Final = _MemberTable({})
    registrar: Final = _Registrar(members)
    with pytest.raises(ProxyException) as no_alias:
        await _resolve(_Prisma(members), CustomerRequestIdentity(customer_member_id="user-7"), register=registrar)
    assert (no_alias.value.code, registrar.calls) == ("401", ())

    failing: Final = _Registrar(members, Failure(ErrorCode.BUDGET_REQUIRED))
    prisma: Final = _Prisma(members, _OrganizationTable((_organization("acme"),)))
    with pytest.raises(ProxyException) as failed:
        await _resolve(
            prisma, CustomerRequestIdentity(customer_member_id="user-7", customer_org_alias="acme"), register=failing
        )
    assert (failed.value.code, failing.calls) == ("401", (("org-1", "user-7"),))
    assert "user-7" not in members.rows
