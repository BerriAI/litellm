from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Final,
    cast,
)  # noqa: TID251  # in-memory adapters implement the external Prisma edge
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.app_auth import app_scope, resolve_app_hierarchy
from litellm.proxy.auth.customer_hierarchy_auth import CustomerBudgetExceededError, check_customer_hierarchy_budget
from litellm.proxy.auth.customer_spend import customer_spend_counter
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.management_helpers.app_groups import AppGroupStore
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters
from litellm.proxy.spend_tracking.customer_cost_margin import key_customer_cost_margin, request_customer_cost_margin
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerScopeAuth,
)
from litellm.types.proxy.management_endpoints.app_groups import AppGroup, AppGroupBudgetPut, AppGroupCreate
from litellm.types.proxy.management_endpoints.apps import (
    App,
    AppCreate,
    AppDefinition,
    AppFailure,
    AppOrganizationPut,
    AppPatch,
    AppResourceType,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget
from tests.unit.proxy.management_helpers.test_apps import AppDatabase, UnavailableAppDatabase, seed_resources
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

if TYPE_CHECKING:
    from prisma import Prisma

    from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource_type", "legacy_key"),
    (
        (AppResourceType.TEAM, False),
        (AppResourceType.PROJECT, False),
        (AppResourceType.VIRTUAL_KEY, False),
        (AppResourceType.VIRTUAL_KEY, True),
    ),
)
async def test_cached_app_auth_preserves_identity_and_enforcement_during_database_outage(
    resource_type: AppResourceType, legacy_key: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    database: Final = AppDatabase()
    writer: Final = AppStore(database, NOW)
    await seed_resources(database)
    await writer.put_definition(AppDefinition(resource_type=resource_type))
    resource: Final = (await writer.resources(0, 10))[0]
    app: Final = await writer.create(AppCreate(resource_id=resource.resource_id, budget=Budget(max_budget=10)), "admin")
    assert isinstance(app, App)
    authenticated: Final = UserAPIKeyAuth(
        team_id="chat",
        team_alias="Chat",
        project_id="project",
        project_alias="Project",
        key_id=None if legacy_key else resource.resource_id,
        token="token-a",
        key_alias="Service key",
    )
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", customer_member_id="member", scopes=())
    cache, shared = shared_cache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)

    async def resolve(
        db: object, assertion: str, identity: UserAPIKeyAuth = authenticated
    ) -> CustomerHierarchyAuth | None:
        connection: Final = cast("PrismaClient", SimpleNamespace(db=db))  # cast-ok: external database edge
        return await resolve_app_hierarchy(
            hierarchy,
            {"x-litellm-app": assertion},
            {},
            connection,
            None,
            None,
            "/v1/chat/completions",
            identity,
        )

    warm: Final = await resolve(database, app.app_id)
    cold, _ = shared_cache(FakeAuthCache(redis=shared.redis, generations=shared.generations))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cold)
    offline: Final = await resolve(UnavailableAppDatabase(), app.app_id)
    assert offline == warm
    assert offline is not None and offline.customer_member_id == "member"
    assert tuple(scope.key for scope in offline.scopes) == ("app", "app_customer_org", "app_customer_member")
    with pytest.raises(ProxyException) as rejected:
        await resolve(UnavailableAppDatabase(), "different-app")
    assert int(rejected.value.code) == 403
    alias_field: Final = {
        AppResourceType.TEAM: "team_alias",
        AppResourceType.PROJECT: "project_alias",
        AppResourceType.VIRTUAL_KEY: "key_alias",
    }[resource_type]
    cleared: Final = authenticated.model_copy(update={alias_field: None})
    without_alias: Final = await resolve(UnavailableAppDatabase(), app.app_id, cleared)
    assert without_alias is not None and without_alias.app_alias == resource.resource_id
    with pytest.raises(ProxyException) as removed_alias:
        await resolve(UnavailableAppDatabase(), resource.alias, cleared)
    assert int(removed_alias.value.code) == 403
    if resource_type == AppResourceType.VIRTUAL_KEY and not legacy_key:
        fresh: Final = UserAPIKeyAuth.model_validate(cleared.model_dump(exclude_none=True))
        with pytest.raises(ProxyException) as omitted_alias:
            await resolve(UnavailableAppDatabase(), resource.alias, fresh)
        assert int(omitted_alias.value.code) == 403

    updater: Final = AppStore(database, NOW, cache=cold, cache_reads=False)
    await updater.patch(app.app_id, AppPatch(budget=Budget(max_budget=0)), "admin")
    await updater.invalidate_cache()
    with pytest.raises(CustomerBudgetExceededError):
        await resolve(database, app.app_id)
    with pytest.raises(CustomerBudgetExceededError):
        await resolve(UnavailableAppDatabase(), app.app_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", (False, True))
@pytest.mark.parametrize(
    "scope_key", ("app", "app_customer_org", "app_customer_member", "app_group", "app_group_customer_org")
)
async def test_cached_app_spend_rechecks_exhausted_budgets_and_preserves_outage_policy(
    scope_key: str, strict: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    database: Final = AppDatabase()
    writer: Final = AppStore(database, NOW)
    await seed_resources(database)
    await database.litellm_customerorganizationtable.create({"customer_org_id": "org"})
    await writer.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    groups: Final = AppGroupStore(writer)
    group: Final = await groups.create(AppGroupCreate(name="Product", budget=Budget(max_budget=10)), "admin")
    assert isinstance(group, AppGroup)
    await groups.put_organization(group.app_group_id, "org", AppGroupBudgetPut(budget=Budget(max_budget=10)), "admin")
    app: Final = await writer.create(
        AppCreate(
            resource_id="chat",
            budget=Budget(max_budget=10),
            app_group_id=group.app_group_id,
            default_customer_member_budget=Budget(max_budget=10),
        ),
        "admin",
    )
    assert isinstance(app, App)
    await writer.put_organization(app.app_id, "org", AppOrganizationPut(budget=Budget(max_budget=10)), "admin")
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: external database edge
    cache, _ = shared_cache()
    redis: Final[dict[str, object]] = {}
    spend: Final = FakeSpendCounters(redis=redis, redis_default=0)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", connection)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache())
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend.store())
    monkeypatch.setattr(proxy_server, "general_settings", {"fail_closed_budget_enforcement": strict})

    async def resolve(db: object = database) -> CustomerHierarchyAuth:
        result: Final = await resolve_app_hierarchy(
            CustomerHierarchyAuth(customer_org_id="org", customer_member_id="member", scopes=()),
            {},
            {},
            cast("PrismaClient", SimpleNamespace(db=db)),
            None,
            None,
            "/v1/chat/completions",
            UserAPIKeyAuth(team_id="chat", team_alias="Chat"),
        )  # cast-ok: external database edge
        assert result is not None
        return result

    warm: Final = await resolve()
    counter: Final = customer_spend_counter(next(scope for scope in warm.scopes if scope.key == scope_key))
    table: Final = {
        "app": database.litellm_apptable,
        "app_group": database.litellm_appgrouptable,
        "app_group_customer_org": database.litellm_appgroupcustomerorgtable,
        "app_customer_org": database.litellm_appcustomerorgtable,
        "app_customer_member": database.litellm_appcustomermembertable,
    }[scope_key]
    for identity, row in tuple(table.rows.items()):
        table.rows[identity] = row.model_copy(update={"spend": 12.0})
    redis[counter] = 5.0
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache())
    with pytest.raises(CustomerBudgetExceededError):
        await resolve()
    assert redis[counter] == 12

    offline: Final = UnavailableAppDatabase()
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=offline))
    with pytest.raises(CustomerBudgetExceededError):
        await resolve(offline)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache())
    redis[counter] = 5.0
    spend.memory.clear()
    admitted: Final = await resolve(offline)
    assert next(scope.spend for scope in admitted.scopes if scope.key == scope_key) == 5
    monkeypatch.setattr(proxy_server, "general_settings", {"fail_closed_budget_enforcement": True})
    spend.fail = True
    with pytest.raises(HTTPException) as rejected:
        await resolve(offline)
    assert rejected.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("with_customer", (False, True))
async def test_resolved_app_adds_only_scopes_with_resolved_customer_parents(
    with_customer: bool,
) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    app: Final = await store.create(AppCreate(resource_id="chat", budget=Budget(max_budget=10)), "admin")
    assert not isinstance(app, AppFailure)
    hierarchy: Final = (
        CustomerHierarchyAuth(
            customer_org_id="org",
            customer_member_id="member",
            scopes=(CustomerScopeAuth(key="customer_org", entity_id="org", spend=1, max_budget=100),),
        )
        if with_customer
        else None
    )
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    result: Final = await resolve_app_hierarchy(
        hierarchy,
        {} if with_customer else {"x-litellm-app": "Chat"},
        {},
        connection,
        None,
        "request-model",
        "/v1/chat/completions",
        UserAPIKeyAuth(team_id="chat"),
    )
    assert result is not None and result.app_id == app.app_id
    assert tuple(scope.key for scope in result.scopes) == (
        ("customer_org", "app", "app_customer_org", "app_customer_member") if with_customer else ("app",)
    )
    assert result.customer_org_id == ("org" if with_customer else None)


@pytest.mark.asyncio
async def test_assertion_header_cannot_select_another_authenticated_team() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    chat: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    search: Final = await store.create(AppCreate(resource_id="search"), "admin")
    assert isinstance(chat, App) and isinstance(search, App)
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    with pytest.raises(ProxyException) as rejected:
        await resolve_app_hierarchy(
            None,
            {"x-litellm-app": "Search"},
            {},
            connection,
            None,
            None,
            "/v1/chat/completions",
            UserAPIKeyAuth(team_id="chat"),
        )
    assert int(rejected.value.code) == 403
    admitted: Final = await resolve_app_hierarchy(
        None,
        {"x-litellm-app": "Chat"},
        {},
        connection,
        None,
        None,
        "/v1/chat/completions",
        UserAPIKeyAuth(team_id="chat"),
    )
    assert admitted is not None and admitted.app_id == chat.app_id


@pytest.mark.asyncio
async def test_unset_definition_preserves_global_behavior_even_with_app_headers() -> None:
    database: Final = AppDatabase()
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=())
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    result: Final = await resolve_app_hierarchy(
        hierarchy,
        {"x-litellm-app": "arbitrary"},
        {"app_alias": "spoof"},
        connection,
        None,
        None,
        "/v1/chat/completions",
        UserAPIKeyAuth(team_id="chat"),
    )
    assert result is hierarchy
    assert not database.litellm_apptable.rows


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", tuple(AppResourceType))
@pytest.mark.parametrize("assertion", (None, "Chat"))
async def test_an_app_assertion_requires_a_resource_on_the_authenticated_identity(
    resource_type: AppResourceType, assertion: str | None
) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await store.put_definition(AppDefinition(resource_type=resource_type))
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=())
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge

    async def resolve() -> CustomerHierarchyAuth | None:
        return await resolve_app_hierarchy(
            hierarchy,
            {"X-LiteLLM-App": assertion} if assertion is not None else {},
            {},
            connection,
            None,
            None,
            "/v1/chat/completions",
            UserAPIKeyAuth(),
        )

    if assertion is None:
        assert await resolve() is hierarchy
    else:
        with pytest.raises(ProxyException) as rejected:
            await resolve()
        assert int(rejected.value.code) == 403
        assert rejected.value.param == "x-litellm-app"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource_id", "assertion", "admitted"),
    (
        ("chat", None, True),
        ("chat", "chat", True),
        ("chat", "  Chat  ", True),
        ("chat", "Search", False),
        ("missing-team", "Chat", False),
    ),
)
async def test_unconfigured_apps_validate_assertions_against_the_authenticated_resource(
    resource_id: str, assertion: str | None, admitted: bool
) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=())
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge

    async def resolve() -> CustomerHierarchyAuth | None:
        return await resolve_app_hierarchy(
            hierarchy,
            {"x-litellm-app": assertion} if assertion is not None else {},
            {},
            connection,
            None,
            None,
            "/v1/chat/completions",
            UserAPIKeyAuth(team_id=resource_id),
        )

    if admitted:
        assert await resolve() is hierarchy
    else:
        with pytest.raises(ProxyException) as rejected:
            await resolve()
        assert int(rejected.value.code) == 403
    assert not database.litellm_apptable.rows


@pytest.mark.asyncio
async def test_an_app_budget_rejection_retains_trusted_attribution() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    app: Final = await store.create(AppCreate(resource_id="chat", budget=Budget(max_budget=0)), "admin")
    assert isinstance(app, App)
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="org",
        scopes=(CustomerScopeAuth(key="customer_org", entity_id="org", spend=0, max_budget=100),),
    )
    with pytest.raises(CustomerBudgetExceededError) as rejected:
        await resolve_app_hierarchy(
            hierarchy,
            {},
            {"app_id": "spoof"},
            connection,
            None,
            None,
            "/v1/chat/completions",
            UserAPIKeyAuth(team_id="chat"),
        )
    assert rejected.value.entity_type == "app"
    assert rejected.value.entity_id == app.app_id
    assert rejected.value.customer_hierarchy.app_id == app.app_id
    assert rejected.value.customer_hierarchy.customer_org_id == "org"
    assert tuple(scope.key for scope in rejected.value.customer_hierarchy.scopes) == (
        "customer_org",
        "app",
        "app_customer_org",
    )


@pytest.mark.asyncio
async def test_project_and_virtual_key_resolution_use_authenticated_resource_ids() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.PROJECT))
    project: Final = await store.create(AppCreate(resource_id="project"), "admin")
    assert isinstance(project, App)
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    result: Final = await resolve_app_hierarchy(
        None,
        {"x-litellm-app": "Project"},
        {},
        connection,
        None,
        None,
        "/v1/chat/completions",
        UserAPIKeyAuth(project_id="project", team_id="chat"),
    )
    assert result is not None and result.app_id == project.app_id


def test_app_org_denies_a_request_that_global_org_admits() -> None:
    global_scope: Final = CustomerScopeAuth(key="customer_org", entity_id="org", spend=4, max_budget=100)
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=(global_scope,))
    check_customer_hierarchy_budget(hierarchy)
    combined: Final = hierarchy.model_copy(
        update={
            "scopes": (
                global_scope,
                app_scope("app_customer_org", "app:org", 4, Budget(max_budget=3)),
            )
        }
    )
    with pytest.raises(litellm.BudgetExceededError) as rejected:
        check_customer_hierarchy_budget(combined)
    assert rejected.value.entity_type == "app_customer_org"
    assert hierarchy.scopes[0] == global_scope


@pytest.mark.parametrize("duration", ("24h", "1mo", "12mo"))
def test_app_duration_cap_uses_a_separate_current_window_not_lifetime_spend(
    duration: str,
) -> None:
    scope: Final = app_scope("app", "application", 1000, Budget(max_budget=10, budget_duration=duration))
    assert scope.max_budget is None
    counters: Final = customer_window_counters(scope.key, scope.entity_id, scope.budget_limits, now=NOW)
    assert len(counters) == 1 and counters[0].max_budget == 10
    assert counters[0].window_duration == duration
    assert counters[0].counter_key.startswith("spend:app:application:window:")
    assert counters[0].reset_at is not None
    next_period: Final = customer_window_counters(
        scope.key,
        scope.entity_id,
        scope.budget_limits,
        now=counters[0].reset_at,
    )
    assert next_period[0].counter_key != counters[0].counter_key


@pytest.mark.asyncio
async def test_no_app_returns_the_same_hierarchy_without_a_database() -> None:
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=())
    before: Final = hierarchy.model_dump_json()
    result: Final = await resolve_app_hierarchy(
        hierarchy,
        {},
        {},
        None,
        None,
        None,
        "/v1/chat/completions",
        UserAPIKeyAuth(team_id="chat"),
    )
    assert result is hierarchy and result.model_dump_json() == before
    assert "app_id" not in result.model_dump()


def test_spoofed_app_metadata_is_removed_when_auth_has_no_app() -> None:
    result: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        {
            "metadata": {
                "user_api_key_app_id": "spoof",
                "user_api_key_app_group_id": "spoof-group",
                "user_api_key_app_alias": "spoof",
            }
        },
        UserAPIKeyAuth(),
        "metadata",
    )
    assert "user_api_key_app_id" not in result["metadata"]
    assert "user_api_key_app_group_id" not in result["metadata"]
    assert "user_api_key_app_alias" not in result["metadata"]


def test_app_only_scopes_stamp_trusted_identity_without_a_customer_org() -> None:
    auth: Final = UserAPIKeyAuth(
        customer_hierarchy=CustomerHierarchyAuth(
            app_id="app",
            app_alias="assistant",
            scopes=(app_scope("app", "app", 0, Budget(max_budget=10)),),
        )
    )
    result: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        {"metadata": {"user_api_key_app_id": "spoof"}},
        auth,
        "metadata",
    )
    assert result["metadata"]["user_api_key_app_id"] == "app"
    assert result["metadata"]["user_api_key_app_alias"] == "assistant"
    assert result["metadata"]["user_api_key_customer_org_id"] is None


@pytest.mark.asyncio
async def test_virtual_key_app_identity_survives_rotation_and_ignores_app_alias_metadata() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    resource: Final = (await store.resources(0, 100, AppResourceType.VIRTUAL_KEY))[0]
    await store.put_definition(AppDefinition(resource_type=AppResourceType.VIRTUAL_KEY))
    app: Final = await store.create(AppCreate(resource_id=resource.resource_id), "admin")
    assert isinstance(app, App)
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=())
    before: Final = await resolve_app_hierarchy(
        hierarchy,
        {},
        {"app_alias": "spoof"},
        connection,
        None,
        None,
        "/v1/chat/completions",
        UserAPIKeyAuth(token="token-a"),
    )
    await database.litellm_verificationtoken.update(where={"token": "token-a"}, data={"token": "rotated-token"})
    after: Final = await resolve_app_hierarchy(
        hierarchy,
        {},
        {},
        connection,
        None,
        None,
        "/v1/chat/completions",
        UserAPIKeyAuth(token="rotated-token"),
    )
    assert before is not None and after is not None
    assert before.app_id == after.app_id == app.app_id


@pytest.mark.asyncio
@pytest.mark.parametrize(("customer_margin", "expected"), [(0.05, 0.05), (0, 0.0), (None, 0.02)])
async def test_a_key_bound_to_a_customer_gets_its_apps_surcharge_without_customer_headers(
    customer_margin: float | None, expected: float
) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    app: Final = await store.create(AppCreate(resource_id="chat", customer_margin=customer_margin), "admin")
    assert not isinstance(app, AppFailure)
    bound_key: Final = UserAPIKeyAuth(token="bound", team_id="chat", metadata={"customer_org_id": "org"})
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge

    resolved: Final = await resolve_app_hierarchy(
        None, {}, {}, connection, None, "request-model", "/v1/chat/completions", bound_key
    )

    assert resolved is not None and resolved.app_id == app.app_id
    with patch.dict(litellm.cost_margin_config, {"customer": 0.02}, clear=True):
        assert key_customer_cost_margin(bound_key, resolved) == expected
        settled: Final = {
            "user_api_key_customer_usage": '{"customer_org_id": "org"}',
            "user_api_key_customer_hierarchy": resolved.model_dump(mode="json"),
        }
        assert request_customer_cost_margin(settled) == expected


@pytest.mark.asyncio
async def test_a_key_with_no_customer_resolves_its_app_but_reserves_no_surcharge() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    await store.create(AppCreate(resource_id="chat", customer_margin=0.05), "admin")
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge

    resolved: Final = await resolve_app_hierarchy(
        None,
        {},
        {},
        connection,
        None,
        "request-model",
        "/v1/chat/completions",
        UserAPIKeyAuth(token="plain", team_id="chat"),
    )

    assert resolved is not None
    assert (resolved.app_alias, resolved.customer_org_id, resolved.app_customer_margin) == ("Chat", None, 0.05)
    with patch.dict(litellm.cost_margin_config, {"customer": 0.02}, clear=True):
        assert key_customer_cost_margin(UserAPIKeyAuth(token="plain", team_id="chat"), resolved) is None


@pytest.mark.parametrize("duration", ("1mo", "12mo", "7d"))
@pytest.mark.asyncio
async def test_soft_only_app_period_is_accounted_without_enforcing_the_warning(duration: str) -> None:
    from pydantic import TypeAdapter

    from litellm.proxy.proxy_server import customer_accounting_counters
    from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyIds, customer_budget_scopes_metadata

    scope: Final = app_scope("app", "application", 1000, Budget(soft_budget=5, budget_duration=duration))
    assert scope.max_budget is None and scope.budget_limits == ()
    snapshot: Final = customer_budget_scopes_metadata(CustomerHierarchyAuth(app_id="application", scopes=(scope,)))
    assert snapshot is not None
    restored: Final = TypeAdapter(tuple[CustomerScopeAuth, ...]).validate_json(snapshot)
    counters: Final = await customer_accounting_counters(
        CustomerHierarchyIds(app_id="application"), restored, started_at=NOW
    )
    assert tuple((counter.entity_type, counter.entity_id, counter.window_duration) for counter in counters) == (
        ("app", "application", duration),
    )


@pytest.mark.asyncio
async def test_moving_app_changes_new_requests_and_preserves_the_inflight_group_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server
    from litellm.types.proxy.auth.customer_hierarchy import customer_hierarchy_ids_from_request

    database: Final = AppDatabase()
    cache, _ = shared_cache()
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    store: Final = AppStore(database, NOW, cache=cache, cache_reads=False)
    groups: Final = AppGroupStore(store)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    first: Final = await groups.create(AppGroupCreate(name="First", budget=Budget(max_budget=100)), "admin")
    second: Final = await groups.create(AppGroupCreate(name="Second", budget=Budget(max_budget=100)), "admin")
    assert isinstance(first, AppGroup) and isinstance(second, AppGroup)
    app: Final = await store.create(AppCreate(resource_id="chat", app_group_id=first.app_group_id), "admin")
    assert isinstance(app, App)
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge

    async def resolve() -> CustomerHierarchyAuth:
        result: Final = await resolve_app_hierarchy(
            None, {}, {}, connection, None, None, "/v1/chat/completions", UserAPIKeyAuth(team_id="chat")
        )
        assert result is not None
        return result

    inflight: Final = await resolve()
    await store.patch(app.app_id, AppPatch(app_group_id=second.app_group_id), "admin")
    await store.invalidate_cache()
    later: Final = await resolve()
    assert inflight.app_group_id == first.app_group_id
    assert later.app_group_id == second.app_group_id
    stamped: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        {"metadata": {"user_api_key_app_group_id": "spoof"}},
        UserAPIKeyAuth(customer_hierarchy=inflight),
        "metadata",
    )
    identity: Final = customer_hierarchy_ids_from_request({"litellm_params": stamped})
    assert identity is not None and identity.app_group_id == first.app_group_id
