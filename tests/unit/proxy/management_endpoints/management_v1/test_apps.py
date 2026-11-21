import asyncio
from types import SimpleNamespace
from typing import Final, cast

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.apps import get_app_store, router
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.management_helpers.customer_hierarchy import BudgetGroupScope
from litellm.redis.stores.auth_cache import AuthObject
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget
from tests.unit.proxy.management_helpers.test_apps import (
    AppDatabase,
    profile,
    seed_resources,
)
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW


def client(
    role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN,
) -> tuple[TestClient, AppStore]:
    app: Final = FastAPI()
    app.add_exception_handler(ManagementProblem, lambda request, error: problem_response(error.problem))
    database: Final = AppDatabase()
    asyncio.run(seed_resources(database))
    store: Final = AppStore(database, NOW)
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id="admin")
    app.dependency_overrides[get_app_store] = lambda: store
    return TestClient(app), store


@pytest.mark.parametrize("invalid", ("missing", "private", "organization", "app_organization", "conflicting"))
def test_app_presets_reject_unrelated_or_conflicting_assignments(invalid: str) -> None:
    http, store = client()
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 200
    app: Final = http.post("/management/v1/apps", json={"resource_id": "chat"}).json()
    scopes: Final = {
        "organization": BudgetGroupScope.ORGANIZATION,
        "app_organization": BudgetGroupScope.APP_ORGANIZATION,
        "conflicting": BudgetGroupScope.APP,
    }
    group: Final = (
        asyncio.run(profile(store, "Standard", 10, scopes[invalid]))
        if invalid in scopes
        else asyncio.run(store.customers.create_budget(Budget(max_budget=10), "admin"))
        if invalid == "private"
        else "missing"
    )
    body: Final = {"budget_id": group, **({"budget": {"max_budget": 5}} if invalid == "conflicting" else {})}
    rejected: Final = http.patch(f"/management/v1/apps/{app['app_id']}", json=body)
    assert rejected.status_code == 422, rejected.text
    assert http.post("/management/v1/apps", json={"resource_id": "search", **body}).status_code == 422
    assert http.get(f"/management/v1/apps/{app['app_id']}").json()["budget_id"] is None


@pytest.mark.parametrize(
    ("method", "suffix", "field"),
    (
        ("PATCH", "", "default_customer_member_budget"),
        ("PUT", "/customer_organizations/org", "customer_member_budget"),
        ("PUT", "/customer_members/member", "budget"),
    ),
)
def test_member_presets_link_fork_and_inherit_without_changing_shared_policy(
    method: str, suffix: str, field: str
) -> None:
    http, store = client()
    asyncio.run(store.customers.organizations.create({"customer_org_id": "org"}))
    asyncio.run(store.customers.members.create({"customer_member_id": "member", "parent_org_id": "org"}))
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 200
    group: Final = asyncio.run(profile(store, "Member Standard", 100, BudgetGroupScope.APP_MEMBER))
    policy: Final = Budget(
        max_budget=100, budget_duration="1mo", tpm_limit=1000, rpm_limit=10, soft_budget=80, allowed_models=("model-a",)
    )
    asyncio.run(store.customers.update_budget(group, policy, "admin"))
    app: Final = http.post(
        "/management/v1/apps", json={"resource_id": "chat", "default_customer_member_budget_id": group}
    ).json()
    path: Final = f"/management/v1/apps/{app['app_id']}{suffix}"
    member_path: Final = f"/management/v1/apps/{app['app_id']}/customer_members/member"
    private: Final = asyncio.run(store.customers.create_budget(Budget(max_budget=10), "admin"))
    unrelated: Final = asyncio.run(profile(store, "Org group", 10))
    for invalid in (private, unrelated, "missing"):
        rejected: Final = http.request(method, path, json={f"{field}_id": invalid})
        assert rejected.status_code == 422, rejected.text
    assert http.request(method, path, json={f"{field}_id": group, field: {"max_budget": 5}}).status_code == 422
    assigned: Final = http.request(method, path, json={f"{field}_id": group})
    assert assigned.status_code == 200, assigned.text
    assert assigned.json()[f"{field}_id"] == group
    assert http.request(method, path, json={}).json()[f"{field}_id"] == group
    asyncio.run(store.customers.update_budget(group, Budget(max_budget=120), "admin"))
    assert http.get(member_path).json()["effective_budget"]["max_budget"] == 120
    customized: Final = http.request(method, path, json={field: {"max_budget": 200}}).json()
    private_id: Final = customized[f"{field}_id"]
    assert private_id is not None and private_id != group
    expected: Final = policy.model_copy(update={"max_budget": 200})
    assert asyncio.run(store.budget(private_id)) == expected
    assert asyncio.run(store.budget(group)) == expected.model_copy(update={"max_budget": 120})
    increased: Final = http.put(
        member_path,
        json={"budget_override": {"max_budget": 300}, "budget_override_expires_at": NOW.replace(year=2028).isoformat()},
    ).json()
    assert increased["effective_budget"]["max_budget"] == 300
    http.request(method, path, json={field: {"max_budget": 250}})
    assert http.get(member_path).json()["effective_budget"]["max_budget"] == 300
    assert http.get(member_path).json()["budget_override_id"] == increased["budget_override_id"]
    expired: Final = http.put(member_path, json={"budget_override_expires_at": NOW.replace(year=2020).isoformat()})
    assert expired.json()["effective_budget"]["max_budget"] == 250
    restored: Final = http.request(method, path, json={f"{field}_id": group})
    assert restored.json()[f"{field}_id"] == group and asyncio.run(store.budget(private_id)) is None
    cleared: Final = http.request(method, path, json={f"{field}_id": None})
    assert cleared.json()[f"{field}_id"] is None
    assert http.get(member_path).json()["effective_budget"] == (
        None if not suffix else expected.model_copy(update={"max_budget": 120}).model_dump(mode="json")
    )
    assert asyncio.run(store.budget(group)) == expected.model_copy(update={"max_budget": 120})


GRANT_ROUTES: Final = ("user", "team", "organization")


async def grant_apps(database: AppDatabase, route: str, app_ids: tuple[str, ...]) -> None:
    """Give user "admin" the apps through their own, their team's or their organization's object permission."""
    await database.litellm_objectpermissiontable.create(
        data={"object_permission_id": "grant", "apps": list(app_ids), "mcp_servers": ["unrelated"]}
    )
    await database.litellm_usertable.create(
        data={
            "user_id": "admin",
            "teams": ["owners"] if route == "team" else [],
            "object_permission_id": "grant" if route == "user" else None,
        }
    )
    await database.litellm_teamtable.create(
        data={"team_id": "owners", "object_permission_id": "grant" if route == "team" else None}
    )
    await database.litellm_organizationtable.create(
        data={"organization_id": "division", "object_permission_id": "grant" if route == "organization" else None}
    )
    if route == "organization":
        await database.litellm_organizationmembership.create(data={"organization_id": "division", "user_id": "admin"})


def two_apps(role: LitellmUserRoles, route: str | None) -> tuple[TestClient, str, str]:
    admin, store = client()
    assert admin.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 200
    chat: Final = admin.post("/management/v1/apps", json={"resource_id": "chat"}).json()["app_id"]
    search: Final = admin.post("/management/v1/apps", json={"resource_id": "search"}).json()["app_id"]
    if route is not None:
        asyncio.run(grant_apps(cast(AppDatabase, store.db), route, (chat,)))  # cast-ok: the store wraps this fake
    grantee: Final = FastAPI()
    grantee.add_exception_handler(ManagementProblem, lambda request, error: problem_response(error.problem))
    grantee.include_router(router)
    grantee.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id="admin")
    grantee.dependency_overrides[get_app_store] = lambda: store
    return TestClient(grantee), chat, search


@pytest.mark.parametrize("route", GRANT_ROUTES)
def test_a_grantee_reads_only_the_apps_granted_to_them_their_team_or_their_organization(route: str) -> None:
    http, chat, search = two_apps(LitellmUserRoles.INTERNAL_USER, route)
    listed: Final = http.get("/management/v1/apps")
    assert listed.status_code == 200, listed.text
    assert [app["app_id"] for app in listed.json()] == [chat]
    assert http.get(f"/management/v1/apps/{chat}").json()["app_id"] == chat
    assert http.get(f"/management/v1/apps/{search}").status_code == 403
    assert http.get(f"/management/v1/apps/{search}/customer_members/anyone").status_code == 403
    assert http.get(f"/management/v1/apps/{search}/customer_organizations/anyone").status_code == 403
    assert http.get("/management/v1/apps/definition").json()["resource_type"] == "team"


@pytest.mark.parametrize("route", GRANT_ROUTES)
def test_a_grantee_cannot_change_an_app_they_can_read(route: str) -> None:
    http, chat, _ = two_apps(LitellmUserRoles.INTERNAL_USER, route)
    assert http.patch(f"/management/v1/apps/{chat}", json={"budget": {"max_budget": 1}}).status_code == 403
    assert http.post("/management/v1/apps", json={"resource_id": "chat"}).status_code == 403
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 403
    assert http.get("/management/v1/apps/resources", params={"resource_type": "team"}).status_code == 403
    assert http.put(f"/management/v1/apps/{chat}/customer_members/anyone", json={}).status_code == 403
    assert http.put(f"/management/v1/apps/{chat}/customer_organizations/anyone", json={}).status_code == 403


def test_a_grantee_reads_only_customer_members_known_to_their_app() -> None:
    admin, store = client()
    asyncio.run(store.customers.organizations.create({"customer_org_id": "org-a"}))
    asyncio.run(store.customers.organizations.create({"customer_org_id": "org-b"}))
    asyncio.run(store.customers.members.create({"customer_member_id": "ours", "parent_org_id": "org-a"}))
    asyncio.run(store.customers.members.create({"customer_member_id": "theirs", "parent_org_id": "org-b"}))
    assert admin.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 200
    chat: Final = admin.post("/management/v1/apps", json={"resource_id": "chat"}).json()["app_id"]
    assert admin.put(f"/management/v1/apps/{chat}/customer_organizations/org-a", json={}).status_code == 200
    asyncio.run(grant_apps(cast(AppDatabase, store.db), "user", (chat,)))  # cast-ok: the store wraps this fake
    grantee: Final = FastAPI()
    grantee.add_exception_handler(ManagementProblem, lambda request, error: problem_response(error.problem))
    grantee.include_router(router)
    grantee.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="admin"
    )
    grantee.dependency_overrides[get_app_store] = lambda: store
    http: Final = TestClient(grantee)
    members: Final = f"/management/v1/apps/{chat}/customer_members"
    assert http.get(f"{members}/ours").json()["customer_member_id"] == "ours"
    unrelated: Final = http.get(f"{members}/theirs")
    unknown: Final = http.get(f"{members}/nobody")
    assert (
        (unrelated.status_code, unrelated.json())
        == (unknown.status_code, unknown.json())
        == (
            404,
            {"detail": "Customer member not found."},
        )
    )
    assert admin.get(f"{members}/theirs").status_code == 200


def test_a_user_without_a_grant_reads_no_apps() -> None:
    http, chat, _ = two_apps(LitellmUserRoles.INTERNAL_USER, None)
    assert http.get("/management/v1/apps").status_code == 403
    assert http.get(f"/management/v1/apps/{chat}").status_code == 403


def test_a_proxy_admin_reads_every_app_without_a_grant() -> None:
    http, chat, search = two_apps(LitellmUserRoles.PROXY_ADMIN, None)
    assert sorted(app["app_id"] for app in http.get("/management/v1/apps").json()) == sorted((chat, search))


def test_choose_definition_discover_and_configure_existing_resource() -> None:
    http, store = client()
    assert http.get("/management/v1/apps/definition").json() == {
        "resource_type": None,
        "auto_create_customers_without_app": True,
    }
    assert http.get("/management/v1/apps/resources").json() == []
    preview: Final = http.get("/management/v1/apps/resources", params={"resource_type": "team"})
    assert preview.status_code == 200 and len(preview.json()) == 2
    rejected: Final = http.post("/management/v1/apps", json={"resource_id": "chat"})
    assert rejected.status_code == 409
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 200
    created: Final = http.post(
        "/management/v1/apps",
        json={"resource_id": "chat", "budget": {"max_budget": 20}},
    )
    assert created.status_code == 201, created.text
    app_id: Final = created.json()["app_id"]
    assert created.json()["alias"] == "Chat" and created.json()["resource_type"] == "team"
    assert created.json()["budget"]["max_budget"] == 20
    changed: Final = http.patch(f"/management/v1/apps/{app_id}", json={"budget": {"max_budget": 30}})
    assert changed.status_code == 200 and changed.json()["budget"]["max_budget"] == 30
    again: Final = http.post("/management/v1/apps", json={"resource_id": "chat"})
    assert again.status_code == 201 and again.json()["app_id"] == app_id
    assert http.put("/management/v1/apps/definition", json={"resource_type": "project"}).status_code == 409
    asyncio.run(store.apps.delete(where={"app_id": app_id}))
    assert http.get(f"/management/v1/apps/{app_id}").status_code == 404
    assert http.put("/management/v1/apps/definition", json={"resource_type": None}).status_code == 409


@pytest.mark.parametrize(
    "role",
    (LitellmUserRoles.TEAM, LitellmUserRoles.ORG_ADMIN, LitellmUserRoles.INTERNAL_USER),
)
def test_only_proxy_admins_can_discover_or_manage_app_budgets(
    role: LitellmUserRoles,
) -> None:
    http, _ = client(role)
    assert http.post("/management/v1/apps", json={"resource_id": "chat"}).status_code == 403
    assert http.get("/management/v1/apps").status_code == 403
    assert http.get("/management/v1/apps/definition").status_code == 403
    assert http.get("/management/v1/apps/resources", params={"resource_type": "virtual_key"}).status_code == 403
    assert http.get("/management/v1/apps/resources/team/chat/models").status_code == 403
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 403


def model_client(
    monkeypatch: pytest.MonkeyPatch,
    team_models: list[str],
    key_models: list[str],
    project_models: list[str],
) -> tuple[TestClient, UserApiKeyCache]:
    from litellm import Router
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_ProjectTableCachedObj, LiteLLM_TeamTableCachedObj
    from litellm.proxy.utils import PrismaClient

    http, store = client()
    cache: Final = UserApiKeyCache()
    team: Final = LiteLLM_TeamTableCachedObj(team_id="chat", models=team_models)
    project: Final = LiteLLM_ProjectTableCachedObj(project_id="project", team_id="chat", models=project_models)
    key: Final = UserAPIKeyAuth(
        token="token-a", team_id="chat", team_models=team_models, models=key_models, project_id="project"
    )
    cache.set_cache(AuthObject.team("chat"), team, model_type=LiteLLM_TeamTableCachedObj)
    cache.set_cache(AuthObject.project("project"), project, model_type=LiteLLM_ProjectTableCachedObj)
    cache.set_cache(AuthObject.virtual_key("token-a"), key, model_type=UserAPIKeyAuth)
    asyncio.run(store.keys.update(where={"token": "token-a"}, data={"key_id": "key-id"}))
    asyncio.run(store.projects.update(where={"project_id": "project"}, data={"team_id": "chat"}))
    router_models: Final = [
        {"model_name": name, "litellm_params": {"model": "openai/test-model", "api_key": "test"}}
        for name in ("alpha", "beta", "other")
    ]
    monkeypatch.setattr(proxy_server, "llm_router", Router(model_list=router_models))
    monkeypatch.setattr(proxy_server, "llm_model_list", router_models)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(
        proxy_server,
        "prisma_client",
        PrismaClient(
            database_url="postgresql://unused:unused@localhost/unused", proxy_logging_obj=proxy_server.proxy_logging_obj
        ),
    )
    return http, cache


@pytest.mark.parametrize(
    ("resource", "team_models", "key_models", "project_models", "expected"),
    (
        ("team/chat", ["alpha"], [], [], ["alpha"]),
        ("team/chat", [], [], [], ["alpha", "beta", "other"]),
        ("team/chat", ["no-default-models"], [], [], []),
        ("virtual_key/key-id", ["alpha", "beta"], ["alpha"], [], ["alpha"]),
        ("virtual_key/key-id", ["alpha"], ["beta"], [], []),
        ("virtual_key/key-id", ["alpha", "beta"], ["all-team-models"], ["beta"], ["beta"]),
        ("virtual_key/key-id", ["a*"], ["*"], [], ["alpha"]),
        ("project/project", ["alpha", "beta"], [], ["beta", "other"], ["beta"]),
    ),
)
def test_app_model_options_follow_resource_permissions(
    monkeypatch: pytest.MonkeyPatch,
    resource: str,
    team_models: list[str],
    key_models: list[str],
    project_models: list[str],
    expected: list[str],
) -> None:
    http, _ = model_client(monkeypatch, team_models, key_models, project_models)
    response: Final = http.get(f"/management/v1/apps/resources/{resource}/models")
    assert response.status_code == 200, response.text
    assert sorted(response.json()) == expected
    assert http.get("/management/v1/apps/resources/team/missing/models").status_code == 404


@pytest.mark.parametrize("resource", ("team/chat", "project/project", "virtual_key/key-id"))
@pytest.mark.parametrize("unavailable", (False, True))
def test_app_model_lookup_errors_describe_the_resource_not_the_admin_session(
    monkeypatch: pytest.MonkeyPatch, resource: str, unavailable: bool
) -> None:
    from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded
    from tests.unit.proxy.rust_auth_stores import FakeKeyStore, FakeTeamStore, RustAuthStores, install_rust_auth_stores

    http, cache = model_client(monkeypatch, [], [], [])
    cache.flush_cache()
    error: Final = DBLookupDeadlineExceeded("app resource", 1)
    install_rust_auth_stores(
        monkeypatch,
        RustAuthStores(
            key=FakeKeyStore(rows={"token-a": error} if unavailable else {}),
            team=FakeTeamStore(rows={"chat": error} if unavailable else {}),
        ),
    )
    response: Final = http.get(f"/management/v1/apps/resources/{resource}/models")
    assert response.status_code == (503 if unavailable else 404), response.text
    assert response.json() == {
        "detail": "Unable to load app models. Try again." if unavailable else "App resource or its parent not found."
    }


@pytest.mark.parametrize("resource_type", ("key", "application", "unknown"))
def test_definition_rejects_unknown_resource_types(resource_type: str) -> None:
    http, _ = client()
    assert http.put("/management/v1/apps/definition", json={"resource_type": resource_type}).status_code == 422
    assert http.get("/management/v1/apps/resources", params={"resource_type": resource_type}).status_code == 422


def test_independent_catalog_creation_and_alias_mutations_are_rejected() -> None:
    http, _ = client()
    http.put("/management/v1/apps/definition", json={"resource_type": "team"})
    assert http.post("/management/v1/apps", json={"alias": "new-app"}).status_code == 422
    assert http.post("/management/v1/apps", json={"resource_id": "missing"}).status_code == 404
    created: Final = http.post("/management/v1/apps", json={"resource_id": "chat"}).json()
    assert http.patch(f"/management/v1/apps/{created['app_id']}", json={"alias": "spoof"}).status_code == 422
    assert http.post(f"/management/v1/apps/{created['app_id']}/tiers", json={"name": "new-tier"}).status_code == 404


def test_app_defaults_reuse_existing_global_budget_groups() -> None:
    http, store = client()
    group: Final = asyncio.run(profile(store, "Standard", 10))
    http.put("/management/v1/apps/definition", json={"resource_type": "team"})
    created: Final = http.post(
        "/management/v1/apps",
        json={"resource_id": "chat", "default_tier_budget_id": group},
    )
    assert created.status_code == 201 and created.json()["default_tier_budget_id"] == group
    unknown: Final = http.patch(
        f"/management/v1/apps/{created.json()['app_id']}",
        json={"default_tier_budget_id": "unknown"},
    )
    assert unknown.status_code == 422


def test_virtual_key_discovery_never_returns_credentials_or_hashes() -> None:
    http, _ = client()
    response: Final = http.get("/management/v1/apps/resources", params={"resource_type": "virtual_key"})
    assert response.status_code == 200
    resource: Final = response.json()[0]
    assert set(resource) == {"resource_type", "resource_id", "alias", "app_id"}
    assert resource["resource_type"] == "virtual_key" and resource["resource_id"] != "token-a"
    assert "token-a" not in response.text


def test_list_and_get_apps_return_saved_budgets_and_period_usage() -> None:
    http, _ = client()
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 200
    created: Final = http.post(
        "/management/v1/apps",
        json={"resource_id": "chat", "budget": {"max_budget": 20, "budget_duration": "1mo"}},
    )
    assert created.status_code == 201, created.text
    saved: Final = created.json()
    assert saved["budget"] == Budget(max_budget=20, budget_duration="1mo", allowed_models=()).model_dump(mode="json")
    assert saved["budget_usage"][0]["spend"] == 0
    assert saved["budget_usage"][0]["max_budget"] == 20
    assert http.get(f"/management/v1/apps/{saved['app_id']}").json() == saved
    listed: Final = http.get("/management/v1/apps", params={"offset": 0, "limit": 1})
    assert listed.status_code == 200 and listed.json() == [saved]
    assert http.get("/management/v1/apps", params={"offset": 1, "limit": 1}).json() == []


def test_customer_and_member_budget_endpoints_keep_apps_independent() -> None:
    http, store = client()
    asyncio.run(store.customers.organizations.create({"customer_org_id": "org-a"}))
    asyncio.run(store.customers.organizations.create({"customer_org_id": "org-b"}))
    asyncio.run(store.customers.members.create({"customer_member_id": "member", "parent_org_id": "org-a"}))
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 200
    chat: Final = http.post("/management/v1/apps", json={"resource_id": "chat"}).json()["app_id"]
    search: Final = http.post(
        "/management/v1/apps",
        json={"resource_id": "search", "default_customer_member_budget": {"max_budget": 20}},
    ).json()["app_id"]
    chat_org: Final = f"/management/v1/apps/{chat}/customer_organizations/org-a"
    search_org: Final = f"/management/v1/apps/{search}/customer_organizations/org-a"
    missing: Final = http.get(chat_org)
    assert missing.status_code == 404 and missing.json() == {"detail": "App customer organization not found."}
    chat_saved: Final = http.put(
        chat_org, json={"budget_override": {"max_budget": 3}, "customer_member_budget": {"max_budget": 2}}
    )
    search_saved: Final = http.put(search_org, json={"budget_override": {"max_budget": 8}})
    assert chat_saved.status_code == 200, chat_saved.text
    assert search_saved.status_code == 200, search_saved.text
    assert chat_saved.json()["effective_budget"] == Budget(max_budget=3, allowed_models=()).model_dump(mode="json")
    assert search_saved.json()["effective_budget"] == Budget(max_budget=8, allowed_models=()).model_dump(mode="json")
    assert http.get(chat_org).json() == chat_saved.json()
    assert http.get(search_org).json() == search_saved.json()
    other_org: Final = http.put(
        f"/management/v1/apps/{chat}/customer_organizations/org-b", json={"budget_override": {"max_budget": 9}}
    )
    assert other_org.status_code == 200, other_org.text
    assert http.get(f"/management/v1/apps/{chat}/customer_organizations/org-b").json() == other_org.json()
    assert http.get(chat_org).json() == chat_saved.json()
    chat_member: Final = f"/management/v1/apps/{chat}/customer_members/member"
    search_member: Final = f"/management/v1/apps/{search}/customer_members/member"
    inherited: Final = http.get(chat_member)
    assert inherited.status_code == 200
    assert inherited.json()["effective_budget"] == Budget(max_budget=2, allowed_models=()).model_dump(mode="json")
    assert inherited.json()["budget_source"] == "customer_default"
    override: Final = http.put(search_member, json={"budget_override": {"max_budget": 1}})
    assert override.status_code == 200, override.text
    assert override.json()["effective_budget"] == Budget(max_budget=1, allowed_models=()).model_dump(mode="json")
    assert override.json()["budget_source"] == "member_override"
    assert http.get(search_member).json() == override.json()
    assert http.get(chat_member).json() == inherited.json()
    restored: Final = http.put(search_member, json={"budget_override": None})
    assert restored.status_code == 200
    assert restored.json()["effective_budget"] == Budget(max_budget=20, allowed_models=()).model_dump(mode="json")
    assert restored.json()["budget_source"] == "app_default"
    permanent: Final = http.put(chat_member, json={"budget": {"max_budget": 7}})
    assert permanent.status_code == 200, permanent.text
    assert permanent.json()["budget"] == Budget(max_budget=7, allowed_models=()).model_dump(mode="json")
    assert http.get(search_member).json()["budget"] is None
    assert permanent.json()["budget_source"] == "member_budget"
    rejected: Final = http.put(chat_org, json={"budget_id": "unknown", "budget": {"max_budget": 7}})
    assert rejected.status_code == 422
    assert http.get(f"/management/v1/apps/{chat}/customer_members/missing").status_code == 404


@pytest.mark.parametrize("available", (True, False))
def test_app_definition_uses_database_or_reports_unavailability(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    from litellm.proxy import proxy_server

    database: Final = AppDatabase()
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=database) if available else None)
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"
    )
    with TestClient(app) as http:
        response: Final = http.get("/management/v1/apps/definition")
    assert response.status_code == (200 if available else 503)
    assert response.json() == (
        {"resource_type": None, "auto_create_customers_without_app": True}
        if available
        else {"detail": "Database unavailable."}
    )


@pytest.mark.parametrize("path", ("/management/v1/apps", "/management/v1/apps/definition", "/management/v1/apps/a"))
def test_unauthenticated_app_read_does_not_open_the_store(path: str) -> None:
    opened: Final[list[str]] = []  # mutable-ok: records store opens across the request

    def unauthenticated() -> UserAPIKeyAuth:
        raise HTTPException(401, "Unauthorized.")

    def opening_store() -> AppStore:
        opened.append(path)
        return AppStore(AppDatabase(), NOW)

    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = unauthenticated
    app.dependency_overrides[get_app_store] = opening_store

    response: Final = TestClient(app).get(path)

    assert (response.status_code, opened) == (401, []), response.text
