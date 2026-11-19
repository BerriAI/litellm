import asyncio
from types import SimpleNamespace
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.apps import get_app_store, router
from litellm.proxy.management_helpers.apps import AppStore
from tests.unit.proxy.management_helpers.test_apps import (
    AppDatabase,
    profile,
    seed_resources,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget
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
    assert http.put("/management/v1/apps/definition", json={"resource_type": "team"}).status_code == 403


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
    asyncio.run(
        store.customers.organizations.create({"customer_org_id": "org-a", "created_by": "admin", "updated_by": "admin"})
    )
    asyncio.run(
        store.customers.organizations.create({"customer_org_id": "org-b", "created_by": "admin", "updated_by": "admin"})
    )
    asyncio.run(
        store.customers.members.create(
            {
                "customer_member_id": "member",
                "parent_org_id": "org-a",
                "parent_team_ids": [],
                "created_by": "admin",
                "updated_by": "admin",
            }
        )
    )
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
    assert http.put(chat_org, json={"budget_id": "unknown"}).status_code == 422
    invalid: Final = http.put(chat_member, json={"budget": {"max_budget": -1}})
    assert invalid.status_code == 422
    assert http.get(chat_member).json()["budget"] == permanent.json()["budget"]
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
