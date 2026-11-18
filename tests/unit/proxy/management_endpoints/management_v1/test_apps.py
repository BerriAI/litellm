import asyncio
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
    http, _ = client()
    assert http.get("/management/v1/apps/definition").json() == {"resource_type": None}
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
    assert http.delete(f"/management/v1/apps/{app_id}").status_code == 204
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
