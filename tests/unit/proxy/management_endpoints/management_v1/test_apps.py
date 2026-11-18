from typing import TYPE_CHECKING, Final, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.apps import get_app_store, router
from litellm.proxy.management_helpers.apps import AppStore
from tests.unit.proxy.management_helpers.test_apps import AppDatabase
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW

if TYPE_CHECKING:
    from prisma import Prisma


def client(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    app: Final = FastAPI()
    app.add_exception_handler(ManagementProblem, lambda request, error: problem_response(error.problem))
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_role=role, user_id="admin")
    app.dependency_overrides[get_app_store] = lambda: store
    return TestClient(app)


def test_apps_crud_and_empty_listing() -> None:
    http: Final = client()
    assert http.get("/management/v1/apps").json() == []
    created: Final = http.post("/management/v1/apps", json={"alias": "assistant", "budget": {"max_budget": 20}})
    assert created.status_code == 201, created.text
    app_id: Final = created.json()["app_id"]
    assert created.json()["budget"]["max_budget"] == 20
    info: Final = http.get(f"/management/v1/apps/{app_id}")
    assert info.status_code == 200 and info.json()["alias"] == "assistant", info.text
    changed: Final = http.patch(f"/management/v1/apps/{app_id}", json={"alias": "new-alias"})
    assert changed.status_code == 200 and changed.json()["alias"] == "new-alias", changed.text
    removed: Final = http.delete(f"/management/v1/apps/{app_id}")
    assert removed.status_code == 204, removed.text
    assert http.get(f"/management/v1/apps/{app_id}").status_code == 404


@pytest.mark.parametrize("role", (LitellmUserRoles.TEAM, LitellmUserRoles.ORG_ADMIN, LitellmUserRoles.INTERNAL_USER))
def test_non_proxy_admins_cannot_manage_apps(role: LitellmUserRoles) -> None:
    http: Final = client(role)
    response: Final = http.post("/management/v1/apps", json={"alias": "assistant"})
    assert response.status_code == 403, response.text
    assert http.get("/management/v1/apps").status_code == 403


def test_tiers_are_managed_through_the_app_and_default_deletion_is_409() -> None:
    http: Final = client()
    app: Final = http.post("/management/v1/apps", json={"alias": "assistant"}).json()
    app_id: Final = app["app_id"]
    tier: Final = http.post(f"/management/v1/apps/{app_id}/tiers", json={"name": "standard", "max_budget": 10})
    assert tier.status_code == 201, tier.text
    budget_id: Final = tier.json()["budget_id"]
    patched: Final = http.patch(
        f"/management/v1/apps/{app_id}/tiers",
        json={"budget_id": budget_id, "max_budget": 15},
    )
    assert patched.status_code == 200 and patched.json()["max_budget"] == 15, patched.text
    default: Final = http.patch(f"/management/v1/apps/{app_id}", json={"default_tier_budget_id": budget_id})
    assert default.status_code == 200, default.text
    rejected: Final = http.request("DELETE", f"/management/v1/apps/{app_id}/tiers", json={"budget_id": budget_id})
    assert rejected.status_code == 409, rejected.text
    foreign: Final = http.patch(f"/management/v1/apps/{app_id}", json={"default_tier_budget_id": "foreign"})
    assert foreign.status_code == 422, foreign.text
