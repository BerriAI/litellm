from typing import TYPE_CHECKING, Final, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.customers import get_customer_hierarchy, router
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy
from tests.unit.proxy.management_helpers.test_customer_hierarchy import Database

if TYPE_CHECKING:
    from prisma import Prisma


def client(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    auth: Final = UserAPIKeyAuth(user_id="admin", user_role=role)
    hierarchy: Final = CustomerHierarchy(cast("Prisma", Database()))  # cast-ok: in-memory Prisma edge
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[get_customer_hierarchy] = lambda: hierarchy
    return TestClient(app)


@pytest.mark.parametrize("entity", ("customer_organizations", "customer_teams", "customer_members"))
@pytest.mark.parametrize("method", ("POST", "GET", "PATCH"))
@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_non_admin_is_forbidden(entity: str, method: str, role: LitellmUserRoles) -> None:
    http: Final = client(role)
    body: Final = {
        "customer_organizations": {},
        "customer_teams": {"parent_org_id": "org"},
        "customer_members": {"parent_team_id": "team"},
    }
    path: Final = f"/management/v1/{entity}" + ("" if method == "POST" else "/id")
    response: Final = http.request(method, path, json=body[entity] if method != "GET" else None)
    assert response.status_code == 403, response.text
    assert response.json()["status"] == 403


def test_admin_http_flow_preserves_member_copy_after_org_patch() -> None:
    http: Final = client()
    org: Final = http.post(
        "/management/v1/customer_organizations",
        json={"customer_org_id": "org", "default_budgets": [{"max_budget": 100, "rpm_limit": 10}]},
    )
    assert org.status_code == 200, org.text
    team: Final = http.post("/management/v1/customer_teams", json={"customer_team_id": "team", "parent_org_id": "org"})
    assert team.status_code == 200, team.text
    member: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "member", "parent_team_id": "team", "budget": {"rpm_limit": 3}},
    )
    assert member.status_code == 200, member.text
    assert member.json()["budget_id"] != org.json()["default_budget_id"]
    assert member.json()["budget"] == {
        "max_budget": 100,
        "tpm_limit": None,
        "rpm_limit": 3,
        "allowed_models": [],
        "budget_duration": None,
    }
    patched: Final = http.patch(
        "/management/v1/customer_organizations/org", json={"default_budgets": [{"max_budget": 200}]}
    )
    assert patched.status_code == 200, patched.text
    fetched: Final = http.get("/management/v1/customer_members/member")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == member.json()


@pytest.mark.parametrize(
    ("path", "body"),
    (
        ("customer_teams", {"parent_org_id": "missing"}),
        ("customer_members", {"parent_team_id": "missing"}),
    ),
)
def test_missing_parent_http_404(path: str, body: dict[str, str]) -> None:
    response: Final = client().post(f"/management/v1/{path}", json=body)
    assert response.status_code == 404, response.text
    assert response.json()["status"] == 404


def test_unknown_query_is_rejected() -> None:
    response: Final = client().post("/management/v1/customer_organizations?unexpected=value", json={})
    assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    "templates",
    (
        [{}, {}],
        [{"is_default": True}, {"is_default": True}],
    ),
)
@pytest.mark.parametrize("method", ("POST", "PATCH"))
def test_invalid_default_selection_returns_422_without_replacing_org(
    templates: list[dict[str, bool]], method: str
) -> None:
    http: Final = client()
    original: Final = http.post(
        "/management/v1/customer_organizations",
        json={"customer_org_id": "org", "default_budgets": [{"max_budget": 10}]},
    )
    assert original.status_code == 200, original.text
    path: Final = "/management/v1/customer_organizations" + ("/org" if method == "PATCH" else "")
    response: Final = http.request(method, path, json={"default_budgets": templates})
    assert response.status_code == 422, response.text
    fetched: Final = http.get("/management/v1/customer_organizations/org")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == original.json()


@pytest.mark.parametrize("templates", ([], [{}], [{"is_default": True}], [{}, {"is_default": True}]))
def test_org_response_reports_exactly_one_default_for_nonempty_set(templates: list[dict[str, bool]]) -> None:
    response: Final = client().post("/management/v1/customer_organizations", json={"default_budgets": templates})
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert len(body["default_budgets"]) == len(templates)
    defaults: Final = tuple(budget for budget in body["default_budgets"] if budget["is_default"])
    assert len(defaults) == (1 if templates else 0)
    assert body["default_budget_id"] == (defaults[0]["budget_id"] if templates else None)


@pytest.mark.parametrize(
    ("entity", "id_field", "body"),
    (
        ("customer_organizations", "customer_org_id", {}),
        ("customer_teams", "customer_team_id", {"parent_org_id": "org"}),
        ("customer_members", "customer_member_id", {"parent_team_id": "team"}),
    ),
)
def test_duplicate_caller_supplied_id_returns_409(entity: str, id_field: str, body: dict[str, str]) -> None:
    http: Final = client()
    assert http.post("/management/v1/customer_organizations", json={"customer_org_id": "org"}).status_code == 200
    assert (
        http.post(
            "/management/v1/customer_teams", json={"customer_team_id": "team", "parent_org_id": "org"}
        ).status_code
        == 200
    )
    created: Final = http.post(f"/management/v1/{entity}", json={**body, id_field: "duplicate"})
    assert created.status_code == 200, created.text
    duplicate: Final = http.post(f"/management/v1/{entity}", json={**body, id_field: "duplicate"})
    assert duplicate.status_code == 409, duplicate.text
    assert duplicate.json()["status"] == 409
