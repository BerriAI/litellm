from datetime import timedelta
from typing import TYPE_CHECKING, Final, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.customers import get_customer_hierarchy
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW, Database, Record

if TYPE_CHECKING:
    from prisma import Prisma


def client(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    database: Final = Database()
    database.litellm_usertable.rows = {
        person: Record(user_id=person) for person in ("first", "second")
    }
    database.litellm_budgettable.rows = {
        "standard": Record(
            budget_id="standard",
            is_profile=True,
            budget_limits=(
                {"budget_duration": "1mo", "max_budget": 100},
                {"budget_duration": "12mo", "max_budget": 1000},
            ),
        ),
    }
    hierarchy: Final = CustomerHierarchy(
        cast("Prisma", database), clock=lambda: NOW
    )  # cast-ok: in-memory Prisma boundary
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    def identity() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_id="admin", user_role=role)

    def service() -> CustomerHierarchy:
        return hierarchy

    app.dependency_overrides[user_api_key_auth] = identity
    app.dependency_overrides[get_customer_hierarchy] = service
    return TestClient(app)


@pytest.mark.parametrize("subject_type", ("user", "customer_member"))
def test_profile_override_and_direct_increase_apply_to_one_person(subject_type: str) -> None:
    http: Final = client()
    if subject_type == "customer_member":
        assert http.post("/management/v1/customer_organizations", json={"customer_org_id": "org"}).status_code == 201
        assert http.post(
            "/management/v1/customer_teams", json={"customer_team_id": "team", "parent_org_id": "org"}
        ).status_code == 201
        for person in ("first", "second"):
            assert http.post(
                "/management/v1/customer_members",
                json={"customer_member_id": person, "parent_team_id": "team", "budget_id": "standard"},
            ).status_code == 201
    else:
        for person in ("first", "second"):
            assert http.patch(
                f"/management/v1/budget_assignments/user/{person}", json={"budget_id": "standard"}
            ).status_code == 200
    path: Final = f"/management/v1/budget_assignments/{subject_type}/first"
    override: Final = http.patch(
        path, json={"budget_override": {"budget_limits": [{"budget_duration": "1mo", "max_budget": 25}]}}
    )
    assert override.status_code == 200, override.text
    assert tuple(window["max_budget"] for window in override.json()["windows"]) == (25, 1000)
    increased: Final = http.patch(path, json={"budget_override": {
        "temp_budget_increase": 10,
        "temp_budget_expiry": (NOW + timedelta(hours=1)).isoformat(),
        "temp_budget_duration": "1mo",
    }})
    assert increased.status_code == 200, increased.text
    assert tuple(window["max_budget"] for window in increased.json()["windows"]) == (35, 1000)
    unchanged: Final = http.get(f"/management/v1/budget_assignments/{subject_type}/second")
    assert tuple(window["max_budget"] for window in unchanged.json()["windows"]) == (100, 1000)
    cleared: Final = http.patch(path, json={"budget_override": None})
    assert cleared.status_code == 200, cleared.text
    assert tuple(window["max_budget"] for window in cleared.json()["windows"]) == (100, 1000)


@pytest.mark.parametrize("override", (
    {"max_budget": 5},
    {"rpm_limit": 5},
    {"temp_budget_increase": 5},
    {"temp_budget_increase": 5, "temp_budget_duration": "1mo", "temp_budget_expiry": NOW.isoformat()},
    {"temp_budget_increase": 5, "temp_budget_duration": "1mo",
     "temp_budget_expiry": (NOW + timedelta(hours=1)).isoformat()},
))
def test_user_assignment_rejects_ineffective_limits_and_invalid_increases(override: dict[str, object]) -> None:
    http: Final = client()
    path: Final = "/management/v1/budget_assignments/user/first"
    rejected: Final = http.patch(path, json={"budget_override": override})
    assert rejected.status_code == 422, rejected.text
    assert http.get(path).json()["budget_override"] is None


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
@pytest.mark.parametrize("method", ("GET", "PATCH"))
def test_only_admins_can_read_or_change_person_budgets(role: LitellmUserRoles, method: str) -> None:
    response: Final = client(role).request(
        method, "/management/v1/budget_assignments/user/first", json={} if method == "PATCH" else None
    )
    assert response.status_code == 403
