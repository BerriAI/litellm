import asyncio
from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Final, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from prisma.errors import DataError, RawQueryError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import customer_member_cache_key, customer_org_alias_cache_key
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.customers import get_customer_hierarchy, get_user_api_key_cache
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy
from litellm.types.proxy.management_endpoints.customer_hierarchy import CustomerMember
from tests.unit.proxy.management_helpers.test_customer_hierarchy import (
    NOW,
    BoundedChildrenTable,
    CapacityErrorDatabase,
    Database,
    DeleteConflictTable,
    Record,
    Table,
)

if TYPE_CHECKING:
    from prisma import Prisma


def client(
    role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN,
    cache: UserApiKeyCache | None = None,
    database: Database | None = None,
    premium_user: bool = True,
    actor: str = "admin",
) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    auth: Final = UserAPIKeyAuth(user_id=actor, user_role=role)
    hierarchy: Final = CustomerHierarchy(
        cast("Prisma", database if database is not None else Database()), premium_user=premium_user, clock=lambda: NOW
    )  # cast-ok: in-memory Prisma edge
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[get_customer_hierarchy] = lambda: hierarchy
    app.dependency_overrides[get_user_api_key_cache] = lambda: cache if cache is not None else UserApiKeyCache()
    return TestClient(app)


@pytest.mark.parametrize("premium", (False, True))
def test_customer_portal_limits_and_free_tier_problem(premium: bool) -> None:
    http: Final = client(premium_user=premium)
    empty: Final = http.get("/management/v1/customer_portal/limits")
    assert empty.status_code == 200, empty.text
    assert empty.json() == {
        "limits_apply": not premium,
        "organization_limit": 5,
        "teams_per_organization_limit": 5,
        "members_per_team_limit": 5,
        "organization_count": 0,
        "any_limit_reached": False,
    }
    for index in range(5):
        response: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": f"org-{index}"})
        assert response.status_code == 201, response.text
    full: Final = http.get("/management/v1/customer_portal/limits")
    assert full.status_code == 200, full.text
    assert full.json() == {**empty.json(), "organization_count": 5, "any_limit_reached": not premium}
    sixth: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "sixth"})
    assert sixth.status_code == (201 if premium else 403), sixth.text
    if not premium:
        assert sixth.json()["detail"] == (
            "Free customer portal limit reached: up to 5 customer organizations, 5 teams per organization and "
            "5 members per team without a LiteLLM Enterprise license. Upgrade to Enterprise to remove the limits: "
            "https://www.litellm.ai/enterprise"
        )


@pytest.mark.parametrize("level", ("teams", "members"))
def test_customer_portal_limits_detect_full_children_before_organization_cap(level: str) -> None:
    http: Final = client(premium_user=False)
    family(http)
    for index in range(5):
        if level == "teams":
            response: Final = http.post(
                "/management/v1/customer_teams", json={"parent_org_id": "org", "customer_team_id": f"extra-{index}"}
            )
            assert response.status_code == (201 if index < 4 else 403), response.text
        else:
            member: Final = http.post(
                "/management/v1/customer_members",
                json={"customer_member_id": f"member-{index}", "parent_team_ids": ["team"]},
            )
            assert member.status_code == 201, member.text
    limits: Final = http.get("/management/v1/customer_portal/limits")
    assert limits.status_code == 200, limits.text
    assert limits.json()["organization_count"] == 1
    assert limits.json()["any_limit_reached"] is True


@pytest.mark.parametrize("premium", (False, True))
def test_member_upsert_respects_capacity_for_default_team_creation_and_added_membership(premium: bool) -> None:
    http: Final = client(premium_user=premium)
    family(http)
    for index in range(4):
        created: Final = http.post(
            "/management/v1/customer_teams", json={"parent_org_id": "org", "customer_team_id": f"extra-{index}"}
        )
        assert created.status_code == 201, created.text
    default_team: Final = http.post(
        "/management/v1/customer_members/upsert", json={"customer_org_id": "org", "customer_member_id": "defaulted"}
    )
    assert default_team.status_code == (200 if premium else 403), default_team.text
    for index in range(5):
        member: Final = http.post(
            "/management/v1/customer_members",
            json={"customer_member_id": f"full-{index}", "parent_team_ids": ["team"]},
        )
        assert member.status_code == 201, member.text
    original: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "moving", "parent_team_ids": ["extra-0"]}
    )
    assert original.status_code == 201, original.text
    for patch in ({}, {"alias": "renamed"}):
        kept: Final = http.post(
            "/management/v1/customer_members/upsert",
            json={"customer_org_id": "org", "customer_member_id": "moving", **patch},
        )
        assert kept.status_code == 200, kept.text
        assert kept.json()["created"] is False
        assert kept.json()["parent_team_ids"] == ["extra-0"]
        assert kept.json()["alias"] == patch.get("alias")
    moved: Final = http.post(
        "/management/v1/customer_members/upsert",
        json={"customer_org_id": "org", "customer_member_id": "moving", "parent_team_id": "team"},
    )
    assert moved.status_code == (200 if premium else 403), moved.text
    if not premium:
        assert http.get("/management/v1/customer_members/moving").json()["parent_team_ids"] == ["extra-0"]
        assert http.get("/management/v1/customer_teams/org:default").status_code == 404


@pytest.mark.parametrize("error_code", ("55P03", "XX000"))
def test_capacity_contention_is_retryable_without_hiding_other_database_errors(error_code: str) -> None:
    http: Final = client(database=CapacityErrorDatabase(error_code, "members:team"), premium_user=False)
    family(http)
    payload: Final = {"customer_member_id": "blocked", "parent_team_ids": ["team"]}
    if error_code != "55P03":
        with pytest.raises(RawQueryError, match="advisory lock failed"):
            http.post("/management/v1/customer_members", json=payload)
        return
    response: Final = http.post("/management/v1/customer_members", json=payload)
    assert response.status_code == 503, response.text
    assert response.json() == {
        "type": "urn:litellm:error:customer-hierarchy-capacity-busy",
        "title": "Customer portal busy",
        "status": 503,
        "detail": "Customer portal capacity is being updated. Retry the request.",
    }
    assert http.get("/management/v1/customer_members/blocked").status_code == 404


def test_customer_portal_limits_require_admin() -> None:
    response: Final = client(role=LitellmUserRoles.INTERNAL_USER).get("/management/v1/customer_portal/limits")
    assert response.status_code == 403, response.text


@pytest.mark.parametrize(
    ("error", "retryable"),
    (
        (DataError({"user_facing_error": {"meta": {"code": "55P03", "message": "lock unavailable"}}}), True),
        (DataError({"user_facing_error": {"message": "canceling statement due to lock timeout"}}), True),
        (DataError({"user_facing_error": {"message": "other database error", "meta": {"code": "XX000"}}}), False),
    ),
    ids=("sqlstate", "message-only", "other-error"),
)
def test_team_model_write_contention_is_retryable_and_rolls_back(error: DataError, retryable: bool) -> None:
    database: Final = Database(team_create_error=error)
    http: Final = client(database=database, premium_user=False)
    created: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "org"})
    assert created.status_code == 201, created.text
    payload: Final = {"customer_team_id": "blocked", "parent_org_id": "org", "budget": {"max_budget": 10}}
    if not retryable:
        with pytest.raises(DataError, match="other database error") as raised:
            http.post("/management/v1/customer_teams", json=payload)
        assert raised.value is error
    else:
        response: Final = http.post("/management/v1/customer_teams", json=payload)
        assert response.status_code == 503, response.text
        assert response.json() == {
            "type": "urn:litellm:error:customer-hierarchy-capacity-busy",
            "title": "Customer portal busy",
            "status": 503,
            "detail": "Customer portal capacity is being updated. Retry the request.",
        }
    assert http.get("/management/v1/customer_teams/blocked").status_code == 404
    assert not database.litellm_budgettable.rows


def family(http: TestClient) -> dict[str, object]:
    org: Final = http.post(
        "/management/v1/customer_organizations",
        json={
            "customer_org_id": "org",
            "budget": {"max_budget": 1000},
            "default_budgets": [
                {
                    "name": "standard",
                    "max_budget": 100,
                    "rpm_limit": 10,
                    "allowed_models": ["model-a"],
                },
                {"name": "small", "max_budget": 20, "rpm_limit": 8},
            ],
            "default_budget_template": "standard",
        },
    )
    assert org.status_code == 201, org.text
    team: Final = http.post(
        "/management/v1/customer_teams",
        json={
            "customer_team_id": "team",
            "parent_org_id": "org",
            "budget": {"max_budget": 999},
        },
    )
    assert team.status_code == 201, team.text
    return org.json()


def test_member_budget_id_must_name_one_of_its_organizations_templates() -> None:
    """An unowned `/budget/new` row and another member's override are both outside the organization (F4)."""
    database: Final = Database()
    asyncio.run(
        database.litellm_budgettable.create(
            data={
                "budget_id": "global",
                "budget_limits": [{"budget_duration": "12mo", "max_budget": 1000}],
            }
        )
    )
    http: Final = client(database=database)
    family(http)
    unowned: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "global-member", "parent_team_ids": ["team"], "budget_id": "global"},
    )
    assert unowned.status_code == 422, unowned.text
    created: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "override-member",
            "parent_team_ids": ["team"],
            "budget_override": {"budget_limits": [{"budget_duration": "1mo", "max_budget": 100}]},
        },
    )
    assert created.status_code == 201, created.text
    shared_override: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "other-member",
            "parent_team_ids": ["team"],
            "budget_id": created.json()["budget_override_id"],
        },
    )
    assert shared_override.status_code == 422, shared_override.text


@pytest.mark.parametrize("clear_limits", (None, []))
def test_member_window_override_round_trip_and_clear(clear_limits: object) -> None:
    http: Final = client()
    family(http)
    template: Final = http.post(
        "/management/v1/customer_organizations/org/default_budgets",
        json={
            "name": "combined",
            "budget_limits": [
                {"budget_duration": "1mo", "max_budget": 100},
                {"budget_duration": "12mo", "max_budget": 1000},
            ],
        },
    )
    assert template.status_code == 201, template.text
    budget_id: Final = next(row["budget_id"] for row in template.json()["default_budgets"] if row["name"] == "combined")
    created: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "window-member",
            "parent_team_ids": ["team"],
            "budget_id": budget_id,
            "budget_override": {"budget_limits": [{"budget_duration": "1mo", "max_budget": 0}]},
        },
    )
    assert created.status_code == 201, created.text
    assert {
        row["budget_duration"]: row["max_budget"] for row in created.json()["effective_budget"]["budget_limits"]
    } == {
        "1mo": 0,
        "12mo": 1000,
    }
    renamed: Final = http.patch("/management/v1/customer_members/window-member", json={"alias": "renamed"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["effective_budget"]["budget_limits"] == created.json()["effective_budget"]["budget_limits"]
    cleared: Final = http.patch(
        "/management/v1/customer_members/window-member",
        json={
            "budget_override": {"budget_limits": clear_limits},
        },
    )
    assert cleared.status_code == 200, cleared.text
    assert {
        row["budget_duration"]: row["max_budget"] for row in cleared.json()["effective_budget"]["budget_limits"]
    } == {
        "1mo": 100,
        "12mo": 1000,
    }


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
@pytest.mark.parametrize(
    "method,path,body",
    (
        ("POST", "/customer_organizations", {}),
        ("POST", "/customer_organizations/upsert", {"alias": "customer"}),
        ("GET", "/customer_organizations", None),
        ("GET", "/customer_organizations/id", None),
        ("PATCH", "/customer_organizations/id", {}),
        ("DELETE", "/customer_organizations/id", None),
        ("DELETE", "/customer_organizations/id?cascade=true", None),
        ("POST", "/customer_teams", {"parent_org_id": "id"}),
        ("GET", "/customer_teams", None),
        ("GET", "/customer_teams/id", None),
        ("PATCH", "/customer_teams/id", {}),
        ("DELETE", "/customer_teams/id", None),
        ("DELETE", "/customer_teams/id?cascade=true", None),
        ("POST", "/customer_members", {"parent_team_ids": ["id"]}),
        ("POST", "/customer_members/upsert", {"customer_member_id": "m", "customer_org_id": "id"}),
        ("GET", "/customer_members", None),
        ("GET", "/customer_members/id", None),
        ("PATCH", "/customer_members/id", {}),
        ("DELETE", "/customer_members/id", None),
        ("POST", "/customer_organizations/id/default_budgets", {"name": "new"}),
        ("PATCH", "/customer_organizations/id/default_budgets/budget", {}),
        ("DELETE", "/customer_organizations/id/default_budgets/budget", None),
    ),
)
def test_every_dispatched_method_is_admin_only(
    role: LitellmUserRoles,
    method: str,
    path: str,
    body: dict[str, object] | None,
) -> None:
    response: Final = client(role).request(method, "/management/v1" + path, json=body)
    assert response.status_code == 403, response.text
    assert response.json()["status"] == 403


def test_template_edits_propagate_and_only_live_override_fields_win() -> None:
    http: Final = client()
    org: Final = family(http)
    templates: Final = org["default_budgets"]
    default_id: Final = templates[0]["budget_id"]
    other_id: Final = templates[1]["budget_id"]
    default: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "default",
            "parent_team_ids": ["team"],
        },
    )
    assert default.status_code == 201, default.text
    assert default.json()["budget_override_id"] is None
    assert default.json()["budget_id"] == default_id
    override: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "override",
            "parent_team_ids": ["team"],
            "budget_id": other_id,
            "budget_override": {"max_budget": 3, "allowed_models": []},
        },
    )
    assert override.status_code == 201, override.text
    expired: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "expired",
            "parent_team_ids": ["team"],
            "budget_override": {"max_budget": 1},
            "budget_override_expires_at": "2025-01-01T00:00:00Z",
        },
    )
    assert expired.status_code == 201, expired.text
    first: Final = http.patch(
        f"/management/v1/customer_organizations/org/default_budgets/{default_id}",
        json={
            "max_budget": 200,
            "rpm_limit": 15,
            "name": "renamed",
        },
    )
    assert first.status_code == 200, first.text
    second: Final = http.patch(
        f"/management/v1/customer_organizations/org/default_budgets/{other_id}",
        json={
            "max_budget": 40,
            "rpm_limit": 12,
            "allowed_models": ["model-b"],
        },
    )
    assert second.status_code == 200, second.text
    fetched: Final = http.get("/management/v1/customer_members/default")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["effective_budget"]["max_budget"] == 200
    assert fetched.json()["effective_budget"]["rpm_limit"] == 15
    selected: Final = http.get("/management/v1/customer_members/override")
    assert selected.status_code == 200, selected.text
    assert selected.json()["effective_budget"] == {
        "max_budget": 3,
        "soft_budget": None,
        "rpm_limit": 12,
        "tpm_limit": None,
        "allowed_models": ["model-b"],
        "budget_duration": None,
        "budget_limits": None,
        "model_tpm_limit": None,
        "model_rpm_limit": None,
        "cap_policy": None,
    }
    fallback: Final = http.get("/management/v1/customer_members/expired")
    assert fallback.status_code == 200, fallback.text
    assert fallback.json()["effective_budget"] == fetched.json()["effective_budget"]
    assert fallback.json()["budget_override"]["max_budget"] == 1
    assert [template["name"] for template in second.json()["default_budgets"]] == ["renamed", "small"]
    assert second.json()["budget"]["max_budget"] == 1000


def test_override_can_be_reexpired_cleared_and_template_switched() -> None:
    http: Final = client()
    org: Final = family(http)
    created: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "member",
            "parent_team_ids": ["team"],
            "budget_override": {"max_budget": 5},
        },
    )
    assert created.status_code == 201, created.text
    expired: Final = http.patch(
        "/management/v1/customer_members/member",
        json={
            "budget_override_expires_at": "2025-01-01T00:00:00Z",
        },
    )
    assert expired.status_code == 200, expired.text
    assert expired.json()["effective_budget"]["max_budget"] == 100
    renewed: Final = http.patch("/management/v1/customer_members/member", json={"budget_override_expires_at": None})
    assert renewed.status_code == 200, renewed.text
    assert renewed.json()["effective_budget"]["max_budget"] == 5
    switched: Final = http.patch(
        "/management/v1/customer_members/member",
        json={
            "budget_id": org["default_budgets"][1]["budget_id"],
            "budget_override": None,
        },
    )
    assert switched.status_code == 200, switched.text
    assert switched.json()["budget_override_id"] is None
    assert switched.json()["budget_override"] is None
    assert switched.json()["effective_budget"]["max_budget"] == 20
    cleared: Final = http.patch("/management/v1/customer_members/member", json={"budget_id": None})
    assert cleared.status_code == 422, cleared.text
    after: Final = http.get("/management/v1/customer_members/member")
    assert after.json() == switched.json()


def test_clearing_an_override_also_clears_its_expiry() -> None:
    http: Final = client()
    family(http)
    created: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "member",
            "parent_team_ids": ["team"],
            "budget_override": {"max_budget": 5},
            "budget_override_expires_at": "2025-01-01T00:00:00Z",
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["effective_budget"]["max_budget"] == 100
    cleared: Final = http.patch("/management/v1/customer_members/member", json={"budget_override": None})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["budget_override"] is None
    assert cleared.json()["budget_override_expires_at"] is None
    renewed: Final = http.patch("/management/v1/customer_members/member", json={"budget_override": {"max_budget": 7}})
    assert renewed.status_code == 200, renewed.text
    assert renewed.json()["budget_override_expires_at"] is None
    assert renewed.json()["effective_budget"]["max_budget"] == 7


_PAST_EXPIRY: Final = "2020-01-01T00:00:00Z"


@pytest.mark.parametrize(
    ("route", "created_without_override"),
    (
        ("/management/v1/customer_members", {"customer_member_id": "row", "parent_team_ids": ["team"]}),
        ("/management/v1/customer_teams", {"customer_team_id": "row", "parent_org_id": "org"}),
    ),
    ids=("member", "team"),
)
def test_an_expiry_sent_without_an_override_is_not_stored_and_a_later_override_applies(
    route: str, created_without_override: dict[str, object]
) -> None:
    http: Final = client()
    family(http)
    created: Final = http.post(route, json={**created_without_override, "budget_override_expires_at": _PAST_EXPIRY})
    assert created.status_code == 201, created.text
    assert created.json()["budget_override_expires_at"] is None
    expiry_only: Final = http.patch(f"{route}/row", json={"budget_override_expires_at": _PAST_EXPIRY})
    assert expiry_only.status_code == 200, expiry_only.text
    assert expiry_only.json()["budget_override_expires_at"] is None
    overridden: Final = http.patch(f"{route}/row", json={"budget_override": {"max_budget": 7}})
    assert overridden.status_code == 200, overridden.text
    assert overridden.json()["budget_override_expires_at"] is None
    assert overridden.json()["effective_budget"]["max_budget"] == 7


def test_an_override_added_to_a_row_holding_a_stale_expiry_does_not_inherit_it() -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    created: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "member", "parent_team_ids": ["team"]}
    )
    assert created.status_code == 201, created.text
    asyncio.run(
        database.litellm_customermembertable.update(
            where={"customer_member_id": "member"},
            data={"budget_override_expires_at": NOW - timedelta(days=365)},
        )
    )
    overridden: Final = http.patch(
        "/management/v1/customer_members/member", json={"budget_override": {"max_budget": 7}}
    )
    assert overridden.status_code == 200, overridden.text
    assert overridden.json()["budget_override_expires_at"] is None
    assert overridden.json()["effective_budget"]["max_budget"] == 7


def test_null_budget_clears_the_overall_budget_of_an_organization_and_a_team() -> None:
    http: Final = client()
    family(http)
    org_cleared: Final = http.patch("/management/v1/customer_organizations/org", json={"budget": None})
    assert org_cleared.status_code == 200, org_cleared.text
    assert org_cleared.json()["budget"] is None
    assert org_cleared.json()["budget_id"] is None
    team_cleared: Final = http.patch("/management/v1/customer_teams/team", json={"budget": None})
    assert team_cleared.status_code == 200, team_cleared.text
    assert team_cleared.json()["budget"] is None
    assert team_cleared.json()["budget_id"] is None
    omitted: Final = http.patch("/management/v1/customer_organizations/org", json={"alias": "renamed"})
    assert omitted.status_code == 200, omitted.text
    assert omitted.json()["budget"] is None
    restored: Final = http.patch("/management/v1/customer_organizations/org", json={"budget": {"max_budget": 5}})
    assert restored.status_code == 200, restored.text
    assert restored.json()["budget"]["max_budget"] == 5


def test_template_deletion_guards_names_and_references() -> None:
    http: Final = client()
    org: Final = family(http)
    default_id: Final = org["default_budget_template_id"]
    selected_id: Final = org["default_budgets"][1]["budget_id"]
    member: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "member",
            "parent_team_ids": ["team"],
            "budget_id": selected_id,
        },
    )
    assert member.status_code == 201, member.text
    for template_id in (default_id, selected_id):
        response: Final = http.delete(f"/management/v1/customer_organizations/org/default_budgets/{template_id}")
        assert response.status_code == 409, response.text
        assert response.json()["type"].endswith("template-in-use")
    duplicate: Final = http.post("/management/v1/customer_organizations/org/default_budgets", json={"name": "small"})
    assert duplicate.status_code == 422, duplicate.text
    rename: Final = http.patch(
        f"/management/v1/customer_organizations/org/default_budgets/{selected_id}", json={"name": "standard"}
    )
    assert rename.status_code == 422, rename.text
    removed: Final = http.delete("/management/v1/customer_members/member")
    assert removed.status_code == 204, removed.text
    fetched: Final = http.get("/management/v1/customer_organizations/org")
    assert fetched.json() == org
    deleted: Final = http.delete(f"/management/v1/customer_organizations/org/default_budgets/{selected_id}")
    assert deleted.status_code == 204, deleted.text
    added: Final = http.post("/management/v1/customer_organizations/org/default_budgets", json={"name": "new"})
    assert added.status_code == 201, added.text
    assert added.json()["default_budget_template_id"] == default_id
    new_id: Final = next(t["budget_id"] for t in added.json()["default_budgets"] if t["name"] == "new")
    moved: Final = http.patch("/management/v1/customer_organizations/org", json={"default_budget_template_id": new_id})
    assert moved.status_code == 200, moved.text
    assert moved.json()["default_budget_template_id"] == new_id
    old_default: Final = http.delete(f"/management/v1/customer_organizations/org/default_budgets/{default_id}")
    assert old_default.status_code == 204, old_default.text


def test_missing_template_requires_custom_budget_and_wrong_org_is_rejected() -> None:
    http: Final = client()
    org: Final = family(http)
    empty: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "empty"})
    assert empty.status_code == 201, empty.text
    team: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "empty-team", "parent_org_id": "empty"}
    )
    assert team.status_code == 201, team.text
    missing: Final = http.post("/management/v1/customer_members", json={"parent_team_ids": ["empty-team"]})
    assert missing.status_code == 422, missing.text
    wrong: Final = http.post(
        "/management/v1/customer_members",
        json={
            "parent_team_ids": ["empty-team"],
            "budget_id": org["default_budget_template_id"],
            "budget_override": {"max_budget": 2},
        },
    )
    assert wrong.status_code == 422, wrong.text
    custom: Final = http.post(
        "/management/v1/customer_members",
        json={"parent_team_ids": ["empty-team"], "budget_override": {"max_budget": 2}},
    )
    assert custom.status_code == 201, custom.text
    assert custom.json()["budget_id"] is None
    assert custom.json()["effective_budget"]["max_budget"] == 2


@pytest.mark.parametrize(
    "entity,body",
    (
        ("customer_organizations", {}),
        ("customer_teams", {"parent_org_id": "org"}),
        ("customer_members", {"parent_team_ids": ["team"]}),
    ),
)
@pytest.mark.parametrize("method", ("POST", "PATCH"))
@pytest.mark.parametrize("field,value", (("spend", 1), ("budget_reset_at", "2025-01-01T00:00:00Z")))
def test_spend_and_reset_fields_are_read_only(
    entity: str, body: dict[str, str], method: str, field: str, value: object
) -> None:
    http: Final = client()
    family(http)
    path: Final = f"/management/v1/{entity}" + ("/id" if method == "PATCH" else "")
    response: Final = http.request(method, path, json={**body, field: value})
    assert response.status_code == 422, response.text


def test_new_entities_report_zero_spend_and_no_reset_and_parent_deletion_conflicts() -> None:
    http: Final = client()
    org: Final = family(http)
    member: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "member", "parent_team_ids": ["team"]}
    )
    assert member.status_code == 201, member.text
    for path in ("customer_organizations/org", "customer_teams/team", "customer_members/member"):
        fetched: Final = http.get("/management/v1/" + path)
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["spend"] == 0
        assert fetched.json()["budget_reset_at"] is None
    for path in ("customer_organizations/org", "customer_teams/team"):
        for query in ("", "?cascade=false"):
            blocked: Final = http.delete("/management/v1/" + path + query)
            assert blocked.status_code == 409, blocked.text
    for path in ("customer_members/member", "customer_teams/team", "customer_organizations/org"):
        deleted: Final = http.delete("/management/v1/" + path)
        assert deleted.status_code == 204, deleted.text
        absent: Final = http.get("/management/v1/" + path)
        assert absent.status_code == 404, absent.text
    assert org["budget_id"] is not None


@pytest.mark.parametrize(
    "body",
    (
        {"default_budgets": [{"name": "a"}, {"name": "b"}]},
        {"default_budgets": [{"name": "a"}, {"name": "b"}], "default_budget_template": "c"},
        {"default_budgets": [{"name": "a"}, {"name": "a"}], "default_budget_template": "a"},
        {"default_budget_template": "a"},
        {"default_budgets": [{"name": "a"}], "default_budget_template_id": "a"},
    ),
)
def test_invalid_template_configuration_does_not_create_an_organization(body: dict[str, object]) -> None:
    http: Final = client()
    response: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "org", **body})
    assert response.status_code == 422, response.text
    missing: Final = http.get("/management/v1/customer_organizations/org")
    assert missing.status_code == 404, missing.text


def test_single_template_defaults_and_org_patch_cannot_replace_templates() -> None:
    http: Final = client()
    created: Final = http.post(
        "/management/v1/customer_organizations",
        json={
            "customer_org_id": "org",
            "default_budgets": [{"name": "only", "max_budget": 10}],
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["default_budget_template_id"] == created.json()["default_budgets"][0]["budget_id"]
    patched: Final = http.patch("/management/v1/customer_organizations/org", json={"default_budgets": []})
    assert patched.status_code == 422, patched.text
    fetched: Final = http.get("/management/v1/customer_organizations/org")
    assert fetched.json() == created.json()


def test_lists_filter_parents_and_link_pages_and_reject_unknown_parameters() -> None:
    http: Final = client()
    family(http)
    other: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "other"})
    assert other.status_code == 201, other.text
    second_team: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "z-team", "parent_org_id": "org"}
    )
    assert second_team.status_code == 201, second_team.text
    listed: Final = http.get("/management/v1/customer_teams?parent_org_id=org&page_size=1")
    assert listed.status_code == 200, listed.text
    assert listed.json()["meta"]["total_count"] == 2
    assert listed.json()["data"][0]["customer_team_id"] == "team"
    next_page: Final = http.get(listed.json()["links"]["next"])
    assert next_page.status_code == 200, next_page.text
    assert next_page.json()["data"][0]["customer_team_id"] == "z-team"
    empty: Final = http.get("/management/v1/customer_members?parent_team_id=team")
    assert empty.status_code == 200, empty.text
    assert empty.json()["data"] == []
    bad: Final = http.get("/management/v1/customer_teams?unknown=1")
    assert bad.status_code == 400, bad.text
    repeated: Final = http.get("/management/v1/customer_teams?page=1&page=2")
    assert repeated.status_code == 400, repeated.text


@pytest.mark.parametrize(
    "entity,payload",
    (
        ("customer_organizations", {"customer_org_id": "duplicate"}),
        ("customer_teams", {"customer_team_id": "duplicate", "parent_org_id": "org"}),
        ("customer_members", {"customer_member_id": "duplicate", "parent_team_ids": ["team"]}),
    ),
)
def test_duplicate_id_conflicts(entity: str, payload: dict[str, str]) -> None:
    http: Final = client()
    family(http)
    first: Final = http.post("/management/v1/" + entity, json=payload)
    assert first.status_code == 201, first.text
    duplicate: Final = http.post("/management/v1/" + entity, json=payload)
    assert duplicate.status_code == 409, duplicate.text


@pytest.mark.parametrize("entity", ("customer_teams", "customer_members"))
def test_missing_parent_is_404(entity: str) -> None:
    parent: Final = {"parent_org_id": "missing"} if entity == "customer_teams" else {"parent_team_ids": ["missing"]}
    response: Final = client().post("/management/v1/" + entity, json=parent)
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "budget",
    (
        {"max_budget": -1},
        {"max_budget": "NaN"},
        {"max_budget": "Infinity"},
        {"tpm_limit": -1},
        {"rpm_limit": -1},
        {"budget_duration": "0s"},
    ),
)
def test_invalid_budget_values_fail_at_the_http_boundary(budget: dict[str, object]) -> None:
    response: Final = client().post("/management/v1/customer_organizations", json={"budget": budget})
    assert response.status_code == 422, response.text


def test_partial_member_override_updates_preserve_other_fields_and_never_change_the_template() -> None:
    http: Final = client()
    organization: Final = family(http)
    created: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "member",
            "parent_team_ids": ["team"],
        },
    )
    assert created.status_code == 201, created.text
    patched: Final = http.patch(
        "/management/v1/customer_members/member",
        json={
            "budget_override": {"max_budget": 5, "soft_budget": 4, "tpm_limit": 2, "allowed_models": ["model-b"]},
            "budget_override_expires_at": "2027-01-01T00:00:00Z",
        },
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["budget_override_id"] != organization["default_budget_template_id"]
    assert patched.json()["effective_budget"]["max_budget"] == 5
    inherited: Final = http.patch(
        "/management/v1/customer_members/member",
        json={
            "budget_override": {"max_budget": None, "allowed_models": []},
        },
    )
    assert inherited.status_code == 200, inherited.text
    assert inherited.json()["budget_override_id"] == patched.json()["budget_override_id"]
    assert inherited.json()["effective_budget"]["max_budget"] == 100
    assert inherited.json()["effective_budget"]["soft_budget"] == 4
    assert inherited.json()["effective_budget"]["tpm_limit"] == 2
    assert inherited.json()["effective_budget"]["allowed_models"] == ["model-a"]
    unchanged: Final = http.get("/management/v1/customer_organizations/org")
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json() == {**organization, "member_count": 1}


def test_first_added_template_is_default_and_org_patch_moves_or_clears_the_default() -> None:
    http: Final = client()
    created: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "org"})
    assert created.status_code == 201, created.text
    first: Final = http.post("/management/v1/customer_organizations/org/default_budgets", json={"name": "first"})
    assert first.status_code == 201, first.text
    first_id: Final = first.json()["default_budget_template_id"]
    second: Final = http.post("/management/v1/customer_organizations/org/default_budgets", json={"name": "second"})
    assert second.status_code == 201, second.text
    assert second.json()["default_budget_template_id"] == first_id
    second_id: Final = next(t["budget_id"] for t in second.json()["default_budgets"] if t["name"] == "second")
    foreign: Final = http.patch(
        "/management/v1/customer_organizations/org", json={"default_budget_template_id": "someone-elses"}
    )
    assert foreign.status_code == 422, foreign.text
    moved: Final = http.patch(
        "/management/v1/customer_organizations/org", json={"default_budget_template_id": second_id}
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["default_budget_template_id"] == second_id
    member: Final = http.post("/management/v1/customer_members", json={"parent_team_ids": [team_in("org", http)]})
    assert member.status_code == 201, member.text
    assert member.json()["budget_id"] == second_id
    cleared: Final = http.patch("/management/v1/customer_organizations/org", json={"default_budget_template_id": None})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["default_budget_template_id"] is None
    assert http.get("/management/v1/customer_organizations/org").json() == cleared.json()


def team_in(organization_id: str, http: TestClient) -> str:
    team: Final = http.post("/management/v1/customer_teams", json={"parent_org_id": organization_id})
    assert team.status_code == 201, team.text
    return cast("str", team.json()["customer_team_id"])


@pytest.mark.parametrize(
    "method,path,body",
    (
        ("POST", "/customer_organizations/missing/default_budgets", {"name": "template"}),
        ("PATCH", "/customer_organizations/missing/default_budgets/budget", {}),
        ("DELETE", "/customer_organizations/missing/default_budgets/budget", None),
    ),
)
def test_template_routes_return_404_for_a_missing_organization(
    method: str,
    path: str,
    body: dict[str, object] | None,
) -> None:
    response: Final = client().request(method, "/management/v1" + path, json=body)
    assert response.status_code == 404, response.text


def test_per_model_rate_limits_round_trip_at_every_level_and_override_the_template_map() -> None:
    http: Final = client()
    family(http)
    team: Final = http.patch(
        "/management/v1/customer_teams/team",
        json={"budget": {"model_tpm_limit": {"model-a": 5000}, "model_rpm_limit": {"model-a": 50}}},
    )
    assert team.status_code == 200, team.text
    assert team.json()["budget"]["model_tpm_limit"] == {"model-a": 5000}
    assert team.json()["budget"]["model_rpm_limit"] == {"model-a": 50}
    template: Final = http.patch(
        "/management/v1/customer_organizations/org/default_budgets/" + family_default_budget_id(http),
        json={"model_tpm_limit": {"model-a": 1000}},
    )
    assert template.status_code == 200, template.text
    created: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "member",
            "parent_team_ids": ["team"],
            "budget_override": {"model_rpm_limit": {"model-a": 1}},
        },
    )
    assert created.status_code == 201, created.text
    effective: Final = created.json()["effective_budget"]
    assert effective["model_tpm_limit"] == {"model-a": 1000}
    assert effective["model_rpm_limit"] == {"model-a": 1}
    overridden: Final = http.patch(
        "/management/v1/customer_members/member", json={"budget_override": {"model_tpm_limit": {"model-a": 10}}}
    )
    assert overridden.status_code == 200, overridden.text
    assert overridden.json()["effective_budget"]["model_tpm_limit"] == {"model-a": 10}
    assert http.get("/management/v1/customer_members/member").json()["effective_budget"]["model_rpm_limit"] == {
        "model-a": 1
    }


def test_budget_edits_at_every_level_drop_the_cached_member_hierarchy() -> None:
    cache: Final = UserApiKeyCache()
    http: Final = client(cache=cache)
    family(http)
    assert (
        http.post(
            "/management/v1/customer_members", json={"customer_member_id": "m", "parent_team_ids": ["team"]}
        ).status_code
        == 201
    )
    cache_key: Final = customer_member_cache_key("m")
    edits: Final = (
        ("/management/v1/customer_members/m", {"budget_override": {"max_budget": 1}}),
        ("/management/v1/customer_teams/team", {"budget": {"max_budget": 2}}),
        ("/management/v1/customer_organizations/org", {"budget": {"max_budget": 3}}),
        (
            f"/management/v1/customer_organizations/org/default_budgets/{family_default_budget_id(http)}",
            {"max_budget": 4},
        ),
    )
    for path, body in edits:
        asyncio.run(cache.async_set_cache(key=cache_key, value="stale hierarchy"))
        assert http.patch(path, json=body).status_code == 200, path
        assert cache.get_cache(cache_key) is None, path
    asyncio.run(cache.async_set_cache(key=cache_key, value="stale hierarchy"))
    assert http.delete("/management/v1/customer_members/m").status_code == 204
    assert cache.get_cache(cache_key) is None


def test_organization_list_counts_all_members_and_searches_before_paginating() -> None:
    http: Final = client()
    family(http)
    for identifier in ("member-a", "member-b"):
        created: Final = http.post(
            "/management/v1/customer_members", json={"customer_member_id": identifier, "parent_team_ids": ["team"]}
        )
        assert created.status_code == 201, created.text
    second_team: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "second-team", "parent_org_id": "org"}
    )
    assert second_team.status_code == 201, second_team.text
    third_member: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "member-c", "parent_team_ids": ["second-team"]}
    )
    assert third_member.status_code == 201, third_member.text
    other: Final = http.post(
        "/management/v1/customer_organizations", json={"customer_org_id": "other", "alias": "Literal%_"}
    )
    assert other.status_code == 201, other.text
    page: Final = http.get("/management/v1/customer_organizations", params={"q": "ORG", "page_size": 1})
    assert page.status_code == 200, page.text
    assert page.json()["meta"]["total_count"] == 1
    assert page.json()["data"][0]["member_count"] == 3
    assert page.json()["data"][0]["spend"] == 0
    assert page.json()["data"][0]["budget"]["max_budget"] == 1000
    literal: Final = http.get("/management/v1/customer_organizations", params={"q": "%_"})
    assert literal.json()["data"][0]["customer_org_id"] == "other"
    assert literal.json()["data"][0]["member_count"] == 0
    assert literal.json()["meta"]["total_count"] == 1


def created_member(http: TestClient, member_id: str, team_id: str, *, override: bool = False) -> CustomerMember:
    response: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": member_id,
            "parent_team_ids": [team_id],
            "budget_override": {"max_budget": 5} if override else None,
        },
    )
    assert response.status_code == 201, response.text
    return CustomerMember.model_validate(response.json())


def test_team_cascade_removes_only_its_members_and_owned_budgets() -> None:
    database: Final = Database()
    http: Final = client(database=database)
    organization: Final = family(http)
    created_member(http, "inherited", "team")
    overridden: Final = created_member(http, "overridden", "team", override=True)
    sibling: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "sibling", "parent_org_id": "org"}
    )
    assert sibling.status_code == 201, sibling.text
    created_member(http, "sibling-member", "sibling", override=True)
    before_budgets: Final = frozenset(database.litellm_budgettable.rows)
    team_budget: Final = database.litellm_customerteamtable.rows["team"].model_dump()["budget_id"]
    deleted: Final = http.delete("/management/v1/customer_teams/team?cascade=true")
    assert deleted.status_code == 204, deleted.text
    assert frozenset(database.litellm_customerteamtable.rows) == {"sibling"}
    assert frozenset(database.litellm_customermembertable.rows) == {"sibling-member"}
    assert frozenset(database.litellm_budgettable.rows) == before_budgets - {team_budget, overridden.budget_override_id}
    fetched: Final = http.get("/management/v1/customer_organizations/org")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == {**organization, "member_count": 1}
    assert http.get("/management/v1/customer_members/sibling-member").status_code == 200


def test_organization_cascade_preserves_the_sibling_customer_and_its_budgets() -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    created_member(http, "inherited", "team")
    created_member(http, "overridden", "team", override=True)
    second_team: Final = http.post(
        "/management/v1/customer_teams",
        json={"customer_team_id": "second-team", "parent_org_id": "org", "budget": {"max_budget": 20}},
    )
    assert second_team.status_code == 201, second_team.text
    created_member(http, "second-member", "second-team", override=True)
    owned_budget_ids: Final = frozenset(database.litellm_budgettable.rows)
    sibling_org: Final = http.post(
        "/management/v1/customer_organizations",
        json={
            "customer_org_id": "other",
            "budget": {"max_budget": 10},
            "default_budgets": [{"name": "default", "max_budget": 3}],
        },
    )
    assert sibling_org.status_code == 201, sibling_org.text
    sibling_team: Final = http.post(
        "/management/v1/customer_teams",
        json={"customer_team_id": "other-team", "parent_org_id": "other", "budget": {"max_budget": 5}},
    )
    assert sibling_team.status_code == 201, sibling_team.text
    created_member(http, "other-member", "other-team", override=True)
    sibling_budgets: Final = {
        key: value for key, value in database.litellm_budgettable.rows.items() if key not in owned_budget_ids
    }
    sibling_rows: Final = (
        database.litellm_customerorganizationtable.rows["other"],
        database.litellm_customerteamtable.rows["other-team"],
        database.litellm_customermembertable.rows["other-member"],
    )
    deleted: Final = http.delete("/management/v1/customer_organizations/org?cascade=true")
    assert deleted.status_code == 204, deleted.text
    assert database.litellm_customerorganizationtable.rows == {"other": sibling_rows[0]}
    assert database.litellm_customerteamtable.rows == {"other-team": sibling_rows[1]}
    assert database.litellm_customermembertable.rows == {"other-member": sibling_rows[2]}
    assert database.litellm_budgettable.rows == sibling_budgets


@pytest.mark.parametrize("resource,entity_id", (("customer_teams", "team"), ("customer_organizations", "org")))
def test_cascade_rolls_back_child_and_budget_deletes_after_a_late_foreign_key_failure(
    resource: str, entity_id: str
) -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    created_member(http, "member", "team", override=True)
    failing: Final = DeleteConflictTable(
        "customer_team_id" if resource == "customer_teams" else "customer_org_id", database
    )
    if resource == "customer_teams":
        failing.rows = database.litellm_customerteamtable.rows
        database.litellm_customerteamtable = failing
    else:
        failing.rows = database.litellm_customerorganizationtable.rows
        database.litellm_customerorganizationtable = failing
    tables: Final = (
        database.litellm_customerorganizationtable,
        database.litellm_customerteamtable,
        database.litellm_customermembertable,
        database.litellm_budgettable,
    )
    snapshots: Final = tuple(table.rows for table in tables)
    response: Final = http.delete(f"/management/v1/{resource}/{entity_id}?cascade=true")
    assert response.status_code == 409, response.text
    assert failing.members_at_delete == 0
    assert tuple(table.rows for table in tables) == snapshots


@pytest.mark.parametrize("resource,entity_id", (("customer_teams", "team"), ("customer_organizations", "org")))
def test_cascade_processes_more_than_two_batches_without_unbounded_child_reads(resource: str, entity_id: str) -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    created_member(http, "member", "team")
    bounded: Final = BoundedChildrenTable(
        "customer_member_id" if resource == "customer_teams" else "customer_team_id", database
    )
    sample: Final = (
        database.litellm_customermembertable.rows["member"]
        if resource == "customer_teams"
        else database.litellm_customerteamtable.rows["team"]
    )
    if resource == "customer_teams":
        bounded.rows = {
            f"member-{index}": sample.model_copy(update={"customer_member_id": f"member-{index}"})
            for index in range(205)
        }
        database.litellm_customermembertable = bounded
    else:
        bounded.rows = {
            "team": sample,
            **{
                f"team-{index}": sample.model_copy(update={"customer_team_id": f"team-{index}", "budget_id": None})
                for index in range(204)
            },
        }
        database.litellm_customerteamtable = bounded
    response: Final = http.delete(f"/management/v1/{resource}/{entity_id}?cascade=true")
    assert response.status_code == 204, response.text
    assert bounded.batch_sizes == (100, 100, 5, 0)
    assert not bounded.rows
    assert not database.litellm_customermembertable.rows


@pytest.mark.parametrize("resource", ("customer_teams", "customer_organizations"))
def test_missing_cascade_parent_does_not_delete_any_existing_rows(resource: str) -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    created_member(http, "member", "team", override=True)
    tables: Final = (
        database.litellm_customerorganizationtable,
        database.litellm_customerteamtable,
        database.litellm_customermembertable,
        database.litellm_budgettable,
    )
    snapshots: Final = tuple(table.rows for table in tables)
    response: Final = http.delete(f"/management/v1/{resource}/missing?cascade=true")
    assert response.status_code == 404, response.text
    assert tuple(table.rows for table in tables) == snapshots


@pytest.mark.parametrize(
    "resource,entity_id,query",
    (
        ("customer_organizations", "org", "?cascade=true"),
        ("customer_teams", "team", "?cascade=true"),
        ("customer_members", "member", ""),
    ),
)
def test_customer_deletion_preserves_budgets_still_referenced_by_gateway_keys_or_projects(
    resource: str, entity_id: str, query: str
) -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    created_member(http, "member", "team", override=True)
    budgets: Final = database.litellm_budgettable.rows
    database.litellm_verificationtoken.rows = {
        f"key-{index}": Record.model_validate({"token": f"key-{index}", "budget_id": budget_id})
        for index, budget_id in enumerate(budgets)
    }
    database.litellm_projecttable.rows = {
        f"project-{index}": Record.model_validate({"project_id": f"project-{index}", "budget_id": budget_id})
        for index, budget_id in enumerate(budgets)
    }
    keys: Final = database.litellm_verificationtoken.rows
    projects: Final = database.litellm_projecttable.rows
    response: Final = http.delete(f"/management/v1/{resource}/{entity_id}{query}")
    assert response.status_code == 204, response.text
    assert database.litellm_budgettable.rows == budgets
    assert database.litellm_verificationtoken.rows == keys
    assert database.litellm_projecttable.rows == projects
    assert http.get(f"/management/v1/{resource}/{entity_id}").status_code == 404


def test_customer_members_are_paginated_across_only_the_selected_organization() -> None:
    http: Final = client()
    family(http)
    created_member(http, "member-a", "team")
    second: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "team-b", "parent_org_id": "org"}
    )
    assert second.status_code == 201, second.text
    created_member(http, "member-b", "team-b")
    other_org: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "other"})
    assert other_org.status_code == 201, other_org.text
    other_team: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "other-team", "parent_org_id": "other"}
    )
    assert other_team.status_code == 201, other_team.text
    created_member(http, "other-member", "other-team", override=True)
    first: Final = http.get("/management/v1/customer_members?parent_org_id=org&page_size=1")
    assert first.status_code == 200, first.text
    assert first.json()["meta"]["total_count"] == 2
    assert [member["customer_member_id"] for member in first.json()["data"]] == ["member-a"]
    second_page: Final = http.get(first.json()["links"]["next"])
    assert second_page.status_code == 200, second_page.text
    assert [member["customer_member_id"] for member in second_page.json()["data"]] == ["member-b"]
    intersected: Final = http.get("/management/v1/customer_members?parent_org_id=org&parent_team_id=other-team")
    assert intersected.status_code == 200, intersected.text
    assert intersected.json()["data"] == []


def family_default_budget_id(http: TestClient) -> str:
    organization: Final = http.get("/management/v1/customer_organizations/org").json()
    return str(organization["default_budget_template_id"])


@pytest.mark.parametrize("use_id", (True, False))
def test_customer_org_upsert_repeats_without_new_budgets_and_updates_by_identity(use_id: bool) -> None:
    http: Final = client()
    identity: Final = {"customer_org_id": "org"} if use_id else {}
    payload: Final = {
        **identity,
        "alias": "customer",
        "budget": {"max_budget": 10},
        "default_budgets": [{"name": "standard", "max_budget": 2}],
    }
    first: Final = http.post("/management/v1/customer_organizations/upsert", json=payload)
    assert first.status_code == 200, first.text
    assert first.json()["created"] is True
    second: Final = http.post("/management/v1/customer_organizations/upsert", json=payload)
    assert second.status_code == 200, second.text
    assert second.json() == {**first.json(), "created": False}
    changed: Final = http.post(
        "/management/v1/customer_organizations/upsert",
        json={**payload, "budget": {"max_budget": 20}, "default_budgets": [{"name": "standard", "max_budget": 4}]},
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["created"] is False
    assert changed.json()["customer_org_id"] == first.json()["customer_org_id"]
    assert changed.json()["budget_id"] == first.json()["budget_id"]
    assert changed.json()["default_budget_template_id"] == first.json()["default_budget_template_id"]
    assert changed.json()["budget"]["max_budget"] == 20
    assert changed.json()["default_budgets"][0]["max_budget"] == 4
    assert http.get("/management/v1/customer_organizations").json()["meta"]["total_count"] == 1


@pytest.mark.parametrize("method", ("POST", "PATCH", "UPSERT"))
def test_duplicate_alias_is_rejected_without_changing_the_customer_org(method: str) -> None:
    http: Final = client()
    first: Final = http.post(
        "/management/v1/customer_organizations", json={"customer_org_id": "first", "alias": "taken"}
    )
    assert first.status_code == 201, first.text
    other: Final = http.post(
        "/management/v1/customer_organizations", json={"customer_org_id": "other", "alias": "other"}
    )
    assert other.status_code == 201, other.text
    requests: Final = {
        "POST": ("POST", "/customer_organizations", {"customer_org_id": "new", "alias": "taken"}),
        "PATCH": ("PATCH", "/customer_organizations/other", {"alias": "taken"}),
        "UPSERT": ("POST", "/customer_organizations/upsert", {"customer_org_id": "other", "alias": "taken"}),
    }
    request_method, path, body = requests[method]
    response: Final = http.request(request_method, "/management/v1" + path, json=body)
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Customer org alias is already used by another customer org."
    assert http.get("/management/v1/customer_organizations/other").json() == other.json()
    assert http.get("/management/v1/customer_organizations").json()["meta"]["total_count"] == 2


def test_customer_member_upsert_inherits_default_and_preserves_budget_ids_on_update() -> None:
    http: Final = client()
    family(http)
    payload: Final = {"customer_member_id": "m", "customer_org_id": "org", "alias": "customer member"}
    first: Final = http.post("/management/v1/customer_members/upsert", json=payload)
    assert first.status_code == 200, first.text
    assert first.json()["created"] is True
    assert first.json()["parent_team_ids"] == ["org:default"]
    assert first.json()["budget_id"] == family_default_budget_id(http)
    repeated: Final = http.post("/management/v1/customer_members/upsert", json=payload)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json() == {**first.json(), "created": False}
    overridden: Final = http.post(
        "/management/v1/customer_members/upsert", json={**payload, "budget_override": {"max_budget": 3}}
    )
    assert overridden.status_code == 200, overridden.text
    assert overridden.json()["effective_budget"]["max_budget"] == 3
    updated: Final = http.post(
        "/management/v1/customer_members/upsert", json={**payload, "budget_override": {"max_budget": 5}}
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["budget_override_id"] == overridden.json()["budget_override_id"]
    assert updated.json()["effective_budget"]["max_budget"] == 5
    cleared: Final = http.post("/management/v1/customer_members/upsert", json={**payload, "budget_override": None})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["budget_override_id"] is None
    assert cleared.json()["effective_budget"]["max_budget"] == 100
    assert http.get("/management/v1/customer_members").json()["meta"]["total_count"] == 1


def test_customer_member_upsert_accepts_customer_org_alias_and_explicit_customer_team() -> None:
    http: Final = client()
    family(http)
    assert http.patch("/management/v1/customer_organizations/org", json={"alias": "customer"}).status_code == 200
    response: Final = http.post(
        "/management/v1/customer_members/upsert",
        json={"customer_member_id": "m", "customer_org_alias": "customer", "parent_team_id": "team"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["created"] is True
    assert response.json()["parent_team_ids"] == ["team"]
    repeated: Final = http.post(
        "/management/v1/customer_members/upsert", json={"customer_member_id": "m", "customer_org_id": "org"}
    )
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["parent_team_ids"] == ["team"]
    assert repeated.json()["created"] is False


@pytest.mark.parametrize("mismatch", ("customer_org", "customer_team", "customer_org_alias"))
def test_customer_member_upsert_rejects_cross_customer_org_identity(mismatch: str) -> None:
    http: Final = client()
    family(http)
    assert http.post("/management/v1/customer_organizations", json={"customer_org_id": "other"}).status_code == 201
    assert (
        http.post(
            "/management/v1/customer_teams", json={"customer_team_id": "foreign", "parent_org_id": "other"}
        ).status_code
        == 201
    )
    original: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "m", "parent_team_ids": ["team"]}
    )
    assert original.status_code == 201, original.text
    edits: Final = {
        "customer_org": {"customer_org_id": "other"},
        "customer_team": {"parent_team_id": "foreign"},
        "customer_org_alias": {"customer_org_alias": "wrong"},
    }
    response: Final = http.post(
        "/management/v1/customer_members/upsert",
        json={"customer_member_id": "m", "customer_org_id": "org", **edits[mismatch]},
    )
    assert response.status_code == 409, response.text
    assert response.json()["title"] == "Customer org mismatch"
    assert http.get("/management/v1/customer_members/m").json() == original.json()


@pytest.mark.parametrize(
    "resource,payload,status",
    (
        ("customer_organizations", {}, 422),
        ("customer_organizations", {"customer_org_id": "", "alias": "new"}, 422),
        ("customer_members", {"customer_member_id": "m"}, 422),
        ("customer_members", {"customer_org_id": "org"}, 422),
        ("customer_members", {"customer_member_id": "m", "customer_org_id": "missing"}, 404),
        ("customer_members", {"customer_member_id": "m", "customer_org_id": "org", "parent_team_id": "missing"}, 404),
        ("customer_members", {"customer_member_id": "m", "customer_org_id": "org", "budget_id": "missing"}, 422),
    ),
)
def test_customer_upsert_missing_identity_or_parent_is_rejected(
    resource: str, payload: dict[str, object], status: int
) -> None:
    http: Final = client()
    family(http)
    response: Final = http.post(f"/management/v1/{resource}/upsert", json=payload)
    assert response.status_code == status, response.text
    assert http.get("/management/v1/customer_members").json()["meta"]["total_count"] == 0


def test_customer_upsert_rejects_missing_budget_without_creating_a_default_customer_team() -> None:
    http: Final = client()
    assert http.post("/management/v1/customer_organizations", json={"customer_org_id": "org"}).status_code == 201
    response: Final = http.post(
        "/management/v1/customer_members/upsert", json={"customer_member_id": "m", "customer_org_id": "org"}
    )
    assert response.status_code == 422, response.text
    assert http.get("/management/v1/customer_teams").json()["meta"]["total_count"] == 0


def test_customer_upsert_invalid_templates_do_not_partially_update_customer_org() -> None:
    http: Final = client()
    original: Final = family(http)
    invalid: Final = http.post(
        "/management/v1/customer_organizations/upsert",
        json={
            "customer_org_id": "org",
            "alias": "changed",
            "budget": {"max_budget": 1},
            "default_budgets": [{"name": "duplicate"}, {"name": "duplicate"}],
            "default_budget_template": "duplicate",
        },
    )
    assert invalid.status_code == 422, invalid.text
    assert http.get("/management/v1/customer_organizations/org").json() == original


def test_customer_upserts_invalidate_customer_member_and_customer_org_alias_caches() -> None:
    cache: Final = UserApiKeyCache()
    http: Final = client(cache=cache)
    family(http)
    assert http.patch("/management/v1/customer_organizations/org", json={"alias": "old"}).status_code == 200
    assert (
        http.post(
            "/management/v1/customer_members", json={"customer_member_id": "m", "parent_team_ids": ["team"]}
        ).status_code
        == 201
    )
    member_key: Final = customer_member_cache_key("m")
    for key in (member_key, customer_org_alias_cache_key("old"), customer_org_alias_cache_key("new")):
        asyncio.run(cache.async_set_cache(key=key, value="stale customer hierarchy"))
    updated: Final = http.post(
        "/management/v1/customer_organizations/upsert", json={"customer_org_id": "org", "alias": "new"}
    )
    assert updated.status_code == 200, updated.text
    for key in (member_key, customer_org_alias_cache_key("old"), customer_org_alias_cache_key("new")):
        assert cache.get_cache(key) is None
    asyncio.run(cache.async_set_cache(key=member_key, value="stale customer hierarchy"))
    member: Final = http.post(
        "/management/v1/customer_members/upsert",
        json={"customer_org_alias": "new", "customer_member_id": "m", "budget_override": {"max_budget": 1}},
    )
    assert member.status_code == 200, member.text
    assert cache.get_cache(member_key) is None


def test_customer_org_upsert_rejects_ambiguous_legacy_alias() -> None:
    database: Final = Database()
    for identifier in ("first", "second"):
        asyncio.run(
            database.litellm_customerorganizationtable.create(
                {"customer_org_id": identifier, "alias": "legacy", "created_by": "admin", "updated_by": "admin"}
            )
        )
    http: Final = client(database=database)
    response: Final = http.post("/management/v1/customer_organizations/upsert", json={"alias": "legacy"})
    assert response.status_code == 409, response.text
    assert http.get("/management/v1/customer_organizations").json()["meta"]["total_count"] == 2


def test_customer_org_upsert_retry_does_not_rewrite_audit_fields() -> None:
    database: Final = Database()
    payload: Final = {"alias": "customer", "budget": {"max_budget": 10}}
    original: Final = client(database=database, actor="creator").post(
        "/management/v1/customer_organizations/upsert", json=payload
    )
    assert original.status_code == 200, original.text
    repeated: Final = client(database=database, actor="retry").post(
        "/management/v1/customer_organizations/upsert", json=payload
    )
    assert repeated.status_code == 200, repeated.text
    assert repeated.json() == {**original.json(), "created": False}


def test_customer_org_upsert_merges_templates_without_replacing_customer_member_budgets() -> None:
    http: Final = client()
    original: Final = family(http)
    member: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "m", "parent_team_ids": ["team"]}
    )
    assert member.status_code == 201, member.text
    payload: Final = {"customer_org_id": "org", "default_budgets": [{"name": "premium", "max_budget": 200}]}
    updated: Final = http.post("/management/v1/customer_organizations/upsert", json=payload)
    assert updated.status_code == 200, updated.text
    original_ids: Final = {template["name"]: template["budget_id"] for template in original["default_budgets"]}
    updated_ids: Final = {template["name"]: template["budget_id"] for template in updated.json()["default_budgets"]}
    assert set(updated_ids) == {"standard", "small", "premium"}
    assert {name: updated_ids[name] for name in original_ids} == original_ids
    assert updated.json()["default_budget_template_id"] == updated_ids["premium"]
    assert http.get("/management/v1/customer_members/m").json() == member.json()
    repeated: Final = http.post("/management/v1/customer_organizations/upsert", json=payload)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json() == updated.json()
    inherited: Final = http.post(
        "/management/v1/customer_members/upsert", json={"customer_member_id": "new", "customer_org_id": "org"}
    )
    assert inherited.status_code == 200, inherited.text
    assert inherited.json()["budget_id"] == updated_ids["premium"]
    assert inherited.json()["effective_budget"]["max_budget"] == 200


def test_resubmitting_a_templates_duration_keeps_its_reset_clock() -> None:
    """An edit that repeats the duration a template already runs must not push its window out: the reset job
    owes that window a clearing, and a re-stamped clock would skip it. A changed duration starts a new window."""
    http: Final = client()
    org: Final = http.post(
        "/management/v1/customer_organizations",
        json={
            "customer_org_id": "clocked",
            "default_budgets": [{"name": "monthly", "max_budget": 100, "budget_duration": "30d"}],
        },
    )
    assert org.status_code == 201, org.text
    template_id: Final = org.json()["default_budget_template_id"]
    team: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "clocked-team", "parent_org_id": "clocked"}
    )
    assert team.status_code == 201, team.text
    member: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "clocked-member", "parent_team_ids": ["clocked-team"]},
    )
    assert member.status_code == 201, member.text
    first_reset_at: Final = member.json()["budget_reset_at"]
    assert first_reset_at is not None

    same_duration: Final = http.patch(
        f"/management/v1/customer_organizations/clocked/default_budgets/{template_id}",
        json={"max_budget": 150, "budget_duration": "30d"},
    )
    assert same_duration.status_code == 200, same_duration.text
    unchanged: Final = http.get("/management/v1/customer_members/clocked-member")
    assert unchanged.json()["budget_reset_at"] == first_reset_at
    assert unchanged.json()["effective_budget"]["max_budget"] == 150

    new_duration: Final = http.patch(
        f"/management/v1/customer_organizations/clocked/default_budgets/{template_id}",
        json={"budget_duration": "7d"},
    )
    assert new_duration.status_code == 200, new_duration.text
    rewound: Final = http.get("/management/v1/customer_members/clocked-member")
    assert rewound.json()["budget_reset_at"] != first_reset_at
    assert rewound.json()["budget_reset_at"] < first_reset_at


def test_budget_reset_at_is_read_from_the_governing_budget_row() -> None:
    """Customer rows carry no reset clock: an org or team reports its budget's window, and a member reports
    its unexpired override's window when that override has a duration, otherwise its template's."""
    http: Final = client()
    org: Final = http.post(
        "/management/v1/customer_organizations",
        json={
            "customer_org_id": "windowed",
            "budget": {"max_budget": 1000, "budget_duration": "30d"},
            "default_budgets": [{"name": "monthly", "max_budget": 100, "budget_duration": "30d"}],
        },
    )
    assert org.status_code == 201, org.text
    assert org.json()["budget_reset_at"] is not None
    team: Final = http.post(
        "/management/v1/customer_teams",
        json={"customer_team_id": "windowed-team", "parent_org_id": "windowed", "budget": {"max_budget": 10}},
    )
    assert team.status_code == 201, team.text
    assert team.json()["budget_reset_at"] is None
    on_template: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "on-template", "parent_team_ids": ["windowed-team"]},
    )
    assert on_template.status_code == 201, on_template.text
    assert on_template.json()["budget_reset_at"] is not None
    override: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "overridden",
            "parent_team_ids": ["windowed-team"],
            "budget_override": {"max_budget": 5, "budget_duration": "1d"},
            "budget_override_expires_at": "2099-01-01T00:00:00Z",
        },
    )
    assert override.status_code == 201, override.text
    assert override.json()["budget_reset_at"] is not None
    assert override.json()["budget_reset_at"] < on_template.json()["budget_reset_at"]
    expired: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "expired-override",
            "parent_team_ids": ["windowed-team"],
            "budget_override": {"max_budget": 5, "budget_duration": "1d"},
            "budget_override_expires_at": "2000-01-01T00:00:00Z",
        },
    )
    assert expired.status_code == 201, expired.text
    assert expired.json()["budget_reset_at"] == on_template.json()["budget_reset_at"]
    cleared: Final = http.patch(
        "/management/v1/customer_organizations/windowed", json={"budget": {"budget_duration": None}}
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["budget_reset_at"] is None


def test_template_rename_shows_on_the_organization_and_leaves_the_other_templates_alone() -> None:
    http: Final = client()
    org: Final = family(http)
    small_id: Final = org["default_budgets"][1]["budget_id"]
    renamed: Final = http.patch(
        f"/management/v1/customer_organizations/org/default_budgets/{small_id}", json={"name": "starter"}
    )
    assert renamed.status_code == 200, renamed.text
    assert "budget_names" not in renamed.json()
    assert [template["name"] for template in renamed.json()["default_budgets"]] == ["standard", "starter"]
    assert all("is_default" not in template for template in renamed.json()["default_budgets"])
    assert renamed.json()["default_budget_template_id"] == org["default_budget_template_id"]
    fetched: Final = http.get("/management/v1/customer_organizations/org")
    assert fetched.json() == renamed.json()


def test_team_override_caps_the_team_until_it_expires_and_is_released_when_cleared() -> None:
    http: Final = client()
    family(http)
    created: Final = http.post(
        "/management/v1/customer_teams",
        json={
            "customer_team_id": "boosted",
            "parent_org_id": "org",
            "budget": {"max_budget": 50, "rpm_limit": 5},
            "budget_override": {"max_budget": 500},
            "budget_override_expires_at": (NOW + timedelta(days=1)).isoformat(),
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["budget_override"]["max_budget"] == 500
    assert created.json()["effective_budget"]["max_budget"] == 500
    assert created.json()["effective_budget"]["rpm_limit"] == 5
    expired: Final = http.patch(
        "/management/v1/customer_teams/boosted",
        json={"budget_override_expires_at": (NOW - timedelta(seconds=1)).isoformat()},
    )
    assert expired.status_code == 200, expired.text
    assert expired.json()["budget_override_id"] == created.json()["budget_override_id"]
    assert expired.json()["effective_budget"]["max_budget"] == 50
    cleared: Final = http.patch("/management/v1/customer_teams/boosted", json={"budget_override": None})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["budget_override_id"] is None
    assert cleared.json()["budget_override_expires_at"] is None
    assert cleared.json()["effective_budget"]["max_budget"] == 50
    listed: Final = http.get("/management/v1/customer_teams", params={"parent_org_id": "org"})
    assert listed.status_code == 200, listed.text
    assert {team["customer_team_id"]: team["effective_budget"]["max_budget"] for team in listed.json()["data"]} == {
        "team": 999,
        "boosted": 50,
    }


@pytest.mark.parametrize("budget", ({"max_budget": 50}, None))
def test_deleting_a_team_removes_its_override_budget_with_or_without_an_overall_budget(
    budget: dict[str, object] | None,
) -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    before_budgets: Final = frozenset(database.litellm_budgettable.rows)
    created: Final = http.post(
        "/management/v1/customer_teams",
        json={
            "customer_team_id": "boosted",
            "parent_org_id": "org",
            "budget": budget,
            "budget_override": {"max_budget": 500},
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["budget_override_id"] in database.litellm_budgettable.rows
    deleted: Final = http.delete("/management/v1/customer_teams/boosted")
    assert deleted.status_code == 204, deleted.text
    assert frozenset(database.litellm_budgettable.rows) == before_budgets


def test_member_carries_its_teams_organization_and_another_organization_is_refused() -> None:
    http: Final = client()
    family(http)
    created: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "member", "parent_team_ids": ["team"]}
    )
    assert created.status_code == 201, created.text
    assert created.json()["parent_org_id"] == "org"
    explicit: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "second", "parent_team_ids": ["team"], "parent_org_id": "org"},
    )
    assert explicit.status_code == 201, explicit.text
    other: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "other"})
    assert other.status_code == 201, other.text
    wrong: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "third", "parent_team_ids": ["team"], "parent_org_id": "other"},
    )
    assert wrong.status_code == 404, wrong.text
    assert wrong.json()["type"].endswith("parent-not-found")


def test_member_in_two_teams_is_listed_under_each_and_teams_from_another_organization_are_refused() -> None:
    http: Final = client()
    family(http)
    second: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "second", "parent_org_id": "org"}
    )
    assert second.status_code == 201, second.text
    created: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "member", "parent_team_ids": ["team", "second", "team"]},
    )
    assert created.status_code == 201, created.text
    assert created.json()["parent_team_ids"] == ["team", "second"]
    assert created.json()["parent_org_id"] == "org"
    for team_id in ("team", "second"):
        listed: Final = http.get("/management/v1/customer_members", params={"parent_team_id": team_id})
        assert [member["customer_member_id"] for member in listed.json()["data"]] == ["member"], team_id
    blocked: Final = http.delete("/management/v1/customer_teams/second")
    assert blocked.status_code == 409, blocked.text
    other: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "other"})
    assert other.status_code == 201, other.text
    foreign: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "foreign", "parent_org_id": "other"}
    )
    assert foreign.status_code == 201, foreign.text
    mixed: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "mixed",
            "parent_team_ids": ["team", "foreign"],
            "budget_override": {"max_budget": 1},
        },
    )
    assert mixed.status_code == 404, mixed.text
    fetched: Final = http.get("/management/v1/customer_organizations/org")
    assert fetched.json()["member_count"] == 1


def test_grouped_configuration_preserves_settings_swaps_names_and_selects_a_new_default() -> None:
    http: Final = client()
    org: Final = family(http)
    first, second = org["default_budgets"]
    response: Final = http.patch(
        "/management/v1/customer_organizations/org/default_budgets",
        json={
            "expected_revision": org["budget_groups_revision"],
            "default_budgets": [
                {"budget_id": first["budget_id"], "name": second["name"], "cap_policy": "pause_allow_increase"},
                {"budget_id": second["budget_id"], "name": first["name"]},
                {"name": "new", "budget_limits": [{"budget_duration": "1mo", "max_budget": 0}]},
            ],
            "default_budget_template": "new",
        },
    )
    assert response.status_code == 200, response.text
    saved: Final = response.json()
    assert saved["default_budgets"][0] == {**first, "name": second["name"], "cap_policy": "pause_allow_increase"}
    assert saved["default_budgets"][1] == {**second, "name": first["name"]}
    assert saved["default_budget_template_id"] == saved["default_budgets"][2]["budget_id"]
    assert saved["budget_groups_revision"] != org["budget_groups_revision"]
    assert http.get("/management/v1/customer_organizations/org").json() == saved


def test_grouped_configuration_rejects_stale_edits_without_losing_external_groups() -> None:
    http: Final = client()
    org: Final = family(http)
    external: Final = http.post("/management/v1/customer_organizations/org/default_budgets", json={"name": "external"})
    assert external.status_code == 201, external.text
    response: Final = http.patch(
        "/management/v1/customer_organizations/org/default_budgets",
        json={
            "expected_revision": org["budget_groups_revision"],
            "default_budgets": [{"budget_id": row["budget_id"], "name": row["name"]} for row in org["default_budgets"]],
            "default_budget_template": "standard",
        },
    )
    assert response.status_code == 409, response.text
    assert response.json()["type"].endswith("configuration-conflict")
    assert http.get("/management/v1/customer_organizations/org").json() == external.json()


def test_grouped_configuration_referenced_deletion_rejects_the_whole_save() -> None:
    http: Final = client()
    org: Final = family(http)
    member: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "m", "parent_team_ids": ["team"]}
    )
    assert member.status_code == 201, member.text
    before: Final = http.get("/management/v1/customer_organizations/org").json()
    response: Final = http.patch(
        "/management/v1/customer_organizations/org/default_budgets",
        json={
            "expected_revision": org["budget_groups_revision"],
            "default_budgets": [
                {"budget_id": org["default_budgets"][1]["budget_id"], "name": "changed", "max_budget": 999},
                {"name": "new", "max_budget": 1},
            ],
            "default_budget_template": "new",
        },
    )
    assert response.status_code == 409, response.text
    assert response.json()["type"].endswith("template-in-use")
    assert http.get("/management/v1/customer_organizations/org").json() == before


def test_grouped_configuration_keeps_window_clocks_and_invalidates_member_cache() -> None:
    cache: Final = UserApiKeyCache()
    http: Final = client(cache=cache)
    org: Final = family(http)
    member: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "m", "parent_team_ids": ["team"]}
    )
    assert member.status_code == 201, member.text
    first: Final = org["default_budgets"][0]
    windows: Final = http.patch(
        f"/management/v1/customer_organizations/org/default_budgets/{first['budget_id']}",
        json={
            "budget_limits": [
                {"budget_duration": "1mo", "max_budget": 10, "reset_at": "2027-01-15T00:00:00Z"},
                {"budget_duration": "12mo", "max_budget": 100, "reset_at": "2027-11-15T00:00:00Z"},
            ]
        },
    )
    assert windows.status_code == 200, windows.text
    advanced: Final = http.patch(
        f'/management/v1/customer_organizations/org/default_budgets/{first["budget_id"]}',
        json={
            "budget_limits": [
                {**row, "reset_at": "2028-01-15T00:00:00Z"}
                for row in windows.json()["default_budgets"][0]["budget_limits"]
            ]
        },
    )
    assert advanced.status_code == 200, advanced.text
    assert advanced.json()["budget_groups_revision"] == windows.json()["budget_groups_revision"]
    cache.set_cache(customer_member_cache_key("m"), "old-budget")
    saved: Final = http.patch(
        "/management/v1/customer_organizations/org/default_budgets",
        json={
            "expected_revision": windows.json()["budget_groups_revision"],
            "default_budgets": [
                {
                    "budget_id": first["budget_id"],
                    "name": "standard",
                    "budget_limits": [
                        {"budget_duration": "1mo", "max_budget": 15},
                        {"budget_duration": "12mo", "max_budget": 100},
                    ],
                }
            ],
            "default_budget_template": "standard",
        },
    )
    assert saved.status_code == 200, saved.text
    previous: Final = advanced.json()["default_budgets"][0]
    assert saved.json()["default_budgets"] == [
        {
            **previous,
            "budget_limits": [
                {**previous["budget_limits"][0], "max_budget": 15},
                previous["budget_limits"][1],
            ],
        }
    ]
    assert cache.get_cache(customer_member_cache_key("m")) is None
    assert (
        http.get("/management/v1/customer_members/m").json()["effective_budget"]["budget_limits"][0]["max_budget"] == 15
    )


@pytest.mark.parametrize("case", ("duplicate_id", "foreign_id", "duplicate_name", "missing_default"))
def test_invalid_grouped_configuration_leaves_existing_groups_untouched(case: str) -> None:
    http: Final = client()
    org: Final = family(http)
    first, second = org["default_budgets"]
    groups: Final = [
        {"budget_id": first["budget_id"], "name": "changed", "max_budget": 999},
        {
            "budget_id": first["budget_id"]
            if case == "duplicate_id"
            else "foreign"
            if case == "foreign_id"
            else second["budget_id"],
            "name": "changed" if case == "duplicate_name" else "second",
        },
    ]
    response: Final = http.patch(
        "/management/v1/customer_organizations/org/default_budgets",
        json={
            "expected_revision": org["budget_groups_revision"],
            "default_budgets": groups,
            "default_budget_template": "missing" if case == "missing_default" else "changed",
        },
    )
    assert response.status_code == 422, response.text
    assert http.get("/management/v1/customer_organizations/org").json() == org


class FailingBudgetInsertTable(Table):
    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> Record:
        from prisma.errors import UniqueViolationError

        if data.get("name") == "failed-insert":
            raise UniqueViolationError({})
        return await super().create(data, include)


def test_grouped_configuration_rolls_back_earlier_writes_when_a_later_database_write_fails() -> None:
    database: Final = Database()
    database.litellm_budgettable = FailingBudgetInsertTable("budget_id", database)
    http: Final = client(database=database)
    org: Final = family(http)
    before_budgets: Final = dict(database.litellm_budgettable.rows)
    response: Final = http.patch(
        "/management/v1/customer_organizations/org/default_budgets",
        json={
            "expected_revision": org["budget_groups_revision"],
            "default_budgets": [
                {"budget_id": org["default_budget_template_id"], "name": "changed", "max_budget": 999},
                {"name": "created-first", "max_budget": 1},
                {"name": "failed-insert", "max_budget": 2},
            ],
            "default_budget_template": "created-first",
        },
    )
    assert response.status_code == 409, response.text
    assert http.get("/management/v1/customer_organizations/org").json() == org
    assert database.litellm_budgettable.rows == before_budgets


def test_list_and_detail_expose_each_effective_budget_with_its_own_recorded_spend() -> None:
    database: Final = Database()
    http: Final = client(database=database)
    family(http)
    created: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "window-member", "parent_team_ids": ["team"],
            "budget_override": {
                "max_budget": 50, "budget_duration": "1mo",
                "budget_limits": [{"budget_duration": "1mo", "max_budget": 10}],
            },
        },
    )
    assert created.status_code == 201, created.text
    database.litellm_budgetwindowspend.rows = {
        "window-member": Record(
            entity_type="customer_member", entity_id="window-member", window_duration="1mo",
            window_start=NOW, spend=8,
        ),
    }
    detail: Final = http.get("/management/v1/customer_members/window-member")
    listing: Final = http.get("/management/v1/customer_members?parent_team_id=team")
    assert detail.status_code == listing.status_code == 200
    assert listing.json()["data"][0]["budget_usage"] == detail.json()["budget_usage"]
    assert [(item["spend"], item["max_budget"]) for item in detail.json()["budget_usage"]] == [(0, 50), (8, 10)]
