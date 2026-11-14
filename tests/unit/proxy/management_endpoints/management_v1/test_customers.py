import asyncio
from typing import TYPE_CHECKING, Final, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import customer_member_cache_key
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.customers import get_customer_hierarchy, get_user_api_key_cache
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW, Database

if TYPE_CHECKING:
    from prisma import Prisma


def client(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN, cache: UserApiKeyCache | None = None) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    auth: Final = UserAPIKeyAuth(user_id="admin", user_role=role)
    hierarchy: Final = CustomerHierarchy(
        cast("Prisma", Database()), clock=lambda: NOW
    )  # cast-ok: in-memory Prisma edge
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[get_customer_hierarchy] = lambda: hierarchy
    app.dependency_overrides[get_user_api_key_cache] = lambda: cache if cache is not None else UserApiKeyCache()
    return TestClient(app)


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
                    "is_default": True,
                },
                {"name": "small", "max_budget": 20, "rpm_limit": 8},
            ],
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


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
@pytest.mark.parametrize(
    "method,path,body",
    (
        ("POST", "/customer_organizations", {}),
        ("GET", "/customer_organizations", None),
        ("GET", "/customer_organizations/id", None),
        ("PATCH", "/customer_organizations/id", {}),
        ("DELETE", "/customer_organizations/id", None),
        ("POST", "/customer_teams", {"parent_org_id": "id"}),
        ("GET", "/customer_teams", None),
        ("GET", "/customer_teams/id", None),
        ("PATCH", "/customer_teams/id", {}),
        ("DELETE", "/customer_teams/id", None),
        ("POST", "/customer_members", {"parent_team_id": "id"}),
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
            "parent_team_id": "team",
        },
    )
    assert default.status_code == 201, default.text
    assert default.json()["budget_override_id"] is None
    assert default.json()["budget_id"] == default_id
    override: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "override",
            "parent_team_id": "team",
            "budget_id": other_id,
            "budget_override": {"max_budget": 3, "allowed_models": []},
        },
    )
    assert override.status_code == 201, override.text
    expired: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "expired",
            "parent_team_id": "team",
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
        "model_tpm_limit": None,
        "model_rpm_limit": None,
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
            "parent_team_id": "team",
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
            "parent_team_id": "team",
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
    default_id: Final = org["default_budget_id"]
    selected_id: Final = org["default_budgets"][1]["budget_id"]
    member: Final = http.post(
        "/management/v1/customer_members",
        json={
            "customer_member_id": "member",
            "parent_team_id": "team",
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
    added: Final = http.post(
        "/management/v1/customer_organizations/org/default_budgets", json={"name": "new", "is_default": True}
    )
    assert added.status_code == 201, added.text
    assert added.json()["default_budget_id"] != default_id
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
    missing: Final = http.post("/management/v1/customer_members", json={"parent_team_id": "empty-team"})
    assert missing.status_code == 422, missing.text
    wrong: Final = http.post(
        "/management/v1/customer_members",
        json={
            "parent_team_id": "empty-team",
            "budget_id": org["default_budget_id"],
            "budget_override": {"max_budget": 2},
        },
    )
    assert wrong.status_code == 422, wrong.text
    custom: Final = http.post(
        "/management/v1/customer_members", json={"parent_team_id": "empty-team", "budget_override": {"max_budget": 2}}
    )
    assert custom.status_code == 201, custom.text
    assert custom.json()["budget_id"] is None
    assert custom.json()["effective_budget"]["max_budget"] == 2


@pytest.mark.parametrize(
    "entity,body",
    (
        ("customer_organizations", {}),
        ("customer_teams", {"parent_org_id": "org"}),
        ("customer_members", {"parent_team_id": "team"}),
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
        "/management/v1/customer_members", json={"customer_member_id": "member", "parent_team_id": "team"}
    )
    assert member.status_code == 201, member.text
    for path in ("customer_organizations/org", "customer_teams/team", "customer_members/member"):
        fetched: Final = http.get("/management/v1/" + path)
        assert fetched.status_code == 200, fetched.text
        assert fetched.json()["spend"] == 0
        assert fetched.json()["budget_reset_at"] is None
    for path in ("customer_organizations/org", "customer_teams/team"):
        blocked: Final = http.delete("/management/v1/" + path)
        assert blocked.status_code == 409, blocked.text
    for path in ("customer_members/member", "customer_teams/team", "customer_organizations/org"):
        deleted: Final = http.delete("/management/v1/" + path)
        assert deleted.status_code == 204, deleted.text
        absent: Final = http.get("/management/v1/" + path)
        assert absent.status_code == 404, absent.text
    assert org["budget_id"] is not None


@pytest.mark.parametrize(
    "templates",
    (
        [{"name": "a"}, {"name": "b"}],
        [{"name": "a", "is_default": True}, {"name": "b", "is_default": True}],
        [{"name": "a"}, {"name": "a", "is_default": True}],
    ),
)
def test_invalid_template_configuration_does_not_create_an_organization(templates: list[dict[str, object]]) -> None:
    http: Final = client()
    response: Final = http.post(
        "/management/v1/customer_organizations", json={"customer_org_id": "org", "default_budgets": templates}
    )
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
    assert created.json()["default_budget_id"] == created.json()["default_budgets"][0]["budget_id"]
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
        ("customer_members", {"customer_member_id": "duplicate", "parent_team_id": "team"}),
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
    field: Final = "parent_org_id" if entity == "customer_teams" else "parent_team_id"
    response: Final = client().post("/management/v1/" + entity, json={field: "missing"})
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
            "parent_team_id": "team",
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
    assert patched.json()["budget_override_id"] != organization["default_budget_id"]
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


def test_first_added_template_is_default_and_default_cannot_be_unset_without_replacement() -> None:
    http: Final = client()
    created: Final = http.post("/management/v1/customer_organizations", json={"customer_org_id": "org"})
    assert created.status_code == 201, created.text
    first: Final = http.post("/management/v1/customer_organizations/org/default_budgets", json={"name": "first"})
    assert first.status_code == 201, first.text
    first_id: Final = first.json()["default_budget_id"]
    second: Final = http.post("/management/v1/customer_organizations/org/default_budgets", json={"name": "second"})
    assert second.status_code == 201, second.text
    assert second.json()["default_budget_id"] == first_id
    invalid: Final = http.patch(
        f"/management/v1/customer_organizations/org/default_budgets/{first_id}", json={"is_default": False}
    )
    assert invalid.status_code == 422, invalid.text
    still_default: Final = http.get("/management/v1/customer_organizations/org")
    assert still_default.json() == second.json()


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
            "parent_team_id": "team",
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
            "/management/v1/customer_members", json={"customer_member_id": "m", "parent_team_id": "team"}
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
            "/management/v1/customer_members", json={"customer_member_id": identifier, "parent_team_id": "team"}
        )
        assert created.status_code == 201, created.text
    second_team: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "second-team", "parent_org_id": "org"}
    )
    assert second_team.status_code == 201, second_team.text
    third_member: Final = http.post(
        "/management/v1/customer_members", json={"customer_member_id": "member-c", "parent_team_id": "second-team"}
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


def family_default_budget_id(http: TestClient) -> str:
    organization: Final = http.get("/management/v1/customer_organizations/org").json()
    return str(organization["default_budget_id"])
