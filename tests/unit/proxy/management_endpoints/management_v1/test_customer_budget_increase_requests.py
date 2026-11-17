from typing import Final

from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from tests.unit.proxy.management_endpoints.management_v1.test_customers import client

REQUESTS: Final = "/management/v1/customer_budget_increase_requests"
EXPIRES: Final = "2026-02-01T00:00:00Z"


def _as(http: TestClient, caller: UserAPIKeyAuth) -> TestClient:
    http.app.dependency_overrides[user_api_key_auth] = lambda: caller  # pyright: ignore[reportAttributeAccessIssue]  # TestClient.app is the FastAPI app
    return http


def _paused_member(http: TestClient, cap_policy: str | None) -> str:
    org: Final = http.post(
        "/management/v1/customer_organizations",
        json={
            "customer_org_id": "paused",
            "default_budgets": [
                {
                    "name": "monthly",
                    "max_budget": 100,
                    "budget_duration": "30d",
                    **({"cap_policy": cap_policy} if cap_policy is not None else {}),
                }
            ],
        },
    )
    assert org.status_code == 201, org.text
    team: Final = http.post(
        "/management/v1/customer_teams", json={"customer_team_id": "paused-team", "parent_org_id": "paused"}
    )
    assert team.status_code == 201, team.text
    member: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "paused-member", "parent_team_ids": ["paused-team"]},
    )
    assert member.status_code == 201, member.text
    assert member.json()["effective_budget"]["cap_policy"] == cap_policy
    return "paused-member"


def test_request_waits_for_review_and_an_approval_lands_as_the_members_override() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, "pause_allow_increase")
    filed: Final = http.post(
        REQUESTS,
        json={
            "customer_member_id": member_id,
            "requested_max_budget": 250,
            "expires_at": "2026-02-01T00:00:00Z",
            "reason": "quarter close",
        },
    )
    assert filed.status_code == 201, filed.text
    assert filed.json()["status"] == "pending_review"
    assert filed.json()["submitted_by"] == "admin"
    request_id: Final = filed.json()["request_id"]

    queue: Final = http.get(REQUESTS, params={"status": "pending_review"})
    assert queue.status_code == 200, queue.text
    assert [row["request_id"] for row in queue.json()["data"]] == [request_id]
    assert queue.json()["meta"]["total_count"] == 1

    approved: Final = http.put(f"{REQUESTS}/{request_id}/approve")
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved"
    assert approved.json()["reviewed_by"] == "admin"
    assert approved.json()["reviewed_at"] is not None

    member: Final = http.get(f"/management/v1/customer_members/{member_id}")
    assert member.json()["budget_override"]["max_budget"] == 250
    assert member.json()["budget_override_expires_at"] == "2026-02-01T00:00:00Z"
    assert member.json()["effective_budget"]["max_budget"] == 250
    assert member.json()["effective_budget"]["cap_policy"] == "pause_allow_increase"

    again: Final = http.put(f"{REQUESTS}/{request_id}/approve")
    assert again.status_code == 409, again.text
    assert http.get(REQUESTS, params={"status": "pending_review"}).json()["meta"]["total_count"] == 0


def test_rejection_keeps_the_member_paused_and_records_the_notes() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, "pause_allow_increase")
    filed: Final = http.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert filed.status_code == 201, filed.text

    rejected: Final = http.put(
        f"{REQUESTS}/{filed.json()['request_id']}/reject", json={"review_notes": "not this quarter"}
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["status"] == "rejected"
    assert rejected.json()["review_notes"] == "not this quarter"
    member: Final = http.get(f"/management/v1/customer_members/{member_id}")
    assert member.json()["budget_override"] is None
    assert member.json()["effective_budget"]["max_budget"] == 100
    assert http.put(f"{REQUESTS}/{filed.json()['request_id']}/approve").status_code == 409


def test_members_whose_budget_pauses_until_reset_cannot_file_a_request() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, None)
    filed: Final = http.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert filed.status_code == 422, filed.text
    assert filed.json()["title"] == "Increase not allowed"
    missing: Final = http.post(
        REQUESTS, json={"customer_member_id": "nobody", "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert missing.status_code == 404, missing.text


def test_only_proxy_admins_review_requests() -> None:
    admin: Final = client()
    member_id: Final = _paused_member(admin, "pause_allow_increase")
    user: Final = client(role=LitellmUserRoles.INTERNAL_USER)
    filed: Final = admin.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert filed.status_code == 201, filed.text
    request_id: Final = filed.json()["request_id"]
    assert user.get(REQUESTS).status_code == 403
    assert user.put(f"{REQUESTS}/{request_id}/approve").status_code == 403
    assert user.put(f"{REQUESTS}/{request_id}/reject").status_code == 403


def test_a_request_needs_a_future_expiry_and_one_pending_request_per_member() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, "pause_allow_increase")
    permanent: Final = http.post(REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250})
    assert permanent.status_code == 422, permanent.text
    past: Final = http.post(
        REQUESTS,
        json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": "2025-12-31T00:00:00Z"},
    )
    assert past.status_code == 422, past.text
    assert past.json()["title"] == "Expiry in past"
    assert (
        http.post(
            REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
        ).status_code
        == 201
    )
    duplicate: Final = http.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 300, "expires_at": EXPIRES}
    )
    assert duplicate.status_code == 409, duplicate.text
    assert duplicate.json()["title"] == "Request already pending"


def test_a_key_pinned_to_another_organization_cannot_file_for_the_member() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, "pause_allow_increase")
    other_org: Final = _as(
        http,
        UserAPIKeyAuth(user_id="svc", user_role=LitellmUserRoles.PROXY_ADMIN, metadata={"customer_org_id": "other"}),
    )
    denied: Final = other_org.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert denied.status_code == 403, denied.text
    assert denied.json()["title"] == "Key not in organization"
    own_org: Final = _as(
        http,
        UserAPIKeyAuth(user_id="svc", user_role=LitellmUserRoles.PROXY_ADMIN, metadata={"customer_org_id": "paused"}),
    )
    allowed: Final = own_org.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert allowed.status_code == 201, allowed.text


def test_submitter_is_recorded_by_user_or_key_alias_never_the_raw_key() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, "pause_allow_increase")
    bearer: Final = _as(
        http,
        UserAPIKeyAuth(
            user_id=None, user_role=LitellmUserRoles.PROXY_ADMIN, api_key="opaque-bearer", key_alias="svc-key"
        ),
    )
    filed: Final = bearer.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert filed.status_code == 201, filed.text
    assert filed.json()["submitted_by"] == "svc-key"
    assert "opaque-bearer" not in filed.text


def test_an_approval_does_not_revive_an_expired_override_or_outlive_an_active_one() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, "pause_allow_increase")
    expired: Final = http.patch(
        f"/management/v1/customer_members/{member_id}",
        json={
            "budget_override": {"max_budget": 50, "allowed_models": ["gpt-5.1"]},
            "budget_override_expires_at": "2025-12-01T00:00:00Z",
        },
    )
    assert expired.status_code == 200, expired.text
    filed: Final = http.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert filed.status_code == 201, filed.text
    assert http.put(f"{REQUESTS}/{filed.json()['request_id']}/approve").status_code == 200
    member: Final = http.get(f"/management/v1/customer_members/{member_id}").json()
    assert member["budget_override"]["max_budget"] == 250
    assert not member["budget_override"]["allowed_models"], member["budget_override"]
    assert member["budget_override_expires_at"] == EXPIRES
    assert not member["effective_budget"]["allowed_models"], member["effective_budget"]

    active: Final = http.patch(
        f"/management/v1/customer_members/{member_id}",
        json={
            "budget_override": {"max_budget": 50, "allowed_models": ["gpt-5.1"]},
            "budget_override_expires_at": "2026-01-15T00:00:00Z",
        },
    )
    assert active.status_code == 200, active.text
    second: Final = http.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 300, "expires_at": EXPIRES}
    )
    assert second.status_code == 201, second.text
    assert http.put(f"{REQUESTS}/{second.json()['request_id']}/approve").status_code == 200
    kept: Final = http.get(f"/management/v1/customer_members/{member_id}").json()
    assert kept["budget_override"]["max_budget"] == 300
    assert kept["budget_override"]["allowed_models"] == ["gpt-5.1"]
    assert kept["budget_override_expires_at"] == "2026-01-15T00:00:00Z"


def test_an_approved_increase_replaces_the_template_windows_as_the_members_cap() -> None:
    http: Final = client()
    org: Final = http.post(
        "/management/v1/customer_organizations",
        json={
            "customer_org_id": "windowed",
            "default_budgets": [
                {
                    "name": "monthly",
                    "budget_limits": [{"budget_duration": "1mo", "max_budget": 100}],
                    "cap_policy": "pause_allow_increase",
                }
            ],
        },
    )
    assert org.status_code == 201, org.text
    assert (
        http.post(
            "/management/v1/customer_teams", json={"customer_team_id": "windowed-team", "parent_org_id": "windowed"}
        ).status_code
        == 201
    )
    member: Final = http.post(
        "/management/v1/customer_members",
        json={"customer_member_id": "windowed-member", "parent_team_ids": ["windowed-team"]},
    )
    assert member.status_code == 201, member.text
    filed: Final = http.post(
        REQUESTS, json={"customer_member_id": "windowed-member", "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert filed.status_code == 201, filed.text
    assert http.put(f"{REQUESTS}/{filed.json()['request_id']}/approve").status_code == 200
    effective: Final = http.get("/management/v1/customer_members/windowed-member").json()["effective_budget"]
    assert effective["max_budget"] == 250, effective
    assert not effective["budget_limits"], effective


def test_a_request_is_reviewed_exactly_once_and_a_late_reject_cannot_undo_an_approval() -> None:
    http: Final = client()
    member_id: Final = _paused_member(http, "pause_allow_increase")
    filed: Final = http.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250, "expires_at": EXPIRES}
    )
    assert filed.status_code == 201, filed.text
    request_id: Final = filed.json()["request_id"]
    assert http.put(f"{REQUESTS}/{request_id}/approve").status_code == 200
    late_reject: Final = http.put(f"{REQUESTS}/{request_id}/reject")
    assert late_reject.status_code == 409, late_reject.text
    assert http.get(f"{REQUESTS}/{request_id}").json()["status"] == "approved"
    assert http.get(f"/management/v1/customer_members/{member_id}").json()["effective_budget"]["max_budget"] == 250
    second: Final = http.post(
        REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 300, "expires_at": EXPIRES}
    )
    assert second.status_code == 201, second.text
    assert http.put(f"{REQUESTS}/{second.json()['request_id']}/reject").status_code == 200
    assert http.put(f"{REQUESTS}/{second.json()['request_id']}/approve").status_code == 409
    assert http.get(f"/management/v1/customer_members/{member_id}").json()["effective_budget"]["max_budget"] == 250


def test_plain_virtual_keys_may_reach_the_increase_request_route() -> None:
    from litellm.proxy._types import LiteLLMRoutes
    from litellm.proxy.auth.route_checks import RouteChecks

    assert RouteChecks.check_route_access(route=REQUESTS, allowed_routes=LiteLLMRoutes.self_managed_routes.value)
    assert not RouteChecks.check_route_access(
        route="/management/v1/customer_members", allowed_routes=LiteLLMRoutes.self_managed_routes.value
    )
