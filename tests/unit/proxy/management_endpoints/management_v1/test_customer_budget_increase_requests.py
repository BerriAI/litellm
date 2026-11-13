from typing import Final

from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles
from tests.unit.proxy.management_endpoints.management_v1.test_customers import client

REQUESTS: Final = "/management/v1/customer_budget_increase_requests"


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
                    "is_default": True,
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
    filed: Final = http.post(REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250})
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
    filed: Final = http.post(REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250})
    assert filed.status_code == 422, filed.text
    assert filed.json()["title"] == "Increase not allowed"
    missing: Final = http.post(REQUESTS, json={"customer_member_id": "nobody", "requested_max_budget": 250})
    assert missing.status_code == 404, missing.text


def test_only_proxy_admins_review_requests() -> None:
    admin: Final = client()
    member_id: Final = _paused_member(admin, "pause_allow_increase")
    user: Final = client(role=LitellmUserRoles.INTERNAL_USER)
    filed: Final = admin.post(REQUESTS, json={"customer_member_id": member_id, "requested_max_budget": 250})
    assert filed.status_code == 201, filed.text
    request_id: Final = filed.json()["request_id"]
    assert user.get(REQUESTS).status_code == 403
    assert user.put(f"{REQUESTS}/{request_id}/approve").status_code == 403
    assert user.put(f"{REQUESTS}/{request_id}/reject").status_code == 403
