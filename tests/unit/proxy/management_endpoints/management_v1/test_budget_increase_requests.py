from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Final, cast

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.management_v1.budget_increase_requests import BudgetIncreaseRequest, router
from litellm.proxy.management_endpoints.management_v1.customers import get_customer_hierarchy
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW, Database, Record

if TYPE_CHECKING:
    from prisma import Prisma

PATH: Final = "/management/v1/budget_increase_requests"
EXPIRY: Final = NOW + timedelta(hours=1)
PAYLOAD: Final = {"amount": 50, "duration": "1mo", "expires_at": EXPIRY.isoformat(), "reason": "Finish a migration"}


class RequestDatabase(Database):
    def __init__(self) -> None:
        super().__init__()
        self.requests: Mapping[str, BudgetIncreaseRequest] = {}
        self.now = NOW
        self.fail_decision = False
        self.litellm_usertable.rows = {user: Record(user_id=user, spend=51, budget_template_id="standard") for user in ("alice", "bob", "admin")}
        self.litellm_budgettable.rows = {"standard": Record(budget_id="standard", is_profile=True, cap_policy="pause_allow_increase", budget_limits=({"budget_duration": "1mo", "max_budget": 100}, {"budget_duration": "12mo", "max_budget": 500}))}

    async def query_raw(self, query: str, *args: object) -> tuple[Mapping[str, object], ...]:
        if '"LiteLLM_BudgetIncreaseRequest"' not in query:
            return await super().query_raw(query, *args)
        if query.lstrip().startswith("INSERT"):
            row: Final = BudgetIncreaseRequest.model_validate(dict(zip(("id", "submitted_by", "amount", "duration", "expires_at", "reason", "created_at"), args)) | {"status": "pending_review"})
            self.requests = {**self.requests, row.id: row}
            return (row.model_dump(),)
        if query.lstrip().startswith("UPDATE"):
            if self.fail_decision:
                raise RuntimeError("decision storage unavailable")
            row: Final = self.requests[str(args[0])].model_copy(update={"status": args[1], "reviewed_by": args[2], "reviewed_at": TypeAdapter(datetime).validate_python(args[3]), "rejection_reason": args[4]})
            self.requests = {**self.requests, row.id: row}
            return (row.model_dump(),)
        if "COUNT(*)" in query:
            cutoff: Final = TypeAdapter(datetime).validate_python(args[1])
            return ({"count": sum(row.submitted_by == args[0] and row.created_at > cutoff for row in self.requests.values())},)
        if "WHERE pending_user_id" in query:
            return tuple(row.model_dump() for row in self.requests.values() if row.submitted_by == args[0] and row.status == "pending_review")
        if "WHERE id" in query:
            row: Final = self.requests.get(str(args[0]))
            return (row.model_dump(),) if row is not None and (args[1] is None or row.submitted_by == args[1]) else ()
        owner, limit, offset = args
        rows: Final = tuple(row.model_dump() for row in reversed(tuple(self.requests.values())) if owner is None or row.submitted_by == owner)
        return rows[int(str(offset)):int(str(offset)) + int(str(limit))]

    async def execute_raw(self, query: str, *args: object) -> int:
        cutoff: Final = TypeAdapter(datetime).validate_python(args[1])
        self.requests = {key: row.model_copy(update={"status": "expired"}) if row.submitted_by == args[0] and row.status == "pending_review" and row.expires_at <= cutoff else row for key, row in self.requests.items()}
        return 0

    @asynccontextmanager
    async def tx(self) -> AsyncIterator["RequestDatabase"]:
        snapshot: Final = self.requests
        try:
            async with super().tx():
                yield self
        except Exception:
            self.requests = snapshot
            raise


def rig() -> tuple[TestClient, RequestDatabase]:
    database: Final = RequestDatabase()
    hierarchy: Final = CustomerHierarchy(cast("Prisma", database), clock=lambda: database.now)
    app: Final = FastAPI()
    app.include_router(router)

    def caller(x_user: str = Header(default="alice"), x_role: LitellmUserRoles = Header(default=LitellmUserRoles.INTERNAL_USER)) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_id=x_user, user_role=x_role)

    def service() -> CustomerHierarchy:
        return hierarchy

    app.dependency_overrides[user_api_key_auth] = caller
    app.dependency_overrides[get_customer_hierarchy] = service
    return TestClient(app, raise_server_exceptions=False), database


ADMIN: Final = {"x-user": "admin", "x-role": "proxy_admin"}


@pytest.mark.parametrize("duration, expected", (("1mo", (150, 500)), ("12mo", (100, 550))))
def test_approval_uses_personal_override_once_and_never_changes_spend(duration: str, expected: tuple[int, int]) -> None:
    http, db = rig()
    submitted: Final = http.post(PATH, json={**PAYLOAD, "duration": duration})
    assert submitted.status_code == 201, submitted.text
    request_id: Final = submitted.json()["id"]
    approved: Final = http.put(f"{PATH}/{request_id}/approve", headers=ADMIN)
    assert approved.status_code == 200, approved.text
    assert approved.json()["reviewed_by"] == "admin"
    assert http.put(f"{PATH}/{request_id}/approve", headers=ADMIN).json() == approved.json()
    assert http.put(f"{PATH}/{request_id}/reject", headers=ADMIN, json={"reason": "No"}).status_code == 409
    own: Final = http.get(PATH + "/me").json()
    assert tuple(window["max_budget"] for window in own["windows"]) == expected
    assert db.litellm_usertable.rows["alice"].model_dump()["spend"] == 51
    assert http.get(PATH + "/me", headers={"x-user": "bob"}).json()["budget_override"] is None
    db.now = EXPIRY
    assert tuple(window["max_budget"] for window in http.get(PATH + "/me").json()["windows"]) == (100, 500)


def test_duplicates_owner_scope_rejection_and_resubmission() -> None:
    http, _ = rig()
    submitted: Final = http.post(PATH, json=PAYLOAD).json()
    assert http.post(PATH, json=PAYLOAD).json() == submitted
    assert http.post(PATH, json={**PAYLOAD, "amount": 99}).status_code == 409
    path: Final = f"{PATH}/{submitted['id']}"
    assert http.get(path, headers={"x-user": "bob"}).status_code == 404
    assert http.get(PATH, headers={"x-user": "bob"}).json() == {"requests": [], "has_more": False}
    assert http.get(PATH, headers=ADMIN).json()["requests"] == [submitted]
    rejected: Final = http.put(path + "/reject", headers=ADMIN, json={"reason": "Use the scheduled reset"})
    assert rejected.status_code == 200, rejected.text
    assert http.get(path).json()["rejection_reason"] == "Use the scheduled reset"
    assert http.get(PATH + "/me").json()["budget_override"] is None
    assert http.post(PATH, json=PAYLOAD).json()["id"] != submitted["id"]


@pytest.mark.parametrize("role", ("internal_user", "internal_user_viewer", "org_admin", "proxy_admin_viewer"))
@pytest.mark.parametrize("action", ("approve", "reject"))
def test_non_admin_roles_cannot_decide(role: str, action: str) -> None:
    http, _ = rig()
    request_id: Final = http.post(PATH, json=PAYLOAD).json()["id"]
    response: Final = http.put(f"{PATH}/{request_id}/{action}", headers={"x-user": "bob", "x-role": role}, json={"reason": "No"})
    assert response.status_code == 403, response.text
    assert http.get(f"{PATH}/{request_id}").json()["status"] == "pending_review"


def test_self_review_and_arbitrary_subject_are_denied() -> None:
    http, _ = rig()
    assert http.post(PATH, json={**PAYLOAD, "subject_id": "bob"}).status_code == 422
    submitted: Final = http.post(PATH, headers=ADMIN, json=PAYLOAD).json()
    assert http.put(f"{PATH}/{submitted['id']}/approve", headers=ADMIN).status_code == 403


@pytest.mark.parametrize("identity", ({"x-user": "__db_unavailable_fallback__"}, {"x-user": "missing"}, {"x-role": "internal_user_viewer"}, {"x-role": "proxy_admin_viewer"}))
def test_missing_or_readonly_requester_cannot_submit(identity: dict[str, str]) -> None:
    http, _ = rig()
    assert http.post(PATH, headers=identity, json=PAYLOAD).status_code in (403, 404)
    assert http.get(PATH, headers=ADMIN).json()["requests"] == []


@pytest.mark.parametrize("amount", (0, -1))
def test_nonpositive_amount_and_timezone_free_expiry_are_denied(amount: int) -> None:
    http, _ = rig()
    assert http.post(PATH, json={**PAYLOAD, "amount": amount}).status_code == 422
    assert http.post(PATH, json={**PAYLOAD, "expires_at": "2026-01-01T01:00:00"}).status_code == 422


def test_approval_rechecks_policy_and_pending_expiry() -> None:
    http, db = rig()
    request_id: Final = http.post(PATH, json=PAYLOAD).json()["id"]
    db.litellm_budgettable.rows = {"standard": db.litellm_budgettable.rows["standard"].model_copy(update={"cap_policy": "pause_until_reset"})}
    assert http.put(f"{PATH}/{request_id}/approve", headers=ADMIN).status_code == 409
    assert http.get(PATH + "/me").json()["budget_override"] is None
    db.now = EXPIRY
    assert http.get(f"{PATH}/{request_id}").json()["status"] == "expired"
    assert http.get(PATH).json()["requests"][0]["status"] == "expired"
    assert http.put(f"{PATH}/{request_id}/approve", headers=ADMIN).status_code == 409


def test_failed_decision_rolls_back_the_grant() -> None:
    http, db = rig()
    request_id: Final = http.post(PATH, json=PAYLOAD).json()["id"]
    db.fail_decision = True
    assert http.put(f"{PATH}/{request_id}/approve", headers=ADMIN).status_code == 500
    assert http.get(PATH + "/me").json()["budget_override"] is None
    assert http.get(f"{PATH}/{request_id}").json()["status"] == "pending_review"


def test_approval_preserves_an_existing_direct_grant() -> None:
    http, db = rig()
    request_id: Final = http.post(PATH, json=PAYLOAD).json()["id"]
    db.litellm_budgettable.rows = {
        **db.litellm_budgettable.rows,
        "personal": Record(budget_id="personal", temp_budget_increase=25, temp_budget_duration="12mo", temp_budget_expiry=EXPIRY),
    }
    db.litellm_usertable.rows = {
        **db.litellm_usertable.rows,
        "alice": db.litellm_usertable.rows["alice"].model_copy(update={"budget_id": "personal"}),
    }
    assert http.put(f"{PATH}/{request_id}/approve", headers=ADMIN).status_code == 409
    assert tuple(window["max_budget"] for window in http.get(PATH + "/me").json()["windows"]) == (100, 525)
    assert http.get(f"{PATH}/{request_id}").json()["status"] == "pending_review"


def test_request_cannot_outlast_the_personal_override() -> None:
    http, db = rig()
    db.litellm_budgettable.rows = {
        **db.litellm_budgettable.rows,
        "personal": Record(budget_id="personal", budget_limits=({"budget_duration": "1mo", "max_budget": 75},)),
    }
    db.litellm_usertable.rows = {
        **db.litellm_usertable.rows,
        "alice": db.litellm_usertable.rows["alice"].model_copy(update={"budget_id": "personal", "budget_override_expires_at": EXPIRY - timedelta(minutes=1)}),
    }
    rejected: Final = http.post(PATH, json=PAYLOAD)
    assert rejected.status_code == 409
    assert "cannot outlast" in rejected.json()["detail"]
    assert http.get(PATH).json()["requests"] == []


@pytest.mark.parametrize("override_expiry", (NOW - timedelta(seconds=1), NOW))
def test_expired_override_allows_profile_increase_without_reviving_old_limits(override_expiry: datetime) -> None:
    http, db = rig()
    db.litellm_budgettable.rows = {
        **db.litellm_budgettable.rows,
        "personal": Record(budget_id="personal", cap_policy="pause_until_reset", budget_limits=({"budget_duration": "1mo", "max_budget": 900},)),
    }
    db.litellm_usertable.rows = {
        **db.litellm_usertable.rows,
        "alice": db.litellm_usertable.rows["alice"].model_copy(update={"budget_id": "personal", "budget_override_expires_at": override_expiry}),
    }
    before: Final = http.get(PATH + "/me").json()
    assert tuple(window["max_budget"] for window in before["windows"]) == (100, 500)
    submitted: Final = http.post(PATH, json=PAYLOAD)
    assert submitted.status_code == 201, submitted.text
    approved: Final = http.put(f"{PATH}/{submitted.json()['id']}/approve", headers=ADMIN)
    assert approved.status_code == 200, approved.text
    own: Final = http.get(PATH + "/me").json()
    assert tuple(window["max_budget"] for window in own["windows"]) == (150, 500)
    assert own["effective_budget"]["cap_policy"] == "pause_allow_increase"
    assert db.litellm_usertable.rows["alice"].model_dump()["spend"] == 51
    db.now = EXPIRY
    assert tuple(window["max_budget"] for window in http.get(PATH + "/me").json()["windows"]) == (100, 500)


def test_new_grant_ignores_expiry_without_a_stored_override() -> None:
    http, db = rig()
    db.litellm_usertable.rows = {
        **db.litellm_usertable.rows,
        "alice": db.litellm_usertable.rows["alice"].model_copy(update={"budget_override_expires_at": NOW + timedelta(minutes=1)}),
    }
    submitted: Final = http.post(PATH, json=PAYLOAD)
    assert submitted.status_code == 201, submitted.text
    approved: Final = http.put(f"{PATH}/{submitted.json()['id']}/approve", headers=ADMIN)
    assert approved.status_code == 200, approved.text
    db.now = NOW + timedelta(minutes=2)
    assert tuple(window["max_budget"] for window in http.get(PATH + "/me").json()["windows"]) == (150, 500)


def test_expired_pending_request_can_be_replaced() -> None:
    http, db = rig()
    request_id: Final = http.post(PATH, json=PAYLOAD).json()["id"]
    db.now = EXPIRY
    replacement: Final = http.post(PATH, json={**PAYLOAD, "expires_at": (EXPIRY + timedelta(hours=1)).isoformat()})
    assert replacement.status_code == 201, replacement.text
    assert replacement.json()["id"] != request_id
    assert http.get(f"{PATH}/{request_id}").json()["status"] == "expired"


def test_submission_quota_counts_expired_requests_but_preserves_pending_retries() -> None:
    http, db = rig()
    for minute in range(10):
        db.now = NOW + timedelta(minutes=minute)
        payload: Final = {**PAYLOAD, "expires_at": (db.now + timedelta(seconds=30)).isoformat()}
        submitted: Final = http.post(PATH, json=payload)
        assert submitted.status_code == 201, submitted.text
        assert http.post(PATH, json=payload).json() == submitted.json()
    db.now = NOW + timedelta(minutes=10)
    assert http.post(PATH, json=PAYLOAD).status_code == 429
    db.now = NOW + timedelta(hours=1)
    available: Final = http.post(PATH, json={**PAYLOAD, "expires_at": (db.now + timedelta(hours=1)).isoformat()})
    assert available.status_code == 201, available.text
