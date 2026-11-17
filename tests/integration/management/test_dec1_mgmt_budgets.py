"""Budget management API contracts: /budget/new, /budget/update, /budget/info, /budget/list, /budget/delete.

`LiteLLM_BudgetTable` moves behind a Rust store this week (draft PR #52176, thread C of the DB W3 brief). These
tests pin the management answers read from `litellm/proxy/management_endpoints/budget_management_endpoints.py`:
the negative value and duplicate id refusals and their `{"detail": {"error": ...}}` envelope (the handlers raise
`HTTPException` without wrapping it), the audit columns written on create, the reset time seeded from
`budget_duration`, the info and list shapes, and the role gates in front of each route. Existing files already
cover duration rescheduling, partial update preservation and list sorting, so those are not repeated.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value
from tests.integration.management.dec1_mgmt_support import (
    NOT_ALLOWED_ACCESS,
    PROXY_ADMIN_ONLY_PREFIX,
    body_of,
    budget_rows,
    caller,
    delete_budget_if_present,
    detail_error,
    get,
    post,
    proxy_error,
    unique,
)

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.parametrize("field", ["max_budget", "soft_budget"])
def test_budget_new_with_a_negative_amount_is_400_and_writes_nothing(gateway: Gateway, field: str) -> None:
    """A negative max_budget or soft_budget must be refused with 400 and detail.error
    "<field> must be a non-negative finite number. Received: -1.0", and no budget row is written for the id."""
    budget_id: Final = unique("neg")
    response: Final = post(gateway, "/budget/new", {"budget_id": budget_id, field: -1.0})
    assert response.status_code == 400, response.text
    assert detail_error(response) == f"{field} must be a non-negative finite number. Received: -1.0"
    assert budget_rows(budget_id) == []


def test_budget_new_with_a_taken_budget_id_is_400_naming_the_id(gateway: Gateway) -> None:
    """Creating a second budget with an existing budget_id must answer 400 with detail.error
    "Budget with id '<id>' already exists." and leave the first row's max_budget unchanged."""
    with gateway.scenario() as scenario:
        budget_id: Final = unique("dup")
        gateway.post("/budget/new", {"budget_id": budget_id, "max_budget": 1.0})
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        response: Final = post(gateway, "/budget/new", {"budget_id": budget_id, "max_budget": 2.0})
        assert response.status_code == 400, response.text
        assert detail_error(response) == f"Budget with id '{budget_id}' already exists."
        assert budget_rows(budget_id)[0]["max_budget"] == 1.0


def test_budget_new_without_budget_id_generates_one_and_returns_the_row(gateway: Gateway) -> None:
    """A budget created without budget_id gets a generated id. The 200 body is the created row itself, carrying
    budget_id, max_budget and tpm_limit as sent, and that row exists in LiteLLM_BudgetTable."""
    with gateway.scenario() as scenario:
        created: Final = gateway.post("/budget/new", {"max_budget": 5.0, "tpm_limit": 10})
        budget_id: Final = created["budget_id"]
        assert isinstance(budget_id, str) and budget_id
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        assert created["max_budget"] == 5.0
        assert created["tpm_limit"] == 10
        row: Final = budget_rows(budget_id)[0]
        assert row["max_budget"] == 5.0
        assert row["tpm_limit"] == 10


def test_budget_new_writes_the_caller_user_id_as_created_by_and_updated_by(gateway: Gateway) -> None:
    """A proxy admin user's key creates a budget. Both created_by and updated_by on the row must be that user's id,
    not the proxy admin name fallback, which is only used when the caller has no user_id."""
    with gateway.scenario() as scenario:
        admin: Final = caller(scenario, "proxy_admin")
        response: Final = post(gateway, "/budget/new", {"max_budget": 1.0}, key=admin.key)
        assert response.status_code == 200, response.text
        budget_id: Final = body_of(response)["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        row: Final = budget_rows(budget_id)[0]
        assert row["created_by"] == admin.user_id
        assert row["updated_by"] == admin.user_id


def test_budget_new_with_a_duration_seeds_budget_reset_at(gateway: Gateway) -> None:
    """With budget_duration "30d" and no budget_reset_at, the handler must seed budget_reset_at with the next reset
    time, so the row has a non null budget_reset_at; a budget without duration keeps it null."""
    with gateway.scenario() as scenario:
        timed: Final = scenario.budget(max_budget=1.0, budget_duration="30d")
        untimed: Final = scenario.budget(max_budget=1.0)
        assert budget_rows(timed)[0]["budget_reset_at"] is not None
        assert budget_rows(timed)[0]["budget_duration"] == "30d"
        assert budget_rows(untimed)[0]["budget_reset_at"] is None


def test_budget_new_with_an_unparseable_duration_is_400(gateway: Gateway) -> None:
    """validate_budget_duration refuses a duration it cannot parse with 400 before anything is written.
    UNVERIFIED: the exact detail.error text comes from `budget_duration_error` in
    `litellm/proxy/common_utils/timezone_utils.py`; this test only pins the status and that no row exists."""
    budget_id: Final = unique("bad-duration")
    response: Final = post(gateway, "/budget/new", {"budget_id": budget_id, "budget_duration": "fortnightly"})
    assert response.status_code == 400, response.text
    assert budget_rows(budget_id) == []


def test_budget_update_without_budget_id_is_400(gateway: Gateway) -> None:
    """/budget/update needs the id of the row to change: a body without budget_id must be refused with 400 and
    detail.error "budget_id is required"."""
    response: Final = post(gateway, "/budget/update", {"max_budget": 3.0})
    assert response.status_code == 400, response.text
    assert detail_error(response) == "budget_id is required"


def test_budget_update_with_a_negative_max_budget_is_400_and_keeps_the_row(gateway: Gateway) -> None:
    """Updating an existing budget to max_budget -2.0 must be refused with 400 and detail.error
    "max_budget must be a non-negative finite number. Received: -2.0", and the stored max_budget stays 1.0."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=1.0)
        response: Final = post(gateway, "/budget/update", {"budget_id": budget_id, "max_budget": -2.0})
        assert response.status_code == 400, response.text
        assert detail_error(response) == "max_budget must be a non-negative finite number. Received: -2.0"
        assert budget_rows(budget_id)[0]["max_budget"] == 1.0


def test_budget_update_writes_updated_by_and_keeps_created_by(gateway: Gateway) -> None:
    """A budget created through the master key is updated by a proxy admin user's key. updated_by must become that
    user's id while created_by keeps the creator's value."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=1.0)
        creator: Final = budget_rows(budget_id)[0]["created_by"]
        admin: Final = caller(scenario, "proxy_admin")
        response: Final = post(gateway, "/budget/update", {"budget_id": budget_id, "max_budget": 2.0}, key=admin.key)
        assert response.status_code == 200, response.text
        row: Final = budget_rows(budget_id)[0]
        assert row["updated_by"] == admin.user_id
        assert row["created_by"] == creator
        assert row["max_budget"] == 2.0


def test_budget_info_with_an_empty_list_is_400(gateway: Gateway) -> None:
    """POST /budget/info with budgets [] must answer 400 with detail.error
    "Specify list of budget id's to query. Passed in=[]"."""
    response: Final = post(gateway, "/budget/info", {"budgets": []})
    assert response.status_code == 400, response.text
    assert detail_error(response) == "Specify list of budget id's to query. Passed in=[]"


def test_budget_info_returns_found_rows_and_silently_skips_unknown_ids(gateway: Gateway) -> None:
    """/budget/info with one existing and one unknown id must answer 200 with a list holding only the existing
    budget; an unknown id is not an error on this route, unlike /customer/info or /tag/info."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=4.0)
        response: Final = post(gateway, "/budget/info", {"budgets": [budget_id, unique("ghost")]})
        assert response.status_code == 200, response.text
        rows: Final = response.json()
        assert isinstance(rows, list)
        assert [object_value(row)["budget_id"] for row in rows] == [budget_id]
        assert object_value(rows[0])["max_budget"] == 4.0


def test_budget_list_includes_a_new_budget_for_the_proxy_admin(gateway: Gateway) -> None:
    """GET /budget/list for the master key returns every budget row as a JSON list, so a budget created in this
    test appears in it with its max_budget."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=6.0)
        response: Final = get(gateway, "/budget/list")
        assert response.status_code == 200, response.text
        listed: Final = {object_value(row)["budget_id"]: object_value(row) for row in response.json()}
        assert listed[budget_id]["max_budget"] == 6.0


def test_budget_list_is_readable_by_a_proxy_admin_viewer(gateway: Gateway) -> None:
    """/budget/list is in admin_viewer_routes and its handler accepts admin view, so a proxy_admin_viewer key must
    get 200 and see a budget created by the admin."""
    with gateway.scenario() as scenario:
        viewer: Final = caller(scenario, "proxy_admin_viewer")
        budget_id: Final = scenario.budget(max_budget=1.0)
        response: Final = get(gateway, "/budget/list", key=viewer.key)
        assert response.status_code == 200, response.text
        assert budget_id in {object_value(row)["budget_id"] for row in response.json()}


@pytest.mark.parametrize(
    ("method", "route"),
    [("POST", "/budget/new"), ("POST", "/budget/update"), ("POST", "/budget/delete"), ("GET", "/budget/list")],
)
def test_budget_routes_are_401_for_an_internal_user_at_the_route_gate(
    gateway: Gateway, method: str, route: str
) -> None:
    """No budget route is in the internal user route lists, so an internal user's key must get the route gate 401
    "Authentication Error, Only proxy admin can be used ... Route=<route>. Your role=internal_user." on each."""
    with gateway.scenario() as scenario:
        internal: Final = caller(scenario, "internal_user")
        body: Final = None if method == "GET" else {"budget_id": unique("x"), "id": unique("x")}
        response: Final = gateway.request(method, route, body, key=internal.key)
        assert response.status_code == 401, response.text
        assert str(proxy_error(response)["message"]).startswith(
            f"{PROXY_ADMIN_ONLY_PREFIX}Route={route}. Your role=internal_user."
        )


@pytest.mark.parametrize("route", ["/budget/new", "/budget/update", "/budget/delete"])
def test_budget_writes_are_403_for_a_proxy_admin_viewer(gateway: Gateway, route: str) -> None:
    """A proxy_admin_viewer posting to a budget write route must get 403 "user not allowed to access this route,
    role= proxy_admin_viewer. Trying to access: <route>" from the viewer gate."""
    with gateway.scenario() as scenario:
        viewer: Final = caller(scenario, "proxy_admin_viewer")
        budget_id: Final = scenario.budget(max_budget=1.0)
        response: Final = post(
            gateway, route, {"budget_id": budget_id, "id": budget_id, "max_budget": 9.0}, key=viewer.key
        )
        assert response.status_code == 403, response.text
        assert proxy_error(response)["message"] == (
            f"user not allowed to access this route, role= proxy_admin_viewer. Trying to access: {route}"
        )
        assert budget_rows(budget_id)[0]["max_budget"] == 1.0


def test_budget_delete_through_an_allowed_route_key_still_needs_proxy_admin(gateway: Gateway) -> None:
    """An internal user's key with allowed_routes ["/budget/delete"] passes the route gate, so the handler's own
    check decides: it must answer 400 (not 403) with detail.error
    "Admin-only endpoint. Not allowed to access this., your role=internal_user" and keep the row."""
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user_id, allowed_routes=["/budget/delete"])
        budget_id: Final = scenario.budget(max_budget=1.0)
        response: Final = post(gateway, "/budget/delete", {"id": budget_id}, key=key)
        assert response.status_code == 400, response.text
        assert detail_error(response) == f"{NOT_ALLOWED_ACCESS}, your role=internal_user"
        assert len(budget_rows(budget_id)) == 1


def test_budget_delete_returns_the_deleted_row(gateway: Gateway) -> None:
    """Deleting a budget answers 200 with the deleted row (its budget_id and max_budget), and the row is gone."""
    budget_id: Final = unique("del")
    gateway.post("/budget/new", {"budget_id": budget_id, "max_budget": 2.5})
    response: Final = post(gateway, "/budget/delete", {"id": budget_id})
    assert response.status_code == 200, response.text
    deleted: Final = body_of(response)
    assert deleted["budget_id"] == budget_id
    assert deleted["max_budget"] == 2.5
    assert budget_rows(budget_id) == []


def test_budget_delete_of_a_budget_linked_to_a_customer_is_refused_and_keeps_both_rows(gateway: Gateway) -> None:
    """A budget referenced by a customer's budget_id cannot simply vanish: the delete must fail and both the budget
    row and the customer link must remain. UNVERIFIED: the handler has no reference check, so the refusal comes
    from the end user table's foreign key and its status (likely 500) and body are not pinned; look at
    `delete_budget` and the `LiteLLM_EndUserTable.budget_id` relation in `schema.prisma`."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=1.0)
        identity: Final = unique("linked")
        gateway.post("/customer/new", {"user_id": identity, "budget_id": budget_id})
        scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [identity]})
        response: Final = post(gateway, "/budget/delete", {"id": budget_id})
        assert response.status_code >= 400, response.text
        assert len(budget_rows(budget_id)) == 1
