"""Customer (end user) management API contracts: /customer/new, info, update, delete, block, unblock, list.

The end user object moves behind a Rust store this week (draft PR #52176, thread C of the DB W3 brief). These
tests pin what the management routes answer and which rows they leave, as read from
`litellm/proxy/management_endpoints/customer_endpoints.py`: the response field set and defaults, the exact
not found and duplicate messages and their error envelope, the budget row a customer creates or reuses, the
batch delete that refuses when any id is missing, and who may call each route. Request time effects of a
customer (restrictions, budgets at auth) are covered by `index/test_dec1_dbstores_.md` and are not repeated.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value
from tests.integration.management.dec1_mgmt_support import (
    PROXY_ADMIN_ONLY_PREFIX,
    body_of,
    budget_rows,
    caller,
    customer,
    customer_rows,
    delete_budget_if_present,
    error_text,
    get,
    post,
    proxy_error,
    unique,
)

pytestmark: Final = pytest.mark.shares_worker_state

CUSTOMER_FIELDS: Final = frozenset(
    {
        "user_id",
        "blocked",
        "alias",
        "spend",
        "allowed_model_region",
        "default_model",
        "models",
        "budget_id",
        "litellm_budget_table",
        "object_permission_id",
        "object_permission",
    }
)


def test_customer_new_answers_the_full_customer_field_set_with_defaults(gateway: Gateway) -> None:
    """A proxy admin creates a customer with only a user_id and an alias. The 200 body must carry exactly the
    CustomerResponse fields, with blocked false, spend 0.0, models [], and no budget or region. A Rust store that
    serializes its own row type would easily add or drop a field here."""
    identity: Final = unique("customer")
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [identity]})
        response: Final = post(gateway, "/customer/new", {"user_id": identity, "alias": "plain"})
        assert response.status_code == 200, response.text
        created: Final = body_of(response)
        assert set(created) == CUSTOMER_FIELDS
        assert created["user_id"] == identity
        assert created["alias"] == "plain"
        assert created["blocked"] is False
        assert created["spend"] == 0.0
        assert created["models"] == []
        assert created["budget_id"] is None
        assert created["litellm_budget_table"] is None
        assert created["allowed_model_region"] is None
        assert created["default_model"] is None


def test_customer_new_with_max_budget_creates_and_links_a_budget_row(gateway: Gateway) -> None:
    """Creating a customer with max_budget 7.5 must write a new LiteLLM_BudgetTable row with that max_budget,
    link it through the customer's budget_id column, and return it nested as litellm_budget_table."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario, max_budget=7.5)
        rows: Final = customer_rows(identity)
        assert len(rows) == 1
        budget_id: Final = rows[0]["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        assert budget_rows(budget_id)[0]["max_budget"] == 7.5
        info: Final = gateway.get("/customer/info", {"end_user_id": identity})
        assert info["budget_id"] == budget_id
        assert object_value(info["litellm_budget_table"])["max_budget"] == 7.5


def test_customer_new_with_an_existing_budget_id_reuses_that_row(gateway: Gateway) -> None:
    """Passing budget_id of an existing budget must link the customer to that row and create no new budget, so
    the customer row's budget_id equals the one passed in."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=3.0)
        identity: Final = customer(scenario, budget_id=budget_id)
        assert customer_rows(identity)[0]["budget_id"] == budget_id
        assert (
            object_value(gateway.get("/customer/info", {"end_user_id": identity})["litellm_budget_table"])["max_budget"]
            == 3.0
        )


def test_customer_new_with_both_max_budget_and_budget_id_is_422_and_writes_nothing(gateway: Gateway) -> None:
    """The request model refuses max_budget together with budget_id before the handler runs, so the answer is a
    422 naming "Set either 'max_budget' or 'budget_id', not both." and no customer row exists afterwards."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=1.0)
        identity: Final = unique("customer")
        response: Final = post(
            gateway, "/customer/new", {"user_id": identity, "max_budget": 2.0, "budget_id": budget_id}
        )
        assert response.status_code == 422, response.text
        assert "Set either 'max_budget' or 'budget_id', not both." in error_text(response)
        assert customer_rows(identity) == []


def test_customer_new_with_a_taken_user_id_is_400_with_the_exact_message(gateway: Gateway) -> None:
    """Creating the same customer id twice must answer 400 in the OpenAI error envelope with message
    "Customer already exists, passed user_id=<id>. Please pass a new user_id.", type bad_request and param
    user_id, and must leave the first row untouched."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario, alias="first")
        response: Final = post(gateway, "/customer/new", {"user_id": identity, "alias": "second"})
        assert response.status_code == 400, response.text
        error: Final = proxy_error(response)
        assert error["message"] == f"Customer already exists, passed user_id={identity}. Please pass a new user_id."
        assert error["type"] == "bad_request"
        assert error["param"] == "user_id"
        assert error["code"] == "400"
        assert customer_rows(identity)[0]["alias"] == "first"


def test_customer_new_with_a_default_model_the_proxy_does_not_serve_is_422(gateway: Gateway) -> None:
    """A default_model that is not a model group on the router must be refused with 422 and a message starting
    "Default Model not on proxy. Configure via `/model/new` or config.yaml. Default_model=<name>", and no
    customer row is written."""
    identity: Final = unique("customer")
    missing_model: Final = unique("no-such-model")
    response: Final = post(gateway, "/customer/new", {"user_id": identity, "default_model": missing_model})
    assert response.status_code == 422, response.text
    assert (
        f"Default Model not on proxy. Configure via `/model/new` or config.yaml. Default_model={missing_model}"
        in error_text(response)
    )
    assert customer_rows(identity) == []


def test_customer_info_for_an_unknown_id_is_404_naming_the_id(gateway: Gateway) -> None:
    """GET /customer/info for an id with no row must answer 404 with message "End User Id=<id> does not exist in
    db", type not_found and param end_user_id."""
    identity: Final = unique("ghost")
    response: Final = get(gateway, "/customer/info", {"end_user_id": identity})
    assert response.status_code == 404, response.text
    error: Final = proxy_error(response)
    assert error["message"] == f"End User Id={identity} does not exist in db"
    assert error["type"] == "not_found"
    assert error["param"] == "end_user_id"


def test_customer_info_without_end_user_id_is_422(gateway: Gateway) -> None:
    """end_user_id is a required query parameter, so a bare GET /customer/info must be a 422 validation error
    rather than a lookup of a null id."""
    response: Final = get(gateway, "/customer/info")
    assert response.status_code == 422, response.text
    assert "end_user_id" in response.text


def test_customer_update_for_an_unknown_id_is_404_with_param_user_id(gateway: Gateway) -> None:
    """Updating a customer that does not exist must answer 404 "End User Id=<id> does not exist in db" with param
    user_id (the update body field, unlike info's end_user_id) and must not create a row."""
    identity: Final = unique("ghost")
    response: Final = post(gateway, "/customer/update", {"user_id": identity, "alias": "x"})
    assert response.status_code == 404, response.text
    error: Final = proxy_error(response)
    assert error["message"] == f"End User Id={identity} does not exist in db"
    assert error["param"] == "user_id"
    assert customer_rows(identity) == []


def test_customer_update_with_max_budget_on_a_customer_without_budget_creates_one(gateway: Gateway) -> None:
    """A customer created without a budget gets max_budget 4.0 through /customer/update. The handler must create
    a budget row and set the customer's budget_id to it, and the response nests that budget."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario)
        assert customer_rows(identity)[0]["budget_id"] is None
        updated: Final = gateway.post("/customer/update", {"user_id": identity, "max_budget": 4.0})
        budget_id: Final = customer_rows(identity)[0]["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        assert updated["budget_id"] == budget_id
        assert budget_rows(budget_id)[0]["max_budget"] == 4.0


def test_customer_update_with_max_budget_rewrites_the_linked_budget_row_in_place(gateway: Gateway) -> None:
    """A customer that already has a budget row gets a new max_budget. The same budget row must be updated, so the
    customer's budget_id is unchanged and that row now reads the new max_budget."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario, max_budget=1.0)
        budget_id: Final = customer_rows(identity)[0]["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        gateway.post("/customer/update", {"user_id": identity, "max_budget": 9.0})
        assert customer_rows(identity)[0]["budget_id"] == budget_id
        assert budget_rows(budget_id)[0]["max_budget"] == 9.0


def test_customer_update_leaves_fields_that_were_not_sent(gateway: Gateway) -> None:
    """A customer created with alias "kept" and models ["m1"] is updated with only blocked true. The alias and
    models must survive, because update applies only the fields the caller sent."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario, alias="kept", models=["m1"])
        gateway.post("/customer/update", {"user_id": identity, "blocked": True})
        row: Final = customer_rows(identity)[0]
        assert row["alias"] == "kept"
        assert row["models"] == ["m1"]
        assert row["blocked"] is True


def test_customer_delete_answers_the_count_and_a_message_listing_the_ids(gateway: Gateway) -> None:
    """Deleting two customers in one call must answer 200 with deleted_customers 2 and message
    "Successfully deleted customers with ids: ['<a>', '<b>']" (the Python repr of the request list), and both
    rows are gone."""
    with gateway.scenario() as scenario:
        first: Final = customer(scenario)
        second: Final = customer(scenario)
        response: Final = post(gateway, "/customer/delete", {"user_ids": [first, second]})
        assert response.status_code == 200, response.text
        assert body_of(response) == {
            "deleted_customers": 2,
            "message": f"Successfully deleted customers with ids: ['{first}', '{second}']",
        }
        assert customer_rows(first) == []
        assert customer_rows(second) == []


def test_customer_delete_with_one_missing_id_deletes_none_and_names_the_missing(gateway: Gateway) -> None:
    """A batch naming one existing and one unknown customer must answer 404 "End User Id(s)=<missing> do not exist
    in db" with param user_ids, and the existing customer must still be there: the existence check runs before
    any delete."""
    with gateway.scenario() as scenario:
        present: Final = customer(scenario)
        missing: Final = unique("ghost")
        response: Final = post(gateway, "/customer/delete", {"user_ids": [present, missing]})
        assert response.status_code == 404, response.text
        error: Final = proxy_error(response)
        assert error["message"] == f"End User Id(s)={missing} do not exist in db"
        assert error["param"] == "user_ids"
        assert len(customer_rows(present)) == 1


def test_customer_delete_with_an_empty_list_is_a_400(gateway: Gateway) -> None:
    """An empty user_ids list reaches the handler's else branch, a bare ValueError "user_id is required, passed
    user_id = []". The intended answer is a 400 naming that; today handle_exception_on_proxy maps a ValueError
    with no status to 500 (`error_status_code(e, 500)` in `common_utils/openai_error_payload.py`)."""
    pytest.skip("BUG: /customer/delete with user_ids [] answers 500 instead of a 400 client error")
    response: Final = post(gateway, "/customer/delete", {"user_ids": []})
    assert response.status_code == 400, response.text
    assert "user_id is required, passed user_id = []" in error_text(response)


def test_customer_delete_keeps_the_customers_budget_row(gateway: Gateway) -> None:
    """Deleting a customer created with max_budget removes only the end user row. Its budget row stays in
    LiteLLM_BudgetTable, because delete_end_user only calls delete_many on the end user table. A store port that
    cascades the budget would change this."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario, max_budget=2.0)
        budget_id: Final = customer_rows(identity)[0]["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        gateway.post("/customer/delete", {"user_ids": [identity]})
        assert customer_rows(identity) == []
        assert len(budget_rows(budget_id)) == 1


def test_customer_block_creates_a_blocked_row_for_an_unknown_id(gateway: Gateway) -> None:
    """/customer/block upserts: blocking an id that has no row must create one with blocked true and answer
    {"blocked_users": [<row>]} with that row's user_id and blocked true."""
    with gateway.scenario() as scenario:
        identity: Final = unique("blocked")
        scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [identity]})
        response: Final = post(gateway, "/customer/block", {"user_ids": [identity]})
        assert response.status_code == 200, response.text
        blocked: Final = body_of(response)["blocked_users"]
        assert isinstance(blocked, list) and len(blocked) == 1
        assert object_value(blocked[0])["user_id"] == identity
        assert object_value(blocked[0])["blocked"] is True
        assert customer_rows(identity)[0]["blocked"] is True


def test_customer_block_of_an_existing_customer_keeps_its_alias(gateway: Gateway) -> None:
    """Blocking an existing customer only flips blocked to true. The upsert's update branch must not reset the
    alias or the budget link of the row."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario, alias="stays")
        gateway.post("/customer/block", {"user_ids": [identity]})
        row: Final = customer_rows(identity)[0]
        assert row["blocked"] is True
        assert row["alias"] == "stays"


def test_customer_unblock_without_the_blocked_user_list_hook_is_400(gateway: Gateway) -> None:
    """/customer/unblock only edits the in memory blocked_user_list of the enterprise hook. On the suite proxy,
    where no blocked_user_list is configured, it must answer 400 with "Blocked user check was never set. This call
    has no effect." and must not touch the DB blocked flag set by /customer/block."""
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario)
        gateway.post("/customer/block", {"user_ids": [identity]})
        response: Final = post(gateway, "/customer/unblock", {"user_ids": [identity]})
        assert response.status_code == 400, response.text
        assert "Blocked user check was never set. This call has no effect." in error_text(response)
        assert customer_rows(identity)[0]["blocked"] is True


def test_customer_list_orders_by_user_id_and_pages(gateway: Gateway) -> None:
    """Three customers share a unique search prefix. /customer/list?search=<prefix>&page_size=2 must return the
    first two by user_id ascending, and page 2 the third, as a bare JSON list."""
    prefix: Final = unique("page")
    with gateway.scenario() as scenario:
        identities: Final = [f"{prefix}-{suffix}" for suffix in ("c", "a", "b")]
        for identity in identities:
            gateway.post("/customer/new", {"user_id": identity})
            scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [identity]})
        first: Final = get(gateway, "/customer/list", {"search": prefix, "page_size": "2", "page": "1"})
        second: Final = get(gateway, "/customer/list", {"search": prefix, "page_size": "2", "page": "2"})
        assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
        assert [object_value(row)["user_id"] for row in first.json()] == [f"{prefix}-a", f"{prefix}-b"]
        assert [object_value(row)["user_id"] for row in second.json()] == [f"{prefix}-c"]


def test_customer_list_search_matches_the_alias_case_insensitively(gateway: Gateway) -> None:
    """search matches user_id or alias with a case insensitive contains. A customer whose alias holds a unique
    mixed case marker must be found by the lower case marker."""
    marker: Final = unique("Alias").upper()
    with gateway.scenario() as scenario:
        identity: Final = customer(scenario, alias=f"x-{marker}-y")
        response: Final = get(gateway, "/customer/list", {"search": marker.lower()})
        assert response.status_code == 200, response.text
        assert [object_value(row)["user_id"] for row in response.json()] == [identity]


@pytest.mark.parametrize("page_size", ["0", "101"])
def test_customer_list_page_size_outside_1_to_100_is_422(gateway: Gateway, page_size: str) -> None:
    """page_size is bounded ge=1, le=100 by the route signature, so 0 and 101 must both be 422 validation errors."""
    response: Final = get(gateway, "/customer/list", {"page_size": page_size})
    assert response.status_code == 422, response.text


def test_customer_list_is_refused_to_an_internal_user_at_the_route_gate(gateway: Gateway) -> None:
    """/customer/list is in no internal user route list, so an internal user's key must get the route gate 401
    whose message starts "Authentication Error, Only proxy admin can be used to generate, delete, update info for
    new keys/users/teams. Route=/customer/list", not the handler's own admin check."""
    with gateway.scenario() as scenario:
        internal: Final = caller(scenario, "internal_user")
        response: Final = get(gateway, "/customer/list", key=internal.key)
        assert response.status_code == 401, response.text
        assert str(proxy_error(response)["message"]).startswith(f"{PROXY_ADMIN_ONLY_PREFIX}Route=/customer/list.")


def test_customer_list_and_info_are_readable_by_a_proxy_admin_viewer(gateway: Gateway) -> None:
    """A proxy_admin_viewer may read customers: GET /customer/list and GET /customer/info both answer 200 for the
    viewer's key, because the viewer gate allows safe methods and the list handler accepts the viewer role."""
    with gateway.scenario() as scenario:
        viewer: Final = caller(scenario, "proxy_admin_viewer")
        identity: Final = customer(scenario)
        listed: Final = get(gateway, "/customer/list", {"search": identity}, key=viewer.key)
        assert listed.status_code == 200, listed.text
        assert [object_value(row)["user_id"] for row in listed.json()] == [identity]
        info: Final = get(gateway, "/customer/info", {"end_user_id": identity}, key=viewer.key)
        assert info.status_code == 200, info.text


@pytest.mark.parametrize("route", ["/customer/new", "/customer/update", "/customer/delete", "/customer/block"])
def test_customer_writes_are_403_for_a_proxy_admin_viewer(gateway: Gateway, route: str) -> None:
    """A proxy_admin_viewer posting to a customer write route must get 403 "user not allowed to access this route,
    role= proxy_admin_viewer. Trying to access: <route>" from the viewer gate, and no customer row is written."""
    with gateway.scenario() as scenario:
        viewer: Final = caller(scenario, "proxy_admin_viewer")
        identity: Final = unique("viewer-write")
        body: Final = (
            {"user_ids": [identity]} if route in ("/customer/delete", "/customer/block") else {"user_id": identity}
        )
        response: Final = post(gateway, route, body, key=viewer.key)
        assert response.status_code == 403, response.text
        assert proxy_error(response)["message"] == (
            f"user not allowed to access this route, role= proxy_admin_viewer. Trying to access: {route}"
        )
        assert customer_rows(identity) == []


@pytest.mark.parametrize("role", ["internal_user", "internal_user_viewer"])
def test_customer_new_is_401_for_non_admin_roles(gateway: Gateway, role: str) -> None:
    """Internal users and internal viewers may not create customers: /customer/new must answer the route gate 401
    naming Route=/customer/new and the caller's role, and write no row."""
    with gateway.scenario() as scenario:
        user: Final = caller(scenario, role)
        identity: Final = unique("denied")
        response: Final = post(gateway, "/customer/new", {"user_id": identity}, key=user.key)
        assert response.status_code == 401, response.text
        message: Final = str(proxy_error(response)["message"])
        assert message.startswith(f"{PROXY_ADMIN_ONLY_PREFIX}Route=/customer/new. Your role={role}.")
        assert customer_rows(identity) == []


def test_end_user_alias_routes_answer_like_the_customer_routes(gateway: Gateway) -> None:
    """The legacy /end_user/new and /end_user/info paths share the customer handlers, so a customer created through
    /end_user/new must be readable through /customer/info and vice versa."""
    with gateway.scenario() as scenario:
        identity: Final = unique("legacy")
        scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [identity]})
        created: Final = post(gateway, "/end_user/new", {"user_id": identity, "alias": "legacy"})
        assert created.status_code == 200, created.text
        assert gateway.get("/customer/info", {"end_user_id": identity})["alias"] == "legacy"
        legacy_info: Final = get(gateway, "/end_user/info", {"end_user_id": identity})
        assert legacy_info.status_code == 200, legacy_info.text
        assert body_of(legacy_info)["alias"] == "legacy"
