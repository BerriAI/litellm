"""Organization management API contracts: /organization/new, update, member_add, info, list, delete.

The organization object moves behind a Rust store this week (draft PR #52173, thread B of the DB W3 brief). These
tests pin, from `litellm/proxy/management_endpoints/organization_endpoints.py` and the route gate in
`litellm/proxy/auth/route_checks.py`: the budget row every organization gets on create, the exact refusals for
negative budgets and missing ids, who may create, read, update and delete an organization, what an organization
delete removes (teams, memberships, keys) and what it leaves (its budget row), and how /organization/list scopes
results for members. Org key revocation on delete and member role updates are already covered by
`test_organization_lifecycle.py`.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, list_value, object_value, string_value
from tests.integration.management.dec1_mgmt_support import (
    PROXY_ADMIN_ONLY_PREFIX,
    body_of,
    budget_rows,
    caller,
    delete_budget_if_present,
    detail_error,
    error_text,
    get,
    org_admin,
    org_membership_rows,
    organization_rows,
    post,
    proxy_error,
    team_rows,
    unique,
    user_rows,
)

pytestmark: Final = pytest.mark.shares_worker_state


def test_organization_new_creates_a_budget_row_from_the_budget_fields(gateway: Gateway) -> None:
    """An organization created with max_budget 12.0 and no budget_id must get a fresh LiteLLM_BudgetTable row with
    that max_budget, linked through budget_id, and the response nests it as litellm_budget_table."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization(max_budget=12.0)
        budget_id: Final = organization_rows(organization_id)[0]["budget_id"]
        assert isinstance(budget_id, str)
        assert budget_rows(budget_id)[0]["max_budget"] == 12.0
        info: Final = gateway.get("/organization/info", {"organization_id": organization_id})
        assert object_value(info["litellm_budget_table"])["max_budget"] == 12.0


def test_organization_new_without_budget_fields_still_creates_an_empty_budget_row(gateway: Gateway) -> None:
    """Every organization needs a budget attached: even with no budget fields the handler creates a budget row with
    a null max_budget and links it, so budget_id is never null on the organization row."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        budget_id: Final = organization_rows(organization_id)[0]["budget_id"]
        assert isinstance(budget_id, str)
        assert budget_rows(budget_id)[0]["max_budget"] is None


def test_organization_new_with_an_existing_budget_id_reuses_it(gateway: Gateway) -> None:
    """Passing budget_id of an existing budget links the organization to it and creates no other budget, so the
    organization row's budget_id equals the one passed in."""
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=2.0)
        alias: Final = unique("org")
        created: Final = gateway.post("/organization/new", {"organization_alias": alias, "budget_id": budget_id})
        organization_id: Final = string_value(created["organization_id"])
        scenario.cleanups.callback(
            gateway.request, "DELETE", "/organization/delete", {"organization_ids": [organization_id]}
        )
        assert organization_rows(organization_id)[0]["budget_id"] == budget_id


@pytest.mark.parametrize("field", ["max_budget", "soft_budget"])
def test_organization_new_with_a_negative_amount_is_400(gateway: Gateway, field: str) -> None:
    """A negative max_budget or soft_budget must be refused with 400 and detail.error
    "<field> must be a non-negative finite number. Received: -5.0", before any budget or organization row."""
    alias: Final = unique("neg-org")
    response: Final = post(gateway, "/organization/new", {"organization_alias": alias, field: -5.0})
    assert response.status_code == 400, response.text
    assert detail_error(response) == f"{field} must be a non-negative finite number. Received: -5.0"
    assert (
        gateway.rows('SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_alias = %s', (alias,))
        == []
    )


@pytest.mark.parametrize("role", ["internal_user", "proxy_admin_viewer"])
def test_organization_new_is_refused_to_non_admin_roles(gateway: Gateway, role: str) -> None:
    """Only a proxy admin may create organizations. An internal user gets the route gate 401 naming
    Route=/organization/new, and a proxy_admin_viewer gets the viewer gate 403; neither writes a row."""
    with gateway.scenario() as scenario:
        user: Final = caller(scenario, role)
        alias: Final = unique("denied-org")
        response: Final = post(gateway, "/organization/new", {"organization_alias": alias}, key=user.key)
        if role == "internal_user":
            assert response.status_code == 401, response.text
            assert str(proxy_error(response)["message"]).startswith(
                f"{PROXY_ADMIN_ONLY_PREFIX}Route=/organization/new."
            )
        else:
            assert response.status_code == 403, response.text
            assert proxy_error(response)["message"] == (
                "user not allowed to access this route, role= proxy_admin_viewer. Trying to access: /organization/new"
            )
        assert (
            gateway.rows(
                'SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_alias = %s', (alias,)
            )
            == []
        )


def test_organization_info_for_an_unknown_id_is_404(gateway: Gateway) -> None:
    """GET /organization/info for an id with no row, asked by the proxy admin, must answer 404 with
    detail.error "Organization not found"."""
    response: Final = get(gateway, "/organization/info", {"organization_id": unique("ghost-org")})
    assert response.status_code == 404, response.text
    assert detail_error(response) == "Organization not found"


def test_organization_info_lists_members_with_their_user_and_teams(gateway: Gateway) -> None:
    """An organization with one org_admin member and one team must answer info with a members list holding that
    user_id and role org_admin, and a teams list holding that team_id."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        admin: Final = org_admin(scenario, organization_id)
        team_id: Final = scenario.team(organization_id=organization_id)
        info: Final = gateway.get("/organization/info", {"organization_id": organization_id})
        members: Final = [object_value(member) for member in list_value(info["members"])]
        assert {(member["user_id"], member["user_role"]) for member in members} == {(admin.user_id, "org_admin")}
        teams: Final = [object_value(team)["team_id"] for team in list_value(info["teams"])]
        assert teams == [team_id]


def test_organization_info_is_readable_by_its_own_org_admin(gateway: Gateway) -> None:
    """An org admin reading their own organization passes verify_org_access and gets 200 with the organization_id.
    UNVERIFIED: the route gate admits /organization/info for an org admin only when request_data carries the
    organization_id; for a GET this assumes the query parameters are part of request_data."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        admin: Final = org_admin(scenario, organization_id)
        response: Final = get(gateway, "/organization/info", {"organization_id": organization_id}, key=admin.key)
        assert response.status_code == 200, response.text
        assert body_of(response)["organization_id"] == organization_id


def test_organization_info_is_refused_to_an_org_admin_of_another_organization(gateway: Gateway) -> None:
    """An org admin of organization A asks for organization B. Whether the route gate (401) or verify_org_access
    (403 "You do not have access to this organization") answers, the caller must be refused and get no body of B."""
    with gateway.scenario() as scenario:
        own: Final = scenario.organization()
        other: Final = scenario.organization()
        admin: Final = org_admin(scenario, own)
        response: Final = get(gateway, "/organization/info", {"organization_id": other}, key=admin.key)
        assert response.status_code in (401, 403), response.text
        assert other not in {str(value) for value in body_of(response).values()}


def test_organization_update_without_organization_id_is_400(gateway: Gateway) -> None:
    """PATCH /organization/update with no organization_id must be refused with 400 and detail.error
    "organization_id is required"."""
    response: Final = gateway.request("PATCH", "/organization/update", {"organization_alias": "nameless"})
    assert response.status_code == 400, response.text
    assert detail_error(response) == "organization_id is required"


def test_organization_update_of_an_unknown_organization_is_404(gateway: Gateway) -> None:
    """Updating an organization id with no row should answer 404 naming the id, as the v2 PATCH route does with
    "Organization not found for organization_id=<id>". Today the legacy handler raises a bare ValueError with that
    text, which reaches the generic exception handler as a 500."""
    pytest.skip("BUG: PATCH /organization/update for an unknown organization_id answers 500 instead of 404")
    organization_id: Final = unique("ghost-org")
    response: Final = gateway.request(
        "PATCH", "/organization/update", {"organization_id": organization_id, "organization_alias": "x"}
    )
    assert response.status_code == 404, response.text
    assert f"Organization not found for organization_id={organization_id}" in error_text(response)


def test_organization_update_with_a_negative_max_budget_is_400_and_keeps_the_budget(gateway: Gateway) -> None:
    """Updating an organization's max_budget to -1.0 must be refused with 400 and detail.error
    "max_budget must be a non-negative finite number. Received: -1.0", leaving the budget row at 3.0."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization(max_budget=3.0)
        response: Final = gateway.request(
            "PATCH", "/organization/update", {"organization_id": organization_id, "max_budget": -1.0}
        )
        assert response.status_code == 400, response.text
        assert detail_error(response) == "max_budget must be a non-negative finite number. Received: -1.0"
        budget_id: Final = string_value(organization_rows(organization_id)[0]["budget_id"])
        assert budget_rows(budget_id)[0]["max_budget"] == 3.0


def test_organization_update_by_an_internal_member_is_refused_and_changes_nothing(gateway: Gateway) -> None:
    """A plain internal_user member of the organization tries to rename it. /organization/update is not in any
    non admin route list, so the route gate must refuse with 401 and the alias stays the same."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        before: Final = organization_rows(organization_id)[0]["organization_alias"]
        member_id: Final = scenario.org_member(organization_id, "internal_user")
        member_key: Final = scenario.key(user_id=member_id)
        response: Final = gateway.request(
            "PATCH",
            "/organization/update",
            {"organization_id": organization_id, "organization_alias": "renamed"},
            key=member_key,
        )
        assert response.status_code == 401, response.text
        assert organization_rows(organization_id)[0]["organization_alias"] == before


def test_organization_member_add_to_an_unknown_organization_is_404(gateway: Gateway) -> None:
    """Adding a member to an organization id with no row must answer 404 with a message containing
    "Organization not found for organization_id=<id>", and create no membership row."""
    organization_id: Final = unique("ghost-org")
    response: Final = post(
        gateway,
        "/organization/member_add",
        {"organization_id": organization_id, "member": {"role": "internal_user", "user_id": unique("u")}},
    )
    assert response.status_code == 404, response.text
    assert f"Organization not found for organization_id={organization_id}" in error_text(response)
    assert org_membership_rows(organization_id) == []


def test_organization_member_add_with_an_unknown_user_id_provisions_the_user(gateway: Gateway) -> None:
    """member_add with a user_id that has no user row creates the user and the membership: afterwards
    LiteLLM_UserTable has the user and LiteLLM_OrganizationMembership has it with role internal_user, and the
    response lists it under updated_organization_memberships. UNVERIFIED: the provisioned user's own user_role is
    not pinned; see `add_member_to_organization`."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        user_id: Final = unique("provisioned")
        response: Final = post(
            gateway,
            "/organization/member_add",
            {"organization_id": organization_id, "member": {"role": "internal_user", "user_id": user_id}},
        )
        scenario.cleanups.callback(scenario.delete_user, user_id)
        assert response.status_code == 200, response.text
        added: Final = body_of(response)
        assert added["organization_id"] == organization_id
        memberships: Final = [object_value(row) for row in list_value(added["updated_organization_memberships"])]
        assert [(row["user_id"], row["user_role"]) for row in memberships] == [(user_id, "internal_user")]
        assert len(user_rows(user_id)) == 1
        assert org_membership_rows(organization_id) == [{"user_id": user_id, "user_role": "internal_user"}]


def test_organization_member_add_by_an_internal_member_is_refused(gateway: Gateway) -> None:
    """A plain internal_user member of an organization may not add members to it: the call must be refused (route
    gate 401 or verify_org_access 403) and the membership table keeps only the caller."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        member_id: Final = scenario.org_member(organization_id, "internal_user")
        member_key: Final = scenario.key(user_id=member_id)
        outsider: Final = scenario.user(user_role="internal_user")
        response: Final = post(
            gateway,
            "/organization/member_add",
            {"organization_id": organization_id, "member": {"role": "org_admin", "user_id": outsider}},
            key=member_key,
        )
        assert response.status_code in (401, 403), response.text
        assert org_membership_rows(organization_id) == [{"user_id": member_id, "user_role": "internal_user"}]


def test_organization_list_for_a_non_member_internal_user_is_empty(gateway: Gateway) -> None:
    """/organization/list is an info route open to internal users, but a user with no organization memberships
    must get 200 and an empty list, not every organization."""
    with gateway.scenario() as scenario:
        scenario.organization()
        loner: Final = caller(scenario, "internal_user")
        response: Final = get(gateway, "/organization/list", key=loner.key)
        assert response.status_code == 200, response.text
        assert response.json() == []


def test_organization_list_for_a_member_returns_only_their_organizations(gateway: Gateway) -> None:
    """An internal user who is a member of organization A, but not B, lists organizations and gets exactly A; asking
    with org_id=B returns an empty list rather than a 403."""
    with gateway.scenario() as scenario:
        mine: Final = scenario.organization()
        other: Final = scenario.organization()
        member_id: Final = scenario.org_member(mine, "internal_user")
        member_key: Final = scenario.key(user_id=member_id)
        listed: Final = get(gateway, "/organization/list", key=member_key)
        assert listed.status_code == 200, listed.text
        assert [object_value(row)["organization_id"] for row in listed.json()] == [mine]
        filtered: Final = get(gateway, "/organization/list", {"org_id": other}, key=member_key)
        assert filtered.status_code == 200, filtered.text
        assert filtered.json() == []


def test_organization_list_org_alias_filter_is_a_case_insensitive_contains(gateway: Gateway) -> None:
    """The proxy admin filters /organization/list with org_alias set to a lower case middle fragment of an upper
    case alias; exactly that organization must come back."""
    marker: Final = unique("frag")
    with gateway.scenario() as scenario:
        created: Final = gateway.post("/organization/new", {"organization_alias": f"PRE-{marker.upper()}-POST"})
        organization_id: Final = string_value(created["organization_id"])
        scenario.cleanups.callback(delete_budget_if_present, gateway, string_value(created["budget_id"]))
        scenario.cleanups.callback(
            gateway.request, "DELETE", "/organization/delete", {"organization_ids": [organization_id]}
        )
        response: Final = get(gateway, "/organization/list", {"org_alias": marker.lower()})
        assert response.status_code == 200, response.text
        assert [object_value(row)["organization_id"] for row in response.json()] == [organization_id]


def test_organization_delete_of_an_unknown_id_is_404_naming_it(gateway: Gateway) -> None:
    """DELETE /organization/delete for an id with no row must answer 404 with detail.error
    "Organization=<id> not found"."""
    organization_id: Final = unique("ghost-org")
    response: Final = gateway.request("DELETE", "/organization/delete", {"organization_ids": [organization_id]})
    assert response.status_code == 404, response.text
    assert detail_error(response) == f"Organization={organization_id} not found"


def test_organization_delete_by_its_own_org_admin_is_refused(gateway: Gateway) -> None:
    """An org admin may not delete their own organization: the delete body names organization_ids, which the
    org admin route check does not read, so the route gate answers 401, and the organization row stays."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        admin: Final = org_admin(scenario, organization_id)
        response: Final = gateway.request(
            "DELETE", "/organization/delete", {"organization_ids": [organization_id]}, key=admin.key
        )
        assert response.status_code == 401, response.text
        assert len(organization_rows(organization_id)) == 1


def test_organization_delete_removes_its_teams_and_memberships_and_keeps_its_budget(gateway: Gateway) -> None:
    """Deleting an organization with one team and one member must remove the organization row, the team row and
    the membership row, answer 200 with a one element list naming the organization, and leave the organization's
    budget row in LiteLLM_BudgetTable (the handler never deletes it)."""
    with gateway.scenario() as scenario:
        alias: Final = unique("cascade")
        created: Final = gateway.post("/organization/new", {"organization_alias": alias, "max_budget": 1.0})
        organization_id: Final = string_value(created["organization_id"])
        budget_id: Final = string_value(created["budget_id"])
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        member_id: Final = scenario.org_member(organization_id, "internal_user")
        team: Final = gateway.post("/team/new", {"team_alias": unique("org-team"), "organization_id": organization_id})
        team_id: Final = string_value(team["team_id"])
        response: Final = gateway.request("DELETE", "/organization/delete", {"organization_ids": [organization_id]})
        assert response.status_code == 200, response.text
        deleted: Final = response.json()
        assert isinstance(deleted, list)
        assert [object_value(row)["organization_id"] for row in deleted] == [organization_id]
        assert organization_rows(organization_id) == []
        assert team_rows(team_id) == []
        assert org_membership_rows(organization_id) == []
        assert len(user_rows(member_id)) == 1
        assert len(budget_rows(budget_id)) == 1
