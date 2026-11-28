"""Internal user management API contracts: /user/new, /user/update, /user/info, /user/list, /user/delete.

The user object moves behind a Rust store this week (draft PR #52176, thread C of the DB W3 brief). These tests pin
what `litellm/proxy/management_endpoints/internal_user_endpoints.py` answers: duplicate id and email refusals
(409, with a case insensitive email match), the key /user/new creates unless auto_create_key is false, who may
create an administrative user, the /user/list envelope and its role gate, and the per target scope of
/user/delete. User delete revoking keys and memberships is already covered by `test_user_delete_revokes_keys.py`.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, list_value, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration.management.dec1_mgmt_support import (
    body_of,
    caller,
    digest,
    error_text,
    get,
    key_rows,
    org_admin,
    post,
    unique,
    user_rows,
)

pytestmark: Final = pytest.mark.shares_worker_state


def _keys_of(user_id: str) -> list[dict[str, object]]:
    return list(read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id = %s', (user_id,)))


def test_user_new_creates_a_personal_key_by_default(gateway: Gateway) -> None:
    """/user/new without auto_create_key returns a key that starts with sk- and is stored with the new user_id, and
    the response carries user_id, user_role and the key; the key is deleted with the user."""
    with gateway.scenario() as scenario:
        user_id: Final = unique("user")
        response: Final = post(gateway, "/user/new", {"user_id": user_id, "user_role": "internal_user"})
        scenario.cleanups.callback(scenario.delete_user, user_id)
        assert response.status_code == 200, response.text
        created: Final = body_of(response)
        key: Final = string_value(created["key"])
        assert key.startswith("sk-")
        assert created["user_id"] == user_id
        assert created["user_role"] == "internal_user"
        assert key_rows(key)[0]["user_id"] == user_id


def test_user_new_with_auto_create_key_false_writes_no_key(gateway: Gateway) -> None:
    """With auto_create_key false only the user row is written: LiteLLM_VerificationToken has no row for the
    user_id. UNVERIFIED: the response key field is then an empty string or null; only the DB state is pinned."""
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        assert len(user_rows(user_id)) == 1
        assert _keys_of(user_id) == []


def test_user_new_without_user_id_generates_a_uuid(gateway: Gateway) -> None:
    """A /user/new body with no user_id gets a generated uuid4 string as user_id, and the row exists under it."""
    with gateway.scenario() as scenario:
        response: Final = post(gateway, "/user/new", {"auto_create_key": False, "user_role": "internal_user"})
        assert response.status_code == 200, response.text
        user_id: Final = string_value(body_of(response)["user_id"])
        scenario.cleanups.callback(scenario.delete_user, user_id)
        assert len(user_id) == 36 and user_id.count("-") == 4
        assert len(user_rows(user_id)) == 1


def test_user_new_with_a_taken_user_id_is_409(gateway: Gateway) -> None:
    """Creating a user whose id already exists must answer 409 with "User with id <id> already exists" and leave the
    original row's role unchanged."""
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        response: Final = post(
            gateway, "/user/new", {"user_id": user_id, "auto_create_key": False, "user_role": "internal_user_viewer"}
        )
        assert response.status_code == 409, response.text
        assert f"User with id {user_id} already exists" in error_text(response)
        assert user_rows(user_id)[0]["user_role"] == "internal_user"


def test_user_new_with_an_email_taken_in_another_case_is_409_naming_the_stored_email(gateway: Gateway) -> None:
    """Email duplicates are matched case insensitively: a user stored with "<x>@example.com" blocks a new user with
    the upper case spelling, and the 409 message names the stored spelling, "User with email <stored> already
    exists"."""
    with gateway.scenario() as scenario:
        email: Final = f"{unique('mail')}@example.com"
        scenario.user(user_role="internal_user", user_email=email)
        other_id: Final = unique("user")
        response: Final = post(
            gateway, "/user/new", {"user_id": other_id, "user_email": email.upper(), "auto_create_key": False}
        )
        assert response.status_code == 409, response.text
        assert f"User with email {email} already exists" in error_text(response)
        assert user_rows(other_id) == []


def test_an_org_admin_cannot_create_a_proxy_admin_user(gateway: Gateway) -> None:
    """An org admin may create users in their organization, but not administrative ones: asking for user_role
    proxy_admin must answer 403 with "Only proxy admins can create administrative users (proxy_admin,
    proxy_admin_viewer). Attempted to create user with role: proxy_admin" and write no row. UNVERIFIED: the
    caller's role in the message is the org admin's key role, internal_user."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        admin: Final = org_admin(scenario, organization_id)
        user_id: Final = unique("escalate")
        response: Final = post(
            gateway,
            "/user/new",
            {
                "user_id": user_id,
                "user_role": "proxy_admin",
                "organizations": [organization_id],
                "auto_create_key": False,
            },
            key=admin.key,
        )
        assert response.status_code == 403, response.text
        assert (
            "Only proxy admins can create administrative users (proxy_admin, proxy_admin_viewer). "
            "Attempted to create user with role: proxy_admin"
        ) in error_text(response)
        assert user_rows(user_id) == []


def test_user_new_is_401_for_an_internal_user(gateway: Gateway) -> None:
    """/user/new is a management route outside the internal user list, so an internal user's key gets the route
    gate 401 and no user is created."""
    with gateway.scenario() as scenario:
        internal: Final = caller(scenario, "internal_user")
        user_id: Final = unique("denied")
        response: Final = post(gateway, "/user/new", {"user_id": user_id, "auto_create_key": False}, key=internal.key)
        assert response.status_code == 401, response.text
        assert user_rows(user_id) == []


def test_user_info_for_an_unknown_user_is_404_naming_it(gateway: Gateway) -> None:
    """The proxy admin asking /user/info for a user_id without a row gets 404 and "User <id> not found"."""
    user_id: Final = unique("ghost")
    response: Final = get(gateway, "/user/info", {"user_id": user_id})
    assert response.status_code == 404, response.text
    assert f"User {user_id} not found" in error_text(response)


def test_user_info_for_another_user_is_403_naming_both_ids(gateway: Gateway) -> None:
    """An internal user asking /user/info for someone else is refused at the route gate with 403 and the message
    "key not allowed to access this user's info. user_id=<other>, key's user_id=<self>"."""
    with gateway.scenario() as scenario:
        me: Final = caller(scenario, "internal_user")
        other: Final = scenario.user(user_role="internal_user")
        response: Final = get(gateway, "/user/info", {"user_id": other}, key=me.key)
        assert response.status_code == 403, response.text
        assert f"key not allowed to access this user's info. user_id={other}, key's user_id={me.user_id}" in error_text(
            response
        )


def test_user_info_for_self_lists_the_users_keys_and_teams(gateway: Gateway) -> None:
    """An internal user reading their own /user/info gets user_id, a user_info object, a keys list holding their
    key, and a teams list holding the team they belong to."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        user_id: Final = scenario.member(team_id)
        key: Final = scenario.key(user_id=user_id)
        response: Final = get(gateway, "/user/info", {"user_id": user_id}, key=key)
        assert response.status_code == 200, response.text
        info: Final = body_of(response)
        assert info["user_id"] == user_id
        assert object_value(info["user_info"])["user_id"] == user_id
        assert digest(key) in {object_value(entry).get("token") for entry in list_value(info["keys"])}
        assert team_id in {object_value(entry)["team_id"] for entry in list_value(info["teams"])}


def test_user_list_envelope_and_key_count(gateway: Gateway) -> None:
    """The proxy admin lists /user/list?user_ids=<id> for a user with two keys and gets the envelope users, total,
    page, page_size and total_pages, with total 1 and the user's key_count 2."""
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        scenario.key(user_id=user_id)
        scenario.key(user_id=user_id)
        listed: Final = gateway.get("/user/list", {"user_ids": user_id})
        assert set(listed) >= {"users", "total", "page", "page_size", "total_pages"}
        assert (listed["total"], listed["page"], listed["page_size"], listed["total_pages"]) == (1, 1, 25, 1)
        users: Final = [object_value(user) for user in list_value(listed["users"])]
        assert [(user["user_id"], user["key_count"]) for user in users] == [(user_id, 2)]


@pytest.mark.parametrize("page_size", ["0", "101"])
def test_user_list_page_size_outside_1_to_100_is_422(gateway: Gateway, page_size: str) -> None:
    """page_size is bounded ge=1, le=100 in the route signature, so 0 and 101 are 422 validation errors."""
    response: Final = get(gateway, "/user/list", {"page_size": page_size})
    assert response.status_code == 422, response.text


def test_user_list_is_403_for_an_internal_user(gateway: Gateway) -> None:
    """/user/list is a self managed route, so an internal user reaches the handler, which answers 403 with
    "Only proxy admins and organization admins can list users."."""
    with gateway.scenario() as scenario:
        internal: Final = caller(scenario, "internal_user")
        response: Final = get(gateway, "/user/list", key=internal.key)
        assert response.status_code == 403, response.text
        assert "Only proxy admins and organization admins can list users." in error_text(response)


def test_user_list_for_an_org_admin_is_scoped_to_their_organization(gateway: Gateway) -> None:
    """An org admin listing users sees members of their organization and not a user outside it; asking for another
    organization's ids answers 403 "You do not have org_admin access to the requested organization(s)."."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        other_org: Final = scenario.organization()
        admin: Final = org_admin(scenario, organization_id)
        member: Final = scenario.org_member(organization_id, "internal_user")
        outsider: Final = scenario.user(user_role="internal_user")
        listed: Final = get(gateway, "/user/list", {"user_ids": f"{member},{outsider}"}, key=admin.key)
        assert listed.status_code == 200, listed.text
        assert [object_value(user)["user_id"] for user in list_value(body_of(listed)["users"])] == [member]
        foreign: Final = get(gateway, "/user/list", {"organization_ids": other_org}, key=admin.key)
        assert foreign.status_code == 403, foreign.text
        assert "You do not have org_admin access to the requested organization(s)." in error_text(foreign)


def test_user_delete_of_an_unknown_user_is_404_and_deletes_nothing_else(gateway: Gateway) -> None:
    """/user/delete validates every id before deleting: a batch with one real and one unknown user answers 404 with
    "User not found, passed user_id=<unknown>" and the real user still exists."""
    with gateway.scenario() as scenario:
        real: Final = scenario.user(user_role="internal_user")
        ghost: Final = unique("ghost")
        response: Final = post(gateway, "/user/delete", {"user_ids": [real, ghost]})
        assert response.status_code == 404, response.text
        assert f"User not found, passed user_id={ghost}" in error_text(response)
        assert len(user_rows(real)) == 1


def test_user_delete_by_an_org_admin_of_a_user_outside_their_orgs_is_403(gateway: Gateway) -> None:
    """An org admin naming their own organization_id in the body passes the route gate, but a target with no
    membership in that organization must be refused with 403 "User <id> is not within your admin scope." and kept."""
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        admin: Final = org_admin(scenario, organization_id)
        outsider: Final = scenario.user(user_role="internal_user")
        response: Final = post(
            gateway, "/user/delete", {"user_ids": [outsider], "organization_id": organization_id}, key=admin.key
        )
        assert response.status_code == 403, response.text
        assert f"User {outsider} is not within your admin scope." in error_text(response)
        assert len(user_rows(outsider)) == 1


def test_user_update_by_a_proxy_admin_viewer_may_only_change_user_email(gateway: Gateway) -> None:
    """A proxy_admin_viewer may post /user/update only with user_email: adding user_role must answer 403 "user not
    allowed to access this route, role= proxy_admin_viewer. Trying to access: /user/update and updating invalid
    param: <param>. only user_email can be updated" and the target keeps its role."""
    with gateway.scenario() as scenario:
        viewer: Final = caller(scenario, "proxy_admin_viewer")
        target: Final = scenario.user(user_role="internal_user")
        response: Final = post(gateway, "/user/update", {"user_id": target, "user_role": "proxy_admin"}, key=viewer.key)
        assert response.status_code == 403, response.text
        assert "only user_email can be updated" in error_text(response)
        assert "Trying to access: /user/update and updating invalid param: user_id" in error_text(response)
        assert user_rows(target)[0]["user_role"] == "internal_user"
