"""Admin and master key only routes refuse virtual keys, and the refusal leaks nothing about the caller's key.

Contract (litellm/proxy/auth/route_checks.py `non_proxy_admin_allowed_routes_check` and
`_raise_admin_only_route_exception`; litellm/proxy/auth/user_api_key_auth.py, the `master_key_only_routes` check
right after the master key comparison; litellm/proxy/_types.py `LiteLLMRoutes`):

* A virtual key whose user is an `internal_user` gets 401 `auth_error` on a route outside the internal user,
  self managed and view only lists. The message is "Authentication Error, Only proxy admin can be used to
  generate, delete, update info for new keys/users/teams. Route=<route>. Your role=<role>. Your
  user_id=<masked>", where the user id keeps its first six and last two characters, and a key with no user
  shows `unknown` and `***`. Nothing is written.
* `/global/spend/reset` and the two in-memory cache diagnostics are for the master key only, whatever the key's
  role, including a key that belongs to a proxy admin user.
* None of these refusals contains the presented key or its hash.

The Rust core (#51374) moves `authz/` route decisions into pure functions; these tests pin the black box answers.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration.security.dec1_authsec_support import (
    ADMIN_ONLY_PREFIX,
    MASTER_ONLY_TEMPLATE,
    assert_auth_error,
    assert_not_echoed,
)

ADMIN_ONLY_WRITES: Final = (
    ("/user/new", {"user_id": "dec1-authsec-never-created"}),
    ("/team/new", {"team_alias": "dec1-authsec-never-created"}),
    ("/organization/new", {"organization_alias": "dec1-authsec-never-created"}),
    ("/budget/new", {"budget_id": "dec1-authsec-never-created", "max_budget": 1}),
    ("/customer/new", {"user_id": "dec1-authsec-never-created"}),
    ("/config/update", {"general_settings": {"dec1_authsec_probe": True}}),
)
MASTER_KEY_ONLY_ROUTES: Final = ("/global/spend/reset",)


def _masked(user_id: str) -> str:
    return user_id[:6] + "*" * (len(user_id) - 8) + user_id[-2:]


@pytest.mark.parametrize(("route", "body"), ADMIN_ONLY_WRITES, ids=[route for route, _ in ADMIN_ONLY_WRITES])
def test_an_internal_users_key_is_refused_on_admin_only_writes(gateway: Gateway, route: str, body: dict) -> None:
    """An internal user's virtual key calls an admin only write route. The proxy must answer 401 `auth_error`
    whose message starts with the admin only prefix and names `Route=<route>. Your role=internal_user`. The key
    must not appear in the body. The object named in the body must not have been created (checked for the user
    and team routes through their tables)."""
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user_id)
        refused: Final = gateway.request("POST", route, body, key=key)
        message: Final = assert_auth_error(refused, 401)
        assert message.startswith(f"{ADMIN_ONLY_PREFIX}Route={route}. Your role=internal_user."), message
        assert_not_echoed(refused, key)
    assert (
        gateway.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', ("dec1-authsec-never-created",)) == []
    )


def test_the_admin_only_refusal_masks_the_callers_user_id(gateway: Gateway) -> None:
    """An internal user with a 40 character user id calls `POST /user/new` with their key. The message must end
    with `Your user_id=<first 6>********************************<last 2>` (`RouteChecks._mask_user_id` with
    visible prefix 6 and suffix 2), never the full id."""
    with gateway.scenario() as scenario:
        user_id: Final = scenario.user(user_id=f"dec1-authsec-{uuid.uuid4().hex[:27]}", user_role="internal_user")
        key: Final = scenario.key(user_id=user_id)
        message: Final = assert_auth_error(gateway.request("POST", "/user/new", {}, key=key), 401)
        assert message.endswith(f"Your user_id={_masked(user_id)}"), message
        assert user_id not in message


def test_a_key_without_a_user_is_refused_on_admin_routes_as_unknown(gateway: Gateway) -> None:
    """A key generated with no `user_id` calls `POST /user/new`. The message must end with
    `Route=/user/new. Your role=unknown. Your user_id=***`: no user row is loaded, so nothing about the key's
    owner can leak. UNVERIFIED: that `/key/generate` without `user_id` leaves the key with no user row."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        message: Final = assert_auth_error(gateway.request("POST", "/user/new", {}, key=key), 401)
        assert message == f"{ADMIN_ONLY_PREFIX}Route=/user/new. Your role=unknown. Your user_id=***", message


def test_a_proxy_admin_users_key_can_call_admin_only_writes(gateway: Gateway) -> None:
    """Positive control: a key owned by a `proxy_admin` user calls `POST /user/new` for a fresh user id. The
    answer must be 200 and the user row must exist, so the refusals above are about the role and not about
    virtual keys in general."""
    with gateway.scenario() as scenario:
        admin: Final = scenario.user(user_role="proxy_admin")
        key: Final = scenario.key(user_id=admin)
        target: Final = f"dec1-authsec-made-{uuid.uuid4().hex}"
        created: Final = gateway.request("POST", "/user/new", {"user_id": target, "auto_create_key": False}, key=key)
        assert created.status_code == 200, created.text
        scenario.cleanups.callback(scenario.delete_user, target)
        assert gateway.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (target,)) != []


@pytest.mark.parametrize("route", MASTER_KEY_ONLY_ROUTES)
def test_a_proxy_admin_users_key_is_refused_on_master_key_only_routes(gateway: Gateway, route: str) -> None:
    """A key owned by a `proxy_admin` user calls a master key only route with nothing in its cache. The proxy
    must answer 401 with exactly `Authentication Error, Tried to access route=<route>, which is only for MASTER
    KEY`. Being an admin is not the same as holding the master key."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin"))
        refused: Final = gateway.request("POST", route, {}, key=key)
        assert_auth_error(refused, 401, MASTER_ONLY_TEMPLATE.format(route=route))


def test_a_warmed_proxy_admin_key_is_still_refused_on_a_master_key_only_route(gateway: Gateway) -> None:
    """A key owned by a `proxy_admin` user first calls `GET /user/list` (200, which puts the key in the auth
    cache), then calls `POST /global/spend/reset`. The second call must still be 401 with the MASTER KEY message.
    UNVERIFIED and suspected to fail: `_user_api_key_auth_builder` returns early for a cached key whose
    `user_role` is `proxy_admin` before it reaches the `master_key_only_routes` check, and #51374 keeps a
    `CachedAdmin` early return; if the cached object carries the admin role, a warmed admin key resets spend."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin"))
        assert gateway.request("GET", "/user/list", key=key).status_code == 200
        refused: Final = gateway.request("POST", "/global/spend/reset", {}, key=key)
        assert_auth_error(refused, 401, MASTER_ONLY_TEMPLATE.format(route="/global/spend/reset"))


def test_an_unknown_key_on_a_master_key_only_route_gets_the_route_refusal_without_echo(gateway: Gateway) -> None:
    """An unknown `sk-` key calls `POST /global/spend/reset`. The answer must be the master key only 401 (the
    route check runs before any database read) and must not contain the key or its hash."""
    secret: Final = f"sk-dec1-authsec-{uuid.uuid4().hex}"
    refused: Final = gateway.request("POST", "/global/spend/reset", {}, key=secret)
    assert_auth_error(refused, 401, MASTER_ONLY_TEMPLATE.format(route="/global/spend/reset"))
    assert_not_echoed(refused, secret)


def test_an_internal_user_key_cannot_read_another_users_info(gateway: Gateway) -> None:
    """An internal user's key calls `GET /user/info?user_id=<another user>`. The proxy must answer 403 whose text
    contains `key not allowed to access this user's info. user_id=<other>`, and must not return the other user's
    row (`RouteChecks.non_proxy_admin_allowed_routes_check`, the `/user/info` branch)."""
    with gateway.scenario() as scenario:
        caller: Final = scenario.user(user_role="internal_user")
        other: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=caller)
        refused: Final = gateway.request("GET", "/user/info", params={"user_id": other}, key=key)
        assert refused.status_code == 403, refused.text
        assert f"key not allowed to access this user's info. user_id={other}" in refused.text


def test_an_internal_user_key_reads_its_own_user_info(gateway: Gateway) -> None:
    """Positive control for the test above: the same kind of key calls `GET /user/info?user_id=<own id>` and must
    get 200."""
    with gateway.scenario() as scenario:
        caller: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=caller)
        assert gateway.request("GET", "/user/info", params={"user_id": caller}, key=key).status_code == 200
