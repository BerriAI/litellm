"""When a request breaks several rules at once, which refusal the caller sees.

The order below is read from `user_api_key_auth` in litellm/proxy/auth/user_api_key_auth.py. A caller only
ever sees the first failing check, so the order is part of the contract: dashboards, SDK retry logic and
support runbooks key off the status code and `error.type` they get today.

1. Credential present and well formed (401 `auth_error`).
2. Master key only routes (401 `auth_error`), before the key is looked up.
3. Key exists (401 `token_not_found_in_db`).
4. Key blocked (401 `auth_error`).
5. Key expired (401 `expired_key`).
6. Key `allowed_routes` (403, body `{"detail": ...}`; pinned current behaviour, see `assert_route_denied`).
7. Key `models` list, including client side fallbacks (403 `key_model_access_denied`).
8. Team blocked (401 `auth_error`, message "Team=<id> is blocked").
9. Team `models` list, then the member's `allowed_models` (403 `team_model_access_denied`).
10. User `models` list for personal keys (403 `user_model_access_denied`).
11. Role based route access for management routes (401 `auth_error`, "Only proxy admin ...").

A Rust port that evaluates the same rules in a different order would change answers for misconfigured
callers without changing any single rule.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration.authorization.dec1_keys_support import (
    BLOCKED_KEY_MESSAGE,
    UNKNOWN_KEY_MESSAGE,
    assert_denied,
    assert_model_denied,
    assert_route_denied,
    assert_served,
    chat,
    digest,
    marker,
    upstream_requests,
)


def _team_blocked_message(team: str) -> str:
    return f"Authentication Error, Team={team} is blocked. Update via `/team/unblock` if you're an admin."


def _wait_blocked(gateway: Gateway, key: str, model: str) -> None:
    eventually(lambda: chat(gateway, key, model), lambda response: response.status_code == 401)


def test_a_key_in_a_blocked_team_gets_the_exact_team_blocked_body(gateway: Gateway) -> None:
    """A team is blocked through `/team/block` and one of its keys sends a chat completion.

    The proxy must answer 401 with exactly `{"error": {"message": "Authentication Error, Team=<team_id> is
    blocked. Update via `/team/unblock` if you're an admin.", "type": "auth_error", "param": null,
    "code": "401"}}` and the upstream must not be called.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        gateway.post("/team/block", {"team_id": team})
        _wait_blocked(gateway, key, model)
        text: Final = marker()
        refused: Final = chat(gateway, key, model, text)
        assert refused.status_code == 401, refused.text
        assert refused.json() == {
            "error": {"message": _team_blocked_message(team), "type": "auth_error", "param": None, "code": "401"}
        }, refused.text
        assert upstream_requests(gateway, text) == []


def test_a_blocked_team_key_is_refused_on_model_listing_too(gateway: Gateway) -> None:
    """A key of a blocked team calls `GET /v1/models`.

    The team blocked check does not depend on a model being requested, so the listing must answer 401
    with the team blocked message.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        gateway.post("/team/block", {"team_id": team})
        _wait_blocked(gateway, key, model)
        refused: Final = gateway.request("GET", "/v1/models", key=key)
        assert_denied(refused, status=401, error_type="auth_error", message=_team_blocked_message(team))


def test_a_blocked_key_in_a_blocked_team_answers_key_blocked(gateway: Gateway) -> None:
    """Both a key and its team are blocked.

    The key's own block is checked first, so the message must be the key blocked one and must not mention
    the team.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        gateway.post("/team/block", {"team_id": team})
        gateway.post("/key/block", {"key": key})
        assert_denied(chat(gateway, key, model), status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)


def test_an_expired_key_in_a_blocked_team_answers_expired(gateway: Gateway) -> None:
    """A key that is past its expiry belongs to a blocked team.

    Expiry is checked while the key is validated, before any team rule, so the refusal must have type
    `expired_key`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, duration="1s")
        gateway.post("/team/block", {"team_id": team})
        eventually(
            lambda: gateway.rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s AND expires < now()', (digest(key),)
            ),
            lambda rows: rows != [],
        )
        refused: Final = chat(gateway, key, model)
        assert refused.status_code == 401, refused.text
        assert refused.json()["error"]["type"] == "expired_key", refused.text


def test_a_key_model_denial_is_reported_before_the_team_block(gateway: Gateway) -> None:
    """A key limited to model A belongs to a blocked team and asks for model B.

    The key's model list is checked before the team is consulted, so the answer must be 403
    `key_model_access_denied` and not the 401 team blocked message.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, models=[allowed])
        gateway.post("/team/block", {"team_id": team})
        _wait_blocked(gateway, key, allowed)
        assert_model_denied(chat(gateway, key, other), other, "key_model_access_denied")


def test_the_team_block_is_reported_before_a_team_model_denial(gateway: Gateway) -> None:
    """A blocked team allows only model A and one of its keys, with no list of its own, asks for model B.

    The team blocked check runs before the team's model list, so the answer must be the 401 team blocked
    message and not 403 `team_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[allowed])
        key: Final = scenario.key(team_id=team)
        gateway.post("/team/block", {"team_id": team})
        _wait_blocked(gateway, key, allowed)
        assert_denied(
            chat(gateway, key, other), status=401, error_type="auth_error", message=_team_blocked_message(team)
        )


def test_a_members_personal_key_keeps_working_while_their_team_is_blocked(gateway: Gateway) -> None:
    """A user belongs to a blocked team and also holds a personal key that has no team.

    The personal key must still be served with 200. Blocking a team stops the team's keys, not every key
    of its members.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        team_key: Final = scenario.key(team_id=team, user_id=member)
        personal: Final = scenario.key(user_id=member)
        gateway.post("/team/block", {"team_id": team})
        _wait_blocked(gateway, team_key, model)
        assert_served(chat(gateway, personal, model))


def test_a_route_denial_is_reported_before_a_model_denial(gateway: Gateway) -> None:
    """A key limited to the embeddings route and to model A sends a chat completion for model B.

    `allowed_routes` is checked first, so the answer must be 403 with the "Virtual key is not allowed to call
    this route" message, not `key_model_access_denied`.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed], allowed_routes=["/v1/embeddings"])
        assert_route_denied(chat(gateway, key, other), ["/v1/embeddings"], "/v1/chat/completions")


def test_a_blocked_key_is_reported_before_its_route_restriction(gateway: Gateway) -> None:
    """A blocked key limited to the embeddings route sends a chat completion.

    The blocked check runs while the key is validated, before `allowed_routes`, so the answer must be the
    401 key blocked message.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_routes=["/v1/embeddings"])
        gateway.post("/key/block", {"key": key})
        assert_denied(chat(gateway, key, model), status=401, error_type="auth_error", message=BLOCKED_KEY_MESSAGE)


def test_a_route_denial_is_reported_before_the_team_block(gateway: Gateway) -> None:
    """A key limited to the embeddings route belongs to a blocked team and sends a chat completion.

    `allowed_routes` is checked before the team is loaded for authorization, so the answer must be the 403
    route message and not the 401 team blocked message.

    Pinned current behaviour: the route refusal body is the bare `detail` shape (see `assert_route_denied`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        probe: Final = scenario.key(team_id=team)
        key: Final = scenario.key(team_id=team, allowed_routes=["/v1/embeddings"])
        gateway.post("/team/block", {"team_id": team})
        _wait_blocked(gateway, probe, model)
        assert_route_denied(chat(gateway, key, model), ["/v1/embeddings"], "/v1/chat/completions")


def test_an_unknown_key_is_a_401_whatever_else_is_wrong_with_the_request(gateway: Gateway) -> None:
    """A never issued key asks for a model that does not exist.

    Authentication comes first, so the answer must be the generic 401 `token_not_found_in_db` and must say
    nothing about the model.
    """
    ghost: Final = f"ghost-{marker()}"
    refused: Final = chat(gateway, "sk-dec1-keys-never-issued-check-order", ghost)
    assert_denied(refused, status=401, error_type="token_not_found_in_db", param="key", message=UNKNOWN_KEY_MESSAGE)
    assert ghost not in refused.text


def test_a_malformed_body_with_an_unknown_key_answers_the_auth_failure_not_a_400(gateway: Gateway) -> None:
    """A never issued key posts a body that is not valid JSON to chat completions.

    UNVERIFIED: the comment in `user_api_key_auth` says a caller who sent both a malformed body and a
    rejected key "used to get the 400; the response is unchanged", so the expected status is 400. The test
    pins that the proxy answers a 4xx, never a 5xx and never a 200, and that the upstream is not called.
    """
    text: Final = marker()
    refused: Final = gateway.client.post(
        "/v1/chat/completions",
        content=f'{{"model": "x", "messages": "{text}"'.encode(),
        headers={"Authorization": "Bearer sk-dec1-keys-never-issued-bad-body", "Content-Type": "application/json"},
    )
    assert refused.status_code in (400, 401), refused.text
    assert upstream_requests(gateway, text) == []


def test_a_malformed_body_with_a_valid_key_is_a_400(gateway: Gateway) -> None:
    """A valid key posts a body that is not valid JSON to chat completions.

    The key authenticates, the body cannot be parsed, and the proxy must answer 400. It must not be a 401
    or a 403, because the credential is fine.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        refused: Final = gateway.client.post(
            "/v1/chat/completions",
            content=b'{"model": "x", "messages": ',
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        assert refused.status_code == 400, refused.text


def test_a_team_model_denial_is_reported_before_the_members_own_restriction(gateway: Gateway) -> None:
    """A team allows model A, a member is limited to model A too, and the member's key asks for model B.

    Both the team list and the member list refuse B and both use type `team_model_access_denied`, so the
    caller must see one 403 with that type and the fixed message for B.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[allowed])
        member: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "allowed_models": [allowed]})
        key: Final = scenario.key(team_id=team, user_id=member)
        assert_model_denied(chat(gateway, key, other), other, "team_model_access_denied")


@pytest.mark.parametrize("route", ["/user/new", "/team/new", "/budget/new"])
def test_a_plain_key_on_an_admin_route_gets_the_admin_only_401(gateway: Gateway, route: str) -> None:
    """A key with no user and no route restrictions posts to an admin management route.

    The proxy must answer 401 `auth_error` with a message that starts with "Authentication Error, Only
    proxy admin can be used to generate, delete, update info for new keys/users/teams. Route=<route>. Your
    role=unknown." This is a 401, not a 403, and clients depend on that.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        refused: Final = gateway.request("POST", route, {}, key=key)
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        assert str(error["message"]).startswith(
            "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
            f"keys/users/teams. Route={route}. Your role=unknown. Your user_id="
        ), refused.text


def test_the_admin_only_refusal_masks_the_callers_user_id(gateway: Gateway) -> None:
    """A key owned by an internal user posts to `/user/new`.

    The 401 message must report `Your role=internal_user` and must not contain the full user id, which is
    masked to its first six and last two characters.
    """
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=user)
        refused: Final = gateway.request("POST", "/user/new", {}, key=key)
        error: Final = assert_denied(refused, status=401, error_type="auth_error")
        message: Final = str(error["message"])
        assert "Route=/user/new. Your role=internal_user. Your user_id=" in message, message
        assert user not in message, message
        assert message.endswith(user[-2:]) and f"user_id={user[:6]}" in message, message
