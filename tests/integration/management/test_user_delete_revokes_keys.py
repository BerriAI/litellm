import uuid
from hashlib import sha256
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from pydantic import JsonValue


def _observed(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = response.json()["requests"]
    assert isinstance(requests, list)
    return requests


def _chat(proxy: Gateway, model: str, key: str, text: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text}]},
        key=key,
    )


def _observation(text: str) -> dict[str, JsonValue]:
    return {
        "path": "/v1/chat/completions",
        "authorization": "Bearer integration-provider-key",
        "body": {"messages": [{"role": "user", "content": text}], "model": "gpt-4o-mini"},
        "method": "POST",
        "api_key": "",
    }


def _token_not_found_error(key: str) -> dict[str, JsonValue]:
    hashed: Final = sha256(key.encode()).hexdigest()
    return {
        "error": {
            "message": (
                "Authentication Error, Invalid proxy server token passed. "
                f"Received API Key = sk-...{key[-4:]}, Key Hash (Token) ={hashed}. "
                "Unable to find token in cache or `LiteLLM_VerificationTokenTable`"
            ),
            "type": "token_not_found_in_db",
            "param": "key",
            "code": "401",
        }
    }


def _blocked_key_error() -> dict[str, JsonValue]:
    return {
        "error": {
            "message": "Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin.",
            "type": "auth_error",
            "param": "None",
            "code": "401",
        }
    }


def _user_rows(user_id: str) -> dict[str, list[dict[str, JsonValue]]]:
    return {
        "user": read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (user_id,)),
        "keys": read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id=%s', (user_id,)),
        "team_memberships": read_rows('SELECT user_id FROM "LiteLLM_TeamMembership" WHERE user_id=%s', (user_id,)),
        "org_memberships": read_rows(
            'SELECT user_id FROM "LiteLLM_OrganizationMembership" WHERE user_id=%s', (user_id,)
        ),
        "invitations": read_rows('SELECT id FROM "LiteLLM_InvitationLink" WHERE user_id=%s', (user_id,)),
    }


def _new_user(gateway: Gateway, **fields: JsonValue) -> str:
    created: Final = gateway.post(
        "/user/new", {"user_id": f"integration-{uuid.uuid4().hex}", "auto_create_key": False, **fields}
    )
    return string_value(created["user_id"])


def _new_key(gateway: Gateway, **fields: JsonValue) -> str:
    created: Final = gateway.post("/key/generate", fields)
    return string_value(created["key"])


def test_user_delete_revokes_keys_and_memberships(gateway: Gateway, peer: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        team_id: Final = scenario.team()
        organization_id: Final = scenario.organization()
        user_id: Final = _new_user(gateway)
        gateway.post("/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}})
        gateway.post(
            "/organization/member_add",
            {"organization_id": organization_id, "member": {"role": "internal_user", "user_id": user_id}},
        )
        keys: Final = (
            _new_key(gateway, user_id=user_id, models=[model]),
            _new_key(gateway, user_id=user_id, models=[model]),
        )
        invitation: Final = gateway.post("/invitation/new", {"user_id": user_id})
        assert string_value(object_value(invitation)["user_id"]) == user_id

        _observed(upstream)
        expected: Final = []
        for proxy in (gateway, peer):
            for key in keys:
                text: Final = f"warm {uuid.uuid4().hex}"
                assert _chat(proxy, model, key, text).status_code == 200
                expected.append(_observation(text))
        assert _observed(upstream) == expected

        deleted: Final = gateway.request("POST", "/user/delete", {"user_ids": [user_id]})
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == 1

        for key in keys:
            denied: Final = _chat(gateway, model, key, f"denied {uuid.uuid4().hex}")
            assert denied.status_code == 401, denied.text
            assert denied.json() == _token_not_found_error(key), denied.text
        for key in keys:
            denied_peer: Final = eventually(
                lambda k=key: _chat(peer, model, k, f"denied {uuid.uuid4().hex}"),
                lambda response: response.status_code == 401,
                seconds=10,
            )
            assert denied_peer.json() == _token_not_found_error(key), denied_peer.text

        _observed(upstream)
        for proxy in (gateway, peer):
            for key in keys:
                denied_again: Final = _chat(proxy, model, key, f"denied {uuid.uuid4().hex}")
                assert denied_again.status_code == 401, denied_again.text
        assert _observed(upstream) == []

        assert _user_rows(user_id) == {
            "user": [],
            "keys": [],
            "team_memberships": [],
            "org_memberships": [],
            "invitations": [],
        }
        team_row: Final = read_rows('SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team_id,))
        assert team_row == [
            {"members_with_roles": [{"role": "admin", "user_email": None, "user_id": "default_user_id"}]}
        ]
        team_info: Final = object_value(object_value(gateway.get("/team/info", {"team_id": team_id}))["team_info"])
        assert team_info["members_with_roles"] == [
            {"role": "admin", "user_alias": None, "user_id": "default_user_id", "user_email": None}
        ]

        info: Final = gateway.request("GET", "/user/info", params={"user_id": user_id})
        assert info.status_code == 404, info.text
        assert info.json() == {
            "error": {
                "message": f"User {user_id} not found",
                "type": "internal_server_error",
                "param": None,
                "code": "404",
            }
        }, info.text


def test_org_admin_can_only_delete_users_inside_their_orgs(
    gateway: Gateway,
) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        org_a: Final = scenario.organization()
        org_b: Final = scenario.organization()
        admin: Final = scenario.org_member(org_a, "org_admin")
        admin_key: Final = scenario.key(user_id=admin)

        victim_b: Final = scenario.org_member(org_b, "internal_user")
        victim_b_key: Final = scenario.key(user_id=victim_b, models=[model])
        victim_a: Final = _new_user(gateway)
        gateway.post(
            "/organization/member_add",
            {"organization_id": org_a, "member": {"role": "internal_user", "user_id": victim_a}},
        )

        _observed(upstream)
        refused: Final = gateway.request(
            "POST",
            "/user/delete",
            {"user_ids": [victim_b], "organization_id": org_a},
            key=admin_key,
        )
        assert refused.status_code == 403, refused.text
        assert refused.json() == {
            "detail": {
                "error": (
                    f"User {victim_b} is not within your admin scope. "
                    "Only PROXY_ADMIN may delete users outside your administered organizations."
                )
            }
        }, refused.text
        assert _user_rows(victim_b)["user"] != []

        alive: Final = f"still alive {uuid.uuid4().hex}"
        served: Final = _chat(gateway, model, victim_b_key, alive)
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [_observation(alive)]

        deleted: Final = gateway.request(
            "POST",
            "/user/delete",
            {"user_ids": [victim_a], "organization_id": org_a},
            key=admin_key,
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == 1
        assert _user_rows(victim_a)["user"] == []


def test_scim_user_delete_revokes_keys(gateway: Gateway, peer: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        user_id: Final = _new_user(gateway)
        key: Final = _new_key(gateway, user_id=user_id, models=[model])
        scenario.cleanups.callback(gateway.request, "POST", "/key/delete", {"keys": [key]})
        hashed: Final = sha256(key.encode()).hexdigest()

        assert _chat(gateway, model, key, f"warm {uuid.uuid4().hex}").status_code == 200
        assert _chat(peer, model, key, f"warm {uuid.uuid4().hex}").status_code == 200
        _observed(upstream)

        deleted: Final = gateway.request("DELETE", f"/scim/v2/Users/{user_id}")
        assert deleted.status_code == 204, deleted.text

        denied: Final = _chat(gateway, model, key, f"denied {uuid.uuid4().hex}")
        assert denied.status_code == 401, denied.text
        assert denied.json() == _blocked_key_error(), denied.text
        denied_peer: Final = eventually(
            lambda: _chat(peer, model, key, f"denied {uuid.uuid4().hex}"),
            lambda response: response.status_code == 401,
            seconds=10,
        )
        assert denied_peer.json() == _blocked_key_error(), denied_peer.text

        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (user_id,)) == []
        assert read_rows('SELECT blocked FROM "LiteLLM_VerificationToken" WHERE token=%s', (hashed,)) == [
            {"blocked": True}
        ]
        _observed(upstream)
        for proxy in (gateway, peer):
            denied_again: Final = _chat(proxy, model, key, f"denied {uuid.uuid4().hex}")
            assert denied_again.status_code == 401, denied_again.text
        assert _observed(upstream) == []
