import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from pydantic import JsonValue

CONCURRENT_CREATES: Final = 8


def _membership_rows(organization_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT user_id, user_role FROM "LiteLLM_OrganizationMembership" WHERE organization_id = %s',
        (organization_id,),
    )


def _listed(gateway: Gateway, organization_id: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", "/organization/list")
    assert response.status_code == 200, response.text
    entries: Final = response.json()
    assert isinstance(entries, list)
    matches: Final = [entry for entry in entries if entry["organization_id"] == organization_id]
    assert len(matches) == 1, f"{organization_id} listed {len(matches)} times"
    return object_value(matches[0])


def test_concurrent_creates_with_one_alias_each_persist_a_distinct_organization(gateway: Gateway) -> None:
    alias: Final = f"integration-{uuid.uuid4().hex}"

    def create(_: int) -> httpx.Response:
        return gateway.request("POST", "/organization/new", {"organization_alias": alias})

    with ThreadPoolExecutor(max_workers=CONCURRENT_CREATES) as pool:
        responses: Final = tuple(pool.map(create, range(CONCURRENT_CREATES)))
    created: Final = tuple(response.json() for response in responses if response.status_code == 200)
    with gateway.scenario() as scenario:
        for body in created:
            scenario.cleanups.callback(
                scenario.delete_organization, string_value(body["organization_id"]), string_value(body["budget_id"])
            )
        assert [response.status_code for response in responses] == [200] * CONCURRENT_CREATES, [
            response.text for response in responses
        ]
        identities: Final = {string_value(body["organization_id"]) for body in created}
        assert len(identities) == CONCURRENT_CREATES
        rows: Final = read_rows(
            'SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_alias = %s', (alias,)
        )
        assert {string_value(row["organization_id"]) for row in rows} == identities


def test_list_returns_each_organization_with_its_budget_and_members(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization(max_budget=3.5, tpm_limit=120)
        member: Final = scenario.org_member(organization_id, role="internal_user")
        listed: Final = _listed(gateway, organization_id)
        budget: Final = object_value(listed["litellm_budget_table"])
        assert (budget["max_budget"], budget["tpm_limit"]) == (3.5, 120)
        members: Final = listed["members"]
        assert isinstance(members, list)
        assert [(object_value(entry)["user_id"], object_value(entry)["user_role"]) for entry in members] == [
            (member, "internal_user")
        ]


def test_member_role_update_and_removal_persist_and_read_back(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        organization_id: Final = scenario.organization()
        member: Final = scenario.org_member(organization_id, role="internal_user")
        assert _membership_rows(organization_id) == [{"user_id": member, "user_role": "internal_user"}]
        updated: Final = gateway.request(
            "PATCH",
            "/organization/member_update",
            {"organization_id": organization_id, "user_id": member, "role": "org_admin"},
        )
        assert updated.status_code == 200, updated.text
        assert _membership_rows(organization_id) == [{"user_id": member, "user_role": "org_admin"}]
        info_members: Final = gateway.get("/organization/info", {"organization_id": organization_id})["members"]
        assert isinstance(info_members, list)
        assert [object_value(entry)["user_role"] for entry in info_members] == ["org_admin"]
        removed: Final = gateway.request(
            "DELETE", "/organization/member_delete", {"organization_id": organization_id, "user_id": member}
        )
        assert removed.status_code == 200, removed.text
        assert _membership_rows(organization_id) == []
        assert _listed(gateway, organization_id)["members"] == []


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


def _new_key(gateway: Gateway, **fields: JsonValue) -> str:
    created: Final = gateway.post("/key/generate", fields)
    return string_value(created["key"])


def _key_rows(*hashes: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = ANY(%s)',
        (list(hashes),),
    )


def test_organization_delete_revokes_org_and_team_keys_on_gateway_and_peer(gateway: Gateway, peer: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        created_org: Final = gateway.post(
            "/organization/new", {"organization_alias": f"integration-{uuid.uuid4().hex}"}
        )
        organization_id: Final = string_value(created_org["organization_id"])
        other_organization_id: Final = scenario.organization()
        member: Final = scenario.user()
        membership: Final = gateway.request(
            "POST",
            "/organization/member_add",
            {"organization_id": organization_id, "member": {"role": "internal_user", "user_id": member}},
        )
        assert membership.status_code == 200, membership.text
        assert _membership_rows(organization_id) == [{"user_id": member, "user_role": "internal_user"}]
        team_id: Final = string_value(
            gateway.post(
                "/team/new",
                {"team_alias": f"integration-{uuid.uuid4().hex}", "organization_id": organization_id},
            )["team_id"]
        )
        organization_key: Final = _new_key(gateway, organization_id=organization_id)
        team_key: Final = _new_key(gateway, team_id=team_id)
        other_key: Final = _new_key(gateway, organization_id=other_organization_id)
        warmed_info: Final = peer.request("GET", "/organization/info", params={"organization_id": organization_id})
        assert warmed_info.status_code == 200, warmed_info.text

        _observed(upstream)
        for proxy in (gateway, peer):
            for key in (organization_key, team_key, other_key):
                text: Final = f"warm {uuid.uuid4().hex}"
                response: Final = _chat(proxy, model, key, text)
                assert response.status_code == 200, response.text
                assert _observed(upstream) == [_observation(text)]

        deleted: Final = gateway.request("DELETE", "/organization/delete", {"organization_ids": [organization_id]})
        assert deleted.status_code == 200, deleted.text

        for key in (organization_key, team_key):
            refused: Final = _chat(gateway, model, key, f"refused {uuid.uuid4().hex}")
            assert refused.status_code == 401, refused.text
            assert refused.json() == _token_not_found_error(key), refused.text
        for key in (organization_key, team_key):
            refused_peer: Final = eventually(
                lambda k=key: _chat(peer, model, k, f"refused {uuid.uuid4().hex}"),
                lambda response: response.status_code == 401,
                seconds=10,
            )
            assert refused_peer.json() == _token_not_found_error(key), refused_peer.text
        _observed(upstream)
        for proxy in (gateway, peer):
            for key in (organization_key, team_key):
                refused_again: Final = _chat(proxy, model, key, f"refused {uuid.uuid4().hex}")
                assert refused_again.status_code == 401, refused_again.text
        assert _observed(upstream) == []

        for proxy in (gateway, peer):
            text: Final = f"served {uuid.uuid4().hex}"
            response: Final = _chat(proxy, model, other_key, text)
            assert response.status_code == 200, response.text
            assert _observed(upstream) == [_observation(text)]

        hashed: Final = tuple(sha256(key.encode()).hexdigest() for key in (organization_key, team_key))
        assert _key_rows(*hashed) == []
        assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,)) == []
        assert _membership_rows(organization_id) == []
        assert (
            read_rows(
                'SELECT organization_id FROM "LiteLLM_OrganizationTable" WHERE organization_id = %s',
                (organization_id,),
            )
            == []
        )
        info: Final = gateway.request("GET", "/organization/info", params={"organization_id": organization_id})
        assert info.status_code == 404, info.text
        assert info.json() == {"detail": {"error": "Organization not found"}}, info.text
        peer_info: Final = eventually(
            lambda: peer.request("GET", "/organization/info", params={"organization_id": organization_id}),
            lambda response: response.status_code == 404,
            seconds=10,
        )
        assert peer_info.json() == {"detail": {"error": "Organization not found"}}, peer_info.text
        alive: Final = gateway.request("GET", "/organization/info", params={"organization_id": other_organization_id})
        assert alive.status_code == 200, alive.text
