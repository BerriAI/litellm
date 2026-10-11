import base64
import re
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    Outcome,
    ScriptedTool,
    call_tool,
    forget_mcp,
    mcp_peer,
    openapi_peer,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
    tool_names,
)
from integration._support.oauth_server import oauth_server
from integration._support.process import owned_proxy
from pydantic import TypeAdapter

ADD: Final = {"a": 2, "b": 3}
STATIC_MODES: Final = (
    ("api_key", b"x-api-key", "{secret}"),
    ("bearer_token", b"authorization", "Bearer {secret}"),
    ("basic", b"authorization", "Basic {basic}"),
    ("authorization", b"authorization", "{secret}"),
)


def _header(call: dict[str, object], name: bytes) -> bytes | None:
    headers: Final = call["headers"]
    assert isinstance(headers, dict)
    value: Final = headers.get(name)
    return value if isinstance(value, bytes) else None


def _one_call(peer: McpPeer) -> dict[str, object]:
    sent: Final = tool_calls(peer.drain())
    assert len(sent) == 1, sent
    return sent[0]


def _plaintext_rows(identity: str, secret: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE server_id = %s '
        "AND (credentials::text LIKE %s OR static_headers::text LIKE %s)",
        (identity, f"%{secret}%", f"%{secret}%"),
    )


@pytest.mark.parametrize(("auth_type", "header", "shape"), STATIC_MODES)
def test_static_credential_reaches_the_peer_in_its_mode_shape_and_is_encrypted_at_rest(
    gateway: Gateway, auth_type: str, header: bytes, shape: str
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        secret: Final = "user:" + uuid.uuid4().hex
        basic: Final = base64.b64encode(secret.encode()).decode()
        identity: Final = register_mcp(
            scenario, peer, "cred" + uuid.uuid4().hex[:8], auth_type=auth_type, credentials={"auth_value": secret}
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        peer.drain()
        response: Final = call_tool(gateway, key, identity, tool_names(gateway, key, identity)["add"], ADD)
        assert response.status_code == 200, response.text
        assert _header(_one_call(peer), header) == shape.format(secret=secret, basic=basic).encode()
        assert _plaintext_rows(identity, secret) == [], "credential stored in plaintext"


def test_create_response_does_not_echo_the_stored_credential(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        secret: Final = "create-secret-" + uuid.uuid4().hex
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        response: Final = gateway.request(
            "POST",
            "/v1/mcp/server",
            {
                "server_name": alias,
                "alias": alias,
                **peer.registration(),
                "auth_type": "bearer_token",
                "credentials": {"auth_value": secret},
            },
        )
        scenario.cleanups.callback(forget_mcp, gateway, response.json()["server_id"])
        assert response.status_code == 201, response.text
        assert response.json()["alias"] == alias
        assert response.json().get("credentials") is None
        assert secret not in response.text


def test_editing_the_credential_rotates_what_the_peer_receives(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        first: Final = "cred-" + uuid.uuid4().hex
        second: Final = "cred-" + uuid.uuid4().hex
        identity: Final = register_mcp(
            scenario, peer, "cred" + uuid.uuid4().hex[:8], auth_type="bearer_token", credentials={"auth_value": first}
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        name: Final = tool_names(gateway, key, identity)["add"]
        peer.drain()
        assert call_tool(gateway, key, identity, name, ADD).status_code == 200
        assert _header(_one_call(peer), b"authorization") == f"Bearer {first}".encode()
        rotated: Final = gateway.request(
            "PUT", "/v1/mcp/server", {"server_id": identity, "credentials": {"auth_value": second}}
        )
        assert rotated.status_code == 202, rotated.text
        observed: Final = eventually(
            lambda: (call_tool(gateway, key, identity, name, ADD).status_code, tool_calls(peer.drain())),
            lambda value: any(_header(call, b"authorization") == f"Bearer {second}".encode() for call in value[1]),
        )
        assert all(_header(call, b"authorization") != f"Bearer {first}".encode() for call in observed[1][-1:])
        assert _plaintext_rows(identity, second) == [] and _plaintext_rows(identity, first) == []


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_caller_headers_for_other_servers_and_unknown_headers_never_reach_the_peer(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with mcp_peer() as peer, mcp_peer() as other, gateway.scenario() as scenario:
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        other_alias: Final = "cred" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        other_id: Final = register_mcp(scenario, other, other_alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity, other_id]})
        leak: Final = "leak-" + uuid.uuid4().hex
        caller: Final = McpCaller(
            gateway,
            key,
            entry,
            alias,
            headers={
                f"x-mcp-{other_alias}-authorization": f"Bearer {leak}",
                "x-integration-unknown": leak,
                "cookie": f"session={leak}",
            },
        )
        peer.drain()
        outcome: Final = caller.call(f"{alias}-add", ADD, identity if entry in ("mcp", "root", "sse", "rest") else None)
        assert outcome.ok, outcome.raw
        call: Final = _one_call(peer)
        assert leak.encode() not in b"".join(_header(call, name) or b"" for name in call["headers"]), call["headers"]
        assert tool_calls(other.drain()) == ()


@pytest.mark.parametrize(
    ("auth_type", "per_server_header"),
    (("none", True), ("true_passthrough", False)),
)
def test_callers_own_litellm_key_never_reaches_the_peer_over_rest(
    gateway: Gateway, auth_type: str, per_server_header: bool
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, auth_type=auth_type)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller_headers: Final = {
            "x-litellm-api-key": f"Bearer {key}",
            (f"x-mcp-{alias}-authorization" if per_server_header else "Authorization"): f"Bearer {key}",
        }
        peer.drain()
        response: Final = call_tool(
            gateway, key, identity, tool_names(gateway, key, identity)["add"], ADD, headers=caller_headers
        )
        assert response.status_code == 200, response.text
        assert response.json()["isError"] is False
        observed: Final = peer.drain()
        calls: Final = tool_calls(observed)
        assert len(calls) == 1, calls
        header_sets: Final = tuple(
            TypeAdapter(dict[bytes, bytes]).validate_python(request["headers"]) for request in observed
        )
        assert all(all(key.encode() not in value for value in headers.values()) for headers in header_sets), (
            "caller key appeared in the recorded MCP header set"
        )


@pytest.mark.parametrize("entry", ("server_mcp", "rest"))
def test_empty_primary_header_does_not_expose_authorization_key(gateway: Gateway, entry: EntryPoint) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, extra_headers=["x-upstream-token", "x-tenant"])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(
            gateway,
            key,
            entry,
            alias,
            headers={
                "x-litellm-api-key": "",
                "Authorization": f"Bearer {key}",
                f"x-mcp-{alias}-authorization": f"Bearer {key}",
                "x-upstream-token": key,
                "x-tenant": "tenant-control",
            },
        )
        peer.drain()
        outcome: Final = caller.call(f"{alias}-add", ADD, identity if entry == "rest" else None)
        assert outcome.ok, outcome.raw
        observed: Final = peer.drain()
        calls: Final = tool_calls(observed)
        assert len(calls) == 1, calls
        assert _header(calls[0], b"x-tenant") == b"tenant-control"
        header_sets: Final = tuple(
            TypeAdapter(dict[bytes, bytes]).validate_python(request["headers"]) for request in observed
        )
        assert all(all(key.encode() not in value for value in headers.values()) for headers in header_sets), (
            "empty primary header prevented scrubbing the admitted Authorization key"
        )


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_extra_headers_cannot_forward_gateway_admission_key(gateway: Gateway, entry: EntryPoint) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, extra_headers=["X-LiteLLM-API-Key", "x-tenant"],
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias, headers={"x-tenant": "tenant-control"})
        peer.drain()
        outcome: Final = caller.call(f"{alias}-add", ADD, identity if entry != "server_mcp" else None)
        assert outcome.ok, outcome.raw
        observed: Final = peer.drain()
        calls: Final = tool_calls(observed)
        assert len(calls) == 1, calls
        assert _header(calls[0], b"x-tenant") == b"tenant-control"
        header_sets: Final = tuple(
            TypeAdapter(dict[bytes, bytes]).validate_python(request["headers"]) for request in observed
        )
        assert all(all(key.encode() not in value for value in headers.values()) for headers in header_sets), (
            "gateway admission key reached the upstream through Extra Headers"
        )


@pytest.mark.parametrize("entry", ("server_mcp", "rest"))
def test_openapi_extra_headers_cannot_forward_gateway_admission_key(gateway: Gateway, entry: EntryPoint) -> None:
    with openapi_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, extra_headers=["X-LiteLLM-API-Key", "x-tenant"],
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias, headers={"x-tenant": "tenant-control"})
        peer.drain()
        outcome: Final = caller.call(f"{alias}-getpet", {"petId": "7"}, identity if entry == "rest" else None)
        assert outcome.ok, outcome.raw
        observed: Final = peer.drain()
        assert [(request["method"], request["path"]) for request in observed] == [("GET", "/pets/7")]
        headers: Final = TypeAdapter(dict[bytes, bytes]).validate_python(observed[0]["headers"])
        assert headers.get(b"x-tenant") == b"tenant-control"
        assert all(key.encode() not in value for value in headers.values()), "gateway admission key reached OpenAPI"


def test_server_scoped_caller_header_reaches_only_its_server(gateway: Gateway) -> None:
    with mcp_peer() as peer, mcp_peer() as other, gateway.scenario() as scenario:
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        other_alias: Final = "cred" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        other_id: Final = register_mcp(scenario, other, other_alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity, other_id]})
        token: Final = "user-" + uuid.uuid4().hex
        caller: Final = McpCaller(
            gateway, key, "mcp", None, headers={f"x-mcp-{alias}-authorization": f"Bearer {token}"}
        )
        peer.drain()
        other.drain()
        assert caller.call(f"{alias}-add", ADD).ok
        assert caller.call(f"{other_alias}-add", ADD).ok
        assert _header(_one_call(peer), b"authorization") == f"Bearer {token}".encode()
        assert _header(_one_call(other), b"authorization") is None


def test_extra_headers_allowlist_forwards_only_named_headers(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "cred" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, extra_headers=["x-tenant"])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "server_mcp", alias, headers={"x-tenant": "acme", "x-other": "no"})
        peer.drain()
        assert caller.call(f"{alias}-add", ADD).ok
        call: Final = _one_call(peer)
        assert _header(call, b"x-tenant") == b"acme"
        assert _header(call, b"x-other") is None


def test_byok_server_uses_the_calling_users_stored_credential_and_fails_closed_without_one(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "byok" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, auth_type="api_key", is_byok=True)
        owner: Final = scenario.user()
        stranger: Final = scenario.user()
        owner_key: Final = scenario.key(user_id=owner, object_permission={"mcp_servers": [identity]})
        stranger_key: Final = scenario.key(user_id=stranger, object_permission={"mcp_servers": [identity]})
        secret: Final = "byok-" + uuid.uuid4().hex
        stored: Final = gateway.client.post(
            f"/v1/mcp/server/{identity}/user-credential",
            json={"credential": secret},
            headers={"x-litellm-api-key": owner_key},
        )
        assert stored.status_code in (200, 201), stored.text
        scenario.cleanups.callback(
            gateway.client.delete,
            f"/v1/mcp/server/{identity}/user-credential",
            headers={"x-litellm-api-key": owner_key},
        )
        assert (
            read_rows(
                'SELECT credential_b64 FROM "LiteLLM_MCPUserCredentials" WHERE server_id = %s AND credential_b64 LIKE %s',
                (identity, f"%{secret}%"),
            )
            == []
        )
        name: Final = f"{alias}-add"
        peer.drain()
        granted: Final = call_tool(gateway, owner_key, identity, name, ADD)
        assert granted.status_code == 200, granted.text
        assert _header(_one_call(peer), b"x-api-key") == secret.encode()
        denied: Final = call_tool(gateway, stranger_key, identity, name, ADD)
        assert denied.status_code == 401, denied.text
        assert tool_calls(peer.drain()) == ()
        removed: Final = gateway.client.delete(
            f"/v1/mcp/server/{identity}/user-credential", headers={"x-litellm-api-key": owner_key}
        )
        assert removed.status_code in (200, 204), removed.text
        eventually(lambda: call_tool(gateway, owner_key, identity, name, ADD), lambda value: value.status_code == 401)
        assert tool_calls(peer.drain()) == ()


def _listings(peer: McpPeer) -> tuple[dict[str, object], ...]:
    return tuple(
        item
        for item in peer.drain()
        if isinstance(item.get("body"), dict) and item["body"].get("method") == "tools/list"
    )


def test_oauth2_byok_listing_sends_the_minted_token_not_the_users_stored_secret(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "cc" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario,
            peer,
            alias,
            auth_type="oauth2",
            oauth2_flow="client_credentials",
            is_byok=True,
            token_url=auth.issuer + "/token",
            credentials={"client_id": "cc-client", "client_secret": "cc-secret-" + uuid.uuid4().hex},
        )
        owner: Final = scenario.user()
        owner_key: Final = scenario.key(user_id=owner, object_permission={"mcp_servers": [identity]})
        secret: Final = "byok-" + uuid.uuid4().hex
        stored: Final = gateway.client.post(
            f"/v1/mcp/server/{identity}/user-credential",
            json={"credential": secret},
            headers={"x-litellm-api-key": owner_key},
        )
        assert stored.status_code in (200, 201), stored.text
        scenario.cleanups.callback(
            gateway.client.delete,
            f"/v1/mcp/server/{identity}/user-credential",
            headers={"x-litellm-api-key": owner_key},
        )
        peer.drain()
        auth.drain()
        response: Final = gateway.client.get(
            "/mcp-rest/tools/list", params={"server_id": identity}, headers={"x-litellm-api-key": owner_key}
        )
        assert response.status_code == 200, response.text
        assert "add" in {tool["name"] for tool in response.json()["tools"]}, response.text
        assert [request["grant_type"] for request in auth.token_requests()] == ["client_credentials"]
        listings: Final = _listings(peer)
        assert len(listings) == 1, listings
        sent: Final = _header(listings[0], b"authorization")
        assert sent is not None and auth.is_live(sent.decode().removeprefix("Bearer ")), sent
        assert secret.encode() not in sent, "stored BYOK secret replaced the minted token on tools/list"


@pytest.mark.parametrize(("auth_type", "header", "shape"), STATIC_MODES[:2])
def test_byok_rest_listing_sends_the_servers_static_credential_not_the_users_stored_secret(
    gateway: Gateway, auth_type: str, header: bytes, shape: str
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "byok" + uuid.uuid4().hex[:8]
        static: Final = "static-" + uuid.uuid4().hex
        identity: Final = register_mcp(
            scenario, peer, alias, auth_type=auth_type, is_byok=True, credentials={"auth_value": static}
        )
        owner: Final = scenario.user()
        owner_key: Final = scenario.key(user_id=owner, object_permission={"mcp_servers": [identity]})
        secret: Final = "byok-" + uuid.uuid4().hex
        stored: Final = gateway.client.post(
            f"/v1/mcp/server/{identity}/user-credential",
            json={"credential": secret},
            headers={"x-litellm-api-key": owner_key},
        )
        assert stored.status_code in (200, 201), stored.text
        scenario.cleanups.callback(
            gateway.client.delete,
            f"/v1/mcp/server/{identity}/user-credential",
            headers={"x-litellm-api-key": owner_key},
        )
        peer.drain()
        response: Final = gateway.client.get(
            "/mcp-rest/tools/list", params={"server_id": identity}, headers={"x-litellm-api-key": owner_key}
        )
        assert response.status_code == 200, response.text
        assert "add" in {tool["name"] for tool in response.json()["tools"]}, response.text
        listings: Final = _listings(peer)
        assert len(listings) == 1, listings
        assert _header(listings[0], header) == shape.format(secret=static, basic="").encode(), listings[0]["headers"]
        peer.drain()
        called: Final = call_tool(gateway, owner_key, identity, f"{alias}-add", ADD)
        assert called.status_code == 200, called.text
        assert _header(_one_call(peer), header) == shape.format(secret=secret, basic="").encode()


def test_deprecated_string_x_mcp_auth_lists_a_byok_server_for_a_key_without_a_user(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "byok" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, auth_type="bearer_token", is_byok=True)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        peer.drain()
        response: Final = gateway.client.get(
            "/mcp-rest/tools/list",
            params={"server_id": identity},
            headers={"x-litellm-api-key": key, "x-mcp-auth": "Bearer hdr"},
        )
        assert response.status_code == 200, response.text
        names: Final = {tool["name"] for tool in response.json()["tools"]}
        assert "add" in names, names
        listings: Final = _listings(peer)
        assert len(listings) == 1, listings
        assert listings[0]["headers"].get(b"authorization") == b"Bearer hdr"


_PROBE: Final = "catalog-probe"
_ECHO: Final = "catalog-echo"
_UNLISTED: Final = ""
_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    f'    if "{_PROBE}" not in list(inputs.get("texts") or []):\n'
    "        return allow()\n"
    '    function = inputs.get("tools", [{}])[0].get("function", {})\n'
    f'    return block("{_ECHO}[" + function.get("description") + "]")\n'
)


_ECHO_GUARDRAIL_YAML: Final = (
    "guardrails:\n"
    "  - guardrail_name: catalog-echo\n"
    "    litellm_params:\n"
    "      guardrail: custom_code\n"
    "      mode: pre_mcp_call\n"
    "      default_on: true\n"
    "      custom_code: |\n" + textwrap.indent(_GUARDRAIL_CODE, 8 * " ")
)


@pytest.fixture(scope="module")
def echo_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("catalog-echo")
    path: Final = directory / "catalog_echo.yaml"
    path.write_text((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text() + _ECHO_GUARDRAIL_YAML)
    with gateway_from_environment() as gateway, owned_proxy(gateway, directory, {}, config=path, workers=2) as rig:
        yield rig


def _echoed_description(outcome: Outcome) -> str:
    found: Final = re.search(rf"{_ECHO}\[(.*?)\]", outcome.raw)
    assert found is not None, outcome.raw
    return found.group(1)


_PROBE_ARGUMENTS: Final = {"probe": _PROBE}
_HEADERS: Final = TypeAdapter(dict[str, str])


def _store_byok_credential(scenario: Scenario, identity: str, key: str, secret: str) -> None:
    stored: Final = scenario.gateway.client.post(
        f"/v1/mcp/server/{identity}/user-credential", json={"credential": secret}, headers={"x-litellm-api-key": key}
    )
    assert stored.status_code in (200, 201), stored.text
    scenario.cleanups.callback(
        scenario.gateway.client.delete, f"/v1/mcp/server/{identity}/user-credential", headers={"x-litellm-api-key": key}
    )


def test_rotating_the_credential_drops_the_callers_listing_until_it_lists_again(echo_rig: Gateway) -> None:
    with mcp_peer() as peer, echo_rig.scenario() as scenario:
        first: Final = "cred-" + uuid.uuid4().hex
        second: Final = "cred-" + uuid.uuid4().hex
        alias: Final = "rot" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, auth_type="bearer_token", credentials={"auth_value": first}
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(echo_rig, key, "mcp", alias)
        assert caller.list_tools().ok
        assert _echoed_description(caller.call(f"{alias}-add", _PROBE_ARGUMENTS)) == "Add two integers"
        rotated: Final = echo_rig.request(
            "PUT", "/v1/mcp/server", {"server_id": identity, "credentials": {"auth_value": second}}
        )
        assert rotated.status_code == 202, rotated.text
        eventually(
            lambda: _echoed_description(caller.call(f"{alias}-add", _PROBE_ARGUMENTS)), lambda seen: seen == _UNLISTED
        )
        peer.drain()
        assert caller.list_tools().ok
        assert _echoed_description(caller.call(f"{alias}-add", _PROBE_ARGUMENTS)) == "Add two integers"
        relisted: Final = _listings(peer)
        assert len(relisted) == 1, relisted
        assert _header(relisted[0], b"authorization") == f"Bearer {second}".encode(), relisted[0]["headers"]


def test_byok_callers_are_evaluated_against_their_own_listing_and_the_stored_secret_never_keys_the_slot(
    echo_rig: Gateway,
) -> None:
    with mcp_peer() as peer, echo_rig.scenario() as scenario:
        alias: Final = "byok" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, auth_type="api_key", is_byok=True)
        owner_key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        stranger_key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        owner_secret: Final = "byok-" + uuid.uuid4().hex
        replacement: Final = "byok-" + uuid.uuid4().hex
        _store_byok_credential(scenario, identity, owner_key, owner_secret)
        _store_byok_credential(scenario, identity, stranger_key, "byok-" + uuid.uuid4().hex)
        owner: Final = McpCaller(echo_rig, owner_key, "mcp", alias)
        stranger: Final = McpCaller(echo_rig, stranger_key, "mcp", alias)
        peer.drain()
        assert owner.list_tools().ok
        listings: Final = _listings(peer)
        assert [_header(item, b"x-api-key") for item in listings] == [owner_secret.encode()], listings
        own: Final = _echoed_description(owner.call(f"{alias}-add", _PROBE_ARGUMENTS))
        other: Final = _echoed_description(stranger.call(f"{alias}-add", _PROBE_ARGUMENTS))
        assert (own, other) == ("Add two integers", _UNLISTED), (own, other)
        _store_byok_credential(scenario, identity, owner_key, replacement)
        assert _echoed_description(owner.call(f"{alias}-add", _PROBE_ARGUMENTS)) == "Add two integers", (
            "the slot is keyed by the client-supplied header, never by the stored credential"
        )
        sent: Final = eventually(
            lambda: (owner.call(f"{alias}-add", ADD).ok, tool_calls(peer.drain())),
            lambda value: any(_header(call, b"x-api-key") == replacement.encode() for call in value[1]),
        )
        assert sent[0], sent


def test_callers_with_different_server_scoped_auth_headers_are_evaluated_against_their_own_listings(
    echo_rig: Gateway,
) -> None:
    tool: Final = ScriptedTool(
        "add",
        lambda _: text_result("3"),
        description=lambda headers: "Adds for " + headers.get("authorization", "nobody"),
    )
    with scripted_peer(tool) as peer, echo_rig.scenario() as scenario:
        alias: Final = "scoped" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        acme_token: Final = "acme-" + uuid.uuid4().hex
        globex_token: Final = "globex-" + uuid.uuid4().hex
        acme: Final = McpCaller(echo_rig, key, "mcp", alias, {f"x-mcp-{alias}-authorization": f"Bearer {acme_token}"})
        globex: Final = McpCaller(
            echo_rig, key, "mcp", alias, {f"x-mcp-{alias}-authorization": f"Bearer {globex_token}"}
        )
        assert acme.list_tools().ok and globex.list_tools().ok
        seen: Final = (
            _echoed_description(acme.call(f"{alias}-add", _PROBE_ARGUMENTS)),
            _echoed_description(globex.call(f"{alias}-add", _PROBE_ARGUMENTS)),
        )
        assert seen == (f"Adds for Bearer {acme_token}", f"Adds for Bearer {globex_token}"), seen
        assert tool_calls(peer.drain()) == (), "a blocked probe reached the peer"


def test_deprecated_string_x_mcp_auth_callers_on_a_user_less_key_own_separate_listings(echo_rig: Gateway) -> None:
    tool: Final = ScriptedTool(
        "add",
        lambda _: text_result("3"),
        description=lambda headers: "Adds for " + headers.get("authorization", "nobody"),
    )
    with scripted_peer(tool) as peer, echo_rig.scenario() as scenario:
        alias: Final = "legacy" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, auth_type="bearer_token")
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        first_token: Final = "first-" + uuid.uuid4().hex
        second_token: Final = "second-" + uuid.uuid4().hex
        first: Final = McpCaller(echo_rig, key, "mcp", alias, {"x-mcp-auth": f"Bearer {first_token}"})
        second: Final = McpCaller(echo_rig, key, "mcp", alias, {"x-mcp-auth": f"Bearer {second_token}"})
        assert _echoed_description(first.call(f"{alias}-add", _PROBE_ARGUMENTS)) == _UNLISTED
        peer.drain()
        assert first.list_tools().ok
        listings: Final = _listings(peer)
        assert len(listings) == 1, listings
        listed_with: Final = _HEADERS.validate_python(listings[0]["headers"])
        assert listed_with.get("authorization") == f"Bearer {first_token}", listed_with
        warm: Final = eventually(
            lambda: _echoed_description(first.call(f"{alias}-add", _PROBE_ARGUMENTS)),
            lambda seen: seen != _UNLISTED,
        )
        assert warm == f"Adds for Bearer {first_token}", warm
        assert _echoed_description(second.call(f"{alias}-add", _PROBE_ARGUMENTS)) == _UNLISTED
        assert second.list_tools().ok
        seen: Final = eventually(
            lambda: (
                _echoed_description(first.call(f"{alias}-add", _PROBE_ARGUMENTS)),
                _echoed_description(second.call(f"{alias}-add", _PROBE_ARGUMENTS)),
            ),
            lambda pair: _UNLISTED not in pair,
        )
        assert seen == (f"Adds for Bearer {first_token}", f"Adds for Bearer {second_token}"), seen
        assert tool_calls(peer.drain()) == (), "a blocked probe reached the peer"


@pytest.mark.parametrize("entry", ("mcp", "server_mcp", "sse"))
@pytest.mark.parametrize("forwarded_token", (False, True))
def test_oauth_passthrough_probe_uses_server_token_without_admission_key(
    gateway: Gateway, entry: EntryPoint, forwarded_token: bool
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "probe" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, auth_type="none", oauth_passthrough=True, extra_headers=["Authorization"]
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        upstream_token: Final = "upstream-" + uuid.uuid4().hex
        caller: Final = McpCaller(
            gateway,
            key,
            entry,
            alias,
            headers={
                "Authorization": f"Bearer {upstream_token if forwarded_token else key}",
                f"x-mcp-{alias}-authorization": "Bearer expired-token"
                if forwarded_token
                else f"Bearer {upstream_token}",
            },
        )
        peer.drain()
        outcome: Final = caller.call(f"{alias}-add", ADD, identity if entry != "server_mcp" else None)
        assert outcome.ok, outcome.raw
        observed: Final = peer.drain()
        probes: Final = tuple(
            request
            for request in observed
            if TypeAdapter(dict[str, object]).validate_python(request["body"]).get("id") == "litellm-mcp-auth-probe"
        )
        assert probes, "expected an upstream initialize authentication probe"
        assert all(_header(probe, b"authorization") == f"Bearer {upstream_token}".encode() for probe in probes)
        assert len(tool_calls(observed)) == 1
        assert all(_header(request, b"authorization") == f"Bearer {upstream_token}".encode() for request in observed)
        header_sets: Final = tuple(
            TypeAdapter(dict[bytes, bytes]).validate_python(request["headers"]) for request in observed
        )
        assert all(all(key.encode() not in value for value in headers.values()) for headers in header_sets), (
            "upstream authentication probe exposed the gateway admission key"
        )
