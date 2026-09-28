import base64
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    call_tool,
    mcp_peer,
    register_mcp,
    tool_calls,
    tool_names,
)

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
