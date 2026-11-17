"""Per-key, per-team and per-organization MCP server and tool allowlists on the Python gateway.

The contract (`MCPRequestHandler.get_allowed_mcp_servers` and `_single_credential_tools` in
`litellm/proxy/mcp_server/auth/user_api_key_auth_mcp.py`): a key that grants no servers of its own
inherits its team's `object_permission.mcp_servers`; a key and a team that both grant servers intersect;
a team with no grant does not restrict. An organization list is a ceiling: with no lower restriction it
becomes the allowed set, otherwise it intersects. Tool lists in `mcp_tool_permissions` follow the same
inheritance, may be keyed by server id or by server name, and an explicit empty list denies every tool on
that server. A config-declared `allowed_params` map refuses arguments outside the list with HTTP 403.

The Rust host has none of this today: its `Registry` resolver hands every authenticated caller every
configured server (`gateway-mcp/src/native/registry.rs`), and only a per-server `allowed_tools` filter
exists. These tests describe what the port must reproduce before any key-scoped deployment moves.
`test_mcp_access_matrix.py`, `test_mcp_tool_permission_merge.py` and `test_mcp_toolsets.py` cover
single-key grants, server allow and deny lists and toolsets; this file covers the inheritance arithmetic.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.mcp import McpCaller, McpPeer, listed_tools, mcp_peer, register_mcp, tool_calls
from integration._support.process import owned_proxy

from tests.integration.mcp.dec1_mcp_support import config_with_servers, server_ids_by_name

pytestmark: Final = pytest.mark.shares_worker_state


def _server(scenario: Scenario, peer: McpPeer, stem: str) -> tuple[str, str]:
    alias: Final = stem + uuid.uuid4().hex[:8]
    return alias, register_mcp(scenario, peer, alias)


def _rest_call(gateway: Gateway, key: str, identity: str, name: str, arguments: dict[str, object]):  # noqa: ANN202
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key},
        json={"server_id": identity, "name": name, "arguments": arguments},
    )


def test_a_team_key_without_its_own_grant_inherits_the_team_servers(gateway: Gateway) -> None:
    """A team grants server A through `object_permission.mcp_servers`; its key grants nothing. The key's
    `/mcp` listing must contain `<A>-add` and a call to it must return `5` (inheritance arm `base = team_set`)."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias, identity = _server(scenario, peer, "dec1inh")
        team: Final = scenario.team(object_permission={"mcp_servers": [identity]})
        key: Final = scenario.key(team_id=team)
        caller: Final = McpCaller(gateway, key, "mcp")
        assert f"{alias}-add" in caller.list_tools().tools
        assert caller.call(f"{alias}-add", {"a": 2, "b": 3}).text == "5"


def test_a_key_grant_outside_the_team_grant_is_cut_by_the_intersection(gateway: Gateway) -> None:
    """The team grants A; its key grants A and B. Both restrict, so the key reaches only A: the listing has
    `<A>-add` and no `<B>-` tool, and a REST call to B is HTTP 403 with
    `{"error": "access_denied", "message": "The key is not allowed to access server <B id>"}` while the peer
    of B sees no `tools/call`."""
    with mcp_peer() as peer_a, mcp_peer() as peer_b, gateway.scenario() as scenario:
        alias_a, a = _server(scenario, peer_a, "dec1ia")
        alias_b, b = _server(scenario, peer_b, "dec1ib")
        team: Final = scenario.team(object_permission={"mcp_servers": [a]})
        key: Final = scenario.key(team_id=team, object_permission={"mcp_servers": [a, b]})
        tools: Final = McpCaller(gateway, key, "mcp").list_tools().tools
        assert f"{alias_a}-add" in tools, tools
        assert not any(name.startswith(f"{alias_b}-") for name in tools), tools
        peer_b.drain()
        denied: Final = _rest_call(gateway, key, b, f"{alias_b}-add", {"a": 1, "b": 1})
        assert denied.status_code == 403, denied.text
        assert denied.json()["detail"] == {
            "error": "access_denied",
            "message": f"The key is not allowed to access server {b}",
        }
        assert tool_calls(peer_b.drain()) == ()


def test_a_team_without_any_mcp_grant_does_not_restrict_its_key(gateway: Gateway) -> None:
    """The team has no `object_permission`; its key grants server B. An empty team set means "no team
    restriction" (`base = key_set`), so `<B>-add` must be listed and return `5`."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias, identity = _server(scenario, peer, "dec1open")
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, object_permission={"mcp_servers": [identity]})
        assert McpCaller(gateway, key, "mcp").call(f"{alias}-add", {"a": 2, "b": 3}).text == "5"


def test_a_key_and_team_with_disjoint_grants_reach_no_server(gateway: Gateway) -> None:
    """The team grants A and its key grants only B. The intersection is empty, so the aggregate listing has
    no `<A>-` or `<B>-` tool and a call to either is refused before any peer sees `tools/call`."""
    with mcp_peer() as peer_a, mcp_peer() as peer_b, gateway.scenario() as scenario:
        alias_a, a = _server(scenario, peer_a, "dec1da")
        alias_b, b = _server(scenario, peer_b, "dec1db")
        team: Final = scenario.team(object_permission={"mcp_servers": [a]})
        key: Final = scenario.key(team_id=team, object_permission={"mcp_servers": [b]})
        listed: Final = McpCaller(gateway, key, "mcp").list_tools()
        assert not any(name.startswith((f"{alias_a}-", f"{alias_b}-")) for name in listed.tools), listed.raw
        peer_a.drain()
        peer_b.drain()
        for alias in (alias_a, alias_b):
            assert McpCaller(gateway, key, "mcp").call(f"{alias}-add", {"a": 1, "b": 1}).error is not None
        assert tool_calls(peer_a.drain()) == () and tool_calls(peer_b.drain()) == ()


def test_team_tool_permissions_cap_a_wider_key_tool_list(gateway: Gateway) -> None:
    """Server A is granted to both. The team's `mcp_tool_permissions` is `{A: ["add"]}` and the key's is
    `{A: ["add", "multiply"]}`. Tool lists intersect, so the key lists `<A>-add` but not `<A>-multiply`, and
    a REST call to `multiply` is HTTP 403 with `detail.error` equal to `Tool 'multiply' is not allowed for
    your key/team on server '<A name>'. Contact proxy admin for access.`
    UNVERIFIED: the server name in the message is assumed to be the alias the server was registered with."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias, identity = _server(scenario, peer, "dec1tt")
        team: Final = scenario.team(
            object_permission={"mcp_servers": [identity], "mcp_tool_permissions": {identity: ["add"]}}
        )
        key: Final = scenario.key(
            team_id=team,
            object_permission={"mcp_servers": [identity], "mcp_tool_permissions": {identity: ["add", "multiply"]}},
        )
        tools: Final = McpCaller(gateway, key, "mcp").list_tools().tools
        assert f"{alias}-add" in tools and f"{alias}-multiply" not in tools, tools
        peer.drain()
        denied: Final = _rest_call(gateway, key, identity, "multiply", {"a": 2, "b": 3})
        assert denied.status_code == 403, denied.text
        assert denied.json()["detail"] == {
            "error": f"Tool 'multiply' is not allowed for your key/team on server '{alias}'. Contact proxy admin for access."
        }, denied.text
        assert tool_calls(peer.drain()) == ()


def test_a_tool_permission_map_keyed_by_server_name_restricts_like_one_keyed_by_id(gateway: Gateway) -> None:
    """The key's `mcp_tool_permissions` is `{"<alias>": ["add"]}` (a name, not the uuid). The map is
    normalized to server ids (`expand_tool_permissions`), so `multiply` must be hidden and refused exactly
    as if the map were keyed by id; a port that only matched ids would silently drop the restriction."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias, identity = _server(scenario, peer, "dec1byname")
        key: Final = scenario.key(
            object_permission={"mcp_servers": [identity], "mcp_tool_permissions": {alias: ["add"]}}
        )
        tools: Final = McpCaller(gateway, key, "mcp").list_tools().tools
        assert f"{alias}-add" in tools and f"{alias}-multiply" not in tools, tools
        assert McpCaller(gateway, key, "mcp").call(f"{alias}-multiply", {"a": 2, "b": 3}).error is not None


def test_an_explicit_empty_tool_list_denies_every_tool_on_that_server(gateway: Gateway) -> None:
    """The key grants server A with `mcp_tool_permissions: {A: []}`. An explicit empty allowlist is a deny
    (`is not None` check), not "no restriction", so the listing has no `<A>-` tool and `<A>-add` is refused
    without reaching the peer."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias, identity = _server(scenario, peer, "dec1empty")
        key: Final = scenario.key(object_permission={"mcp_servers": [identity], "mcp_tool_permissions": {identity: []}})
        tools: Final = McpCaller(gateway, key, "mcp").list_tools().tools
        assert not any(name.startswith(f"{alias}-") for name in tools), tools
        peer.drain()
        assert McpCaller(gateway, key, "mcp").call(f"{alias}-add", {"a": 1, "b": 1}).error is not None
        assert tool_calls(peer.drain()) == ()


def test_tool_permissions_for_one_server_do_not_restrict_another(gateway: Gateway) -> None:
    """The key grants A and B with `mcp_tool_permissions: {A: ["add"]}` only. B has no entry, so every B
    tool stays callable: `<B>-multiply` must return `6`."""
    with mcp_peer() as peer_a, mcp_peer() as peer_b, gateway.scenario() as scenario:
        _, a = _server(scenario, peer_a, "dec1pa")
        alias_b, b = _server(scenario, peer_b, "dec1pb")
        key: Final = scenario.key(object_permission={"mcp_servers": [a, b], "mcp_tool_permissions": {a: ["add"]}})
        assert McpCaller(gateway, key, "mcp").call(f"{alias_b}-multiply", {"a": 2, "b": 3}).text == "6"


def test_an_organization_list_becomes_the_ceiling_for_an_unrestricted_key(gateway: Gateway) -> None:
    """An organization's `object_permission.mcp_servers` is `[A]`; a key in that organization grants
    nothing and has no team. With no lower restriction the org list becomes the allowed set
    (`_apply_org_server_ceiling`), so the key lists `<A>-add` and no tool of a second server B granted to
    nobody. UNVERIFIED: `/organization/new` is assumed to persist `object_permission` like `/team/new`."""
    with mcp_peer() as peer_a, mcp_peer() as peer_b, gateway.scenario() as scenario:
        alias_a, a = _server(scenario, peer_a, "dec1oa")
        alias_b, _ = _server(scenario, peer_b, "dec1ob")
        organization: Final = scenario.organization(object_permission={"mcp_servers": [a]})
        key: Final = scenario.key(organization_id=organization)
        tools: Final = McpCaller(gateway, key, "mcp").list_tools().tools
        assert f"{alias_a}-add" in tools, tools
        assert not any(name.startswith(f"{alias_b}-") for name in tools), tools


def test_an_organization_list_cuts_a_key_grant_outside_it(gateway: Gateway) -> None:
    """The organization allows only A; its key grants A and B. The org ceiling intersects, so `<B>-add` is
    refused and its peer sees no `tools/call`, while `<A>-add` returns `5`."""
    with mcp_peer() as peer_a, mcp_peer() as peer_b, gateway.scenario() as scenario:
        alias_a, a = _server(scenario, peer_a, "dec1oc")
        alias_b, b = _server(scenario, peer_b, "dec1od")
        organization: Final = scenario.organization(object_permission={"mcp_servers": [a]})
        key: Final = scenario.key(organization_id=organization, object_permission={"mcp_servers": [a, b]})
        caller: Final = McpCaller(gateway, key, "mcp")
        assert caller.call(f"{alias_a}-add", {"a": 2, "b": 3}).text == "5"
        peer_b.drain()
        assert caller.call(f"{alias_b}-add", {"a": 2, "b": 3}).error is not None
        assert tool_calls(peer_b.drain()) == ()


def test_allowed_params_refuses_an_argument_outside_the_list_with_the_exact_message(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A config-declared server has `allowed_params: {add: [a, b]}`. A REST call to `add` with
    `{a: 1, b: 2, c: 3}` must be HTTP 403 with `detail.error` equal to `Parameters ['c'] are not allowed for
    tool add. Allowed parameters: ['a', 'b']. Contact proxy admin to allow these parameters.` and never
    reach the peer; the same call without `c` returns `3`. The Rust host refuses unknown server settings at
    boot, so `allowed_params` is an unported feature."""
    with mcp_peer() as peer:
        name: Final = "dec1params" + uuid.uuid4().hex[:8]
        config: Final = config_with_servers(
            tmp_path, {name: {**peer.registration(), "allow_all_keys": True, "allowed_params": {"add": ["a", "b"]}}}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as owned:
            identity: Final = server_ids_by_name(owned)[name]
            peer.drain()
            refused: Final = _rest_call(owned, owned.key, identity, f"{name}-add", {"a": 1, "b": 2, "c": 3})
            assert refused.status_code == 403, refused.text
            assert refused.json()["detail"] == {
                "error": "Parameters ['c'] are not allowed for tool add. Allowed parameters: ['a', 'b']. "
                "Contact proxy admin to allow these parameters."
            }, refused.text
            assert tool_calls(peer.drain()) == ()
            allowed: Final = _rest_call(owned, owned.key, identity, f"{name}-add", {"a": 1, "b": 2})
            assert allowed.status_code == 200, allowed.text
            assert allowed.json()["content"][0]["text"] == "3", allowed.text


def test_allowed_params_keyed_by_the_prefixed_tool_name_also_applies(gateway: Gateway, tmp_path: Path) -> None:
    """`allowed_params` is keyed `<name>-add: [a]`. `match_known_tool_name` accepts every spelling routing
    accepts, so a call to `add` with `{a: 1, b: 2}` must still be HTTP 403 naming `['b']`. The tools/call
    over `/mcp` is refused too (JSON-RPC error or `isError`)."""
    with mcp_peer() as peer:
        name: Final = "dec1pparams" + uuid.uuid4().hex[:8]
        config: Final = config_with_servers(
            tmp_path, {name: {**peer.registration(), "allow_all_keys": True, "allowed_params": {f"{name}-add": ["a"]}}}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as owned:
            identity: Final = server_ids_by_name(owned)[name]
            refused: Final = _rest_call(owned, owned.key, identity, f"{name}-add", {"a": 1, "b": 2})
            assert refused.status_code == 403, refused.text
            assert "Parameters ['b'] are not allowed" in refused.text, refused.text
            assert McpCaller(owned, owned.key, "mcp").call(f"{name}-add", {"a": 1, "b": 2}).error is not None


def test_allowed_params_for_one_tool_leaves_other_tools_unrestricted(gateway: Gateway, tmp_path: Path) -> None:
    """`allowed_params: {add: [a, b]}` says nothing about `multiply`, so `multiply` with `{a: 2, b: 3}` is
    served (`6`): `validate_allowed_params` returns early when no entry matches the tool."""
    with mcp_peer() as peer:
        name: Final = "dec1oparams" + uuid.uuid4().hex[:8]
        config: Final = config_with_servers(
            tmp_path, {name: {**peer.registration(), "allow_all_keys": True, "allowed_params": {"add": ["a", "b"]}}}
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as owned:
            assert McpCaller(owned, owned.key, "mcp").call(f"{name}-multiply", {"a": 2, "b": 3}).text == "6"


def test_an_invalid_tool_display_name_is_refused_at_registration(gateway: Gateway) -> None:
    """`POST /v1/mcp/server` with `tool_name_to_display_name: {add: "add numbers"}` must be HTTP 400 with
    `detail.error` equal to `Invalid display name 'add numbers' for tool 'add'. Display names may only
    contain letters, digits, underscores, and hyphens (no spaces or other special characters), since they
    replace the tool name sent to the LLM provider.` (pattern `^[a-zA-Z0-9_-]+$`)."""
    with mcp_peer() as peer:
        response: Final = gateway.request(
            "POST",
            "/v1/mcp/server",
            {
                "server_name": "dec1disp" + uuid.uuid4().hex[:8],
                "tool_name_to_display_name": {"add": "add numbers"},
                **peer.registration(),
            },
        )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {
        "error": "Invalid display name 'add numbers' for tool 'add'. Display names may only contain letters, "
        "digits, underscores, and hyphens (no spaces or other special characters), since they replace the "
        "tool name sent to the LLM provider."
    }


def test_a_rest_listing_with_no_grant_at_all_answers_200_with_an_unexpected_error_body(gateway: Gateway) -> None:
    """A key with no MCP grant (and no team, no org) calls `GET /mcp-rest/tools/list` with no `server_id`.
    The handler raises a 403 `access_denied` internally but the aggregate branch keeps the legacy shape:
    HTTP 200 with `tools == []`, `error == "unexpected_error"` and a message starting `An unexpected error
    occurred:` that contains `The key is not allowed to access any MCP servers.` (`rest_endpoints.py`).
    UNVERIFIED: assumes no `allow_all_keys` server is configured on the shared proxy; if one is, the key
    sees that server instead and the test is skipped."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = gateway.client.get("/mcp-rest/tools/list", headers={"x-litellm-api-key": key})
    assert response.status_code == 200, response.text
    body: Final = response.json()
    if body.get("tools"):
        pytest.skip("the shared proxy exposes an allow_all_keys server to every key")
    assert body["error"] == "unexpected_error", body
    assert str(body["message"]).startswith("An unexpected error occurred:"), body
    assert "The key is not allowed to access any MCP servers." in str(body["message"]), body


def test_a_rest_listing_of_an_unknown_server_id_is_404_with_server_not_found(gateway: Gateway) -> None:
    """A granted key asks `GET /mcp-rest/tools/list?server_id=<random uuid>`. The single-server branch relays
    the truthful status: HTTP 404 with `detail == {"error": "server_not_found", "message": "MCP server
    '<uuid>' was not found"}`."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        _, identity = _server(scenario, peer, "dec1known")
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        missing: Final = str(uuid.uuid4())
        response: Final = gateway.client.get(
            "/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params={"server_id": missing}
        )
    assert response.status_code == 404, response.text
    assert response.json()["detail"] == {"error": "server_not_found", "message": f"MCP server '{missing}' was not found"}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"name": "add", "arguments": {}}, "server_id is required in request body"),
        ({"server_id": "PLACEHOLDER", "arguments": {}}, "name is required in request body"),
    ],
)
def test_a_rest_call_missing_a_required_field_is_400_missing_parameter(
    gateway: Gateway, body: dict[str, object], message: str
) -> None:
    """`POST /mcp-rest/tools/call` without `server_id` (or without `name`) must be HTTP 400 with
    `detail == {"error": "missing_parameter", "message": "<field> is required in request body"}`. The Rust
    host answers the same status with a flat `{"detail": "<field> is required in request body"}`, a body
    shape change clients that parse `detail.error` would notice."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        _, identity = _server(scenario, peer, "dec1miss")
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        sent: Final = {name: (identity if value == "PLACEHOLDER" else value) for name, value in body.items()}
        response: Final = gateway.client.post("/mcp-rest/tools/call", headers={"x-litellm-api-key": key}, json=sent)
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"error": "missing_parameter", "message": message}


def test_rest_aggregate_listing_reports_each_servers_outcome_by_its_prefix(gateway: Gateway) -> None:
    """A key granted two healthy servers lists `GET /mcp-rest/tools/list`. The body must carry
    `error: null`, `message: "Successfully retrieved tools"` and `server_outcomes` keyed by each server's
    alias with value `{"status": "ok", "tool_count": 3}` (the plain peer has `add`, `multiply`, `fail`).
    The Rust host keys outcomes by the same prefix (`catalog.rs`)."""
    with mcp_peer() as peer_a, mcp_peer() as peer_b, gateway.scenario() as scenario:
        alias_a, a = _server(scenario, peer_a, "dec1outa")
        alias_b, b = _server(scenario, peer_b, "dec1outb")
        key: Final = scenario.key(object_permission={"mcp_servers": [a, b]})
        response: Final = gateway.client.get("/mcp-rest/tools/list", headers={"x-litellm-api-key": key})
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["error"] is None and body["message"] == "Successfully retrieved tools", body
    assert body["server_outcomes"][alias_a] == {"status": "ok", "tool_count": 3}, body
    assert body["server_outcomes"][alias_b] == {"status": "ok", "tool_count": 3}, body
    assert {tool["mcp_info"]["server_id"] for tool in body["tools"]} >= {a, b}, body


def test_rest_aggregate_listing_keeps_healthy_servers_when_one_is_down(gateway: Gateway) -> None:
    """One server's peer is stopped after registration; the other is healthy. The aggregate REST listing
    must still be HTTP 200 with the healthy server's tools, `error: null` (only an all-failed listing is a
    `partial_failure`), and the dead server's outcome status must be one of `unreachable`, `timeout`,
    `upstream_error` or `internal`."""
    with mcp_peer() as healthy, gateway.scenario() as scenario:
        alias_ok, ok = _server(scenario, healthy, "dec1up")
        with mcp_peer() as doomed:
            alias_dead, dead = _server(scenario, doomed, "dec1down")
        key: Final = scenario.key(object_permission={"mcp_servers": [ok, dead]})
        body: Final = gateway.client.get("/mcp-rest/tools/list", headers={"x-litellm-api-key": key}).json()
        assert body["error"] is None, body
        assert any(tool["mcp_info"]["server_id"] == ok for tool in body["tools"]), body
        assert body["server_outcomes"][alias_ok]["status"] == "ok", body
        assert body["server_outcomes"][alias_dead]["status"] in ("unreachable", "timeout", "upstream_error", "internal"), body
        assert listed_tools(gateway, key, ok), body
