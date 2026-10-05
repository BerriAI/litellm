import json
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    INITIALIZE,
    PEER_KINDS,
    EntryPoint,
    McpCaller,
    Outcome,
    PeerKind,
    _outcome_from_rpc,
    mcp_peer,
    official_client_outcomes,
    peer_of,
    register_mcp,
    tool_calls,
)
from integration._support.mcp_grants import create_toolset

ADD: Final = {"http": "add", "sse": "add", "stdio": "add", "openapi": "getpet"}
ARGUMENTS: Final = {"add": {"a": 3, "b": 4}, "getpet": {"petId": "7"}}
EXPECTED: Final = {"add": "7", "getpet": json.dumps({"id": "7", "name": "integration-pet"})}


def _streamable_rpc(
    gateway: Gateway, path: str, key: str, method: str, params: Mapping[str, object]
) -> httpx.Response:
    return gateway.client.post(
        path,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json, text/event-stream"},
    )


def _streamable_tools(gateway: Gateway, path: str, key: str) -> tuple[httpx.Response, Outcome]:
    response: Final = _streamable_rpc(gateway, path, key, "tools/list", {})
    return response, _outcome_from_rpc(response)


def _assert_upstream_call(
    call: Mapping[str, object], name: str, arguments: Mapping[str, object]
) -> None:
    body: Final = call["body"]
    assert isinstance(body, Mapping), call
    params: Final = body["params"]
    assert isinstance(params, Mapping), call
    assert set(params) == {"name", "arguments", "_meta"}, call
    assert params["name"] == name, call
    assert params["arguments"] == arguments, call
    metadata: Final = params["_meta"]
    assert isinstance(metadata, Mapping), call
    assert set(metadata) == {"progressToken"}, call


def _open_aliases() -> frozenset[str]:
    rows: Final = read_rows('SELECT alias FROM "LiteLLM_MCPServerTable" WHERE allow_all_keys', ())
    return frozenset(str(row["alias"]) for row in rows)


def _without_foreign_open_servers(tools: tuple[str, ...], open_aliases: frozenset[str]) -> set[str]:
    prefixes: Final = tuple(f"{alias}-" for alias in open_aliases)
    return {tool for tool in tools if not tool.startswith(prefixes)}


def _peer_saw_call(peer_kind: PeerKind, observed: tuple[dict[str, object], ...], tool: str) -> bool:
    if peer_kind == "openapi":
        return any(item.get("path") == "/pets/7" and item.get("method") == "GET" for item in observed)
    calls: Final = tool_calls(observed)
    return len(calls) == 1 and calls[0]["body"]["params"]["name"] == tool


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("peer_kind", PEER_KINDS)
def test_every_entry_point_lists_and_calls_every_peer_transport(
    gateway: Gateway, peer_kind: PeerKind, entry: EntryPoint
) -> None:
    with peer_of(peer_kind) as peer, gateway.scenario() as scenario:
        alias: Final = "tr" + uuid.uuid4().hex[:10]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        tool: Final = ADD[peer_kind]
        listed: Final = caller.list_tools(identity if entry == "rest" else None)
        assert listed.ok, listed.raw
        prefixed: Final = tool if entry == "rest" else f"{alias}-{tool}"
        assert prefixed in listed.tools, listed.tools
        peer.drain()
        called: Final = caller.call(prefixed, ARGUMENTS[tool], identity if entry == "rest" else None)
        assert called.ok, called.raw
        assert called.text is not None and json.loads(called.text) == json.loads(EXPECTED[tool]), called.raw
        assert _peer_saw_call(peer_kind, peer.drain(), tool)


@pytest.mark.parametrize("peer_kind", ("http", "sse", "stdio"))
def test_rest_and_streamable_http_agree_on_tool_list_and_result(gateway: Gateway, peer_kind: PeerKind) -> None:
    with peer_of(peer_kind) as peer, gateway.scenario() as scenario:
        alias: Final = "agree" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        rest: Final = McpCaller(gateway, key, "rest", alias)
        rpc: Final = McpCaller(gateway, key, "mcp", alias)
        rest_tools: Final = rest.list_tools(identity).tools
        rpc_tools: Final = rpc.list_tools().tools
        assert tuple(f"{alias}-{name}" for name in rest_tools) == rpc_tools, (rest_tools, rpc_tools)
        rest_result: Final = rest.call("multiply", {"a": 6, "b": 7}, identity)
        rpc_result: Final = rpc.call(f"{alias}-multiply", {"a": 6, "b": 7})
        assert rest_result.ok and rpc_result.ok, (rest_result.raw, rpc_result.raw)
        assert rest_result.text == rpc_result.text == "42"
        rest_failure: Final = rest.call("fail", {}, identity)
        rpc_failure: Final = rpc.call(f"{alias}-fail", {})
        assert rest_failure.error is not None and rpc_failure.error is not None, (rest_failure.raw, rpc_failure.raw)
        assert rest_failure.text == rpc_failure.text


@pytest.mark.parametrize(
    ("path_kind", "legacy_sse"),
    (("aggregate", False), ("named", False), ("legacy_sse", True)),
    ids=("official-client-/mcp", "official-client-/{server}/mcp", "official-client-/mcp/sse"),
)
@pytest.mark.parametrize("peer_kind", ("http", "sse"))
def test_official_client_session_lists_and_calls_through_gateway(
    gateway: Gateway, peer_kind: PeerKind, path_kind: str, legacy_sse: bool
) -> None:
    with peer_of(peer_kind) as peer, gateway.scenario() as scenario:
        alias: Final = "sdk" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        path: Final = {"aggregate": "/mcp", "named": f"/{alias}/mcp", "legacy_sse": "/mcp/sse"}[path_kind]
        peer.drain()
        listed, called = official_client_outcomes(
            gateway, key, path, f"{alias}-add", {"a": 20, "b": 22}, legacy_sse=legacy_sse
        )
        assert set(listed.tools) == {f"{alias}-add", f"{alias}-multiply", f"{alias}-fail"}, listed.tools
        assert called.ok and called.text == "42", called
        assert _peer_saw_call(peer_kind, peer.drain(), "add")


def test_standard_per_server_path_scopes_the_session_to_the_named_servers(gateway: Gateway) -> None:
    open_before: Final = _open_aliases()
    with mcp_peer() as peer_a, mcp_peer() as peer_b, mcp_peer() as peer_c, gateway.scenario() as scenario:
        alias_a: Final = "rt1a" + uuid.uuid4().hex[:10]
        alias_b: Final = "rt1b" + uuid.uuid4().hex[:10]
        alias_c: Final = "rt1c" + uuid.uuid4().hex[:10]
        server_a: Final = register_mcp(scenario, peer_a, alias_a)
        server_b: Final = register_mcp(scenario, peer_b, alias_b)
        server_c: Final = register_mcp(scenario, peer_c, alias_c)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_a, server_b, server_c]})
        open_aliases: Final = open_before | _open_aliases()

        first_path: Final = f"/mcp/{alias_a}"
        initialized: Final = _streamable_rpc(gateway, first_path, key, "initialize", dict(INITIALIZE))
        assert initialized.status_code == 200, initialized.text
        listed_response, listed = _streamable_tools(gateway, first_path, key)
        assert listed_response.status_code == 200, listed_response.text
        assert listed.ok, listed.raw
        assert _without_foreign_open_servers(listed.tools, open_aliases) == {
            f"{alias_a}-add",
            f"{alias_a}-multiply",
            f"{alias_a}-fail",
        }, listed.tools
        called_response: Final = _streamable_rpc(
            gateway, first_path, key, "tools/call", {"name": f"{alias_a}-add", "arguments": {"a": 20, "b": 22}}
        )
        called: Final = _outcome_from_rpc(called_response)
        assert called_response.status_code == 200, called_response.text
        assert called.ok and called.text == "42", called.raw
        calls_a: Final = tool_calls(peer_a.drain())
        assert len(calls_a) == 1, calls_a
        _assert_upstream_call(calls_a[0], "add", {"a": 20, "b": 22})
        assert tool_calls(peer_b.drain()) == ()
        assert tool_calls(peer_c.drain()) == ()

        combined_path: Final = f"/mcp/{alias_a},{alias_b}"
        combined_initialized: Final = _streamable_rpc(
            gateway, combined_path, key, "initialize", dict(INITIALIZE)
        )
        assert combined_initialized.status_code == 200, combined_initialized.text
        combined_listed_response, combined_listed = _streamable_tools(gateway, combined_path, key)
        assert combined_listed_response.status_code == 200, combined_listed_response.text
        assert combined_listed.ok, combined_listed.raw
        assert _without_foreign_open_servers(combined_listed.tools, open_aliases) == {
            f"{alias_a}-add",
            f"{alias_a}-multiply",
            f"{alias_a}-fail",
            f"{alias_b}-add",
            f"{alias_b}-multiply",
            f"{alias_b}-fail",
        }, combined_listed.tools
        combined_called_response: Final = _streamable_rpc(
            gateway,
            combined_path,
            key,
            "tools/call",
            {"name": f"{alias_b}-multiply", "arguments": {"a": 6, "b": 7}},
        )
        combined_called: Final = _outcome_from_rpc(combined_called_response)
        assert combined_called_response.status_code == 200, combined_called_response.text
        assert combined_called.ok and combined_called.text == "42", combined_called.raw
        assert tool_calls(peer_a.drain()) == ()
        calls_b: Final = tool_calls(peer_b.drain())
        assert len(calls_b) == 1, calls_b
        _assert_upstream_call(calls_b[0], "multiply", {"a": 6, "b": 7})
        assert tool_calls(peer_c.drain()) == ()


def test_legacy_server_path_resolves_lists_groups_and_toolsets_and_fails_closed_on_unknown_names(
    gateway: Gateway,
) -> None:
    open_before: Final = _open_aliases()
    with mcp_peer() as peer_a, mcp_peer() as peer_b, mcp_peer() as peer_c, gateway.scenario() as scenario:
        alias_a: Final = "rt2a" + uuid.uuid4().hex[:10]
        alias_b: Final = "rt2b" + uuid.uuid4().hex[:10]
        alias_c: Final = "rt2c" + uuid.uuid4().hex[:10]
        group: Final = "rt2g" + uuid.uuid4().hex[:10]
        server_a: Final = register_mcp(scenario, peer_a, alias_a, mcp_access_groups=[group])
        server_b: Final = register_mcp(scenario, peer_b, alias_b, mcp_access_groups=[group])
        server_c: Final = register_mcp(scenario, peer_c, alias_c)
        toolset_name: Final = "rt2t" + uuid.uuid4().hex[:10]
        toolset_id: Final = create_toolset(scenario, ((server_a, "add"),), toolset_name=toolset_name)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_a, server_b, server_c]})
        toolset_key: Final = scenario.key(object_permission={"mcp_toolsets": [toolset_id]})
        open_aliases: Final = open_before | _open_aliases()

        direct_path: Final = f"/{alias_a},{alias_b}/mcp"
        direct_initialized: Final = _streamable_rpc(gateway, direct_path, key, "initialize", dict(INITIALIZE))
        assert direct_initialized.status_code == 200, direct_initialized.text
        direct_response, direct_listed = _streamable_tools(gateway, direct_path, key)
        assert direct_response.status_code == 200, direct_response.text
        assert direct_listed.ok, direct_listed.raw
        assert _without_foreign_open_servers(direct_listed.tools, open_aliases) == {
            f"{alias_a}-add",
            f"{alias_a}-multiply",
            f"{alias_a}-fail",
            f"{alias_b}-add",
            f"{alias_b}-multiply",
            f"{alias_b}-fail",
        }, direct_listed.tools
        direct_called_response: Final = _streamable_rpc(
            gateway, direct_path, key, "tools/call", {"name": f"{alias_a}-add", "arguments": {"a": 20, "b": 22}}
        )
        direct_called: Final = _outcome_from_rpc(direct_called_response)
        assert direct_called_response.status_code == 200, direct_called_response.text
        assert direct_called.ok and direct_called.text == "42", direct_called.raw
        direct_calls_a: Final = tool_calls(peer_a.drain())
        assert len(direct_calls_a) == 1, direct_calls_a
        _assert_upstream_call(direct_calls_a[0], "add", {"a": 20, "b": 22})
        assert tool_calls(peer_b.drain()) == ()
        assert tool_calls(peer_c.drain()) == ()

        mixed_path: Final = f"/{alias_a},%20{alias_a}%20,nope{uuid.uuid4().hex}/mcp"
        mixed_initialized: Final = _streamable_rpc(gateway, mixed_path, key, "initialize", dict(INITIALIZE))
        assert mixed_initialized.status_code == 200, mixed_initialized.text
        mixed_response, mixed_listed = _streamable_tools(gateway, mixed_path, key)
        assert mixed_response.status_code == 200, mixed_response.text
        assert mixed_listed.ok, mixed_listed.raw
        assert _without_foreign_open_servers(mixed_listed.tools, open_aliases) == {
            f"{alias_a}-add",
            f"{alias_a}-multiply",
            f"{alias_a}-fail",
        }, mixed_listed.tools
        mixed_called_response: Final = _streamable_rpc(
            gateway, mixed_path, key, "tools/call", {"name": f"{alias_a}-multiply", "arguments": {"a": 6, "b": 7}}
        )
        mixed_called: Final = _outcome_from_rpc(mixed_called_response)
        assert mixed_called_response.status_code == 200, mixed_called_response.text
        assert mixed_called.ok and mixed_called.text == "42", mixed_called.raw
        mixed_calls_a: Final = tool_calls(peer_a.drain())
        assert len(mixed_calls_a) == 1, mixed_calls_a
        _assert_upstream_call(mixed_calls_a[0], "multiply", {"a": 6, "b": 7})
        assert tool_calls(peer_b.drain()) == ()
        assert tool_calls(peer_c.drain()) == ()

        unknown_path: Final = f"/nope1{uuid.uuid4().hex},nope2{uuid.uuid4().hex}/mcp"
        assert peer_a.drain() == ()
        assert peer_b.drain() == ()
        assert peer_c.drain() == ()
        unknown_initialized: Final = _streamable_rpc(gateway, unknown_path, key, "initialize", dict(INITIALIZE))
        assert unknown_initialized.status_code == 404, unknown_initialized.text
        unknown_listed: Final = _streamable_rpc(gateway, unknown_path, key, "tools/list", {})
        assert unknown_listed.status_code == 404, unknown_listed.text
        assert peer_a.drain() == ()
        assert peer_b.drain() == ()
        assert peer_c.drain() == ()

        group_path: Final = f"/{group}/mcp"
        group_initialized: Final = _streamable_rpc(gateway, group_path, key, "initialize", dict(INITIALIZE))
        assert group_initialized.status_code == 200, group_initialized.text
        group_response, group_listed = _streamable_tools(gateway, group_path, key)
        assert group_response.status_code == 200, group_response.text
        assert group_listed.ok, group_listed.raw
        assert _without_foreign_open_servers(group_listed.tools, open_aliases) == {
            f"{alias_a}-add",
            f"{alias_a}-multiply",
            f"{alias_a}-fail",
            f"{alias_b}-add",
            f"{alias_b}-multiply",
            f"{alias_b}-fail",
        }, group_listed.tools
        group_called_response: Final = _streamable_rpc(
            gateway, group_path, key, "tools/call", {"name": f"{alias_b}-multiply", "arguments": {"a": 6, "b": 7}}
        )
        group_called: Final = _outcome_from_rpc(group_called_response)
        assert group_called_response.status_code == 200, group_called_response.text
        assert group_called.ok and group_called.text == "42", group_called.raw
        assert tool_calls(peer_a.drain()) == ()
        group_calls_b: Final = tool_calls(peer_b.drain())
        assert len(group_calls_b) == 1, group_calls_b
        _assert_upstream_call(group_calls_b[0], "multiply", {"a": 6, "b": 7})
        assert tool_calls(peer_c.drain()) == ()

        toolset_path: Final = f"/{toolset_name}/mcp"
        toolset_initialized: Final = _streamable_rpc(
            gateway, toolset_path, toolset_key, "initialize", dict(INITIALIZE)
        )
        assert toolset_initialized.status_code == 200, toolset_initialized.text
        toolset_response, toolset_listed = _streamable_tools(gateway, toolset_path, toolset_key)
        assert toolset_response.status_code == 200, toolset_response.text
        assert toolset_listed.ok, toolset_listed.raw
        assert _without_foreign_open_servers(toolset_listed.tools, open_aliases) == {f"{alias_a}-add"}, (
            toolset_listed.tools
        )
        toolset_called_response: Final = _streamable_rpc(
            gateway,
            toolset_path,
            toolset_key,
            "tools/call",
            {"name": f"{alias_a}-add", "arguments": {"a": 20, "b": 22}},
        )
        toolset_called: Final = _outcome_from_rpc(toolset_called_response)
        assert toolset_called_response.status_code == 200, toolset_called_response.text
        assert toolset_called.ok and toolset_called.text == "42", toolset_called.raw
        toolset_calls_a: Final = tool_calls(peer_a.drain())
        assert len(toolset_calls_a) == 1, toolset_calls_a
        _assert_upstream_call(toolset_calls_a[0], "add", {"a": 20, "b": 22})
        assert tool_calls(peer_b.drain()) == ()
        assert tool_calls(peer_c.drain()) == ()

@pytest.mark.parametrize("peer_kind", ("http", "sse", "stdio"))
def test_prompts_resources_and_templates_are_proxied_from_rich_peer(gateway: Gateway, peer_kind: PeerKind) -> None:
    with peer_of(peer_kind, rich=True) as peer, gateway.scenario() as scenario:
        alias: Final = "rich" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "server_mcp", alias)
        prompts: Final = caller.rpc("prompts/list").text
        assert f"{alias}-greeting" in prompts, prompts
        prompt: Final = caller.rpc("prompts/get", {"name": f"{alias}-greeting", "arguments": {"name": "Ada"}}).text
        assert "Hello, Ada" in prompt, prompt
        resources: Final = caller.rpc("resources/list").text
        assert "status://ready" in resources and f"{alias}-status" in resources, resources
        read: Final = caller.rpc("resources/read", {"uri": "status://ready"}).text
        assert '"text":"ready"' in read.replace(" ", ""), read
        templates: Final = caller.rpc("resources/templates/list").text
        assert "greeting://{name}" in templates, templates
        templated: Final = caller.rpc("resources/read", {"uri": "greeting://Bob"}).text
        assert "Hello, Bob" in templated, templated
        methods: Final = {item["body"].get("method") for item in peer.drain() if isinstance(item.get("body"), dict)}
        assert {
            "prompts/list",
            "prompts/get",
            "resources/list",
            "resources/read",
            "resources/templates/list",
        } <= methods


@pytest.mark.parametrize("peer_kind", ("http", "sse", "stdio"))
def test_progress_notifications_do_not_break_result_and_slow_tool_completes(
    gateway: Gateway, peer_kind: PeerKind
) -> None:
    with peer_of(peer_kind, rich=True) as peer, gateway.scenario() as scenario:
        alias: Final = "prog" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        progressed: Final = caller.call(f"{alias}-progress", {"steps": 3})
        assert progressed.ok and progressed.text == "3 steps", progressed.raw
        slow: Final = caller.call(f"{alias}-slow", {"seconds": 1.5})
        assert slow.ok and slow.text == "slept", slow.raw


@pytest.mark.parametrize("tool", ("sample", "elicit"))
@pytest.mark.parametrize("entry", ("mcp", "rest"))
def test_server_initiated_sampling_and_elicitation_surface_as_errors_not_success(
    gateway: Gateway, entry: EntryPoint, tool: str
) -> None:
    with mcp_peer(rich=True) as peer, gateway.scenario() as scenario:
        alias: Final = "back" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        name: Final = tool if entry == "rest" else f"{alias}-{tool}"
        peer.drain()
        outcome: Final = caller.call(name, {"prompt": "hi"} if tool == "sample" else {"question": "ok?"}, identity)
        assert outcome.error is not None, outcome.raw
        assert outcome.text is None or not outcome.text.startswith(("sampled:", "elicited:")), outcome.raw
        assert len(tool_calls(peer.drain())) == 1


@pytest.mark.parametrize("downstream", ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"))
@pytest.mark.parametrize("upstream", ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"))
@pytest.mark.parametrize("peer_kind", ("http", "sse", "stdio"))
@pytest.mark.parametrize("ingress", ("http", "sse"))
def test_pinned_revision_pairs_list_and_call_through_gateway(
    gateway: Gateway, downstream: str, upstream: str, peer_kind: PeerKind, ingress: str
) -> None:
    import asyncio

    from mcp.types import CallToolRequestParams

    from litellm.experimental_mcp_client.client import MCPClient
    from litellm.types.mcp import MCPTransport

    with peer_of(peer_kind) as peer, gateway.scenario() as scenario:
        alias: Final = "versions" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, mcp_info={"protocol_version": upstream})
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        endpoint: Final = str(gateway.client.base_url).rstrip("/") + ("/mcp/sse" if ingress == "sse" else "/mcp")
        client: Final = MCPClient(
            server_url=endpoint, transport_type=MCPTransport(ingress), protocol_version=downstream,
            extra_headers={"Authorization": f"Bearer {key}", "x-mcp-servers": identity}, timeout=15,
        )

        async def exercise() -> None:
            tools: Final = await client.list_tools(raise_on_error=True)
            assert f"{alias}-add" in tuple(tool.name for tool in tools)
            result: Final = await client.call_tool(CallToolRequestParams(name=f"{alias}-add", arguments={"a": 3, "b": 4}))
            assert result.is_error is False
            assert result.content[0].text == "7"

        peer.drain()
        asyncio.run(exercise())
        observed: Final = peer.drain()
        negotiations: Final = tuple(item["body"] for item in observed if item["body"].get("method") == "initialize")
        assert negotiations, "The operation must reach the upstream negotiation"
        assert all(request["params"]["protocolVersion"] == upstream for request in negotiations), negotiations
        assert len(tool_calls(observed)) == 1


@pytest.mark.parametrize("downstream", ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"))
@pytest.mark.parametrize("peer_kind", ("http", "stdio"))
def test_legacy_gateway_calls_modern_upstream_without_initialize(
    gateway: Gateway,
    downstream: str,
    peer_kind: PeerKind,
) -> None:
    import asyncio

    from mcp.types import CallToolRequestParams

    from litellm.experimental_mcp_client.client import MCPClient
    from litellm.types.mcp import MCPTransport

    with peer_of(peer_kind) as peer, gateway.scenario() as scenario:
        alias: Final = "modern" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, mcp_info={"protocol_version": "2026-07-28"})
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        client: Final = MCPClient(
            server_url=str(gateway.client.base_url).rstrip("/") + "/mcp",
            transport_type=MCPTransport.http,
            protocol_version=downstream,
            extra_headers={"Authorization": f"Bearer {key}"},
        )

        async def exercise() -> None:
            listed: Final = await client.list_tools(raise_on_error=True)
            assert f"{alias}-add" in tuple(tool.name for tool in listed)
            called: Final = await client.call_tool(
                CallToolRequestParams(name=f"{alias}-add", arguments={"a": 2, "b": 3}),
                raise_on_error=True,
            )
            assert called.is_error is False
            assert called.content[0].text == "5"

        peer.drain()
        asyncio.run(exercise())
        observed: Final = peer.drain()
        assert len(tool_calls(observed)) == 1
        assert all(row["body"].get("method") not in ("initialize", "notifications/initialized") for row in observed)
        requests: Final = tuple(row for row in observed if "id" in row["body"])
        assert requests
        for row in requests:
            metadata: Final = row["body"]["params"]["_meta"]
            assert metadata["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
            assert "io.modelcontextprotocol/clientCapabilities" in metadata
            assert b"mcp-session-id" not in row.get("headers", {})
