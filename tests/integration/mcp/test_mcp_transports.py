import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.mcp import (
    ENTRY_POINTS,
    PEER_KINDS,
    EntryPoint,
    McpCaller,
    PeerKind,
    mcp_peer,
    official_client_outcomes,
    peer_of,
    register_mcp,
    tool_calls,
)

ADD: Final = {"http": "add", "sse": "add", "stdio": "add", "openapi": "getpet"}
ARGUMENTS: Final = {"add": {"a": 3, "b": 4}, "getpet": {"petId": "7"}}
EXPECTED: Final = {"add": "7", "getpet": json.dumps({"id": "7", "name": "integration-pet"})}


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
