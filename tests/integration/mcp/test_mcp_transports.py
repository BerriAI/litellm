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


@pytest.mark.parametrize("guardrail_action", ["off", "MASK", "BLOCK"])
@pytest.mark.parametrize("protocol", ["2025-11-25", "2026-07-28"])
def test_sdk_progress_and_cancellation_reach_upstream_without_losing_session(
    tmp_path, protocol: str, guardrail_action: str
) -> None:
    import asyncio
    import os
    from pathlib import Path
    from threading import Event
    from contextlib import asynccontextmanager

    import httpx
    import httpx2
    import yaml
    from mcp import ClientSession, MCPError
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import DiscoverResult
    from sse_starlette.sse import AppStatus
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from integration._support.asgi import asgi_server
    from integration._support.mcp import _drain_sse_streams
    from integration._support.process import owned_proxy

    upstream: Final = MCPServer("progress-cancellation-peer")
    started: Final = Event()
    stopped: Final = Event()
    completed: Final = Event()

    @upstream.tool()
    async def progress(ctx: Context) -> str:
        await ctx.report_progress(1, 2, "halfway")
        return "done"

    @upstream.tool()
    async def final_text() -> str:
        return "halfway"

    @upstream.tool()
    async def slow() -> str:
        started.set()
        try:
            await asyncio.sleep(20)
            completed.set()
            return "completed"
        finally:
            stopped.set()

    async def exercise(gateway: Gateway) -> None:
        observed: Final[list[tuple[float, float | None, str | None]]] = []

        async def receive_progress(value: float, total: float | None, message: str | None) -> None:
            observed.append((value, total, message))

        async with httpx2.AsyncClient(headers={"Authorization": "Bearer " + gateway.key}) as client:
            async with streamable_http_client(
                str(gateway.client.base_url).rstrip("/") + "/mcp/", http_client=client
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    if protocol == "2026-07-28":
                        session.adopt(DiscoverResult.model_validate(await session.send_discover(protocol)))
                    else:
                        await session.initialize()
                    listed = await session.list_tools()
                    assert {"peer-progress", "peer-slow"} <= {tool.name for tool in listed.tools}
                    if guardrail_action != "off":
                        try:
                            checked = await session.call_tool("peer-final_text", {})
                        except MCPError:
                            assert guardrail_action == "BLOCK"
                        else:
                            if guardrail_action == "BLOCK":
                                assert checked.is_error
                            else:
                                assert not checked.is_error
                                assert "halfway" not in str(checked.content)
                    result = await session.call_tool("peer-progress", {}, progress_callback=receive_progress)
                    assert not result.is_error
                    assert len(observed) == 1
                    assert observed[0][:2] == (1.0, 2.0)
                    if guardrail_action == "BLOCK":
                        assert observed[0][2] is None
                    elif guardrail_action == "MASK":
                        assert observed[0][2] and "halfway" not in observed[0][2]
                    else:
                        assert observed[0][2] == "halfway"
                    call = asyncio.create_task(session.call_tool("peer-slow", {}))
                    try:
                        assert await asyncio.to_thread(started.wait, 5), "Upstream tool did not start"
                        call.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await call
                        assert await asyncio.to_thread(stopped.wait, 2), "Cancelled upstream work kept running"
                        assert not completed.is_set()
                        followup = await session.call_tool("peer-progress", {})
                        assert not followup.is_error, "Cancellation must leave later calls usable"
                    finally:
                        call.cancel()
                        await asyncio.gather(call, return_exceptions=True)

    app: Final = upstream.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )

    @asynccontextmanager
    async def lifespan(_: Starlette):
        AppStatus.should_exit = False
        async with app.router.lifespan_context(app):
            yield
        _drain_sse_streams()
        watchers = tuple(task for task in asyncio.all_tasks() if "_shutdown_watcher" in repr(task.get_coro()))
        await asyncio.wait_for(asyncio.gather(*watchers), 2)

    hosted: Final = Starlette(routes=[Mount("/", app=app)], lifespan=lifespan)
    with asgi_server(hosted) as upstream_url, httpx.Client() as client:
        config: Final = tmp_path / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [],
                    "guardrails": []
                    if guardrail_action == "off"
                    else [
                        {
                            "guardrail_name": "progress-policy",
                            "litellm_params": {
                                "guardrail": "litellm_content_filter",
                                "mode": "post_mcp_call",
                                "default_on": True,
                                "blocked_words": [{"keyword": "halfway", "action": guardrail_action}],
                            },
                        }
                    ],
                    "mcp_servers": {
                        "peer": {"url": upstream_url + "/mcp", "transport": "http", "protocol_version": protocol}
                    },
                    "general_settings": {
                        "master_key": "sk-progress-cancellation-test",
                        "store_model_in_db": False,
                        "mcp_advertised_versions": ["2025-11-25", "2026-07-28"],
                    },
                }
            )
        )
        source: Final = Path(__file__).resolve().parents[3]
        environment: Final = {
            "PYTHONPATH": os.pathsep.join((str(source), str(source / "tests"))),
            "STORE_MODEL_IN_DB": "False",
            "DISABLE_SCHEMA_UPDATE": "true",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY": "true",
        }
        seed: Final = Gateway(client, "sk-progress-cancellation-test", upstream_url)
        with owned_proxy(
            seed,
            tmp_path / "gateway",
            environment,
            config=config,
            database_setup=(),
            remove_environment=(
                "DATABASE_URL",
                "DATABASE_URL_READ_REPLICA",
                "DIRECT_URL",
                "INTEGRATION_PROXY_DATABASE_URL",
                "INTEGRATION_PROXY_READ_REPLICA_URL",
            ),
        ) as gateway:
            asyncio.run(exercise(gateway))
