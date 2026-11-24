import json
import queue
import uuid
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.mcp import McpCaller, McpPeer, call_tool, register_mcp, tool_names
from integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.shares_worker_state


URL_ELICITATION_ERROR: Final = {
    "code": -32042,
    "message": "This request requires more information.",
    "data": {
        "elicitations": [
            {
                "mode": "url",
                "elicitationId": "el-1",
                "url": "https://consent.example.invalid/oauth2/authorize?request_uri=FAKE",
                "message": "Authorize to continue.",
            }
        ]
    },
}


def _protocol_peer(fail_error: Mapping[str, object]) -> Callable[[Request], Reply]:
    def provider(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=405)
        body: Final = json.loads(request.body)
        method: Final = body["method"]
        if "id" not in body:
            return Reply(status=202)
        base: Final = {"jsonrpc": "2.0", "id": body["id"]}
        if method == "initialize":
            return Reply(
                body=json.dumps(
                    {
                        **base,
                        "result": {
                            "protocolVersion": body["params"]["protocolVersion"],
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "synthetic-protocol-peer", "version": "1"},
                        },
                    }
                ).encode()
            )
        if method == "tools/list":
            return Reply(
                body=json.dumps(
                    {
                        **base,
                        "result": {
                            "tools": [
                                {"name": name, "inputSchema": {"type": "object"}}
                                for name in ("add", "multiply", "fail")
                            ]
                        },
                    }
                ).encode()
            )
        assert method == "tools/call"
        name: Final = body["params"]["name"]
        if name == "fail":
            return Reply(body=json.dumps({**base, "error": fail_error}).encode())
        result: Final = (
            {"content": "synthetic malformed content"}
            if name == "multiply"
            else {"content": [{"type": "text", "text": "8"}], "isError": False}
        )
        return Reply(body=json.dumps({**base, "result": result}).encode())

    return provider


@pytest.mark.covers("other.mcp.errors.protocol_and_malformed_results_cannot_be_empty_success")
def test_jsonrpc_error_and_malformed_tool_result_remain_errors(gateway: Gateway) -> None:
    provider: Final = _protocol_peer({"code": -32042, "message": "synthetic JSON-RPC error"})
    with wire_server(provider) as wire, gateway.scenario() as scenario:
        identity: Final = register_mcp(
            scenario, McpPeer(wire.url + "/mcp", queue.Queue()), "integration" + uuid.uuid4().hex
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        names: Final = tool_names(gateway, key, identity)
        for name, expected in (("fail", "synthetic JSON-RPC error"), ("multiply", "validation")):
            wire.drain()
            response: Final = call_tool(gateway, key, identity, names[name], {})
            assert response.status_code == 200 and response.json()["isError"] is True, response.text
            assert expected.lower() in response.json()["content"][0]["text"].lower(), response.text
            assert (
                len(
                    tuple(
                        item
                        for item in wire.drain()
                        if item.method == "POST" and json.loads(item.body).get("method") == "tools/call"
                    )
                )
                == 1
            )
            control: Final = call_tool(gateway, key, identity, names["add"], {"a": 3, "b": 5})
            assert control.status_code == 200 and control.json()["isError"] is False, control.text
            assert control.json()["content"][0]["text"] == "8"


def test_upstream_url_elicitation_required_reaches_mcp_client_as_jsonrpc_error(gateway: Gateway) -> None:
    with wire_server(_protocol_peer(URL_ELICITATION_ERROR)) as upstream, gateway.scenario() as scenario:
        alias: Final = "elicit" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, McpPeer(upstream.url + "/mcp", queue.Queue()), alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp")
        assert caller.initialize().ok
        elicitation: Final = caller.call(f"{alias}-fail", {})
        assert elicitation.status == 200 and elicitation.text is None and elicitation.error, elicitation.raw
        assert json.loads(elicitation.error) == URL_ELICITATION_ERROR, elicitation.raw
        control: Final = caller.call(f"{alias}-add", {"a": 3, "b": 5})
        assert control.ok and control.text == "8", control.raw


@pytest.mark.parametrize("ingress", ("http", "sse"))
def test_configured_revision_blocks_unadvertised_handshake_and_keeps_allowed_control(
    gateway: Gateway, tmp_path, ingress: str
) -> None:
    import asyncio
    from pathlib import Path

    import yaml
    from integration._support.mcp import mcp_peer
    from integration._support.process import owned_proxy
    from mcp import MCPError
    from mcp.types import CallToolRequestParams

    from litellm.mcp_client.client import MCPClient
    from litellm.types.mcp import MCPTransport

    with mcp_peer() as upstream, gateway.scenario() as scenario:
        alias: Final = "restricted" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, upstream, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"]["mcp_advertised_versions"] = ["2024-11-05"]
        config_path: Final = tmp_path / "restricted.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {"DISABLE_SCHEMA_UPDATE": "true"}, config=config_path) as restricted:
            endpoint: Final = str(restricted.client.base_url).rstrip("/") + ("/mcp/sse" if ingress == "sse" else "/mcp")
            headers: Final = {"Authorization": f"Bearer {key}", "x-mcp-servers": identity}
            denied: Final = MCPClient(server_url=endpoint, transport_type=MCPTransport(ingress), protocol_version="2025-11-25", extra_headers=headers)
            allowed: Final = MCPClient(server_url=endpoint, transport_type=MCPTransport(ingress), protocol_version="2024-11-05", extra_headers=headers)

            async def exercise() -> None:
                with pytest.raises(MCPError, match="Unsupported MCP protocol version"):
                    await denied.list_tools(raise_on_error=True)
                assert f"{alias}-add" in tuple(tool.name for tool in await allowed.list_tools(raise_on_error=True))
                result: Final = await allowed.call_tool(CallToolRequestParams(name=f"{alias}-add", arguments={"a": 2, "b": 5}))
                assert result.is_error is False and result.content[0].text == "7"

            asyncio.run(exercise())


def test_virtual_tool_call_url_elicitation_is_a_rest_error_result_and_an_mcp_jsonrpc_error(gateway: Gateway) -> None:
    with wire_server(_protocol_peer(URL_ELICITATION_ERROR)) as upstream, gateway.scenario() as scenario:
        alias: Final = "elicit" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, McpPeer(upstream.url + "/mcp", queue.Queue()), alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity], "mcp_tool_search_enabled": True})
        caller: Final = McpCaller(gateway, key, "rest")
        elicitation: Final = caller.call("mcp_tool_call", {"tool_name": f"{alias}-fail", "arguments": {}})
        assert elicitation.status == 200, elicitation.raw
        assert elicitation.error == f"MCPError: {URL_ELICITATION_ERROR['message']}", elicitation.raw
        control: Final = caller.call("mcp_tool_call", {"tool_name": f"{alias}-add", "arguments": {"a": 3, "b": 5}})
        assert control.ok and control.text == "8", control.raw
        mcp_caller: Final = McpCaller(gateway, key, "mcp")
        assert mcp_caller.initialize().ok
        relayed: Final = mcp_caller.call("mcp_tool_call", {"tool_name": f"{alias}-fail", "arguments": {}})
        assert relayed.status == 200 and relayed.text is None and relayed.error, relayed.raw
        assert json.loads(relayed.error) == URL_ELICITATION_ERROR, relayed.raw
