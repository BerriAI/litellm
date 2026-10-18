import json
import queue
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.mcp import McpPeer, call_tool, register_mcp, tool_names
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter


@pytest.mark.covers("other.mcp.errors.protocol_and_malformed_results_cannot_be_empty_success")
def test_jsonrpc_error_and_malformed_tool_result_remain_errors(gateway: Gateway) -> None:
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
            return Reply(
                body=json.dumps({**base, "error": {"code": -32042, "message": "synthetic JSON-RPC error"}}).encode()
            )
        result: Final = (
            {"content": "synthetic malformed content"}
            if name == "multiply"
            else {"content": [{"type": "text", "text": "8"}], "isError": False}
        )
        return Reply(body=json.dumps({**base, "result": result}).encode())

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


@pytest.mark.covers("other.mcp.transport.large_sse_tool_result_reaches_caller")
def test_tool_result_larger_than_one_mib_in_a_single_sse_event_reaches_the_caller(gateway: Gateway) -> None:
    payload: Final = "x" * (3 * 1024 * 1024)
    request_json: Final = TypeAdapter(dict[str, JsonValue])

    def provider(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=405)
        body: Final = request_json.validate_json(request.body)
        if "id" not in body:
            return Reply(status=202)
        method: Final = body["method"]
        if method == "initialize":
            params: Final = request_json.validate_python(body["params"])
            return Reply(
                body=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "protocolVersion": params["protocolVersion"],
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "large-result", "version": "1"},
                        },
                    }
                ).encode()
            )
        if method == "tools/list":
            return Reply(
                body=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "tools": [{"name": "add", "inputSchema": {"type": "object"}}],
                        },
                    }
                ).encode()
            )
        assert method == "tools/call"
        response: Final = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": payload}], "isError": False},
            }
        )
        return Reply(content_type="text/event-stream", body=f"event: message\ndata: {response}\n\n".encode())

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        peer: Final = McpPeer(wire.url + "/mcp", queue.Queue())
        identity: Final = register_mcp(scenario, peer, "bigsse" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        names: Final = tool_names(gateway, key, identity)
        response: Final = call_tool(gateway, key, identity, names["add"], {})
        assert response.status_code == 200, response.text[:500]
        body: Final = response.json()
        assert body["isError"] is False, response.text[:500]
        assert body["content"][0]["text"] == payload
