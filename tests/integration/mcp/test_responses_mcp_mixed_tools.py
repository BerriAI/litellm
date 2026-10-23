import json
import uuid
from typing import Final

from integration._support.client import Gateway
from integration._support.mcp import mcp_peer, register_mcp, tool_calls
from integration._support.wire import Reply, Request, wire_server

_FUNCTION_TOOL: Final = {
    "type": "function",
    "name": "lookup_weather",
    "description": "Look up the forecast for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def test_responses_with_gateway_mcp_and_caller_function_tool_hands_both_to_model_and_returns_the_function_call(
    gateway: Gateway,
) -> None:
    alias: Final = "mix" + uuid.uuid4().hex[:8]
    upstream_tools: list[tuple[str, ...]] = []

    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.target.endswith("/responses"), request.target
        body: Final = json.loads(request.body)
        upstream_tools.append(tuple(str(tool.get("name")) for tool in body.get("tools", ())))
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_mixed",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_weather",
                            "call_id": "call_weather",
                            "name": "lookup_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    with mcp_peer() as peer, wire_server(respond) as wire, gateway.scenario() as scenario:
        server_id: Final = register_mcp(scenario, peer, alias)
        model: Final = scenario.model(model="openai/responses/gpt-4o-mini", api_base=wire.url + "/v1")
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]})
        peer.drain()
        response: Final = gateway.client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "input": "what is the weather in Paris",
                "tools": [
                    {
                        "type": "mcp",
                        "server_url": "litellm_proxy",
                        "server_label": "litellm",
                        "require_approval": "never",
                    },
                    _FUNCTION_TOOL,
                ],
            },
            timeout=90,
        )
        assert response.status_code == 200, response.text
        assert upstream_tools, "model was never called"
        assert all("lookup_weather" in names and f"{alias}-add" in names for names in upstream_tools), upstream_tools
        calls: Final = [item for item in response.json()["output"] if item.get("type") == "function_call"]
        assert [call["name"] for call in calls] == ["lookup_weather"], response.text
        assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
        assert tool_calls(peer.drain()) == (), "a caller-owned function call must never reach the MCP peer"
