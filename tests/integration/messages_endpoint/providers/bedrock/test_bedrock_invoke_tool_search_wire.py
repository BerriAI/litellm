import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL_ID: Final = "us.anthropic.claude-sonnet-5"
TOKEN: Final = "synthetic-bedrock-bearer"
TOOL_SEARCH_TOOL: Final = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
DEFERRED_TOOL: Final = {
    "name": "get_weather",
    "description": "Weather lookup",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    "defer_loading": True,
}
RESPONSE: Final = json.dumps(
    {
        "id": "msg_tool_search_control",
        "type": "message",
        "role": "assistant",
        "model": MODEL_ID,
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_control",
                "name": "tool_search_tool_regex",
                "input": {"pattern": "weather"},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_control",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
                },
            },
            {"type": "text", "text": "tool search wire control"},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 6},
    }
).encode()


def tool_search_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == f"/model/{MODEL_ID}/invoke", request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body["anthropic_beta"] == ["tool-search-tool-2025-10-19"], body
    assert body["messages"] == [{"role": "user", "content": "find the weather tool"}]
    assert body["tools"] == [TOOL_SEARCH_TOOL, DEFERRED_TOOL], body["tools"]
    assert body["max_tokens"] == 64
    assert "model" not in body
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.bedrock_invoke.tool_search_gen5_claude_sends_bedrock_beta_and_reports_support")
def test_gen5_claude_bedrock_invoke_messages_tool_search_sends_bedrock_beta_field(gateway: Gateway) -> None:
    with wire_server(tool_search_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/invoke/{MODEL_ID}",
            api_key=TOKEN,
            aws_region_name="us-east-1",
            api_base=wire.url,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "find the weather tool"}],
                "tools": [TOOL_SEARCH_TOOL, DEFERRED_TOOL],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["content"][2] == {"type": "text", "text": "tool search wire control"}, response.text
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["input_tokens"] == 12 and body["usage"]["output_tokens"] == 6
        assert len(wire.drain()) == 1
        entries: Final = gateway.get("/v1/model/info")["data"]
        assert isinstance(entries, list)
        info: Final = next(entry for entry in entries if isinstance(entry, dict) and entry["model_name"] == model)
        assert isinstance(info["model_info"], dict)
        assert info["model_info"]["supports_tool_search"] is True, info["model_info"]
