import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

MODEL: Final = "bedrock_mantle/openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-mantle-bearer"
CIPHERTEXT: Final = "synthetic-compaction-ciphertext"
CALL_ID: Final = "call_synthetic_shell"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
ACTION: Final[dict[str, JsonValue]] = {"type": "exec", "command": ["ls", "-la"], "timeout_ms": 1000}
RESPONSE: Final = json.dumps(
    {
        "id": "resp_synthetic_mantle",
        "object": "response",
        "created_at": 1789788253,
        "status": "completed",
        "model": "openai.gpt-5.6-sol",
        "output": [
            {
                "type": "message",
                "id": "msg_synthetic_mantle",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "mantle wire control", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 21, "output_tokens": 4, "total_tokens": 25},
    }
).encode()


def user_turn(text: str) -> JsonValue:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def codex_history(marker: str) -> tuple[JsonValue, ...]:
    return (
        user_turn(f"first turn {marker}"),
        {"type": "agent_message", "role": "assistant", "content": [{"type": "output_text", "text": "sub-agent reply"}]},
        {"type": "context_compaction", "encrypted_content": CIPHERTEXT},
        {"type": "local_shell_call", "call_id": CALL_ID, "status": "completed", "action": ACTION},
        {"type": "function_call_output", "call_id": CALL_ID, "output": "synthetic shell output"},
        user_turn(f"next turn {marker}"),
    )


def mantle_history(marker: str) -> tuple[JsonValue, ...]:
    return (
        user_turn(f"first turn {marker}"),
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "sub-agent reply"}]},
        {"type": "compaction", "encrypted_content": CIPHERTEXT},
        {"type": "function_call", "call_id": CALL_ID, "name": "local_shell", "arguments": json.dumps(ACTION)},
        {"type": "function_call_output", "call_id": CALL_ID, "output": "synthetic shell output"},
        user_turn(f"next turn {marker}"),
    )


@pytest.mark.covers("other.provider_wire.bedrock_mantle.codex_history_items_reach_mantle_as_supported_types")
def test_codex_agent_message_context_compaction_and_local_shell_call_reach_mantle_as_supported_items(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    expected_input: Final = list(mantle_history(marker))

    def mantle_peer(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/openai/v1/responses", request.target
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "openai.gpt-5.6-sol", body
        assert body["input"] == expected_input, body["input"]
        return Reply(body=RESPONSE)

    with wire_server(mantle_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=MODEL, api_key=TOKEN, api_base=wire.url, aws_region_name="us-east-2")
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": list(codex_history(marker)), "store": False}
        )
        assert response.status_code == 200, response.text
        assert response.json()["output"][0]["content"][0]["text"] == "mantle wire control", response.text
        assert response.json()["usage"]["total_tokens"] == 25, response.text
        forwarded: Final = wire.drain()
        assert len(forwarded) == 1, forwarded
        assert JSON_OBJECT.validate_json(forwarded[0].body)["input"] == expected_input, forwarded[0].body


SHELL_TOOL: Final[JsonValue] = {
    "type": "function",
    "name": "shell",
    "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
}
APPLY_PATCH_TOOL: Final[JsonValue] = {
    "type": "function",
    "name": "apply_patch",
    "description": "apply a diff",
    "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]},
}


@pytest.mark.covers("providers.bedrock_mantle.codex_additional_tools_input_item_is_hoisted_to_top_level_tools")
def test_codex_additional_tools_input_item_reaches_mantle_as_top_level_tools(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    expected_input: Final[list[JsonValue]] = [user_turn(f"hoist tools {marker}")]
    expected_tools: Final[list[JsonValue]] = [SHELL_TOOL, APPLY_PATCH_TOOL]

    def mantle_peer(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/openai/v1/responses", request.target
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        body: Final = JSON_OBJECT.validate_json(request.body)
        assert body["model"] == "openai.gpt-5.6-sol", body
        assert body["input"] == expected_input, body["input"]
        assert body["tools"] == expected_tools, body
        return Reply(body=RESPONSE)

    with wire_server(mantle_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=MODEL, api_key=TOKEN, api_base=wire.url, aws_region_name="us-east-2")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "input": [
                    {"type": "additional_tools", "role": "developer", "tools": [APPLY_PATCH_TOOL]},
                    user_turn(f"hoist tools {marker}"),
                ],
                "tools": [SHELL_TOOL],
                "store": False,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["output"][0]["content"][0]["text"] == "mantle wire control", response.text
        forwarded: Final = wire.drain()
        assert len(forwarded) == 1, forwarded
        forwarded_body: Final = JSON_OBJECT.validate_json(forwarded[0].body)
        assert forwarded_body["input"] == expected_input, forwarded[0].body
        assert forwarded_body["tools"] == expected_tools, forwarded[0].body
