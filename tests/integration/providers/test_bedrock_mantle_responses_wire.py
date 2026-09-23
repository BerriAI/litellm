import json
from collections.abc import Callable
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "bedrock_mantle/openai.gpt-5.6-sol"
_TOKEN: Final = "synthetic-mantle-bearer"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SHELL_ACTION: Final[dict[str, JsonValue]] = {"type": "exec", "command": ["ls", "-la"], "timeout_ms": 1000}
_OUTPUT_MESSAGE: Final[dict[str, JsonValue]] = {
    "type": "message",
    "id": "msg_mantle",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "mantle wire control", "annotations": []}],
}
_RESPONSE: Final = json.dumps(
    {
        "id": "resp_mantle",
        "object": "response",
        "status": "completed",
        "created_at": 1700000000,
        "model": "gpt-5.6-sol",
        "output": [_OUTPUT_MESSAGE],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
).encode()


def _codex_history(marker: str) -> list[JsonValue]:
    return [
        {"type": "message", "role": "user", "content": f"delegate to a subagent {marker}"},
        {
            "type": "agent_message",
            "id": "msg_agent",
            "content": [{"type": "text", "text": "sub-agent said "}, {"type": "text", "encrypted_content": "hello"}],
        },
        {"type": "context_compaction", "id": "cmp_1", "encrypted_content": "compacted-history"},
        {
            "type": "local_shell_call",
            "id": "lsc_1",
            "call_id": "call_shell",
            "status": "completed",
            "action": _SHELL_ACTION,
        },
        {"type": "function_call_output", "call_id": "call_shell", "output": "total 0"},
    ]


def _mantle_history(marker: str) -> list[JsonValue]:
    return [
        {"type": "message", "role": "user", "content": f"delegate to a subagent {marker}"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "sub-agent said hello"}]},
        {"type": "compaction", "encrypted_content": "compacted-history"},
        {
            "type": "function_call",
            "call_id": "call_shell",
            "name": "local_shell",
            "arguments": json.dumps(_SHELL_ACTION),
        },
        {"type": "function_call_output", "call_id": "call_shell", "output": "total 0"},
    ]


def _mantle_peer(marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/openai/v1/responses", request.target
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["input"] == _mantle_history(marker), json.dumps(body["input"])
        return Reply(body=_RESPONSE)

    return respond


@pytest.mark.covers("providers.bedrock_mantle.codex_history_items_reach_the_wire_as_supported_input_items")
def test_codex_agent_message_compaction_and_local_shell_items_are_rewritten_for_mantle(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    with wire_server(_mantle_peer(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_TOKEN, aws_region_name="us-east-1")
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": _codex_history(marker), "stream": False}
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["output"] == [
            {
                **_OUTPUT_MESSAGE,
                "phase": None,
                "content": [
                    {"type": "output_text", "text": "mantle wire control", "annotations": [], "logprobs": None}
                ],
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/openai/v1/responses")]


_MANTLE_MIN_MAX_OUTPUT_TOKENS: Final = 16


def _mantle_peer_expecting_max_output_tokens(marker: str, expected: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/openai/v1/responses", request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["max_output_tokens"] == expected, request.body.decode()
        assert body["input"] == f"clamp probe {marker}", request.body.decode()
        return Reply(body=_RESPONSE)

    return respond


@pytest.mark.covers("providers.bedrock_mantle.max_output_tokens_below_minimum_is_clamped_to_16_on_the_wire")
def test_max_output_tokens_below_mantle_minimum_is_raised_to_16_before_reaching_mantle(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    peer: Final = _mantle_peer_expecting_max_output_tokens(marker, _MANTLE_MIN_MAX_OUTPUT_TOKENS)
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_TOKEN, aws_region_name="us-east-1")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": f"clamp probe {marker}", "max_output_tokens": 5, "stream": False},
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["status"] == "completed", response.text
        assert payload["output"] == [
            {
                **_OUTPUT_MESSAGE,
                "phase": None,
                "content": [
                    {"type": "output_text", "text": "mantle wire control", "annotations": [], "logprobs": None}
                ],
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/openai/v1/responses")]
