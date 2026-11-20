"""us.anthropic.claude-fable-5-2 on Bedrock Converse: the row's capability flags shape the Converse body.

Fable rejects sampling params and forced tool choice and always thinks, so the proxy must strip
or refuse those before the request reaches Bedrock, the same way it does on the Anthropic API.
"""

import json
from collections.abc import Callable
from typing import Final

from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL_ID: Final = "us.anthropic.claude-fable-5-2"
_TARGET: Final = f"/model/{_MODEL_ID}/converse"
_TOKEN: Final = "synthetic-bedrock-bearer"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_FORCED_TOOL_CHOICES: Final[tuple[JsonValue, ...]] = ("required", {"type": "function", "function": {"name": "lookup"}})
_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a synthetic record.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }
]


def _reply(usage: dict[str, JsonValue] | None = None) -> bytes:
    return json.dumps(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "scripted thought", "signature": "scripted-signature"}
                            }
                        },
                        {"text": "done"},
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": usage or {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15},
            "metrics": {"latencyMs": 1},
        }
    ).encode()


def _deployment(scenario: Scenario, api_base: str, *, drop_params: bool = False) -> str:
    if drop_params:
        return scenario.model(
            model=f"bedrock/converse/{_MODEL_ID}",
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=api_base,
            num_retries=0,
            drop_params=True,
        )
    return scenario.model(
        model=f"bedrock/converse/{_MODEL_ID}",
        api_key=_TOKEN,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=api_base,
        num_retries=0,
    )


def _peer(on_body: Callable[[dict[str, JsonValue]], bytes]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == _TARGET, (request.method, request.target)
        assert request.headers["authorization"] == f"Bearer {_TOKEN}", request.headers
        return Reply(body=on_body(_JSON_OBJECT.validate_json(request.body)))

    return respond


def _chat_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "synthetic fable request"}],
        "max_tokens": 64,
        **extra,
    }


def test_drop_params_removes_temperature_and_top_p_before_bedrock(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert body["inferenceConfig"] == {"maxTokens": 64}, body
        return _reply()

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url, drop_params=True)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, temperature=0.2, top_p=0.9))
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "done", response.text
        assert len(wire.drain()) == 1


def test_sampling_params_are_refused_before_reaching_bedrock_without_drop_params(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        raise AssertionError(f"temperature must not reach Bedrock: {body}")

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, temperature=0.2))
        assert response.status_code == 400, response.text
        assert "temperature" in response.text and "drop_params" in response.text, response.text
        assert len(wire.drain()) == 0


def test_drop_params_downgrades_forced_and_named_tool_choice_to_auto(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        tool_config: Final = body["toolConfig"]
        assert isinstance(tool_config, dict), body
        assert tool_config.get("toolChoice") in (None, {"auto": {}}), body
        tools: Final = tool_config["tools"]
        assert isinstance(tools, list) and len(tools) == 1, body
        return _reply()

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url, drop_params=True)
        responses: Final = tuple(
            gateway.request("POST", "/v1/chat/completions", _chat_body(model, tools=_TOOLS, tool_choice=tool_choice))
            for tool_choice in _FORCED_TOOL_CHOICES
        )
        assert [response.status_code for response in responses] == [200, 200], [response.text for response in responses]
        assert len(wire.drain()) == 2


def test_forced_tool_choice_is_refused_before_reaching_bedrock_without_drop_params(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        raise AssertionError(f"forced tool_choice must not reach Bedrock: {body}")

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, tools=_TOOLS, tool_choice="required")
        )
        assert response.status_code == 400, response.text
        assert "tool_choice" in response.text, response.text
        assert len(wire.drain()) == 0


def test_thinking_disabled_is_dropped_for_an_always_on_model(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert "thinking" not in json.dumps(body.get("additionalModelRequestFields", {})), body
        return _reply()

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, thinking={"type": "disabled"})
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["reasoning_content"] == "scripted thought", response.text
        assert len(wire.drain()) == 1


def test_reasoning_effort_becomes_adaptive_thinking_with_output_config_effort(gateway: Gateway) -> None:
    def on_body(body: dict[str, JsonValue]) -> bytes:
        extra: Final = body["additionalModelRequestFields"]
        assert isinstance(extra, dict), body
        assert extra["thinking"] == {"type": "adaptive", "display": "summarized"}, body
        assert extra["output_config"] == {"effort": "max"}, body
        assert "budget_tokens" not in json.dumps(body), body
        return _reply()

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, reasoning_effort="max"))
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_one_hour_cache_point_reaches_bedrock_and_its_usage_comes_back(gateway: Gateway) -> None:
    usage: Final[dict[str, JsonValue]] = {
        "inputTokens": 9,
        "outputTokens": 3,
        "totalTokens": 7237,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 7225,
        "cacheDetails": [{"ttl": "1h", "inputTokens": 7225}],
    }

    def on_body(body: dict[str, JsonValue]) -> bytes:
        assert body["system"] == [
            {"text": "synthetic cached system prompt"},
            {"cachePoint": {"type": "default", "ttl": "1h"}},
        ], body
        return _reply(usage)

    with wire_server(_peer(on_body)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(
                model,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "synthetic cached system prompt",
                                "cache_control": {"type": "ephemeral", "ttl": "1h"},
                            }
                        ],
                    },
                    {"role": "user", "content": "synthetic fable request"},
                ],
            ),
        )
        assert response.status_code == 200, response.text
        usage_out: Final = _JSON_OBJECT.validate_json(response.text)["usage"]
        assert isinstance(usage_out, dict), response.text
        prompt_details: Final = usage_out["prompt_tokens_details"]
        assert isinstance(prompt_details, dict), response.text
        assert prompt_details["cache_creation_token_details"] == {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 7225,
        }, response.text
        assert usage_out["cache_creation_input_tokens"] == 7225, response.text
        assert len(wire.drain()) == 1
