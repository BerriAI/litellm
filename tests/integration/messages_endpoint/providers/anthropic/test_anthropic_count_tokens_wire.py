import json
from collections.abc import Callable
from typing import Final

import anthropic
import pytest
from anthropic.types import MessageTokensCount
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-4-6"
_API_KEY: Final = "synthetic-anthropic-key"
_COUNT_BETA: Final = "token-counting-2024-11-01"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Count this synthetic request."}]
_SYSTEM: Final[list[JsonValue]] = [{"type": "text", "text": "Count the supplied message and tool."}]
_TOOLS: Final[list[JsonValue]] = [
    {
        "name": "lookup",
        "description": "Look up a synthetic record.",
        "input_schema": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    }
]
_COUNTED_INPUT_TOKENS: Final = 73
_UPSTREAM_ERROR: Final = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": "The scripted count request was rejected."},
}


def _client(gateway: Gateway, key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=key,
        base_url=str(gateway.client.base_url),
        max_retries=0,
        http_client=gateway.client,
    )


def _count_peer(expected_body: dict[str, JsonValue], result: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages/count_tokens", request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert request.headers["anthropic-beta"] == _COUNT_BETA, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_body, request.body
        return result

    return respond


def test_anthropic_count_tokens_forwards_body_and_refuses_a_key_without_model_access(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with wire_server(
            _count_peer(
                {"model": _BACKEND, "messages": _MESSAGES, "system": _SYSTEM, "tools": _TOOLS},
                Reply(body=json.dumps({"input_tokens": _COUNTED_INPUT_TOKENS}).encode()),
            )
        ) as wire:
            model: Final = scenario.model(
                model=f"anthropic/{_BACKEND}",
                api_base=wire.url,
                api_key=_API_KEY,
                num_retries=0,
            )
            restricted_model: Final = scenario.model(
                model="anthropic/claude-haiku-4-5-20251001",
                api_base=wire.url,
                api_key=_API_KEY,
                num_retries=0,
            )
            restricted_key: Final = scenario.key(models=[restricted_model])

            client: Final = _client(gateway, gateway.key)
            result: Final[MessageTokensCount] = client.messages.count_tokens(
                model=model,
                messages=_MESSAGES,
                system=_SYSTEM,
                tools=_TOOLS,
                extra_headers={"anthropic-beta": _COUNT_BETA},
            )
            assert result.model_dump() == {"input_tokens": _COUNTED_INPUT_TOKENS}
            requests: Final = wire.drain()
            assert len(requests) == 1, requests

            restricted_client: Final = _client(gateway, restricted_key)
            with pytest.raises(anthropic.APIStatusError) as raised:
                restricted_client.messages.count_tokens(
                    model=model,
                    messages=_MESSAGES,
                    system=_SYSTEM,
                    tools=_TOOLS,
                    extra_headers={"anthropic-beta": _COUNT_BETA},
                )
            assert raised.value.status_code == 403, raised.value.response.text
            assert _JSON_OBJECT.validate_json(raised.value.response.content) == {
                "error": {
                    "message": (
                        f"The requested model '{model}' is not available for this API key, or the model name is "
                        "invalid. Check the models available to you and try again."
                    ),
                    "type": "key_model_access_denied",
                    "param": "model",
                    "code": "403",
                }
            }, raised.value.response.text
            assert wire.drain() == ()


def test_anthropic_count_tokens_uses_local_count_after_an_upstream_server_error(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with wire_server(
            _count_peer(
                {"model": _BACKEND, "messages": _MESSAGES, "system": _SYSTEM, "tools": _TOOLS},
                Reply(status=500, body=json.dumps(_UPSTREAM_ERROR).encode()),
            )
        ) as wire:
            model: Final = scenario.model(
                model=f"anthropic/{_BACKEND}",
                api_base=wire.url,
                api_key=_API_KEY,
                num_retries=0,
            )
            body: Final = {"model": model, "messages": _MESSAGES, "system": _SYSTEM, "tools": _TOOLS}
            client: Final = _client(gateway, gateway.key)
            result: Final = client.messages.count_tokens(
                model=model,
                messages=_MESSAGES,
                system=_SYSTEM,
                tools=_TOOLS,
                extra_headers={"anthropic-beta": _COUNT_BETA},
            )
            assert len(wire.drain()) == 1
            local_response: Final = gateway.request(
                "POST", "/utils/token_counter", body, params={"call_endpoint": "false"}
            )
            assert local_response.status_code == 200, local_response.text
            local_body: Final = _JSON_OBJECT.validate_json(local_response.content)
            assert result.input_tokens == local_body["total_tokens"], local_response.text


def test_anthropic_count_tokens_falls_back_locally_after_an_upstream_client_error(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with wire_server(
            _count_peer(
                {"model": _BACKEND, "messages": _MESSAGES, "system": _SYSTEM, "tools": _TOOLS},
                Reply(status=400, body=json.dumps(_UPSTREAM_ERROR).encode()),
            )
        ) as wire:
            model: Final = scenario.model(
                model=f"anthropic/{_BACKEND}",
                api_base=wire.url,
                api_key=_API_KEY,
                num_retries=0,
            )
            client: Final = _client(gateway, gateway.key)
            result: Final = client.messages.count_tokens(
                model=model,
                messages=_MESSAGES,
                system=_SYSTEM,
                tools=_TOOLS,
                extra_headers={"anthropic-beta": _COUNT_BETA},
            )
            assert len(wire.drain()) == 1
            local_response: Final = gateway.request(
                "POST",
                "/utils/token_counter",
                {"model": model, "messages": _MESSAGES, "system": _SYSTEM, "tools": _TOOLS},
                params={"call_endpoint": "false"},
            )
            assert local_response.status_code == 200, local_response.text
            local_body: Final = _JSON_OBJECT.validate_json(local_response.content)
            assert result.input_tokens == local_body["total_tokens"], local_response.text


def test_anthropic_count_tokens_preserves_thinking_tool_choice_and_output_config(gateway: Gateway) -> None:
    pytest.skip("BUG: /v1/messages/count_tokens drops thinking, tool_choice and output_config before the provider count call")

    thinking: Final = {"type": "enabled", "budget_tokens": 1024}
    tool_choice: Final = {"type": "auto", "disable_parallel_tool_use": True}
    output_config: Final = {"effort": "medium"}
    expected_body: Final = {
        "model": _BACKEND,
        "messages": _MESSAGES,
        "system": _SYSTEM,
        "tools": _TOOLS,
        "thinking": thinking,
        "tool_choice": tool_choice,
        "output_config": output_config,
    }

    with gateway.scenario() as scenario:
        with wire_server(
            _count_peer(expected_body, Reply(body=json.dumps({"input_tokens": _COUNTED_INPUT_TOKENS}).encode()))
        ) as wire:
            model: Final = scenario.model(
                model=f"anthropic/{_BACKEND}",
                api_base=wire.url,
                api_key=_API_KEY,
                num_retries=0,
            )
            client: Final = _client(gateway, gateway.key)
            result: Final = client.messages.count_tokens(
                model=model,
                messages=_MESSAGES,
                system=_SYSTEM,
                tools=_TOOLS,
                thinking=thinking,
                tool_choice=tool_choice,
                output_config=output_config,
                extra_headers={"anthropic-beta": _COUNT_BETA},
            )
            assert result.input_tokens == _COUNTED_INPUT_TOKENS
            assert len(wire.drain()) == 1
