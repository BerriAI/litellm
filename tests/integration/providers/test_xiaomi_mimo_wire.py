import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_BACKENDS: Final = ("mimo-v2.6-pro", "mimo-v2.6-flash")
_API_KEY: Final = "synthetic-xiaomi-key"
_ARITHMETIC_PROMPT: Final = "What is 17 + 26? Answer with just the number."
_WEATHER_PROMPT: Final = "What is the weather in Paris? Use the tool."
_COUNTING_PROMPT: Final = "Count from 1 to 5, one number per line."
_WEATHER_TOOL: Final[JsonValue] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_COST_MAP_PATH: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, object]])


class _Delta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    content: str | None = None
    reasoning_content: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore")
    delta: _Delta
    finish_reason: str | None = None


class _Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    choices: tuple[_Choice, ...]


def _catalog_cost(backend: str, field: str) -> float:
    cost_map: Final = _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())
    cost_value: Final = cost_map[f"xiaomi_mimo/{backend}"][field]
    assert isinstance(cost_value, (int, float))
    return float(cost_value)


def _approx(value: float) -> object:
    return pytest.approx(value, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]  # pytest lacks typed approx stubs


def _completion(identity: str, backend: str, message: Mapping[str, object], finish: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": backend,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 23, "completion_tokens": 41, "total_tokens": 64},
        }
    ).encode()


def _frame(identity: str, backend: str, delta: Mapping[str, object], finish: str | None = None) -> bytes:
    value: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": backend,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def _assert_provider_request(request: Request, backend: str, prompt: str) -> dict[str, JsonValue]:
    assert request.method == "POST"
    assert request.target == "/chat/completions"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    assert request.headers["content-type"] == "application/json"
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert body["model"] == backend
    assert body["messages"] == [{"role": "user", "content": prompt}]
    return body


@pytest.mark.covers("other.provider_wire.xiaomi_mimo.reasoning_content_and_registry_pricing")
@pytest.mark.parametrize("backend", _BACKENDS)
def test_xiaomi_mimo_nonstream_surfaces_reasoning_and_charges_registry_price(gateway: Gateway, backend: str) -> None:
    identity: Final = f"xiaomi-cost-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _assert_provider_request(request, backend, _ARITHMETIC_PROMPT)
        assert body["max_tokens"] == 256
        assert "max_completion_tokens" not in body
        return Reply(
            body=_completion(
                identity,
                backend,
                {"role": "assistant", "content": "43", "reasoning_content": "17 plus 26 is 43."},
                "stop",
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xiaomi_mimo/{backend}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _ARITHMETIC_PROMPT}],
                "max_completion_tokens": 256,
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "43",
                    "reasoning_content": "17 plus 26 is 43.",
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ]
        assert payload["usage"] == {"prompt_tokens": 23, "completion_tokens": 41, "total_tokens": 64}
        expected_cost: Final = 23 * _catalog_cost(backend, "input_cost_per_token") + 41 * _catalog_cost(
            backend, "output_cost_per_token"
        )
        assert float(response.headers["x-litellm-response-cost"]) == _approx(expected_cost)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (23, 41)
        spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float, str))
    assert float(spend) == _approx(expected_cost)


@pytest.mark.covers("other.provider_wire.xiaomi_mimo.reasoning_and_answer_stream_as_deltas")
def test_xiaomi_mimo_stream_delivers_reasoning_then_answer_deltas(gateway: Gateway) -> None:
    backend: Final = _BACKENDS[0]
    identity: Final = f"xiaomi-stream-{uuid.uuid4().hex}"
    frames: Final = (
        _frame(identity, backend, {"role": "assistant", "reasoning_content": "Count "}),
        _frame(identity, backend, {"reasoning_content": "up by one."}),
        _frame(identity, backend, {"content": "1\n2\n"}),
        _frame(identity, backend, {"content": "3\n4\n5"}),
        _frame(identity, backend, {}, finish="stop"),
        b"data: [DONE]\n\n",
    )

    def respond(request: Request) -> Reply:
        body: Final = _assert_provider_request(request, backend, _COUNTING_PROMPT)
        assert body["stream"] is True
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xiaomi_mimo/{backend}", api_base=wire.url, api_key=_API_KEY)
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": _COUNTING_PROMPT}], "stream": True},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read()
            lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        assert lines[-1] == "data: [DONE]"
        chunks: Final = tuple(_Chunk.model_validate_json(line.removeprefix("data: ")) for line in lines[:-1])
        assert {chunk.id for chunk in chunks} == {identity}
        choices: Final = tuple(choice for chunk in chunks for choice in chunk.choices)
        assert "".join(choice.delta.reasoning_content or "" for choice in choices) == "Count up by one."
        assert "".join(choice.delta.content or "" for choice in choices) == "1\n2\n3\n4\n5"
        assert tuple(choice.finish_reason for choice in choices if choice.finish_reason) == ("stop",)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]


@pytest.mark.covers("other.provider_wire.xiaomi_mimo.tool_call_survives_translation")
def test_xiaomi_mimo_tool_call_is_forwarded_and_returned(gateway: Gateway) -> None:
    backend: Final = _BACKENDS[1]
    identity: Final = f"xiaomi-tool-{uuid.uuid4().hex}"
    tool_call: Final = {
        "id": "call_paris",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
    }

    def respond(request: Request) -> Reply:
        body: Final = _assert_provider_request(request, backend, _WEATHER_PROMPT)
        assert body["tools"] == [_WEATHER_TOOL]
        assert body["tool_choice"] == "auto"
        return Reply(
            body=_completion(
                identity,
                backend,
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Need the tool.",
                    "tool_calls": [tool_call],
                },
                "tool_calls",
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"xiaomi_mimo/{backend}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": _WEATHER_PROMPT}],
                "tools": [_WEATHER_TOOL],
                "tool_choice": "auto",
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"] == [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Need the tool.",
                    "tool_calls": [tool_call],
                    "provider_specific_fields": {"refusal": None},
                },
                "provider_specific_fields": {},
            }
        ]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/chat/completions")]
