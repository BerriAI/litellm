import json
import re
from collections.abc import Callable, Mapping
from typing import Final
from uuid import uuid4

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.sigv4 import signature
from integration._support.wire import Reply, Request, wire_server
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter

from litellm.constants import DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS

_HAIKU: Final = "anthropic.claude-haiku-4-5"
_OPUS: Final = "anthropic.claude-opus-5-5"
_SAFEGUARD: Final = "openai.gpt-oss-safeguard-120b"
_API_KEY: Final = "synthetic-mantle-bearer"
_ACCESS_KEY: Final = "AKIAINTEGRATION000009"
_SECRET_KEY: Final = "synthetic-secret-key-for-testing"
_MESSAGES_PATH: Final = "/anthropic/v1/messages"
_BRIDGE_VERSION: Final = "bedrock-2023-05-31"
_INPUT_TOKENS: Final = 23
_OUTPUT_TOKENS: Final = 7
_USAGE: Final = (_INPUT_TOKENS, _OUTPUT_TOKENS, _INPUT_TOKENS + _OUTPUT_TOKENS)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_CITY_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_weather", "description": "Weather lookup", "parameters": _CITY_SCHEMA},
}
_WEATHER_TOOL_OUTBOUND: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "input_schema": _CITY_SCHEMA,
    "type": "custom",
    "description": "Weather lookup",
}
_SPEND_COLUMNS: Final = (
    "SELECT request_id, call_type, status, spend, prompt_tokens, completion_tokens, model, model_group,"
    ' custom_llm_provider, cache_hit, litellm_call_id FROM "LiteLLM_SpendLogs"'
)
_BY_REQUEST_ID: Final = _SPEND_COLUMNS + " WHERE request_id=%s"
_BY_CALL_ID: Final = _SPEND_COLUMNS + " WHERE litellm_call_id=%s"
_BY_MODEL_GROUP: Final = _SPEND_COLUMNS + " WHERE model_group=%s ORDER BY request_id"


def _prompt(marker: str) -> str:
    return f"mantle claude chat marker-{marker}"


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _user_turn(marker: str) -> dict[str, JsonValue]:
    return {"role": "user", "content": [{"type": "text", "text": _prompt(marker)}]}


def _chat_messages(marker: str) -> list[JsonValue]:
    return [{"role": "user", "content": _prompt(marker)}]


def _outbound(backend: str, messages: list[JsonValue], **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "messages": messages,
        "max_tokens": DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS,
        "anthropic_version": _BRIDGE_VERSION,
        "model": backend,
        **fields,
    }


def _message_reply(marker: str, backend: str, content: list[JsonValue], stop_reason: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"msg_bdrk_{marker}",
                "type": "message",
                "role": "assistant",
                "model": backend,
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS},
            }
        ).encode()
    )


def _text_reply(marker: str, backend: str = _HAIKU) -> Reply:
    return _message_reply(marker, backend, [{"type": "text", "text": _answer(marker)}], "end_turn")


def _tool_reply(marker: str, backend: str, name: str) -> Reply:
    block: Final[JsonValue] = {"type": "tool_use", "id": f"toolu_{marker}", "name": name, "input": {"city": "Paris"}}
    return _message_reply(marker, backend, [block], "tool_use")


def _stream_reply(
    marker: str, block: Mapping[str, JsonValue], deltas: tuple[Mapping[str, JsonValue], ...], stop_reason: str
) -> Reply:
    opening: Final = {
        "id": f"msg_bdrk_{marker}",
        "type": "message",
        "role": "assistant",
        "model": _HAIKU,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 1},
    }
    events: Final = (
        ("message_start", {"message": opening}),
        ("content_block_start", {"index": 0, "content_block": block}),
        *(("content_block_delta", {"index": 0, "delta": delta}) for delta in deltas),
        ("content_block_stop", {"index": 0}),
        (
            "message_delta",
            {"delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": _OUTPUT_TOKENS}},
        ),
        ("message_stop", {}),
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(
            f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n".encode() for kind, payload in events
        ),
    )


def _text_stream(marker: str) -> Reply:
    return _stream_reply(
        marker,
        {"type": "text", "text": ""},
        ({"type": "text_delta", "text": "answer "}, {"type": "text_delta", "text": f"marker-{marker}"}),
        "end_turn",
    )


def _tool_stream(marker: str) -> Reply:
    return _stream_reply(
        marker,
        {"type": "tool_use", "id": f"toolu_{marker}", "name": "get_weather", "input": {}},
        (
            {"type": "input_json_delta", "partial_json": '{"city": '},
            {"type": "input_json_delta", "partial_json": '"Paris"}'},
        ),
        "tool_use",
    )


def _bearer_peer(expected: Mapping[str, JsonValue], reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _MESSAGES_PATH), request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", sorted(request.headers)
        assert _JSON_OBJECT.validate_json(request.body) == expected, request.body
        return reply

    return respond


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _cost_map_row(gateway: Gateway, name: str) -> dict[str, JsonValue]:
    return object_value(gateway.get("/public/litellm_model_cost_map")[name])


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def _mantle_cost(
    gateway: Gateway, backend: str, input_tokens: int = _INPUT_TOKENS, output_tokens: int = _OUTPUT_TOKENS
) -> float:
    row: Final = _cost_map_row(gateway, f"bedrock_mantle/{backend}")
    cost: Final = input_tokens * _number(row["input_cost_per_token"]) + output_tokens * _number(
        row["output_cost_per_token"]
    )
    assert cost > 0, row
    return cost


def _mantle_claude_where(gateway: Gateway, flag: str, disabled: bool) -> str:
    rows: Final = gateway.get("/public/litellm_model_cost_map")
    names: Final = sorted(
        name
        for name, row in rows.items()
        if name.startswith("bedrock_mantle/anthropic.claude") and (object_value(row).get(flag) is False) is disabled
    )
    assert names, f"The cost map has no Mantle Claude row with {flag} disabled={disabled}"
    return names[0].removeprefix("bedrock_mantle/")


def _spend_row(query: str, identity: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(query, (identity,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _assert_success_row(
    row: Mapping[str, JsonValue], model_group: str, backend: str, call_type: str, cost: float
) -> None:
    assert (row["status"], row["call_type"], row["cache_hit"] == "True") == ("success", call_type, False), row
    assert (row["model_group"], row["model"], row["custom_llm_provider"]) == (
        model_group,
        f"bedrock_mantle/{backend}",
        "bedrock_mantle",
    ), row
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_INPUT_TOKENS, _OUTPUT_TOKENS), row
    assert _number(row["spend"]) == pytest.approx(cost), row


def _only_choice(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    choices: Final = body["choices"]
    assert isinstance(choices, list) and len(choices) == 1, body
    return object_value(choices[0])


def _token_usage(body: Mapping[str, JsonValue]) -> tuple[JsonValue, JsonValue, JsonValue]:
    usage: Final = object_value(body["usage"])
    return usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]


def _assert_text_completion(response: httpx.Response, model: str, marker: str) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    body: Final = _JSON_OBJECT.validate_json(response.content)
    assert (body["object"], body["model"]) == ("chat.completion", model), response.text
    choice: Final = _only_choice(body)
    message: Final = object_value(choice["message"])
    assert (choice["finish_reason"], message["role"], message["content"], message.get("tool_calls")) == (
        "stop",
        "assistant",
        _answer(marker),
        None,
    ), response.text
    assert _token_usage(body) == _USAGE, response.text
    return body


def _sse_objects(text: str) -> tuple[dict[str, JsonValue], ...]:
    data: Final = tuple(line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: "))
    assert data and data[-1] == "[DONE]", text
    return tuple(_JSON_OBJECT.validate_json(item) for item in data[:-1])


def _assert_chunk_envelope(chunks: tuple[dict[str, JsonValue], ...], model: str, text: str) -> str:
    assert {(chunk["object"], chunk["model"]) for chunk in chunks} == {("chat.completion.chunk", model)}, text
    identities: Final = {string_value(chunk["id"]) for chunk in chunks}
    assert len(identities) == 1, text
    return next(iter(identities))


def _finish_reasons(chunks: tuple[dict[str, JsonValue], ...]) -> list[JsonValue]:
    reasons: Final = (_only_choice(chunk).get("finish_reason") for chunk in chunks)
    return [reason for reason in reasons if reason is not None]


def _deltas(chunks: tuple[dict[str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(_only_choice(chunk)["delta"]) for chunk in chunks)


def _sdk_base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


def _error_message(response: httpx.Response) -> str:
    return string_value(object_value(_JSON_OBJECT.validate_json(response.content)["error"])["message"])


@pytest.mark.parametrize("backend", [_HAIKU, _OPUS], ids=["haiku", "opus_with_a_cheaper_bare_twin"])
def test_chat_completion_on_a_mantle_claude_id_posts_native_messages_and_spends_at_the_mantle_price(
    gateway: Gateway, backend: str
) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(backend, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker, backend))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{backend}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker)}
        )
        body: Final = _assert_text_completion(response, model, marker)
        cost: Final = _mantle_cost(gateway, backend)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(cost), response.text
        _assert_success_row(_spend_row(_BY_REQUEST_ID, string_value(body["id"])), model, backend, "acompletion", cost)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


@pytest.mark.parametrize(
    ("stream_options", "usage_chunks"),
    [
        pytest.param({"stream_options": {"include_usage": True}}, [_USAGE], id="include_usage"),
        pytest.param({}, [], id="no_usage"),
    ],
)
def test_chat_stream_on_a_mantle_claude_id_relays_openai_chunks_from_the_native_stream(
    gateway: Gateway, stream_options: dict[str, JsonValue], usage_chunks: list[tuple[int, int, int]]
) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)], stream=True)
    with wire_server(_bearer_peer(expected, _text_stream(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "stream": True, "messages": _chat_messages(marker), **stream_options},
        )
        assert response.status_code == 200, response.text
        chunks: Final = _sse_objects(response.text)
        identity: Final = _assert_chunk_envelope(chunks, model, response.text)
        assert "".join(str(delta.get("content") or "") for delta in _deltas(chunks)) == _answer(marker), response.text
        assert _finish_reasons(chunks) == ["stop"], response.text
        assert [_token_usage(chunk) for chunk in chunks if chunk.get("usage") is not None] == usage_chunks, (
            response.text
        )
        _assert_success_row(
            _spend_row(_BY_REQUEST_ID, identity), model, _HAIKU, "acompletion", _mantle_cost(gateway, _HAIKU)
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


@pytest.mark.parametrize(
    ("leading_messages", "request_fields", "outbound_fields"),
    [
        pytest.param(
            [{"role": "system", "content": "be terse"}],
            {},
            {"system": [{"type": "text", "text": "be terse"}]},
            id="system_prompt",
        ),
        pytest.param([], {"max_tokens": 77}, {"max_tokens": 77}, id="explicit_max_tokens"),
    ],
)
def test_chat_request_fields_are_translated_to_the_native_messages_shape(
    gateway: Gateway,
    leading_messages: list[JsonValue],
    request_fields: dict[str, JsonValue],
    outbound_fields: dict[str, JsonValue],
) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)], **outbound_fields)
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [*leading_messages, *_chat_messages(marker)], **request_fields},
        )
        _assert_text_completion(response, model, marker)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


@pytest.mark.parametrize(
    ("tool_choice", "outbound_choice"),
    [pytest.param("auto", {"type": "auto"}, id="auto"), pytest.param("required", {"type": "any"}, id="required")],
)
def test_chat_tools_reach_mantle_as_input_schema_and_the_tool_use_block_returns_as_a_tool_call(
    gateway: Gateway, tool_choice: str, outbound_choice: dict[str, JsonValue]
) -> None:
    marker: Final = uuid4().hex
    backend: Final = _mantle_claude_where(gateway, "supports_forced_tool_use", disabled=False)
    expected: Final = _outbound(
        backend, [_user_turn(marker)], tools=[_WEATHER_TOOL_OUTBOUND], tool_choice=outbound_choice
    )
    reply: Final = _tool_reply(marker, backend, "get_weather")
    with wire_server(_bearer_peer(expected, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{backend}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _chat_messages(marker), "tools": [_WEATHER_TOOL], "tool_choice": tool_choice},
        )
        assert response.status_code == 200, response.text
        choice: Final = _only_choice(_JSON_OBJECT.validate_json(response.content))
        message: Final = object_value(choice["message"])
        calls: Final = message["tool_calls"]
        assert isinstance(calls, list) and len(calls) == 1, response.text
        call: Final = object_value(calls[0])
        function: Final = object_value(call["function"])
        assert (choice["finish_reason"], message["content"], call["id"], call["type"], function["name"]) == (
            "tool_calls",
            None,
            f"toolu_{marker}",
            "function",
            "get_weather",
        ), response.text
        assert json.loads(string_value(function["arguments"])) == {"city": "Paris"}, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_chat_stream_relays_a_native_tool_use_block_as_tool_call_chunks(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)], tools=[_WEATHER_TOOL_OUTBOUND], stream=True)
    with wire_server(_bearer_peer(expected, _tool_stream(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "stream": True, "messages": _chat_messages(marker), "tools": [_WEATHER_TOOL]},
        )
        assert response.status_code == 200, response.text
        chunks: Final = _sse_objects(response.text)
        _assert_chunk_envelope(chunks, model, response.text)
        fragments: Final = tuple(
            object_value(object_value(calls[0])["function"])
            for calls in (delta.get("tool_calls") for delta in _deltas(chunks))
            if isinstance(calls, list)
        )
        opening: Final = object_value(object_value(_deltas(chunks)[0])["tool_calls"][0])
        assert (opening["id"], opening["type"], fragments[0]["name"]) == (
            f"toolu_{marker}",
            "function",
            "get_weather",
        ), response.text
        arguments: Final = "".join(string_value(fragment["arguments"]) for fragment in fragments)
        assert json.loads(arguments) == {"city": "Paris"}, response.text
        assert _finish_reasons(chunks) == ["tool_calls"], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_chat_tool_result_turn_reaches_mantle_as_tool_use_and_tool_result_blocks(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    tool_call: Final = f"toolu_{marker}"
    expected: Final = _outbound(
        _HAIKU,
        [
            _user_turn(marker),
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_call, "name": "get_weather", "input": {"city": "Paris"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_call, "content": "sunny"}]},
        ],
        tools=[_WEATHER_TOOL_OUTBOUND],
    )
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "tools": [_WEATHER_TOOL],
                "messages": [
                    *_chat_messages(marker),
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call,
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": tool_call, "content": "sunny"},
                ],
            },
        )
        _assert_text_completion(response, model, marker)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_chat_json_schema_response_format_is_sent_as_a_forced_tool_and_returned_as_json_content(
    gateway: Gateway,
) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(
        _HAIKU,
        [_user_turn(marker)],
        tools=[{"name": "json_tool_call", "input_schema": _CITY_SCHEMA}],
        tool_choice={"name": "json_tool_call", "type": "tool"},
    )
    reply: Final = _tool_reply(marker, _HAIKU, "json_tool_call")
    with wire_server(_bearer_peer(expected, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": _chat_messages(marker),
                "response_format": {"type": "json_schema", "json_schema": {"name": "weather", "schema": _CITY_SCHEMA}},
            },
        )
        assert response.status_code == 200, response.text
        choice: Final = _only_choice(_JSON_OBJECT.validate_json(response.content))
        message: Final = object_value(choice["message"])
        assert (choice["finish_reason"], message.get("tool_calls")) == ("stop", None), response.text
        assert json.loads(string_value(message["content"])) == {"city": "Paris"}, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def _authorization_field(part: str) -> tuple[str, str]:
    name, _, value = part.partition("=")
    return name, value


def _sigv4_peer(expected: Mapping[str, JsonValue], reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _MESSAGES_PATH), request.target
        authorization: Final = request.headers["authorization"]
        assert authorization.startswith("AWS4-HMAC-SHA256 "), sorted(request.headers)
        fields: Final = dict(
            _authorization_field(part) for part in authorization.removeprefix("AWS4-HMAC-SHA256 ").split(", ")
        )
        access_key, scope = fields["Credential"].split("/", 1)
        assert access_key == _ACCESS_KEY, authorization
        assert scope == f"{request.headers['x-amz-date'][:8]}/us-east-1/bedrock/aws4_request", authorization
        signed: Final = signature(
            "POST", _MESSAGES_PATH, request.headers, fields["SignedHeaders"], request.body, _SECRET_KEY, scope
        )
        assert fields["Signature"] == signed[1], authorization
        assert _JSON_OBJECT.validate_json(request.body) == expected, request.body
        return reply

    return respond


def test_chat_completion_signs_the_native_messages_request_with_the_deployment_aws_keys(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)])
    with wire_server(_sigv4_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock_mantle/{_HAIKU}",
            api_base=wire.url,
            api_key=None,
            aws_access_key_id=_ACCESS_KEY,
            aws_secret_access_key=_SECRET_KEY,
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker)}
        )
        _assert_text_completion(response, model, marker)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_region_prefix_in_a_mantle_claude_id_is_routing_only_and_keeps_the_mantle_price(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/us-east-2/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker)}
        )
        body: Final = _assert_text_completion(response, model, marker)
        cost: Final = _mantle_cost(gateway, _HAIKU)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(cost), response.text
        row: Final = _spend_row(_BY_REQUEST_ID, string_value(body["id"]))
        assert (row["status"], row["model_group"], row["custom_llm_provider"]) == (
            "success",
            model,
            "bedrock_mantle",
        ), row
        assert _number(row["spend"]) == pytest.approx(cost), row
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_openai_sdk_chat_completion_on_a_mantle_claude_id(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        with OpenAI(
            api_key=gateway.key,
            base_url=_sdk_base(gateway),
            max_retries=0,
            http_client=httpx.Client(timeout=30, trust_env=False),
        ) as client:
            completion: Final = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": _prompt(marker)}]
            )
        assert (completion.object, completion.model) == ("chat.completion", model), completion
        assert (completion.choices[0].message.content, completion.choices[0].finish_reason) == (
            _answer(marker),
            "stop",
        ), completion
        assert completion.usage is not None
        assert (completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens) == (
            _USAGE
        ), completion
        _assert_success_row(
            _spend_row(_BY_REQUEST_ID, completion.id), model, _HAIKU, "acompletion", _mantle_cost(gateway, _HAIKU)
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_openai_sdk_chat_stream_on_a_mantle_claude_id(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)], stream=True)
    with wire_server(_bearer_peer(expected, _text_stream(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        with OpenAI(
            api_key=gateway.key,
            base_url=_sdk_base(gateway),
            max_retries=0,
            http_client=httpx.Client(timeout=30, trust_env=False),
        ) as client:
            chunks: Final = tuple(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": _prompt(marker)}],
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
        identities: Final = {chunk.id for chunk in chunks}
        assert len(identities) == 1 and {chunk.model for chunk in chunks} == {model}, chunks
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _answer(marker), chunks
        assert [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices[0].finish_reason] == ["stop"], (
            chunks
        )
        assert [
            (chunk.usage.prompt_tokens, chunk.usage.completion_tokens, chunk.usage.total_tokens)
            for chunk in chunks
            if chunk.usage is not None
        ] == [_USAGE], chunks
        _assert_success_row(
            _spend_row(_BY_REQUEST_ID, next(iter(identities))),
            model,
            _HAIKU,
            "acompletion",
            _mantle_cost(gateway, _HAIKU),
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


async def test_async_openai_sdk_chat_completion_on_a_mantle_claude_id(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        async with AsyncOpenAI(
            api_key=gateway.key,
            base_url=_sdk_base(gateway),
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=30, trust_env=False),
        ) as client:
            completion: Final = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": _prompt(marker)}]
            )
        assert (completion.object, completion.model) == ("chat.completion", model), completion
        assert (completion.choices[0].message.content, completion.choices[0].finish_reason) == (
            _answer(marker),
            "stop",
        ), completion
        assert completion.usage is not None
        assert (completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens) == (
            _USAGE
        ), completion
        _assert_success_row(
            _spend_row(_BY_REQUEST_ID, completion.id), model, _HAIKU, "acompletion", _mantle_cost(gateway, _HAIKU)
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


async def test_async_openai_sdk_chat_stream_on_a_mantle_claude_id(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)], stream=True)
    with wire_server(_bearer_peer(expected, _text_stream(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        async with AsyncOpenAI(
            api_key=gateway.key,
            base_url=_sdk_base(gateway),
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=30, trust_env=False),
        ) as client:
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt(marker)}],
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks: Final = tuple([chunk async for chunk in stream])
        identities: Final = {chunk.id for chunk in chunks}
        assert len(identities) == 1 and {chunk.model for chunk in chunks} == {model}, chunks
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _answer(marker), chunks
        assert [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices[0].finish_reason] == ["stop"], (
            chunks
        )
        assert [
            (chunk.usage.prompt_tokens, chunk.usage.completion_tokens, chunk.usage.total_tokens)
            for chunk in chunks
            if chunk.usage is not None
        ] == [_USAGE], chunks
        _assert_success_row(
            _spend_row(_BY_REQUEST_ID, next(iter(identities))),
            model,
            _HAIKU,
            "acompletion",
            _mantle_cost(gateway, _HAIKU),
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_identical_chat_request_is_served_from_the_response_cache_and_logged_as_a_free_cache_hit(
    gateway: Gateway,
) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        request: Final[dict[str, JsonValue]] = {"model": model, "messages": _chat_messages(marker)}
        first: Final = gateway.request("POST", "/v1/chat/completions", request)
        identity: Final = string_value(_assert_text_completion(first, model, marker)["id"])
        assert "x-litellm-cache-key" not in first.headers, sorted(first.headers)
        second: Final = gateway.request("POST", "/v1/chat/completions", request)
        assert _assert_text_completion(second, model, marker)["id"] == identity, second.text
        assert second.headers["x-litellm-cache-key"], sorted(second.headers)
        rows: Final = eventually(
            lambda: read_rows(_BY_MODEL_GROUP, (model,)), lambda found: len(found) == 2, seconds=70
        )
        priced, cached = rows
        _assert_success_row(priced, model, _HAIKU, "acompletion", _mantle_cost(gateway, _HAIKU))
        assert priced["request_id"] == identity, rows
        assert string_value(cached["request_id"]).startswith(f"{identity}_cache_hit"), rows
        assert (cached["status"], cached["cache_hit"], _number(cached["spend"])) == ("success", "True", 0), rows
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def _open_weight_peer(prompt: str, identity: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", sorted(request.headers)
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": _SAFEGUARD,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "stream": False,
        }, request.body
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": _SAFEGUARD,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "safeguard answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
                }
            ).encode()
        )

    return respond


def test_open_weight_mantle_id_keeps_the_openai_compatible_chat_route_and_its_sampling_params(
    gateway: Gateway,
) -> None:
    marker: Final = uuid4().hex
    identity: Final = f"chatcmpl-safeguard-{marker}"
    with wire_server(_open_weight_peer(_prompt(marker), identity)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_SAFEGUARD}", api_base=wire.url + "/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _chat_messages(marker), "temperature": 0.3, "stream": False},
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        message: Final = object_value(_only_choice(body)["message"])
        assert (body["id"], message["role"], message["content"]) == (
            identity,
            "assistant",
            "safeguard answer",
        ), response.text
        cost: Final = _mantle_cost(gateway, _SAFEGUARD, input_tokens=9, output_tokens=3)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(cost), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/chat/completions")]


@pytest.mark.parametrize(
    "suffix",
    [
        pytest.param("/v1", id="v1"),
        pytest.param("/anthropic/v1/messages", id="anthropic_v1_messages"),
        pytest.param("/openai/v1", id="openai_v1"),
    ],
)
def test_api_base_with_a_route_suffix_still_posts_to_the_native_messages_path(gateway: Gateway, suffix: str) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url + suffix, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker)}
        )
        _assert_text_completion(response, model, marker)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_responses_api_on_a_mantle_claude_id_bridges_to_native_messages(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": _prompt(marker)})
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert (body["object"], body["status"], body["model"]) == ("response", "completed", model), response.text
        output: Final = body["output"]
        assert isinstance(output, list) and len(output) == 1, response.text
        content: Final = object_value(output[0])["content"]
        assert isinstance(content, list) and len(content) == 1, response.text
        part: Final = object_value(content[0])
        assert (part["type"], part["text"]) == ("output_text", _answer(marker)), response.text
        _assert_success_row(
            _spend_row(_BY_CALL_ID, response.headers["x-litellm-call-id"]),
            model,
            _HAIKU,
            "aresponses",
            _mantle_cost(gateway, _HAIKU),
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_responses_api_stream_on_a_mantle_claude_id_bridges_to_the_native_stream(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final = _outbound(_HAIKU, [_user_turn(marker)], stream=True)
    with wire_server(_bearer_peer(expected, _text_stream(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": _prompt(marker), "stream": True}
        )
        assert response.status_code == 200, response.text
        events: Final = _sse_objects(response.text)
        kinds: Final = [event["type"] for event in events]
        assert (kinds[0], kinds[-1]) == ("response.created", "response.completed"), kinds
        assert "".join(
            string_value(event["delta"]) for event in events if event["type"] == "response.output_text.delta"
        ) == _answer(marker), response.text
        assert {event["model"] for event in events} == {model}, response.text
        completed: Final = object_value(events[-1]["response"])
        usage: Final = object_value(completed["usage"])
        assert (completed["status"], usage["input_tokens"], usage["output_tokens"]) == (
            "completed",
            _INPUT_TOKENS,
            _OUTPUT_TOKENS,
        ), response.text
        _assert_success_row(
            _spend_row(_BY_CALL_ID, response.headers["x-litellm-call-id"]),
            model,
            _HAIKU,
            "aresponses",
            _mantle_cost(gateway, _HAIKU),
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def _health_peer(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", _MESSAGES_PATH), request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}", sorted(request.headers)
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert (body["model"], body["anthropic_version"]) == (_HAIKU, _BRIDGE_VERSION), request.body
    return _text_reply(uuid4().hex)


def _health_counts(gateway: Gateway, model: str) -> tuple[int, JsonValue, JsonValue]:
    response: Final = gateway.request("GET", "/health", params={"model": model})
    body: Final = _JSON_OBJECT.validate_json(response.content)
    return response.status_code, body.get("healthy_count"), body.get("unhealthy_count")


def test_health_check_on_a_mantle_claude_deployment_probes_the_native_messages_route(gateway: Gateway) -> None:
    with wire_server(_health_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        counts: Final = eventually(
            lambda: _health_counts(gateway, model),
            lambda found: found[1:] != (0, 0),
            seconds=30,
            return_last_on_timeout=True,
        )
        assert counts == (200, 1, 0), counts
        assert {(request.method, request.target) for request in wire.drain()} == {("POST", _MESSAGES_PATH)}


def _native_messages_peer(expected: Mapping[str, JsonValue], reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _MESSAGES_PATH), request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", sorted(request.headers)
        assert request.headers["anthropic-version"] == "2023-06-01", sorted(request.headers)
        assert _JSON_OBJECT.validate_json(request.body) == expected, request.body
        return reply

    return respond


def test_messages_api_on_a_mantle_claude_id_keeps_forwarding_the_caller_shape(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    expected: Final[dict[str, JsonValue]] = {"messages": _chat_messages(marker), "max_tokens": 64, "model": _HAIKU}
    with wire_server(_native_messages_peer(expected, _text_reply(marker))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 64, "messages": _chat_messages(marker)},
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        assert (body["id"], body["type"], body["model"], body["stop_reason"]) == (
            f"msg_bdrk_{marker}",
            "message",
            model,
            "end_turn",
        ), response.text
        assert body["content"] == [{"type": "text", "text": _answer(marker)}], response.text
        _assert_success_row(
            _spend_row(_BY_REQUEST_ID, f"msg_bdrk_{marker}"),
            model,
            _HAIKU,
            "anthropic_messages",
            _mantle_cost(gateway, _HAIKU),
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


_UNSUPPORTED: Final = (
    pytest.param(None, {"n": 2}, {}, "does not support parameters: ['n']", id="n"),
    pytest.param(
        "supports_sampling_params",
        {"temperature": 0.2},
        {},
        "Only temperature=1 is supported",
        id="temperature_on_a_fixed_sampling_model",
    ),
    pytest.param(
        "supports_forced_tool_use",
        {"tools": [_WEATHER_TOOL], "tool_choice": "required"},
        {"tools": [_WEATHER_TOOL_OUTBOUND], "tool_choice": {"type": "auto"}},
        "does not support forced tool use",
        id="forced_tool_use_on_a_model_without_it",
    ),
)


def _backend_without(gateway: Gateway, flag: str | None) -> str:
    return _HAIKU if flag is None else _mantle_claude_where(gateway, flag, disabled=True)


@pytest.mark.parametrize(("flag", "request_fields", "outbound_fields", "reason"), _UNSUPPORTED)
def test_unsupported_claude_param_is_a_400_that_never_reaches_mantle(
    gateway: Gateway,
    flag: str | None,
    request_fields: dict[str, JsonValue],
    outbound_fields: dict[str, JsonValue],
    reason: str,
) -> None:
    marker: Final = uuid4().hex
    backend: Final = _backend_without(gateway, flag)
    with wire_server(lambda request: _text_reply(_marker_of(request), backend)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{backend}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker), **request_fields}
        )
        assert response.status_code == 400, response.text
        message: Final = _error_message(response)
        assert "UnsupportedParamsError" in message and reason in message, response.text
        assert wire.drain() == ()


@pytest.mark.parametrize(("flag", "request_fields", "outbound_fields", "reason"), _UNSUPPORTED)
def test_unsupported_claude_param_is_dropped_under_drop_params(
    gateway: Gateway,
    flag: str | None,
    request_fields: dict[str, JsonValue],
    outbound_fields: dict[str, JsonValue],
    reason: str,
) -> None:
    marker: Final = uuid4().hex
    backend: Final = _backend_without(gateway, flag)
    expected: Final = _outbound(backend, [_user_turn(marker)], **outbound_fields)
    with wire_server(_bearer_peer(expected, _text_reply(marker, backend))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{backend}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": _chat_messages(marker), "drop_params": True, **request_fields},
        )
        _assert_text_completion(response, model, marker)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def _error_reply(status: int, kind: str, message: str) -> Reply:
    return Reply(
        status=status, body=json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()
    )


def _failing_peer(failing: str, error: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _MESSAGES_PATH), request.target
        marker: Final = _marker_of(request)
        return error if marker == failing else _text_reply(marker)

    return respond


@pytest.mark.parametrize(
    ("upstream", "status", "error_class"),
    [
        pytest.param(
            _error_reply(400, "invalid_request_error", "messages: synthetic bad request"),
            400,
            "BadRequestError",
            id="bad_request",
        ),
        pytest.param(
            _error_reply(401, "authentication_error", "synthetic invalid bearer"),
            401,
            "AuthenticationError",
            id="unauthorized",
        ),
        pytest.param(
            _error_reply(429, "rate_limit_error", "synthetic throttle"), 429, "RateLimitError", id="rate_limited"
        ),
        pytest.param(
            _error_reply(500, "api_error", "synthetic upstream fault"),
            503,
            "ServiceUnavailableError",
            id="upstream_fault",
        ),
        pytest.param(
            _error_reply(400, "invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum"),
            400,
            "ContextWindowExceededError",
            id="context_overflow",
        ),
    ],
)
def test_mantle_error_on_the_native_route_reaches_the_caller_and_the_deployment_keeps_serving(
    gateway: Gateway, upstream: Reply, status: int, error_class: str
) -> None:
    failing: Final = uuid4().hex
    following: Final = uuid4().hex
    upstream_message: Final = string_value(object_value(_JSON_OBJECT.validate_json(upstream.body)["error"])["message"])
    with wire_server(_failing_peer(failing, upstream)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_HAIKU}", api_base=wire.url, api_key=_API_KEY)
        failed: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(failing)}
        )
        assert failed.status_code == status, failed.text
        message: Final = _error_message(failed)
        assert error_class in message and upstream_message in message, failed.text
        row: Final = _spend_row(_BY_CALL_ID, failed.headers["x-litellm-call-id"])
        assert (row["status"], row["model_group"], _number(row["spend"])) == ("failure", model, 0), row
        served: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(following)}
        )
        _assert_text_completion(served, model, following)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)] * 2


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(123, id="int"),
        pytest.param([f"bedrock_mantle/{_HAIKU}"], id="list"),
        pytest.param("", id="empty_string"),
    ],
)
def test_malformed_model_field_is_a_400_with_an_error_body_and_the_proxy_keeps_serving(
    gateway: Gateway, model: JsonValue
) -> None:
    response: Final = gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(uuid4().hex)}
    )
    assert response.status_code == 400, response.text
    assert "model" in _error_message(response).lower(), response.text
    assert gateway.request("GET", "/health/liveliness").status_code == 200


def _wildcard_prefix(gateway: Gateway, scenario: Scenario, api_base: str) -> str:
    prefix: Final = f"integration-{uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": f"{prefix}/*",
            "litellm_params": {"model": "bedrock_mantle/*", "api_base": api_base, "api_key": _API_KEY},
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return prefix


def test_uppercase_claude_id_through_a_mantle_wildcard_is_sent_to_native_messages(gateway: Gateway) -> None:
    marker: Final = uuid4().hex
    backend: Final = _HAIKU.upper()
    expected: Final = _outbound(backend, [_user_turn(marker)])
    with wire_server(_bearer_peer(expected, _text_reply(marker, backend))) as wire, gateway.scenario() as scenario:
        model: Final = f"{_wildcard_prefix(gateway, scenario, wire.url)}/{backend}"
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker)}
        )
        _assert_text_completion(response, model, marker)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


def test_five_kilobyte_claude_id_through_a_mantle_wildcard_reaches_mantle_and_its_404_reaches_the_caller(
    gateway: Gateway,
) -> None:
    marker: Final = uuid4().hex
    backend: Final = f"{_HAIKU}-{'x' * 5000}"
    expected: Final = _outbound(backend, [_user_turn(marker)])
    unknown: Final = _error_reply(404, "not_found_error", "model: synthetic unknown model")
    with wire_server(_bearer_peer(expected, unknown)) as wire, gateway.scenario() as scenario:
        model: Final = f"{_wildcard_prefix(gateway, scenario, wire.url)}/{backend}"
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker)}
        )
        assert response.status_code == 404, response.text[:600]
        message: Final = _error_message(response)
        assert "NotFoundError" in message and "model: synthetic unknown model" in message, message[:600]
        row: Final = _spend_row(_BY_CALL_ID, response.headers["x-litellm-call-id"])
        assert (row["status"], row["model_group"]) == ("failure", model), row["status"]
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _MESSAGES_PATH)]


_LUNA: Final = "openai.gpt-6-luna"
_RESPONSES_PATH: Final = "/openai/v1/responses"


def _responses_bridge_peer(marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", _RESPONSES_PATH), request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", sorted(request.headers)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _LUNA, body
        assert _prompt(marker) in json.dumps(body["input"]), body
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp-{marker}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": _LUNA,
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg-{marker}",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": _answer(marker), "annotations": []}],
                        }
                    ],
                    "usage": {
                        "input_tokens": _INPUT_TOKENS,
                        "output_tokens": _OUTPUT_TOKENS,
                        "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
                    },
                }
            ).encode()
        )

    return respond


def test_non_claude_mantle_id_with_a_bare_bedrock_twin_keeps_its_route_and_spends_at_the_mantle_price(
    gateway: Gateway,
) -> None:
    marker: Final = uuid4().hex
    with wire_server(_responses_bridge_peer(marker)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"bedrock_mantle/{_LUNA}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": _chat_messages(marker)}
        )
        assert response.status_code == 200, response.text
        body: Final = _JSON_OBJECT.validate_json(response.content)
        message: Final = object_value(_only_choice(body)["message"])
        assert (body["model"], message["role"], message["content"]) == (model, "assistant", _answer(marker)), (
            response.text
        )
        assert _token_usage(body) == _USAGE, response.text
        cost: Final = _mantle_cost(gateway, _LUNA)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(cost), response.text
        _assert_success_row(
            _spend_row(_BY_CALL_ID, response.headers["x-litellm-call-id"]), model, _LUNA, "responses", cost
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", _RESPONSES_PATH)]
