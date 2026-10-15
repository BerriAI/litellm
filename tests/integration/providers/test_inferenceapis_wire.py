import json
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import anthropic
import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "deepseek-ai/DeepSeek-V4-Flash"
_MODEL: Final = f"inferenceapis/{_BACKEND}"
_API_KEY: Final = "synthetic-inferenceapis-key"
_OPENAI_KEY: Final = "synthetic-openai-key"
_OVERFLOW_TEXT: Final = "Input too long: 150055 input tokens, limit is 131072 for this model"
_PARAM_ERROR_TEXT: Final = (
    "Input should be 'error' or 'truncate', field: 'context_length_exceeded_behavior', value: 'foo'"
)
_WEATHER_TOOL: Final[JsonValue] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_COST_MAP_PATH: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, JsonValue]])
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])


def _identity(label: str) -> str:
    return f"inferenceapis-{label}-{uuid.uuid4().hex}"


def _catalog_cost(field: str) -> float:
    value: Final = _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())[_MODEL][field]
    assert isinstance(value, (int, float))
    return float(value)


def _approx(value: float) -> object:
    return pytest.approx(value, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]  # pytest lacks typed approx stubs


def _sse(events: tuple[Mapping[str, JsonValue], ...], named: bool) -> tuple[bytes, ...]:
    return tuple(
        (f"event: {event['type']}\n" if named else "").encode() + f"data: {json.dumps(event)}\n\n".encode()
        for event in events
    )


def _chat_completion(identity: str, message: Mapping[str, JsonValue], finish: str = "stop") -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": _BACKEND,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 23, "completion_tokens": 41, "total_tokens": 64},
            }
        ).encode()
    )


def _chat_frames(identity: str, pieces: tuple[str, ...]) -> Reply:
    chunk: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _BACKEND,
    }
    deltas: Final[tuple[dict[str, JsonValue], ...]] = tuple(
        {**chunk, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]} for piece in pieces
    )
    finish: Final[dict[str, JsonValue]] = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return Reply(
        content_type="text/event-stream",
        chunks=(*_sse((*deltas, finish), named=False), b"data: [DONE]\n\n"),
    )


def _response_object(identity: str, text: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": _BACKEND,
        "output": [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        if status == "completed"
        else [],
        "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
    }


def _responses_reply(identity: str, text: str, stream: bool) -> Reply:
    completed: Final = _response_object(identity, text, "completed")
    if not stream:
        return Reply(body=json.dumps(completed).encode())
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"type": "response.created", "sequence_number": 0, "response": _response_object(identity, text, "in_progress")},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.completed", "sequence_number": 2, "response": completed},
    )
    return Reply(content_type="text/event-stream", chunks=_sse(events, named=True))


def _message(identity: str, text: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": _BACKEND,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }


def _messages_reply(identity: str, text: str, stream: bool) -> Reply:
    if not stream:
        return Reply(body=json.dumps(_message(identity, text)).encode())
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {"type": "message_start", "message": {**_message(identity, ""), "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    )
    return Reply(content_type="text/event-stream", chunks=_sse(events, named=True))


def _error(status: int, message: str, code: str | None, kind: str = "invalid_request_error") -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message, "type": kind, "code": code}}).encode())


def _body(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _assert_vendor_request(request: Request, target: str) -> dict[str, JsonValue]:
    assert (request.method, request.target) == ("POST", target), request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    body: Final = _body(request)
    assert body["model"] == _BACKEND
    return body


def _chat(model: str, prompt: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], **extra}


def _messages(model: str, prompt: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": prompt}], **extra}


def _responses(model: str, prompt: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": prompt, **extra}


def _consumed(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[httpx.Response, str]:
    with gateway.client.stream("POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}) as response:
        text: Final = b"".join(response.iter_bytes()).decode()
    return response, text


def _data_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _targets(wire: Wire) -> list[tuple[str, str]]:
    return [(request.method, request.target) for request in wire.drain()]


def test_chat_completion_reaches_the_vendor_and_bills_the_catalog_price(gateway: Gateway) -> None:
    identity: Final = _identity("chat")
    prompt: Final = f"chat {identity}"

    def respond(request: Request) -> Reply:
        body: Final = _assert_vendor_request(request, "/v1/chat/completions")
        assert body["messages"] == [{"role": "user", "content": prompt}]
        return _chat_completion(identity, {"role": "assistant", "content": "43"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        choice: Final = object_value(_JSON_LIST.validate_python(payload["choices"])[0])
        assert object_value(choice["message"])["content"] == "43"
        expected_cost: Final = 23 * _catalog_cost("input_cost_per_token") + 41 * _catalog_cost("output_cost_per_token")
        assert float(response.headers["x-litellm-response-cost"]) == _approx(expected_cost)
        assert _targets(wire) == [("POST", "/v1/chat/completions")]
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


def test_streamed_chat_completion_delivers_every_delta_and_one_done(gateway: Gateway) -> None:
    identity: Final = _identity("chat-stream")
    pieces: Final = ("1\n2\n", "3\n4\n", "5")

    def respond(request: Request) -> Reply:
        assert _assert_vendor_request(request, "/v1/chat/completions")["stream"] is True
        return _chat_frames(identity, pieces)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/chat/completions", _chat(model, f"count {identity}", stream=True))
        assert response.status_code == 200, text
        assert text.count("data: [DONE]") == 1
        events: Final = _data_events(text)
        assert {str(event["id"]) for event in events} == {identity}
        deltas: Final = tuple(
            object_value(object_value(choice)["delta"]).get("content") or ""
            for event in events
            for choice in _JSON_LIST.validate_python(event["choices"])
        )
        assert "".join(str(delta) for delta in deltas) == "".join(pieces)
        assert _targets(wire) == [("POST", "/v1/chat/completions")]


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_responses_call_goes_to_the_vendor_responses_endpoint(gateway: Gateway, stream: bool) -> None:
    identity: Final = _identity("responses")
    answer: Final = f"answer {identity}"
    prompt: Final = f"responses {identity}"

    def respond(request: Request) -> Reply:
        body: Final = _assert_vendor_request(request, "/v1/responses")
        assert body["input"] == prompt
        assert bool(body.get("stream")) is stream
        return _responses_reply(identity, answer, stream)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/responses", _responses(model, prompt, stream=stream))
        assert response.status_code == 200, text
        completed: Final = (
            _JSON_OBJECT.validate_json(text)
            if not stream
            else object_value(
                next(event for event in _data_events(text) if event["type"] == "response.completed")["response"]
            )
        )
        assert completed["status"] == "completed"
        assert answer in json.dumps(completed["output"])
        assert _targets(wire) == [("POST", "/v1/responses")]


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_messages_call_goes_to_the_vendor_messages_endpoint_untranslated(gateway: Gateway, stream: bool) -> None:
    identity: Final = _identity("messages")
    answer: Final = f"answer {identity}"
    prompt: Final = f"messages {identity}"

    def respond(request: Request) -> Reply:
        body: Final = _assert_vendor_request(request, "/v1/messages")
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert body["max_tokens"] == 32
        assert "anthropic-version" in request.headers
        assert bool(body.get("stream")) is stream
        return _messages_reply(identity, answer, stream)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/messages", _messages(model, prompt, stream=stream))
        assert response.status_code == 200, text
        if stream:
            events: Final = _data_events(text)
            assert [event["type"] for event in events][-1] == "message_stop"
            assert object_value(events[0]["message"])["id"] == identity
            assert answer in text
        else:
            payload: Final = _JSON_OBJECT.validate_json(text)
            assert payload["id"] == identity
            assert payload["content"] == [{"type": "text", "text": answer}]
        assert _targets(wire) == [("POST", "/v1/messages")]


def test_chat_tool_call_round_trips(gateway: Gateway) -> None:
    identity: Final = _identity("tool")
    tool_call: Final[JsonValue] = {
        "id": "call_paris",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
    }

    def respond(request: Request) -> Reply:
        body: Final = _assert_vendor_request(request, "/v1/chat/completions")
        assert body["tools"] == [_WEATHER_TOOL]
        assert body["tool_choice"] == "auto"
        message: Final[dict[str, JsonValue]] = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        return _chat_completion(identity, message, "tool_calls")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat(model, "weather in Paris", tools=[_WEATHER_TOOL], tool_choice="auto"),
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["id"] == identity
        choice: Final = object_value(_JSON_LIST.validate_python(payload["choices"])[0])
        assert choice["finish_reason"] == "tool_calls"
        assert object_value(choice["message"])["tool_calls"] == [tool_call]
        assert _targets(wire) == [("POST", "/v1/chat/completions")]


def test_non_openai_params_reach_the_vendor_body_top_level(gateway: Gateway) -> None:
    identity: Final = _identity("extra")

    def respond(request: Request) -> Reply:
        body: Final = _assert_vendor_request(request, "/v1/chat/completions")
        assert body["top_k"] == 5
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "extra_body" not in body
        return _chat_completion(identity, {"role": "assistant", "content": "ok"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat(model, "extra params", top_k=5, chat_template_kwargs={"enable_thinking": False}),
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["id"] == identity
        assert _targets(wire) == [("POST", "/v1/chat/completions")]


_OVERFLOW_CASES: Final = (
    ("/v1/chat/completions", "/v1/chat/completions", "context_length_exceeded"),
    ("/v1/chat/completions", "/v1/chat/completions", "upstream_rejected"),
    ("/v1/messages", "/v1/messages", "upstream_rejected"),
    ("/v1/responses", "/v1/responses", "upstream_rejected"),
)
_REQUEST_BUILDERS: Final[Mapping[str, Callable[[str, str], dict[str, JsonValue]]]] = {
    "/v1/chat/completions": _chat,
    "/v1/messages": _messages,
    "/v1/responses": _responses,
}


@pytest.mark.parametrize(("route", "target", "code"), _OVERFLOW_CASES)
def test_vendor_overflow_reaches_the_caller_as_a_context_window_error(
    gateway: Gateway, route: str, target: str, code: str
) -> None:
    prompt: Final = f"overflow {uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        _assert_vendor_request(request, target)
        return _error(400, _OVERFLOW_TEXT, code)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request("POST", route, _REQUEST_BUILDERS[route](model, prompt))
        assert response.status_code == 400, response.text
        assert "ContextWindowExceededError" in response.text, response.text
        assert _OVERFLOW_TEXT in response.text, response.text
        assert _targets(wire) == [("POST", target)]


@pytest.mark.parametrize(
    ("status", "message", "code", "kind", "caller_status", "caller_class"),
    (
        (400, _PARAM_ERROR_TEXT, None, "invalid_request_error", 400, "BadRequestError"),
        (401, "Invalid API key", "invalid_api_key", "authentication_error", 401, "AuthenticationError"),
    ),
    ids=("param-error-naming-the-overflow-field", "bad-key"),
)
def test_vendor_errors_that_are_not_overflow_keep_their_class(
    gateway: Gateway, status: int, message: str, code: str | None, kind: str, caller_status: int, caller_class: str
) -> None:
    def respond(request: Request) -> Reply:
        _assert_vendor_request(request, "/v1/chat/completions")
        return _error(status, message, code, kind)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, f"error {uuid.uuid4().hex}"))
        assert response.status_code == caller_status, response.text
        assert caller_class in response.text, response.text
        assert "ContextWindowExceededError" not in response.text, response.text
        assert _targets(wire) == [("POST", "/v1/chat/completions")]


def _echoing(_: Request) -> Reply:
    return _chat_completion(_identity("echo"), {"role": "assistant", "content": "ok"})


def _forwarded_top_k(wire: Wire) -> list[JsonValue]:
    return [_body(request).get("top_k") for request in wire.drain()]


@pytest.mark.parametrize(
    "value",
    (5, [1, 2], "", "x" * 5120),
    ids=("int", "list", "empty-string", "five-kb"),
)
def test_a_non_openai_param_of_any_shape_forwards_like_it_does_for_an_openai_deployment(
    gateway: Gateway, value: JsonValue
) -> None:
    with (
        wire_server(_echoing) as vendor,
        wire_server(_echoing) as openai_peer,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_MODEL, api_base=f"{vendor.url}/v1", api_key=_API_KEY)
        control: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{openai_peer.url}/v1", api_key=_OPENAI_KEY
        )
        responses: Final = tuple(
            gateway.request("POST", "/v1/chat/completions", _chat(name, "shape", top_k=value))
            for name in (model, control)
        )
        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        assert _forwarded_top_k(vendor) == [value]
        assert _forwarded_top_k(openai_peer) == [value]


def test_a_param_sent_twice_forwards_the_last_value_like_an_openai_deployment(gateway: Gateway) -> None:
    with (
        wire_server(_echoing) as vendor,
        wire_server(_echoing) as openai_peer,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_MODEL, api_base=f"{vendor.url}/v1", api_key=_API_KEY)
        control: Final = scenario.model(
            model="openai/gpt-4o-mini", api_base=f"{openai_peer.url}/v1", api_key=_OPENAI_KEY
        )
        responses: Final = tuple(
            gateway.client.post(
                "/v1/chat/completions",
                content=(
                    f'{{"model": "{name}", "messages": [{{"role": "user", "content": "twice"}}], '
                    '"top_k": 1, "top_k": 2}'
                ).encode(),
                headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
            )
            for name in (model, control)
        )
        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        assert _forwarded_top_k(vendor) == [2]
        assert _forwarded_top_k(openai_peer) == [2]


def test_messages_on_a_json_provider_without_native_messages_forwards_top_k(gateway: Gateway) -> None:
    identity: Final = _identity("bridge")

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request.target
        assert _body(request)["top_k"] == 5
        return _chat_completion(identity, {"role": "assistant", "content": "bridged"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="veniceai/venice-uncensored", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", _messages(model, "bridge", top_k=5))
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["content"] == [{"type": "text", "text": "bridged"}]
        assert _targets(wire) == [("POST", "/v1/chat/completions")]


def _lists_inferenceapis(endpoint: Mapping[str, JsonValue]) -> bool:
    providers: Final = _JSON_LIST.validate_python(endpoint["providers"])
    return any(object_value(provider)["slug"] == "inferenceapis" for provider in providers)


def test_public_routes_list_the_provider_with_its_endpoints_and_fields(gateway: Gateway) -> None:
    providers: Final = gateway.request("GET", "/public/providers")
    assert providers.status_code == 200, providers.text
    assert "inferenceapis" in _JSON_LIST.validate_json(providers.content)

    fields_response: Final = gateway.request("GET", "/public/providers/fields")
    assert fields_response.status_code == 200, fields_response.text
    entries: Final = tuple(
        object_value(entry)
        for entry in _JSON_LIST.validate_json(fields_response.content)
        if object_value(entry)["litellm_provider"] == "inferenceapis"
    )
    assert len(entries) == 1
    required: Final = {
        object_value(field)["key"]: object_value(field)["required"]
        for field in _JSON_LIST.validate_python(entries[0]["credential_fields"])
    }
    assert required == {"api_base": False, "api_key": True}

    endpoints_response: Final = gateway.request("GET", "/public/endpoints")
    assert endpoints_response.status_code == 200, endpoints_response.text
    endpoints: Final = tuple(
        object_value(endpoint)
        for endpoint in _JSON_LIST.validate_python(_JSON_OBJECT.validate_json(endpoints_response.content)["endpoints"])
    )
    listed: Final = frozenset(str(endpoint["key"]) for endpoint in endpoints if _lists_inferenceapis(endpoint))
    assert listed == {"chat_completions", "messages", "responses"}


def test_openai_and_anthropic_sdk_clients_reach_the_vendor(gateway: Gateway) -> None:
    chat_identity: Final = _identity("sdk-chat")
    message_identity: Final = _identity("sdk-messages")

    def respond(request: Request) -> Reply:
        if request.target == "/v1/messages":
            _assert_vendor_request(request, "/v1/messages")
            return _messages_reply(message_identity, "from messages", stream=False)
        _assert_vendor_request(request, "/v1/chat/completions")
        return _chat_completion(chat_identity, {"role": "assistant", "content": "from chat"})

    base_url: Final = str(gateway.client.base_url).rstrip("/")
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=f"{wire.url}/v1", api_key=_API_KEY)
        openai_client: Final = OpenAI(
            api_key=gateway.key,
            base_url=f"{base_url}/v1",
            max_retries=0,
            http_client=httpx.Client(timeout=30, trust_env=False),
        )
        completion: Final = openai_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "sdk chat"}]
        )
        assert completion.id == chat_identity
        assert completion.choices[0].message.content == "from chat"
        anthropic_client: Final = anthropic.Anthropic(
            base_url=base_url, api_key=gateway.key, auth_token=gateway.key, max_retries=0
        )
        message: Final = anthropic_client.messages.create(
            model=model, max_tokens=32, messages=[{"role": "user", "content": "sdk messages"}]
        )
        assert message.id == message_identity
        assert len(message.content) == 1
        block: Final = message.content[0]
        assert isinstance(block, anthropic.types.TextBlock)
        assert block.text == "from messages"
        assert sorted(_targets(wire)) == [("POST", "/v1/chat/completions"), ("POST", "/v1/messages")]
