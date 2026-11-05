import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.responses_stream import (
    AZURE_TARGET,
    OPENAI_TARGET,
    RATE_LIMIT_MESSAGE,
    RATE_LIMIT_PREFIX,
    SENTINEL_PREFIX,
    assert_provider_typed_rate_limit,
    chat_body,
    chat_content,
    created,
    data_frames,
    delta,
    error_body,
    frame,
    function_tools,
    healthy_stream,
    messages_body,
    openai_error_event,
    openai_error_stream,
    serve,
    spend_row,
    sse_events,
)
from integration._support.wire import Wire, wire_server
from openai.types.responses import ResponseFailedEvent, ResponseStreamEvent
from pydantic import JsonValue

_AZURE_KEY: Final = "synthetic-azure-key"
_QUOTA_MESSAGE: Final = "You exceeded your current quota, please check your plan and billing details."
_SERVER_MESSAGE: Final = "The server had an error while processing your request. Sorry about that!"
_ANTHROPIC_HEADERS: Final = {"anthropic-version": "2023-06-01"}


@dataclass(frozen=True, slots=True)
class _Served:
    response: httpx.Response
    wire: Wire


def _azure_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model="azure/gpt-6", api_base=wire.url, api_key=_AZURE_KEY)


def _post(gateway: Gateway, path: str, body: Mapping[str, JsonValue], stream: tuple[bytes, ...]) -> _Served:
    with wire_server(serve(stream, AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = _azure_model(scenario, wire)
        response: Final = gateway.request("POST", path, {**body, "model": model}, headers=_ANTHROPIC_HEADERS)
        return _Served(response, wire)


def _chat(gateway: Gateway, stream: tuple[bytes, ...]) -> tuple[httpx.Response, str]:
    marker: Final = uuid.uuid4().hex
    served: Final = _post(gateway, "/v1/chat/completions", chat_body("", marker), stream)
    _assert_one_attempt(served.wire, marker)
    return served.response, marker


def _assert_one_attempt(wire: Wire, marker: str) -> None:
    received: Final = wire.drain()
    assert len(received) == 1 and marker.encode() in received[0].body, [request.target for request in received]


def _assert_failure_logged(call_id: str, logged_code: str, provider_message: str) -> None:
    row: Final = spend_row(call_id)
    assert row["status"] == "failure", row
    error: Final = object_value(object_value(row["metadata"])["error_information"])
    assert str(error["error_code"]) == logged_code, error
    assert provider_message in string_value(error["error_message"]), error


def _identity() -> str:
    return "resp_" + uuid.uuid4().hex


def test_openai_sdk_chat_stream_top_level_rate_limit_is_a_throttling_error_with_the_provider_message(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = openai_error_stream(_identity(), "rate_limit_exceeded", RATE_LIMIT_MESSAGE)
    with wire_server(serve(stream, AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = _azure_model(scenario, wire)
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        with pytest.raises(openai.RateLimitError) as raised:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                tools=function_tools(),
                stream=True,
                extra_body={"num_retries": 0, "cache": {"no-cache": True}},
            )
        assert raised.value.status_code == 429
        assert_provider_typed_rate_limit(object_value(raised.value.body))
        _assert_one_attempt(wire, marker)
        _assert_failure_logged(raised.value.response.headers["x-litellm-call-id"], "429", RATE_LIMIT_MESSAGE)


async def test_async_openai_sdk_codex_chat_stream_top_level_rate_limit_is_a_throttling_error(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = openai_error_stream(_identity(), "rate_limit_exceeded", RATE_LIMIT_MESSAGE)
    with wire_server(serve(stream, OPENAI_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-5.3-codex", api_base=wire.url, api_key="synthetic-openai-key")
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        with pytest.raises(openai.RateLimitError) as raised:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                stream=True,
                extra_body={"num_retries": 0, "cache": {"no-cache": True}},
            )
        assert raised.value.status_code == 429
        assert_provider_typed_rate_limit(object_value(raised.value.body))
        _assert_one_attempt(wire, marker)


@pytest.mark.parametrize(
    ("code", "message", "status"),
    [
        pytest.param("insufficient_quota", _QUOTA_MESSAGE, 429, id="insufficient_quota"),
        pytest.param("server_error", _SERVER_MESSAGE, 500, id="server_error"),
        pytest.param("invalid_prompt", "Invalid prompt: your prompt was flagged.", 400, id="invalid_prompt"),
        pytest.param(
            "context_length_exceeded",
            "Your input exceeds the context window of this model.",
            400,
            id="context_length_exceeded",
        ),
        pytest.param("503", "The engine is currently overloaded.", 503, id="decimal_503"),
        pytest.param("brand_new_error_code", "Something new went wrong upstream.", 500, id="unknown_code"),
        pytest.param("", "An error with an empty code.", 500, id="empty_code"),
    ],
)
def test_top_level_error_code_sets_the_chat_status_and_keeps_the_provider_message(
    gateway: Gateway, code: str, message: str, status: int
) -> None:
    response, _ = _chat(gateway, openai_error_stream(_identity(), code, message))
    assert response.status_code == status, response.text
    error: Final = error_body(response)
    assert str(error["code"]) == str(status), error
    client_message: Final = string_value(error["message"])
    assert message in client_message and SENTINEL_PREFIX not in client_message, client_message
    logged_code: Final = code if status == 400 else str(status)
    _assert_failure_logged(response.headers["x-litellm-call-id"], logged_code, message)


def test_top_level_rate_limit_after_text_is_a_throttling_error_frame_behind_the_text(gateway: Gateway) -> None:
    identity: Final = _identity()
    stream: Final = (
        frame(created(identity)),
        frame(delta(identity, "Hello")),
        frame(delta(identity, " there")),
        frame(openai_error_event("rate_limit_exceeded", RATE_LIMIT_MESSAGE)),
    )
    response, _ = _chat(gateway, stream)
    assert response.status_code == 200, response.text
    frames: Final = data_frames(response.text)
    assert chat_content(frames) == "Hello there", response.text
    assert_provider_typed_rate_limit(object_value(frames[-1]["error"]))


def test_nested_error_object_wins_over_top_level_fields(gateway: Gateway) -> None:
    event: Final = {
        **openai_error_event("rate_limit_exceeded", "top-level says slow down"),
        "error": {"type": "server_error", "code": "server_error", "message": "nested says the server broke"},
    }
    response, _ = _chat(gateway, (frame(created(_identity())), frame(event)))
    assert response.status_code == 500, response.text
    message: Final = string_value(error_body(response)["message"])
    assert "nested says the server broke" in message and "top-level says" not in message, message


def _raw_event(body: str) -> bytes:
    return f"event: error\ndata: {body}\n\n".encode()


_FIVE_KB_MESSAGE: Final = "Rate limit reached for gpt-6. " + "x" * 5120


@pytest.mark.parametrize(
    ("event", "provider_message"),
    [
        pytest.param(
            frame({**openai_error_event("rate_limit_exceeded", RATE_LIMIT_MESSAGE), "error": None}),
            RATE_LIMIT_MESSAGE,
            id="error_null",
        ),
        pytest.param(
            frame({"type": "error", "sequence_number": 1, "code": 429, "message": RATE_LIMIT_MESSAGE, "param": None}),
            RATE_LIMIT_MESSAGE,
            id="integer_code",
        ),
        pytest.param(frame(openai_error_event("rate_limit_exceeded", "")), "", id="empty_message"),
        pytest.param(
            frame({"type": "error", "sequence_number": 1, "code": "rate_limit_exceeded", "message": {"text": "x"}}),
            "",
            id="object_message",
        ),
        pytest.param(frame(openai_error_event("rate_limit_exceeded", _FIVE_KB_MESSAGE)), _FIVE_KB_MESSAGE, id="5kb"),
        pytest.param(
            _raw_event(
                '{"type": "error", "sequence_number": 1, "code": "rate_limit_exceeded", '
                f'"code": "rate_limit_exceeded", "message": {json.dumps(RATE_LIMIT_MESSAGE)}, "param": null}}'
            ),
            RATE_LIMIT_MESSAGE,
            id="duplicated_code",
        ),
    ],
)
def test_malformed_top_level_fields_still_answer_a_throttling_error(
    gateway: Gateway, event: bytes, provider_message: str
) -> None:
    response, _ = _chat(gateway, (frame(created(_identity())), event))
    assert response.status_code == 429, response.text
    assert_provider_typed_rate_limit(error_body(response), provider_message)
    healthy: Final = _post(
        gateway, "/v1/chat/completions", chat_body("", "after"), healthy_stream(_identity(), "still serving")
    )
    assert chat_content(data_frames(healthy.response.text)) == "still serving", healthy.response.text


def _cached_chat(gateway: Gateway, stream: tuple[bytes, ...]) -> tuple[tuple[httpx.Response, ...], int]:
    marker: Final = uuid.uuid4().hex
    body: Final = {key: value for key, value in chat_body("", marker).items() if key != "cache"}
    with wire_server(serve(stream, AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = _azure_model(scenario, wire)
        responses: Final = tuple(
            gateway.request("POST", "/v1/chat/completions", {**body, "model": model}) for _ in range(2)
        )
        return responses, len(wire.drain())


def test_a_top_level_error_stream_is_never_served_from_the_response_cache(gateway: Gateway) -> None:
    responses, attempts = _cached_chat(
        gateway, openai_error_stream(_identity(), "rate_limit_exceeded", RATE_LIMIT_MESSAGE)
    )
    assert attempts == 2, attempts
    for response in responses:
        assert response.status_code == 429, response.text
        assert_provider_typed_rate_limit(error_body(response))


def _responses_frames(gateway: Gateway, stream: tuple[bytes, ...]) -> tuple[httpx.Response, Mapping[str, JsonValue]]:
    marker: Final = uuid.uuid4().hex
    served: Final = _post(
        gateway, "/v1/responses", {"input": marker, "stream": True, "cache": {"no-cache": True}}, stream
    )
    _assert_one_attempt(served.wire, marker)
    assert served.response.status_code == 200, served.response.text
    assert served.response.text.rstrip().endswith("data: [DONE]"), served.response.text
    frames: Final = data_frames(served.response.text)
    assert frames[-1]["type"] == "response.failed", served.response.text
    return served.response, object_value(object_value(frames[-1]["response"])["error"])


def _assert_rate_limit_failure(error: Mapping[str, JsonValue]) -> None:
    assert error["code"] == "rate_limit_exceeded", error
    assert RATE_LIMIT_MESSAGE in string_value(error["message"]), error


def test_native_responses_stream_fails_with_the_provider_code_and_message(gateway: Gateway) -> None:
    identity: Final = _identity()
    response, error = _responses_frames(
        gateway, openai_error_stream(identity, "rate_limit_exceeded", RATE_LIMIT_MESSAGE)
    )
    assert [frame["type"] for frame in data_frames(response.text)] == ["response.created", "response.failed"]
    _assert_rate_limit_failure(error)
    _assert_failure_logged(response.headers["x-litellm-call-id"], "429", RATE_LIMIT_MESSAGE)


def test_native_responses_stream_error_after_text_fails_with_the_provider_code(gateway: Gateway) -> None:
    identity: Final = _identity()
    stream: Final = (
        frame(created(identity)),
        frame(delta(identity, "partial")),
        frame(openai_error_event("rate_limit_exceeded", RATE_LIMIT_MESSAGE)),
    )
    response, error = _responses_frames(gateway, stream)
    kinds: Final = [frame["type"] for frame in data_frames(response.text)]
    assert kinds == ["response.created", "response.output_text.delta", "response.failed"], kinds
    _assert_rate_limit_failure(error)


def _failed_error(events: list[ResponseStreamEvent]) -> tuple[str, str]:
    assert [event.type for event in events] == ["response.created", "response.failed"], events
    failed: Final = events[-1]
    assert isinstance(failed, ResponseFailedEvent) and failed.response.error is not None, failed
    return failed.response.error.code, failed.response.error.message


def test_openai_sdk_responses_stream_fails_with_insufficient_quota(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = openai_error_stream(_identity(), "insufficient_quota", _QUOTA_MESSAGE)
    with wire_server(serve(stream, OPENAI_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-5.3-codex", api_base=wire.url, api_key="synthetic-openai-key")
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        events: Final = list(
            client.responses.create(model=model, input=marker, stream=True, extra_body={"cache": {"no-cache": True}})
        )
        code, message = _failed_error(events)
        assert code == "insufficient_quota" and _QUOTA_MESSAGE in message, events
        _assert_one_attempt(wire, marker)


async def test_async_openai_sdk_responses_stream_fails_with_the_server_error_message(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = openai_error_stream(_identity(), "server_error", _SERVER_MESSAGE)
    with wire_server(serve(stream, AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = _azure_model(scenario, wire)
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key, max_retries=0)
        response: Final = await client.responses.create(
            model=model, input=marker, stream=True, extra_body={"cache": {"no-cache": True}}
        )
        events: Final = [event async for event in response]
        code, message = _failed_error(events)
        assert code == "server_error" and _SERVER_MESSAGE in message, events
        _assert_one_attempt(wire, marker)


def _assert_messages_error(error: Mapping[str, JsonValue], error_type: str, provider_message: str) -> None:
    assert error["type"] == error_type, error
    message: Final = string_value(error["message"])
    assert provider_message in message and SENTINEL_PREFIX not in message, message


def test_messages_stream_top_level_rate_limit_is_a_rate_limit_errorframe(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = openai_error_stream(_identity(), "rate_limit_exceeded", RATE_LIMIT_MESSAGE)
    served: Final = _post(gateway, "/v1/messages", messages_body("", marker), stream)
    assert served.response.status_code == 200, served.response.text
    events: Final = sse_events(served.response.text)
    assert events[0][0] == "message_start" and events[-1][0] == "error", events
    error: Final = object_value(events[-1][1]["error"])
    _assert_messages_error(error, "rate_limit_error", RATE_LIMIT_MESSAGE)
    assert string_value(error["message"]).count(RATE_LIMIT_PREFIX) == 1, error
    _assert_one_attempt(served.wire, marker)


def _anthropic_tools() -> list[anthropic.types.ToolParam]:  # mutable-ok: the Anthropic SDK tools parameter is a list
    return [
        {
            "name": "get_weather",
            "description": "Weather for a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]


async def _drain_async_messages_stream(client: anthropic.AsyncAnthropic, model: str, marker: str) -> None:
    async with client.messages.stream(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": marker}],
        tools=_anthropic_tools(),
        extra_body={"num_retries": 0, "cache": {"no-cache": True}},
    ) as events:
        async for _ in events:
            pass


def _drain_messages_stream(client: anthropic.Anthropic, model: str, marker: str) -> None:
    with client.messages.stream(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": marker}],
        tools=_anthropic_tools(),
        extra_body={"num_retries": 0, "cache": {"no-cache": True}},
    ) as events:
        for _ in events:
            pass


async def test_async_anthropic_sdk_stream_raises_the_top_level_rate_limit(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = openai_error_stream(_identity(), "rate_limit_exceeded", RATE_LIMIT_MESSAGE)
    with wire_server(serve(stream, AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = _azure_model(scenario, wire)
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        with pytest.raises(anthropic.APIStatusError) as raised:
            await _drain_async_messages_stream(client, model, marker)
        _assert_messages_error(
            object_value(object_value(raised.value.body)["error"]), "rate_limit_error", RATE_LIMIT_MESSAGE
        )
        _assert_one_attempt(wire, marker)


def test_sync_anthropic_sdk_stream_raises_the_top_level_server_error_message(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    stream: Final = openai_error_stream(_identity(), "server_error", _SERVER_MESSAGE)
    with wire_server(serve(stream, AZURE_TARGET)) as wire, gateway.scenario() as scenario:
        model: Final = _azure_model(scenario, wire)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        with pytest.raises(anthropic.APIStatusError) as raised:
            _drain_messages_stream(client, model, marker)
        _assert_messages_error(object_value(object_value(raised.value.body)["error"]), "api_error", _SERVER_MESSAGE)
        _assert_one_attempt(wire, marker)
