import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Final
from uuid import uuid4

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "openai.gpt-5.6-luna"
_MODEL: Final = f"bedrock_mantle/{_BACKEND}"
_API_KEY: Final = "synthetic-mantle-bearer"
_RESPONSES_PATH: Final = "/openai/v1/responses"
_OPENAI_MODEL: Final = "openai/gpt-4o-mini"
_OPENAI_CHAT_PATH: Final = "/v1/chat/completions"
_OPENAI_RESPONSES_PATH: Final = "/v1/responses"
_OPENAI_API_KEY: Final = "synthetic-openai-key"
_GENERIC: Final = "prompt is too long: your prompt exceeds the model's context window"
_TOO_LONG: Final = "prompt is too long"
_UPSTREAM_MESSAGE: Final = (
    "Your input exceeds the context window of this model. Please adjust your input and try again."
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_VALUE: Final = TypeAdapter(JsonValue)
_INVALID_INPUT_MESSAGE: Final = "Invalid 'input': expected a string or array"
_INVALID_PROMPT_MESSAGE: Final = "Invalid prompt: your prompt was flagged as potentially violating our usage policy."
_OPENAI_OVERFLOW_MESSAGE: Final = (
    "This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens. "
    "Please reduce the length of the messages."
)
_OPENAI_OVERFLOW_MARK: Final = "maximum context length is 128000 tokens"
_LEGACY_PROMPT_TOKENS: Final = 1055489
_LEGACY_MODEL_MAXIMUM: Final = 1050000
_LEGACY_MESSAGE: Final = (
    f"prompt tokens ({_LEGACY_PROMPT_TOKENS}) exceed model maximum ({_LEGACY_MODEL_MAXIMUM}) for {_BACKEND}"
)
_HAPPY_TEXT: Final = "mantle overflow audit control"


def _envelope(code: str | None, message: str) -> bytes:
    return json.dumps(
        {"error": {"code": code, "message": message, "param": "input", "type": "invalid_request_error"}}
    ).encode()


_OVERFLOW_BODY: Final = _envelope("context_length_exceeded", _UPSTREAM_MESSAGE)
_LEGACY_BODY: Final = _envelope("validation_error", _LEGACY_MESSAGE)
_BAD_INPUT_BODY: Final = _envelope(None, _INVALID_INPUT_MESSAGE)
_OPENAI_OVERFLOW_ERROR: Final[dict[str, JsonValue]] = {
    "message": _OPENAI_OVERFLOW_MESSAGE,
    "type": "invalid_request_error",
    "param": "messages",
    "code": "context_length_exceeded",
}
_OPENAI_OVERFLOW_BODY: Final = json.dumps({"error": _OPENAI_OVERFLOW_ERROR}).encode()
_OVERFLOW: Final = Reply(status=400, body=_OVERFLOW_BODY)
_OVERFLOW_STREAM_ERROR: Final[dict[str, JsonValue]] = {"code": "context_length_exceeded", "message": _UPSTREAM_MESSAGE}


def _response_object(identity: str, status: str, text: str | None, model: str = _BACKEND) -> dict[str, JsonValue]:
    output: Final[list[JsonValue]] = (
        []
        if text is None
        else [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    )
    return {
        "id": f"resp_{identity}",
        "object": "response",
        "created_at": 1789788253,
        "status": status,
        "model": model,
        "output": output,
        "usage": {"input_tokens": 21, "output_tokens": 4, "total_tokens": 25},
    }


def _frames(events: tuple[dict[str, JsonValue], ...]) -> tuple[bytes, ...]:
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _failed_stream(error: JsonValue, model: str = _BACKEND) -> Reply:
    identity: Final = uuid4().hex
    return Reply(
        content_type="text/event-stream",
        chunks=_frames(
            (
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": _response_object(identity, "in_progress", None, model),
                },
                {
                    "type": "response.failed",
                    "sequence_number": 1,
                    "response": {**_response_object(identity, "failed", None, model), "error": error},
                },
            )
        ),
    )


def _happy_reply(stream: bool) -> Reply:
    identity: Final = uuid4().hex
    completed: Final = _response_object(identity, "completed", _HAPPY_TEXT)
    if not stream:
        return Reply(body=json.dumps(completed).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=_frames(
            (
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": _response_object(identity, "in_progress", None),
                },
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": f"msg_{identity}",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": _HAPPY_TEXT,
                },
                {"type": "response.completed", "sequence_number": 2, "response": completed},
            )
        ),
    )


def _mantle_peer(prompt: str, reply: Reply) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == _RESPONSES_PATH, request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", dict(request.headers)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND, body
        assert prompt in json.dumps(body["input"]), body
        return reply

    return respond


def _overflowing_mantle_peer(prompt: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == _RESPONSES_PATH, request.target
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert prompt in json.dumps(body["input"]), body
        return _failed_stream(_OVERFLOW_STREAM_ERROR) if body.get("stream") is True else _OVERFLOW

    return respond


def _openai_peer(prompt: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.headers["authorization"] == f"Bearer {_OPENAI_API_KEY}", dict(request.headers)
        assert prompt in request.body.decode(), request.body
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if request.target == _OPENAI_RESPONSES_PATH and body.get("stream") is True:
            return _failed_stream(_OPENAI_OVERFLOW_ERROR, "gpt-4o-mini")
        assert request.target in (_OPENAI_CHAT_PATH, _OPENAI_RESPONSES_PATH), request.target
        return Reply(status=400, body=_OPENAI_OVERFLOW_BODY)

    return respond


def _chat(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _messages(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def _responses(model: str, prompt: str, stream: bool = False) -> dict[str, JsonValue]:
    return {"model": model, "input": prompt, "stream": stream}


def _consumed(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[httpx.Response, str]:
    with gateway.client.stream("POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}) as response:
        text: Final = b"".join(response.iter_bytes()).decode()
    return response, text


def _events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _error_events(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(event for event in _events(text) if event.get("type") == "error")


def _only_error(text: str) -> dict[str, JsonValue]:
    errors: Final = _error_events(text)
    assert len(errors) == 1, text
    return object_value(errors[0]["error"])


def _only_failed(text: str) -> dict[str, JsonValue]:
    failed: Final = tuple(event for event in _events(text) if event.get("type") == "response.failed")
    assert len(failed) == 1, text
    return object_value(object_value(failed[0]["response"])["error"])


def _error_object(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(_JSON_OBJECT.validate_json(response.content)["error"])


def _error_message(response: httpx.Response) -> str:
    return str(_error_object(response)["message"])


def _only_call(wire: Wire, target: str = _RESPONSES_PATH) -> None:
    assert [(request.method, request.target) for request in wire.drain()] == [("POST", target)]


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,))


def _single_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: _spend_rows(call_id), lambda values: len(values) == 1, seconds=70)
    return rows[0]


def _failure_message(call_id: str) -> str:
    row: Final = _single_row(call_id)
    assert row["status"] == "failure", row
    metadata: Final = row["metadata"]
    parsed: Final = _JSON_OBJECT.validate_json(metadata) if isinstance(metadata, str) else object_value(metadata)
    return str(object_value(parsed["error_information"])["error_message"])


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{_proxy_url(gateway)}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(timeout=15, trust_env=False),
    )


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=_proxy_url(gateway),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(timeout=15, trust_env=False),
    )


def test_chat_completions_overflow_envelope_returns_400_prompt_too_long_and_logs_failure(gateway: Gateway) -> None:
    prompt: Final = f"overflow chat {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == 400, response.text
        assert _GENERIC in _error_message(response), response.text
        _only_call(wire)
        assert _GENERIC in _failure_message(response.headers["x-litellm-call-id"])


def test_chat_completions_stream_overflow_envelope_returns_400_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"overflow chat stream {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/chat/completions", _chat(model, prompt, stream=True))
        assert response.status_code == 400, text
        assert _GENERIC in text, text
        _only_call(wire)


def test_messages_overflow_envelope_returns_400_invalid_request_and_logs_failure(gateway: Gateway) -> None:
    prompt: Final = f"overflow messages {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", _messages(model, prompt))
        assert response.status_code == 400, response.text
        error: Final = _error_object(response)
        assert error["type"] == "invalid_request_error", response.text
        assert _GENERIC in str(error["message"]), response.text
        _only_call(wire)
        assert _GENERIC in _failure_message(response.headers["x-litellm-call-id"])


def test_messages_stream_overflow_envelope_before_the_stream_returns_400_invalid_request(gateway: Gateway) -> None:
    prompt: Final = f"overflow messages stream {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/messages", _messages(model, prompt, stream=True))
        assert response.status_code == 400, text
        error: Final = object_value(_JSON_OBJECT.validate_json(text)["error"])
        assert error["type"] == "invalid_request_error", text
        assert _GENERIC in str(error["message"]), text
        _only_call(wire)


def test_responses_overflow_envelope_returns_400_prompt_too_long_and_logs_failure(gateway: Gateway) -> None:
    prompt: Final = f"overflow responses {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/responses", _responses(model, prompt))
        assert response.status_code == 400, response.text
        assert _GENERIC in _error_message(response), response.text
        _only_call(wire)
        assert _GENERIC in _failure_message(response.headers["x-litellm-call-id"])


def test_responses_stream_overflow_envelope_before_the_stream_returns_400_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"overflow responses stream {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/responses", _responses(model, prompt, stream=True))
        assert response.status_code == 400, text
        assert _GENERIC in str(object_value(_JSON_OBJECT.validate_json(text)["error"])["message"]), text
        _only_call(wire)


def test_openai_sdk_chat_completions_raises_bad_request_saying_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"overflow sdk chat {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(openai.BadRequestError) as caught:
            _openai_client(gateway).chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        assert caught.value.status_code == 400
        assert _GENERIC in str(caught.value), str(caught.value)
        _only_call(wire)


async def test_async_openai_sdk_chat_completions_raises_bad_request_saying_prompt_too_long(
    gateway: Gateway,
) -> None:
    prompt: Final = f"overflow async sdk chat {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        async with openai.AsyncOpenAI(
            base_url=f"{_proxy_url(gateway)}/v1",
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=15, trust_env=False),
        ) as client:
            with pytest.raises(openai.BadRequestError) as caught:
                await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        assert caught.value.status_code == 400
        assert _GENERIC in str(caught.value), str(caught.value)
        _only_call(wire)


def test_openai_sdk_responses_raises_bad_request_saying_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"overflow sdk responses {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(openai.BadRequestError) as caught:
            _openai_client(gateway).responses.create(model=model, input=prompt)
        assert caught.value.status_code == 400
        assert _GENERIC in str(caught.value), str(caught.value)
        _only_call(wire)


def test_anthropic_sdk_messages_raises_bad_request_saying_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"overflow sdk messages {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(anthropic.BadRequestError) as caught:
            _anthropic_client(gateway).messages.create(
                model=model, max_tokens=32, messages=[{"role": "user", "content": prompt}]
            )
        assert caught.value.status_code == 400
        assert _GENERIC in str(caught.value), str(caught.value)
        _only_call(wire)


async def test_async_anthropic_sdk_messages_raises_bad_request_saying_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"overflow async sdk messages {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        async with anthropic.AsyncAnthropic(
            base_url=_proxy_url(gateway),
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=15, trust_env=False),
        ) as client:
            with pytest.raises(anthropic.BadRequestError) as caught:
                await client.messages.create(model=model, max_tokens=32, messages=[{"role": "user", "content": prompt}])
        assert caught.value.status_code == 400
        assert _GENERIC in str(caught.value), str(caught.value)
        _only_call(wire)


def test_anthropic_sdk_messages_stream_raises_invalid_request_error_saying_prompt_too_long(
    gateway: Gateway,
) -> None:
    prompt: Final = f"overflow sdk messages stream {uuid4().hex}"
    reply: Final = _failed_stream(_OVERFLOW_STREAM_ERROR)
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        with pytest.raises(anthropic.APIStatusError) as caught:
            for _ in _anthropic_client(gateway).messages.create(
                model=model, max_tokens=32, messages=[{"role": "user", "content": prompt}], stream=True
            ):
                pass
        body: Final = object_value(_JSON_VALUE.validate_python(caught.value.body))
        error: Final = object_value(body["error"])
        assert error["type"] == "invalid_request_error", body
        assert _GENERIC in str(error["message"]), body
        _only_call(wire)


def test_chat_completions_stream_overflow_in_response_failed_event_returns_400_prompt_too_long(
    gateway: Gateway,
) -> None:
    prompt: Final = f"failed event chat {uuid4().hex}"
    reply: Final = _failed_stream(_OVERFLOW_STREAM_ERROR)
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/chat/completions", _chat(model, prompt, stream=True))
        assert response.status_code == 400, text
        assert _GENERIC in text, text
        _only_call(wire)


def test_messages_stream_overflow_in_response_failed_event_emits_invalid_request_error_event(
    gateway: Gateway,
) -> None:
    prompt: Final = f"failed event messages {uuid4().hex}"
    reply: Final = _failed_stream(_OVERFLOW_STREAM_ERROR)
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/messages", _messages(model, prompt, stream=True))
        assert response.status_code == 200, text
        assert response.headers["content-type"].startswith("text/event-stream"), dict(response.headers)
        error: Final = _only_error(text)
        assert error["type"] == "invalid_request_error", text
        assert _GENERIC in str(error["message"]), text
        assert not any(event["type"] == "content_block_delta" for event in _events(text)), text
        _only_call(wire)


def test_responses_stream_overflow_in_response_failed_event_relays_failure_saying_prompt_too_long(
    gateway: Gateway,
) -> None:
    prompt: Final = f"failed event responses {uuid4().hex}"
    reply: Final = _failed_stream(_OVERFLOW_STREAM_ERROR)
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/responses", _responses(model, prompt, stream=True))
        assert response.status_code == 200, text
        assert _GENERIC in str(_only_failed(text)["message"]), text
        _only_call(wire)


def test_chat_completions_non_overflow_400_keeps_the_upstream_message(gateway: Gateway) -> None:
    prompt: Final = f"bad input chat {uuid4().hex}"
    reply: Final = Reply(status=400, body=_BAD_INPUT_BODY)
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == 400, response.text
        assert _INVALID_INPUT_MESSAGE in response.text, response.text
        assert _TOO_LONG not in response.text, response.text
        _only_call(wire)


def test_messages_stream_non_overflow_400_before_the_stream_returns_400_with_the_upstream_message(
    gateway: Gateway,
) -> None:
    prompt: Final = f"bad input messages stream {uuid4().hex}"
    reply: Final = Reply(status=400, body=_BAD_INPUT_BODY)
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/messages", _messages(model, prompt, stream=True))
        assert response.status_code == 400, text
        assert _INVALID_INPUT_MESSAGE in text, text
        assert _TOO_LONG not in text, text
        _only_call(wire)


def test_messages_stream_non_overflow_response_failed_event_still_returns_500_api_error(gateway: Gateway) -> None:
    prompt: Final = f"invalid prompt messages stream {uuid4().hex}"
    reply: Final = _failed_stream({"code": "invalid_prompt", "message": _INVALID_PROMPT_MESSAGE})
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/messages", _messages(model, prompt, stream=True))
        assert response.status_code == 500, text
        error: Final = object_value(_JSON_OBJECT.validate_json(text)["error"])
        assert error["type"] == "api_error", text
        assert _INVALID_PROMPT_MESSAGE in str(error["message"]), text
        assert _TOO_LONG not in text, text
        _only_call(wire)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    (
        (401, b'{"error":{"message":"invalid bearer","type":"authentication_error","code":null}}', 401),
        (429, b'{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit_exceeded"}}', 429),
        (500, b'{"error":{"message":"boom","type":"server_error","code":null}}', 503),
        (
            404,
            b'{"error":{"message":"The model `x` does not exist","type":"invalid_request_error","code":"model_not_found"}}',
            404,
        ),
    ),
    ids=("401", "429", "500", "404"),
)
def test_chat_completions_other_upstream_statuses_keep_their_mapping(
    gateway: Gateway, status: int, body: bytes, expected: int
) -> None:
    prompt: Final = f"status {status} chat {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, Reply(status=status, body=body))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == expected, response.text
        assert _TOO_LONG not in response.text, response.text
        _only_call(wire)


def test_messages_stream_legacy_token_count_envelope_returns_400_with_the_counts(gateway: Gateway) -> None:
    prompt: Final = f"legacy messages stream {uuid4().hex}"
    with (
        wire_server(_mantle_peer(prompt, Reply(status=400, body=_LEGACY_BODY))) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/messages", _messages(model, prompt, stream=True))
        assert response.status_code == 400, text
        error: Final = object_value(_JSON_OBJECT.validate_json(text)["error"])
        assert error["type"] == "invalid_request_error", text
        assert f"prompt is too long: {_LEGACY_PROMPT_TOKENS} tokens > {_LEGACY_MODEL_MAXIMUM} maximum" in str(
            error["message"]
        ), text
        _only_call(wire)


def test_chat_completions_code_only_overflow_envelope_returns_400_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"code only chat {uuid4().hex}"
    reply: Final = Reply(status=400, body=b'{"error":{"code":"context_length_exceeded"}}')
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == 400, response.text
        assert _GENERIC in _error_message(response), response.text
        _only_call(wire)


def test_chat_completions_text_plain_overflow_body_returns_400_prompt_too_long(gateway: Gateway) -> None:
    prompt: Final = f"text plain chat {uuid4().hex}"
    reply: Final = Reply(status=400, body=b"request rejected: context_length_exceeded", content_type="text/plain")
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == 400, response.text
        assert _GENERIC in _error_message(response), response.text
        _only_call(wire)


def test_chat_completions_five_kilobyte_overflow_message_returns_400_and_keeps_the_proxy_alive(
    gateway: Gateway,
) -> None:
    prompt: Final = f"large envelope chat {uuid4().hex}"
    reply: Final = Reply(status=400, body=_envelope("context_length_exceeded", "x" * 5120))
    with wire_server(_mantle_peer(prompt, reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == 400, response.text
        assert _GENERIC in _error_message(response), response.text
        _only_call(wire)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_chat_completions_stream_response_failed_without_error_fails_and_keeps_the_proxy_alive(
    gateway: Gateway,
) -> None:
    prompt: Final = f"null error chat stream {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _failed_stream(None))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response, text = _consumed(gateway, "/v1/chat/completions", _chat(model, prompt, stream=True))
        assert response.status_code >= 400, text
        assert _TOO_LONG not in text, text
        _only_call(wire)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_chat_completions_repeated_overflow_logs_one_failure_row_per_call(gateway: Gateway) -> None:
    prompt: Final = f"repeated overflow chat {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _OVERFLOW)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        responses: Final = tuple(
            gateway.request("POST", "/v1/chat/completions", _chat(model, prompt)) for _ in range(2)
        )
        call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
        assert len(set(call_ids)) == 2, call_ids
        assert [request.target for request in wire.drain()] == [_RESPONSES_PATH, _RESPONSES_PATH]
        for response, call_id in zip(responses, call_ids, strict=True):
            assert response.status_code == 400, response.text
            assert _GENERIC in _failure_message(call_id)


def test_chat_completions_prompt_naming_the_error_code_still_succeeds(gateway: Gateway) -> None:
    prompt: Final = f"my prompt mentions context_length_exceeded {uuid4().hex}"
    with wire_server(_mantle_peer(prompt, _happy_reply(stream=False))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat(model, prompt))
        assert response.status_code == 200, response.text
        assert _HAPPY_TEXT in response.text, response.text
        _only_call(wire)


def test_messages_stream_openai_deployment_overflow_in_the_stream_emits_invalid_request_error_event(
    gateway: Gateway,
) -> None:
    prompt: Final = f"openai overflow messages stream {uuid4().hex}"
    with wire_server(_openai_peer(prompt)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_OPENAI_MODEL, api_base=f"{wire.url}/v1", api_key=_OPENAI_API_KEY)
        response, text = _consumed(gateway, "/v1/messages", _messages(model, prompt, stream=True))
        assert response.status_code == 200, text
        error: Final = _only_error(text)
        assert error["type"] == "invalid_request_error", text
        assert _OPENAI_OVERFLOW_MARK in str(error["message"]), text
        _only_call(wire, _OPENAI_RESPONSES_PATH)


def test_chat_completions_stream_openai_deployment_overflow_returns_400(gateway: Gateway) -> None:
    prompt: Final = f"openai overflow chat stream {uuid4().hex}"
    with wire_server(_openai_peer(prompt)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_OPENAI_MODEL, api_base=f"{wire.url}/v1", api_key=_OPENAI_API_KEY)
        response, text = _consumed(gateway, "/v1/chat/completions", _chat(model, prompt, stream=True))
        assert response.status_code == 400, text
        assert _OPENAI_OVERFLOW_MARK in text, text
        _only_call(wire, _OPENAI_CHAT_PATH)


def test_messages_openai_deployment_overflow_returns_400_invalid_request(gateway: Gateway) -> None:
    prompt: Final = f"openai overflow messages {uuid4().hex}"
    with wire_server(_openai_peer(prompt)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_OPENAI_MODEL, api_base=f"{wire.url}/v1", api_key=_OPENAI_API_KEY)
        response: Final = gateway.request("POST", "/v1/messages", _messages(model, prompt))
        assert response.status_code == 400, response.text
        error: Final = _error_object(response)
        assert error["type"] == "invalid_request_error", response.text
        assert _OPENAI_OVERFLOW_MARK in str(error["message"]), response.text
        _only_call(wire, _OPENAI_RESPONSES_PATH)


_OVERFLOW_MARKER: Final = "chaos-overflow"
_HAPPY_MARKER: Final = "chaos-happy"
_Call = tuple[str, dict[str, JsonValue], bool, str]
_Outcome = tuple[str, bool, bool, int, str, str]
_Builder = Callable[[str, str, bool], dict[str, JsonValue]]
_Cell = tuple[tuple[str, _Builder], str, bool]
_BUILDERS: Final = (("/v1/chat/completions", _chat), ("/v1/messages", _messages), ("/v1/responses", _responses))
_STREAMED_MESSAGES_OVERFLOW: Final = ("/v1/messages", True, True)


def _logs_a_spend_row(path: str, overflow: bool, stream: bool) -> bool:
    return (path, overflow, stream) != _STREAMED_MESSAGES_OVERFLOW


def _chaos_peer(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    stream: Final = body.get("stream") is True
    if _OVERFLOW_MARKER in json.dumps(body["input"]):
        return _failed_stream(_OVERFLOW_STREAM_ERROR) if stream else _OVERFLOW
    return _happy_reply(stream)


def _burst_bodies(model: str, round_name: str) -> tuple[_Call, ...]:
    def call(cell: _Cell) -> _Call:
        (path, build), marker, stream = cell
        tag: Final = f"chaos-{round_name}-{uuid4().hex}"
        prompt: Final = f"{marker} {round_name} {path} stream={stream} {tag}"
        return path, build(model, prompt, stream), marker == _OVERFLOW_MARKER, tag

    return tuple(call(cell) for cell in product(_BUILDERS, (_HAPPY_MARKER, _OVERFLOW_MARKER), (False, True)))


def _tagged_rows(tag: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT status FROM "LiteLLM_SpendLogs" WHERE request_tags::jsonb @> %s::jsonb', (json.dumps([tag]),)
    )


def _single_tagged_status(tag: str) -> str:
    rows: Final = eventually(lambda: _tagged_rows(tag), lambda values: len(values) == 1, seconds=70)
    return str(rows[0]["status"])


def _fire(gateway: Gateway, bodies: tuple[_Call, ...]) -> tuple[_Outcome, ...]:
    def one(item: _Call) -> _Outcome:
        path, body, overflow, tag = item
        with gateway.client.stream(
            "POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-tags": tag}
        ) as response:
            text: Final = b"".join(response.iter_bytes()).decode()
        return path, overflow, body["stream"] is True, response.status_code, tag, text

    with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        return tuple(pool.map(one, bodies))


def _assert_served(outcomes: tuple[_Outcome, ...]) -> None:
    for path, overflow, stream, status, tag, text in outcomes:
        if not overflow:
            assert status == 200, (path, text)
            assert _HAPPY_TEXT in text, (path, text)
            assert _single_tagged_status(tag) == "success", (path, tag)
            continue
        assert _GENERIC in text, (path, status, text)
        assert status in (200, 400), (path, status, text)
        if _logs_a_spend_row(path, overflow, stream):
            assert _single_tagged_status(tag) == "failure", (path, tag)


def test_messages_stream_overflow_logs_one_failure_row(gateway: Gateway) -> None:
    pytest.skip("BUG: a streamed /v1/messages context overflow writes no LiteLLM_SpendLogs row (LIT-9132)")
    with wire_server(_chaos_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
        tag: Final = f"chaos-single-{uuid4().hex}"
        body: Final = _messages(model, f"{_OVERFLOW_MARKER} single {tag}", True)
        ((_, _, _, status, _, text),) = _fire(gateway, (("/v1/messages", body, True, tag),))
        assert status == 200, text
        assert _GENERIC in text, text
        assert _single_tagged_status(tag) == "failure", tag


def test_chaos_mantle_peer_outage_mid_burst_logs_every_call_once_and_keeps_the_proxy_alive(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with wire_server(_chaos_peer) as wire:
            port: Final = int(wire.url.rsplit(":", 1)[1])
            model: Final = scenario.model(model=_MODEL, api_base=wire.url, api_key=_API_KEY)
            before: Final = _fire(gateway, _burst_bodies(model, "before"))
            assert len(wire.drain()) == len(before)
        during: Final = _fire(gateway, _burst_bodies(model, "during"))
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        with wire_server(_chaos_peer, port=port) as revived:
            after: Final = _fire(gateway, _burst_bodies(model, "after"))
            assert len(revived.drain()) == len(after)
        _assert_served(before)
        _assert_served(after)
        for path, _, _, status, tag, text in during:
            assert status >= 500 or _error_events(text) or '"response.failed"' in text, (path, status, text)
            assert _TOO_LONG not in text, (path, text)
            assert _single_tagged_status(tag) == "failure", (path, tag)
