from typing import Final

import anthropic
import pytest
from anthropic.types import Message
from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-4-6"
_API_KEY: Final = "synthetic-azure-ai-key"
_SUPPORTED_BETA: Final = "interleaved-thinking-2025-05-14"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGE: Final = TypeAdapter(Message)
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Read the synthetic record."}]
_SYSTEM: Final = [{"type": "text", "text": "Use the supplied tool when relevant."}]
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
_CONTENT: Final[tuple[dict[str, JsonValue], ...]] = ({"type": "text", "text": "The record is ready."},)
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 8, "output_tokens": 4}


def _without_none(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


@pytest.mark.parametrize("base_suffix", ["", "/anthropic"], ids=["base", "anthropic_base"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_azure_ai_claude_messages_preserves_target_auth_beta_and_typed_response(
    gateway: Gateway, base_suffix: str, stream: bool
) -> None:
    identity: Final = f"msg_azure_claude_{base_suffix.strip('/') or 'base'}_{stream}"
    api_path: Final = f"/anthropic/v1/messages"
    reply_body: Final = cc.message_reply(identity, _BACKEND, _CONTENT, _USAGE)
    stream_reply: Final = cc.message_stream(identity, _BACKEND, _CONTENT, _USAGE)
    request_body_template: Final = {
        "max_tokens": 1024,
        "messages": _MESSAGES,
        "model": "",
        "stream": stream,
        "system": _SYSTEM,
        "tools": _TOOLS,
    }
    expected_provider_body: Final = {**request_body_template, "model": _BACKEND}

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == api_path, request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert request.headers.get("anthropic-beta") is None, request.headers
        assert "authorization" not in request.headers and "api-key" not in request.headers, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_provider_body, request.body
        if stream:
            return Reply(content_type="text/event-stream", chunks=stream_reply)
        return Reply(body=reply_body)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure_ai/{_BACKEND}",
            api_base=f"{wire.url}{base_suffix}",
            api_key=_API_KEY,
            num_retries=0,
        )
        with anthropic.Anthropic(
            api_key=gateway.key,
            base_url=str(gateway.client.base_url),
            max_retries=0,
        ) as client:
            if stream:
                stream_response: Final = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=_MESSAGES,
                    system=_SYSTEM,
                    tools=_TOOLS,
                    stream=True,
                    extra_headers={"anthropic-beta": "made-up-future-beta-2099-01-01"},
                )
                events: Final = tuple(stream_response)
                expected_events: Final = tuple(
                    _JSON_OBJECT.validate_python(
                        _without_none(
                            {
                                **data,
                                **(
                                    {
                                        "message": {
                                            **_JSON_OBJECT.validate_python(data["message"]),
                                            "model": model,
                                        }
                                    }
                                    if event_name == "message_start"
                                    else {}
                                ),
                            }
                        )
                    )
                    for event_name, data in cc.sse_events(b"".join(stream_reply).decode())
                )
                actual_events: Final = tuple(
                    _JSON_OBJECT.validate_python(event.model_dump(mode="json", exclude_none=True)) for event in events
                )
                assert actual_events == expected_events, actual_events
                started_message: Final = next(event.message for event in events if event.type == "message_start")
                assert started_message.id == identity and started_message.model == model, actual_events
            else:
                response: Final = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=_MESSAGES,
                    system=_SYSTEM,
                    tools=_TOOLS,
                    stream=False,
                    extra_headers={"anthropic-beta": "made-up-future-beta-2099-01-01"},
                )
                parsed_message: Final = _MESSAGE.validate_python(response)
                expected_message: Final = _MESSAGE.validate_python(
                    {**_JSON_OBJECT.validate_json(reply_body), "model": model}
                )
                assert parsed_message.model_dump(mode="json") == expected_message.model_dump(mode="json"), (response,)

        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        provider_request: Final = requests[0]
        assert provider_request.method == "POST" and provider_request.target == api_path, provider_request.target
        assert provider_request.headers["x-api-key"] == _API_KEY, provider_request.headers
        assert provider_request.headers["anthropic-version"] == "2023-06-01", provider_request.headers
        assert provider_request.headers.get("anthropic-beta") is None, provider_request.headers
        assert "authorization" not in provider_request.headers and "api-key" not in provider_request.headers
        assert _JSON_OBJECT.validate_json(provider_request.body) == expected_provider_body, provider_request.body


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_azure_ai_claude_messages_filters_auto_injected_betas_by_azure_support(gateway: Gateway, stream: bool) -> None:
    identity: Final = f"msg_azure_claude_beta_filter_{stream}"
    api_path: Final = "/anthropic/v1/messages"
    reply_body: Final = cc.message_reply(identity, _BACKEND, _CONTENT, _USAGE)
    stream_reply: Final = cc.message_stream(identity, _BACKEND, _CONTENT, _USAGE)
    output_config: Final = {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        }
    }
    context_management: Final = {"edits": [{"type": "compact_20260112"}]}
    request_body_template: Final = {
        "max_tokens": 1024,
        "messages": _MESSAGES,
        "model": "",
        "stream": stream,
        "system": _SYSTEM,
        "tools": _TOOLS,
        "output_config": output_config,
        "context_management": context_management,
    }
    expected_provider_body: Final = {**request_body_template, "model": _BACKEND}
    expected_beta: Final = "structured-outputs-2025-11-13"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == api_path, request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert request.headers.get("anthropic-beta") == expected_beta, request.headers
        assert "authorization" not in request.headers and "api-key" not in request.headers, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_provider_body, request.body
        if stream:
            return Reply(content_type="text/event-stream", chunks=stream_reply)
        return Reply(body=reply_body)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure_ai/{_BACKEND}",
            api_base=wire.url,
            api_key=_API_KEY,
            num_retries=0,
        )
        with anthropic.Anthropic(
            api_key=gateway.key,
            base_url=str(gateway.client.base_url),
            max_retries=0,
        ) as client:
            if stream:
                stream_response: Final = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=_MESSAGES,
                    system=_SYSTEM,
                    tools=_TOOLS,
                    stream=True,
                    output_config=output_config,
                    extra_body={"context_management": context_management},
                )
                events: Final = tuple(stream_response)
                expected_events: Final = tuple(
                    _JSON_OBJECT.validate_python(
                        _without_none(
                            {
                                **data,
                                **(
                                    {
                                        "message": {
                                            **_JSON_OBJECT.validate_python(data["message"]),
                                            "model": model,
                                        }
                                    }
                                    if event_name == "message_start"
                                    else {}
                                ),
                            }
                        )
                    )
                    for event_name, data in cc.sse_events(b"".join(stream_reply).decode())
                )
                actual_events: Final = tuple(
                    _JSON_OBJECT.validate_python(event.model_dump(mode="json", exclude_none=True)) for event in events
                )
                assert actual_events == expected_events, actual_events
                started_message: Final = next(event.message for event in events if event.type == "message_start")
                assert started_message.id == identity and started_message.model == model, actual_events
            else:
                response: Final = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=_MESSAGES,
                    system=_SYSTEM,
                    tools=_TOOLS,
                    stream=False,
                    output_config=output_config,
                    extra_body={"context_management": context_management},
                )
                parsed_message: Final = _MESSAGE.validate_python(response)
                expected_message: Final = _MESSAGE.validate_python(
                    {**_JSON_OBJECT.validate_json(reply_body), "model": model}
                )
                assert parsed_message.model_dump(mode="json") == expected_message.model_dump(mode="json"), (response,)

        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        provider_request: Final = requests[0]
        assert provider_request.method == "POST" and provider_request.target == api_path, provider_request.target
        assert provider_request.headers["x-api-key"] == _API_KEY, provider_request.headers
        assert provider_request.headers["anthropic-version"] == "2023-06-01", provider_request.headers
        assert provider_request.headers.get("anthropic-beta") == expected_beta, provider_request.headers
        assert "authorization" not in provider_request.headers and "api-key" not in provider_request.headers
        assert _JSON_OBJECT.validate_json(provider_request.body) == expected_provider_body, provider_request.body


def test_azure_ai_claude_messages_forwards_supported_beta(gateway: Gateway) -> None:
    pytest.skip(f"BUG: Azure AI Claude drops the supported beta header {_SUPPORTED_BETA}")

    identity: Final = "msg_azure_claude_supported_beta"
    reply_body: Final = cc.message_reply(identity, _BACKEND, _CONTENT, _USAGE)
    request_body: Final = {
        "max_tokens": 1024,
        "messages": _MESSAGES,
        "model": "",
        "stream": False,
        "system": _SYSTEM,
        "tools": _TOOLS,
    }
    expected_provider_body: Final = {**request_body, "model": _BACKEND}

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/anthropic/v1/messages", request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert request.headers.get("anthropic-beta") == _SUPPORTED_BETA, request.headers
        assert "authorization" not in request.headers and "api-key" not in request.headers, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_provider_body, request.body
        return Reply(body=reply_body)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure_ai/{_BACKEND}",
            api_base=wire.url,
            api_key=_API_KEY,
            num_retries=0,
        )
        with anthropic.Anthropic(
            api_key=gateway.key,
            base_url=str(gateway.client.base_url),
            max_retries=0,
        ) as client:
            response: Final = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=_MESSAGES,
                system=_SYSTEM,
                tools=_TOOLS,
                stream=False,
                extra_headers={"anthropic-beta": f"{_SUPPORTED_BETA},made-up-future-beta-2099-01-01"},
            )
            parsed_message: Final = _MESSAGE.validate_python(response)
            expected_message: Final = _MESSAGE.validate_python(
                {**_JSON_OBJECT.validate_json(reply_body), "model": model}
            )
            assert parsed_message.model_dump(mode="json") == expected_message.model_dump(mode="json"), response

        requests: Final = wire.drain()
        assert len(requests) == 1, requests
        provider_request: Final = requests[0]
        assert provider_request.method == "POST" and provider_request.target == "/anthropic/v1/messages"
        assert provider_request.headers["x-api-key"] == _API_KEY, provider_request.headers
        assert provider_request.headers.get("anthropic-beta") == _SUPPORTED_BETA, provider_request.headers
        assert _JSON_OBJECT.validate_json(provider_request.body) == expected_provider_body, provider_request.body
