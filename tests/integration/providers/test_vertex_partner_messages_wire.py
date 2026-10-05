import json
from collections.abc import Callable
from typing import Final
from urllib.parse import parse_qs

import anthropic
import pytest
from anthropic.types import Message
from integration._support import claude_code as cc
from integration._support.client import Gateway, Scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "claude-sonnet-4-6"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_MODEL_TARGET: Final = (
    f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/{_BACKEND}:rawPredict"
)
_STREAM_TARGET: Final = (
    f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/{_BACKEND}:streamRawPredict?alt=sse"
)
_VERTEX_TOKEN: Final = "scripted-vertex-access-token"
_SUPPORTED_BETA: Final = "context-management-2025-06-27"
_BETA_HEADER: Final = f"{_SUPPORTED_BETA},made-up-future-beta-2099-01-01"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGE: Final = TypeAdapter(Message)
_SYSTEM: Final[list[JsonValue]] = [{"type": "text", "text": "Answer briefly and use the supplied tool when relevant."}]
_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Look up a synthetic record."}]
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
_THINKING: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024}
_CONTENT: Final[tuple[dict[str, JsonValue], ...]] = (
    {"type": "text", "text": "The record is ready."},
)
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 8, "output_tokens": 4}


def _without_none(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _deployment(scenario: Scenario, api_base: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, api_base),
        num_retries=0,
    )


def _peer(
    expected_body: dict[str, JsonValue],
    target: str,
    reply_body: bytes,
    stream_reply: tuple[bytes, ...],
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/_oauth/token":
            form: Final = parse_qs(request.body.decode())
            assert request.method == "POST", request.method
            assert request.headers["content-type"] == "application/x-www-form-urlencoded", request.headers
            assert set(form) == {"assertion", "grant_type"}, form
            assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"], form
            assert form["assertion"], form
            return Reply(
                body=json.dumps(
                    {"access_token": _VERTEX_TOKEN, "token_type": "Bearer", "expires_in": 3600}
                ).encode()
            )

        assert request.method == "POST" and request.target == target, request.target
        assert request.headers["authorization"] == f"Bearer {_VERTEX_TOKEN}", request.headers
        assert request.headers["anthropic-beta"] == _SUPPORTED_BETA, request.headers
        assert _JSON_OBJECT.validate_json(request.body) == expected_body, request.body
        if target == _STREAM_TARGET:
            return Reply(content_type="text/event-stream", chunks=stream_reply)
        return Reply(body=reply_body)

    return respond


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_partner_messages_uses_raw_predict_contract_and_typed_response(gateway: Gateway, stream: bool) -> None:
    identity: Final = f"msg_vertex_partner_{stream}"
    reply_body: Final = cc.message_reply(identity, _BACKEND, _CONTENT, _USAGE)
    stream_reply: Final = cc.message_stream(identity, _BACKEND, _CONTENT, _USAGE)

    with gateway.scenario() as scenario:
        with wire_server(
            _peer(
                {
                    "max_tokens": 1024,
                    "messages": _MESSAGES,
                    "system": _SYSTEM,
                    "tools": _TOOLS,
                    "thinking": _THINKING,
                    "stream": stream,
                    "anthropic_version": "vertex-2023-10-16",
                },
                _STREAM_TARGET if stream else _MODEL_TARGET,
                reply_body,
                stream_reply,
            )
        ) as wire:
            model: Final = _deployment(scenario, wire.url)
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
                        thinking=_THINKING,
                        stream=True,
                        extra_headers={"anthropic-beta": _BETA_HEADER},
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
                        _JSON_OBJECT.validate_python(event.model_dump(mode="json", exclude_none=True))
                        for event in events
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
                        thinking=_THINKING,
                        stream=False,
                        extra_headers={"anthropic-beta": _BETA_HEADER},
                    )
                    parsed_message: Final = _MESSAGE.validate_python(response)
                    expected_message: Final = _MESSAGE.validate_python(
                        {**_JSON_OBJECT.validate_json(reply_body), "model": model}
                    )
                    assert parsed_message.model_dump(mode="json") == expected_message.model_dump(mode="json"), (
                        response,
                    )

            requests: Final = wire.drain()
            assert tuple(request.target for request in requests) == ("/_oauth/token", _STREAM_TARGET if stream else _MODEL_TARGET), requests
            token_request: Final = requests[0]
            assert token_request.method == "POST", token_request
            assert token_request.headers["content-type"] == "application/x-www-form-urlencoded", token_request.headers
            model_request: Final = requests[1]
            assert model_request.method == "POST", model_request
            assert model_request.headers["authorization"] == f"Bearer {_VERTEX_TOKEN}", model_request.headers
