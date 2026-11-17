import itertools
import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-5.4-mini"
_API_KEY: Final = "synthetic-openai-key"
_CORRECTION: Final = "Stop refactoring the parser and only fix the failing test instead."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_OBJECTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_PIXEL: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
_PASTED_URL: Final = "https://example.invalid/pasted.png"
_MISSING_DETAIL: Final = json.dumps(
    {"error": {"message": "1 validation error: input.0.content.1.detail Field required", "type": "BadRequestError"}}
).encode()


def _responses_reply(identity: str, content: str) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": _BACKEND,
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
        }
    ).encode()


def _image_parts(body: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    contents: Final = (
        _JSON_OBJECTS.validate_python(item.get("content", ())) for item in _JSON_OBJECTS.validate_python(body["input"])
    )
    return tuple(part for part in itertools.chain.from_iterable(contents) if part.get("type") == "input_image")


@pytest.mark.parametrize(
    ("source", "image_url"),
    (
        ({"type": "base64", "media_type": "image/png", "data": _PIXEL}, f"data:image/png;base64,{_PIXEL}"),
        ({"type": "url", "url": _PASTED_URL}, _PASTED_URL),
    ),
    ids=("base64", "url"),
)
def test_pasted_image_reaches_a_responses_server_that_requires_image_detail(
    gateway: Gateway, source: dict[str, JsonValue], image_url: str
) -> None:
    identity: Final = f"openai-image-detail-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        if any("detail" not in part for part in _image_parts(_JSON_OBJECT.validate_json(request.body))):
            return Reply(status=400, body=_MISSING_DETAIL)
        return Reply(body=_responses_reply(identity, "A red circle."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image", "source": source},
                        ],
                    }
                ],
            },
        )
        requests: Final = wire.drain()
        assert len(requests) == 1, response.text
        assert _JSON_OBJECT.validate_json(requests[0].body)["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ],
            }
        ], requests[0].body
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["content"] == [{"type": "text", "text": "A red circle."}], (
            response.text
        )


@pytest.mark.covers("providers.anthropic_messages_openai_bridge.midturn_system_correction_reaches_the_wire")
def test_midturn_system_correction_is_forwarded_to_openai_responses(gateway: Gateway) -> None:
    identity: Final = f"openai-midturn-system-{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/responses"
        assert request.headers["authorization"] == f"Bearer {_API_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["model"] == _BACKEND
        assert body["instructions"] == "You are a coding agent."
        assert body["input"] == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the failing test."}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will start by refactoring the parser."}],
            },
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": _CORRECTION}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Continue."}]},
        ], body
        return Reply(body=_responses_reply(identity, "Understood."))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "system": "You are a coding agent.",
                "messages": [
                    {"role": "user", "content": "Fix the failing test."},
                    {"role": "assistant", "content": "I will start by refactoring the parser."},
                    {"role": "system", "content": _CORRECTION},
                    {"role": "user", "content": "Continue."},
                ],
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["content"] == [{"type": "text", "text": "Understood."}], response.text
        assert payload["stop_reason"] == "end_turn", response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/responses")]
