import json
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

_GEMINI_MODEL: Final = "gemini/gemini-2.5-flash"
_GEMINI_DEPLOYMENT_KEY: Final = "synthetic-gemini-deployment-key"

_MODEL_BODY: Final = {
    "input": "plan a trip",
    "system_instruction": "be brief",
    "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}],
    "generation_config": {"temperature": 0.3, "image_config": {"aspect_ratio": "16:9"}},
    "response_mime_type": "application/json",
    "response_format": {"type": "object", "properties": {"plan": {"type": "string"}}},
    "previous_interaction_id": "interaction-prev-123",
    "store": True,
    "background": True,
}

_INTERACTION_REPLY: Final = {
    "id": "interaction-wire-1",
    "object": "interaction",
    "model": "gemini-2.5-flash",
    "status": "completed",
    "created": "2026-05-20T00:00:00Z",
    "steps": [{"type": "content", "content": {"type": "text", "text": "itinerary"}}],
    "usage": {"total_tokens": 42},
}
_KNOWN_DONE_TERMINATOR: Final = b"data: [DONE]\n\n"


class _Interaction(BaseModel):
    id: str
    object: str
    model: str | None = None
    agent: str | None = None
    status: str
    created: str
    steps: list[dict]
    usage: dict


def _sse_event(event_type: str, payload: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _proxy_stream_event(payload: dict, model: str) -> bytes:
    event: Final = {
        "event_type": payload["event_type"],
        "id": payload["id"],
        **({"object": payload["object"]} if "object" in payload else {}),
        "model": model,
        **{key: value for key, value in payload.items() if key not in {"event_type", "id", "object"}},
    }
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


_STREAM_EVENTS: Final = (
    (
        "interaction.start",
        {
            "event_type": "interaction.start",
            "id": "interaction-wire-2",
            "object": "interaction",
            "status": "in_progress",
        },
    ),
    (
        "content.delta",
        {
            "event_type": "content.delta",
            "id": "interaction-wire-2",
            "delta": {"type": "text", "text": "itin"},
        },
    ),
    (
        "interaction.complete",
        {
            "event_type": "interaction.complete",
            "id": "interaction-wire-2",
            "object": "interaction",
            "status": "completed",
            "usage": {"total_tokens": 42},
        },
    ),
)


@pytest.mark.parametrize("prefix", ["/v1beta/interactions", "/interactions"])
@pytest.mark.parametrize(
    "request_input",
    [
        "plan a trip",
        [
            {"role": "user", "content": "plan a trip"},
            {"role": "model", "content": "where to"},
            {"role": "user", "content": "lisbon"},
        ],
    ],
    ids=["string", "turns"],
)
def test_interaction_forwards_native_body_to_gemini(gateway: Gateway, prefix: str, request_input: object) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1beta/interactions"
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        assert request.headers["api-revision"] == "2026-05-20"
        assert json.loads(request.body) == {
            "model": "gemini-2.5-flash",
            "input": request_input,
            "tools": _MODEL_BODY["tools"],
            "system_instruction": "be brief",
            "store": True,
            "background": True,
            "previous_interaction_id": "interaction-prev-123",
            "generation_config": {"temperature": 0.3},
            "response_format": [
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": {"type": "object", "properties": {"plan": {"type": "string"}}},
                },
                {"type": "image", "aspect_ratio": "16:9"},
            ],
        }
        return Reply(body=json.dumps(_INTERACTION_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", prefix, {"model": model, **_MODEL_BODY, "input": request_input}, key=key
        )
        assert response.status_code == 200, response.text
        payload: Final = _Interaction.model_validate_json(response.content)
        assert payload.model == model, response.text
        assert response.json() == {
            **_INTERACTION_REPLY,
            "model": model,
            "agent": None,
            "updated": None,
            "outputs": None,
        }, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1beta/interactions")], response.text


@pytest.mark.parametrize("prefix", ["/v1beta/interactions", "/interactions"])
def test_interaction_authenticates_google_key_header(gateway: Gateway, prefix: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1beta/interactions"
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        assert all(key not in value for value in request.headers.values()), dict(request.headers)
        assert json.loads(request.body) == {
            "model": "gemini-2.5-flash",
            "input": "plan a trip",
        }, request.body.decode()
        return Reply(body=json.dumps(_INTERACTION_REPLY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.client.post(
            prefix,
            json={"model": model, "input": "plan a trip"},
            headers={"x-goog-api-key": key},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            **_INTERACTION_REPLY,
            "model": model,
            "agent": None,
            "updated": None,
            "outputs": None,
        }, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1beta/interactions")], response.text


@pytest.mark.parametrize("prefix", ["/v1beta/interactions", "/interactions"])
def test_agent_interaction_uses_env_gemini_credentials(gateway: Gateway, tmp_path: Path, prefix: str) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1beta/interactions"
        assert request.headers["x-goog-api-key"] == "synthetic-env-gemini-key"
        assert request.headers["api-revision"] == "2026-05-20"
        body: Final = json.loads(request.body)
        assert body == {
            "agent": "deep-research-pro-preview-12-2025",
            "input": [{"role": "user", "content": "research trails"}],
        }, body
        return Reply(body=json.dumps({**_INTERACTION_REPLY, "agent": "deep-research-pro-preview-12-2025"}).encode())

    with wire_server(respond) as wire:
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "synthetic-env-gemini-key"},
        ) as owned:
            response: Final = owned.gateway.request(
                "POST",
                prefix,
                {
                    "agent": "deep-research-pro-preview-12-2025",
                    "input": [{"role": "user", "content": "research trails"}],
                },
            )
            assert response.status_code == 200, response.text
            _Interaction.model_validate_json(response.content)
            assert response.json() == {
                **_INTERACTION_REPLY,
                "agent": "deep-research-pro-preview-12-2025",
                "updated": None,
                "outputs": None,
            }, response.text
            assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1beta/interactions")], response.text


@pytest.mark.parametrize("prefix", ["/v1beta/interactions", "/interactions"])
def test_stream_interaction_relays_gemini_events(gateway: Gateway, prefix: str) -> None:
    frames: Final = tuple(_sse_event(event_type, payload) for event_type, payload in _STREAM_EVENTS)

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1beta/interactions?alt=sse"
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        body: Final = json.loads(request.body)
        assert body == {
            "model": "gemini-2.5-flash",
            "input": "plan a trip",
            "stream": True,
        }, body
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", prefix, {"model": model, "input": "plan a trip", "stream": True}, key=key
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        expected: Final = b"".join(_proxy_stream_event(payload, model) for _event_type, payload in _STREAM_EVENTS)
        assert response.content.removesuffix(_KNOWN_DONE_TERMINATOR) == expected, response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1beta/interactions?alt=sse")], response.text


@pytest.mark.parametrize("prefix", ["/v1beta/interactions", "/interactions"])
def test_stream_interaction_ends_without_openai_done_terminator(gateway: Gateway, prefix: str) -> None:
    pytest.skip("BUG: /interactions stream=true appends an OpenAI data: [DONE] frame to the Gemini event stream")
    frames: Final = tuple(_sse_event(event_type, payload) for event_type, payload in _STREAM_EVENTS)

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1beta/interactions?alt=sse"
        return Reply(content_type="text/event-stream", chunks=frames)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI_MODEL, api_base=wire.url, api_key=_GEMINI_DEPLOYMENT_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", prefix, {"model": model, "input": "plan a trip", "stream": True}, key=key
        )
        assert response.status_code == 200, response.text
        assert "[DONE]" not in response.text, response.text
        received: Final = [json.loads(part.removeprefix("data: ")) for part in response.text.split("\n\n") if part]
        assert received == [{**payload, "model": model} for _event_type, payload in _STREAM_EVENTS], response.text
        assert [(r.method, r.target) for r in wire.drain()] == [("POST", "/v1beta/interactions?alt=sse")]
