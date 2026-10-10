from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

BACKEND: Final = "gemini-3.7-flash"
PROJECT: Final = "scripted-project"
LOCATION: Final = "us-central1"
MODEL_PATH: Final = f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{BACKEND}"
CLIP: Final = base64.b64encode(b"\x00\x00\x00\x18ftypmp42" + bytes(24)).decode()
DATA_URI: Final = f"data:video/mp4;base64,{CLIP}"
PROMPT: Final = "describe the clip"
FILE_BLOCK: Final[dict[str, JsonValue]] = {
    "type": "file",
    "file": {"file_data": DATA_URI, "video_metadata": {"fps": 1.0, "start_offset": "0s", "end_offset": "3s"}},
}
TEXT_PART: Final[dict[str, JsonValue]] = {"text": PROMPT}
BARE_VIDEO_PART: Final[dict[str, JsonValue]] = {"inline_data": {"mime_type": "video/mp4", "data": CLIP}}
VIDEO_PART: Final[dict[str, JsonValue]] = {
    **BARE_VIDEO_PART,
    "video_metadata": {"fps": 1.0, "startOffset": "0s", "endOffset": "3s"},
}
REPLY: Final[dict[str, JsonValue]] = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "a cat"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
}


def chat_peer(path: str, authorization: str | None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request.method
        assert authorization is None or request.headers["authorization"] == authorization
        target, _, query = request.target.partition("?")
        if target == f"{path}:streamGenerateContent":
            assert "alt=sse" in query, request.target
            return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(REPLY)}\n\n".encode(),))
        assert target == f"{path}:generateContent", request.target
        return Reply(body=json.dumps(REPLY).encode())

    return respond


def wire_parts(wire: Wire) -> list[JsonValue]:
    received: Final = wire.drain()
    assert len(received) == 1, [item.target for item in received]
    body: Final = object_value(json.loads(received[0].body))
    contents: Final = body["contents"]
    assert isinstance(contents, list) and len(contents) == 1, body
    parts: Final = object_value(contents[0])["parts"]
    assert isinstance(parts, list), body
    return parts


def gemini_model(scenario: Scenario, url: str) -> str:
    return scenario.model(model=f"gemini/{BACKEND}", api_key="scripted-gemini-key", api_base=url)


def vertex_model(gateway: Gateway, scenario: Scenario, url: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{BACKEND}",
        api_key=None,
        api_base=url,
        vertex_project=PROJECT,
        vertex_location=LOCATION,
        vertex_credentials=service_account_json(PROJECT, gateway.upstream_url),
    )


def chat(gateway: Gateway, model: str, stream: bool) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}, FILE_BLOCK]}],
        "stream": stream,
    }
    return gateway.request("POST", "/v1/chat/completions", body)


def answered(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    assert "a cat" in response.text, response.text


@pytest.mark.parametrize("stream", (False, True), ids=("non-stream", "stream"))
def test_gemini_chat_file_block_video_metadata_reaches_the_wire(gateway: Gateway, stream: bool) -> None:
    with wire_server(chat_peer(f"/models/{BACKEND}", None)) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        answered(chat(gateway, model, stream))
        assert wire_parts(wire) == [TEXT_PART, VIDEO_PART]


@pytest.mark.parametrize("stream", (False, True), ids=("non-stream", "stream"))
def test_vertex_chat_file_block_video_metadata_reaches_the_wire(gateway: Gateway, stream: bool) -> None:
    with wire_server(chat_peer(MODEL_PATH, "Bearer scripted-token")) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        answered(chat(gateway, model, stream))
        assert wire_parts(wire) == [TEXT_PART, VIDEO_PART]


def test_responses_input_file_video_reaches_the_wire_as_inline_data(gateway: Gateway) -> None:
    with wire_server(chat_peer(f"/models/{BACKEND}", None)) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": PROMPT},
                        {"type": "input_file", "file_data": DATA_URI, "filename": "clip.mp4"},
                    ],
                }
            ],
        }
        answered(gateway.request("POST", "/v1/responses", body))
        assert wire_parts(wire) == [TEXT_PART, BARE_VIDEO_PART]


def test_messages_document_video_reaches_the_wire_as_inline_data(gateway: Gateway) -> None:
    with wire_server(chat_peer(f"/models/{BACKEND}", None)) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "document", "source": {"type": "base64", "media_type": "video/mp4", "data": CLIP}},
                    ],
                }
            ],
        }
        answered(gateway.request("POST", "/v1/messages", body))
        assert wire_parts(wire) == [TEXT_PART, BARE_VIDEO_PART]
