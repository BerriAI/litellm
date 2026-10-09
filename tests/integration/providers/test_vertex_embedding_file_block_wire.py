from __future__ import annotations

import json
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, object_value
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

BACKEND: Final = "gemini-embedding-2-preview"
PROJECT: Final = "scripted-project"
LOCATION: Final = "us-central1"
TARGET: Final = f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{BACKEND}:embedContent"
GCS_URI: Final = "gs://scripted-bucket/clips/animals.mp4"
METADATA: Final[dict[str, JsonValue]] = {"fps": 1.0, "start_offset": "0s", "end_offset": "3s"}
GCS_PART: Final[dict[str, JsonValue]] = {
    "file_data": {"mime_type": "video/mp4", "file_uri": GCS_URI},
    "video_metadata": {"fps": 1.0, "startOffset": "0s", "endOffset": "3s"},
}
TEXT_PART: Final[dict[str, JsonValue]] = {"text": "a red bus"}
VALUES: Final = [0.75, 0.25]


def block(**file: JsonValue) -> dict[str, JsonValue]:
    return {"type": "file", "file": file}


def embed_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target == TARGET, request.target
    assert request.headers["authorization"] == "Bearer scripted-token"
    return Reply(body=json.dumps({"embedding": {"values": VALUES}}).encode())


def wire_parts(wire: Wire) -> list[JsonValue]:
    received: Final = wire.drain()
    assert len(received) == 1, [item.target for item in received]
    body: Final = object_value(json.loads(received[0].body))
    parts: Final = object_value(body["content"])["parts"]
    assert isinstance(parts, list), body
    return parts


def vertex_model(gateway: Gateway, scenario: Scenario, url: str, **litellm_params: JsonValue) -> str:
    return scenario.model(
        model=f"vertex_ai/{BACKEND}",
        api_key=None,
        api_base=url,
        vertex_project=PROJECT,
        vertex_location=LOCATION,
        vertex_credentials=service_account_json(PROJECT, gateway.upstream_url),
        **litellm_params,
    )


def embed(gateway: Gateway, model: str, elements: JsonValue, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/embeddings", {"model": model, "input": elements, **extra})


def single_embedding(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    data: Final = object_value(response.json())["data"]
    assert isinstance(data, list) and len(data) == 1, response.text
    item: Final = object_value(data[0])
    assert item["index"] == 0 and item["object"] == "embedding", response.text
    return item["embedding"]


def rejected(gateway: Gateway, wire: Wire, model: str, elements: JsonValue, fragment: str) -> None:
    response: Final = embed(gateway, model, elements)
    assert response.status_code == 400, response.text
    assert fragment in response.text, response.text
    assert wire.drain() == (), "the rejected input reached the provider"


def test_string_input_reaches_embed_content_as_a_text_part(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        assert single_embedding(embed(gateway, model, ["a red bus"])) == VALUES
        assert wire_parts(wire) == [TEXT_PART]


def test_file_block_with_video_metadata_reaches_embed_content(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        assert single_embedding(embed(gateway, model, [block(file_id=GCS_URI, video_metadata=METADATA)])) == VALUES
        assert wire_parts(wire) == [GCS_PART]


def test_request_drop_params_strips_unknown_keys_at_every_level(gateway: Gateway) -> None:
    noisy: Final[dict[str, JsonValue]] = {
        **block(file_id=GCS_URI, mime_type="video/mp4", video_metadata={**METADATA, "frame_rate": 2}),
        "caption": "unknown block key",
    }
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        assert single_embedding(embed(gateway, model, [noisy], drop_params=True)) == VALUES
        assert wire_parts(wire) == [GCS_PART]


def test_bad_fps_answers_400_before_any_provider_call(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        rejected(gateway, wire, model, [block(file_id=GCS_URI, video_metadata={"fps": "1"})], "file.video_metadata.fps")


def test_files_api_reference_answers_400_naming_the_gemini_provider(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        rejected(
            gateway,
            wire,
            model,
            [block(file_id="files/abc", video_metadata=METADATA)],
            "Gemini Files API references are only supported through the gemini/ provider",
        )
