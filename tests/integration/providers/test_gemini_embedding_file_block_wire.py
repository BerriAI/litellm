from __future__ import annotations

import asyncio
import base64
import json
import uuid
import zlib
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse
from pydantic import JsonValue

BACKEND: Final = "gemini-embedding-001"
TARGET: Final = f"/models/{BACKEND}:batchEmbedContents"
CLIP: Final = base64.b64encode(b"\x00\x00\x00\x18ftypmp42" + bytes(24)).decode()
DATA_URI: Final = f"data:video/mp4;base64,{CLIP}"
GCS_URI: Final = "gs://scripted-bucket/clips/animals.mp4"
METADATA: Final[dict[str, JsonValue]] = {"fps": 1.0, "start_offset": "0s", "end_offset": "3s"}
WIRE_METADATA: Final[dict[str, JsonValue]] = {"fps": 1.0, "startOffset": "0s", "endOffset": "3s"}
INLINE_DATA: Final[dict[str, JsonValue]] = {"mime_type": "video/mp4", "data": CLIP}
INLINE_PART: Final[dict[str, JsonValue]] = {"inline_data": INLINE_DATA, "video_metadata": WIRE_METADATA}
TEXT_PART: Final[dict[str, JsonValue]] = {"text": "a red bus"}
ACCEPTED_FORMS: Final = "must be a data: URI, a gs:// URL, a files/ reference, or a Gemini Files API URI"
LONG_OFFSET: Final = "9" * 5000 + "s"


def block(**file: JsonValue) -> dict[str, JsonValue]:
    return {"type": "file", "file": file}


def gcs_part(uri: str = GCS_URI, metadata: dict[str, JsonValue] | None = WIRE_METADATA) -> dict[str, JsonValue]:
    file_data: Final[dict[str, JsonValue]] = {"file_data": {"mime_type": "video/mp4", "file_uri": uri}}
    return file_data if metadata is None else {**file_data, "video_metadata": metadata}


GCS_PART: Final = gcs_part()
NOISY_BLOCK: Final[dict[str, JsonValue]] = {
    **block(file_id=GCS_URI, mime_type="video/mp4", video_metadata={**METADATA, "frame_rate": 2}),
    "caption": "unknown block key",
}


def list_value(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value


def source_of(part: JsonValue) -> str:
    item: Final = object_value(part)
    if "text" in item:
        return string_value(item["text"])
    if "file_data" in item:
        return string_value(object_value(item["file_data"])["file_uri"])
    return string_value(object_value(item["inline_data"])["data"])


def vector(parts: list[JsonValue]) -> list[float]:
    return [zlib.crc32("|".join(source_of(part) for part in parts).encode()) / 2**32, 0.5]


def request_parts(request: Request) -> list[list[JsonValue]]:
    body: Final = object_value(json.loads(request.body))
    return [list_value(object_value(object_value(item)["content"])["parts"]) for item in list_value(body["requests"])]


def embed_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target.split("?")[0] == TARGET, request.target
    embeddings: Final = [{"values": vector(parts)} for parts in request_parts(request)]
    return Reply(body=json.dumps({"embeddings": embeddings}).encode())


def wire_parts(wire: Wire) -> list[list[JsonValue]]:
    received: Final = wire.drain()
    assert len(received) == 1, [item.target for item in received]
    return request_parts(received[0])


def embeddings(response: httpx.Response) -> list[JsonValue]:
    assert response.status_code == 200, response.text
    data: Final = [object_value(item) for item in list_value(object_value(response.json())["data"])]
    assert [item["index"] for item in data] == list(range(len(data))), response.text
    assert all(item["object"] == "embedding" for item in data), response.text
    return [item["embedding"] for item in data]


def gemini_model(scenario: Scenario, url: str, **litellm_params: JsonValue) -> str:
    return scenario.model(model=f"gemini/{BACKEND}", api_key="scripted-gemini-key", api_base=url, **litellm_params)


def embed(gateway: Gateway, model: str, elements: JsonValue, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/embeddings", {"model": model, "input": elements, **extra})


def proxy_root(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def write_config(directory: Path, url: str, name: str) -> Path:
    config: Final = directory / f"gemini_embeddings_{uuid.uuid4().hex}.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": name,
                        "litellm_params": {
                            "model": f"gemini/{BACKEND}",
                            "api_key": "scripted-gemini-key",
                            "api_base": url,
                        },
                    }
                ],
                "litellm_settings": {"drop_params": True},
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    return config


def test_string_input_reaches_the_wire_as_a_text_part(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        assert embeddings(embed(gateway, model, ["a red bus"])) == [vector([TEXT_PART])]
        assert wire_parts(wire) == [[TEXT_PART]]


@pytest.mark.parametrize(
    ("file", "part"),
    (
        pytest.param({"file_data": DATA_URI, "video_metadata": METADATA}, INLINE_PART, id="data-uri"),
        pytest.param({"file_id": GCS_URI, "video_metadata": METADATA}, GCS_PART, id="gcs"),
        pytest.param(
            {"file_id": GCS_URI, "video_metadata": {"fps": 1, "start_offset": "", "end_offset": LONG_OFFSET}},
            gcs_part(metadata={"fps": 1, "startOffset": "", "endOffset": LONG_OFFSET}),
            id="verbatim-strings-and-int-fps",
        ),
        pytest.param(
            {"file_data": DATA_URI, "format": "video/quicktime"},
            {"inline_data": {"mime_type": "video/quicktime", "data": CLIP}},
            id="format-overrides-the-data-uri-mime",
        ),
        pytest.param(
            {"file_id": "gs://scripted-bucket/clips/clip.bin", "format": "video/mp4"},
            gcs_part("gs://scripted-bucket/clips/clip.bin", None),
            id="format-names-an-unlisted-extension",
        ),
        pytest.param({"file_id": GCS_URI}, gcs_part(metadata=None), id="no-metadata"),
        pytest.param({"file_id": GCS_URI, "video_metadata": {}}, gcs_part(metadata=None), id="empty-metadata"),
    ),
)
def test_file_block_reaches_the_wire_as_one_part(
    gateway: Gateway, file: dict[str, JsonValue], part: dict[str, JsonValue]
) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        assert embeddings(embed(gateway, model, [block(**file)])) == [vector([part])]
        assert wire_parts(wire) == [[part]]


def test_nested_text_and_block_become_one_request_with_two_parts(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        response: Final = embed(gateway, model, [["a red bus", block(file_id=GCS_URI, video_metadata=METADATA)]])
        assert embeddings(response) == [vector([TEXT_PART, GCS_PART])]
        assert wire_parts(wire) == [[TEXT_PART, GCS_PART]]


def test_repeated_blocks_become_one_request_each(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        response: Final = embed(gateway, model, [block(file_id=GCS_URI, video_metadata=METADATA)] * 2)
        assert embeddings(response) == [vector([GCS_PART])] * 2
        assert wire_parts(wire) == [[GCS_PART], [GCS_PART]]


@pytest.mark.parametrize(
    ("element", "fragment"),
    (
        pytest.param(block(file_id=GCS_URI, video_metadata={"fps": "1"}), "file.video_metadata.fps", id="fps-string"),
        pytest.param(
            block(file_id=GCS_URI, video_metadata={"start_offset": 0}),
            "file.video_metadata.start_offset",
            id="offset-int",
        ),
        pytest.param(
            block(file_id=GCS_URI, video_metadata={"end_offset": ["3s"]}),
            "file.video_metadata.end_offset",
            id="offset-list",
        ),
        pytest.param(block(file_id=GCS_URI, video_metadata="1fps"), "file.video_metadata", id="metadata-string"),
        pytest.param(
            block(file_id=GCS_URI, video_metadata={**METADATA, "frame_rate": 2}),
            "frame_rate",
            id="unknown-metadata-key",
        ),
        pytest.param(block(file_id=GCS_URI, mime_type="video/mp4"), "mime_type", id="unknown-file-key"),
        pytest.param({**block(file_id=GCS_URI), "caption": "x"}, "caption", id="unknown-block-key"),
        pytest.param({"type": "video", "file": {"file_id": GCS_URI}}, "type", id="wrong-type"),
        pytest.param(
            block(file_id=GCS_URI, file_data=DATA_URI),
            "takes file.file_id or file.file_data, not both",
            id="both-sources",
        ),
        pytest.param(block(video_metadata=METADATA), "needs file.file_id or file.file_data", id="no-source"),
        pytest.param(block(file_data=CLIP), ACCEPTED_FORMS, id="bare-base64"),
        pytest.param(block(file_id="https://example.com/clip.mp4"), ACCEPTED_FORMS, id="http-url"),
        pytest.param(block(file_id=""), ACCEPTED_FORMS, id="empty-file-id"),
        pytest.param(block(file_id=GCS_URI, format=""), "file.format", id="empty-format"),
        pytest.param(block(file_data=5), "file.file_data", id="file-data-int"),
        pytest.param(block(file_data=[DATA_URI]), "file.file_data", id="file-data-list"),
        pytest.param(1, "got int", id="int-element"),
    ),
)
def test_malformed_blocks_answer_400_before_any_provider_call(
    gateway: Gateway, element: JsonValue, fragment: str
) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        response: Final = embed(gateway, model, [element])
        assert response.status_code == 400, response.text
        assert fragment in response.text, response.text
        assert wire.drain() == (), "the rejected input reached the provider"


def test_bare_block_input_answers_400_before_any_provider_call(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        response: Final = embed(gateway, model, block(file_id=GCS_URI, video_metadata=METADATA))
        assert response.status_code == 400, response.text
        assert "input must be a string or a list" in response.text, response.text
        assert wire.drain() == (), "the rejected input reached the provider"


def test_request_drop_params_strips_unknown_keys_at_every_level(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        assert embeddings(embed(gateway, model, [NOISY_BLOCK], drop_params=True)) == [vector([GCS_PART])]
        assert wire_parts(wire) == [[GCS_PART]]


def test_deployment_drop_params_strips_unknown_keys_at_every_level(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url, drop_params=True)
        assert embeddings(embed(gateway, model, [NOISY_BLOCK])) == [vector([GCS_PART])]
        assert wire_parts(wire) == [[GCS_PART]]


@pytest.mark.timeout(300)
def test_yaml_drop_params_strips_unknown_keys_at_every_level(gateway: Gateway, tmp_path: Path) -> None:
    name: Final = f"gemini-embeddings-{uuid.uuid4().hex}"
    with wire_server(embed_peer) as wire:
        config: Final = write_config(tmp_path, wire.url, name)
        with owned_proxy_process(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config) as owned:
            assert embeddings(embed(owned.gateway, name, [NOISY_BLOCK])) == [vector([GCS_PART])]
            assert wire_parts(wire) == [[GCS_PART]]


def test_openai_sdk_clients_send_blocks_through_the_proxy(gateway: Gateway) -> None:
    sync_uri: Final = "gs://scripted-bucket/clips/sync.mp4"
    async_uri: Final = "gs://scripted-bucket/clips/async.mp4"
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        with OpenAI(base_url=f"{proxy_root(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
            served: Final = client.post(
                "/embeddings",
                body={"model": model, "input": [block(file_id=sync_uri, video_metadata=METADATA)]},
                cast_to=CreateEmbeddingResponse,
            )
        assert served.data[0].embedding == vector([gcs_part(sync_uri)]), served.model_dump_json()
        assert wire_parts(wire) == [[gcs_part(sync_uri)]]

        async def drive() -> CreateEmbeddingResponse:
            async with AsyncOpenAI(base_url=f"{proxy_root(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
                return await client.post(
                    "/embeddings",
                    body={"model": model, "input": [block(file_id=async_uri, video_metadata=METADATA)]},
                    cast_to=CreateEmbeddingResponse,
                )

        served_async: Final = asyncio.run(drive())
        assert served_async.data[0].embedding == vector([gcs_part(async_uri)]), served_async.model_dump_json()
        assert wire_parts(wire) == [[gcs_part(async_uri)]]
