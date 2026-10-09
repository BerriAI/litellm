from __future__ import annotations

import asyncio
import json
import threading
import zlib
from collections.abc import Mapping
from hashlib import sha256
from typing import Final

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, Wire, wire_server

BACKEND: Final = "gemini-embedding-001"
TARGET: Final = f"/models/{BACKEND}:batchEmbedContents"
METADATA: Final[dict[str, JsonValue]] = {"fps": 1.0, "start_offset": "0s", "end_offset": "3s"}
WIRE_METADATA: Final[dict[str, JsonValue]] = {"fps": 1.0, "startOffset": "0s", "endOffset": "3s"}
TEXT_PART: Final[dict[str, JsonValue]] = {"text": "a red bus"}
SETTLE_SECONDS: Final = 15
SPEND_SQL: Final = (
    'SELECT request_id, cache_hit, spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime", request_id'
)


def clip(name: str) -> str:
    return f"gs://scripted-bucket/cache/{name}.mp4"


def block(name: str, metadata: Mapping[str, JsonValue] | None = METADATA) -> dict[str, JsonValue]:
    file: Final[dict[str, JsonValue]] = {"file_id": clip(name)}
    return {"type": "file", "file": file if metadata is None else {**file, "video_metadata": dict(metadata)}}


def gcs_part(name: str, metadata: Mapping[str, JsonValue] | None = WIRE_METADATA) -> dict[str, JsonValue]:
    part: Final[dict[str, JsonValue]] = {"file_data": {"mime_type": "video/mp4", "file_uri": clip(name)}}
    return part if metadata is None else {**part, "video_metadata": dict(metadata)}


def list_value(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value


def source_of(part: JsonValue) -> str:
    item: Final = object_value(part)
    if "text" in item:
        return string_value(item["text"])
    return string_value(object_value(item["file_data"])["file_uri"])


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


def embeddings_of(answer: Mapping[str, JsonValue]) -> list[JsonValue]:
    data: Final = [object_value(item) for item in list_value(answer["data"])]
    assert [item["index"] for item in data] == list(range(len(data))), answer
    return [item["embedding"] for item in data]


def gemini_model(scenario: Scenario, url: str) -> str:
    return scenario.model(model=f"gemini/{BACKEND}", api_key="scripted-gemini-key", api_base=url)


def embed_body(model: str, elements: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": elements}


def spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),))


def served_from_cache(
    gateway: Gateway, wire: Wire, body: Mapping[str, JsonValue], key: str
) -> dict[str, JsonValue] | None:
    answer: Final = gateway.post("/v1/embeddings", body, key=key)
    return None if wire.drain() else answer


def await_hit(gateway: Gateway, wire: Wire, body: Mapping[str, JsonValue], key: str) -> dict[str, JsonValue]:
    served: Final = eventually(
        lambda: served_from_cache(gateway, wire, body, key), lambda answer: answer is not None, SETTLE_SECONDS
    )
    assert served is not None
    return served


def proxy_root(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def test_block_embedding_is_served_from_the_cache_with_a_zero_spend_row(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        key: Final = scenario.key(models=[model])
        body: Final = embed_body(model, [block("one")])
        first: Final = gateway.request("POST", "/v1/embeddings", body, key=key)
        assert first.status_code == 200, first.text
        first_id: Final = first.headers["x-litellm-call-id"]
        assert wire_parts(wire) == [[gcs_part("one")]]
        served: Final = await_hit(gateway, wire, body, key)
        assert embeddings_of(served) == embeddings_of(object_value(first.json())) == [vector([gcs_part("one")])]
        rows: Final = eventually(
            lambda: spend_rows(key), lambda found: any(row["cache_hit"] == "True" for row in found), seconds=70
        )
        by_id: Final = {string_value(row["request_id"]): row for row in rows}
        assert by_id[first_id]["cache_hit"] != "True", rows
        hits: Final = [row for row in rows if row["cache_hit"] == "True"]
        assert hits and all(float(str(row["spend"])) == 0 for row in hits), rows


def test_cached_block_is_not_resent_when_a_new_text_joins_it(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        key: Final = scenario.key(models=[model])
        cached: Final = embed_body(model, [block("one")])
        gateway.post("/v1/embeddings", cached, key=key)
        assert wire_parts(wire) == [[gcs_part("one")]]
        await_hit(gateway, wire, cached, key)
        mixed: Final = gateway.post("/v1/embeddings", embed_body(model, [block("one"), "a red bus"]), key=key)
        assert wire_parts(wire) == [[TEXT_PART]]
        assert embeddings_of(mixed) == [vector([gcs_part("one")]), vector([TEXT_PART])]


@pytest.mark.parametrize("names", (("a", "b", "c", "d", "e"), ("same", "same")), ids=("five-distinct", "two-identical"))
def test_block_lists_are_cached_whole(gateway: Gateway, names: tuple[str, ...]) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        key: Final = scenario.key(models=[model])
        body: Final = embed_body(model, [block(name) for name in names])
        first: Final = gateway.post("/v1/embeddings", body, key=key)
        assert wire_parts(wire) == [[gcs_part(name)] for name in names]
        assert embeddings_of(first) == [vector([gcs_part(name)]) for name in names]
        assert embeddings_of(await_hit(gateway, wire, body, key)) == embeddings_of(first)


def test_provider_failure_is_not_cached(gateway: Gateway) -> None:
    healed: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if healed.is_set():
            return embed_peer(request)
        return Reply(status=500, body=b'{"error": {"message": "scripted outage"}}')

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        key: Final = scenario.key(models=[model])
        body: Final = embed_body(model, [block("flaky")])
        failed: Final = gateway.request("POST", "/v1/embeddings", body, key=key)
        assert failed.status_code >= 500, failed.text
        assert wire.drain(), "the failing call never reached the provider"
        healed.set()
        recovered: Final = gateway.post("/v1/embeddings", body, key=key)
        assert wire_parts(wire) == [[gcs_part("flaky")]]
        assert embeddings_of(await_hit(gateway, wire, body, key)) == embeddings_of(recovered)


def test_video_metadata_is_part_of_the_cache_key(gateway: Gateway) -> None:
    two_seconds: Final[dict[str, JsonValue]] = {**METADATA, "end_offset": "2s"}
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        key: Final = scenario.key(models=[model])
        settled: Final = embed_body(model, [block("keyed")])
        gateway.post("/v1/embeddings", settled, key=key)
        assert wire_parts(wire) == [[gcs_part("keyed")]]
        await_hit(gateway, wire, settled, key)
        variants: Final = (
            (block("keyed", two_seconds), gcs_part("keyed", {**WIRE_METADATA, "endOffset": "2s"})),
            (block("keyed", None), gcs_part("keyed", None)),
        )
        for variant, part in variants:
            gateway.post("/v1/embeddings", embed_body(model, [variant]), key=key)
            assert wire_parts(wire) == [[part]]


def test_openai_sdk_clients_fill_and_hit_the_cache(gateway: Gateway) -> None:
    with wire_server(embed_peer) as wire, gateway.scenario() as scenario:
        model: Final = gemini_model(scenario, wire.url)
        key: Final = scenario.key(models=[model])
        body: Final = embed_body(model, [block("sdk")])
        with OpenAI(base_url=f"{proxy_root(gateway)}/v1", api_key=key, max_retries=0) as client:
            filled: Final = client.post("/embeddings", body=body, cast_to=CreateEmbeddingResponse)
        assert wire_parts(wire) == [[gcs_part("sdk")]]
        await_hit(gateway, wire, body, key)

        async def drive() -> CreateEmbeddingResponse:
            async with AsyncOpenAI(base_url=f"{proxy_root(gateway)}/v1", api_key=key, max_retries=0) as client:
                return await client.post("/embeddings", body=body, cast_to=CreateEmbeddingResponse)

        served: Final = asyncio.run(drive())
        assert served.data[0].embedding == filled.data[0].embedding == vector([gcs_part("sdk")])
        assert wire.drain() == (), "the cached answer reached the provider again"
