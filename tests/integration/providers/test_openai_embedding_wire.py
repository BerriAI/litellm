import base64
import json
import struct
from typing import Final, Literal

import httpx
import openai
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_MODEL: Final = "openai/text-embedding-3-small"
_SDK_KEY: Final = "synthetic-openai-key"
_INPUTS: Final = ("embedding alpha marker", "embedding bravo marker")
_RAW_INPUT: Final = "single raw embedding marker"
_VECTORS: Final = ((0.25, 0.5, 0.75), (-1.0, 0.125, 2.0))
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _Embedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["embedding"]
    index: int
    embedding: list[float]


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class _EmbeddingResponse(BaseModel):
    data: tuple[_Embedding, ...]
    usage: _Usage


def _encoded(vector: tuple[float, ...]) -> str:
    return base64.b64encode(struct.pack("<3f", *vector)).decode("ascii")


def _embedding_reply(embeddings: list[str | list[float]]) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": embedding}
                    for index, embedding in enumerate(embeddings)
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
        ).encode()
    )


# openai-python 2.33.0 defaults omitted encoding_format to base64: https://github.com/openai/openai-python/blob/v2.33.0/src/openai/resources/embeddings.py
def test_openai_sdk_embedding_preserves_base64_and_decodes_vectors(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/embeddings"
        assert request.headers["authorization"] == f"Bearer {_SDK_KEY}"
        assert request.headers["authorization"] != f"Bearer {virtual_key}"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "model": "text-embedding-3-small",
            "input": list(_INPUTS),
            "encoding_format": "base64",
            "dimensions": 3,
            "user": "u1",
        }
        return _embedding_reply([_encoded(vector) for vector in _VECTORS])

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=f"{wire.url}/v1",
            api_key=_SDK_KEY,
        )
        virtual_key: Final = scenario.key()
        with (
            httpx.Client(trust_env=False) as http_client,
            openai.OpenAI(
                base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
                api_key=virtual_key,
                max_retries=0,
                http_client=http_client,
            ) as client,
        ):
            response: Final = client.embeddings.create(
                model=model,
                input=list(_INPUTS),
                dimensions=3,
                user="u1",
            )
        assert [item.embedding for item in response.data] == [list(vector) for vector in _VECTORS], (
            response.model_dump_json()
        )
        assert [item.index for item in response.data] == [0, 1], response.model_dump_json()
        assert (response.usage.prompt_tokens, response.usage.total_tokens) == (4, 4), response.model_dump_json()
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/embeddings")]


@pytest.mark.parametrize("encoding_format", (None, "float"), ids=("omitted", "float"))
def test_openai_embedding_forwards_omitted_or_float_encoding_format(
    gateway: Gateway,
    encoding_format: str | None,
) -> None:
    expected_body: Final = (
        {"model": "text-embedding-3-small", "input": _RAW_INPUT}
        if encoding_format is None
        else {"model": "text-embedding-3-small", "input": _RAW_INPUT, "encoding_format": "float"}
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/embeddings"
        assert request.headers["authorization"] == f"Bearer {_SDK_KEY}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if encoding_format is None and "encoding_format" in body:
            return Reply(status=400, body=b'{"error":"encoding_format must be omitted"}')
        assert body == expected_body, body
        return _embedding_reply([[0.5, -0.25, 1.0]])

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=f"{wire.url}/v1",
            api_key=_SDK_KEY,
        )
        virtual_key: Final = scenario.key()
        request_body: Final = (
            {"model": model, "input": _RAW_INPUT}
            if encoding_format is None
            else {"model": model, "input": _RAW_INPUT, "encoding_format": encoding_format}
        )
        response: Final = gateway.request("POST", "/v1/embeddings", request_body, key=virtual_key)
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        assert [item.model_dump() for item in payload.data] == [
            {"object": "embedding", "index": 0, "embedding": [0.5, -0.25, 1.0]}
        ], response.text
        assert (payload.usage.prompt_tokens, payload.usage.total_tokens) == (4, 4), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/embeddings")]
