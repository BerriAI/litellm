import base64
import json
import struct
from typing import Final, Literal
from urllib.parse import parse_qs

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_DEPLOYMENT: Final = "embedding-deployment"
_INPUTS: Final = ("azure embedding alpha", "azure embedding bravo")
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


def test_azure_embedding_uses_deployment_api_version_and_normalizes_response(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        path, separator, query = request.target.partition("?")
        assert path == f"/openai/deployments/{_DEPLOYMENT}/embeddings", request.target
        assert separator == "?", request.target
        assert parse_qs(query) == {"api-version": ["2024-06-01"]}, request.target
        assert request.headers["api-key"] == "synthetic-azure-key"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        # openai-python 2.33.0 defaults omitted encoding_format to base64: https://github.com/openai/openai-python/blob/v2.33.0/src/openai/resources/embeddings.py
        assert body == {
            "model": _DEPLOYMENT,
            "input": list(_INPUTS),
            "dimensions": 256,
            "encoding_format": "base64",
        }, body
        embedding_values: Final = [_encoded(vector) for vector in _VECTORS]
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": index, "embedding": embedding}
                        for index, embedding in enumerate(embedding_values)
                    ],
                    "model": _DEPLOYMENT,
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{_DEPLOYMENT}",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="2024-06-01",
        )
        virtual_key: Final = scenario.key()
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": list(_INPUTS), "dimensions": 256},
            key=virtual_key,
        )
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        assert [item.model_dump() for item in payload.data] == [
            {"object": "embedding", "index": index, "embedding": list(vector)} for index, vector in enumerate(_VECTORS)
        ], response.text
        assert (payload.usage.prompt_tokens, payload.usage.total_tokens) == (4, 4), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"/openai/deployments/{_DEPLOYMENT}/embeddings?api-version=2024-06-01")
        ]
