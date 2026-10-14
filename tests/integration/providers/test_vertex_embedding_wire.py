import json
from typing import Final, Literal

import pytest
from integration._support.client import Gateway
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_PROJECT: Final = "scripted-embedding-project"
_LOCATION: Final = "us-central1"
_VERTEX_MODEL: Final = "text-embedding-005"
_VERTEX_INPUTS: Final = ("vertex embedding alpha", "vertex embedding bravo")
_GEMINI_INPUTS: Final = ("gemini embedding alpha", "gemini embedding bravo")
_VECTOR_A: Final = (1.0,) + (0.0,) * 255
_VECTOR_B: Final = (0.0, 1.0) + (0.0,) * 254
_VECTORS: Final = (_VECTOR_A, _VECTOR_B)
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


@pytest.mark.parametrize(
    ("embedding_input", "expected_inputs"),
    ((list(_VERTEX_INPUTS), _VERTEX_INPUTS), (_VERTEX_INPUTS[0], (_VERTEX_INPUTS[0],))),
    ids=("list", "string"),
)
def test_vertex_text_embedding_sends_dimensionality_and_preserves_input_order(
    gateway: Gateway,
    embedding_input: str | list[str],
    expected_inputs: tuple[str, ...],
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == (
            f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_VERTEX_MODEL}:predict"
        )
        assert request.headers["authorization"] == "Bearer scripted-token"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "instances": [{"content": text} for text in expected_inputs],
            "parameters": {"outputDimensionality": 256},
        }
        return Reply(
            body=json.dumps(
                {
                    "predictions": [
                        {
                            "embeddings": {
                                "values": list(_VECTORS[index]),
                                "statistics": {"token_count": index + 2, "truncated": False},
                            }
                        }
                        for index, _ in enumerate(expected_inputs)
                    ]
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"vertex_ai/{_VERTEX_MODEL}",
            api_base=wire.url,
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        )
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": embedding_input, "dimensions": 256},
        )
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        assert [item.model_dump() for item in payload.data] == [
            {"object": "embedding", "index": index, "embedding": list(_VECTORS[index])}
            for index, _ in enumerate(expected_inputs)
        ], response.text
        expected_tokens: Final = sum(index + 2 for index in range(len(expected_inputs)))
        assert (payload.usage.prompt_tokens, payload.usage.total_tokens) == (
            expected_tokens,
            expected_tokens,
        ), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            (
                "POST",
                f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_VERTEX_MODEL}:predict",
            )
        ]


@pytest.mark.parametrize(
    ("embedding_input", "expected_inputs"),
    ((list(_GEMINI_INPUTS), _GEMINI_INPUTS), (_GEMINI_INPUTS[0], (_GEMINI_INPUTS[0],))),
    ids=("list", "string"),
)
def test_gemini_embedding_sends_batch_contents_and_dimensionality(
    gateway: Gateway,
    embedding_input: str | list[str],
    expected_inputs: tuple[str, ...],
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/models/gemini-embedding-001:batchEmbedContents"
        assert request.headers["x-goog-api-key"] == "synthetic-gemini-key"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "requests": [
                {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": 256,
                }
                for text in expected_inputs
            ]
        }
        return Reply(
            body=json.dumps(
                {
                    "embeddings": [
                        {"values": list(_VECTORS[index]), "statistics": {"token_count": index + 2}}
                        for index, _ in enumerate(expected_inputs)
                    ]
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="gemini/gemini-embedding-001",
            api_base=wire.url,
            api_key="synthetic-gemini-key",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": embedding_input, "dimensions": 256},
        )
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        assert [item.model_dump() for item in payload.data] == [
            {"object": "embedding", "index": index, "embedding": list(_VECTORS[index])}
            for index, _ in enumerate(expected_inputs)
        ], response.text
        assert payload.usage.prompt_tokens == payload.usage.total_tokens, response.text
        assert payload.usage.prompt_tokens > 0, response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/models/gemini-embedding-001:batchEmbedContents")
        ]
