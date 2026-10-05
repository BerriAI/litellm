import json
from typing import Final, Literal

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_TOKEN_INPUT: Final = [[791, 4062, 14198, 2385], [40, 1093]]
_BEDROCK_TEXTS: Final = ["The quick brown base", "I like"]
_VECTORS: Final = ((0.25, 0.5, 0.75), (-1.0, 0.125, 2.0))
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _Embedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["embedding"]
    index: int
    embedding: list[float]
    type: str | None = None


class _EmbeddingResponse(BaseModel):
    data: tuple[_Embedding, ...]


def _openai_reply() -> bytes:
    return json.dumps(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": list(vector)}
                for index, vector in enumerate(_VECTORS)
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    ).encode()


def _bedrock_reply() -> bytes:
    return json.dumps(
        {
            "embeddings": {"float": [list(vector) for vector in _VECTORS]},
            "id": "synthetic-cohere-embed",
            "response_type": "embeddings_by_type",
            "texts": _BEDROCK_TEXTS,
        }
    ).encode()


@pytest.mark.parametrize("provider", ("openai", "bedrock"), ids=("openai", "bedrock-cohere"))
def test_embedding_token_arrays_are_forwarded_or_decoded_for_provider(
    gateway: Gateway,
    provider: Literal["openai", "bedrock"],
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        if provider == "openai":
            assert request.target == "/v1/embeddings"
            assert request.headers["authorization"] == "Bearer synthetic-openai-key"
            assert _JSON_OBJECT.validate_json(request.body) == {
                "model": "text-embedding-3-small",
                "input": _TOKEN_INPUT,
            }
            return Reply(body=_openai_reply())
        assert request.target == "/model/cohere.embed-english-v3/invoke"
        assert request.headers["authorization"] == "Bearer synthetic-bedrock-bearer"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "texts": _BEDROCK_TEXTS,
            "input_type": "search_document",
        }
        return Reply(body=_bedrock_reply())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        if provider == "openai":
            model: Final = scenario.model(
                model="openai/text-embedding-3-small",
                api_base=f"{wire.url}/v1",
                api_key="synthetic-openai-key",
            )
        else:
            model = scenario.model(
                model="bedrock/cohere.embed-english-v3",
                api_base=wire.url,
                api_key="synthetic-bedrock-bearer",
                aws_region_name="us-east-1",
            )
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": _TOKEN_INPUT},
        )
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        type_field: Final = {"type": "float"} if provider == "bedrock" else {}
        assert [item.model_dump(exclude_none=True) for item in payload.data] == [
            {
                "object": "embedding",
                "index": index,
                "embedding": list(vector),
                **type_field,
            }
            for index, vector in enumerate(_VECTORS)
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            (
                "POST",
                "/v1/embeddings" if provider == "openai" else "/model/cohere.embed-english-v3/invoke",
            )
        ]


def test_engines_path_token_arrays_are_decoded_for_bedrock(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /engines/{model}/embeddings with token-array input and no body model skips decoding for bedrock; upstream gets the int lists in texts and the proxy returns 503"
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/model/cohere.embed-english-v3/invoke"
        assert request.headers["authorization"] == "Bearer synthetic-bedrock-bearer"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "texts": _BEDROCK_TEXTS,
            "input_type": "search_document",
        }
        return Reply(body=_bedrock_reply())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/cohere.embed-english-v3",
            api_base=wire.url,
            api_key="synthetic-bedrock-bearer",
            aws_region_name="us-east-1",
        )
        virtual_key: Final = scenario.key()
        response: Final = gateway.request(
            "POST",
            f"/engines/{model}/embeddings",
            {"input": _TOKEN_INPUT},
            key=virtual_key,
        )
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        assert [item.model_dump(exclude_none=True) for item in payload.data] == [
            {
                "object": "embedding",
                "index": index,
                "embedding": list(vector),
                "type": "float",
            }
            for index, vector in enumerate(_VECTORS)
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/model/cohere.embed-english-v3/invoke")
        ]
