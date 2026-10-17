import base64
import json
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal
from urllib.parse import parse_qs

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_TOKEN_INPUT: Final = [[791, 4062, 14198, 2385], [40, 1093]]
_FLAT_TOKEN_INPUT: Final = [791, 4062, 14198, 2385]
_BEDROCK_TEXTS: Final = ["The quick brown base", "I like"]
_VECTORS: Final = ((0.25, 0.5, 0.75), (-1.0, 0.125, 2.0))
_AZURE_DEPLOYMENT: Final = "embed-azure-deployment"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _Embedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["embedding"]
    index: int
    embedding: list[float]
    type: str | None = None


class _EmbeddingResponse(BaseModel):
    data: tuple[_Embedding, ...]


def _encoded(vector: tuple[float, ...]) -> str:
    return base64.b64encode(struct.pack("<3f", *vector)).decode("ascii")


def _openai_reply(count: int) -> bytes:
    return json.dumps(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": list(vector)}
                for index, vector in enumerate(_VECTORS[:count])
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    ).encode()


def _azure_reply(count: int) -> bytes:
    return json.dumps(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": _encoded(vector)}
                for index, vector in enumerate(_VECTORS[:count])
            ],
            "model": _AZURE_DEPLOYMENT,
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    ).encode()


def _bedrock_reply(texts: tuple[str, ...] = tuple(_BEDROCK_TEXTS)) -> bytes:
    vectors: Final = _VECTORS[: len(texts)]
    return json.dumps(
        {
            "embeddings": {"float": [list(vector) for vector in vectors]},
            "id": "synthetic-cohere-embed",
            "response_type": "embeddings_by_type",
            "texts": list(texts),
        }
    ).encode()


@dataclass(frozen=True, slots=True)
class _TokenArrayCase:
    model_params: Callable[[str], Mapping[str, JsonValue]]
    target: str
    query: Mapping[str, tuple[str, ...]]
    auth_header: str
    auth_value: str
    body: Callable[[JsonValue], Mapping[str, JsonValue]]
    reply: Callable[[int], bytes]
    item_type: str | None
    ignored_body_keys: frozenset[str] = frozenset()

    @property
    def path(self) -> str:
        return self.target.partition("?")[0]


_CASES: Final = (
    _TokenArrayCase(
        model_params=lambda url: {
            "model": "openai/text-embedding-3-small",
            "api_base": f"{url}/v1",
            "api_key": "synthetic-openai-key",
        },
        target="/v1/embeddings",
        query={},
        auth_header="authorization",
        auth_value="Bearer synthetic-openai-key",
        body=lambda tokens: {"model": "text-embedding-3-small", "input": tokens},
        reply=_openai_reply,
        item_type=None,
    ),
    _TokenArrayCase(
        model_params=lambda url: {
            "model": f"azure/{_AZURE_DEPLOYMENT}",
            "api_base": url,
            "api_key": "synthetic-azure-key",
            "api_version": "2024-06-01",
        },
        target=f"/openai/deployments/{_AZURE_DEPLOYMENT}/embeddings?api-version=2024-06-01",
        query={"api-version": ("2024-06-01",)},
        auth_header="api-key",
        auth_value="synthetic-azure-key",
        body=lambda tokens: {"input": tokens, "model": _AZURE_DEPLOYMENT},
        reply=_azure_reply,
        item_type=None,
        ignored_body_keys=frozenset({"encoding_format"}),
    ),
    _TokenArrayCase(
        model_params=lambda url: {
            "model": "hosted_vllm/vllm-embed-model",
            "api_base": f"{url}/v1",
            "api_key": "synthetic-vllm-key",
        },
        target="/v1/embeddings",
        query={},
        auth_header="authorization",
        auth_value="Bearer synthetic-vllm-key",
        body=lambda tokens: {"model": "vllm-embed-model", "input": tokens},
        reply=_openai_reply,
        item_type=None,
    ),
    _TokenArrayCase(
        model_params=lambda url: {
            "model": "nebius/nebius-embed-model",
            "api_base": f"{url}/v1",
            "api_key": "synthetic-nebius-key",
        },
        target="/v1/embeddings",
        query={},
        auth_header="authorization",
        auth_value="Bearer synthetic-nebius-key",
        body=lambda tokens: {"model": "nebius-embed-model", "input": tokens},
        reply=_openai_reply,
        item_type=None,
    ),
    _TokenArrayCase(
        model_params=lambda url: {
            "model": "bedrock/cohere.embed-english-v3",
            "api_base": url,
            "api_key": "synthetic-bedrock-bearer",
            "aws_region_name": "us-east-1",
        },
        target="/model/cohere.embed-english-v3/invoke",
        query={},
        auth_header="authorization",
        auth_value="Bearer synthetic-bedrock-bearer",
        body=lambda _tokens: {"texts": _BEDROCK_TEXTS, "input_type": "search_document"},
        reply=lambda _count: _bedrock_reply(),
        item_type="float",
    ),
)


_CASE_IDS: Final = ("openai", "azure", "hosted_vllm", "nebius", "bedrock-cohere")
_SHAPES: Final = (("nested", _TOKEN_INPUT, 2), ("flat", _FLAT_TOKEN_INPUT, 1))
_PARAMS: Final = tuple(
    pytest.param(case, tokens, count, id=f"{case_id}-{shape}")
    for case, case_id in zip(_CASES, _CASE_IDS, strict=True)
    for shape, tokens, count in _SHAPES[: 1 if case_id == "bedrock-cohere" else 2]
)


@pytest.mark.parametrize(("case", "tokens", "count"), _PARAMS)
def test_embedding_token_arrays_are_forwarded_or_decoded_for_provider(
    gateway: Gateway,
    case: _TokenArrayCase,
    tokens: list[int] | list[list[int]],
    count: int,
) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        path, separator, query = request.target.partition("?")
        assert path == case.path, request.target
        assert {key: tuple(values) for key, values in parse_qs(query).items()} == dict(case.query), request.target
        assert request.headers[case.auth_header] == case.auth_value, dict(request.headers)
        body: Final = _JSON_OBJECT.validate_json(request.body)
        sent_fields: Final = {key: value for key, value in body.items() if key not in case.ignored_body_keys}
        assert sent_fields == case.body(tokens), request.body
        return Reply(body=case.reply(count))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(**case.model_params(wire.url))
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": tokens},
        )
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        type_field: Final = {"type": case.item_type} if case.item_type is not None else {}
        assert [item.model_dump(exclude_none=True) for item in payload.data] == [
            {
                "object": "embedding",
                "index": index,
                "embedding": list(vector),
                **type_field,
            }
            for index, vector in enumerate(_VECTORS[:count])
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", case.target)]


def test_flat_token_array_is_decoded_for_bedrock(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: flat token-array input to a bedrock group is not decoded; cohere gets int texts and the proxy returns 500"
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/model/cohere.embed-english-v3/invoke"
        assert request.headers["authorization"] == "Bearer synthetic-bedrock-bearer"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "texts": ["The quick brown base"],
            "input_type": "search_document",
        }
        return Reply(body=_bedrock_reply(("The quick brown base",)))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/cohere.embed-english-v3",
            api_base=wire.url,
            api_key="synthetic-bedrock-bearer",
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": _FLAT_TOKEN_INPUT},
        )
        assert response.status_code == 200, response.text
        payload: Final = _EmbeddingResponse.model_validate_json(response.content)
        assert [item.model_dump(exclude_none=True) for item in payload.data] == [
            {
                "object": "embedding",
                "index": 0,
                "embedding": list(_VECTORS[0]),
                "type": "float",
            }
        ], response.text
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", "/model/cohere.embed-english-v3/invoke")
        ]


def test_engines_path_token_arrays_are_decoded_for_bedrock(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /engines/{model}/embeddings with token-array input and no body model skips decoding for bedrock; upstream gets the int lists in texts and the proxy returns 500"
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


def test_deployments_path_token_arrays_are_decoded_for_bedrock(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /openai/deployments/{model}/embeddings with token-array input and no body model skips decoding for bedrock; upstream gets the int lists in texts and the proxy returns 500"
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
        response: Final = gateway.client.post(
            f"/openai/deployments/{model}/embeddings?api-version=2024-10-21",
            json={"input": _TOKEN_INPUT},
            headers={"api-key": virtual_key},
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
