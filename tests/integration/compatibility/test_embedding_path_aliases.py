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

_TEXT: Final = "path alias embedding marker"
_VECTOR: Final = (0.25, 0.5, 0.75)
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


def _assert_raw_embedding_response(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    payload: Final = _EmbeddingResponse.model_validate_json(response.content)
    assert [item.model_dump() for item in payload.data] == [
        {"object": "embedding", "index": 0, "embedding": list(_VECTOR)}
    ], response.text
    assert (payload.usage.prompt_tokens, payload.usage.total_tokens) == (1, 1), response.text


def _reply(encoding_format: str | None) -> Reply:
    embedding: Final = (
        base64.b64encode(struct.pack("<3f", *_VECTOR)).decode("ascii") if encoding_format == "base64" else list(_VECTOR)
    )
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": embedding}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        ).encode()
    )


_AZURE_SDK_BODY: Final = {
    "model": "text-embedding-3-small",
    "input": _TEXT,
    "encoding_format": "base64",
}


def _azure_sdk_embedding(
    gateway: Gateway,
    deployment: str | None,
    api_key: str,
    model: str,
) -> openai.types.CreateEmbeddingResponse:
    with (
        httpx.Client(trust_env=False) as http_client,
        openai.AzureOpenAI(
            azure_endpoint=str(gateway.client.base_url).rstrip("/"),
            azure_deployment=deployment,
            api_key=api_key,
            api_version="2024-10-21",
            max_retries=0,
            http_client=http_client,
        ) as client,
    ):
        return client.embeddings.create(model=model, input=_TEXT)


@pytest.mark.parametrize(
    "surface",
    ("azure_sdk", "azure_sdk_path_over_body", "deployments_raw", "engines_raw", "bare"),
    ids=("azure-sdk", "azure-sdk-path-over-body", "deployments-path", "engines-path", "bare-model"),
)
def test_embedding_model_is_resolved_from_each_endpoint_surface(
    gateway: Gateway,
    surface: Literal["azure_sdk", "azure_sdk_path_over_body", "deployments_raw", "engines_raw", "bare"],
) -> None:
    expected_key: Final = (
        "synthetic-key-a"
        if surface in ("azure_sdk", "azure_sdk_path_over_body", "deployments_raw")
        else "synthetic-key-b"
    )
    # openai-python 2.33.0 defaults omitted encoding_format to base64: https://github.com/openai/openai-python/blob/v2.33.0/src/openai/resources/embeddings.py
    expected_body: Final = (
        _AZURE_SDK_BODY
        if surface in ("azure_sdk", "azure_sdk_path_over_body")
        else {"model": "text-embedding-3-small", "input": _TEXT}
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/embeddings"
        assert request.headers["authorization"] == f"Bearer {expected_key}"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body == expected_body, body
        return _reply("base64" if surface in ("azure_sdk", "azure_sdk_path_over_body") else None)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        group_a: Final = scenario.model(
            model="openai/text-embedding-3-small",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-key-a",
        )
        group_b: Final = scenario.model(
            model="openai/text-embedding-3-small",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-key-b",
        )
        virtual_key: Final = scenario.key()

        if surface in ("azure_sdk", "azure_sdk_path_over_body"):
            deployment: Final = group_a if surface == "azure_sdk_path_over_body" else None
            request_model: Final = group_b if surface == "azure_sdk_path_over_body" else group_a
            response: Final = _azure_sdk_embedding(gateway, deployment, virtual_key, request_model)
            if surface == "azure_sdk":
                assert response.model == group_a, response.model_dump_json()
            assert [item.embedding for item in response.data] == [list(_VECTOR)], response.model_dump_json()
            assert [item.index for item in response.data] == [0], response.model_dump_json()
            assert (response.usage.prompt_tokens, response.usage.total_tokens) == (1, 1), response.model_dump_json()
        elif surface == "deployments_raw":
            raw_response: Final = gateway.client.post(
                f"/openai/deployments/{group_a}/embeddings?api-version=2024-10-21",
                json={"input": _TEXT},
                headers={"api-key": virtual_key},
            )
            _assert_raw_embedding_response(raw_response)
        elif surface == "engines_raw":
            raw_response = gateway.request(
                "POST",
                f"/engines/{group_b}/embeddings",
                {"input": _TEXT},
                key=virtual_key,
            )
            _assert_raw_embedding_response(raw_response)
        else:
            raw_response = gateway.request(
                "POST",
                "/embeddings",
                {"model": group_b, "input": _TEXT},
                key=virtual_key,
            )
            _assert_raw_embedding_response(raw_response)
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/embeddings")]


def test_azure_sdk_response_reports_the_path_group_that_served_it(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: AzureOpenAI with azure_deployment=<group_a> and model=<group_b> is served by group_a but the response model reports group_b"
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer synthetic-key-a"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body == _AZURE_SDK_BODY, body
        return _reply("base64")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        group_a: Final = scenario.model(
            model="openai/text-embedding-3-small",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-key-a",
        )
        group_b: Final = scenario.model(
            model="openai/text-embedding-3-small",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-key-b",
        )
        virtual_key: Final = scenario.key()
        response: Final = _azure_sdk_embedding(gateway, group_a, virtual_key, group_b)
        assert response.model == group_a, response.model_dump_json()
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/embeddings")]


def test_deployments_embedding_rejects_invalid_key_without_upstream_request(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"Invalid key reached the peer: {request.method} {request.target}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/text-embedding-3-small",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-key-a",
        )
        response: Final = gateway.client.post(
            f"/openai/deployments/{model}/embeddings?api-version=2024-10-21",
            json={"input": _TEXT},
            headers={"api-key": "sk-not-a-real-key"},
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == ()
