import json
from typing import Final, Literal

import httpx
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_VECTORS: Final = {
    "alpha": (1.0, 0.0, 0.0),
    "bravo": (0.0, 1.0, 0.0),
    "charlie": (0.0, 0.0, 1.0),
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TEXT_LIST: Final = TypeAdapter(list[str])


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


def _input_values(request: Request) -> tuple[str, ...]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    return tuple(_TEXT_LIST.validate_python(body["input"]))


def _expected_data(inputs: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return [
        {
            "object": "embedding",
            "index": index,
            "embedding": list(_VECTORS[text]),
        }
        for index, text in enumerate(inputs)
    ]


def test_embedding_cache_reuses_cached_items_during_partial_hits(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer synthetic-cache-key"
        body: Final = _JSON_OBJECT.validate_json(request.body)
        inputs: Final = tuple(_TEXT_LIST.validate_python(body["input"]))
        assert body == {"model": "text-embedding-3-small", "input": list(inputs)}
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": list(_VECTORS[text]),
                        }
                        for index, text in enumerate(inputs)
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/text-embedding-3-small",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-cache-key",
        )

        def attempt_alpha() -> tuple[httpx.Response, tuple[Request, ...]]:
            response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": ["alpha"]})
            return response, wire.drain()

        first_warm: Final = attempt_alpha()
        assert first_warm[0].status_code == 200, first_warm[0].text
        assert len(first_warm[1]) == 1, "The unique embedding group unexpectedly hit the response cache"
        assert _input_values(first_warm[1][0]) == ("alpha",)
        warm_cache_hit: Final = eventually(attempt_alpha, lambda attempt: not attempt[1])
        assert warm_cache_hit[0].status_code == 200, warm_cache_hit[0].text
        warm_payload: Final = _EmbeddingResponse.model_validate_json(warm_cache_hit[0].content)
        assert [item.model_dump() for item in warm_payload.data] == _expected_data(("alpha",)), warm_cache_hit[0].text
        assert (warm_payload.usage.prompt_tokens, warm_payload.usage.total_tokens) == (
            1,
            1,
        ), warm_cache_hit[0].text

        partial: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": ["bravo", "alpha", "charlie"]},
        )
        partial_requests: Final = wire.drain()
        assert partial.status_code == 200, partial.text
        assert len(partial_requests) == 1, f"Expected one uncached provider call, received {len(partial_requests)}"
        assert _input_values(partial_requests[0]) == ("bravo", "charlie")
        partial_payload: Final = _EmbeddingResponse.model_validate_json(partial.content)
        assert [item.model_dump() for item in partial_payload.data] == _expected_data(("bravo", "alpha", "charlie")), (
            partial.text
        )
        assert (partial_payload.usage.prompt_tokens, partial_payload.usage.total_tokens) == (
            3,
            3,
        ), partial.text

        def attempt_partial() -> tuple[httpx.Response, tuple[Request, ...]]:
            response: Final = gateway.request(
                "POST",
                "/v1/embeddings",
                {"model": model, "input": ["bravo", "alpha", "charlie"]},
            )
            return response, wire.drain()

        full_cache_hit: Final = eventually(attempt_partial, lambda attempt: not attempt[1])
        assert full_cache_hit[0].status_code == 200, full_cache_hit[0].text
        final_payload: Final = _EmbeddingResponse.model_validate_json(full_cache_hit[0].content)
        assert [item.model_dump() for item in final_payload.data] == _expected_data(("bravo", "alpha", "charlie")), (
            full_cache_hit[0].text
        )
        assert (final_payload.usage.prompt_tokens, final_payload.usage.total_tokens) == (
            3,
            3,
        ), full_cache_hit[0].text
        assert [(request.method, request.target) for request in wire.drain()] == []
