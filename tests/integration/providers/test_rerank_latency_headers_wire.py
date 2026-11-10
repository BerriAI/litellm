import json
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel

MODEL: Final = "cohere/synthetic-rerank-model-without-pricing"
QUERY: Final = "which document mentions the gateway"
DOCUMENTS: Final = ("the gateway proxies rerank calls", "unrelated synthetic text")
RESPONSE: Final = json.dumps(
    {
        "id": "synthetic-rerank-id",
        "results": [{"index": 0, "relevance_score": 0.91}, {"index": 1, "relevance_score": 0.03}],
        "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 1}},
    }
).encode()


def rerank_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/rerank"), request.target
    body: Final = json.loads(request.body)
    assert body["query"] == QUERY and body["documents"] == list(DOCUMENTS), request.body
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.rerank.response_carries_latency_and_cost_headers")
def test_rerank_response_carries_call_id_latency_and_cost_headers_like_chat_completions(gateway: Gateway) -> None:
    with wire_server(rerank_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key="synthetic-cohere-key",
            api_base=wire.url,
            model_info={"mode": "rerank"},
        )
        response: Final = gateway.request(
            "POST", "/v1/rerank", {"model": model, "query": QUERY, "documents": list(DOCUMENTS), "top_n": 2}
        )
        assert response.status_code == 200, response.text
        assert [(result["index"], result["relevance_score"]) for result in response.json()["results"]] == [
            (0, 0.91),
            (1, 0.03),
        ], response.text
        assert len(wire.drain()) == 1, "Expected exactly one provider rerank call"
        assert response.headers["x-litellm-model-group"] == model, response.text
        assert uuid.UUID(response.headers["x-litellm-call-id"]).version == 4, response.headers
        assert float(response.headers["x-litellm-response-cost"]) == 0.0, response.headers
        assert float(response.headers["x-litellm-response-duration-ms"]) > 0, response.headers
        assert float(response.headers["x-litellm-overhead-duration-ms"]) >= 0, response.headers


COHERE_KEY: Final = "synthetic-cohere-key"
COHERE_V2_BODY: Final = {"model": "rerank-v3.5", "query": QUERY, "top_n": 1, "documents": list(DOCUMENTS)}
COHERE_V1_BODY: Final = {**COHERE_V2_BODY, "return_documents": False, "max_chunks_per_doc": 3}
COHERE_V2_SDK_BODY: Final = {**COHERE_V2_BODY, "max_tokens_per_doc": 512}


class _RerankResult(BaseModel):
    index: int
    relevance_score: float


class _RerankResponse(BaseModel):
    results: tuple[_RerankResult, ...]


def _cohere_peer(expected: dict[str, object], route: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", route), request.target
        assert request.headers["authorization"] == f"Bearer {COHERE_KEY}", request.headers
        assert json.loads(request.body) == expected, request.body
        return Reply(body=RESPONSE)

    return respond


@pytest.mark.parametrize("path", ["/v1/rerank", "/v2/rerank", "/rerank"])
@pytest.mark.parametrize(
    ("client_fields", "api_base_suffix", "expected_body", "expected_route"),
    [
        pytest.param({}, "", COHERE_V2_BODY, "/v2/rerank", id="default-v2"),
        pytest.param(
            {"max_chunks_per_doc": 3, "return_documents": False}, "", COHERE_V1_BODY, "/v1/rerank", id="v1-sdk-fields"
        ),
        pytest.param({}, "/v1/rerank", COHERE_V2_BODY, "/v1/rerank", id="v1-api-base"),
        pytest.param({"max_tokens_per_doc": 512}, "", COHERE_V2_SDK_BODY, "/v2/rerank", id="v2-sdk-fields"),
    ],
)
def test_cohere_rerank_selects_v1_or_v2_from_client_fields_and_sends_exact_body(
    gateway: Gateway,
    path: str,
    client_fields: dict[str, object],
    api_base_suffix: str,
    expected_body: dict[str, object],
    expected_route: str,
) -> None:
    with wire_server(_cohere_peer(expected_body, expected_route)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="cohere/rerank-v3.5",
            api_key=COHERE_KEY,
            api_base=wire.url + api_base_suffix,
            model_info={"mode": "rerank"},
        )
        response: Final = gateway.request(
            "POST", path, {"model": model, "query": QUERY, "documents": list(DOCUMENTS), "top_n": 1, **client_fields}
        )
        assert response.status_code == 200, response.text
        assert _RerankResponse.model_validate_json(response.content) == _RerankResponse(
            results=(_RerankResult(index=0, relevance_score=0.91), _RerankResult(index=1, relevance_score=0.03))
        ), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", expected_route)]


OBJECT_DOCUMENTS: Final = ({"text": DOCUMENTS[0]}, {"text": DOCUMENTS[1]})


@pytest.mark.parametrize(
    ("client_fields", "expected_route"),
    [
        pytest.param({}, "/v2/rerank", id="v2"),
        pytest.param({"max_chunks_per_doc": 3, "return_documents": False}, "/v1/rerank", id="v1-sdk-fields"),
    ],
)
def test_cohere_v1_sdk_object_documents_reach_cohere_unchanged_on_both_api_versions(
    gateway: Gateway, client_fields: dict[str, object], expected_route: str
) -> None:
    expected_body: Final = {
        "model": "rerank-v3.5",
        "query": QUERY,
        "top_n": 1,
        "documents": list(OBJECT_DOCUMENTS),
        **client_fields,
    }
    with wire_server(_cohere_peer(expected_body, expected_route)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="cohere/rerank-v3.5", api_key=COHERE_KEY, api_base=wire.url, model_info={"mode": "rerank"}
        )
        response: Final = gateway.request(
            "POST",
            "/v1/rerank",
            {"model": model, "query": QUERY, "documents": list(OBJECT_DOCUMENTS), "top_n": 1, **client_fields},
        )
        assert response.status_code == 200, response.text
        assert _RerankResponse.model_validate_json(response.content) == _RerankResponse(
            results=(_RerankResult(index=0, relevance_score=0.91), _RerankResult(index=1, relevance_score=0.03))
        ), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", expected_route)]
