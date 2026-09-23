import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

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
