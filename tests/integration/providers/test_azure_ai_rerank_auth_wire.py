import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "azure_ai/Cohere-rerank-v4.0-fast"
ENTRA_TOKEN: Final = "synthetic-entra-access-token"
QUERY: Final = "which document mentions the gateway"
DOCUMENTS: Final = ("the gateway proxies rerank calls", "unrelated synthetic text")
RESPONSE: Final = json.dumps(
    {
        "id": "synthetic-rerank-id",
        "results": [{"index": 0, "relevance_score": 0.91}, {"index": 1, "relevance_score": 0.03}],
        "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 1}},
    }
).encode()


def entra_rerank_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/providers/cohere/v2/rerank"
    assert request.headers["authorization"] == f"Bearer {ENTRA_TOKEN}"
    assert "api-key" not in request.headers
    body: Final = json.loads(request.body)
    assert body == {"model": "Cohere-rerank-v4.0-fast", "query": QUERY, "documents": list(DOCUMENTS), "top_n": 2}
    return Reply(body=RESPONSE)


@pytest.mark.covers("other.provider_wire.azure_ai.rerank_entra_token_without_api_key_reaches_provider")
def test_azure_ai_rerank_with_entra_token_and_no_api_key_sends_bearer_to_provider(gateway: Gateway) -> None:
    with wire_server(entra_rerank_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=None,
            api_base=f"{wire.url}/providers/cohere/v2",
            azure_ad_token=ENTRA_TOKEN,
            model_info={"mode": "rerank"},
        )
        response: Final = gateway.request(
            "POST", "/v1/rerank", {"model": model, "query": QUERY, "documents": list(DOCUMENTS), "top_n": 2}
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert [(result["index"], result["relevance_score"]) for result in body["results"]] == [(0, 0.91), (1, 0.03)]
        assert len(wire.drain()) == 1, "Expected exactly one provider rerank call"
