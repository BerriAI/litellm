import json
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2"
QUERY: Final = "which passage shows the gateway diagram"
IMAGE_PASSAGE: Final = "data:image/png;base64,aW50ZWdyYXRpb24tc3ludGhldGljLWltYWdl"
TEXT_PASSAGE: Final = "the gateway proxies rerank calls"
RESPONSE: Final = json.dumps(
    {"rankings": [{"index": 0, "logit": 0.82}, {"index": 1, "logit": -1.4}], "usage": {"total_tokens": 11}}
).encode()


def ranking_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/ranking", request.target
    assert request.headers["authorization"] == "Bearer integration-provider-key"
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert body == {
        "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        "query": {"text": QUERY},
        "passages": [{"image": IMAGE_PASSAGE}, {"text": TEXT_PASSAGE}],
    }, body
    return Reply(body=RESPONSE)


@pytest.mark.covers(
    "providers.nvidia_nim_ranking.image_passages_reach_ranking_without_top_k_and_top_n_is_applied_locally"
)
def test_nvidia_nim_ranking_keeps_image_passages_and_applies_top_n_without_sending_top_k(gateway: Gateway) -> None:
    with wire_server(ranking_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=MODEL, api_base=wire.url, model_info={"mode": "rerank"})
        response: Final = gateway.request(
            "POST",
            "/v1/rerank",
            {
                "model": model,
                "query": QUERY,
                "documents": [{"image": IMAGE_PASSAGE}, {"text": TEXT_PASSAGE}],
                "top_n": 1,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 0, "relevance_score": 0.82}], response.text
        assert len(wire.drain()) == 1, "Expected exactly one provider ranking call"
