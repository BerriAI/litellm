import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "huggingface/BAAI/bge-reranker-base"
QUERY: Final = "what is a fox"
TEXT: Final = "a short doc about foxes"
RESPONSE: Final = json.dumps([{"index": 0, "score": 0.82}]).encode()
TRUNCATION_CASES: Final = (
    pytest.param(
        MappingProxyType({}),
        MappingProxyType({"truncation_direction": "Right"}),
        id="omitted-truncate-uses-server-default",
    ),
    pytest.param(
        MappingProxyType({"truncate": True, "truncation_direction": "Left"}),
        MappingProxyType({"truncate": True, "truncation_direction": "Left"}),
        id="explicit-true-and-left-are-forwarded",
    ),
    pytest.param(
        MappingProxyType({"truncate": False}),
        MappingProxyType({"truncate": False, "truncation_direction": "Right"}),
        id="explicit-false-is-forwarded",
    ),
)


def rerank_peer(request: Request, expected_body: Mapping[str, object]) -> Reply:
    assert request.method == "POST" and request.target == "/rerank", request.target
    assert request.headers["authorization"] == "Bearer integration-provider-key"
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert body == expected_body, body
    return Reply(body=RESPONSE)


@pytest.mark.parametrize(("request_params", "expected_truncation_params"), TRUNCATION_CASES)
def test_huggingface_rerank_forwards_truncation_options(
    gateway: Gateway,
    request_params: Mapping[str, bool | str],
    expected_truncation_params: Mapping[str, bool | str],
) -> None:
    expected_body: Final = {
        "raw_scores": False,
        "truncation_direction": "Right",
        "query": QUERY,
        "texts": [TEXT],
        "return_text": False,
        **expected_truncation_params,
    }
    with wire_server(lambda request: rerank_peer(request, expected_body)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_base=wire.url,
            model_info={"mode": "rerank"},
        )
        response: Final = gateway.request(
            "POST",
            "/v1/rerank",
            {
                "model": model,
                "query": QUERY,
                "documents": [TEXT],
                "return_documents": False,
                **request_params,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 0, "relevance_score": 0.82}], response.text
        assert len(wire.drain()) == 1, "Expected exactly one provider rerank call"
