import json
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Reply, Request, wire_server

_MODERATION_RESULT: Final = {
    "id": "modr-integration",
    "model": "omni-moderation-latest",
    "results": [
        {
            "flagged": True,
            "categories": {"violence": True},
            "category_scores": {"violence": 0.99},
        }
    ],
}


def _moderation_response(request: Request) -> Reply:
    if (request.method, request.target) == ("GET", "/v1/models"):
        return Reply(body=b'{"object":"list","data":[]}')
    if (request.method, request.target) == ("POST", "/v1/moderations"):
        return Reply(body=json.dumps(_MODERATION_RESULT).encode())
    return Reply(status=404, body=b'{"error":"unexpected moderation request"}')


def test_moderations_forwards_the_selected_model_and_input(gateway: Gateway) -> None:
    with wire_server(_moderation_response) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="omni-moderation-latest",
            custom_llm_provider="openai",
            api_key="moderation-integration-key",
            api_base=f"{wire.url}/v1",
        )
        text: Final = "I want to harm someone"
        response: Final = gateway.request(
            "POST",
            "/v1/moderations",
            {"input": text, "model": model},
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["results"][0]["flagged"] is True

        requests: Final = tuple(
            request for request in wire.drain() if request.target == "/v1/moderations"
        )
        assert len(requests) == 1
        assert json.loads(requests[0].body) == {"input": text, "model": "omni-moderation-latest"}
