import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/us.twelvelabs.marengo-embed-3-0-v1:0"
TOKEN: Final = "synthetic-bedrock-bearer"
INPUT: Final = "hello world"
VECTOR: Final = [0.1, 0.2, 0.3]
RESPONSE: Final = json.dumps({"data": [{"embedding": VECTOR}]}).encode()


def marengo_3_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target == "/model/us.twelvelabs.marengo-embed-3-0-v1%3A0/invoke", request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request.body) == {"inputType": "text", "text": {"inputText": INPUT}}, request.body
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.bedrock_embedding.marengo_3_text_input_reaches_bedrock_nested_under_input_type")
def test_marengo_3_text_embedding_nests_input_text_under_input_type(gateway: Gateway) -> None:
    with wire_server(marengo_3_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=TOKEN,
            api_base=wire.url,
            aws_region_name="us-east-1",
        )
        response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": INPUT})
        assert response.status_code == 200, response.text
        assert response.json()["data"] == [{"object": "embedding", "index": 0, "embedding": VECTOR}], response.text
        assert len(wire.drain()) == 1, "the embedding request never reached Bedrock"
