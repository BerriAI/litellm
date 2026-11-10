import json
from typing import Final

from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "triton/my-triton-model"
TOKEN: Final = "scripted-triton-key"
INPUT: Final = ["good morning from litellm"]
OUTPUT: Final = [0.1, 0.2]
RESPONSE: Final = json.dumps(
    {
        "model_name": "my-triton-model",
        "outputs": [
            {
                "name": "output",
                "datatype": "FP32",
                "shape": [1, 2],
                "data": OUTPUT,
            }
        ],
    }
).encode()


def triton_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target == "/triton/embeddings", request.target
    assert json.loads(request.body) == {
        "inputs": [
            {
                "name": "input_text",
                "shape": [1],
                "datatype": "BYTES",
                "data": INPUT,
            }
        ]
    }, request.body
    return Reply(body=RESPONSE)


def test_triton_embeddings(gateway: Gateway) -> None:
    with wire_server(triton_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=TOKEN,
            api_base=f"{wire.url}/triton/embeddings",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": INPUT},
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"] == [
            {"object": "embedding", "index": 0, "embedding": OUTPUT}
        ], response.text
        assert len(wire.drain()) == 1
