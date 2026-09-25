import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/cohere.embed-english-v3"
TOKEN: Final = "synthetic-bedrock-bearer"
INPUT: Final = "hello world"
VECTOR: Final = [0.1, 0.2, 0.3]
RESPONSE: Final = json.dumps(
    {
        "embeddings": {"float": [VECTOR]},
        "id": "synthetic-cohere-embed",
        "response_type": "embeddings_by_type",
        "texts": [INPUT],
    }
).encode()


def cohere_english_v3_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/model/cohere.embed-english-v3/invoke"
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request.body) == {
        "texts": [INPUT],
        "input_type": "search_document",
        "embedding_types": ["float"],
        "output_dimension": 512,
    }
    return Reply(body=RESPONSE)


@pytest.mark.covers("other.provider_wire.bedrock.cohere_embed_english_v3_accepts_encoding_format")
def test_cohere_embed_english_v3_accepts_encoding_format_and_dimensions(gateway: Gateway) -> None:
    with wire_server(cohere_english_v3_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=TOKEN,
            api_base=wire.url,
            aws_region_name="us-east-1",
        )
        for encoding_format in ("float", "base64"):
            response: Final = gateway.request(
                "POST",
                "/v1/embeddings",
                {
                    "model": model,
                    "input": INPUT,
                    "encoding_format": encoding_format,
                    "dimensions": 512,
                },
            )
            assert response.status_code == 200, f"encoding_format={encoding_format}: {response.text}"
            assert response.json()["data"] == [
                {"object": "embedding", "index": 0, "embedding": VECTOR, "type": "float"},
            ], response.text
            assert len(wire.drain()) == 1, f"encoding_format={encoding_format} never reached Bedrock"
