import json
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL_ID: Final = "amazon.nova-2-multimodal-embeddings-v1:0"
TOKEN: Final = "synthetic-bedrock-bearer"
INPUT: Final = "hello world"
S3_URI: Final = "s3://synthetic-bucket/out/"
SEGMENTATION: Final = {"durationSeconds": 10}
INVOCATION_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"


def start_async_invoke_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/async-invoke", request.target
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body: Final = json.loads(request.body)
    assert body.keys() == {"modelId", "modelInput", "outputDataConfig"}, request.body
    assert body["modelId"] == MODEL_ID, request.body
    assert body["outputDataConfig"] == {"s3OutputDataConfig": {"s3Uri": S3_URI}}, request.body
    assert body["modelInput"]["segmentationConfig"] == SEGMENTATION, request.body
    return Reply(body=json.dumps({"invocationArn": INVOCATION_ARN}).encode())


def test_async_invoke_embedding_extra_body_fields_reach_bedrock_inside_model_input(gateway: Gateway) -> None:
    with wire_server(start_async_invoke_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/async_invoke/{MODEL_ID}",
            api_key=TOKEN,
            api_base=wire.url,
            aws_region_name="us-east-1",
            output_s3_uri=S3_URI,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": INPUT, "extra_body": {"segmentationConfig": SEGMENTATION}},
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1, "the async-invoke request never reached Bedrock"
