import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
BUCKET: Final = "integration-batch-bucket"
PROMPT: Final = "synthetic completions prompt"
RESPONSES_INPUT: Final = "synthetic responses input"
INPUT_LINES: Final = (
    {
        "custom_id": "completions-record",
        "method": "POST",
        "url": "/v1/completions",
        "body": {"model": MODEL, "prompt": PROMPT, "max_tokens": 64},
    },
    {
        "custom_id": "responses-record",
        "method": "POST",
        "url": "/v1/responses",
        "body": {"model": MODEL, "input": RESPONSES_INPUT, "max_output_tokens": 16},
    },
)
EXPECTED_S3_OBJECT: Final = (
    {
        "recordId": "completions-record",
        "modelInput": {
            "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
            "max_tokens": 64,
            "anthropic_version": "bedrock-2023-05-31",
        },
    },
    {
        "recordId": "responses-record",
        "modelInput": {
            "messages": [{"role": "user", "content": [{"type": "text", "text": RESPONSES_INPUT}]}],
            "max_tokens": 16,
            "anthropic_version": "bedrock-2023-05-31",
        },
    },
)


def s3_peer(request: Request) -> Reply:
    assert request.method == "PUT" and request.target.startswith(f"/{BUCKET}/"), request.target
    assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    return Reply(body=b"")


@pytest.mark.covers(
    "other.provider_wire.bedrock.batch_file_completions_and_responses_records_reach_s3_as_user_messages"
)
def test_completions_and_responses_batch_records_upload_as_anthropic_user_messages(gateway: Gateway) -> None:
    with wire_server(s3_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=None,
            api_base=None,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name="us-east-1",
            s3_bucket_name=BUCKET,
            s3_endpoint_url=wire.url,
        )
        jsonl: Final = "\n".join(json.dumps(line, separators=(",", ":")) for line in INPUT_LINES) + "\n"
        response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model},
            {"file": ("in.jsonl", jsonl.encode(), "application/jsonl")},
        )
        assert response.status_code == 200, response.text
        assert response.json()["object"] == "file" and response.json()["purpose"] == "batch", response.text
        uploads: Final = wire.drain()
        assert len(uploads) == 1, f"Expected exactly one S3 PUT, saw {[upload.target for upload in uploads]}"
        stored: Final = tuple(json.loads(line) for line in uploads[0].body.decode().splitlines() if line.strip())
        assert stored == EXPECTED_S3_OBJECT
