import json
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
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


_JSON_TOOL: Final = "json_tool_call"
_SCHEMA: Final = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "backend", ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-opus-4-8"], ids=["haiku_5_5", "opus_4_8"]
)
def test_json_schema_batch_record_with_thinking_uploads_the_json_tool_and_no_native_output(
    gateway: Gateway, backend: str
) -> None:
    record: Final = {
        "custom_id": "schema-record",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": f"bedrock/{backend}",
            "messages": [{"role": "user", "content": "Name a city."}],
            "max_tokens": 2048,
            "thinking": {"type": "adaptive"},
            "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": _SCHEMA}},
        },
    }
    with wire_server(s3_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/{backend}",
            api_key=None,
            api_base=None,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name="us-east-1",
            s3_bucket_name=BUCKET,
            s3_endpoint_url=wire.url,
        )
        response: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model},
            {"file": ("in.jsonl", (json.dumps(record) + "\n").encode(), "application/jsonl")},
        )
        assert response.status_code == 200, response.text
        (upload,) = wire.drain()
    (stored,) = tuple(JSON_OBJECT.validate_json(line) for line in upload.body.decode().splitlines() if line.strip())
    model_input: Final = object_value(stored["modelInput"])
    assert "format" not in object_value(model_input.get("output_config", {})), model_input
    (tool,) = (object_value(entry) for entry in list_value(model_input["tools"]))
    assert tool["name"] == _JSON_TOOL, model_input
    assert object_value(tool["input_schema"])["properties"] == _SCHEMA["properties"], model_input
