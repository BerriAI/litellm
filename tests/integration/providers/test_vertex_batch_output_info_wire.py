import base64
import functools
import json
from typing import Final

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

PROJECT: Final = "cc-scripted-project"
LOCATION: Final = "us-central1"
MODEL: Final = "vertex_ai/gemini-2.5-flash"
VERTEX_MODEL_RESOURCE: Final = "publishers/google/models/gemini-2.5-flash"
BUCKET: Final = "integration-batch-bucket"
INPUT_FILE_ID: Final = f"gs://{BUCKET}/litellm-vertex-files/{VERTEX_MODEL_RESOURCE}/input.jsonl"
OUTPUT_PREFIX: Final = INPUT_FILE_ID.rsplit("/", 1)[0]
JOB_NAME: Final = f"projects/{PROJECT}/locations/{LOCATION}/batchPredictionJobs/7412345678901234567"
JOB_ID: Final = JOB_NAME.rsplit("/", 1)[-1]
EXPECTED_VERTEX_BODY: Final = {
    "inputConfig": {"gcsSource": {"uris": [INPUT_FILE_ID]}, "instancesFormat": "jsonl"},
    "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": OUTPUT_PREFIX}},
    "model": VERTEX_MODEL_RESOURCE,
}
VERTEX_REPLY: Final = {
    "name": JOB_NAME,
    "displayName": "litellm-vertex-batch-scripted",
    "model": VERTEX_MODEL_RESOURCE,
    "inputConfig": {"gcsSource": {"uris": [INPUT_FILE_ID]}, "instancesFormat": "jsonl"},
    "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": OUTPUT_PREFIX}},
    "outputInfo": None,
    "state": "JOB_STATE_PENDING",
    "createTime": "2026-07-24T20:00:00.000000Z",
    "updateTime": "2026-07-24T20:00:00.000000Z",
}


@functools.cache
def _vertex_private_key_pem() -> str:
    return (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )


def _vertex_service_account_json(url: str) -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": PROJECT,
            "private_key_id": "scripted",
            "private_key": _vertex_private_key_pem(),
            "client_email": f"scripted@{PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{url}/_oauth/authorize",
            "token_uri": f"{url}/_oauth/token",
        }
    )


def _encoded(raw: str, model: str, prefix: str) -> str:
    return prefix + base64.urlsafe_b64encode(f"litellm:{raw};model,{model}".encode()).decode().rstrip("=")


def vertex_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target == f"/v1/projects/{PROJECT}/locations/{LOCATION}/batchPredictionJobs", request.target
    assert request.headers["authorization"] == "Bearer scripted-token"
    assert request.headers["content-type"] == "application/json; charset=utf-8"
    body: Final = json.loads(request.body)
    display_name: Final = body.pop("displayName")
    assert isinstance(display_name, str) and display_name.startswith("litellm-vertex-batch-"), display_name
    assert body == EXPECTED_VERTEX_BODY, body
    return Reply(body=json.dumps(VERTEX_REPLY).encode())


@pytest.mark.covers("other.provider_wire.vertex_ai.batch_create_with_null_output_info_returns_batch_instead_of_500")
def test_vertex_batch_create_survives_explicit_null_output_info(gateway: Gateway) -> None:
    with wire_server(vertex_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=MODEL,
            api_key=None,
            api_base=wire.url,
            vertex_project=PROJECT,
            vertex_location=LOCATION,
            vertex_credentials=_vertex_service_account_json(gateway.upstream_url),
        )
        response: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": INPUT_FILE_ID,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model,
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert (
            body["id"],
            body["object"],
            body["status"],
            body["input_file_id"],
            body["output_file_id"],
            body["error_file_id"],
            body["completion_window"],
        ) == (
            _encoded(JOB_ID, model, "batch_"),
            "batch",
            "validating",
            _encoded(INPUT_FILE_ID, model, "file-"),
            _encoded(f"{OUTPUT_PREFIX}/predictions.jsonl", model, "file-"),
            None,
            "24h",
        ), response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, f"Expected exactly one Vertex POST, saw {[request.target for request in requests]}"
