"""
Tests for the proxy /v1/batches/{batch_id} retrieve flow and the
/v1/files/{file_id}/content download flow with model-encoded IDs (Bedrock).

Regression (retrieve): when the proxy decoded `model` from the encoded
batch_id, it did not forward `model` as a kwarg to `litellm.aretrieve_batch`.
That caused litellm to skip the `BedrockBatchesConfig` provider_config path
and fall into the legacy provider switch, which raises BadRequestError for
bedrock.

The download path is included to lock in the end-to-end Bedrock batch flow:
retrieve returns an `output_file_id` re-encoded with model info, and that ID
must round-trip through `client.files.content(...)` back to bedrock with AWS
credentials and the raw S3 URI intact.
"""

import os
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.openai_files_endpoints.common_utils import (
    encode_file_id_with_model,
)
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import LiteLLMBatch

client = TestClient(app)

BEDROCK_MODEL = "bedrock-claude-test"
BEDROCK_BATCH_ARN = (
    "arn:aws:bedrock:us-east-1:000000000000:model-invocation-job/test-job-id"
)
BEDROCK_OUTPUT_S3_URI = (
    "s3://test-bedrock-batch-output/job-output/test-job-id/output.jsonl.out"
)


@pytest.fixture
def bedrock_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": BEDROCK_MODEL,
                "litellm_params": {
                    "model": f"bedrock/{BEDROCK_MODEL}",
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": "test-access-key",
                    "aws_secret_access_key": "test-secret-key",
                },
                "model_info": {"id": "bedrock-claude-test-id"},
            },
        ]
    )


def _setup_proxy(monkeypatch, llm_router: Router):
    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)


def _encoded_bedrock_batch_id() -> str:
    return encode_file_id_with_model(
        file_id=BEDROCK_BATCH_ARN, model=BEDROCK_MODEL, id_type="batch"
    )


def _make_in_progress_batch_response(batch_id: str) -> LiteLLMBatch:
    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id="file-input",
        object="batch",
        status="in_progress",
    )


def test_retrieve_batch_passes_model_for_bedrock_encoded_id(
    monkeypatch, bedrock_router
):
    """Encoded batch_id → proxy must pass `model` to litellm.aretrieve_batch
    so BedrockBatchesConfig is loaded.

    Without this, litellm falls into the legacy provider switch and raises
    'LiteLLM doesn't support bedrock for retrieve_batch'.
    """
    _setup_proxy(monkeypatch, bedrock_router)

    user_key = UserAPIKeyAuth(api_key="test-key")
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    encoded_batch_id = _encoded_bedrock_batch_id()
    captured_kwargs: dict = {}

    async def mock_aretrieve_batch(**kwargs):
        captured_kwargs.update(kwargs)
        return _make_in_progress_batch_response(BEDROCK_BATCH_ARN)

    monkeypatch.setattr(litellm, "aretrieve_batch", mock_aretrieve_batch)

    try:
        response = client.get(
            f"/v1/batches/{encoded_batch_id}",
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200, response.text
    finally:
        app.dependency_overrides.clear()

    assert captured_kwargs.get("custom_llm_provider") == "bedrock"
    assert captured_kwargs.get("model") == BEDROCK_MODEL, (
        "model must be forwarded to litellm.aretrieve_batch so the bedrock "
        "provider_config is loaded; got kwargs: " + repr(captured_kwargs)
    )
    assert captured_kwargs.get("batch_id") == BEDROCK_BATCH_ARN


def test_retrieve_batch_response_id_is_re_encoded_with_model(
    monkeypatch, bedrock_router
):
    """After provider returns the raw ARN, the proxy must re-encode the
    response id with the model so subsequent client calls keep routing to
    bedrock."""
    _setup_proxy(monkeypatch, bedrock_router)

    user_key = UserAPIKeyAuth(api_key="test-key")
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    encoded_batch_id = _encoded_bedrock_batch_id()

    async def mock_aretrieve_batch(**kwargs):
        return _make_in_progress_batch_response(BEDROCK_BATCH_ARN)

    monkeypatch.setattr(litellm, "aretrieve_batch", mock_aretrieve_batch)

    try:
        response = client.get(
            f"/v1/batches/{encoded_batch_id}",
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
    finally:
        app.dependency_overrides.clear()

    assert body["id"] == encoded_batch_id


def test_file_content_routes_to_bedrock_for_encoded_output_file_id(
    monkeypatch, bedrock_router
):
    """`client.files.content(output_file_id)` for a bedrock-encoded file ID
    must reach `litellm.afile_content` with `custom_llm_provider="bedrock"`,
    the raw S3 URI as `file_id`, and AWS credentials sourced from the router.

    This is the second half of the bedrock batch flow (the first being
    retrieve). Without this round-trip, callers have to bypass the proxy and
    call `litellm.file_content(...)` directly with hand-rolled AWS args.
    """
    _setup_proxy(monkeypatch, bedrock_router)

    user_key = UserAPIKeyAuth(api_key="test-key")
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    encoded_file_id = encode_file_id_with_model(
        file_id=BEDROCK_OUTPUT_S3_URI, model=BEDROCK_MODEL, id_type="file"
    )
    captured_kwargs: dict = {}
    file_bytes = b'{"custom_id":"r1","response":{"body":{"choices":[]}}}\n'

    async def mock_afile_content(**kwargs):
        captured_kwargs.update(kwargs)
        return HttpxBinaryResponseContent(
            response=httpx.Response(
                status_code=200,
                content=file_bytes,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request(method="GET", url=BEDROCK_OUTPUT_S3_URI),
            )
        )

    monkeypatch.setattr(litellm, "afile_content", mock_afile_content)

    try:
        response = client.get(
            f"/v1/files/{encoded_file_id}/content",
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200, response.text
        assert response.content == file_bytes
    finally:
        app.dependency_overrides.clear()

    assert captured_kwargs.get("custom_llm_provider") == "bedrock"
    assert captured_kwargs.get("file_id") == BEDROCK_OUTPUT_S3_URI, (
        "file_id must be decoded back to the raw S3 URI before reaching "
        "litellm.afile_content; got kwargs: " + repr(captured_kwargs)
    )
    assert captured_kwargs.get("aws_region_name") == "us-east-1"
    assert captured_kwargs.get("aws_access_key_id") == "test-access-key"
    assert captured_kwargs.get("aws_secret_access_key") == "test-secret-key"
