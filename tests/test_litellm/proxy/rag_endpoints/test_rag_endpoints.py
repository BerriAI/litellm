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
