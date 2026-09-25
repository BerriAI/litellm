"""
Tests for RAG proxy endpoints.

Covers:
- internal_user_viewer restriction: can only ingest to existing vector stores (must provide vector_store_id)
"""

import base64
import io
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPResponseEncodingError, HTTPResponseLimitError
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app
from litellm.proxy.rag_endpoints.upload_security import (
    MAX_UPLOAD_SIZE_BYTES,
    EicarTestMalwareScanner,
    ScanResult,
    ScanVerdict,
)


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
        data={"request": '{"ingest_options":{"vector_store":{"custom_llm_provider":"openai"}}}'},
    )

    assert response.status_code == 403
    detail = response.json()
    assert "detail" in detail
    error_msg = detail["detail"]["error"] if isinstance(detail["detail"], dict) else str(detail["detail"])
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
        f"internal_user_viewer with vector_store_id should pass role check. Response: {response.json()}"
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
            data={"request": '{"ingest_options":{"vector_store":{"custom_llm_provider":"openai"}}}'},
        )

    # Should not be 403
    assert response.status_code != 403, (
        f"internal_user should be allowed to create new vector stores. Response: {response.json()}"
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
    assert response.status_code == 400, (
        f"Expected 400 when '{blocked_field}' is set clientside, got {response.status_code}: {response.json()}"
    )
    body = response.json()
    assert blocked_field in str(body), f"Response should mention '{blocked_field}': {body}"


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
    def test_ssrf_field_in_vector_store_config_rejected(self, field, value, client_internal_user):
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
        error_text = detail.get("error", "") if isinstance(detail, dict) else str(detail)
        assert field in error_text, f"Error should name the offending field: {error_text}"

    def test_clean_bedrock_ingest_options_not_rejected(self, client_internal_user, monkeypatch):
        fetch_url, _seen = _fetcher_answering(200, b"%PDF-1.7\n1 0 obj<<>>endobj\n", content_type="application/pdf")
        monkeypatch.setattr("litellm.proxy.rag_endpoints.endpoints.fetch_file_url", fetch_url)
        with patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new_callable=AsyncMock,
            return_value={"vector_store_id": "vs_bedrock", "file_id": "file_123"},
        ):
            response = client_internal_user.post(
                "/v1/rag/ingest",
                json={
                    "file_url": "https://example.com/doc.pdf",
                    "ingest_options": {"vector_store": {"custom_llm_provider": "bedrock"}},
                },
            )
        assert response.status_code != 400, f"Clean Bedrock ingest_options should not be rejected: {response.json()}"


S3_REGISTRY_STORE = {
    "vector_store_id": "s3-store",
    "custom_llm_provider": "s3_vectors",
    "litellm_params": {"aws_region_name": "eu-west-1", "vector_bucket_name": "bkt", "index_name": "docs"},
}
DB_MANAGED_STORE = {
    "vector_store_id": "db-store",
    "custom_llm_provider": "openai",
    "litellm_credential_name": None,
    "litellm_params": {"ttl_days": 7},
}
AZURE_REGISTRY_STORE = {
    "vector_store_id": "my-azure-index",
    "custom_llm_provider": "azure_ai",
    "litellm_params": {
        "api_key": "azure-search-key",
        "api_base": "https://search.example.net",
        "api_version": "2024-07-01",
    },
}
BEDROCK_REGISTRY_STORE = {
    "vector_store_id": "kb-store",
    "custom_llm_provider": "bedrock",
    "litellm_params": {
        "aws_region_name": "eu-west-1",
        "aws_access_key_id": "AKIA-registry",
        "aws_secret_access_key": "registry-secret",
    },
}
CREDENTIALED_REGISTRY_STORE = {
    "vector_store_id": "cred-store",
    "custom_llm_provider": "openai",
    "litellm_credential_name": "registry-openai",
    "litellm_params": {},
}
VERTEX_REGISTRY_STORE = {
    "vector_store_id": "projects/registry-project/locations/us-central1/ragCorpora/42",
    "custom_llm_provider": "vertex_ai",
    "litellm_params": {"vertex_project": "registry-project", "vertex_location": "us-central1"},
}
UNSUPPORTED_INGEST_PROVIDER_ERROR = (
    "Provider '{provider}' is not supported for RAG ingestion. "
    "Supported providers: openai, bedrock, gemini, s3_vectors, vertex_ai"
)


def _registry_with(store):
    registry = MagicMock()
    registry.get_litellm_managed_vector_store_from_registry.return_value = store
    return registry


def _ingest_form(vector_store):
    return {
        "files": {"file": ("sample.txt", io.BytesIO(b"test content"), "text/plain")},
        "data": {"request": json.dumps({"ingest_options": {"vector_store": vector_store}})},
    }


def _patched_ingest_boundary(registry_store, aingest_response):
    return (
        patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; tests assert the forwarded options
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new=AsyncMock(return_value=aingest_response),
        ),
        patch.object(  # test-quality-ok: seeds the managed-store registry the merge under test reads
            litellm,
            "vector_store_registry",
            _registry_with(registry_store),
        ),
    )


def _patched_prisma_client(prisma_client):
    return patch(  # test-quality-ok: proxy module global, no injection seam
        "litellm.proxy.proxy_server.prisma_client",
        prisma_client,
    )


def test_rag_ingest_resolves_registry_store_provider_and_params(client_internal_user):
    aingest_patch, registry_patch = _patched_ingest_boundary(
        S3_REGISTRY_STORE, {"vector_store_id": "s3-store", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post("/v1/rag/ingest", **_ingest_form({"vector_store_id": "s3-store"}))

    assert response.status_code == 200, response.json()
    mock_aingest.assert_awaited_once()
    forwarded = mock_aingest.await_args.kwargs["ingest_options"]["vector_store"]
    assert forwarded["vector_store_id"] == "s3-store"
    assert forwarded["custom_llm_provider"] == "s3_vectors"
    assert forwarded["aws_region_name"] == "eu-west-1"
    assert forwarded["vector_bucket_name"] == "bkt"
    assert forwarded["index_name"] == "docs"


def test_rag_ingest_registry_store_wins_over_request_provider_and_params(client_internal_user):
    aingest_patch, registry_patch = _patched_ingest_boundary(
        S3_REGISTRY_STORE, {"vector_store_id": "s3-store", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            **_ingest_form(
                {"vector_store_id": "s3-store", "custom_llm_provider": "openai", "aws_region_name": "us-east-1"}
            ),
        )

    assert response.status_code == 200, response.json()
    forwarded = mock_aingest.await_args.kwargs["ingest_options"]["vector_store"]
    assert forwarded["custom_llm_provider"] == "s3_vectors"
    assert forwarded["aws_region_name"] == "eu-west-1"


def test_rag_ingest_registry_store_drops_caller_destinations_and_keeps_upload_options(client_internal_user):
    aingest_patch, registry_patch = _patched_ingest_boundary(
        BEDROCK_REGISTRY_STORE, {"vector_store_id": "kb-store", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            **_ingest_form(
                {
                    "vector_store_id": "kb-store",
                    "s3_bucket": "someone-elses-bucket",
                    "s3_prefix": "other-kb/",
                    "vector_bucket_name": "someone-elses-vectors",
                    "index_name": "other-index",
                    "vertex_project": "other-project",
                    "data_source_id": "DS2",
                    "wait_for_ingestion": True,
                    "ingestion_timeout": 60,
                }
            ),
        )

    assert response.status_code == 200, response.json()
    forwarded = mock_aingest.await_args.kwargs["ingest_options"]["vector_store"]
    assert forwarded == {
        "vector_store_id": "kb-store",
        "custom_llm_provider": "bedrock",
        "aws_region_name": "eu-west-1",
        "aws_access_key_id": "AKIA-registry",
        "aws_secret_access_key": "registry-secret",
        "data_source_id": "DS2",
        "wait_for_ingestion": True,
        "ingestion_timeout": 60,
    }


def test_rag_ingest_unmanaged_store_keeps_the_callers_full_config(client_internal_user):
    caller_config = {
        "vector_store_id": "KB-unmanaged",
        "custom_llm_provider": "bedrock",
        "s3_bucket": "callers-bucket",
        "s3_prefix": "docs/",
    }
    aingest_patch, registry_patch = _patched_ingest_boundary(
        None, {"vector_store_id": "KB-unmanaged", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post("/v1/rag/ingest", **_ingest_form(caller_config))

    assert response.status_code == 200, response.json()
    assert mock_aingest.await_args.kwargs["ingest_options"]["vector_store"] == caller_config


def test_rag_ingest_db_managed_store_drops_the_callers_credential_name(client_internal_user):
    aingest_patch, registry_patch = _patched_ingest_boundary(
        DB_MANAGED_STORE, {"vector_store_id": "db-store", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            **_ingest_form({"vector_store_id": "db-store", "litellm_credential_name": "team-openai"}),
        )

    assert response.status_code == 200, response.json()
    forwarded = mock_aingest.await_args.kwargs["ingest_options"]["vector_store"]
    assert "litellm_credential_name" not in forwarded
    assert forwarded["custom_llm_provider"] == "openai"
    assert forwarded["ttl_days"] == 7


def test_rag_ingest_registry_store_credential_name_beats_the_callers(client_internal_user):
    aingest_patch, registry_patch = _patched_ingest_boundary(
        CREDENTIALED_REGISTRY_STORE, {"vector_store_id": "cred-store", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            **_ingest_form({"vector_store_id": "cred-store", "litellm_credential_name": "team-openai"}),
        )

    assert response.status_code == 200, response.json()
    forwarded = mock_aingest.await_args.kwargs["ingest_options"]["vector_store"]
    assert forwarded["litellm_credential_name"] == "registry-openai"


def test_rag_ingest_registry_store_keeps_the_callers_vertex_embedding_throttle(client_internal_user):
    aingest_patch, registry_patch = _patched_ingest_boundary(
        VERTEX_REGISTRY_STORE, {"vector_store_id": VERTEX_REGISTRY_STORE["vector_store_id"], "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            **_ingest_form(
                {
                    "vector_store_id": VERTEX_REGISTRY_STORE["vector_store_id"],
                    "max_embedding_requests_per_min": 500,
                    "vector_db_config": {"pinecone": {"index_name": "attacker-index"}},
                }
            ),
        )

    assert response.status_code == 200, response.json()
    assert mock_aingest.await_args.kwargs["ingest_options"]["vector_store"] == {
        "vector_store_id": VERTEX_REGISTRY_STORE["vector_store_id"],
        "custom_llm_provider": "vertex_ai",
        "vertex_project": "registry-project",
        "vertex_location": "us-central1",
        "max_embedding_requests_per_min": 500,
    }


def test_rag_ingest_rejects_registry_store_provider_without_ingestion_support(client_internal_user):
    aingest_patch, registry_patch = _patched_ingest_boundary(
        AZURE_REGISTRY_STORE, {"vector_store_id": "my-azure-index", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(None),
    ):
        response = client_internal_user.post("/v1/rag/ingest", **_ingest_form({"vector_store_id": "my-azure-index"}))

    assert response.status_code == 400, response.json()
    assert response.json()["detail"]["error"] == UNSUPPORTED_INGEST_PROVIDER_ERROR.format(provider="azure_ai")
    mock_aingest.assert_not_awaited()


def test_rag_ingest_rejects_request_provider_without_ingestion_support(client_internal_user):
    with (
        patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; the test asserts it is never reached
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new=AsyncMock(return_value={"vector_store_id": "vs_new", "file_id": "file-test"}),
        ) as mock_aingest,
        patch("litellm.vector_store_registry", None),  # test-quality-ok: proxy module global, no injection seam
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            json={"file_id": "file-test", "ingest_options": {"vector_store": {"custom_llm_provider": "milvus"}}},
        )

    assert response.status_code == 400, response.json()
    assert response.json() == {"detail": {"error": UNSUPPORTED_INGEST_PROVIDER_ERROR.format(provider="milvus")}}
    mock_aingest.assert_not_awaited()


def test_rag_ingest_rejects_non_string_provider(client_internal_user):
    with (
        patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; the test asserts it is never reached
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new=AsyncMock(return_value={"vector_store_id": "vs_new", "file_id": "file-test"}),
        ) as mock_aingest,
        patch("litellm.vector_store_registry", None),  # test-quality-ok: proxy module global, no injection seam
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            json={
                "file_id": "file-test",
                "ingest_options": {"vector_store": {"custom_llm_provider": {"provider": "milvus"}}},
            },
        )

    assert response.status_code == 400, response.json()
    assert response.json() == {"detail": {"error": "custom_llm_provider must be a string"}}
    mock_aingest.assert_not_awaited()


def test_rag_ingest_never_creates_db_row_for_registry_store(client_internal_user):
    prisma_client = MagicMock()
    prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(return_value=None)
    create_in_db = AsyncMock()
    aingest_patch, registry_patch = _patched_ingest_boundary(
        S3_REGISTRY_STORE, {"vector_store_id": "s3-store", "file_id": "file_123"}
    )
    with (
        aingest_patch,
        registry_patch,
        _patched_prisma_client(prisma_client),
        patch(  # test-quality-ok: the DB write boundary the guard under test must never reach
            "litellm.proxy.vector_store_endpoints.management_endpoints.create_vector_store_in_db",
            new=create_in_db,
        ),
    ):
        response = client_internal_user.post("/v1/rag/ingest", **_ingest_form({"vector_store_id": "s3-store"}))

    assert response.status_code == 200, response.json()
    prisma_client.db.litellm_managedvectorstorestable.find_unique.assert_awaited_once()
    create_in_db.assert_not_awaited()
    prisma_client.db.litellm_managedvectorstorestable.update.assert_not_called()


def test_rag_ingest_fresh_store_creates_db_row_with_the_requesters_params(client_internal_user):
    prisma_client = MagicMock()
    prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(return_value=None)
    create_in_db = AsyncMock()
    with (
        patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; persistence is what the test asserts
            "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
            new=AsyncMock(return_value={"vector_store_id": "vs_new", "file_id": "file_123"}),
        ),
        patch("litellm.vector_store_registry", None),  # test-quality-ok: proxy module global, no injection seam
        _patched_prisma_client(prisma_client),
        patch(  # test-quality-ok: the DB write boundary whose inputs the test asserts
            "litellm.proxy.vector_store_endpoints.management_endpoints.create_vector_store_in_db",
            new=create_in_db,
        ),
    ):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            **_ingest_form({"custom_llm_provider": "bedrock", "aws_region_name": "us-east-1"}),
        )

    assert response.status_code == 200, response.json()
    create_in_db.assert_awaited_once()
    created = create_in_db.await_args.kwargs
    assert created["vector_store_id"] == "vs_new"
    assert created["custom_llm_provider"] == "bedrock"
    assert created["litellm_params"] == {"aws_region_name": "us-east-1"}


def test_rag_ingest_hands_persistence_the_requesters_options_not_registry_credentials(client_internal_user):
    save_helper = AsyncMock()
    aingest_patch, registry_patch = _patched_ingest_boundary(
        BEDROCK_REGISTRY_STORE, {"vector_store_id": "kb-store", "file_id": "file_123"}
    )
    with (
        aingest_patch as mock_aingest,
        registry_patch,
        _patched_prisma_client(MagicMock()),
        patch(  # test-quality-ok: the persistence seam whose inputs the test asserts
            "litellm.proxy.rag_endpoints.endpoints._save_vector_store_to_db_from_rag_ingest",
            new=save_helper,
        ),
    ):
        response = client_internal_user.post("/v1/rag/ingest", **_ingest_form({"vector_store_id": "kb-store"}))

    assert response.status_code == 200, response.json()
    forwarded = mock_aingest.await_args.kwargs["ingest_options"]["vector_store"]
    assert forwarded["aws_secret_access_key"] == "registry-secret"
    save_helper.assert_awaited_once()
    assert save_helper.await_args.kwargs["ingest_options"]["vector_store"] == {"vector_store_id": "kb-store"}
    assert save_helper.await_args.kwargs["store_is_managed"] is True


async def test_save_vector_store_from_rag_ingest_appends_file_to_db_managed_store():
    from litellm.proxy.rag_endpoints.endpoints import _save_vector_store_to_db_from_rag_ingest

    existing_row = MagicMock()
    existing_row.vector_store_metadata = {"ingested_files": [{"file_id": "file_old"}]}
    prisma_client = MagicMock()
    table = prisma_client.db.litellm_managedvectorstorestable
    table.find_unique = AsyncMock(return_value=existing_row)
    table.update = AsyncMock()
    create_in_db = AsyncMock()

    with patch(  # test-quality-ok: the DB write boundary the append branch must not reach
        "litellm.proxy.vector_store_endpoints.management_endpoints.create_vector_store_in_db",
        new=create_in_db,
    ):
        await _save_vector_store_to_db_from_rag_ingest(
            response={"vector_store_id": "vs_db_managed", "file_id": "file_new"},
            ingest_options={"vector_store": {"vector_store_id": "vs_db_managed"}},
            prisma_client=prisma_client,
            user_api_key_dict=UserAPIKeyAuth(user_id="user-1", team_id="team-1"),
            store_is_managed=True,
        )

    create_in_db.assert_not_awaited()
    table.update.assert_awaited_once()
    stored_metadata = json.loads(table.update.await_args.kwargs["data"]["vector_store_metadata"])
    assert [entry["file_id"] for entry in stored_metadata["ingested_files"]] == ["file_old", "file_new"]


async def test_save_vector_store_from_rag_ingest_still_creates_row_for_fresh_store():
    from litellm.proxy.rag_endpoints.endpoints import _save_vector_store_to_db_from_rag_ingest

    prisma_client = MagicMock()
    prisma_client.db.litellm_managedvectorstorestable.find_unique = AsyncMock(return_value=None)
    create_in_db = AsyncMock()

    with patch(  # test-quality-ok: the DB write boundary whose inputs the test asserts
        "litellm.proxy.vector_store_endpoints.management_endpoints.create_vector_store_in_db",
        new=create_in_db,
    ):
        await _save_vector_store_to_db_from_rag_ingest(
            response={"vector_store_id": "vs_new", "file_id": "file_new"},
            ingest_options={"vector_store": {"custom_llm_provider": "bedrock", "aws_region_name": "us-east-1"}},
            prisma_client=prisma_client,
            user_api_key_dict=UserAPIKeyAuth(user_id="user-1", team_id="team-1"),
            store_is_managed=False,
        )

    create_in_db.assert_awaited_once()
    created = create_in_db.await_args.kwargs
    assert created["vector_store_id"] == "vs_new"
    assert created["custom_llm_provider"] == "bedrock"
    assert created["litellm_params"] == {"aws_region_name": "us-east-1"}
    assert created["team_id"] == "team-1"


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

    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        patch("litellm.vector_store_registry", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
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


@pytest.mark.parametrize(
    ("upstream_error", "expected_status"),
    [
        (litellm.BadRequestError(message="filter andAll needs two clauses", model="kb", llm_provider="bedrock"), 400),
        (litellm.NotFoundError(message="Knowledge Base does not exist", model="kb", llm_provider="bedrock"), 404),
        (RuntimeError("pipeline blew up"), 500),
    ],
)
def test_rag_query_surfaces_upstream_status_code(client_internal_user, upstream_error, expected_status):
    """A vector store rejection must reach the caller with its own status code, never a blanket 500."""
    with (
        patch(  # test-quality-ok: the handler calls the module-level litellm.aquery directly; no injection seam
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new=AsyncMock(side_effect=upstream_error),
        ),
        patch("litellm.vector_store_registry", None),  # test-quality-ok: proxy module global, no injection seam
        patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ),  # test-quality-ok: proxy module global, no injection seam
    ):
        response = client_internal_user.post(
            "/v1/rag/query",
            json={
                "model": "bedrock/us.anthropic.claude-sonnet-5",
                "messages": [{"role": "user", "content": "How was this document ingested?"}],
                "retrieval_config": {
                    "vector_store_id": "L7INRFMVQT",
                    "custom_llm_provider": "bedrock",
                    "retrieval_filter": {"andAll": [{"equals": {"key": "department", "value": "billing"}}]},
                },
            },
        )

    assert response.status_code == expected_status, response.text
    assert str(upstream_error) in response.json()["detail"]["error"]


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

    with (
        patch(
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new=AsyncMock(side_effect=fake_aquery),
        ),
        patch("litellm.vector_store_registry", None),
        patch("litellm.proxy.proxy_server.prisma_client", None),
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
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert "data: [DONE]" in response.text


def test_rag_query_stream_pings_while_retrieval_is_still_running(client_internal_user, monkeypatch):
    import asyncio

    import litellm as litellm_module

    monkeypatch.setattr(litellm_module, "sse_keepalive_ping_interval_seconds", 0.05)

    async def slow_aquery(**kwargs):
        await asyncio.sleep(0.3)
        return await litellm_module.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the codename?"}],
            mock_response="The codename is AZURE-FALCON-42.",
            stream=True,
            api_key="test-key",
        )

    with (
        patch(  # test-quality-ok: the handler calls the module-level litellm.aquery directly; no injection seam
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new=AsyncMock(side_effect=slow_aquery),
        ),
        patch("litellm.vector_store_registry", None),  # test-quality-ok: proxy module global, no injection seam
        patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ),  # test-quality-ok: proxy module global, no injection seam
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
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.startswith(": ping\n\n")
    assert response.text.count(": ping\n\n") >= 3
    assert '"object":"chat.completion.chunk"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_rag_query_stream_keeps_response_headers_when_retrieval_beats_the_keepalive(client_internal_user, monkeypatch):
    import litellm as litellm_module

    monkeypatch.setattr(litellm_module, "sse_keepalive_ping_interval_seconds", 5)

    async def fast_aquery(**kwargs):
        response = await litellm_module.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the codename?"}],
            mock_response="The codename is AZURE-FALCON-42.",
            stream=True,
            api_key="test-key",
        )
        response._hidden_params["response_cost"] = 3.45e-06
        return response

    with (
        patch(  # test-quality-ok: the handler calls the module-level litellm.aquery directly; no injection seam
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new=AsyncMock(side_effect=fast_aquery),
        ),
        patch("litellm.vector_store_registry", None),  # test-quality-ok: proxy module global, no injection seam
        patch(
            "litellm.proxy.proxy_server.prisma_client", None
        ),  # test-quality-ok: proxy module global, no injection seam
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
                "stream": True,
            },
        )

    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert response.headers.get("x-litellm-response-cost") == "3.45e-06"
    assert not response.text.startswith(": ping")
    assert '"object":"chat.completion.chunk"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_rag_query_merges_managed_store_params(client_internal_user):
    """
    Regression: /v1/rag/query must consult the managed vector store registry
    (like the direct /v1/vector_stores/{id}/search endpoint does) so that
    provider, region, embedding model, etc. don't have to be repeated in
    retrieval_config. Pre-fix the registry was never read, so managed S3
    Vectors stores failed with "aws_region_name is required".
    """
    import litellm
    from litellm.types.utils import ModelResponse

    mock_vector_store = {
        "vector_store_id": "s3-store",
        "custom_llm_provider": "s3_vectors",
        "litellm_params": {
            "aws_region_name": "eu-west-1",
            "embedding_model": "my-embed",
            "vector_bucket_name": "bkt",
        },
    }
    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = mock_vector_store

    mock_response = ModelResponse(
        id="chatcmpl-test",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        model="gpt-4o-mini",
    )

    with (
        patch(  # test-quality-ok: aquery is the endpoint's downstream boundary; the forwarded config is what the test asserts
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aquery,
        patch.object(litellm, "vector_store_registry", mock_registry),
        patch(  # test-quality-ok: seeds the managed-store registry the merge under test reads and grants access so real store resolution runs
            "litellm.proxy.vector_store_endpoints.utils.can_user_access_vector_store",
            new=AsyncMock(return_value=True),
        ),
    ):
        response = client_internal_user.post(
            "/v1/rag/query",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "retrieval_config": {"vector_store_id": "s3-store"},
            },
        )

    assert response.status_code == 200, response.json()
    mock_aquery.assert_awaited_once()
    forwarded_config = mock_aquery.await_args.kwargs["retrieval_config"]
    assert forwarded_config["vector_store_id"] == "s3-store"
    assert forwarded_config["custom_llm_provider"] == "s3_vectors"
    assert forwarded_config["aws_region_name"] == "eu-west-1"
    assert forwarded_config["embedding_model"] == "my-embed"
    assert forwarded_config["vector_bucket_name"] == "bkt"


def test_rag_query_store_params_win_over_user_retrieval_config(client_internal_user):
    """Registry values must win over user-supplied retrieval_config keys so callers cannot override store credentials."""
    import litellm
    from litellm.types.utils import ModelResponse

    mock_vector_store = {
        "vector_store_id": "s3-store",
        "custom_llm_provider": "s3_vectors",
        "litellm_params": {"aws_region_name": "eu-west-1"},
    }
    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = mock_vector_store

    mock_response = ModelResponse(
        id="chatcmpl-test",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        model="gpt-4o-mini",
    )

    with (
        patch(  # test-quality-ok: aquery is the endpoint's downstream boundary; the forwarded config is what the test asserts
            "litellm.proxy.rag_endpoints.endpoints.litellm.aquery",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_aquery,
        patch.object(litellm, "vector_store_registry", mock_registry),
        patch(  # test-quality-ok: seeds the managed-store registry the merge under test reads and grants access so real store resolution runs
            "litellm.proxy.vector_store_endpoints.utils.can_user_access_vector_store",
            new=AsyncMock(return_value=True),
        ),
    ):
        response = client_internal_user.post(
            "/v1/rag/query",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "retrieval_config": {"vector_store_id": "s3-store", "aws_region_name": "us-east-1"},
            },
        )

    assert response.status_code == 200, response.json()
    forwarded_config = mock_aquery.await_args.kwargs["retrieval_config"]
    assert forwarded_config["aws_region_name"] == "eu-west-1"


def test_rag_query_forwards_managed_store_credentials_to_search(client_internal_user):
    """
    Regression for LIT-6773: the registry store's api_key / api_base and its
    provider extras (Milvus outputFields, milvus_text_field) must reach the
    vector store search the way the direct /v1/vector_stores/{id}/search
    endpoint forwards them. Pre-fix the RAG path allowlisted them away and a
    managed Milvus store 500'd with "MILVUS_API_KEY is not set".
    """
    import litellm
    from litellm import Router
    from litellm.types.vector_stores import VectorStoreSearchResponse

    mock_vector_store = {
        "vector_store_id": "customer_kb",
        "custom_llm_provider": "milvus",
        "litellm_params": {
            "vector_store_id": "customer_kb",
            "custom_llm_provider": "milvus",
            "api_base": "http://127.0.0.1:19530",
            "api_key": "root:Milvus",
            "litellm_embedding_model": "multilingual-e5-large",
            "milvus_text_field": "book_intro_text",
            "outputFields": ["book_intro_text"],
        },
    }
    mock_registry = MagicMock()
    mock_registry.get_litellm_managed_vector_store_from_registry.return_value = mock_vector_store
    fake_search = AsyncMock(
        return_value=VectorStoreSearchResponse(object="vector_store.search_results.page", search_query="q", data=[])
    )
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test", "mock_response": "hi"},
            }
        ]
    )

    with (
        patch("litellm.vector_stores.asearch", new=fake_search),  # test-quality-ok: the search boundary under test
        patch.object(litellm, "vector_store_registry", mock_registry),  # test-quality-ok: seeds the store under test
        patch("litellm.proxy.proxy_server.llm_router", router),  # test-quality-ok: mock-response router for completion
        patch(  # test-quality-ok: store access is not under test, so the request reaches the search boundary
            "litellm.proxy.vector_store_endpoints.utils.can_user_access_vector_store",
            new=AsyncMock(return_value=True),
        ),
    ):
        response = client_internal_user.post(
            "/v1/rag/query",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "which database is built for similarity search?"}],
                "retrieval_config": {"vector_store_id": "customer_kb", "custom_llm_provider": "milvus", "top_k": 2},
            },
        )

    assert response.status_code == 200, response.json()
    fake_search.assert_awaited_once()
    search_kwargs = fake_search.await_args.kwargs
    assert search_kwargs["vector_store_id"] == "customer_kb"
    assert search_kwargs["custom_llm_provider"] == "milvus"
    assert search_kwargs["max_num_results"] == 2
    assert search_kwargs["api_base"] == "http://127.0.0.1:19530"
    assert search_kwargs["api_key"] == "root:Milvus"
    assert search_kwargs["litellm_embedding_model"] == "multilingual-e5-large"
    assert search_kwargs["milvus_text_field"] == "book_intro_text"
    assert search_kwargs["outputFields"] == ["book_intro_text"]


@pytest.mark.parametrize(
    "blocked_key",
    ["embedding_model", "litellm_embedding_model", "litellm_embedding_config", "litellm_credential_name"],
)
def test_rag_query_rejects_caller_embedding_selection_params(client_internal_user, blocked_key):
    """
    Regression: a caller must not pick the embedding model or credential used at
    search time. Those resolve through the Router with the proxy's credentials,
    bypassing the key's model permissions, so they may only come from the
    managed store's server-side registration.
    """
    response = client_internal_user.post(
        "/v1/rag/query",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "retrieval_config": {"vector_store_id": "s3-store", blocked_key: "attacker-choice"},
        },
    )

    assert response.status_code == 400, response.json()
    assert blocked_key in str(response.json())


EICAR = r"X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
INGEST_REQUEST = '{"ingest_options":{"vector_store":{"custom_llm_provider":"openai"}}}'


def _multipart_ingest_request(*, filename: str, content: bytes, content_type: str):
    from starlette.requests import Request

    boundary = "litellmuploadtestboundary"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    tail = (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="request"\r\n\r\n'
        f"{INGEST_REQUEST}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    body = head + content + tail
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/rag/ingest",
        "headers": [
            (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "state": {},
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


async def _never_fetch(url: str):
    raise AssertionError(f"no fetch expected, got {url}")


@dataclass(frozen=True)
class _FixedScanner:
    verdict: ScanVerdict

    def scan(self, content: bytes) -> ScanResult:
        return ScanResult(verdict=self.verdict, signature="Custom.Sig")


def _fetcher_answering(status_code: int, content: bytes, content_type: str = "text/plain"):
    seen: list[str] = []

    async def fetch_url(url: str) -> httpx.Response:
        seen.append(url)
        return httpx.Response(status_code, content=content, headers={"content-type": content_type})

    return fetch_url, seen


FILE_URL_BODY = {
    "file_url": "https://docs.example.com/reports/q3.txt",
    "ingest_options": json.loads(INGEST_REQUEST)["ingest_options"],
}


class TestFileUrlUploadControls:
    """A ``file_url`` is fetched by the proxy and goes through the same controls as an inline upload."""

    async def _fetch(self, fetch_url, scanner=None):
        from litellm.proxy.rag_endpoints.endpoints import fetch_and_secure_file_url

        return await fetch_and_secure_file_url(
            FILE_URL_BODY["file_url"],
            scanner=scanner or EicarTestMalwareScanner(),
            fetch_url=fetch_url,
        )

    async def _expect_rejection(self, fetch_url, reason: str | None, scanner=None) -> dict:
        with pytest.raises(HTTPException) as excinfo:
            await self._fetch(fetch_url, scanner=scanner)
        assert excinfo.value.status_code == 400
        detail = excinfo.value.detail
        assert detail.get("reason") == reason, detail
        return detail

    async def test_eicar_behind_a_url_is_blocked(self):
        fetch_url, seen = _fetcher_answering(200, EICAR.encode())
        await self._expect_rejection(fetch_url, "malware_detected")
        assert seen == [FILE_URL_BODY["file_url"]]

    async def test_shebang_behind_a_url_is_blocked(self):
        fetch_url, _seen = _fetcher_answering(200, b"#!/bin/sh\nrm -rf /\n")
        await self._expect_rejection(fetch_url, "executable_not_allowed")

    async def test_oversize_url_is_blocked_before_it_is_buffered(self):
        async def fetch_url(url: str) -> httpx.Response:
            raise HTTPResponseLimitError(f"Response exceeded {MAX_UPLOAD_SIZE_BYTES} bytes")

        detail = await self._expect_rejection(fetch_url, "file_too_large")
        assert str(MAX_UPLOAD_SIZE_BYTES) in detail["error"]

    async def test_compressed_url_response_is_refused_without_calling_it_too_large(self):
        async def fetch_url(url: str) -> httpx.Response:
            raise HTTPResponseEncodingError("Response size limits require an uncompressed response")

        detail = await self._expect_rejection(fetch_url, None)
        assert "uncompressed" in detail["error"]

    @pytest.mark.parametrize("status_code", [301, 403, 404, 500])
    async def test_non_2xx_fetch_is_a_client_error_naming_the_status(self, status_code):
        fetch_url, _seen = _fetcher_answering(status_code, b"benign document text")
        detail = await self._expect_rejection(fetch_url, None)
        assert f"HTTP {status_code}" in detail["error"]

    async def test_ssrf_or_transport_failure_is_a_client_error(self):
        async def fetch_url(url: str) -> httpx.Response:
            raise ValueError("Blocked private address")

        detail = await self._expect_rejection(fetch_url, None)
        assert "Blocked private address" in detail["error"]

    async def test_clean_text_behind_a_url_is_handed_on_as_bytes(self):
        fetch_url, _seen = _fetcher_answering(200, b"benign document text\n", content_type="text/x-whatever")
        server_filename, content_bytes, content_type = await self._fetch(fetch_url)
        assert content_bytes == b"benign document text\n"
        assert server_filename.endswith(".txt") and "/" not in server_filename
        assert content_type == "text/plain"

    async def test_url_bytes_go_through_the_injected_scanner(self):
        fetch_url, _seen = _fetcher_answering(200, b"benign document text\n")
        await self._expect_rejection(fetch_url, "malware_detected", scanner=_FixedScanner(ScanVerdict.INFECTED))

    def test_route_prefers_an_inline_file_and_never_fetches_the_url(self, client_internal_user, monkeypatch):
        monkeypatch.setattr("litellm.proxy.rag_endpoints.endpoints.fetch_file_url", _never_fetch)
        inline = {"filename": "inline.txt", "content": base64.b64encode(b"inline text").decode()}
        with (
            patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; the test asserts the forwarded bytes
                "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
                new_callable=AsyncMock,
                return_value={"vector_store_id": "vs_new", "file_id": "file_123"},
            ) as mock_aingest
        ):
            response = client_internal_user.post("/v1/rag/ingest", json={**FILE_URL_BODY, "file": inline})
        assert response.status_code == 200, response.text
        assert mock_aingest.await_args.kwargs["file_data"][1] == b"inline text"

    @pytest.mark.parametrize("field", [{"file_url": ["https://docs.example.com/q3.txt"]}, {"file_id": {"id": 1}}])
    def test_route_rejects_a_non_string_file_reference(self, client_internal_user, monkeypatch, field):
        monkeypatch.setattr("litellm.proxy.rag_endpoints.endpoints.fetch_file_url", _never_fetch)
        body = {"ingest_options": FILE_URL_BODY["ingest_options"], **field}
        response = client_internal_user.post("/v1/rag/ingest", json=body)
        assert response.status_code == 400, response.text
        assert f"'{next(iter(field))}' must be a string" in response.json()["detail"]["error"]

    def test_route_validates_the_request_before_fetching_the_url(self, client_internal_user, monkeypatch):
        monkeypatch.setattr("litellm.proxy.rag_endpoints.endpoints.fetch_file_url", _never_fetch)
        vector_store = {"custom_llm_provider": "bedrock", "aws_sts_endpoint": "https://attacker.example/sts"}
        body = {**FILE_URL_BODY, "ingest_options": {"vector_store": vector_store}}
        response = client_internal_user.post("/v1/rag/ingest", json=body)
        assert response.status_code == 400, response.text
        assert "aws_sts_endpoint" in response.json()["detail"]["error"]

    def test_route_hands_fetched_bytes_to_ingestion_and_never_the_url(self, client_internal_user, monkeypatch):
        fetch_url, seen = _fetcher_answering(200, b"benign document text\n")
        monkeypatch.setattr("litellm.proxy.rag_endpoints.endpoints.fetch_file_url", fetch_url)
        with (
            patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; the test asserts the forwarded bytes
                "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
                new_callable=AsyncMock,
                return_value={"vector_store_id": "vs_new", "file_id": "file_123"},
            ) as mock_aingest
        ):
            response = client_internal_user.post("/v1/rag/ingest", json=FILE_URL_BODY)
        assert response.status_code == 200, response.text
        assert seen == [FILE_URL_BODY["file_url"]]
        forwarded = mock_aingest.await_args.kwargs
        assert forwarded["file_data"][1] == b"benign document text\n"
        assert "file_url" not in forwarded

    def test_route_rejects_eicar_behind_a_url(self, client_internal_user, monkeypatch):
        fetch_url, _seen = _fetcher_answering(200, EICAR.encode())
        monkeypatch.setattr("litellm.proxy.rag_endpoints.endpoints.fetch_file_url", fetch_url)
        with (
            patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; the test asserts it is never reached
                "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
                new_callable=AsyncMock,
            ) as mock_aingest
        ):
            response = client_internal_user.post("/v1/rag/ingest", json=FILE_URL_BODY)
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["reason"] == "malware_detected"
        mock_aingest.assert_not_awaited()


class TestVectorStoreUploadControls:
    """End-to-end enforcement of pentest M4 upload controls on /v1/rag/ingest."""

    def test_eicar_upload_blocked_by_malware_scanner(self, client_internal_user):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            files={"file": ("clean_name.txt", io.BytesIO(EICAR.encode()), "text/plain")},
            data={"request": INGEST_REQUEST},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["reason"] == "malware_detected"

    def test_executable_upload_rejected(self, client_internal_user):
        elf = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 40
        response = client_internal_user.post(
            "/v1/rag/ingest",
            files={"file": ("doc.txt", io.BytesIO(elf), "text/plain")},
            data={"request": INGEST_REQUEST},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["reason"] == "executable_not_allowed"

    def test_zip_archive_upload_rejected(self, client_internal_user):
        response = client_internal_user.post(
            "/v1/rag/ingest",
            files={"file": ("doc.pdf", io.BytesIO(b"PK\x03\x04\x14\x00\x00\x00payload"), "application/pdf")},
            data={"request": INGEST_REQUEST},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["reason"] == "archive_not_allowed"

    def test_route_runs_the_configured_scanner(self, client_internal_user, monkeypatch):
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.rag_upload_malware_scanner", _FixedScanner(ScanVerdict.INFECTED)
        )
        with (
            patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; the test asserts it is never reached
                "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
                new_callable=AsyncMock,
            ) as mock_aingest
        ):
            response = client_internal_user.post(
                "/v1/rag/ingest",
                files={"file": ("doc.txt", io.BytesIO(b"benign document text"), "text/plain")},
                data={"request": INGEST_REQUEST},
            )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["reason"] == "malware_detected"
        assert "Custom.Sig" in response.json()["detail"]["error"]
        mock_aingest.assert_not_awaited()

    def test_route_lets_a_clean_verdict_from_the_configured_scanner_through(self, client_internal_user, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.rag_upload_malware_scanner", _FixedScanner(ScanVerdict.CLEAN))
        with (
            patch(  # test-quality-ok: aingest is the endpoint's downstream boundary; the test asserts the forwarded bytes
                "litellm.proxy.rag_endpoints.endpoints.litellm.aingest",
                new_callable=AsyncMock,
                return_value={"vector_store_id": "vs_new", "file_id": "file_123"},
            ) as mock_aingest
        ):
            response = client_internal_user.post(
                "/v1/rag/ingest",
                files={"file": ("clean_name.txt", io.BytesIO(EICAR.encode()), "text/plain")},
                data={"request": INGEST_REQUEST},
            )
        assert response.status_code == 200, response.text
        assert mock_aingest.await_args.kwargs["file_data"][1] == EICAR.encode()

    async def test_clean_text_upload_gets_server_generated_filename(self):
        from litellm.proxy.rag_endpoints.endpoints import parse_rag_ingest_request
        from litellm.proxy.rag_endpoints.upload_security import EicarTestMalwareScanner

        request = _multipart_ingest_request(
            filename="../../etc/passwd",
            content=b"benign document text\n",
            content_type="text/plain",
        )
        _options, file_data, _url, _file_id = await parse_rag_ingest_request(request, scanner=EicarTestMalwareScanner())
        assert file_data is not None
        server_filename, content_bytes, secured_content_type = file_data
        assert server_filename != "../../etc/passwd"
        assert "/" not in server_filename and "\\" not in server_filename
        assert server_filename.endswith(".txt")
        assert secured_content_type == "text/plain"
        assert content_bytes == b"benign document text\n"
