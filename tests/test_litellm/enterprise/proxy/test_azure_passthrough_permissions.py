"""
Tests for Azure pass-through permission management.

Tests cover:
- URL classification for all resource types and operations
- Pre-request ownership checks and ID rewriting
- Post-response ID encoding and ownership storage
- List filtering by user
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.common_utils import (
    encode_file_id_with_model,
)
from litellm_enterprise.proxy.hooks.azure_passthrough_permissions import (
    AzurePassthroughOp,
    classify_azure_passthrough_request,
    passthrough_list_filter,
    passthrough_post_response,
    passthrough_pre_request,
)


# ============================================================================
#                    URL CLASSIFICATION TESTS
# ============================================================================


class TestClassifyAzurePassthroughRequest:
    def test_create_batch(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/batches", "POST"
        )
        assert op is not None
        assert op.resource_type == "batch"
        assert op.operation == "create"
        assert op.resource_id is None

    def test_retrieve_batch(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/batches/batch_abc123", "GET"
        )
        assert op is not None
        assert op.resource_type == "batch"
        assert op.operation == "retrieve"
        assert op.resource_id == "batch_abc123"

    def test_cancel_batch(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/batches/batch_abc123/cancel", "POST"
        )
        assert op is not None
        assert op.resource_type == "batch"
        assert op.operation == "cancel"
        assert op.resource_id == "batch_abc123"

    def test_list_batches(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/batches", "GET"
        )
        assert op is not None
        assert op.resource_type == "batch"
        assert op.operation == "list"
        assert op.resource_id is None

    def test_create_file(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/files", "POST"
        )
        assert op is not None
        assert op.resource_type == "file"
        assert op.operation == "create"
        assert op.resource_id is None

    def test_retrieve_file(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/files/file-abc123", "GET"
        )
        assert op is not None
        assert op.resource_type == "file"
        assert op.operation == "retrieve"
        assert op.resource_id == "file-abc123"

    def test_delete_file(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/files/file-abc123", "DELETE"
        )
        assert op is not None
        assert op.resource_type == "file"
        assert op.operation == "delete"
        assert op.resource_id == "file-abc123"

    def test_get_file_content(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/files/file-abc123/content", "GET"
        )
        assert op is not None
        assert op.resource_type == "file"
        assert op.operation == "content"
        assert op.resource_id == "file-abc123"

    def test_list_files(self):
        op = classify_azure_passthrough_request("openai/deployments/gpt-4/files", "GET")
        assert op is not None
        assert op.resource_type == "file"
        assert op.operation == "list"
        assert op.resource_id is None

    def test_create_response(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/responses", "POST"
        )
        assert op is not None
        assert op.resource_type == "response"
        assert op.operation == "create"
        assert op.resource_id is None

    def test_retrieve_response(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/responses/resp_abc123", "GET"
        )
        assert op is not None
        assert op.resource_type == "response"
        assert op.operation == "retrieve"
        assert op.resource_id == "resp_abc123"

    def test_delete_response(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/responses/resp_abc123", "DELETE"
        )
        assert op is not None
        assert op.resource_type == "response"
        assert op.operation == "delete"
        assert op.resource_id == "resp_abc123"

    def test_non_matching_endpoint(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/completions", "POST"
        )
        assert op is None

    def test_non_matching_chat_completions(self):
        op = classify_azure_passthrough_request(
            "openai/deployments/gpt-4/chat/completions", "POST"
        )
        assert op is None


# ============================================================================
#                    PRE-REQUEST TESTS
# ============================================================================


class TestPassthroughPreRequest:
    @pytest.mark.asyncio
    async def test_non_matching_endpoint_passes_through(self):
        endpoint = "openai/deployments/gpt-4/completions"
        new_endpoint, op = await passthrough_pre_request(
            endpoint=endpoint,
            request_method="POST",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=None,
        )
        assert new_endpoint == endpoint
        assert op is None

    @pytest.mark.asyncio
    async def test_create_operation_passes_through(self):
        endpoint = "openai/deployments/gpt-4/batches"
        new_endpoint, op = await passthrough_pre_request(
            endpoint=endpoint,
            request_method="POST",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=MagicMock(),
        )
        assert new_endpoint == endpoint
        assert op is not None
        assert op.operation == "create"

    @pytest.mark.asyncio
    async def test_retrieve_with_managed_batch_id_owner_allowed(self):
        """Owner can retrieve a managed batch."""
        # Encode a batch ID with model info
        managed_id = encode_file_id_with_model(
            file_id="batch_real123", model="gpt-4", id_type="batch"
        )

        managed_files_obj = AsyncMock()
        managed_files_obj.can_user_call_unified_object_id = AsyncMock(return_value=True)

        endpoint = f"openai/deployments/gpt-4/batches/{managed_id}"
        new_endpoint, op = await passthrough_pre_request(
            endpoint=endpoint,
            request_method="GET",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
        )
        assert op is not None
        assert op.operation == "retrieve"
        # Endpoint should be rewritten with the real provider ID
        assert "batch_real123" in new_endpoint
        assert managed_id not in new_endpoint

    @pytest.mark.asyncio
    async def test_retrieve_with_managed_batch_id_non_owner_blocked(self):
        """Non-owner gets 403 when trying to retrieve a managed batch."""
        managed_id = encode_file_id_with_model(
            file_id="batch_real123", model="gpt-4", id_type="batch"
        )

        managed_files_obj = AsyncMock()
        managed_files_obj.can_user_call_unified_object_id = AsyncMock(
            return_value=False
        )

        endpoint = f"openai/deployments/gpt-4/batches/{managed_id}"
        with pytest.raises(HTTPException) as exc_info:
            await passthrough_pre_request(
                endpoint=endpoint,
                request_method="GET",
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="user2", parent_otel_span=MagicMock()
                ),
                managed_files_obj=managed_files_obj,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_retrieve_with_managed_file_id_owner_allowed(self):
        """Owner can retrieve a managed file."""
        managed_id = encode_file_id_with_model(
            file_id="file-real456", model="gpt-4", id_type="file"
        )

        managed_files_obj = AsyncMock()
        managed_files_obj.can_user_call_unified_file_id = AsyncMock(return_value=True)

        endpoint = f"openai/deployments/gpt-4/files/{managed_id}"
        new_endpoint, op = await passthrough_pre_request(
            endpoint=endpoint,
            request_method="GET",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
        )
        assert "file-real456" in new_endpoint
        assert managed_id not in new_endpoint

    @pytest.mark.asyncio
    async def test_retrieve_with_managed_file_id_non_owner_blocked(self):
        """Non-owner gets 403 when trying to retrieve a managed file."""
        managed_id = encode_file_id_with_model(
            file_id="file-real456", model="gpt-4", id_type="file"
        )

        managed_files_obj = AsyncMock()
        managed_files_obj.can_user_call_unified_file_id = AsyncMock(return_value=False)

        endpoint = f"openai/deployments/gpt-4/files/{managed_id}"
        with pytest.raises(HTTPException) as exc_info:
            await passthrough_pre_request(
                endpoint=endpoint,
                request_method="DELETE",
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="user2", parent_otel_span=MagicMock()
                ),
                managed_files_obj=managed_files_obj,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_raw_provider_id_ownership_check_via_db(self):
        """Raw (non-encoded) IDs are checked via model_object_id lookup."""
        managed_files_obj = AsyncMock()
        db_obj = MagicMock()
        db_obj.created_by = "user1"
        managed_files_obj.prisma_client.db.litellm_managedobjecttable.find_first = (
            AsyncMock(return_value=db_obj)
        )

        endpoint = "openai/deployments/gpt-4/batches/batch_raw789"
        new_endpoint, op = await passthrough_pre_request(
            endpoint=endpoint,
            request_method="GET",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
        )
        # Should pass through since ownership matches
        assert new_endpoint == endpoint

    @pytest.mark.asyncio
    async def test_raw_provider_id_non_owner_blocked(self):
        """Raw ID with different owner gets 403."""
        managed_files_obj = AsyncMock()
        db_obj = MagicMock()
        db_obj.created_by = "user1"
        managed_files_obj.prisma_client.db.litellm_managedobjecttable.find_first = (
            AsyncMock(return_value=db_obj)
        )

        endpoint = "openai/deployments/gpt-4/batches/batch_raw789"
        with pytest.raises(HTTPException) as exc_info:
            await passthrough_pre_request(
                endpoint=endpoint,
                request_method="GET",
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="user2", parent_otel_span=MagicMock()
                ),
                managed_files_obj=managed_files_obj,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_untracked_raw_id_passes_through(self):
        """Raw ID not in DB passes through (backward compat)."""
        managed_files_obj = AsyncMock()
        managed_files_obj.prisma_client.db.litellm_managedobjecttable.find_first = (
            AsyncMock(return_value=None)
        )

        endpoint = "openai/deployments/gpt-4/batches/batch_unknown"
        new_endpoint, op = await passthrough_pre_request(
            endpoint=endpoint,
            request_method="GET",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
        )
        assert new_endpoint == endpoint

    @pytest.mark.asyncio
    async def test_no_managed_files_obj_passes_through(self):
        """When enterprise is not available, everything passes through."""
        endpoint = "openai/deployments/gpt-4/batches/batch_abc"
        new_endpoint, op = await passthrough_pre_request(
            endpoint=endpoint,
            request_method="GET",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=None,
        )
        assert new_endpoint == endpoint
        assert op is not None
        assert op.operation == "retrieve"


# ============================================================================
#                    POST-RESPONSE TESTS
# ============================================================================


class TestPassthroughPostResponse:
    @pytest.mark.asyncio
    async def test_create_batch_encodes_id_and_stores(self):
        """Batch create response gets encoded ID and stored in DB."""
        response_json = {
            "id": "batch_abc123",
            "object": "batch",
            "status": "validating",
            "input_file_id": "file-input456",
            "output_file_id": None,
            "error_file_id": None,
            "completion_window": "24h",
            "created_at": 1234567890,
            "endpoint": "/v1/chat/completions",
        }
        response_body = json.dumps(response_json).encode()

        managed_files_obj = AsyncMock()
        managed_files_obj.store_unified_object_id = AsyncMock()

        op = AzurePassthroughOp(
            resource_type="batch", operation="create", resource_id=None
        )
        result = await passthrough_post_response(
            response_body=response_body,
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
            deployment_name="gpt-4",
        )

        result_json = json.loads(result)
        # ID should be encoded (not the raw provider ID)
        assert result_json["id"] != "batch_abc123"
        assert result_json["id"].startswith("batch_")
        # input_file_id should also be encoded
        assert result_json["input_file_id"] != "file-input456"
        assert result_json["input_file_id"].startswith("file-")
        # DB storage should have been called
        managed_files_obj.store_unified_object_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_file_encodes_id_and_stores(self):
        """File create response gets encoded ID and stored in DB."""
        response_json = {
            "id": "file-xyz789",
            "object": "file",
            "bytes": 1234,
            "created_at": 1234567890,
            "filename": "test.jsonl",
            "purpose": "batch",
        }
        response_body = json.dumps(response_json).encode()

        managed_files_obj = AsyncMock()
        managed_files_obj.store_unified_file_id = AsyncMock()

        op = AzurePassthroughOp(
            resource_type="file", operation="create", resource_id=None
        )
        result = await passthrough_post_response(
            response_body=response_body,
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
            deployment_name="gpt-4",
        )

        result_json = json.loads(result)
        assert result_json["id"] != "file-xyz789"
        assert result_json["id"].startswith("file-")
        managed_files_obj.store_unified_file_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_create_operation_passes_through(self):
        """Non-create operations don't modify the response."""
        response_body = b'{"id": "batch_abc", "status": "completed"}'

        op = AzurePassthroughOp(
            resource_type="batch", operation="retrieve", resource_id="batch_abc"
        )
        result = await passthrough_post_response(
            response_body=response_body,
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=AsyncMock(),
        )
        assert result == response_body

    @pytest.mark.asyncio
    async def test_invalid_json_passes_through(self):
        """Invalid JSON response is returned unchanged."""
        response_body = b"not valid json"

        op = AzurePassthroughOp(
            resource_type="batch", operation="create", resource_id=None
        )
        result = await passthrough_post_response(
            response_body=response_body,
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=AsyncMock(),
        )
        assert result == response_body

    @pytest.mark.asyncio
    async def test_no_managed_files_passes_through(self):
        """Without managed_files_obj, response passes through unchanged."""
        response_body = b'{"id": "batch_abc"}'

        op = AzurePassthroughOp(
            resource_type="batch", operation="create", resource_id=None
        )
        result = await passthrough_post_response(
            response_body=response_body,
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=None,
        )
        assert result == response_body


# ============================================================================
#                    LIST FILTER TESTS
# ============================================================================


class TestPassthroughListFilter:
    @pytest.mark.asyncio
    async def test_list_batches_filters_by_user(self):
        """List batches returns user-filtered results from managed tables."""
        managed_files_obj = AsyncMock()
        managed_files_obj.list_user_batches = AsyncMock(
            return_value={
                "object": "list",
                "data": [{"id": "batch_1", "status": "completed"}],
                "has_more": False,
            }
        )

        op = AzurePassthroughOp(
            resource_type="batch", operation="list", resource_id=None
        )
        result = await passthrough_list_filter(
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["data"][0]["id"] == "batch_1"

    @pytest.mark.asyncio
    async def test_list_files_filters_by_user(self):
        """List files returns user-filtered results from managed tables."""
        mock_file = MagicMock()
        mock_file.file_object = json.dumps(
            {
                "id": "file-raw",
                "object": "file",
                "bytes": 100,
                "created_at": 1234567890,
                "filename": "test.jsonl",
                "purpose": "batch",
            }
        )
        mock_file.unified_file_id = "file-managed123"

        managed_files_obj = AsyncMock()
        managed_files_obj.prisma_client.db.litellm_managedfiletable.find_many = (
            AsyncMock(return_value=[mock_file])
        )

        op = AzurePassthroughOp(
            resource_type="file", operation="list", resource_id=None
        )
        result = await passthrough_list_filter(
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=managed_files_obj,
        )
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "file-managed123"

    @pytest.mark.asyncio
    async def test_non_list_operation_returns_none(self):
        """Non-list operations return None (fall through to Azure)."""
        op = AzurePassthroughOp(
            resource_type="batch", operation="create", resource_id=None
        )
        result = await passthrough_list_filter(
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=AsyncMock(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_managed_files_returns_none(self):
        """Without managed_files_obj, returns None (fall through)."""
        op = AzurePassthroughOp(
            resource_type="batch", operation="list", resource_id=None
        )
        result = await passthrough_list_filter(
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_response_list_returns_none(self):
        """List responses not yet supported, returns None."""
        op = AzurePassthroughOp(
            resource_type="response", operation="list", resource_id=None
        )
        result = await passthrough_list_filter(
            op=op,
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user1", parent_otel_span=MagicMock()
            ),
            managed_files_obj=AsyncMock(),
        )
        assert result is None
