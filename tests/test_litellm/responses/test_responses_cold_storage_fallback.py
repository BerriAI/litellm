"""
Unit tests for the cold storage fallback paths in GET/DELETE /responses/{id}.

When a provider (Bedrock, Anthropic, etc.) doesn't support native GET/DELETE
on the Responses API, the proxy falls back to querying SpendLogs DB (and
optionally S3 cold storage).
"""

import json
import sys
from types import ModuleType
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.spend_tracking.cold_storage_handler import ColdStorageHandler
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.responses.main import DeleteResponseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RESPONSE_DICT = {
    "id": "resp_test123",
    "created_at": 1700000000,
    "model": "bedrock/anthropic.claude-v2",
    "object": "response",
    "output": [
        {
            "type": "message",
            "id": "msg_001",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!"}],
            "status": "completed",
        }
    ],
    "status": "completed",
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}

# Encoded response_id that decodes to a bedrock provider
# (We mock the decode so the actual base64 value doesn't matter)
ENCODED_RESPONSE_ID = "resp_dGVzdA=="


def _make_spend_log_row(
    response: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Build a minimal SpendLogs row dict as returned by query_raw."""
    row: dict = {
        "request_id": "test-request-id",
        "response": response or SAMPLE_RESPONSE_DICT,
    }
    if metadata is not None:
        row["metadata"] = json.dumps(metadata)
    else:
        row["metadata"] = "{}"
    return row


def _setup_mock_proxy_server(prisma_client_value):
    """
    Inject a mock litellm.proxy.proxy_server module into sys.modules
    so the lazy import inside session_handler works without loading
    the real (heavy) proxy_server module.
    """
    mock_module = ModuleType("litellm.proxy.proxy_server")
    mock_module.prisma_client = prisma_client_value  # type: ignore
    sys.modules["litellm.proxy.proxy_server"] = mock_module
    return mock_module


def _teardown_mock_proxy_server():
    """Remove the mock proxy_server module from sys.modules."""
    sys.modules.pop("litellm.proxy.proxy_server", None)


# ---------------------------------------------------------------------------
# get_response_from_spend_logs
# ---------------------------------------------------------------------------


class TestGetResponseFromSpendLogs:
    def teardown_method(self):
        _teardown_mock_proxy_server()

    @pytest.mark.asyncio
    async def test_success(self):
        """SpendLog has a valid response dict -> return ResponsesAPIResponse."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(
            return_value=[_make_spend_log_row()]
        )
        _setup_mock_proxy_server(mock_prisma)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "test-request-id", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            result = await ResponsesSessionHandler.get_response_from_spend_logs(
                ENCODED_RESPONSE_ID
            )

        assert result is not None
        assert isinstance(result, ResponsesAPIResponse)
        assert result.id == "resp_test123"

    @pytest.mark.asyncio
    async def test_not_found(self):
        """No matching SpendLog -> return None."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(return_value=[])
        _setup_mock_proxy_server(mock_prisma)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "missing-id", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            result = await ResponsesSessionHandler.get_response_from_spend_logs(
                ENCODED_RESPONSE_ID
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_corrupted_response(self):
        """SpendLog has invalid JSON in response -> return None."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(
            return_value=[{"request_id": "bad", "response": "not-valid-json{{{", "metadata": "{}"}]
        )
        _setup_mock_proxy_server(mock_prisma)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "bad", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            result = await ResponsesSessionHandler.get_response_from_spend_logs(
                ENCODED_RESPONSE_ID
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_no_prisma_client(self):
        """No prisma client configured -> return None."""
        _setup_mock_proxy_server(None)

        result = await ResponsesSessionHandler.get_response_from_spend_logs(
            ENCODED_RESPONSE_ID
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_response_as_json_string(self):
        """SpendLog stores response as a JSON string (not dict) -> parse and return."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(
            return_value=[{
                "request_id": "str-resp",
                "response": json.dumps(SAMPLE_RESPONSE_DICT),
                "metadata": "{}",
            }]
        )
        _setup_mock_proxy_server(mock_prisma)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "str-resp", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            result = await ResponsesSessionHandler.get_response_from_spend_logs(
                ENCODED_RESPONSE_ID
            )

        assert result is not None
        assert result.id == "resp_test123"


# ---------------------------------------------------------------------------
# delete_response_from_spend_logs
# ---------------------------------------------------------------------------


class TestDeleteResponseFromSpendLogs:
    def teardown_method(self):
        _teardown_mock_proxy_server()

    @pytest.mark.asyncio
    async def test_success_with_cold_storage(self):
        """Delete SpendLog + cold storage object when key is present."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(
            return_value=[_make_spend_log_row(
                metadata={"cold_storage_object_key": "logs/test.json"}
            )]
        )
        mock_prisma.db.execute_raw = AsyncMock(return_value=1)
        _setup_mock_proxy_server(mock_prisma)

        mock_cold_storage = AsyncMock(return_value=True)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "test-request-id", "custom_llm_provider": "bedrock", "model_id": None},
        ), patch.object(
            ColdStorageHandler,
            "delete_object_from_cold_storage",
            mock_cold_storage,
        ):
            result = await ResponsesSessionHandler.delete_response_from_spend_logs(
                ENCODED_RESPONSE_ID
            )

        assert result is True
        mock_cold_storage.assert_awaited_once_with(object_key="logs/test.json")
        mock_prisma.db.execute_raw.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_no_cold_storage_key(self):
        """Delete SpendLog only - no cold storage key in metadata."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(
            return_value=[_make_spend_log_row()]
        )
        mock_prisma.db.execute_raw = AsyncMock(return_value=1)
        _setup_mock_proxy_server(mock_prisma)

        mock_cold_storage = AsyncMock(return_value=True)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "test-request-id", "custom_llm_provider": "bedrock", "model_id": None},
        ), patch.object(
            ColdStorageHandler,
            "delete_object_from_cold_storage",
            mock_cold_storage,
        ):
            result = await ResponsesSessionHandler.delete_response_from_spend_logs(
                ENCODED_RESPONSE_ID
            )

        assert result is True
        mock_cold_storage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_not_found(self):
        """No SpendLog row -> return False."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(return_value=[])
        _setup_mock_proxy_server(mock_prisma)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "missing-id", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            result = await ResponsesSessionHandler.delete_response_from_spend_logs(
                ENCODED_RESPONSE_ID
            )

        assert result is False


# ---------------------------------------------------------------------------
# S3Logger._delete_object_from_s3
# ---------------------------------------------------------------------------


try:
    import botocore  # noqa: F401
    HAS_BOTOCORE = True
except ImportError:
    HAS_BOTOCORE = False


@pytest.mark.skipif(not HAS_BOTOCORE, reason="botocore not installed")
class TestS3DeleteObject:
    @pytest.mark.asyncio
    async def test_successful_delete(self):
        """S3 returns 204 -> return True."""
        from litellm.integrations.s3_v2 import S3Logger

        logger = S3Logger.__new__(S3Logger)
        logger.s3_bucket_name = "test-bucket"
        logger.s3_region_name = "us-east-1"
        logger.s3_endpoint_url = None
        logger.s3_aws_access_key_id = "AKID"
        logger.s3_aws_secret_access_key = "secret"
        logger.s3_aws_session_token = None
        logger.s3_aws_session_name = None
        logger.s3_aws_profile_name = None
        logger.s3_aws_role_name = None
        logger.s3_aws_web_identity_token = None
        logger.s3_aws_sts_endpoint = None
        logger.s3_use_virtual_hosted_style = False

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        logger.async_httpx_client = mock_client

        mock_creds = MagicMock()

        with patch.object(
            S3Logger, "get_credentials", return_value=mock_creds
        ), patch(
            "botocore.auth.SigV4Auth"
        ), patch(
            "litellm.integrations.s3_v2.asyncify",
            return_value=AsyncMock(return_value=mock_creds),
        ):
            result = await logger._delete_object_from_s3("logs/test.json")

        assert result is True
        mock_client.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_not_found_returns_false(self):
        """S3 returns 404 -> return False."""
        from litellm.integrations.s3_v2 import S3Logger

        logger = S3Logger.__new__(S3Logger)
        logger.s3_bucket_name = "test-bucket"
        logger.s3_region_name = "us-east-1"
        logger.s3_endpoint_url = None
        logger.s3_aws_access_key_id = "AKID"
        logger.s3_aws_secret_access_key = "secret"
        logger.s3_aws_session_token = None
        logger.s3_aws_session_name = None
        logger.s3_aws_profile_name = None
        logger.s3_aws_role_name = None
        logger.s3_aws_web_identity_token = None
        logger.s3_aws_sts_endpoint = None
        logger.s3_use_virtual_hosted_style = False

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        logger.async_httpx_client = mock_client

        mock_creds = MagicMock()

        with patch.object(
            S3Logger, "get_credentials", return_value=mock_creds
        ), patch(
            "botocore.auth.SigV4Auth"
        ), patch(
            "litellm.integrations.s3_v2.asyncify",
            return_value=AsyncMock(return_value=mock_creds),
        ):
            result = await logger._delete_object_from_s3("logs/missing.json")

        assert result is False


# ---------------------------------------------------------------------------
# ColdStorageHandler.delete_object_from_cold_storage
# ---------------------------------------------------------------------------


class TestColdStorageHandlerDelete:
    @pytest.mark.asyncio
    async def test_delegates_to_custom_logger(self):
        """Delegates to the active custom logger's delete method."""
        mock_logger = AsyncMock()
        mock_logger.delete_object_from_cold_storage = AsyncMock(return_value=True)

        handler = ColdStorageHandler()

        with patch.object(
            handler, "_select_custom_logger_for_cold_storage", return_value="s3"
        ), patch(
            "litellm.proxy.spend_tracking.cold_storage_handler.litellm.logging_callback_manager.get_active_custom_logger_for_callback_name",
            return_value=mock_logger,
        ):
            result = await handler.delete_object_from_cold_storage("key.json")

        assert result is True
        mock_logger.delete_object_from_cold_storage.assert_awaited_once_with(
            object_key="key.json"
        )

    @pytest.mark.asyncio
    async def test_no_cold_storage_configured(self):
        """No cold storage logger -> return False."""
        handler = ColdStorageHandler()

        with patch.object(
            handler, "_select_custom_logger_for_cold_storage", return_value=None
        ):
            result = await handler.delete_object_from_cold_storage("key.json")

        assert result is False


# ---------------------------------------------------------------------------
# Endpoint-level fallback integration tests
#
# These tests verify the ValueError catch + fallback at the proxy endpoint
# layer. They mock the session_handler methods rather than DB access since
# the endpoint tests import the full proxy machinery.
# ---------------------------------------------------------------------------


class TestEndpointFallback:
    """Test the GET/DELETE endpoint fallback logic at the proxy layer.

    These tests call the session handler methods directly (the unit under test),
    rather than the full FastAPI endpoint handler, to avoid importing the heavy
    proxy_server module which requires many optional dependencies.
    """

    @pytest.mark.asyncio
    async def test_get_fallback_returns_stored_response(self):
        """Simulates the GET fallback: ValueError raised -> session handler called -> response returned."""
        _setup_mock_proxy_server(MagicMock())

        stored_response = ResponsesAPIResponse(
            id="resp_test123",
            created_at=1700000000,
            model="bedrock/anthropic.claude-v2",
            output=[],
            status="completed",
        )

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "test-request-id", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            # Simulate what the endpoint does: catch ValueError, call get_response_from_spend_logs
            mock_prisma = MagicMock()
            mock_prisma.db.query_raw = AsyncMock(
                return_value=[_make_spend_log_row()]
            )
            _setup_mock_proxy_server(mock_prisma)

            result = await ResponsesSessionHandler.get_response_from_spend_logs(
                response_id=ENCODED_RESPONSE_ID
            )

        assert result is not None
        assert isinstance(result, ResponsesAPIResponse)
        assert result.id == "resp_test123"

    def teardown_method(self):
        _teardown_mock_proxy_server()

    @pytest.mark.asyncio
    async def test_delete_fallback_returns_true(self):
        """Simulates the DELETE fallback: ValueError raised -> session handler called -> deleted."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(
            return_value=[_make_spend_log_row()]
        )
        mock_prisma.db.execute_raw = AsyncMock(return_value=1)
        _setup_mock_proxy_server(mock_prisma)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "test-request-id", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            result = await ResponsesSessionHandler.delete_response_from_spend_logs(
                response_id=ENCODED_RESPONSE_ID
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_fallback_not_found_returns_none(self):
        """When no SpendLog exists, fallback returns None (endpoint would raise 404)."""
        mock_prisma = MagicMock()
        mock_prisma.db.query_raw = AsyncMock(return_value=[])
        _setup_mock_proxy_server(mock_prisma)

        with patch(
            "litellm.responses.litellm_completion_transformation.session_handler.ResponsesAPIRequestUtils._decode_responses_api_response_id",
            return_value={"response_id": "missing", "custom_llm_provider": "bedrock", "model_id": None},
        ):
            result = await ResponsesSessionHandler.get_response_from_spend_logs(
                response_id=ENCODED_RESPONSE_ID
            )

        assert result is None
