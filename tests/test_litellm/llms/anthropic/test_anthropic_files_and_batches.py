"""
Test Anthropic Files Handler and Batch Retrieval

Tests for:
1. AnthropicFilesHandler.afile_content() - retrieving batch results
2. AnthropicBatchesConfig.transform_retrieve_batch_response() - transforming batch responses
3. Transformation of Anthropic batch results to OpenAI format
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../../"))

import httpx
import pytest

from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig
from litellm.llms.anthropic.files.handler import AnthropicFilesHandler
from litellm.types.llms.openai import FileContentRequest, HttpxBinaryResponseContent


class TestAnthropicFilesHandler:
    """Test Anthropic Files Handler for batch results retrieval"""

    @pytest.fixture
    def handler(self):
        """Create AnthropicFilesHandler instance"""
        return AnthropicFilesHandler()

    @pytest.fixture
    def mock_anthropic_batch_results_succeeded(self):
        """Mock Anthropic batch results with succeeded status"""
        return json.dumps({
            "custom_id": "test-request-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_123",
                    "model": "claude-3-5-sonnet-20241022",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello, world!"
                        }
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5
                    }
                }
            }
        }).encode("utf-8")

    @pytest.fixture
    def mock_anthropic_batch_results_errored(self):
        """Mock Anthropic batch results with errored status"""
        return json.dumps({
            "custom_id": "test-request-2",
            "result": {
                "type": "errored",
                "error": {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Invalid request"
                    },
                    "request_id": "req_456"
                }
            }
        }).encode("utf-8")

    @pytest.fixture
    def mock_anthropic_batch_results_canceled(self):
        """Mock Anthropic batch results with canceled status"""
        return json.dumps({
            "custom_id": "test-request-3",
            "result": {
                "type": "canceled"
            }
        }).encode("utf-8")

    @pytest.fixture
    def mock_anthropic_batch_results_mixed(self):
        """Mock Anthropic batch results with multiple result types"""
        lines = [
            json.dumps({
                "custom_id": "test-request-1",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_123",
                        "model": "claude-3-5-sonnet-20241022",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Success"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5}
                    }
                }
            }),
            json.dumps({
                "custom_id": "test-request-2",
                "result": {
                    "type": "errored",
                    "error": {
                        "error": {
                            "type": "rate_limit_error",
                            "message": "Rate limit exceeded"
                        },
                        "request_id": "req_456"
                    }
                }
            }),
            json.dumps({
                "custom_id": "test-request-3",
                "result": {
                    "type": "expired"
                }
            })
        ]
        return "\n".join(lines).encode("utf-8")

    @pytest.mark.asyncio
    async def test_afile_content_success(self, handler, mock_anthropic_batch_results_succeeded):
        """Test successful file content retrieval and transformation"""
        file_content_request: FileContentRequest = {
            "file_id": "batch_123",
            "extra_headers": None,
            "extra_body": None
        }

        # Mock the httpx client
        mock_response = httpx.Response(
            status_code=200,
            content=mock_anthropic_batch_results_succeeded,
            headers={"content-type": "application/json"},
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123/results")
        )

        with patch("litellm.llms.anthropic.files.handler.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(handler.anthropic_model_info, "get_api_key", return_value="test-api-key"):
                with patch.object(handler.anthropic_model_info, "get_api_base", return_value="https://api.anthropic.com"):
                    result = await handler.afile_content(
                        file_content_request=file_content_request,
                        api_key="test-api-key"
                    )

                    # Verify result
                    assert isinstance(result, HttpxBinaryResponseContent)
                    assert result.response.status_code == 200

                    # Verify transformation to OpenAI format
                    content = result.response.content.decode("utf-8")
                    lines = [line for line in content.strip().split("\n") if line.strip()]
                    assert len(lines) == 1

                    transformed_result = json.loads(lines[0])
                    assert transformed_result["custom_id"] == "test-request-1"
                    assert transformed_result["response"]["status_code"] == 200
                    assert "body" in transformed_result["response"]
                    # Verify body has required OpenAI format fields
                    assert "id" in transformed_result["response"]["body"]
                    assert transformed_result["response"]["body"]["object"] == "chat.completion"
                    assert "choices" in transformed_result["response"]["body"]
                    # Verify request_id matches the original message id
                    assert transformed_result["response"]["request_id"] == "msg_123"

    @pytest.mark.asyncio
    async def test_afile_content_with_prefix(self, handler, mock_anthropic_batch_results_succeeded):
        """Test file content retrieval with anthropic_batch_results: prefix"""
        file_content_request: FileContentRequest = {
            "file_id": "anthropic_batch_results:batch_123",
            "extra_headers": None,
            "extra_body": None
        }

        mock_response = httpx.Response(
            status_code=200,
            content=mock_anthropic_batch_results_succeeded,
            headers={"content-type": "application/json"},
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123/results")
        )

        with patch("litellm.llms.anthropic.files.handler.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(handler.anthropic_model_info, "get_api_key", return_value="test-api-key"):
                with patch.object(handler.anthropic_model_info, "get_api_base", return_value="https://api.anthropic.com"):
                    result = await handler.afile_content(
                        file_content_request=file_content_request,
                        api_key="test-api-key"
                    )

                    assert isinstance(result, HttpxBinaryResponseContent)
                    # Verify the URL was constructed correctly (batch_id extracted from prefix)
                    mock_client.get.assert_called_once()
                    call_url = mock_client.get.call_args[1]["url"]
                    assert "batch_123" in call_url

    @pytest.mark.asyncio
    async def test_afile_content_errored_result(self, handler, mock_anthropic_batch_results_errored):
        """Test transformation of errored batch results"""
        file_content_request: FileContentRequest = {
            "file_id": "batch_123",
            "extra_headers": None,
            "extra_body": None
        }

        mock_response = httpx.Response(
            status_code=200,
            content=mock_anthropic_batch_results_errored,
            headers={"content-type": "application/json"},
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123/results")
        )

        with patch("litellm.llms.anthropic.files.handler.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(handler.anthropic_model_info, "get_api_key", return_value="test-api-key"):
                with patch.object(handler.anthropic_model_info, "get_api_base", return_value="https://api.anthropic.com"):
                    result = await handler.afile_content(
                        file_content_request=file_content_request,
                        api_key="test-api-key"
                    )

                    content = result.response.content.decode("utf-8")
                    lines = [line for line in content.strip().split("\n") if line.strip()]
                    assert len(lines) == 1

                    transformed_result = json.loads(lines[0])
                    assert transformed_result["custom_id"] == "test-request-2"
                    assert transformed_result["response"]["status_code"] == 400  # invalid_request_error maps to 400
                    assert transformed_result["response"]["body"]["error"]["type"] == "invalid_request_error"
                    assert transformed_result["response"]["body"]["error"]["message"] == "Invalid request"

    @pytest.mark.asyncio
    async def test_afile_content_canceled_result(self, handler, mock_anthropic_batch_results_canceled):
        """Test transformation of canceled batch results"""
        file_content_request: FileContentRequest = {
            "file_id": "batch_123",
            "extra_headers": None,
            "extra_body": None
        }

        mock_response = httpx.Response(
            status_code=200,
            content=mock_anthropic_batch_results_canceled,
            headers={"content-type": "application/json"},
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123/results")
        )

        with patch("litellm.llms.anthropic.files.handler.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(handler.anthropic_model_info, "get_api_key", return_value="test-api-key"):
                with patch.object(handler.anthropic_model_info, "get_api_base", return_value="https://api.anthropic.com"):
                    result = await handler.afile_content(
                        file_content_request=file_content_request,
                        api_key="test-api-key"
                    )

                    content = result.response.content.decode("utf-8")
                    lines = [line for line in content.strip().split("\n") if line.strip()]
                    assert len(lines) == 1

                    transformed_result = json.loads(lines[0])
                    assert transformed_result["custom_id"] == "test-request-3"
                    assert transformed_result["response"]["status_code"] == 400
                    assert "Batch request was canceled" in transformed_result["response"]["body"]["error"]["message"]

    @pytest.mark.asyncio
    async def test_afile_content_mixed_results(self, handler, mock_anthropic_batch_results_mixed):
        """Test transformation of mixed batch results (succeeded, errored, expired)"""
        file_content_request: FileContentRequest = {
            "file_id": "batch_123",
            "extra_headers": None,
            "extra_body": None
        }

        mock_response = httpx.Response(
            status_code=200,
            content=mock_anthropic_batch_results_mixed,
            headers={"content-type": "application/json"},
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123/results")
        )

        with patch("litellm.llms.anthropic.files.handler.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(handler.anthropic_model_info, "get_api_key", return_value="test-api-key"):
                with patch.object(handler.anthropic_model_info, "get_api_base", return_value="https://api.anthropic.com"):
                    result = await handler.afile_content(
                        file_content_request=file_content_request,
                        api_key="test-api-key"
                    )

                    content = result.response.content.decode("utf-8")
                    lines = [line for line in content.strip().split("\n") if line.strip()]
                    assert len(lines) == 3

                    # Check first result (succeeded)
                    result1 = json.loads(lines[0])
                    assert result1["response"]["status_code"] == 200

                    # Check second result (errored)
                    result2 = json.loads(lines[1])
                    assert result2["response"]["status_code"] == 429  # rate_limit_error maps to 429

                    # Check third result (expired)
                    result3 = json.loads(lines[2])
                    assert result3["response"]["status_code"] == 400
                    assert "expired" in result3["response"]["body"]["error"]["message"]

    @pytest.mark.asyncio
    async def test_afile_content_missing_api_key(self, handler):
        """Test file content retrieval with missing API key"""
        file_content_request: FileContentRequest = {
            "file_id": "batch_123",
            "extra_headers": None,
            "extra_body": None
        }

        with patch.object(handler.anthropic_model_info, "get_api_key", return_value=None):
            with pytest.raises(ValueError, match="Missing Anthropic API Key"):
                await handler.afile_content(
                    file_content_request=file_content_request,
                    api_key=None
                )

    @pytest.mark.asyncio
    async def test_afile_content_missing_file_id(self, handler):
        """Test file content retrieval with missing file_id"""
        file_content_request: FileContentRequest = {
            "file_id": None,
            "extra_headers": None,
            "extra_body": None
        }

        with pytest.raises(ValueError, match="file_id is required"):
            await handler.afile_content(
                file_content_request=file_content_request,
                api_key="test-api-key"
            )

    @pytest.mark.asyncio
    async def test_afile_content_http_error(self, handler):
        """Test file content retrieval with HTTP error"""
        file_content_request: FileContentRequest = {
            "file_id": "batch_123",
            "extra_headers": None,
            "extra_body": None
        }

        mock_response = httpx.Response(
            status_code=404,
            content=b"Not Found",
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123/results")
        )
        mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Not Found", request=mock_response.request, response=mock_response))

        with patch("litellm.llms.anthropic.files.handler.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with patch.object(handler.anthropic_model_info, "get_api_key", return_value="test-api-key"):
                with patch.object(handler.anthropic_model_info, "get_api_base", return_value="https://api.anthropic.com"):
                    with pytest.raises(httpx.HTTPStatusError):
                        await handler.afile_content(
                            file_content_request=file_content_request,
                            api_key="test-api-key"
                        )


class TestAnthropicBatchesConfig:
    """Test Anthropic Batches Config for batch retrieval transformation"""

    @pytest.fixture
    def config(self):
        """Create AnthropicBatchesConfig instance"""
        return AnthropicBatchesConfig()

    @pytest.fixture
    def mock_anthropic_batch_response_in_progress(self):
        """Mock Anthropic batch response with in_progress status"""
        return {
            "id": "batch_123",
            "processing_status": "in_progress",
            "created_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-02T00:00:00Z",
            "request_counts": {
                "processing": 5,
                "succeeded": 3,
                "errored": 1,
                "canceled": 0,
                "expired": 0
            }
        }

    @pytest.fixture
    def mock_anthropic_batch_response_completed(self):
        """Mock Anthropic batch response with completed status"""
        return {
            "id": "batch_456",
            "processing_status": "ended",
            "created_at": "2024-01-01T00:00:00Z",
            "ended_at": "2024-01-01T12:00:00Z",
            "expires_at": "2024-01-02T00:00:00Z",
            "request_counts": {
                "processing": 0,
                "succeeded": 10,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        }

    @pytest.fixture
    def mock_anthropic_batch_response_canceling(self):
        """Mock Anthropic batch response with canceling status"""
        return {
            "id": "batch_789",
            "processing_status": "canceling",
            "created_at": "2024-01-01T00:00:00Z",
            "cancel_initiated_at": "2024-01-01T06:00:00Z",
            "ended_at": "2024-01-01T07:00:00Z",
            "expires_at": "2024-01-02T00:00:00Z",
            "request_counts": {
                "processing": 0,
                "succeeded": 5,
                "errored": 0,
                "canceled": 3,
                "expired": 0
            }
        }

    def test_get_retrieve_batch_url(self, config):
        """Test URL construction for batch retrieval"""
        url = config.get_retrieve_batch_url(
            api_base="https://api.anthropic.com",
            batch_id="batch_123",
            optional_params={},
            litellm_params={}
        )
        assert url == "https://api.anthropic.com/v1/messages/batches/batch_123"

        # Test with trailing slash
        url = config.get_retrieve_batch_url(
            api_base="https://api.anthropic.com/",
            batch_id="batch_123",
            optional_params={},
            litellm_params={}
        )
        assert url == "https://api.anthropic.com/v1/messages/batches/batch_123"

    def test_transform_retrieve_batch_response_in_progress(self, config, mock_anthropic_batch_response_in_progress):
        """Test transformation of in_progress batch response"""
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(mock_anthropic_batch_response_in_progress).encode("utf-8"),
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123")
        )

        logging_obj = MagicMock()
        batch = config.transform_retrieve_batch_response(
            model="claude-3-5-sonnet-20241022",
            raw_response=mock_response,
            logging_obj=logging_obj,
            litellm_params={}
        )

        assert batch.id == "batch_123"
        assert batch.object == "batch"
        assert batch.status == "in_progress"
        assert batch.endpoint == "/v1/messages"
        assert batch.output_file_id == "batch_123"
        assert batch.request_counts.total == 9  # 5 + 3 + 1
        assert batch.request_counts.completed == 3
        assert batch.request_counts.failed == 1
        assert batch.in_progress_at is not None
        assert batch.completed_at is None

    def test_transform_retrieve_batch_response_completed(self, config, mock_anthropic_batch_response_completed):
        """Test transformation of completed batch response"""
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(mock_anthropic_batch_response_completed).encode("utf-8"),
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_456")
        )

        logging_obj = MagicMock()
        batch = config.transform_retrieve_batch_response(
            model="claude-3-5-sonnet-20241022",
            raw_response=mock_response,
            logging_obj=logging_obj,
            litellm_params={}
        )

        assert batch.id == "batch_456"
        assert batch.status == "completed"
        assert batch.completed_at is not None
        assert batch.request_counts.total == 10
        assert batch.request_counts.completed == 10
        assert batch.request_counts.failed == 0

    def test_transform_retrieve_batch_response_canceling(self, config, mock_anthropic_batch_response_canceling):
        """Test transformation of canceling batch response"""
        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(mock_anthropic_batch_response_canceling).encode("utf-8"),
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_789")
        )

        logging_obj = MagicMock()
        batch = config.transform_retrieve_batch_response(
            model="claude-3-5-sonnet-20241022",
            raw_response=mock_response,
            logging_obj=logging_obj,
            litellm_params={}
        )

        assert batch.id == "batch_789"
        assert batch.status == "cancelling"
        assert batch.cancelling_at is not None
        assert batch.cancelled_at is not None
        assert batch.request_counts.total == 8  # 5 + 3

    def test_transform_retrieve_batch_response_invalid_json(self, config):
        """Test transformation with invalid JSON response"""
        mock_response = httpx.Response(
            status_code=200,
            content=b"invalid json",
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123")
        )

        logging_obj = MagicMock()
        with pytest.raises(ValueError, match="Failed to parse Anthropic batch response"):
            config.transform_retrieve_batch_response(
                model="claude-3-5-sonnet-20241022",
                raw_response=mock_response,
                logging_obj=logging_obj,
                litellm_params={}
            )

    def test_transform_retrieve_batch_response_timestamp_parsing(self, config):
        """Test timestamp parsing in batch response"""
        batch_data = {
            "id": "batch_123",
            "processing_status": "ended",
            "created_at": "2024-01-01T12:00:00Z",
            "ended_at": "2024-01-01T13:30:45Z",
            "expires_at": "2024-01-02T12:00:00Z",
            "archived_at": "2024-01-03T00:00:00Z",
            "request_counts": {
                "processing": 0,
                "succeeded": 1,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        }

        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(batch_data).encode("utf-8"),
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123")
        )

        logging_obj = MagicMock()
        batch = config.transform_retrieve_batch_response(
            model="claude-3-5-sonnet-20241022",
            raw_response=mock_response,
            logging_obj=logging_obj,
            litellm_params={}
        )

        # Verify timestamps are parsed correctly
        assert batch.created_at is not None
        assert batch.completed_at is not None
        assert batch.expires_at is not None
        assert batch.expired_at is not None

        # Verify timestamps are integers (Unix timestamps)
        assert isinstance(batch.created_at, int)
        assert isinstance(batch.completed_at, int)
        assert isinstance(batch.expires_at, int)
        assert isinstance(batch.expired_at, int)

    def test_transform_retrieve_batch_response_missing_fields(self, config):
        """Test transformation with missing optional fields"""
        batch_data = {
            "id": "batch_123",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 1,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            }
        }

        mock_response = httpx.Response(
            status_code=200,
            content=json.dumps(batch_data).encode("utf-8"),
            request=httpx.Request(method="GET", url="https://api.anthropic.com/v1/messages/batches/batch_123")
        )

        logging_obj = MagicMock()
        batch = config.transform_retrieve_batch_response(
            model="claude-3-5-sonnet-20241022",
            raw_response=mock_response,
            logging_obj=logging_obj,
            litellm_params={}
        )

        # Should still work with missing optional fields
        assert batch.id == "batch_123"
        assert batch.status == "in_progress"
        assert batch.created_at is not None  # Should default to current time if missing
        assert batch.expires_at is None
        assert batch.completed_at is None

