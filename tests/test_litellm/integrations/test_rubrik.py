"""Unit tests for Rubrik integration logger"""
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.rubrik import RubrikLogger
from litellm.types.utils import StandardLoggingPayload


class MockStandardLoggingPayload(dict):
    """Mock object that mimics StandardLoggingPayload for testing
    Inherits from dict to be JSON serializable while allowing attribute access
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        self[key] = value
        super().__setattr__(key, value)


class TestRubrikIntegration:
    """Test suite for Rubrik integration"""

    def setup_method(self):
        """Set up test environment"""
        # Clear environment variables before each test
        os.environ.pop("RUBRIK_API_KEY", None)
        os.environ.pop("RUBRIK_WEBHOOK_URL", None)
        os.environ.pop("RUBRIK_SAMPLING_RATE", None)
        os.environ.pop("RUBRIK_BATCH_SIZE", None)

    def teardown_method(self):
        """Clean up test environment"""
        # Clean up environment variables after each test
        os.environ.pop("RUBRIK_API_KEY", None)
        os.environ.pop("RUBRIK_WEBHOOK_URL", None)
        os.environ.pop("RUBRIK_SAMPLING_RATE", None)
        os.environ.pop("RUBRIK_BATCH_SIZE", None)

    @pytest.mark.asyncio
    async def test_rubrik_logger_initialization(self):
        """Test that RubrikLogger initializes correctly with required env vars"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            assert logger is not None
            assert logger.webhook_endpoint == "https://test.rubrik.com/litellm/batch"
            assert logger.key is None
            assert logger.sampling_rate == 1.0
            assert logger.log_queue == []

    @pytest.mark.asyncio
    async def test_rubrik_logger_initialization_with_api_key(self):
        """Test that RubrikLogger initializes correctly with API key"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"
        os.environ["RUBRIK_API_KEY"] = "test-api-key-123"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            assert logger is not None
            assert logger.webhook_endpoint == "https://test.rubrik.com/litellm/batch"
            assert logger.key == "test-api-key-123"

    @pytest.mark.asyncio
    async def test_rubrik_logger_missing_webhook_url(self):
        """Test that RubrikLogger raises ValueError when webhook URL is missing"""
        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            with pytest.raises(ValueError, match="environment variable RUBRIK_WEBHOOK_URL not set"):
                RubrikLogger()

    @pytest.mark.asyncio
    async def test_rubrik_logger_removes_trailing_slash(self):
        """Test that RubrikLogger removes trailing slash from webhook URL"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com/"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            assert logger.webhook_endpoint == "https://test.rubrik.com/litellm/batch"

    @pytest.mark.asyncio
    async def test_rubrik_logger_keeps_url_without_trailing_slash(self):
        """Test that RubrikLogger keeps URL without trailing slash unchanged"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            assert logger.webhook_endpoint == "https://test.rubrik.com/litellm/batch"

    @pytest.mark.asyncio
    async def test_rubrik_logger_custom_batch_size(self):
        """Test that RubrikLogger respects custom batch size"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"
        os.environ["RUBRIK_BATCH_SIZE"] = "100"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            assert logger.batch_size == 100

    @pytest.mark.asyncio
    async def test_rubrik_logger_sampling_rate(self):
        """Test that RubrikLogger respects sampling rate"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"
        # Note: sampling rate parsing requires digit string
        os.environ["RUBRIK_SAMPLING_RATE"] = "1"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            assert logger.sampling_rate == 1.0

    @pytest.mark.asyncio
    async def test_async_log_success_event_adds_to_queue(self):
        """Test that async_log_success_event adds payload to queue"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            logger.batch_size = 100  # Set high batch size to prevent auto-flush

            mock_payload = MockStandardLoggingPayload(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                response={"choices": [{"message": {"content": "Hi there!"}}]},
            )

            kwargs = {
                "standard_logging_object": mock_payload,
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify payload was added to queue
            assert len(logger.log_queue) == 1
            assert logger.log_queue[0] == mock_payload

    @pytest.mark.asyncio
    async def test_async_log_success_event_flushes_when_batch_size_reached(self):
        """Test that async_log_success_event flushes queue when batch size is reached"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            logger = RubrikLogger()
            logger.batch_size = 2  # Small batch size for testing

            mock_payload = MockStandardLoggingPayload(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

            kwargs = {
                "standard_logging_object": mock_payload,
            }

            # Add first item - should not flush
            await logger.async_log_success_event(kwargs, None, None, None)
            assert len(logger.log_queue) == 1

            # Add second item - should trigger flush
            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify post was called (batch was sent)
            mock_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_async_send_batch_with_api_key(self):
        """Test async_send_batch with API key authentication"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"
        os.environ["RUBRIK_API_KEY"] = "test-api-key-456"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            logger = RubrikLogger()

            mock_payload = MockStandardLoggingPayload(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
            )

            logger.log_queue.append(mock_payload)
            await logger.async_send_batch()

            # Verify post was called with Authorization header
            call_args = mock_client.post.call_args
            assert call_args.kwargs["headers"]["Content-Type"] == "application/json"
            assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-api-key-456"

    @pytest.mark.asyncio
    async def test_async_log_success_event_with_system_prompt(self):
        """Test async_log_success_event with Anthropic-style system prompt"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            logger.batch_size = 100  # Prevent auto-flush

            # Create a mock object that behaves like StandardLoggingPayload
            mock_payload = MockStandardLoggingPayload(
                model="claude-3-opus",
                messages=[{"role": "user", "content": "Hello"}],
                response={"choices": [{"message": {"content": "Hi!"}}]},
            )

            kwargs = {
                "standard_logging_object": mock_payload,
                "system": "You are a helpful assistant.",  # Anthropic system prompt
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify payload includes system prompt at the beginning
            assert mock_payload.messages[0]["role"] == "system"
            assert mock_payload.messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_async_log_success_event_with_system_prompt_list(self):
        """Test async_log_success_event with system prompt as list"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            logger.batch_size = 100  # Prevent auto-flush

            system_prompt_list = [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
            mock_payload = MockStandardLoggingPayload(
                model="claude-3-opus",
                messages=[{"role": "user", "content": "Hello"}],
                response={"choices": [{"message": {"content": "Hi!"}}]},
            )

            kwargs = {
                "standard_logging_object": mock_payload,
                "system": system_prompt_list,
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify payload includes system prompt at the beginning
            assert mock_payload.messages[0]["role"] == "system"
            assert mock_payload.messages[0]["content"] == system_prompt_list

    @pytest.mark.asyncio
    async def test_async_send_batch_empty_queue(self):
        """Test that async_send_batch does nothing with empty queue"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client

            logger = RubrikLogger()
            logger.log_queue = []

            await logger.async_send_batch()

            # Verify post was NOT called
            mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_batch_to_rubrik_error_handling(self):
        """Test error handling in _log_batch_to_rubrik"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Network error"))
            mock_get_client.return_value = mock_client

            logger = RubrikLogger()

            # Should not raise an exception - errors are caught and logged
            await logger._log_batch_to_rubrik([{"test": "data"}])

    @pytest.mark.asyncio
    async def test_log_batch_to_rubrik_http_error(self):
        """Test handling of HTTP error responses"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        import httpx

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_request = MagicMock()
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError("HTTP 500 Error", request=mock_request, response=mock_response)
            )
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            logger = RubrikLogger()

            # Should not raise an exception - errors are caught and logged
            await logger._log_batch_to_rubrik([{"test": "data"}])

    @pytest.mark.asyncio
    async def test_async_send_batch_webhook_endpoint_format(self):
        """Test that webhook endpoint is correctly formatted"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            logger = RubrikLogger()
            logger.log_queue = [{"test": "data"}]

            await logger.async_send_batch()

            # Verify the webhook endpoint format
            call_args = mock_client.post.call_args
            called_url = call_args.kwargs["url"]
            assert called_url == "https://test.rubrik.com/litellm/batch"

    @pytest.mark.asyncio
    async def test_async_send_batch_payload_format(self):
        """Test that payload is wrapped in data field"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            logger = RubrikLogger()
            test_data = [{"model": "gpt-4", "messages": []}]
            logger.log_queue = test_data

            await logger.async_send_batch()

            # Verify the payload format
            call_args = mock_client.post.call_args
            payload = call_args.kwargs["json"]
            assert "data" in payload
            assert payload["data"] == test_data

    @pytest.mark.asyncio
    async def test_async_log_success_event_without_system_prompt(self):
        """Test that messages are not modified when no system prompt exists"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            logger.batch_size = 100  # Prevent auto-flush

            original_messages = [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"},
            ]
            mock_payload = MockStandardLoggingPayload(
                model="gpt-4",
                messages=original_messages.copy(),
                response={"choices": [{"message": {"content": "Response"}}]},
            )

            kwargs = {
                "standard_logging_object": mock_payload,
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify messages were not modified
            assert mock_payload.messages == original_messages

    @pytest.mark.asyncio
    async def test_async_log_success_event_sampling_skips_when_rate_exceeded(self):
        """Test that logging is skipped based on sampling rate"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            logger.sampling_rate = 0.0  # Set to 0 to always skip

            mock_payload = MockStandardLoggingPayload(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

            kwargs = {
                "standard_logging_object": mock_payload,
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify nothing was added to queue
            assert len(logger.log_queue) == 0

    @pytest.mark.asyncio
    async def test_send_batch_sync(self):
        """Test _send_batch synchronous wrapper"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        with patch("litellm.integrations.rubrik.get_async_httpx_client") as mock_get_client:
            mock_get_client.return_value = MagicMock()
            logger = RubrikLogger()
            logger.log_queue = []

            # Should not raise an error with empty queue
            logger._send_batch()
