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

    def teardown_method(self):
        """Clean up test environment"""
        # Clean up environment variables after each test
        os.environ.pop("RUBRIK_API_KEY", None)
        os.environ.pop("RUBRIK_WEBHOOK_URL", None)

    def test_rubrik_logger_initialization(self):
        """Test that RubrikLogger initializes correctly with required env vars"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()
        assert logger is not None
        assert logger.webhook_url == "https://test.rubrik.com"
        assert logger.key is None
        assert logger.client is None

    def test_rubrik_logger_initialization_with_api_key(self):
        """Test that RubrikLogger initializes correctly with API key"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"
        os.environ["RUBRIK_API_KEY"] = "test-api-key-123"

        logger = RubrikLogger()
        assert logger is not None
        assert logger.webhook_url == "https://test.rubrik.com"
        assert logger.key == "test-api-key-123"

    def test_rubrik_logger_missing_webhook_url(self):
        """Test that RubrikLogger raises ValueError when webhook URL is missing"""
        with pytest.raises(ValueError, match="environment variable RUBRIK_WEBHOOK_URL not set"):
            RubrikLogger()

    def test_rubrik_logger_removes_trailing_slash(self):
        """Test that RubrikLogger removes trailing slash from webhook URL"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com/"

        logger = RubrikLogger()
        assert logger.webhook_url == "https://test.rubrik.com"

    def test_rubrik_logger_keeps_url_without_trailing_slash(self):
        """Test that RubrikLogger keeps URL without trailing slash unchanged"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()
        assert logger.webhook_url == "https://test.rubrik.com"

    @pytest.mark.asyncio
    async def test_async_log_success_event_basic(self):
        """Test basic async_log_success_event without API key"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        # Mock httpx.AsyncClient
        with patch("httpx.AsyncClient", return_value=mock_client):
            kwargs = {
                "standard_logging_object": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "response": {"choices": [{"message": {"content": "Hi there!"}}]},
                }
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify client was created and cached
            assert logger.client is not None

            # Verify post was called
            mock_client.post.assert_called_once()

            # Verify the call arguments
            call_args = mock_client.post.call_args
            assert call_args[1]["headers"]["Content-Type"] == "application/json"
            assert "Authorization" not in call_args[1]["headers"]
            assert "https://test.rubrik.com/litellm" in call_args[0][0]
            assert call_args[1]["timeout"] == 10.0

            # Verify payload structure
            payload = json.loads(call_args[1]["content"])
            assert payload["model"] == "gpt-4"
            assert payload["messages"][0]["role"] == "user"
            assert payload["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_async_log_success_event_with_api_key(self):
        """Test async_log_success_event with API key authentication"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"
        os.environ["RUBRIK_API_KEY"] = "test-api-key-456"

        logger = RubrikLogger()

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            kwargs = {
                "standard_logging_object": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test message"}],
                    "response": {"choices": [{"message": {"content": "Test response"}}]},
                }
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify post was called with Authorization header
            call_args = mock_client.post.call_args
            assert call_args[1]["headers"]["Content-Type"] == "application/json"
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key-456"

    @pytest.mark.asyncio
    async def test_async_log_success_event_with_system_prompt(self):
        """Test async_log_success_event with Anthropic-style system prompt"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
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

            # Verify post was called
            mock_client.post.assert_called_once()

            # Verify payload includes system prompt at the beginning
            call_args = mock_client.post.call_args
            payload_json = call_args[1]["content"]

            # The payload should contain the system message at the beginning
            assert '"role": "system"' in payload_json
            assert '"You are a helpful assistant."' in payload_json
            assert mock_payload.messages[0]["role"] == "system"
            assert mock_payload.messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_async_log_success_event_with_system_prompt_list(self):
        """Test async_log_success_event with system prompt as list"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
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
    async def test_async_log_success_event_client_reuse(self):
        """Test that httpx client is cached and reused"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client) as mock_async_client:
            kwargs = {
                "standard_logging_object": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Test"}],
                    "response": {"choices": [{"message": {"content": "Response"}}]},
                }
            }

            # Call twice
            await logger.async_log_success_event(kwargs, None, None, None)
            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify AsyncClient was only instantiated once
            assert mock_async_client.call_count == 1
            # Verify post was called twice with the same client
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_async_log_success_event_error_handling(self):
        """Test error handling in async_log_success_event"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client to raise an exception
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Network error"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            kwargs = {
                "standard_logging_object": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Test"}],
                    "response": {"choices": [{"message": {"content": "Response"}}]},
                }
            }

            # Should not raise an exception - errors are caught and logged
            await logger.async_log_success_event(kwargs, None, None, None)

    @pytest.mark.asyncio
    async def test_async_log_success_event_http_error(self):
        """Test handling of HTTP error responses"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client with an HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("HTTP 500 Error")
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            kwargs = {
                "standard_logging_object": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Test"}],
                    "response": {"choices": [{"message": {"content": "Response"}}]},
                }
            }

            # Should not raise an exception - errors are caught and logged
            await logger.async_log_success_event(kwargs, None, None, None)

    @pytest.mark.asyncio
    async def test_async_log_success_event_webhook_endpoint_format(self):
        """Test that webhook endpoint is correctly formatted"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            kwargs = {
                "standard_logging_object": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Test"}],
                    "response": {"choices": [{"message": {"content": "Response"}}]},
                }
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify the webhook endpoint format
            call_args = mock_client.post.call_args
            called_url = call_args[0][0]
            assert called_url == "https://test.rubrik.com/litellm"

    @pytest.mark.asyncio
    async def test_async_log_success_event_without_system_prompt(self):
        """Test that messages are not modified when no system prompt exists"""
        os.environ["RUBRIK_WEBHOOK_URL"] = "https://test.rubrik.com"

        logger = RubrikLogger()

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            original_messages = [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"},
            ]
            kwargs = {
                "standard_logging_object": {
                    "model": "gpt-4",
                    "messages": original_messages.copy(),
                    "response": {"choices": [{"message": {"content": "Response"}}]},
                }
            }

            await logger.async_log_success_event(kwargs, None, None, None)

            # Verify messages were not modified
            call_args = mock_client.post.call_args
            payload = json.loads(call_args[1]["content"])
            assert payload["messages"] == original_messages
