import os
import sys
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
    ContextCachingEndpoints,
)


class TestContextCachingEndpoints:
    """Test class for ContextCachingEndpoints methods"""

    def setup_method(self):
        """Setup for each test method"""
        self.context_caching = ContextCachingEndpoints()
        self.mock_logging = MagicMock(spec=Logging)
        self.mock_client = MagicMock(spec=HTTPHandler)
        self.mock_async_client = MagicMock(spec=AsyncHTTPHandler)

        # Sample messages for testing
        self.sample_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant",
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Sample tools for testing
        self.sample_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        self.sample_optional_params = {"tools": self.sample_tools.copy()}

    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    def test_check_and_create_cache_with_cached_content(
        self, mock_cache_obj, mock_separate
    ):
        """Test check_and_create_cache when cached_content is provided"""
        # Setup
        cached_content = "cached_content_123"
        optional_params = self.sample_optional_params.copy()

        # Execute
        result = self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            cached_content=cached_content,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache == cached_content

        # Verify mocks weren't called since we short-circuited
        mock_separate.assert_not_called()
        mock_cache_obj.get_cache_key.assert_not_called()

    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    def test_check_and_create_cache_no_cached_messages(self, mock_separate):
        """Test check_and_create_cache when no cached messages are found"""
        # Setup
        mock_separate.return_value = ([], self.sample_messages)  # No cached messages
        optional_params = self.sample_optional_params.copy()

        # Execute
        result = self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache is None

    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    def test_check_and_create_cache_existing_cache_found(
        self, mock_check_cache, mock_cache_obj, mock_separate
    ):
        """Test check_and_create_cache when existing cache is found"""
        # Setup
        cached_messages = [self.sample_messages[0]]  # System message with cache_control
        non_cached_messages = [self.sample_messages[1]]  # User message
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = "existing_cache_name"

        optional_params = self.sample_optional_params.copy()

        # Execute
        result = self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == non_cached_messages
        assert returned_params == optional_params
        assert returned_cache == "existing_cache_name"

        # Verify cache key was generated with tools
        mock_cache_obj.get_cache_key.assert_called_once_with(
            messages=cached_messages, tools=self.sample_tools
        )

    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.transform_openai_messages_to_gemini_context_caching"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    def test_check_and_create_cache_create_new_cache(
        self,
        mock_get_token_url,
        mock_check_cache,
        mock_transform,
        mock_cache_obj,
        mock_separate,
    ):
        """Test check_and_create_cache when creating new cache"""
        # Setup
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None  # No existing cache
        mock_get_token_url.return_value = ("token", "https://test-url.com")

        mock_transform.return_value = {"model": "gemini-1.5-pro", "contents": []}

        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "new_cache_name",
            "model": "gemini-1.5-pro",
        }
        self.mock_client.post.return_value = mock_response

        optional_params = self.sample_optional_params.copy()

        # Execute
        result = self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == non_cached_messages
        assert returned_params == optional_params
        assert returned_cache == "new_cache_name"

        # Verify HTTP request was made
        self.mock_client.post.assert_called_once()
        call_args = self.mock_client.post.call_args
        assert "tools" in call_args.kwargs["json"]
        assert call_args.kwargs["json"]["tools"] == self.sample_tools

    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    def test_check_and_create_cache_http_error(
        self, mock_get_token_url, mock_check_cache, mock_cache_obj, mock_separate
    ):
        """Test check_and_create_cache handles HTTP errors properly"""
        # Setup
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None
        mock_get_token_url.return_value = ("token", "https://test-url.com")

        # Mock HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        http_error = httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=mock_response
        )
        self.mock_client.post.side_effect = http_error

        optional_params = self.sample_optional_params.copy()

        # Execute and Assert
        with pytest.raises(VertexAIError) as exc_info:
            self.context_caching.check_and_create_cache(
                messages=self.sample_messages,
                optional_params=optional_params,
                api_key="test_key",
                api_base=None,
                model="gemini-1.5-pro",
                client=self.mock_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
            )

        assert exc_info.value.status_code == 400
        assert "Bad Request" in str(exc_info.value.message)

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    async def test_async_check_and_create_cache_with_cached_content(
        self, mock_cache_obj, mock_separate
    ):
        """Test async_check_and_create_cache when cached_content is provided"""
        # Setup
        cached_content = "cached_content_123"
        optional_params = self.sample_optional_params.copy()

        # Execute
        result = await self.context_caching.async_check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_async_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            cached_content=cached_content,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache == cached_content

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    async def test_async_check_and_create_cache_no_cached_messages(self, mock_separate):
        """Test async_check_and_create_cache when no cached messages are found"""
        # Setup
        mock_separate.return_value = ([], self.sample_messages)
        optional_params = self.sample_optional_params.copy()

        # Execute
        result = await self.context_caching.async_check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_async_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache is None

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "async_check_cache")
    async def test_async_check_and_create_cache_existing_cache_found(
        self, mock_async_check_cache, mock_cache_obj, mock_separate
    ):
        """Test async_check_and_create_cache when existing cache is found"""
        # Setup
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_async_check_cache.return_value = "existing_cache_name"

        optional_params = self.sample_optional_params.copy()

        # Execute
        result = await self.context_caching.async_check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_async_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == non_cached_messages
        assert returned_params == optional_params
        assert returned_cache == "existing_cache_name"

        # Verify cache key was generated with tools
        mock_cache_obj.get_cache_key.assert_called_once_with(
            messages=cached_messages, tools=self.sample_tools
        )

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.transform_openai_messages_to_gemini_context_caching"
    )
    @patch.object(ContextCachingEndpoints, "async_check_cache")
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.get_async_httpx_client"
    )
    async def test_async_check_and_create_cache_create_new_cache(
        self,
        mock_get_client,
        mock_get_token_url,
        mock_async_check_cache,
        mock_transform,
        mock_cache_obj,
        mock_separate,
    ):
        """Test async_check_and_create_cache when creating new cache"""
        # Setup
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_async_check_cache.return_value = None
        mock_get_token_url.return_value = ("token", "https://test-url.com")

        mock_transform.return_value = {"model": "gemini-1.5-pro", "contents": []}

        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "new_cache_name",
            "model": "gemini-1.5-pro",
        }
        self.mock_async_client.post = AsyncMock(return_value=mock_response)

        optional_params = self.sample_optional_params.copy()

        # Execute
        result = await self.context_caching.async_check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_async_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == non_cached_messages
        assert returned_params == optional_params
        assert returned_cache == "new_cache_name"

        # Verify HTTP request was made
        self.mock_async_client.post.assert_called_once()
        call_args = self.mock_async_client.post.call_args
        assert "tools" in call_args.kwargs["json"]
        assert call_args.kwargs["json"]["tools"] == self.sample_tools

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "async_check_cache")
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.get_async_httpx_client"
    )
    async def test_async_check_and_create_cache_timeout_error(
        self,
        mock_get_client,
        mock_get_token_url,
        mock_async_check_cache,
        mock_cache_obj,
        mock_separate,
    ):
        """Test async_check_and_create_cache handles timeout errors properly"""
        # Setup
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_async_check_cache.return_value = None
        mock_get_token_url.return_value = ("token", "https://test-url.com")

        # Mock timeout error
        self.mock_async_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        optional_params = self.sample_optional_params.copy()

        # Execute and Assert
        with pytest.raises(VertexAIError) as exc_info:
            await self.context_caching.async_check_and_create_cache(
                messages=self.sample_messages,
                optional_params=optional_params,
                api_key="test_key",
                api_base=None,
                model="gemini-1.5-pro",
                client=self.mock_async_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
            )

        assert exc_info.value.status_code == 408
        assert "Timeout error occurred" in str(exc_info.value.message)

    def test_check_and_create_cache_tools_popped_from_optional_params(self):
        """Test that tools are properly popped from optional_params when there are cached messages"""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            # Mock to return cached messages so tools get popped
            cached_messages = [
                self.sample_messages[0]
            ]  # System message with cache_control
            non_cached_messages = [self.sample_messages[1]]  # User message
            mock_separate.return_value = (cached_messages, non_cached_messages)

            optional_params = self.sample_optional_params.copy()
            original_tools = optional_params["tools"].copy()

            # Mock the check_cache to return existing cache so we don't make HTTP calls
            with patch.object(
                self.context_caching, "check_cache", return_value="existing_cache"
            ):
                # Execute
                result = self.context_caching.check_and_create_cache(
                    messages=self.sample_messages,
                    optional_params=optional_params,
                    api_key="test_key",
                    api_base=None,
                    model="gemini-1.5-pro",
                    client=self.mock_client,
                    timeout=30.0,
                    logging_obj=self.mock_logging,
                )

            # Assert tools were popped from optional_params
            assert "tools" not in optional_params

            # But original tools should still be available for comparison
            assert original_tools == self.sample_tools

    def test_check_and_create_cache_tools_not_popped_when_no_cached_messages(self):
        """Test that tools are NOT popped from optional_params when there are no cached messages"""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            mock_separate.return_value = (
                [],
                self.sample_messages,
            )  # No cached messages

            optional_params = self.sample_optional_params.copy()
            original_tools = optional_params["tools"].copy()

            # Execute
            result = self.context_caching.check_and_create_cache(
                messages=self.sample_messages,
                optional_params=optional_params,
                api_key="test_key",
                api_base=None,
                model="gemini-1.5-pro",
                client=self.mock_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
            )

            # Assert tools were NOT popped from optional_params (early return)
            assert "tools" in optional_params
            assert optional_params["tools"] == original_tools

    @pytest.mark.asyncio
    async def test_async_check_and_create_cache_tools_not_popped_when_no_cached_messages(
        self,
    ):
        """Test that tools are NOT popped from optional_params in async version when there are no cached messages"""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            mock_separate.return_value = (
                [],
                self.sample_messages,
            )  # No cached messages

            optional_params = self.sample_optional_params.copy()
            original_tools = optional_params["tools"].copy()

            # Execute
            result = await self.context_caching.async_check_and_create_cache(
                messages=self.sample_messages,
                optional_params=optional_params,
                api_key="test_key",
                api_base=None,
                model="gemini-1.5-pro",
                client=self.mock_async_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
            )

            # Assert tools were NOT popped from optional_params (early return)
            assert "tools" in optional_params
            assert optional_params["tools"] == original_tools

    @pytest.mark.asyncio
    async def test_async_check_and_create_cache_tools_popped_from_optional_params(self):
        """Test that tools are properly popped from optional_params in async version when there are cached messages"""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            # Mock to return cached messages so tools get popped
            cached_messages = [
                self.sample_messages[0]
            ]  # System message with cache_control
            non_cached_messages = [self.sample_messages[1]]  # User message
            mock_separate.return_value = (cached_messages, non_cached_messages)

            optional_params = self.sample_optional_params.copy()
            original_tools = optional_params["tools"].copy()

            # Mock the async_check_cache to return existing cache so we don't make HTTP calls
            with patch.object(
                self.context_caching, "async_check_cache", return_value="existing_cache"
            ):
                # Execute
                result = await self.context_caching.async_check_and_create_cache(
                    messages=self.sample_messages,
                    optional_params=optional_params,
                    api_key="test_key",
                    api_base=None,
                    model="gemini-1.5-pro",
                    client=self.mock_async_client,
                    timeout=30.0,
                    logging_obj=self.mock_logging,
                )

            # Assert tools were popped from optional_params
            assert "tools" not in optional_params

            # But original tools should still be available for comparison
            assert original_tools == self.sample_tools
