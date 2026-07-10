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
    MAX_PAGINATION_PAGES,
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

        # Mock is_prompt_caching_valid_prompt to return True by default.
        # This avoids token counting in unit tests. The min-token guard is
        # tested explicitly in test_check_and_create_cache_skips_when_below_min_tokens.
        self._token_check_patcher = patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.is_prompt_caching_valid_prompt",
            return_value=True,
        )
        self._token_check_patcher.start()

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

    def teardown_method(self):
        """Teardown for each test method"""
        self._token_check_patcher.stop()

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    def test_check_and_create_cache_with_cached_content(
        self, mock_cache_obj, mock_separate, custom_llm_provider
    ):
        """Test check_and_create_cache when cached_content is provided"""
        # Setup
        cached_content = "cached_content_123"
        optional_params = self.sample_optional_params.copy()
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache == cached_content

        # Verify mocks weren't called since we short-circuited
        mock_separate.assert_not_called()
        mock_cache_obj.get_cache_key.assert_not_called()

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    def test_check_and_create_cache_no_cached_messages(
        self, mock_separate, custom_llm_provider
    ):
        """Test check_and_create_cache when no cached messages are found"""
        # Setup
        mock_separate.return_value = ([], self.sample_messages)  # No cached messages
        optional_params = self.sample_optional_params.copy()
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache is None

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    def test_check_and_create_cache_existing_cache_found(
        self, mock_check_cache, mock_cache_obj, mock_separate, custom_llm_provider
    ):
        """Test check_and_create_cache when existing cache is found"""
        # Setup
        cached_messages = [self.sample_messages[0]]  # System message with cache_control
        non_cached_messages = [self.sample_messages[1]]  # User message
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = "existing_cache_name"

        optional_params = self.sample_optional_params.copy()
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == non_cached_messages
        assert returned_params == optional_params
        assert returned_cache == "existing_cache_name"

        # Verify cache key was generated with tools, tool_choice and model
        mock_cache_obj.get_cache_key.assert_called_once_with(
            messages=cached_messages,
            tools=self.sample_tools,
            tool_choice=None,
            model="gemini-1.5-pro",
        )

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
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
        custom_llm_provider,
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
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
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

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    def test_check_and_create_cache_http_error(
        self,
        mock_get_token_url,
        mock_check_cache,
        mock_cache_obj,
        mock_separate,
        custom_llm_provider,
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
        test_project = "test_project"
        test_location = "us-central1"

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
                custom_llm_provider=custom_llm_provider,
                vertex_project=test_project,
                vertex_location=test_location,
                vertex_auth_header="vertext_test_token",
            )

        assert exc_info.value.status_code == 400
        assert "Bad Request" in str(exc_info.value.message)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    async def test_async_check_and_create_cache_with_cached_content(
        self, mock_cache_obj, mock_separate, custom_llm_provider
    ):
        """Test async_check_and_create_cache when cached_content is provided"""
        # Setup
        cached_content = "cached_content_123"
        optional_params = self.sample_optional_params.copy()
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache == cached_content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    async def test_async_check_and_create_cache_no_cached_messages(
        self, mock_separate, custom_llm_provider
    ):
        """Test async_check_and_create_cache when no cached messages are found"""
        # Setup
        mock_separate.return_value = ([], self.sample_messages)
        optional_params = self.sample_optional_params.copy()
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == self.sample_messages
        assert returned_params == optional_params
        assert returned_cache is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.local_cache_obj"
    )
    @patch.object(ContextCachingEndpoints, "async_check_cache")
    async def test_async_check_and_create_cache_existing_cache_found(
        self, mock_async_check_cache, mock_cache_obj, mock_separate, custom_llm_provider
    ):
        """Test async_check_and_create_cache when existing cache is found"""
        # Setup
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)

        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_async_check_cache.return_value = "existing_cache_name"

        optional_params = self.sample_optional_params.copy()
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
        )

        # Assert
        messages, returned_params, returned_cache = result
        assert messages == non_cached_messages
        assert returned_params == optional_params
        assert returned_cache == "existing_cache_name"

        # Verify cache key was generated with tools, tool_choice and model
        mock_cache_obj.get_cache_key.assert_called_once_with(
            messages=cached_messages,
            tools=self.sample_tools,
            tool_choice=None,
            model="gemini-1.5-pro",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
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
        custom_llm_provider,
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
        test_project = "test_project"
        test_location = "us-central1"

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
            custom_llm_provider=custom_llm_provider,
            vertex_project=test_project,
            vertex_location=test_location,
            vertex_auth_header="vertext_test_token",
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
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
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
        custom_llm_provider,
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
        test_project = "test_project"
        test_location = "us-central1"

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
                custom_llm_provider=custom_llm_provider,
                vertex_project=test_project,
                vertex_location=test_location,
                vertex_auth_header="vertext_test_token",
            )

        assert exc_info.value.status_code == 408
        assert "Timeout error occurred" in str(exc_info.value.message)

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_check_and_create_cache_tools_popped_from_optional_params(
        self, custom_llm_provider
    ):
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
            test_project = "test_project"
            test_location = "us-central1"

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
                    custom_llm_provider=custom_llm_provider,
                    vertex_project=test_project,
                    vertex_location=test_location,
                    vertex_auth_header="vertext_test_token",
                )

            # Assert tools were popped from optional_params
            assert "tools" not in optional_params

            # But original tools should still be available for comparison
            assert original_tools == self.sample_tools

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_check_and_create_cache_tools_not_popped_when_no_cached_messages(
        self, custom_llm_provider
    ):
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
            test_project = "test_project"
            test_location = "us-central1"

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
                custom_llm_provider=custom_llm_provider,
                vertex_project=test_project,
                vertex_location=test_location,
                vertex_auth_header="vertext_test_token",
            )

            # Assert tools were NOT popped from optional_params (early return)
            assert "tools" in optional_params
            assert optional_params["tools"] == original_tools

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    async def test_async_check_and_create_cache_tools_not_popped_when_no_cached_messages(
        self, custom_llm_provider
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
            test_project = "test_project"
            test_location = "us-central1"

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
                custom_llm_provider=custom_llm_provider,
                vertex_project=test_project,
                vertex_location=test_location,
                vertex_auth_header="vertext_test_token",
            )

            # Assert tools were NOT popped from optional_params (early return)
            assert "tools" in optional_params
            assert optional_params["tools"] == original_tools

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    async def test_async_check_and_create_cache_tools_popped_from_optional_params(
        self, custom_llm_provider
    ):
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
            test_project = "test_project"
            test_location = "us-central1"

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
                    custom_llm_provider=custom_llm_provider,
                    vertex_project=test_project,
                    vertex_location=test_location,
                    vertex_auth_header="vertext_test_token",
                )

            # Assert tools were popped from optional_params
            assert "tools" not in optional_params

            # But original tools should still be available for comparison
            assert original_tools == self.sample_tools

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_check_and_create_cache_tool_choice_popped_from_optional_params(
        self, custom_llm_provider
    ):
        """tool_choice is popped from optional_params when cached messages exist."""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            cached_messages = [self.sample_messages[0]]
            non_cached_messages = [self.sample_messages[1]]
            mock_separate.return_value = (cached_messages, non_cached_messages)

            optional_params = self.sample_optional_params.copy()
            optional_params["tool_choice"] = {"functionCallingConfig": {"mode": "ANY"}}

            with patch.object(
                self.context_caching, "check_cache", return_value="existing_cache"
            ):
                self.context_caching.check_and_create_cache(
                    messages=self.sample_messages,
                    optional_params=optional_params,
                    api_key="test_key",
                    api_base=None,
                    model="gemini-1.5-pro",
                    client=self.mock_client,
                    timeout=30.0,
                    logging_obj=self.mock_logging,
                    custom_llm_provider=custom_llm_provider,
                    vertex_project="test_project",
                    vertex_location="us-central1",
                    vertex_auth_header="vertext_test_token",
                )

            assert "tool_choice" not in optional_params

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    def test_check_and_create_cache_tool_choice_not_popped_when_no_cached_messages(
        self, custom_llm_provider
    ):
        """tool_choice is NOT popped when there are no cached messages (early return)."""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            mock_separate.return_value = ([], self.sample_messages)

            tool_choice = {"functionCallingConfig": {"mode": "AUTO"}}
            optional_params = self.sample_optional_params.copy()
            optional_params["tool_choice"] = tool_choice

            self.context_caching.check_and_create_cache(
                messages=self.sample_messages,
                optional_params=optional_params,
                api_key="test_key",
                api_base=None,
                model="gemini-1.5-pro",
                client=self.mock_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
                custom_llm_provider=custom_llm_provider,
                vertex_project="test_project",
                vertex_location="us-central1",
                vertex_auth_header="vertext_test_token",
            )

            assert optional_params.get("tool_choice") == tool_choice

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    async def test_async_check_and_create_cache_tool_choice_popped_from_optional_params(
        self, custom_llm_provider
    ):
        """Async equivalent of test_check_and_create_cache_tool_choice_popped_from_optional_params."""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            cached_messages = [self.sample_messages[0]]
            non_cached_messages = [self.sample_messages[1]]
            mock_separate.return_value = (cached_messages, non_cached_messages)

            optional_params = self.sample_optional_params.copy()
            optional_params["tool_choice"] = {"functionCallingConfig": {"mode": "ANY"}}

            with patch.object(
                self.context_caching, "async_check_cache", return_value="existing_cache"
            ):
                await self.context_caching.async_check_and_create_cache(
                    messages=self.sample_messages,
                    optional_params=optional_params,
                    api_key="test_key",
                    api_base=None,
                    model="gemini-1.5-pro",
                    client=self.mock_async_client,
                    timeout=30.0,
                    logging_obj=self.mock_logging,
                    custom_llm_provider=custom_llm_provider,
                    vertex_project="test_project",
                    vertex_location="us-central1",
                    vertex_auth_header="vertext_test_token",
                )

            assert "tool_choice" not in optional_params

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    async def test_async_check_and_create_cache_tool_choice_not_popped_when_no_cached_messages(
        self, custom_llm_provider
    ):
        """Async equivalent of test_check_and_create_cache_tool_choice_not_popped_when_no_cached_messages."""
        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
        ) as mock_separate:
            mock_separate.return_value = ([], self.sample_messages)

            tool_choice = {"functionCallingConfig": {"mode": "AUTO"}}
            optional_params = self.sample_optional_params.copy()
            optional_params["tool_choice"] = tool_choice

            await self.context_caching.async_check_and_create_cache(
                messages=self.sample_messages,
                optional_params=optional_params,
                api_key="test_key",
                api_base=None,
                model="gemini-1.5-pro",
                client=self.mock_async_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
                custom_llm_provider=custom_llm_provider,
                vertex_project="test_project",
                vertex_location="us-central1",
                vertex_auth_header="vertext_test_token",
            )

            assert optional_params.get("tool_choice") == tool_choice

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
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
    def test_check_and_create_cache_tool_choice_in_request_body(
        self,
        mock_get_token_url,
        mock_check_cache,
        mock_transform,
        mock_cache_obj,
        mock_separate,
        custom_llm_provider,
    ):
        """End-to-end: tool_choice ends up as `toolConfig` on the cache-creation HTTP POST body."""
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None  # cache miss -> create new
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        mock_transform.return_value = {"model": "gemini-1.5-pro", "contents": []}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "new_cache_name",
            "model": "gemini-1.5-pro",
        }
        self.mock_client.post.return_value = mock_response

        tool_choice = {"functionCallingConfig": {"mode": "ANY"}}
        optional_params = self.sample_optional_params.copy()
        optional_params["tool_choice"] = tool_choice

        self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="vertext_test_token",
        )

        self.mock_client.post.assert_called_once()
        call_args = self.mock_client.post.call_args
        assert call_args.kwargs["json"]["tools"] == self.sample_tools
        assert call_args.kwargs["json"]["toolConfig"] == tool_choice
        mock_cache_obj.get_cache_key.assert_called_once_with(
            messages=cached_messages,
            tools=self.sample_tools,
            tool_choice=tool_choice,
            model="gemini-1.5-pro",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
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
    @patch.object(ContextCachingEndpoints, "async_check_cache")
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    async def test_async_check_and_create_cache_tool_choice_in_request_body(
        self,
        mock_get_token_url,
        mock_check_cache,
        mock_transform,
        mock_cache_obj,
        mock_separate,
        custom_llm_provider,
    ):
        """Async equivalent of test_check_and_create_cache_tool_choice_in_request_body."""
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        mock_transform.return_value = {"model": "gemini-1.5-pro", "contents": []}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "new_cache_name",
            "model": "gemini-1.5-pro",
        }
        self.mock_async_client.post = AsyncMock(return_value=mock_response)

        tool_choice = {"functionCallingConfig": {"mode": "ANY"}}
        optional_params = self.sample_optional_params.copy()
        optional_params["tool_choice"] = tool_choice

        await self.context_caching.async_check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_async_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="vertext_test_token",
        )

        call_args = self.mock_async_client.post.call_args
        assert call_args.kwargs["json"]["tools"] == self.sample_tools
        assert call_args.kwargs["json"]["toolConfig"] == tool_choice
        mock_cache_obj.get_cache_key.assert_called_once_with(
            messages=cached_messages,
            tools=self.sample_tools,
            tool_choice=tool_choice,
            model="gemini-1.5-pro",
        )

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
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
    def test_check_and_create_cache_omits_tool_config_when_tool_choice_unset(
        self,
        mock_get_token_url,
        mock_check_cache,
        mock_transform,
        mock_cache_obj,
        mock_separate,
        custom_llm_provider,
    ):
        """When the caller didn't pass tool_choice, toolConfig must NOT appear in the cache body."""
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        mock_transform.return_value = {"model": "gemini-1.5-pro", "contents": []}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "new_cache_name",
            "model": "gemini-1.5-pro",
        }
        self.mock_client.post.return_value = mock_response

        optional_params = self.sample_optional_params.copy()

        self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="vertext_test_token",
        )

        call_args = self.mock_client.post.call_args
        assert "tools" in call_args.kwargs["json"]
        assert "toolConfig" not in call_args.kwargs["json"]

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
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
    def test_check_and_create_cache_tool_choice_function_pin(
        self,
        mock_get_token_url,
        mock_check_cache,
        mock_transform,
        mock_cache_obj,
        mock_separate,
        custom_llm_provider,
    ):
        """tool_choice as a function-pin dict survives the cache body intact."""
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        mock_transform.return_value = {"model": "gemini-1.5-pro", "contents": []}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "new_cache_name",
            "model": "gemini-1.5-pro",
        }
        self.mock_client.post.return_value = mock_response

        function_pin = {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowed_function_names": ["get_current_weather"],
            }
        }
        optional_params = self.sample_optional_params.copy()
        optional_params["tool_choice"] = function_pin

        self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="vertext_test_token",
        )

        call_args = self.mock_client.post.call_args
        assert call_args.kwargs["json"]["toolConfig"] == function_pin

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
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
    def test_check_and_create_cache_tool_choice_typed_constructor(
        self,
        mock_get_token_url,
        mock_check_cache,
        mock_transform,
        mock_cache_obj,
        mock_separate,
        custom_llm_provider,
    ):
        """Exercise the actual ToolConfig(FunctionCallingConfig(...)) constructor that map_tool_choice_values produces.

        ToolConfig / FunctionCallingConfig are TypedDicts (litellm/types/llms/vertex_ai.py:158, 277)
        so this is functionally identical to the dict-literal tests above at
        runtime — but exercising the typed constructor pins the test to the
        same call shape map_tool_choice_values uses and auto-follows if
        either type ever migrates to a Pydantic model upstream.
        """
        from litellm.types.llms.vertex_ai import (
            FunctionCallingConfig,
            ToolConfig,
        )

        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)
        mock_cache_obj.get_cache_key.return_value = "test_cache_key"
        mock_check_cache.return_value = None
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        mock_transform.return_value = {"model": "gemini-1.5-pro", "contents": []}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "new_cache_name",
            "model": "gemini-1.5-pro",
        }
        self.mock_client.post.return_value = mock_response

        tool_choice = ToolConfig(
            functionCallingConfig=FunctionCallingConfig(mode="ANY")
        )
        optional_params = self.sample_optional_params.copy()
        optional_params["tool_choice"] = tool_choice

        self.context_caching.check_and_create_cache(
            messages=self.sample_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="vertext_test_token",
        )

        call_args = self.mock_client.post.call_args
        assert call_args.kwargs["json"]["toolConfig"] == tool_choice
        assert call_args.kwargs["json"]["toolConfig"] == {
            "functionCallingConfig": {"mode": "ANY"}
        }
        mock_cache_obj.get_cache_key.assert_called_once_with(
            messages=cached_messages,
            tools=self.sample_tools,
            tool_choice=tool_choice,
            model="gemini-1.5-pro",
        )

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @patch.object(ContextCachingEndpoints, "check_cache")
    def test_check_and_create_cache_distinct_tool_choices_use_distinct_keys(
        self,
        mock_check_cache,
        mock_separate,
        custom_llm_provider,
    ):
        """Two requests with different tool_choice values must produce different cache keys.

        Runs the real local_cache_obj.get_cache_key to verify the hashed
        output actually differs — mocking it would only prove that distinct
        arguments are forwarded, not that they produce distinct keys.
        """
        cached_messages = [self.sample_messages[0]]
        non_cached_messages = [self.sample_messages[1]]
        mock_separate.return_value = (cached_messages, non_cached_messages)
        mock_check_cache.return_value = "existing_cache"

        auto_tool_choice = {"functionCallingConfig": {"mode": "AUTO"}}
        any_tool_choice = {"functionCallingConfig": {"mode": "ANY"}}
        for choice in (auto_tool_choice, any_tool_choice):
            optional_params = self.sample_optional_params.copy()
            optional_params["tool_choice"] = choice
            self.context_caching.check_and_create_cache(
                messages=self.sample_messages,
                optional_params=optional_params,
                api_key="test_key",
                api_base=None,
                model="gemini-1.5-pro",
                client=self.mock_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
                custom_llm_provider=custom_llm_provider,
                vertex_project="test_project",
                vertex_location="us-central1",
                vertex_auth_header="vertext_test_token",
            )

        check_cache_calls = mock_check_cache.call_args_list
        assert len(check_cache_calls) == 2
        first_cache_key = check_cache_calls[0].kwargs["cache_key"]
        second_cache_key = check_cache_calls[1].kwargs["cache_key"]
        assert first_cache_key != second_cache_key

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    def test_check_and_create_cache_skips_when_below_min_tokens(
        self, mock_separate, custom_llm_provider
    ):
        """Test that context caching is skipped when cached content is below 1024 tokens.

        Gemini requires a minimum of 1024 tokens for context caching. If the cached
        content is too small, the request should proceed without caching instead of
        failing with a Gemini API error.
        """
        # Stop the default mock so the real token count check runs
        self._token_check_patcher.stop()

        short_cached_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        non_cached_messages = [
            {"role": "user", "content": "Hello"},
        ]
        all_messages = short_cached_messages + non_cached_messages
        mock_separate.return_value = (short_cached_messages, non_cached_messages)
        optional_params = self.sample_optional_params.copy()

        result = self.context_caching.check_and_create_cache(
            messages=all_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            cached_content=None,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="test_token",
        )

        messages, returned_params, returned_cache = result
        assert messages == all_messages
        assert returned_cache is None

        # Restart the patcher so teardown_method can stop it cleanly
        self._token_check_patcher.start()

    @pytest.mark.parametrize(
        "model, expected_min",
        [
            ("gemini-3.5-flash", 4096),
            ("gemini/gemini-3.5-flash", 4096),
            ("gemini-3.1-pro-preview", 4096),
            ("gemini-2.5-flash", 2048),
            ("gemini-2.5-pro", 2048),
        ],
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    def test_check_and_create_cache_uses_model_specific_min_tokens(
        self, mock_separate, model, expected_min
    ):
        """The Gemini per-model floor must be forwarded to the token-count guard.

        A flat 1024 floor let content between 1024 and the real minimum (2048 for
        2.5, 4096 for 3.x) reach Gemini and 400. Assert the model-derived floor is
        passed so the guard skips instead of erroring.
        """
        self._token_check_patcher.stop()

        cached_messages = [
            {
                "role": "system",
                "content": "cached",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        non_cached_messages = [{"role": "user", "content": "Hello"}]
        mock_separate.return_value = (cached_messages, non_cached_messages)

        with patch(
            "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.is_prompt_caching_valid_prompt",
            return_value=False,
        ) as mock_valid:
            self.context_caching.check_and_create_cache(
                messages=cached_messages + non_cached_messages,
                optional_params=self.sample_optional_params.copy(),
                api_key="test_key",
                api_base=None,
                model=model,
                client=self.mock_client,
                timeout=30.0,
                logging_obj=self.mock_logging,
                cached_content=None,
                custom_llm_provider="gemini",
                vertex_project="test_project",
                vertex_location="us-central1",
                vertex_auth_header="test_token",
            )

        assert mock_valid.call_args.kwargs["min_token_count"] == expected_min

        self._token_check_patcher.start()

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch(
        "litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching.separate_cached_messages"
    )
    @pytest.mark.asyncio
    async def test_async_check_and_create_cache_skips_when_below_min_tokens(
        self, mock_separate, custom_llm_provider
    ):
        """Test that async context caching is skipped when cached content is below 1024 tokens."""
        # Stop the default mock so the real token count check runs
        self._token_check_patcher.stop()

        short_cached_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        non_cached_messages = [
            {"role": "user", "content": "Hello"},
        ]
        all_messages = short_cached_messages + non_cached_messages
        mock_separate.return_value = (short_cached_messages, non_cached_messages)
        optional_params = self.sample_optional_params.copy()

        result = await self.context_caching.async_check_and_create_cache(
            messages=all_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-1.5-pro",
            client=self.mock_async_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            cached_content=None,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="test_token",
        )

        messages, returned_params, returned_cache = result
        assert messages == all_messages
        assert returned_cache is None

        # Restart the patcher so teardown_method can stop it cleanly
        self._token_check_patcher.start()


class TestCheckCachePagination:
    """Test pagination logic in check_cache and async_check_cache methods."""

    def setup_method(self):
        """Setup for each test method"""
        self.context_caching = ContextCachingEndpoints()
        self.mock_logging = MagicMock(spec=Logging)
        self.mock_client = MagicMock(spec=HTTPHandler)
        self.mock_async_client = MagicMock(spec=AsyncHTTPHandler)

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    def test_check_cache_pagination_finds_cache_on_second_page(
        self, mock_get_token_url, custom_llm_provider
    ):
        """Test that check_cache correctly handles pagination and finds cache on second page"""
        # Setup
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "target_cache_key"

        # Mock first page response (no match, has nextPageToken)
        first_page_response = MagicMock()
        first_page_response.json.return_value = {
            "cachedContents": [
                {"name": "cache_1", "displayName": "cache_key_1"},
                {"name": "cache_2", "displayName": "cache_key_2"},
            ],
            "nextPageToken": "token_page_2",
        }

        # Mock second page response (has match, no nextPageToken)
        second_page_response = MagicMock()
        second_page_response.json.return_value = {
            "cachedContents": [
                {"name": "cache_3", "displayName": cache_key_to_find},
                {"name": "cache_4", "displayName": "cache_key_4"},
            ]
        }

        # Setup mock client to return different responses
        self.mock_client.get.side_effect = [first_page_response, second_page_response]

        # Execute
        result = self.context_caching.check_cache(
            cache_key=cache_key_to_find,
            client=self.mock_client,
            headers={"Authorization": "Bearer token"},
            api_key="test_key",
            api_base=None,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer test-token",
        )

        # Assert
        assert result == "cache_3"
        assert self.mock_client.get.call_count == 2
        # Check that second call includes pageToken
        second_call_url = self.mock_client.get.call_args_list[1].kwargs["url"]
        assert "pageToken=token_page_2" in second_call_url

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    def test_check_cache_pagination_stops_when_no_next_token(
        self, mock_get_token_url, custom_llm_provider
    ):
        """Test that check_cache stops pagination when no nextPageToken is present"""
        # Setup
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "nonexistent_cache_key"

        # Mock response without nextPageToken
        response = MagicMock()
        response.json.return_value = {
            "cachedContents": [
                {"name": "cache_1", "displayName": "cache_key_1"},
                {"name": "cache_2", "displayName": "cache_key_2"},
            ]
        }

        self.mock_client.get.return_value = response

        # Execute
        result = self.context_caching.check_cache(
            cache_key=cache_key_to_find,
            client=self.mock_client,
            headers={"Authorization": "Bearer token"},
            api_key="test_key",
            api_base=None,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer test-token",
        )

        # Assert
        assert result is None
        assert self.mock_client.get.call_count == 1

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    def test_check_cache_pagination_multiple_pages(
        self, mock_get_token_url, custom_llm_provider
    ):
        """Test that check_cache correctly iterates through multiple pages"""
        # Setup
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "target_cache_key"

        # Mock three pages
        page1 = MagicMock()
        page1.json.return_value = {
            "cachedContents": [{"name": "cache_1", "displayName": "cache_key_1"}],
            "nextPageToken": "token_page_2",
        }

        page2 = MagicMock()
        page2.json.return_value = {
            "cachedContents": [{"name": "cache_2", "displayName": "cache_key_2"}],
            "nextPageToken": "token_page_3",
        }

        page3 = MagicMock()
        page3.json.return_value = {
            "cachedContents": [{"name": "cache_3", "displayName": cache_key_to_find}],
        }

        self.mock_client.get.side_effect = [page1, page2, page3]

        # Execute
        result = self.context_caching.check_cache(
            cache_key=cache_key_to_find,
            client=self.mock_client,
            headers={"Authorization": "Bearer token"},
            api_key="test_key",
            api_base=None,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer test-token",
        )

        # Assert
        assert result == "cache_3"
        assert self.mock_client.get.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    async def test_async_check_cache_pagination_finds_cache_on_second_page(
        self, mock_get_token_url, custom_llm_provider
    ):
        """Test that async_check_cache correctly handles pagination and finds cache on second page"""
        # Setup
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "target_cache_key"

        # Mock first page response (no match, has nextPageToken)
        first_page_response = MagicMock()
        first_page_response.json.return_value = {
            "cachedContents": [
                {"name": "cache_1", "displayName": "cache_key_1"},
                {"name": "cache_2", "displayName": "cache_key_2"},
            ],
            "nextPageToken": "token_page_2",
        }

        # Mock second page response (has match, no nextPageToken)
        second_page_response = MagicMock()
        second_page_response.json.return_value = {
            "cachedContents": [
                {"name": "cache_3", "displayName": cache_key_to_find},
                {"name": "cache_4", "displayName": "cache_key_4"},
            ]
        }

        # Setup mock async client to return different responses
        self.mock_async_client.get = AsyncMock(
            side_effect=[first_page_response, second_page_response]
        )

        # Execute
        result = await self.context_caching.async_check_cache(
            cache_key=cache_key_to_find,
            client=self.mock_async_client,
            headers={"Authorization": "Bearer token"},
            api_key="test_key",
            api_base=None,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer test-token",
        )

        # Assert
        assert result == "cache_3"
        assert self.mock_async_client.get.call_count == 2
        # Check that second call includes pageToken
        second_call_url = self.mock_async_client.get.call_args_list[1].kwargs["url"]
        assert "pageToken=token_page_2" in second_call_url

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    async def test_async_check_cache_pagination_stops_when_no_next_token(
        self, mock_get_token_url, custom_llm_provider
    ):
        """Test that async_check_cache stops pagination when no nextPageToken is present"""
        # Setup
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "nonexistent_cache_key"

        # Mock response without nextPageToken
        response = MagicMock()
        response.json.return_value = {
            "cachedContents": [
                {"name": "cache_1", "displayName": "cache_key_1"},
                {"name": "cache_2", "displayName": "cache_key_2"},
            ]
        }

        self.mock_async_client.get = AsyncMock(return_value=response)

        # Execute
        result = await self.context_caching.async_check_cache(
            cache_key=cache_key_to_find,
            client=self.mock_async_client,
            headers={"Authorization": "Bearer token"},
            api_key="test_key",
            api_base=None,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer test-token",
        )

        # Assert
        assert result is None
        assert self.mock_async_client.get.call_count == 1

    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    def test_check_cache_pagination_max_pages_limit(
        self, mock_get_token_url, custom_llm_provider
    ):
        """Test that pagination stops after MAX_PAGINATION_PAGES iterations"""
        # Setup
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "nonexistent_cache_key"

        # Create mock response that always has nextPageToken (infinite pagination scenario)
        def create_page_response(page_num):
            response = MagicMock()
            response.json.return_value = {
                "cachedContents": [
                    {"name": f"cache_{page_num}", "displayName": f"key_{page_num}"}
                ],
                "nextPageToken": f"token_page_{page_num + 1}",
            }
            return response

        # Create MAX_PAGINATION_PAGES responses, each with a nextPageToken
        self.mock_client.get.side_effect = [
            create_page_response(i) for i in range(MAX_PAGINATION_PAGES)
        ]

        # Execute
        result = self.context_caching.check_cache(
            cache_key=cache_key_to_find,
            client=self.mock_client,
            headers={"Authorization": "Bearer token"},
            api_key="test_key",
            api_base=None,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer test-token",
        )

        # Assert - should return None after exhausting all pages without finding match
        assert result is None
        # Verify exactly MAX_PAGINATION_PAGES API calls were made (not more)
        assert self.mock_client.get.call_count == MAX_PAGINATION_PAGES

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    async def test_async_check_cache_pagination_max_pages_limit(
        self, mock_get_token_url, custom_llm_provider
    ):
        """Test that async pagination stops after MAX_PAGINATION_PAGES iterations"""
        # Setup
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "nonexistent_cache_key"

        # Create mock response that always has nextPageToken (infinite pagination scenario)
        def create_page_response(page_num):
            response = MagicMock()
            response.json.return_value = {
                "cachedContents": [
                    {"name": f"cache_{page_num}", "displayName": f"key_{page_num}"}
                ],
                "nextPageToken": f"token_page_{page_num + 1}",
            }
            return response

        # Create MAX_PAGINATION_PAGES responses, each with a nextPageToken
        self.mock_async_client.get = AsyncMock(
            side_effect=[create_page_response(i) for i in range(MAX_PAGINATION_PAGES)]
        )

        # Execute
        result = await self.context_caching.async_check_cache(
            cache_key=cache_key_to_find,
            client=self.mock_async_client,
            headers={"Authorization": "Bearer token"},
            api_key="test_key",
            api_base=None,
            logging_obj=self.mock_logging,
            custom_llm_provider=custom_llm_provider,
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer test-token",
        )

        # Assert - should return None after exhausting all pages without finding match
        assert result is None
        # Verify exactly MAX_PAGINATION_PAGES async API calls were made (not more)
        assert self.mock_async_client.get.call_count == MAX_PAGINATION_PAGES


class TestVertexAIGlobalLocation:
    """Test global location handling in context caching."""

    def test_global_location_url_construction_v1(self):
        """Test that global location uses correct URL (no location prefix) for v1 API."""
        caching = ContextCachingEndpoints()

        # Mock the _check_custom_proxy to return the URL unchanged
        with patch.object(
            caching,
            "_check_custom_proxy",
            side_effect=lambda **kwargs: (kwargs.get("auth_header"), kwargs.get("url")),
        ):
            auth_header, url = caching._get_token_and_url_context_caching(
                gemini_api_key=None,
                custom_llm_provider="vertex_ai",
                api_base=None,
                vertex_project="test-project",
                vertex_location="global",
                vertex_auth_header="Bearer test-token",
            )

            # Assert correct URL format for global
            expected_url = "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/cachedContents"
            assert url == expected_url, f"Expected {expected_url}, got {url}"
            assert (
                "global-aiplatform" not in url
            ), "URL should not contain 'global-aiplatform' prefix"

    def test_regional_location_url_construction_v1(self):
        """Test that regional location uses correct URL (with location prefix) for v1 API."""
        caching = ContextCachingEndpoints()

        with patch.object(
            caching,
            "_check_custom_proxy",
            side_effect=lambda **kwargs: (kwargs.get("auth_header"), kwargs.get("url")),
        ):
            auth_header, url = caching._get_token_and_url_context_caching(
                gemini_api_key=None,
                custom_llm_provider="vertex_ai",
                api_base=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_auth_header="Bearer test-token",
            )

            # Assert correct URL format for regional
            expected_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/cachedContents"
            assert url == expected_url, f"Expected {expected_url}, got {url}"

    def test_global_location_url_construction_v1beta1(self):
        """Test that global location uses correct URL for v1beta1 API."""
        caching = ContextCachingEndpoints()

        with patch.object(
            caching,
            "_check_custom_proxy",
            side_effect=lambda **kwargs: (kwargs.get("auth_header"), kwargs.get("url")),
        ):
            auth_header, url = caching._get_token_and_url_context_caching(
                gemini_api_key=None,
                custom_llm_provider="vertex_ai_beta",
                api_base=None,
                vertex_project="test-project",
                vertex_location="global",
                vertex_auth_header="Bearer test-token",
            )

            # Assert correct URL format for global with beta API
            expected_url = "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/cachedContents"
            assert url == expected_url, f"Expected {expected_url}, got {url}"
            assert (
                "global-aiplatform" not in url
            ), "URL should not contain 'global-aiplatform' prefix"

    def test_gemini_context_caching_with_custom_api_base_passes_model(self):
        """Gemini context caching with custom api_base must pass model to _check_custom_proxy.

        Regression test for https://github.com/BerriAI/litellm/issues/23846
        Previously model was hardcoded to None, causing ValueError when api_base was set.
        """
        caching = ContextCachingEndpoints()

        auth_header, url = caching._get_token_and_url_context_caching(
            gemini_api_key="test-key",
            custom_llm_provider="gemini",
            api_base="https://my-proxy.example.com",
            vertex_project=None,
            vertex_location=None,
            vertex_auth_header=None,
            model="gemini-1.5-pro",
        )

        assert "models/gemini-1.5-pro" in url
        assert url.startswith("https://my-proxy.example.com/")

    def test_gemini_context_caching_without_api_base_ignores_model(self):
        """Without custom api_base, model param is not needed (default URL is used)."""
        caching = ContextCachingEndpoints()

        auth_header, url = caching._get_token_and_url_context_caching(
            gemini_api_key="test-key",
            custom_llm_provider="gemini",
            api_base=None,
            vertex_project=None,
            vertex_location=None,
            vertex_auth_header=None,
        )

        assert "generativelanguage.googleapis.com" in url
        assert "cachedContents" in url


class TestContextCachingMultiRegionUrls:
    """Regression coverage for #29571: multi-region vertex_location values
    (`eu`, `us`) must resolve to the REP host (`aiplatform.{geo}.rep.googleapis.com`)
    on the cachedContents endpoint, matching the inference path (already
    fixed in #27293). Previously the URL was hardcoded to
    `{location}-aiplatform.googleapis.com`, which doesn't exist for
    multi-region locations and 404'd."""

    def setup_method(self):
        self.caching = ContextCachingEndpoints()

    @pytest.mark.parametrize("location", ["eu", "us"])
    def test_vertex_ai_multi_region_uses_rep_host(self, location):
        _, url = self.caching._get_token_and_url_context_caching(
            gemini_api_key=None,
            custom_llm_provider="vertex_ai",
            api_base=None,
            vertex_project="my-project",
            vertex_location=location,
            vertex_auth_header="Bearer token",
        )

        assert url.startswith(f"https://aiplatform.{location}.rep.googleapis.com/")
        assert f"/locations/{location}/cachedContents" in url
        # Old broken host must no longer appear.
        assert f"{location}-aiplatform.googleapis.com" not in url

    def test_vertex_ai_regional_still_uses_regional_host(self):
        _, url = self.caching._get_token_and_url_context_caching(
            gemini_api_key=None,
            custom_llm_provider="vertex_ai",
            api_base=None,
            vertex_project="my-project",
            vertex_location="us-central1",
            vertex_auth_header="Bearer token",
        )

        assert url.startswith("https://us-central1-aiplatform.googleapis.com/")
        assert "/locations/us-central1/cachedContents" in url

    def test_vertex_ai_global_still_uses_global_host(self):
        _, url = self.caching._get_token_and_url_context_caching(
            gemini_api_key=None,
            custom_llm_provider="vertex_ai",
            api_base=None,
            vertex_project="my-project",
            vertex_location="global",
            vertex_auth_header="Bearer token",
        )

        assert url.startswith("https://aiplatform.googleapis.com/")
        assert "/locations/global/cachedContents" in url
