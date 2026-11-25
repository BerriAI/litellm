import asyncio
import base64
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import chain
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
import responses
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import CONVERTED_SYSTEM_NOTE
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.context_caching.transformation import transform_openai_messages_to_gemini_context_caching
from litellm.llms.vertex_ai.context_caching.vertex_ai_context_caching import (
    ContextCachingEndpoints,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionImageUrlObject,
    ChatCompletionSystemMessage,
    ChatCompletionToolParam,
    ChatCompletionUserMessage,
)


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Force the bundled in-repo cost map so capability and pricing assertions do not
    depend on the network-fetched ``main`` copy, which lags this branch until merge.

    ``get_model_info`` is lru_cached, so swapping ``model_cost`` is not enough on its
    own; clear on the way in and out so entries warmed against either map never leak
    across tests."""
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


class TestContextCachingEndpoints:
    """Test class for ContextCachingEndpoints methods"""

    def setup_method(self):
        """Setup for each test method"""
        self.context_caching = ContextCachingEndpoints()
        self.mock_logging = MagicMock(spec=Logging)
        self.mock_logging.model_call_details = {}
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

    def test_check_and_create_cache_skips_between_default_and_gemini_2_5_pro_minimum(
        self, local_model_cost_map
    ):
        model = "gemini-2.5-pro"
        self._token_check_patcher.stop()

        cached_messages = [
            {
                "role": "system",
                "content": " ".join(["word"] * 1500),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        non_cached_messages = [{"role": "user", "content": "Hello"}]

        messages, _, returned_cache = self.context_caching.check_and_create_cache(
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

        assert messages == cached_messages + non_cached_messages
        assert returned_cache is None
        self.mock_client.post.assert_not_called()

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

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai"]
    )
    @pytest.mark.asyncio
    async def test_check_and_create_cache_considers_tools_for_min_tokens(
        self, custom_llm_provider, is_async
    ):
        """Test that context caching accounts for tools when validating minimum token count.

        Fixes #42804: When messages alone are below the threshold, but tools push the total
        over the minimum token count, context caching must proceed and include tools.
        """
        self._token_check_patcher.stop()

        short_cached_messages = [
            {
                "role": "system",
                "content": "Short system instruction.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        non_cached_messages = [
            {"role": "user", "content": "Hello world"},
        ]
        all_messages = short_cached_messages + non_cached_messages

        openai_large_tools: Final = [
            {
                "type": "function",
                "function": {
                    "name": f"synthetic_tool_{i}",
                    "description": "A very descriptive explanation of a synthetic tool designed to add tokens to the prompt cache prefix " * 8,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            f"arg_{j}": {"type": "string", "description": "Argument description for caching verification " * 4}
                            for j in range(10)
                        },
                        "required": [f"arg_{j}" for j in range(5)],
                    },
                },
            }
            for i in range(12)
        ]
        large_tools: Final = litellm.utils.get_optional_params(
            model="gemini-1.5-pro",
            custom_llm_provider="gemini",
            tools=openai_large_tools,
        )["tools"]

        optional_params = {
            **self.sample_optional_params,
            "tools": large_tools,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "cachedContents/test_cache_id",
            "model": "gemini-1.5-pro",
        }
        mock_response.status_code = 200
        self.mock_client.post.return_value = mock_response
        self.mock_async_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            self.context_caching,
            "_get_token_and_url_context_caching",
            return_value=("fake_token", "https://fake.url/cachedContents"),
        ), patch.object(
            self.context_caching,
            "check_cache",
            return_value=None,
        ), patch.object(
            self.context_caching,
            "async_check_cache",
            new_callable=AsyncMock,
            return_value=None,
        ):
            if is_async:
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
            else:
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
        assert messages == non_cached_messages
        assert returned_cache == "cachedContents/test_cache_id"
        assert "tools" not in returned_params

        post_mock = self.mock_async_client.post if is_async else self.mock_client.post
        post_mock.assert_called_once()
        call_kwargs = post_mock.call_args.kwargs
        assert call_kwargs["json"]["tools"] == large_tools
        assert call_kwargs["json"]["contents"] == [
            {"role": "user", "parts": [{"text": "Short system instruction."}]}
        ]

        self._token_check_patcher.start()


    def _model_turn_final_messages(self, final_cached_role):
        tool_call = {
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'},
        }
        cached_tail = {
            "assistant": [],
            "tool": [
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "72F and sunny",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "system": [
                {
                    "role": "system",
                    "content": "Tool results are authoritative.",
                    "cache_control": {"type": "ephemeral"},
                },
                {"role": "tool", "tool_call_id": "call_abc123", "content": "72F and sunny"},
            ],
            "system_before_an_uncached_reminder": [
                {
                    "role": "system",
                    "content": "Tool results are authoritative.",
                    "cache_control": {"type": "ephemeral"},
                },
                {"role": "system", "content": "2 of 8 turns used"},
                {"role": "tool", "tool_call_id": "call_abc123", "content": "72F and sunny"},
            ],
        }[final_cached_role]
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Use the weather tool for every answer.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call],
                "cache_control": {"type": "ephemeral"},
            },
            *cached_tail,
            {"role": "user", "content": "What is the weather in Boston?"},
        ]

    @pytest.mark.parametrize("final_cached_role", ["assistant", "tool", "system", "system_before_an_uncached_reminder"])
    def test_check_and_create_cache_skips_when_cached_block_ends_on_model_turn(
        self, final_cached_role
    ):
        """The cachedContents API rejects contents ending on an assistant or tool turn
        with HTTP 400 "Requests ending with a model turn are not supported", so the
        request must proceed uncached instead of failing.
        """
        all_messages = self._model_turn_final_messages(final_cached_role)
        optional_params = self.sample_optional_params.copy()

        result = self.context_caching.check_and_create_cache(
            messages=all_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-3.6-flash",
            client=self.mock_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            cached_content=None,
            custom_llm_provider="vertex_ai",
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="test_token",
        )

        messages, returned_params, returned_cache = result
        assert messages == all_messages
        assert returned_cache is None
        assert "tools" in returned_params
        self.mock_client.get.assert_not_called()
        self.mock_client.post.assert_not_called()

    @pytest.mark.parametrize("final_cached_role", ["assistant", "tool", "system", "system_before_an_uncached_reminder"])
    @pytest.mark.asyncio
    async def test_async_check_and_create_cache_skips_when_cached_block_ends_on_model_turn(
        self, final_cached_role
    ):
        """Async variant: an unsupported terminal turn skips caching instead of failing."""
        all_messages = self._model_turn_final_messages(final_cached_role)
        optional_params = self.sample_optional_params.copy()

        result = await self.context_caching.async_check_and_create_cache(
            messages=all_messages,
            optional_params=optional_params,
            api_key="test_key",
            api_base=None,
            model="gemini-3.6-flash",
            client=self.mock_async_client,
            timeout=30.0,
            logging_obj=self.mock_logging,
            cached_content=None,
            custom_llm_provider="vertex_ai",
            vertex_project="test_project",
            vertex_location="us-central1",
            vertex_auth_header="test_token",
        )

        messages, returned_params, returned_cache = result
        assert messages == all_messages
        assert returned_cache is None
        assert "tools" in returned_params
        self.mock_async_client.get.assert_not_called()
        self.mock_async_client.post.assert_not_called()


def _marked_user(text: str) -> ChatCompletionUserMessage:
    return {"role": "user", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}


_SYSTEM: Final[ChatCompletionSystemMessage] = {"role": "system", "content": "sys"}
_Q1: Final[ChatCompletionUserMessage] = {"role": "user", "content": "Q1"}
_A1: Final[ChatCompletionAssistantMessage] = {"role": "assistant", "content": "A1"}
_A2: Final[ChatCompletionAssistantMessage] = {"role": "assistant", "content": "A2"}
_Q3: Final[ChatCompletionUserMessage] = {"role": "user", "content": "Q3"}
_MARKED_Q2: Final = _marked_user("Q2")
_MARKED_SYSTEM: Final[ChatCompletionSystemMessage] = {
    "role": "system",
    "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
}


@pytest.mark.parametrize(
    ("messages", "cache_accepts", "expected_cached", "expected_non_cached"),
    [
        pytest.param(
            [_SYSTEM, _Q1, _A1, _MARKED_Q2],
            True,
            [_SYSTEM],
            [_Q1, _A1, _MARKED_Q2],
            id="marked-latest-turn",
        ),
        pytest.param(
            [_MARKED_SYSTEM, _MARKED_Q2],
            True,
            [_MARKED_SYSTEM, _MARKED_Q2],
            [],
            id="fully-marked",
        ),
        pytest.param(
            [_SYSTEM, _MARKED_Q2],
            True,
            [_SYSTEM, _MARKED_Q2],
            [],
            id="one-marked-turn",
        ),
        pytest.param(
            [_MARKED_Q2],
            True,
            [_MARKED_Q2],
            [],
            id="one-marked-turn-without-system",
        ),
        pytest.param(
            [_Q1, _A1, _MARKED_Q2],
            True,
            [],
            [_Q1, _A1, _MARKED_Q2],
            id="marked-latest-turn-without-system",
        ),
        pytest.param(
            [_Q1, _A1, _MARKED_Q2, _A2, _Q3],
            True,
            [_Q1, _A1, _MARKED_Q2],
            [_A2, _Q3],
            id="marked-middle-turn",
        ),
        pytest.param(
            [_SYSTEM, _Q1, _A1, _MARKED_Q2, _A2, _Q3],
            False,
            [_SYSTEM],
            [_Q1, _A1, _MARKED_Q2, _A2, _Q3],
            id="marked-middle-turn-the-cache-does-not-accept",
        ),
        pytest.param(
            [_MARKED_SYSTEM, _Q1],
            True,
            [_MARKED_SYSTEM],
            [_Q1],
            id="marked-system",
        ),
    ],
)
def test_separate_cached_messages_caches_prefix_through_marked_block(
    messages: list[AllMessageValues],
    cache_accepts: bool,
    expected_cached: list[AllMessageValues],
    expected_non_cached: list[AllMessageValues],
) -> None:
    from litellm.llms.vertex_ai.context_caching.transformation import separate_cached_messages

    cached_messages, non_cached_messages = separate_cached_messages(messages, lambda _: cache_accepts)

    assert cached_messages == expected_cached
    assert non_cached_messages == expected_non_cached


def test_cached_messages_end_on_supported_turn():
    from litellm.llms.vertex_ai.context_caching.transformation import (
        cached_messages_end_on_supported_turn,
    )

    assert (
        cached_messages_end_on_supported_turn(
            [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "hello"}]
        )
        is True
    )
    assert cached_messages_end_on_supported_turn([{"role": "system", "content": "be brief"}]) is True
    assert cached_messages_end_on_supported_turn([{"role": "assistant", "content": "hi"}]) is False
    assert (
        cached_messages_end_on_supported_turn(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "system", "content": "be brief"},
            ]
        )
        is False
    )
    assert (
        cached_messages_end_on_supported_turn(
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
        )
        is True
    )
    assert (
        cached_messages_end_on_supported_turn([{"role": "tool", "tool_call_id": "x", "content": "y"}])
        is False
    )
    assert (
        cached_messages_end_on_supported_turn([{"role": "function", "name": "f", "content": "y"}])
        is False
    )
    assert cached_messages_end_on_supported_turn([]) is False


class TestCheckCachePagination:
    """Test pagination logic in check_cache and async_check_cache methods."""

    def setup_method(self):
        """Setup for each test method"""
        self.context_caching = ContextCachingEndpoints()
        self.mock_logging = MagicMock(spec=Logging)
        self.mock_logging.model_call_details = {}
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
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "nonexistent_cache_key"

        def create_page_response(page_num):
            response = MagicMock()
            response.json.return_value = {
                "cachedContents": [
                    {"name": f"cache_{page_num}", "displayName": f"key_{page_num}"}
                ],
                "nextPageToken": f"token_page_{page_num + 1}",
            }
            return response

        self.mock_client.get.side_effect = [
            create_page_response(i) for i in range(100)
        ]

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

        assert result is None
        assert self.mock_client.get.call_count == 100

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "custom_llm_provider", ["gemini", "vertex_ai", "vertex_ai_beta"]
    )
    @patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching")
    async def test_async_check_cache_pagination_max_pages_limit(
        self, mock_get_token_url, custom_llm_provider
    ):
        mock_get_token_url.return_value = ("token", "https://test-url.com")
        cache_key_to_find = "nonexistent_cache_key"

        def create_page_response(page_num):
            response = MagicMock()
            response.json.return_value = {
                "cachedContents": [
                    {"name": f"cache_{page_num}", "displayName": f"key_{page_num}"}
                ],
                "nextPageToken": f"token_page_{page_num + 1}",
            }
            return response

        self.mock_async_client.get = AsyncMock(
            side_effect=[create_page_response(i) for i in range(100)]
        )

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

        assert result is None
        assert self.mock_async_client.get.call_count == 100




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


def _created_cache_body(total_token_count: int) -> dict:
    return {
        "name": "cachedContents/created",
        "model": "models/gemini-3.1-pro",
        "usageMetadata": {"totalTokenCount": total_token_count},
    }


def _cacheable_messages() -> list:
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "cacheable prefix " * 1200, "cache_control": {"type": "ephemeral"}}
            ],
        },
        {"role": "user", "content": "question"},
    ]


@patch.object(ContextCachingEndpoints, "check_cache", return_value=None)
@patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching", return_value=("token", "https://u"))
def test_check_and_create_cache_records_creation_tokens_for_billing(mock_url, mock_check):
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = MagicMock(json=lambda: _created_cache_body(1280))

    _, _, cache_name = ContextCachingEndpoints().check_and_create_cache(
        messages=_cacheable_messages(),
        optional_params={},
        api_key="k",
        api_base=None,
        model="gemini-3.1-pro",
        client=client,
        timeout=30.0,
        logging_obj=logging_obj,
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )

    assert cache_name == "cachedContents/created"
    assert logging_obj.vertex_ai_cache_creation_input_tokens == 1280


@patch.object(ContextCachingEndpoints, "check_cache", return_value="cachedContents/existing")
@patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching", return_value=("token", "https://u"))
def test_check_and_create_cache_records_no_creation_tokens_when_cache_exists(mock_url, mock_check):
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.vertex_ai_cache_creation_input_tokens = 0
    client = MagicMock(spec=HTTPHandler)

    _, _, cache_name = ContextCachingEndpoints().check_and_create_cache(
        messages=_cacheable_messages(),
        optional_params={},
        api_key="k",
        api_base=None,
        model="gemini-3.1-pro",
        client=client,
        timeout=30.0,
        logging_obj=logging_obj,
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )

    assert cache_name == "cachedContents/existing"
    client.post.assert_not_called()
    assert logging_obj.vertex_ai_cache_creation_input_tokens == 0


@pytest.mark.asyncio
@patch.object(ContextCachingEndpoints, "async_check_cache", return_value=None)
@patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching", return_value=("token", "https://u"))
async def test_async_check_and_create_cache_records_creation_tokens_for_billing(mock_url, mock_check):
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    client = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=MagicMock(json=lambda: _created_cache_body(2048)))

    _, _, cache_name = await ContextCachingEndpoints().async_check_and_create_cache(
        messages=_cacheable_messages(),
        optional_params={},
        api_key="k",
        api_base=None,
        model="gemini-3.1-pro",
        client=client,
        timeout=30.0,
        logging_obj=logging_obj,
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
        vertex_auth_header=None,
    )

    assert cache_name == "cachedContents/created"
    assert logging_obj.vertex_ai_cache_creation_input_tokens == 2048


def test_check_and_create_cache_clamps_negative_creation_tokens() -> None:
    logging_obj = MagicMock(spec=Logging)
    logging_obj.model_call_details = {}
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = MagicMock(json=lambda: _created_cache_body(-500))

    with (
        patch.object(ContextCachingEndpoints, "check_cache", return_value=None),
        patch.object(ContextCachingEndpoints, "_get_token_and_url_context_caching", return_value=("token", "https://u")),
    ):
        _, _, cache_name = ContextCachingEndpoints().check_and_create_cache(
            messages=_cacheable_messages(),
            optional_params={},
            api_key="k",
            api_base=None,
            model="gemini-3.1-pro",
            client=client,
            timeout=30.0,
            logging_obj=logging_obj,
            custom_llm_provider="gemini",
            vertex_project=None,
            vertex_location=None,
            vertex_auth_header=None,
        )

    assert cache_name == "cachedContents/created"
    assert logging_obj.vertex_ai_cache_creation_input_tokens == 0


def _caches_created_so_far(calls: list[httpx.Request]) -> dict[str, list[dict[str, str]]]:
    return {
        "cachedContents": [
            {"name": "cachedContents/created", "displayName": json.loads(call.content)["displayName"]}
            for call in calls
            if call.method == "POST" and call.url.path.endswith("/cachedContents")
        ]
    }


def _recording_gemini_route(calls: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def route(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/cachedContents"):
            return httpx.Response(
                200, json=_caches_created_so_far(calls) if request.method == "GET" else _created_cache_body(3000)
            )
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 3000, "candidatesTokenCount": 1, "totalTokenCount": 3001},
            },
        )

    return route


def _recording_gemini_client(calls: list[httpx.Request]) -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_recording_gemini_route(calls))))


async def _gemini_calls(
    messages: list[AllMessageValues], use_async: bool, model: str = "gemini/gemini-flash-latest"
) -> list[httpx.Request]:
    calls: Final[list[httpx.Request]] = []
    transport: Final = httpx.MockTransport(_recording_gemini_route(calls))
    if use_async:
        await litellm.acompletion(
            model=model,
            messages=messages,
            api_key="test-key",
            client=AsyncHTTPHandler(transport=transport),
        )
    else:
        litellm.completion(
            model=model,
            messages=messages,
            api_key="test-key",
            client=HTTPHandler(client=httpx.Client(transport=transport)),
        )
    return calls


def _cached_conversation_with_tool_arguments(arguments: str) -> list[Mapping[str, object]]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": "cached prefix " * 3000, "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": arguments}}
            ],
            "cache_control": {"type": "ephemeral"},
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny", "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": [{"type": "text", "text": "Thanks.", "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "Reply with OK only."},
    ]


async def _complete_sync(request: Mapping[str, object], calls: list[httpx.Request]) -> None:
    litellm.completion(**request, client=_recording_gemini_client(calls))


async def _complete_async(request: Mapping[str, object], calls: list[httpx.Request]) -> None:
    await litellm.acompletion(
        **request, client=AsyncHTTPHandler(transport=httpx.MockTransport(_recording_gemini_route(calls)))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [_complete_sync, _complete_async])
async def test_cache_creation_drops_tool_call_arguments_gemini_cannot_use_when_drop_params_is_set(
    local_model_cost_map: None, complete: Callable[[Mapping[str, object], list[httpx.Request]], Awaitable[None]]
) -> None:
    calls: Final[list[httpx.Request]] = []
    await complete(
        {
            "model": "gemini/gemini-flash-latest",
            "messages": _cached_conversation_with_tool_arguments('["Boston"]'),
            "api_key": "test-key",
            "drop_params": True,
        },
        calls,
    )

    created: Final = next(
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    function_call: Final = created["contents"][1]["parts"][0]["function_call"]
    assert (function_call["name"], function_call["args"]) == ("get_weather", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [_complete_sync, _complete_async])
async def test_a_strict_request_rejects_tool_call_arguments_even_when_a_lenient_request_cached_that_history(
    local_model_cost_map: None, complete: Callable[[Mapping[str, object], list[httpx.Request]], Awaitable[None]]
) -> None:
    calls: Final[list[httpx.Request]] = []
    request: Final = {
        "model": "gemini/gemini-flash-latest",
        "messages": _cached_conversation_with_tool_arguments('["Boston"]'),
        "api_key": "test-key",
    }
    await complete({**request, "drop_params": True}, calls)

    with pytest.raises(litellm.BadRequestError, match="get_weather"):
        await complete({**request, "drop_params": False}, calls)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_mark_on_the_latest_turn_sends_the_whole_conversation_uncached_in_order(
    local_model_cost_map: None, use_async: bool
) -> None:
    calls: Final = await _gemini_calls(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What's the weather in Tokyo today?"},
            {"role": "assistant", "content": "Sunny."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And Nagoya tomorrow? " * 3000, "cache_control": {"type": "ephemeral"}},
                ],
            },
        ],
        use_async,
    )

    generated: Final = next(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert not any(call.url.path.endswith("/cachedContents") for call in calls)
    assert "cachedContent" not in generated
    assert generated["system_instruction"] == {"parts": [{"text": "You are terse."}]}
    assert [content["role"] for content in generated["contents"]] == ["user", "model", "user"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_reminder_after_the_marked_latest_turn_still_sends_the_conversation_uncached_in_order(
    local_model_cost_map, use_async: bool
) -> None:
    calls: Final = await _gemini_calls(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What's the weather in Tokyo today?"},
            {"role": "assistant", "content": "Sunny."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "And Nagoya tomorrow? " * 3000, "cache_control": {"type": "ephemeral"}},
                ],
            },
            {"role": "system", "content": "2 of 8 turns used"},
        ],
        use_async,
    )

    generated: Final = next(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert not any(call.url.path.endswith("/cachedContents") for call in calls)
    assert "cachedContent" not in generated
    assert generated["system_instruction"] == {"parts": [{"text": "You are terse."}]}
    assert generated["contents"][-1]["parts"][-1] == {"text": "2 of 8 turns used"}
    assert [content["role"] for content in generated["contents"]] == ["user", "model", "user"]


async def _complete_each_turn(
    turns: Sequence[list[AllMessageValues]], use_async: bool, tools: list[ChatCompletionToolParam] | None = None
) -> list[httpx.Request]:
    calls: Final[list[httpx.Request]] = []
    transport: Final = httpx.MockTransport(_recording_gemini_route(calls))
    for messages in turns:
        if use_async:
            await litellm.acompletion(
                model="gemini/gemini-flash-latest",
                messages=messages,
                tools=tools,
                api_key="test-key",
                client=AsyncHTTPHandler(transport=transport),
            )
        else:
            litellm.completion(
                model="gemini/gemini-flash-latest",
                messages=messages,
                tools=tools,
                api_key="test-key",
                client=HTTPHandler(client=httpx.Client(transport=transport)),
            )
    return calls


def _created_caches(calls: Sequence[httpx.Request]) -> list[dict[str, object]]:
    return [
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    ]


def _generate_requests(calls: Sequence[httpx.Request]) -> list[dict[str, object]]:
    return [json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent")]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_mark_on_each_newest_turn_caches_the_system_prompt_from_the_second_turn_and_the_third_turn_reads_it(
    local_model_cost_map, use_async: bool
) -> None:
    system: Final[ChatCompletionSystemMessage] = {"role": "system", "content": "house rules " * 3000}
    first_turn: Final[list[AllMessageValues]] = [
        {"role": "user", "content": "What's the weather in Tokyo today?"},
        {"role": "assistant", "content": "Sunny."},
    ]
    second_turn: Final[list[AllMessageValues]] = [
        {"role": "user", "content": "And Nagoya tomorrow?"},
        {"role": "assistant", "content": "Cloudy."},
    ]

    calls: Final = await _complete_each_turn(
        (
            [system, _marked_user("What's the weather in Tokyo today?")],
            [system, *first_turn, _marked_user("And Nagoya tomorrow?")],
            [system, *first_turn, *second_turn, _marked_user("And Osaka on Friday?")],
        ),
        use_async,
    )

    created: Final = _created_caches(calls)
    generated: Final = _generate_requests(calls)
    assert [cache["system_instruction"] for cache in created] == [{"parts": [{"text": "house rules " * 3000}]}] * 2
    assert "Tokyo" in json.dumps(created[0]["contents"])
    assert "Tokyo" not in json.dumps(created[1].get("contents", []))
    assert [request["cachedContent"] for request in generated] == ["cachedContents/created"] * 3
    assert [[content["role"] for content in request["contents"]] for request in generated[1:]] == [
        ["user", "model", "user"],
        ["user", "model", "user", "model", "user"],
    ]
    assert generated[2]["contents"][-1]["parts"] == [{"text": "And Osaka on Friday?"}]


@pytest.mark.parametrize(
    "system_prompt", ["You are terse.", "house rules " * 3000], ids=["short-system-prompt", "big-system-prompt"]
)
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_repeat_of_one_marked_turn_under_an_unmarked_system_prompt_reads_the_gemini_cache_it_made(
    local_model_cost_map, system_prompt: str, use_async: bool
) -> None:
    messages: Final[list[AllMessageValues]] = [
        {"role": "system", "content": system_prompt},
        _marked_user("contract clause " * 3000),
    ]

    calls: Final = await _complete_each_turn((messages, messages), use_async)

    created: Final = _created_caches(calls)
    generated: Final = _generate_requests(calls)
    assert len(created) == 1
    assert created[0]["system_instruction"] == {"parts": [{"text": system_prompt}]}
    assert "contract clause" in json.dumps(created[0]["contents"])
    assert [request["cachedContent"] for request in generated] == ["cachedContents/created"] * 2
    assert all("contract clause" not in json.dumps(request["contents"]) for request in generated)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_one_marked_turn_without_a_system_prompt_goes_into_the_gemini_cache_beside_tools_that_clear_the_floor(
    local_model_cost_map, use_async: bool
) -> None:
    tools: Final[list[ChatCompletionToolParam]] = [
        {
            "type": "function",
            "function": {
                "name": f"archive_lookup_{index}",
                "description": "Looks up one archive record. " * 200,
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}},
            },
        }
        for index in range(4)
    ]
    messages: Final[list[AllMessageValues]] = [_marked_user("Which archive holds the Tokyo station logs?")]

    calls: Final = await _complete_each_turn((messages, messages), use_async, tools=tools)

    created: Final = _created_caches(calls)
    generated: Final = _generate_requests(calls)
    assert len(created) == 1
    assert "Tokyo station logs" in json.dumps(created[0]["contents"])
    assert [request["cachedContent"] for request in generated] == ["cachedContents/created"] * 2
    assert all("Tokyo station logs" not in json.dumps(request["contents"]) for request in generated)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_repeat_of_a_marked_system_prompt_and_marked_turn_reads_one_gemini_cache_of_both_with_the_turn_ttl(
    local_model_cost_map, use_async: bool
) -> None:
    messages: Final[list[AllMessageValues]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "house rules " * 3000, "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Which archive holds the Tokyo station logs?",
                    "cache_control": {"type": "ephemeral", "ttl": "300s"},
                }
            ],
        },
    ]

    calls: Final = await _complete_each_turn((messages, messages), use_async)

    created: Final = _created_caches(calls)
    generated: Final = _generate_requests(calls)
    assert len(created) == 1
    assert created[0]["ttl"] == "300s"
    assert "Tokyo station logs" in json.dumps(created[0]["contents"])
    assert [request["cachedContent"] for request in generated] == ["cachedContents/created"] * 2
    assert all("Tokyo station logs" not in json.dumps(request.get("contents", [])) for request in generated)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_reminder_inside_the_cached_prefix_stays_in_its_place_in_the_gemini_cache(
    local_model_cost_map, use_async: bool
) -> None:
    calls: Final = await _gemini_calls(
        [
            {"role": "system", "content": "shared policy " * 3000},
            {"role": "user", "content": "question 1"},
            {"role": "system", "content": "1 of 8 turns used"},
            {"role": "assistant", "content": "answer 1"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "question 2", "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "assistant", "content": "answer 2"},
            {"role": "user", "content": "question 3"},
        ],
        use_async,
    )

    created: Final = next(
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    generated: Final = next(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert created["system_instruction"] == {"parts": [{"text": "shared policy " * 3000}]}
    assert created["contents"] == [
        {
            "role": "user",
            "parts": [{"text": "question 1"}, {"text": CONVERTED_SYSTEM_NOTE}, {"text": "1 of 8 turns used"}],
        },
        {"role": "model", "parts": [{"text": "answer 1"}]},
        {"role": "user", "parts": [{"text": "question 2"}]},
    ]
    assert generated["cachedContent"] == "cachedContents/created"
    assert generated["contents"] == [
        {"role": "model", "parts": [{"text": "answer 2"}]},
        {"role": "user", "parts": [{"text": "question 3"}]},
    ]


_SERVED_IMAGE: Final = b"\x89PNG\r\n\x1a\nserved"


def _serve_images(monkeypatch: pytest.MonkeyPatch, status_code: int) -> list[str]:
    fetched: Final[list[str]] = []

    def serve(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        return httpx.Response(status_code, content=_SERVED_IMAGE, headers={"content-type": "image/png"})

    transport: Final = httpx.MockTransport(serve)
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(litellm, "module_level_client", HTTPHandler(client=httpx.Client(transport=transport)))
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler(transport=transport))
    return fetched


@pytest.fixture
def served_images(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return _serve_images(monkeypatch, 200)


@pytest.fixture
def missing_images(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return _serve_images(monkeypatch, 404)


_CACHED_NOTES: Final[AllMessageValues] = {
    "role": "user",
    "content": [{"type": "text", "text": "Background notes. " * 3000, "cache_control": {"type": "ephemeral"}}],
}


def _conversation_cached_through_notes(*opening: AllMessageValues) -> list[AllMessageValues]:
    return [
        *opening,
        _CACHED_NOTES,
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "What color was the button?"},
    ]


def _button_question(image_url: ChatCompletionImageUrlObject) -> AllMessageValues:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "What does this button say?"},
            {"type": "image_url", "image_url": image_url},
        ],
    }


def _button_link_opening(url: str) -> tuple[AllMessageValues, ...]:
    return _button_question({"url": url}), {"role": "assistant", "content": "Deploy."}


def _system_link_opening(url: str) -> tuple[AllMessageValues, ...]:
    return (
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Match our house style."},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
    )


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_web_image_before_the_mark_reaches_the_gemini_cache_inlined(
    local_model_cost_map: None, served_images: list[str], use_async: bool
) -> None:
    image_url: Final = f"https://images.example.com/button-{use_async}.png"
    calls: Final = await _gemini_calls(
        _conversation_cached_through_notes(
            _button_question({"url": image_url}), {"role": "assistant", "content": "Deploy."}
        ),
        use_async,
    )

    created: Final = next(
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    generated: Final = next(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert served_images == [image_url]
    assert created["contents"][0]["parts"][1] == {
        "inline_data": {"data": base64.b64encode(_SERVED_IMAGE).decode(), "mime_type": "image/png"}
    }
    assert generated["cachedContent"] == "cachedContents/created"
    assert [content["role"] for content in generated["contents"]] == ["model", "user"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_web_image_the_proxy_cannot_fetch_fails_the_gemini_request_after_one_fetch(
    local_model_cost_map: None, missing_images: list[str], use_async: bool
) -> None:
    image_url: Final = f"https://images.example.com/missing-{use_async}.png"

    with pytest.raises(litellm.BadRequestError):
        await _gemini_calls(
            _conversation_cached_through_notes(
                _button_question({"url": image_url}), {"role": "assistant", "content": "Deploy."}
            ),
            use_async,
        )

    assert missing_images == [image_url]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_an_image_in_a_cached_system_message_is_never_downloaded(
    local_model_cost_map: None, missing_images: list[str], use_async: bool
) -> None:
    calls: Final = await _gemini_calls(
        _conversation_cached_through_notes(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are terse."},
                    {"type": "image_url", "image_url": {"url": f"https://images.example.com/logo-{use_async}.png"}},
                ],
            }
        ),
        use_async,
    )

    created: Final = next(
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    generated: Final = next(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert missing_images == []
    assert created["system_instruction"] == {"parts": [{"text": "You are terse."}]}
    assert generated["cachedContent"] == "cachedContents/created"
    assert [content["role"] for content in generated["contents"]] == ["model", "user"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_system_image_is_inlined_into_the_gemini_cache_when_the_system_prompt_is_sent_as_a_turn(
    local_model_cost_map: None, served_images: list[str], use_async: bool
) -> None:
    image_url: Final = f"https://images.example.com/house-style-{use_async}.png"
    calls: Final = await _gemini_calls(
        _conversation_cached_through_notes(*_system_link_opening(image_url)),
        use_async,
        model="gemini/house-style-tuned-model",
    )

    created: Final = next(
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    assert served_images == [image_url]
    assert "system_instruction" not in created
    assert created["contents"][0]["parts"][1] == {
        "inline_data": {"data": base64.b64encode(_SERVED_IMAGE).decode(), "mime_type": "image/png"}
    }


@pytest.mark.parametrize("api_version", ["v1beta", "v1"])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_files_api_file_before_the_mark_reaches_the_gemini_cache_as_a_link(
    local_model_cost_map: None, served_images: list[str], use_async: bool, api_version: str
) -> None:
    file_uri: Final = f"https://generativelanguage.googleapis.com/{api_version}/files/button-{use_async}"
    calls: Final = await _gemini_calls(
        _conversation_cached_through_notes(
            _button_question({"url": file_uri, "format": "image/png"}), {"role": "assistant", "content": "Deploy."}
        ),
        use_async,
    )

    created: Final = next(
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    assert served_images == []
    assert created["contents"][0]["parts"][1] == {"file_data": {"mime_type": "image/png", "file_uri": file_uri}}


async def _vertex_cache_check(
    messages: list[AllMessageValues],
    optional_params: dict[str, object],
    model: str,
    use_async: bool,
    litellm_params: Mapping[str, object] | None = None,
) -> tuple[tuple[list[AllMessageValues], dict, str | None], list[httpx.Request]]:
    calls: Final[list[httpx.Request]] = []
    transport: Final = httpx.MockTransport(_recording_gemini_route(calls))
    request: Final = {
        "messages": messages,
        "optional_params": optional_params,
        "api_key": "",
        "api_base": None,
        "model": model,
        "timeout": 30.0,
        "logging_obj": MagicMock(),
        "custom_llm_provider": "vertex_ai",
        "vertex_project": "test-project",
        "vertex_location": "us-central1",
        "vertex_auth_header": "test-token",
        "litellm_params": litellm_params,
    }
    result: Final = (
        await ContextCachingEndpoints().async_check_and_create_cache(
            client=AsyncHTTPHandler(transport=transport), **request
        )
        if use_async
        else ContextCachingEndpoints().check_and_create_cache(
            client=HTTPHandler(client=httpx.Client(transport=transport)), **request
        )
    )
    return result, calls


_PRIVATE_BUTTON_METADATA: Final = {"host": "storage.googleapis.com", "path": "/storage/v1/b/private-bucket/o/button"}


def _service_account_credentials() -> str:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "request-project",
            "private_key_id": "request-key",
            "private_key": key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ).decode(),
            "client_email": "cache@request-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_private_extensionless_gs_image_before_the_mark_is_typed_with_the_request_credentials(
    use_async: bool,
) -> None:
    with responses.RequestsMock() as google_auth, respx.mock(using="httpcore") as gcs:
        google_auth.add(
            responses.POST,
            "https://oauth2.googleapis.com/token",
            json={"access_token": "request-token", "expires_in": 3600, "token_type": "Bearer"},
        )
        metadata: Final = gcs.get(**_PRIVATE_BUTTON_METADATA).respond(json={"contentType": "image/png"})
        _, calls = await _vertex_cache_check(
            _conversation_cached_through_notes(
                _button_question({"url": "gs://private-bucket/button"}), {"role": "assistant", "content": "Deploy."}
            ),
            {},
            "gemini-2.5-flash",
            use_async,
            litellm_params={"vertex_project": "request-project", "vertex_credentials": _service_account_credentials()},
        )

    created: Final = next(
        json.loads(call.content)
        for call in calls
        if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    assert [call.request.headers.get("Authorization") for call in metadata.calls] == ["Bearer request-token"]
    assert created["contents"][0]["parts"][1] == {
        "file_data": {"mime_type": "image/png", "file_uri": "gs://private-bucket/button"}
    }


def _runs_on_an_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def test_an_async_gemini_cache_types_an_extensionless_gs_image_off_the_event_loop() -> None:
    lookups_on_the_loop: Final[list[bool]] = []

    def serve_metadata(request: httpx.Request) -> httpx.Response:
        lookups_on_the_loop.append(_runs_on_an_event_loop())
        return httpx.Response(200, json={"contentType": "image/png"})

    with respx.mock(using="httpcore") as gcs:
        gcs.get(**_PRIVATE_BUTTON_METADATA).mock(side_effect=serve_metadata)
        (_, _, cache_name), _ = await _vertex_cache_check(
            _conversation_cached_through_notes(
                _button_question({"url": "gs://private-bucket/button"}), {"role": "assistant", "content": "Deploy."}
            ),
            {},
            "gemini-2.5-flash",
            use_async=True,
        )

    assert cache_name == "cachedContents/created"
    assert lookups_on_the_loop == [False]


@pytest.mark.parametrize(
    ("model", "opening"),
    [
        ("gemini-2.5-flash", _button_link_opening("https://images.example.com/button.png")),
        (
            "gemini-2.5-flash",
            (
                _button_question({"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "format": "video/mp4"}),
                {"role": "assistant", "content": "Deploy."},
            ),
        ),
        (
            "gemini-2.5-flash",
            (
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize this spec."},
                        {"type": "file", "file": {"file_id": "https://docs.example.com/spec.pdf"}},
                    ],
                },
                {"role": "assistant", "content": "Deploy."},
            ),
        ),
        ("house-style-tuned-model", _system_link_opening("https://images.example.com/house-style.png")),
    ],
    ids=["image-link", "typed-video-link", "pdf-file-link", "system-image-link-sent-as-a-turn"],
)
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_web_link_vertex_fetches_itself_leaves_the_request_uncached_with_its_tools(
    local_model_cost_map: None,
    served_images: list[str],
    model: str,
    opening: tuple[AllMessageValues, ...],
    use_async: bool,
) -> None:
    messages: Final = _conversation_cached_through_notes(*opening)
    tools: Final = [{"function_declarations": [{"name": "get_weather", "parameters": {"type": "object"}}]}]

    result, calls = await _vertex_cache_check(messages, {"tools": tools}, model, use_async)

    assert result == (messages, {"tools": tools}, None)
    assert served_images == []
    assert calls == []


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_web_link_vertex_fetches_itself_leaves_the_turns_uncached_and_caches_the_system_prompt_alone(
    local_model_cost_map: None, served_images: list[str], use_async: bool
) -> None:
    system: Final[ChatCompletionSystemMessage] = {"role": "system", "content": "house rules " * 3000}
    turns: Final = _conversation_cached_through_notes(*_button_link_opening("https://images.example.com/button.png"))

    (uncached, _, cache_name), calls = await _vertex_cache_check([system, *turns], {}, "gemini-2.5-flash", use_async)

    created: Final = [json.loads(call.content) for call in calls if call.method == "POST"]
    assert cache_name == "cachedContents/created"
    assert uncached == turns
    assert [cache["system_instruction"] for cache in created] == [{"parts": [{"text": "house rules " * 3000}]}]
    assert "images.example.com" not in json.dumps(created)
    assert served_images == []


@pytest.mark.parametrize(
    ("opening_for", "url_template", "downloads"),
    [
        (_system_link_opening, "https://images.example.com/vertex-house-style-{}.png", 0),
        (_button_link_opening, "http://images.example.com/plain-button-{}.png", 1),
        (_button_link_opening, "https://images.example.com/untyped-button-{}", 1),
    ],
    ids=["system-image-link", "plain-http-image-link", "untyped-image-link"],
)
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_vertex_caches_a_prefix_whose_web_media_it_downloads_or_reads_as_instruction(
    local_model_cost_map: None,
    served_images: list[str],
    opening_for: Callable[[str], tuple[AllMessageValues, ...]],
    url_template: str,
    downloads: int,
    use_async: bool,
) -> None:
    url: Final = url_template.format(use_async)

    result, _ = await _vertex_cache_check(
        _conversation_cached_through_notes(*opening_for(url)), {}, "gemini-2.5-flash", use_async
    )

    assert result[2] == "cachedContents/created"
    assert served_images == [url] * downloads


@pytest.mark.parametrize("empty_content", [pytest.param("", id="empty_string"), pytest.param(None, id="null")])
def test_a_cached_segment_ending_on_an_empty_user_message_caches_no_blank_user_turn(empty_content: str | None) -> None:
    cached: Final = transform_openai_messages_to_gemini_context_caching(
        model="gemini-2.5-pro",
        messages=cast(
            list[AllMessageValues],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": empty_content},
            ],
        ),
        custom_llm_provider="gemini",
        cache_key="test-cache-key",
        ttl=None,
        vertex_project=None,
        vertex_location=None,
    )

    assert {"text": " "} not in chain.from_iterable(content["parts"] for content in cached["contents"])


_GET_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


@pytest.fixture
def search_with_functions_model(local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(
        litellm.model_cost, "gemini-3-search-next-to-functions", {"supports_search_tools_with_functions": True}
    )
    return "gemini/gemini-3-search-next-to-functions"


def _cached_request_with_search_and_functions(model: str) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "cached prefix " * 3000, "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": "Search the web, then reply with OK only."},
        ],
        "tools": [{"type": "web_search"}, _GET_WEATHER_TOOL],
        "api_key": "test-key",
    }


def _created_and_generated_bodies(calls: list[httpx.Request]) -> tuple[dict, dict]:
    created: Final = next(
        json.loads(call.content) for call in calls if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    generated: Final = next(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    return created, generated


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [_complete_sync, _complete_async])
async def test_search_kept_next_to_functions_goes_into_the_gemini_cache_with_its_server_side_tool_flag(
    search_with_functions_model: str,
    complete: Callable[[Mapping[str, object], list[httpx.Request]], Awaitable[None]],
) -> None:
    calls: Final[list[httpx.Request]] = []
    await complete(_cached_request_with_search_and_functions(search_with_functions_model), calls)

    created, generated = _created_and_generated_bodies(calls)
    assert {key for tool in created["tools"] for key in tool} == {"function_declarations", "googleSearch"}
    assert created["toolConfig"] == {"includeServerSideToolInvocations": True}
    assert generated["cachedContent"] == "cachedContents/created"
    assert "toolConfig" not in generated
    assert "tools" not in generated


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [_complete_sync, _complete_async])
async def test_a_caller_who_turns_server_side_tools_off_caches_functions_without_search(
    search_with_functions_model: str,
    complete: Callable[[Mapping[str, object], list[httpx.Request]], Awaitable[None]],
) -> None:
    calls: Final[list[httpx.Request]] = []
    await complete(
        {**_cached_request_with_search_and_functions(search_with_functions_model), "includeServerSideToolInvocations": False},
        calls,
    )

    created, generated = _created_and_generated_bodies(calls)
    assert {key for tool in created["tools"] for key in tool} == {"function_declarations"}
    assert "toolConfig" not in created
    assert "toolConfig" not in generated


_CACHED_PREFIX_WITH_A_REMINDER: Final[tuple[AllMessageValues, ...]] = (
    {
        "role": "system",
        "content": [{"type": "text", "text": "shared policy " * 3000, "cache_control": {"type": "ephemeral"}}],
    },
    {"role": "user", "content": [{"type": "text", "text": "question 1", "cache_control": {"type": "ephemeral"}}]},
    {"role": "system", "content": "1 of 8 turns used"},
)


def _later_turn_with_a_reminder(turn: int) -> tuple[AllMessageValues, ...]:
    return (
        {"role": "assistant", "content": f"answer {turn - 1}"},
        {"role": "user", "content": f"question {turn}"},
        {"role": "system", "content": f"{turn} of 8 turns used"},
    )


def _cached_conversation_with_a_reminder_after_each_user_turn(user_turns: int) -> list[AllMessageValues]:
    later_turns: Final = tuple(chain.from_iterable(map(_later_turn_with_a_reminder, range(2, user_turns + 1))))
    return [*_CACHED_PREFIX_WITH_A_REMINDER, *later_turns]


def test_a_reminder_after_the_cached_block_stays_out_of_the_gemini_cache_and_reaches_the_request(
    local_model_cost_map,
) -> None:
    calls: Final[list[httpx.Request]] = []
    for user_turns in (1, 2):
        litellm.completion(
            model="gemini/gemini-flash-latest",
            messages=_cached_conversation_with_a_reminder_after_each_user_turn(user_turns),
            api_key="test-key",
            client=_recording_gemini_client(calls),
        )

    created: Final = tuple(
        json.loads(call.content) for call in calls if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    generated: Final = tuple(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert len(created) == 1
    assert created[0]["system_instruction"] == {"parts": [{"text": "shared policy " * 3000}]}
    assert created[0]["contents"] == [{"role": "user", "parts": [{"text": "question 1"}]}]
    assert generated[1]["cachedContent"] == "cachedContents/created"
    assert "system_instruction" not in generated[1]
    assert generated[1]["contents"] == [
        {"role": "user", "parts": [{"text": CONVERTED_SYSTEM_NOTE}, {"text": "1 of 8 turns used"}]},
        {"role": "model", "parts": [{"text": "answer 1"}]},
        {"role": "user", "parts": [{"text": "question 2"}, {"text": CONVERTED_SYSTEM_NOTE}, {"text": "2 of 8 turns used"}]},
    ]


@pytest.mark.parametrize(
    "system_prompt",
    [
        pytest.param("You are terse.", id="short_system_prompt"),
        pytest.param("house rules " * 3000, id="big_system_prompt"),
    ],
)
def test_a_cache_mark_on_the_last_reminder_makes_no_gemini_cache_and_keeps_the_reminder_in_place(
    local_model_cost_map, system_prompt: str
) -> None:
    calls: Final[list[httpx.Request]] = []
    reminder: Final = "remaining budget " * 3000

    litellm.completion(
        model="gemini/gemini-flash-latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "question 1"},
            {"role": "system", "content": reminder},
        ],
        cache_control_injection_points=[{"location": "message", "index": -1}],
        api_key="test-key",
        client=_recording_gemini_client(calls),
    )

    generated: Final = next(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert [call for call in calls if call.url.path.endswith("/cachedContents")] == []
    assert generated["system_instruction"] == {"parts": [{"text": system_prompt}]}
    assert generated["contents"] == [
        {"role": "user", "parts": [{"text": "question 1"}, {"text": CONVERTED_SYSTEM_NOTE}, {"text": reminder}]}
    ]


def test_the_leading_system_message_goes_into_the_cache_and_a_later_reminder_comes_back_converted():
    from litellm.llms.vertex_ai.context_caching.transformation import separate_cached_messages

    leading: Final = {"role": "system", "content": "You are terse."}
    cached_doc: Final = {
        "role": "user",
        "content": [{"type": "text", "text": "contract text", "cache_control": {"type": "ephemeral"}}],
    }

    cached, remaining = separate_cached_messages(
        [leading, cached_doc, {"role": "system", "content": "1 of 8 turns used"}, {"role": "user", "content": "question 2"}],
        lambda _: True,
    )

    assert cached == [leading, cached_doc]
    assert remaining == [
        {
            "role": "user",
            "content": [{"type": "text", "text": CONVERTED_SYSTEM_NOTE}, {"type": "text", "text": "1 of 8 turns used"}],
        },
        {"role": "user", "content": "question 2"},
    ]


def test_a_mark_inside_the_leading_system_run_caches_the_whole_run():
    from litellm.llms.vertex_ai.context_caching.transformation import separate_cached_messages

    run: Final = [
        {"role": "system", "content": "You are terse."},
        {"role": "system", "content": [{"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}}]},
        {"role": "system", "content": "Answer in English."},
    ]
    question: Final = {"role": "user", "content": "question 1"}

    cached, remaining = separate_cached_messages([*run, question], lambda _: True)

    assert cached == run
    assert remaining == [question]


def _cached_question_then_a_reminder_marked(
    reminder_mark: Mapping[str, str], user_turns: int
) -> list[AllMessageValues]:
    later_turns: Final = tuple(chain.from_iterable(map(_later_turn_with_a_reminder, range(2, user_turns + 1))))
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "shared policy " * 3000, "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "user", "content": [{"type": "text", "text": "question 1", "cache_control": {"type": "ephemeral"}}]},
        {"role": "system", "content": [{"type": "text", "text": "1 of 8 turns used", "cache_control": reminder_mark}]},
        *later_turns,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [_complete_sync, _complete_async])
async def test_a_ttl_on_a_reminder_that_ends_the_cached_block_still_sets_how_long_the_gemini_cache_lives(
    local_model_cost_map: None, complete: Callable[[Mapping[str, object], list[httpx.Request]], Awaitable[None]]
) -> None:
    calls: Final[list[httpx.Request]] = []
    for user_turns in (1, 2):
        await complete(
            {
                "model": "gemini/gemini-flash-latest",
                "messages": _cached_question_then_a_reminder_marked({"type": "ephemeral", "ttl": "5m"}, user_turns),
                "api_key": "test-key",
            },
            calls,
        )

    created: Final = tuple(
        json.loads(call.content) for call in calls if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    generated: Final = tuple(json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert [(body["contents"], body.get("ttl")) for body in created] == [
        ([{"role": "user", "parts": [{"text": "question 1"}]}], "300s")
    ]
    assert [request["cachedContent"] for request in generated] == ["cachedContents/created"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [_complete_sync, _complete_async])
async def test_a_reminder_ttl_gets_its_own_gemini_cache_beside_one_made_without_a_ttl(
    local_model_cost_map: None, complete: Callable[[Mapping[str, object], list[httpx.Request]], Awaitable[None]]
) -> None:
    calls: Final[list[httpx.Request]] = []
    for reminder_mark in ({"type": "ephemeral"}, {"type": "ephemeral", "ttl": "5m"}):
        await complete(
            {
                "model": "gemini/gemini-flash-latest",
                "messages": _cached_question_then_a_reminder_marked(reminder_mark, 2),
                "api_key": "test-key",
            },
            calls,
        )

    created: Final = tuple(
        json.loads(call.content) for call in calls if call.method == "POST" and call.url.path.endswith("/cachedContents")
    )
    assert [body.get("ttl") for body in created] == [None, "300s"]
    assert created[0]["displayName"] != created[1]["displayName"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_ttl_on_a_later_mark_in_the_system_prompt_sets_how_long_the_gemini_cache_lives(
    local_model_cost_map, use_async: bool
) -> None:
    calls: Final = await _gemini_calls(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "house rules " * 3000, "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "system", "content": "Answer in English."},
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral", "ttl": "600s"}}
                ],
            },
            {"role": "user", "content": "question 1"},
        ],
        use_async,
    )

    assert [body.get("ttl") for body in _created_caches(calls)] == ["600s"]


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_a_mark_on_a_model_turn_caches_the_system_prompt_alone_and_sends_the_turns_uncached(
    local_model_cost_map, use_async: bool
) -> None:
    calls: Final = await _gemini_calls(
        [
            {"role": "system", "content": "house rules " * 3000},
            {"role": "user", "content": "question 1"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "answer 1", "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": "question 2"},
        ],
        use_async,
    )

    created: Final = _created_caches(calls)
    generated: Final = _generate_requests(calls)
    assert len(created) == 1
    assert created[0]["system_instruction"] == {"parts": [{"text": "house rules " * 3000}]}
    assert "answer 1" not in json.dumps(created[0]["contents"])
    assert [request["cachedContent"] for request in generated] == ["cachedContents/created"]
    assert generated[0]["contents"] == [
        {"role": "user", "parts": [{"text": "question 1"}]},
        {"role": "model", "parts": [{"text": "answer 1"}]},
        {"role": "user", "parts": [{"text": "question 2"}]},
    ]


def test_a_cached_block_from_a_later_turn_that_ends_on_a_reminder_goes_out_uncached_and_in_order(
    local_model_cost_map,
) -> None:
    calls: Final[list[httpx.Request]] = []
    litellm.completion(
        model="gemini/gemini-flash-latest",
        messages=[
            {"role": "system", "content": "shared policy"},
            {"role": "user", "content": "question 1"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "answer 1 " * 3000, "cache_control": {"type": "ephemeral"}}],
            },
            {
                "role": "system",
                "content": [{"type": "text", "text": "1 of 8 turns used", "cache_control": {"type": "ephemeral"}}],
            },
            {"role": "user", "content": "question 2"},
        ],
        api_key="test-key",
        client=_recording_gemini_client(calls),
    )

    assert not any(call.url.path.endswith("/cachedContents") for call in calls)
    (generated,) = (json.loads(call.content) for call in calls if call.url.path.endswith(":generateContent"))
    assert generated["system_instruction"] == {"parts": [{"text": "shared policy"}]}
    assert generated["contents"] == [
        {"role": "user", "parts": [{"text": "question 1"}]},
        {"role": "model", "parts": [{"text": "answer 1 " * 3000}]},
        {"role": "user", "parts": [{"text": CONVERTED_SYSTEM_NOTE}, {"text": "1 of 8 turns used"}, {"text": "question 2"}]},
    ]


def test_a_video_in_a_system_message_after_the_cached_block_is_a_bad_request_before_any_cache_call(
    local_model_cost_map,
) -> None:
    calls: Final[list[httpx.Request]] = []
    messages: Final[list[AllMessageValues]] = [
        *_cached_conversation_with_a_reminder_after_each_user_turn(1),
        {"role": "assistant", "content": "answer 1"},
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Watch this first"},
                {"type": "video_url", "video_url": {"url": "https://videos.example/clip.mp4"}},
            ],
        },
        {"role": "user", "content": "Describe the clip"},
    ]

    with pytest.raises(litellm.BadRequestError, match="system instructions take text only"):
        litellm.completion(
            model="gemini/gemini-flash-latest",
            messages=messages,
            api_key="test-key",
            client=_recording_gemini_client(calls),
        )
    assert calls == []


def _last_generated_contents(calls: list[httpx.Request]) -> list[object]:
    return next(json.loads(call.content) for call in reversed(calls) if call.url.path.endswith(":generateContent"))[
        "contents"
    ]


def _last_gemini_request_contents(messages: list[AllMessageValues], model: str) -> object:
    calls: Final[list[httpx.Request]] = []
    litellm.completion(model=model, messages=messages, api_key="test-key", client=_recording_gemini_client(calls))
    return _last_generated_contents(calls)[-1]


def test_a_model_without_system_messages_gets_a_later_reminder_the_same_way_with_or_without_a_cache(
    local_model_cost_map,
) -> None:
    later_turns: Final[list[AllMessageValues]] = [
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "question 2"},
        {"role": "system", "content": "2 of 8 turns used"},
    ]
    cached: Final[list[AllMessageValues]] = [
        {"role": "user", "content": [{"type": "text", "text": "shared policy " * 3000, "cache_control": {"type": "ephemeral"}}]},
        *later_turns,
    ]
    uncached: Final[list[AllMessageValues]] = [{"role": "user", "content": "shared policy " * 3000}, *later_turns]

    with_cache: Final = _last_gemini_request_contents(cached, "gemini/gemma-3-27b-it")

    assert with_cache == _last_gemini_request_contents(uncached, "gemini/gemma-3-27b-it")
    assert with_cache == {"role": "user", "parts": [{"text": "question 2"}, {"text": "2 of 8 turns used"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [_complete_sync, _complete_async])
async def test_a_model_without_system_messages_takes_a_video_in_a_later_system_message_the_same_way_with_or_without_a_cache(
    local_model_cost_map: None, complete: Callable[[Mapping[str, object], list[httpx.Request]], Awaitable[None]]
) -> None:
    later_turns: Final[list[AllMessageValues]] = [
        {"role": "assistant", "content": "answer 1"},
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Watch this first"},
                {"type": "video_url", "video_url": {"url": "gs://videos-example/clip.mp4"}},
            ],
        },
        {"role": "user", "content": "Describe the clip"},
    ]
    cached: Final[list[AllMessageValues]] = [
        {"role": "user", "content": [{"type": "text", "text": "shared policy " * 3000, "cache_control": {"type": "ephemeral"}}]},
        *later_turns,
    ]
    uncached: Final[list[AllMessageValues]] = [{"role": "user", "content": "shared policy " * 3000}, *later_turns]
    cached_calls: Final[list[httpx.Request]] = []
    uncached_calls: Final[list[httpx.Request]] = []

    await complete({"model": "gemini/gemma-3-27b-it", "messages": cached, "api_key": "test-key"}, cached_calls)
    await complete({"model": "gemini/gemma-3-27b-it", "messages": uncached, "api_key": "test-key"}, uncached_calls)

    assert _last_generated_contents(cached_calls)[-1] == _last_generated_contents(uncached_calls)[-1]
