"""
Unit tests for WebSearch Interception Handler

Tests the WebSearchInterceptionLogger class and helper functions.
"""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.types.integrations.websearch_interception import (
    WebSearchInterceptionConfig,
)
from litellm.types.utils import LlmProviders


class TestWebSearchInterceptionLoggerInit:
    """Test WebSearchInterceptionLogger initialization"""

    def test_init_default_provider(self):
        """Test initialization with default provider (bedrock)"""
        logger = WebSearchInterceptionLogger()
        assert logger.enabled_providers == [LlmProviders.BEDROCK.value]
        assert logger.search_tool_name is None

    def test_init_with_string_providers(self):
        """Test initialization with string provider names"""
        logger = WebSearchInterceptionLogger(
            enabled_providers=["bedrock", "vertex_ai"]
        )
        assert "bedrock" in logger.enabled_providers
        assert "vertex_ai" in logger.enabled_providers

    def test_init_with_enum_providers(self):
        """Test initialization with LlmProviders enum values"""
        logger = WebSearchInterceptionLogger(
            enabled_providers=[LlmProviders.BEDROCK, LlmProviders.VERTEX_AI]
        )
        assert LlmProviders.BEDROCK.value in logger.enabled_providers
        assert LlmProviders.VERTEX_AI.value in logger.enabled_providers

    def test_init_with_search_tool_name(self):
        """Test initialization with custom search tool name"""
        logger = WebSearchInterceptionLogger(
            enabled_providers=["bedrock"],
            search_tool_name="my-custom-search",
        )
        assert logger.search_tool_name == "my-custom-search"


class TestWebSearchInterceptionLoggerFromConfigYaml:
    """Test from_config_yaml classmethod"""

    def test_from_config_yaml_minimal(self):
        """Test initialization from minimal config"""
        config: WebSearchInterceptionConfig = {}
        logger = WebSearchInterceptionLogger.from_config_yaml(config)
        # Should use defaults
        assert logger.enabled_providers == [LlmProviders.BEDROCK.value]
        assert logger.search_tool_name is None

    def test_from_config_yaml_with_providers(self):
        """Test initialization from config with providers"""
        config: WebSearchInterceptionConfig = {
            "enabled_providers": ["bedrock", "vertex_ai"]
        }
        logger = WebSearchInterceptionLogger.from_config_yaml(config)
        assert LlmProviders.BEDROCK.value in logger.enabled_providers
        assert LlmProviders.VERTEX_AI.value in logger.enabled_providers

    def test_from_config_yaml_with_search_tool(self):
        """Test initialization from config with search tool name"""
        config: WebSearchInterceptionConfig = {
            "enabled_providers": ["bedrock"],
            "search_tool_name": "my-perplexity-search",
        }
        logger = WebSearchInterceptionLogger.from_config_yaml(config)
        assert logger.search_tool_name == "my-perplexity-search"

    def test_from_config_yaml_invalid_provider(self):
        """Test initialization with invalid provider name (should keep as string)"""
        config: WebSearchInterceptionConfig = {
            "enabled_providers": ["bedrock", "invalid_provider"]
        }
        logger = WebSearchInterceptionLogger.from_config_yaml(config)
        # Should convert valid provider and keep invalid one as string
        assert LlmProviders.BEDROCK.value in logger.enabled_providers
        assert "invalid_provider" in logger.enabled_providers


class TestWebSearchInterceptionLoggerInitializeFromProxyConfig:
    """Test initialize_from_proxy_config static method"""

    def test_initialize_from_litellm_settings(self):
        """Test initialization from litellm_settings"""
        litellm_settings = {
            "websearch_interception_params": {
                "enabled_providers": ["bedrock"],
                "search_tool_name": "my-search",
            }
        }
        callback_specific_params = {}

        logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
            litellm_settings=litellm_settings,
            callback_specific_params=callback_specific_params,
        )

        assert LlmProviders.BEDROCK.value in logger.enabled_providers
        assert logger.search_tool_name == "my-search"

    def test_initialize_from_callback_specific_params(self):
        """Test initialization from callback_specific_params"""
        litellm_settings = {}
        callback_specific_params = {
            "websearch_interception": {
                "enabled_providers": ["vertex_ai"],
                "search_tool_name": "my-other-search",
            }
        }

        logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
            litellm_settings=litellm_settings,
            callback_specific_params=callback_specific_params,
        )

        assert LlmProviders.VERTEX_AI.value in logger.enabled_providers
        assert logger.search_tool_name == "my-other-search"

    def test_initialize_preference_for_litellm_settings(self):
        """Test that litellm_settings takes precedence over callback_specific_params"""
        litellm_settings = {
            "websearch_interception_params": {
                "enabled_providers": ["bedrock"],
            }
        }
        callback_specific_params = {
            "websearch_interception": {
                "enabled_providers": ["vertex_ai"],
            }
        }

        logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
            litellm_settings=litellm_settings,
            callback_specific_params=callback_specific_params,
        )

        # Should use litellm_settings
        assert LlmProviders.BEDROCK.value in logger.enabled_providers
        assert LlmProviders.VERTEX_AI.value not in logger.enabled_providers


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_wrong_provider():
    """Test that agentic loop is NOT triggered for non-enabled providers"""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    response = Mock()
    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="gpt-4",
        messages=[],
        tools=[{"name": "WebSearch"}],
        stream=False,
        custom_llm_provider="openai",  # Not in enabled_providers
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_no_websearch_tool():
    """Test that agentic loop is NOT triggered without WebSearch tool"""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    response = Mock()
    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="bedrock/claude",
        messages=[],
        tools=[{"name": "SomeOtherTool"}],  # No WebSearch
        stream=False,
        custom_llm_provider="bedrock",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_streaming():
    """Test that streaming requests with WebSearch are handled correctly"""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    # Mock response with no tool_use (streaming conversion happens earlier)
    response = Mock()
    response.content = [{"type": "text", "text": "Some response"}]

    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="bedrock/claude",
        messages=[],
        tools=[{"name": "WebSearch"}],
        stream=True,  # Streaming request
        custom_llm_provider="bedrock",
        kwargs={},
    )

    # Streaming is converted earlier, so no tool_use in response
    assert should_run is False


@pytest.mark.asyncio
async def test_create_empty_search_result():
    """Test _create_empty_search_result method"""
    logger = WebSearchInterceptionLogger()
    result = await logger._create_empty_search_result()
    assert result == "No search query provided"
    assert isinstance(result, str)
