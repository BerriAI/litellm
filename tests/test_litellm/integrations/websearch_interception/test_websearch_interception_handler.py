"""
Unit tests for WebSearch Interception Handler

Tests the WebSearchInterceptionLogger class and helper functions.
"""

import asyncio
from unittest.mock import Mock

import pytest

from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool,
    is_web_search_tool,
)
from litellm.types.utils import LlmProviders


class TestIsWebSearchTool:
    """Tests for is_web_search_tool() helper function"""

    def test_litellm_standard_tool(self):
        """Should detect LiteLLM standard web search tool"""
        tool = {"name": LITELLM_WEB_SEARCH_TOOL_NAME}
        assert is_web_search_tool(tool) is True

    def test_anthropic_native_web_search(self):
        """Should detect Anthropic native web_search_* type"""
        tool = {"type": "web_search_20250305", "name": "web_search"}
        assert is_web_search_tool(tool) is True

    def test_anthropic_native_future_version(self):
        """Should detect future versions of Anthropic web_search type"""
        tool = {"type": "web_search_20260101", "name": "web_search"}
        assert is_web_search_tool(tool) is True

    def test_claude_code_web_search(self):
        """Should detect Claude Code's web_search with type field"""
        tool = {"name": "web_search", "type": "web_search_20250305"}
        assert is_web_search_tool(tool) is True

    def test_legacy_websearch_format(self):
        """Should detect legacy WebSearch format"""
        tool = {"name": "WebSearch"}
        assert is_web_search_tool(tool) is True

    def test_non_websearch_tool(self):
        """Should not detect non-web-search tools"""
        assert is_web_search_tool({"name": "calculator"}) is False
        assert is_web_search_tool({"name": "read_file"}) is False
        assert is_web_search_tool({"type": "function", "name": "search"}) is False

    def test_web_search_name_without_type(self):
        """Should NOT detect 'web_search' name without type field (could be custom tool)"""
        tool = {"name": "web_search"}  # No type field
        assert is_web_search_tool(tool) is False


class TestGetLitellmWebSearchTool:
    """Tests for get_litellm_web_search_tool() helper function"""

    def test_returns_valid_tool_definition(self):
        """Should return a valid tool definition"""
        tool = get_litellm_web_search_tool()

        assert tool["name"] == LITELLM_WEB_SEARCH_TOOL_NAME
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "query" in tool["input_schema"]["properties"]


class TestWebSearchInterceptionLoggerInit:
    """Tests for WebSearchInterceptionLogger initialization"""

    def test_default_initialization(self):
        """Test default initialization with no parameters"""
        logger = WebSearchInterceptionLogger()

        # Default should have bedrock enabled
        assert "bedrock" in logger.enabled_providers
        assert logger.search_tool_name is None

    def test_custom_providers(self):
        """Test initialization with custom providers"""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock", "vertex_ai", "openai"])

        assert "bedrock" in logger.enabled_providers
        assert "vertex_ai" in logger.enabled_providers
        assert "openai" in logger.enabled_providers

    def test_custom_search_tool_name(self):
        """Test initialization with custom search tool name"""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"], search_tool_name="custom-search-tool")

        assert logger.search_tool_name == "custom-search-tool"

    def test_llm_providers_enum_conversion(self):
        """Test that LlmProviders enum values are converted to strings"""
        logger = WebSearchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK, LlmProviders.VERTEX_AI])

        # Should be stored as string values
        assert "bedrock" in logger.enabled_providers
        assert "vertex_ai" in logger.enabled_providers


def test_initialize_from_proxy_config():
    """Test initialization from proxy config with litellm_settings"""
    litellm_settings = {
        "websearch_interception_params": {
            "enabled_providers": ["bedrock", "vertex_ai"],
            "search_tool_name": "my-search",
        }
    }
    callback_specific_params = {}

    logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
        litellm_settings=litellm_settings,
        callback_specific_params=callback_specific_params,
    )

    assert LlmProviders.BEDROCK.value in logger.enabled_providers
    assert LlmProviders.VERTEX_AI.value in logger.enabled_providers
    assert logger.search_tool_name == "my-search"


def test_initialize_from_proxy_config_defaults():
    """Test initialization from proxy config with defaults when params missing"""
    litellm_settings = {}
    callback_specific_params = {}

    logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
        litellm_settings=litellm_settings,
        callback_specific_params=callback_specific_params,
    )

    # Should use default bedrock provider
    assert "bedrock" in logger.enabled_providers


def test_async_should_run_agentic_loop_wrong_provider():
    """Test that agentic loop is NOT triggered for wrong provider"""

    async def _test():
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = Mock()
        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="gpt-4",
            messages=[],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            stream=False,
            custom_llm_provider="openai",  # Not in enabled_providers
            kwargs={},
        )

        assert should_run is False
        assert tools_dict == {}

    asyncio.run(_test())


def test_async_should_run_agentic_loop_no_websearch_tool():
    """Test that agentic loop is NOT triggered when no WebSearch tool in request"""

    async def _test():
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = Mock()
        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[{"name": "calculator"}],  # No WebSearch tool
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is False
        assert tools_dict == {}

    asyncio.run(_test())


def test_async_should_run_agentic_loop_no_websearch_in_response():
    """Test that agentic loop is NOT triggered when response has no WebSearch tool_use"""

    async def _test():
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        # Response with text only, no tool_use
        response = {"content": [{"type": "text", "text": "I don't need to search for this."}]}

        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is False
        assert tools_dict == {}

    asyncio.run(_test())


def test_async_execute_tool_calls_positive_case():
    """Test that async_execute_tool_calls returns results for WebSearch tool_use"""
    from unittest.mock import AsyncMock, patch

    async def _test():
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        # Response with WebSearch tool_use
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "WebSearch",
                    "input": {"query": "weather in SF"},
                }
            ]
        }

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock, return_value="Sunny, 72F"
        ):
            results = await logger.async_execute_tool_calls(
                response=response,
                kwargs={"custom_llm_provider": "bedrock"},
            )

        assert len(results) == 1
        assert results[0].tool_call_id == "tool_123"
        assert results[0].content == "Sunny, 72F"
        assert results[0].is_error is False

    asyncio.run(_test())


def test_async_execute_tool_calls_with_thinking_blocks():
    """Test that async_execute_tool_calls works alongside thinking blocks in response"""
    from unittest.mock import AsyncMock, patch

    async def _test():
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        # Response with thinking block and WebSearch tool_use
        response = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me search for the weather...",
                },
                {
                    "type": "tool_use",
                    "id": "tool_456",
                    "name": "WebSearch",
                    "input": {"query": "current weather SF"},
                },
            ]
        }

        with patch.object(
            logger, "_execute_search", new_callable=AsyncMock, return_value="Cloudy, 60F"
        ):
            results = await logger.async_execute_tool_calls(
                response=response,
                kwargs={"custom_llm_provider": "bedrock"},
            )

        # Should return results for the tool_use block (thinking blocks are
        # handled by the framework, not the callback)
        assert len(results) == 1
        assert results[0].tool_call_id == "tool_456"
        assert results[0].content == "Cloudy, 60F"
        assert results[0].is_error is False

    asyncio.run(_test())


def test_async_should_run_agentic_loop_empty_tools_list():
    """Test with empty tools list"""

    async def _test():
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = Mock()
        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[],  # Empty tools list
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is False
        assert tools_dict == {}

    asyncio.run(_test())


def test_async_should_run_agentic_loop_none_tools():
    """Test with None tools"""

    async def _test():
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        response = Mock()
        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=None,  # None tools
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is False
        assert tools_dict == {}

    asyncio.run(_test())



@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_litellm_params_provider():
    """Test that async_pre_call_deployment_hook reads custom_llm_provider from litellm_params."""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "messages": [{"role": "user", "content": "Search the web for LiteLLM"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "function", "function": {"name": "other_tool", "parameters": {}}},
        ],
        "litellm_params": {"custom_llm_provider": "bedrock"},
        "api_key": "fake-key",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    # The web_search tool should be converted to litellm_web_search (Anthropic format)
    assert any(t.get("name") == "litellm_web_search" for t in result["tools"])
    # The non-web-search tool should be preserved
    assert any(
        t.get("type") == "function" and t.get("function", {}).get("name") == "other_tool"
        for t in result["tools"]
    )


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_returns_full_kwargs():
    """Test that async_pre_call_deployment_hook returns the full kwargs dict."""
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Search for something"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
        ],
        "litellm_params": {"custom_llm_provider": "openai"},
        "api_key": "sk-fake",
        "temperature": 0.7,
        "metadata": {"user": "test"},
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    assert result["model"] == "gpt-4o"
    assert result["messages"] == [{"role": "user", "content": "Search for something"}]
    assert result["api_key"] == "sk-fake"
    assert result["temperature"] == 0.7
    assert result["metadata"] == {"user": "test"}
    assert any(t.get("name") == "litellm_web_search" for t in result["tools"])


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_skips_disabled_provider():
    """Test that the hook returns None for providers not in enabled_providers."""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "litellm_params": {"custom_llm_provider": "openai"},  # Not in enabled_providers
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)
    assert result is None


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_skips_no_websearch_tools():
    """Test that the hook returns None when no web search tools are present."""
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}},
        ],
        "litellm_params": {"custom_llm_provider": "openai"},
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)
    assert result is None
