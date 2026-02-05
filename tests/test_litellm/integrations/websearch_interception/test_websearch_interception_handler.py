"""
Unit tests for WebSearch Interception Handler

Tests the WebSearchInterceptionLogger class and helper functions.
"""

import asyncio
from unittest.mock import Mock


from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool,
    is_web_search_tool,
)
from litellm.litellm_core_utils.core_helpers import filter_internal_params
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


def test_async_should_run_agentic_loop_positive_case():
    """Test that agentic loop IS triggered when WebSearch tool_use in response"""

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

        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
            messages=[],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is True
        assert "tool_calls" in tools_dict
        assert len(tools_dict["tool_calls"]) == 1
        assert tools_dict["tool_calls"][0]["id"] == "tool_123"
        assert tools_dict["tool_type"] == "websearch"

    asyncio.run(_test())


def test_async_should_run_agentic_loop_includes_thinking_blocks():
    """Test that thinking blocks are captured in tools_dict"""

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

        should_run, tools_dict = await logger.async_should_run_agentic_loop(
            response=response,
            model="bedrock/claude",
            messages=[],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            stream=False,
            custom_llm_provider="bedrock",
            kwargs={},
        )

        assert should_run is True
        assert "thinking_blocks" in tools_dict
        assert len(tools_dict["thinking_blocks"]) == 1
        assert tools_dict["thinking_blocks"][0]["type"] == "thinking"

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


def test_internal_flags_filtered_from_followup_kwargs():
    """Test that internal _websearch_interception flags are filtered from follow-up request kwargs.

    Regression test for bug where _websearch_interception_converted_stream was passed
    to the follow-up LLM request, causing "Extra inputs are not permitted" errors
    from providers like Bedrock that use strict parameter validation.
    """

    # Simulate kwargs that would be passed during agentic loop execution
    kwargs_with_internal_flags = {
        "_websearch_interception_converted_stream": True,
        "_websearch_interception_other_flag": "test",
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    # Use the actual filter function from the codebase
    kwargs_for_followup = filter_internal_params(kwargs_with_internal_flags)

    # Verify internal flags are filtered out
    assert "_websearch_interception_converted_stream" not in kwargs_for_followup
    assert "_websearch_interception_other_flag" not in kwargs_for_followup

    # Verify regular kwargs are preserved
    assert kwargs_for_followup["temperature"] == 0.7
    assert kwargs_for_followup["max_tokens"] == 1024
