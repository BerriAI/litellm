"""
Unit tests for AnthropicWebSearchHook

Tests the hook that intercepts Anthropic web_search tool calls and routes them
through LiteLLM's search API for non-Anthropic providers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.integrations.anthropic_web_search_hook import AnthropicWebSearchHook
from litellm.types.utils import CallTypes


class TestAnthropicWebSearchHookDetection:
    """Tests for web_search tool detection."""

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return AnthropicWebSearchHook()

    def test_detect_web_search_tool_present(self, hook):
        """Test detection when web_search tool is present."""
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "custom", "name": "other_tool"},
        ]
        result = hook._detect_web_search_tool(tools)

        assert result is not None
        assert result["type"] == "web_search_20250305"
        assert result["name"] == "web_search"

    def test_detect_web_search_tool_not_present(self, hook):
        """Test detection when web_search tool is not present."""
        tools = [
            {"type": "custom", "name": "tool1"},
            {"type": "function", "name": "tool2"},
        ]
        result = hook._detect_web_search_tool(tools)
        assert result is None

    def test_detect_web_search_tool_empty_list(self, hook):
        """Test detection with empty tools list."""
        result = hook._detect_web_search_tool([])
        assert result is None

    def test_detect_web_search_tool_none(self, hook):
        """Test detection with None tools."""
        result = hook._detect_web_search_tool(None)
        assert result is None

    def test_detect_web_search_tool_with_max_uses(self, hook):
        """Test detection of web_search tool with max_uses parameter."""
        tools = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
                "user_location": {"type": "approximate", "country": "US"},
            },
        ]
        result = hook._detect_web_search_tool(tools)

        assert result is not None
        assert result["max_uses"] == 5
        assert result["user_location"]["country"] == "US"


class TestAnthropicWebSearchHookRemoval:
    """Tests for web_search tool removal."""

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return AnthropicWebSearchHook()

    def test_remove_web_search_tool(self, hook):
        """Test removal of web_search tool from tools list."""
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "custom", "name": "other_tool", "description": "Other tool"},
        ]
        filtered_tools, web_search_tool = hook._remove_web_search_tool(tools)

        assert len(filtered_tools) == 1
        assert filtered_tools[0]["name"] == "other_tool"
        assert web_search_tool is not None
        assert web_search_tool["type"] == "web_search_20250305"

    def test_remove_web_search_tool_only_web_search(self, hook):
        """Test removal when only web_search tool is present."""
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        filtered_tools, web_search_tool = hook._remove_web_search_tool(tools)

        assert len(filtered_tools) == 0
        assert web_search_tool is not None

    def test_remove_web_search_tool_no_web_search(self, hook):
        """Test removal when no web_search tool is present."""
        tools = [
            {"type": "custom", "name": "tool1"},
            {"type": "function", "name": "tool2"},
        ]
        filtered_tools, web_search_tool = hook._remove_web_search_tool(tools)

        assert len(filtered_tools) == 2
        assert web_search_tool is None


class TestAnthropicWebSearchHookParamExtraction:
    """Tests for search parameter extraction."""

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return AnthropicWebSearchHook()

    def test_extract_search_params_max_uses(self, hook):
        """Test extraction of max_uses as max_results."""
        web_search_tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 10,
        }
        params = hook._extract_search_params_from_tool(web_search_tool)

        assert params["max_results"] == 10

    def test_extract_search_params_user_location(self, hook):
        """Test extraction of country from user_location."""
        web_search_tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "user_location": {
                "type": "approximate",
                "country": "US",
                "city": "San Francisco",
            },
        }
        params = hook._extract_search_params_from_tool(web_search_tool)

        assert params["country"] == "US"

    def test_extract_search_params_empty(self, hook):
        """Test extraction with no optional params."""
        web_search_tool = {
            "type": "web_search_20250305",
            "name": "web_search",
        }
        params = hook._extract_search_params_from_tool(web_search_tool)

        assert params == {}


class TestAnthropicWebSearchHookFormatting:
    """Tests for search results formatting."""

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return AnthropicWebSearchHook()

    def test_format_search_results(self, hook):
        """Test formatting of search results for injection."""
        search_results = {
            "type": "web_search_results",
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "snippet": "This is the first result.",
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "snippet": "This is the second result.",
                    "date": "2024-01-15",
                },
            ],
        }
        formatted = hook._format_search_results_for_injection(search_results)

        assert "Web Search Results:" in formatted
        assert "Test Result 1" in formatted
        assert "https://example.com/1" in formatted
        assert "Test Result 2" in formatted
        assert "2024-01-15" in formatted

    def test_format_search_results_empty(self, hook):
        """Test formatting with empty results."""
        search_results = {"type": "web_search_results", "results": []}
        formatted = hook._format_search_results_for_injection(search_results)

        assert formatted == "No search results found."


class TestAnthropicWebSearchHookPreCallDeployment:
    """Tests for async_pre_call_deployment_hook."""

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return AnthropicWebSearchHook()

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_non_anthropic_messages(self, hook):
        """Test hook passes through for non-anthropic_messages call types."""
        kwargs = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.completion
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_no_tools(self, hook):
        """Test hook passes through when no tools present."""
        kwargs = {"model": "gpt-4", "messages": []}
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_no_web_search(self, hook):
        """Test hook passes through when no web_search tool present."""
        kwargs = {
            "model": "gpt-4",
            "tools": [{"type": "custom", "name": "regular_tool"}],
        }
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_anthropic_provider(self, hook):
        """Test hook passes through for native Anthropic provider."""
        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "custom_llm_provider": "anthropic",
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )
        # Should pass through to let Anthropic handle web search natively
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_non_anthropic_provider(self, hook):
        """Test hook transforms request for non-Anthropic providers."""
        kwargs = {
            "model": "bedrock/claude-3-5-sonnet",
            "custom_llm_provider": "bedrock",
            "tools": [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
                {"type": "custom", "name": "other_tool", "description": "Other tool"},
            ],
        }
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )

        assert result is not None
        assert "_web_search_config" in result
        assert "_web_search_params" in result
        assert result["_web_search_params"]["max_results"] == 5

        # Check tools were transformed
        tool_names = [t.get("name") for t in result["tools"]]
        assert "other_tool" in tool_names
        assert "web_search" not in tool_names  # Removed

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_removes_all_tools_if_only_web_search(
        self, hook
    ):
        """Test hook sets tools to None if only web_search was present."""
        kwargs = {
            "model": "bedrock/claude-3-5-sonnet",
            "custom_llm_provider": "bedrock",
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )

        assert result is not None
        assert result["tools"] is None


class TestAnthropicWebSearchHookShouldUse:
    """Tests for should_use_web_search_hook static method."""

    def test_should_use_no_router(self):
        """Test returns False when no router provided."""
        kwargs = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}
        result = AnthropicWebSearchHook.should_use_web_search_hook(
            kwargs, llm_router=None
        )
        assert result is False

    def test_should_use_no_search_tools(self):
        """Test returns False when router has no search tools."""
        mock_router = MagicMock()
        mock_router.search_tools = []

        kwargs = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}
        result = AnthropicWebSearchHook.should_use_web_search_hook(
            kwargs, llm_router=mock_router
        )
        assert result is False

    def test_should_use_no_web_search_tool(self):
        """Test returns False when no web_search tool in request."""
        mock_router = MagicMock()
        mock_router.search_tools = [{"search_tool_name": "default_search"}]

        kwargs = {"tools": [{"type": "custom", "name": "other_tool"}]}
        result = AnthropicWebSearchHook.should_use_web_search_hook(
            kwargs, llm_router=mock_router
        )
        assert result is False

    def test_should_use_anthropic_provider(self):
        """Test returns False for native Anthropic provider."""
        mock_router = MagicMock()
        mock_router.search_tools = [{"search_tool_name": "default_search"}]

        kwargs = {
            "custom_llm_provider": "anthropic",
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        result = AnthropicWebSearchHook.should_use_web_search_hook(
            kwargs, llm_router=mock_router
        )
        assert result is False

    def test_should_use_all_conditions_met(self):
        """Test returns True when all conditions are met."""
        mock_router = MagicMock()
        mock_router.search_tools = [{"search_tool_name": "default_search"}]

        kwargs = {
            "custom_llm_provider": "bedrock",
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
        result = AnthropicWebSearchHook.should_use_web_search_hook(
            kwargs, llm_router=mock_router
        )
        assert result is True


class TestAnthropicWebSearchHookPerformSearch:
    """Tests for _perform_search method."""

    @pytest.fixture
    def hook_with_router(self):
        """Create hook instance with mock router."""
        mock_router = MagicMock()
        hook = AnthropicWebSearchHook(llm_router=mock_router)
        return hook, mock_router

    @pytest.mark.asyncio
    async def test_perform_search_no_router(self):
        """Test _perform_search returns None when no router."""
        hook = AnthropicWebSearchHook()
        result = await hook._perform_search("test query")
        assert result is None

    @pytest.mark.asyncio
    async def test_perform_search_success(self, hook_with_router):
        """Test _perform_search returns formatted results on success."""
        hook, mock_router = hook_with_router

        # Create mock search response
        mock_result = MagicMock()
        mock_result.title = "Test Title"
        mock_result.url = "https://example.com"
        mock_result.snippet = "Test snippet"
        mock_result.date = "2024-01-15"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_router.asearch = AsyncMock(return_value=mock_response)

        result = await hook._perform_search("test query", {"max_results": 5})

        assert result is not None
        assert result["type"] == "web_search_results"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test Title"
        assert result["results"][0]["url"] == "https://example.com"

        mock_router.asearch.assert_called_once_with(
            query="test query",
            search_tool_name="default_search",
            max_results=5,
        )

    @pytest.mark.asyncio
    async def test_perform_search_error(self, hook_with_router):
        """Test _perform_search returns None on error."""
        hook, mock_router = hook_with_router
        mock_router.asearch = AsyncMock(side_effect=Exception("Search failed"))

        result = await hook._perform_search("test query")
        assert result is None
