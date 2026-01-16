"""
Unit tests for ToolSearchPreCallHook

Tests client-side tool search functionality (BM25 + regex) for
providers that don't support server-side tool search.
"""

import pytest

from litellm.integrations.tool_search_pre_call_hook import (
    ClientSideToolSearch,
    ToolSearchPreCallHook,
)
from litellm.types.utils import CallTypes


class TestClientSideToolSearch:
    """Tests for ClientSideToolSearch class."""

    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions for testing."""
        return [
            {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or coordinates",
                        },
                        "units": {
                            "type": "string",
                            "description": "Temperature units (celsius/fahrenheit)",
                        },
                    },
                },
            },
            {
                "name": "search_flights",
                "description": "Search for available flights between destinations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Departure airport"},
                        "destination": {
                            "type": "string",
                            "description": "Arrival airport",
                        },
                    },
                },
            },
            {
                "name": "book_hotel",
                "description": "Book a hotel room for specified dates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "check_in": {"type": "string", "description": "Check-in date"},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_distance",
                    "description": "Calculate distance between two points",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "point1": {"type": "string"},
                            "point2": {"type": "string"},
                        },
                    },
                },
            },
        ]

    def test_bm25_search_basic(self, sample_tools):
        """Test BM25 search returns relevant tools."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_bm25("weather forecast", max_results=5)

        assert len(results) > 0
        assert results[0]["type"] == "tool_reference"
        assert results[0]["tool_name"] == "get_weather"

    def test_bm25_search_flight(self, sample_tools):
        """Test BM25 search for flights."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_bm25("find flights to paris", max_results=5)

        assert len(results) > 0
        tool_names = [r["tool_name"] for r in results]
        assert "search_flights" in tool_names

    def test_bm25_search_empty_query(self, sample_tools):
        """Test BM25 search with empty query returns empty."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_bm25("", max_results=5)
        assert results == []

    def test_regex_search_basic(self, sample_tools):
        """Test regex search returns matching tools."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_regex("weather", max_results=5)

        assert len(results) > 0
        assert results[0]["tool_name"] == "get_weather"

    def test_regex_search_pattern(self, sample_tools):
        """Test regex search with pattern."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_regex("book.*|search.*", max_results=5)

        assert len(results) >= 2
        tool_names = [r["tool_name"] for r in results]
        assert "book_hotel" in tool_names
        assert "search_flights" in tool_names

    def test_regex_search_invalid_pattern(self, sample_tools):
        """Test regex search with invalid pattern returns empty."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_regex("[invalid", max_results=5)
        assert results == []

    def test_regex_search_too_long_pattern(self, sample_tools):
        """Test regex search with too long pattern returns empty."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_regex("a" * 201, max_results=5)
        assert results == []

    def test_search_function_format_tools(self, sample_tools):
        """Test search handles both native and function-wrapped tool formats."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_bm25("calculate distance", max_results=5)

        assert len(results) > 0
        assert results[0]["tool_name"] == "calculate_distance"

    def test_max_results_limit(self, sample_tools):
        """Test max_results limits number of results."""
        search = ClientSideToolSearch(sample_tools)
        results = search.search_bm25("location city", max_results=2)

        assert len(results) <= 2


class TestToolSearchPreCallHook:
    """Tests for ToolSearchPreCallHook class."""

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return ToolSearchPreCallHook()

    def test_detect_tool_search_bm25(self, hook):
        """Test detection of BM25 tool search."""
        tools = [
            {"type": "tool_search_tool_bm25_20251119", "name": "my_search"},
            {"name": "other_tool"},
        ]
        config = hook._detect_tool_search(tools)

        assert config is not None
        assert config["search_type"] == "bm25"
        assert config["name"] == "my_search"

    def test_detect_tool_search_regex(self, hook):
        """Test detection of regex tool search."""
        tools = [
            {"type": "tool_search_tool_regex_20251119", "name": "regex_search"},
            {"name": "other_tool"},
        ]
        config = hook._detect_tool_search(tools)

        assert config is not None
        assert config["search_type"] == "regex"
        assert config["name"] == "regex_search"

    def test_detect_tool_search_not_present(self, hook):
        """Test no detection when tool_search not present."""
        tools = [{"name": "regular_tool"}, {"name": "another_tool"}]
        config = hook._detect_tool_search(tools)
        assert config is None

    def test_prepare_tools_separates_deferred(self, hook):
        """Test _prepare_tools separates deferred tools."""
        tools = [
            {"type": "tool_search_tool_bm25_20251119", "name": "tool_search"},
            {"name": "always_available", "description": "Always available tool"},
            {
                "name": "deferred_tool",
                "defer_loading": True,
                "description": "Deferred tool",
            },
        ]
        config = {"search_type": "bm25", "name": "tool_search"}

        modified_tools, deferred_tools = hook._prepare_tools(tools, config)

        # Deferred tool should be separated
        assert len(deferred_tools) == 1
        assert deferred_tools[0]["name"] == "deferred_tool"

        # Modified tools should have non-deferred + synthetic tool_search
        # All tools are now in Anthropic format (top-level name)
        tool_names = [t.get("name") for t in modified_tools]
        assert "always_available" in tool_names
        assert "tool_search" in tool_names  # Synthetic function in Anthropic format
        assert "deferred_tool" not in tool_names

    def test_create_tool_search_function_bm25(self, hook):
        """Test creation of synthetic BM25 tool_search function in Anthropic format."""
        config = {"search_type": "bm25", "name": "my_search"}
        func = hook._create_tool_search_function(config)

        # Should be in Anthropic format (top-level name, description, input_schema)
        assert func["name"] == "my_search"
        assert "natural language" in func["description"].lower()
        assert "query" in func["input_schema"]["properties"]

    def test_create_tool_search_function_regex(self, hook):
        """Test creation of synthetic regex tool_search function in Anthropic format."""
        config = {"search_type": "regex", "name": "regex_search"}
        func = hook._create_tool_search_function(config)

        # Should be in Anthropic format (top-level name, description, input_schema)
        assert func["name"] == "regex_search"
        assert "regex" in func["description"].lower()


class TestToolSearchPreCallHookAsync:
    """Async tests for ToolSearchPreCallHook."""

    @pytest.fixture
    def hook(self):
        """Create hook instance."""
        return ToolSearchPreCallHook()

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_non_anthropic(self, hook):
        """Test hook passes through for non-anthropic call types."""
        kwargs = {"tools": [{"type": "tool_search_tool_bm25_20251119"}]}
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.completion
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_no_tools(self, hook):
        """Test hook passes through when no tools present."""
        kwargs = {"model": "openai/gpt-4o", "messages": []}
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_no_tool_search(self, hook):
        """Test hook passes through when no tool_search_tool present."""
        kwargs = {
            "model": "openai/gpt-4o",
            "tools": [{"name": "regular_tool"}],
        }
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_anthropic_passthrough(self, hook):
        """Test hook passes through for anthropic provider (server-side support)."""
        kwargs = {
            "model": "claude-3-5-sonnet-20241022",
            "custom_llm_provider": "anthropic",
            "tools": [{"type": "tool_search_tool_bm25_20251119", "name": "tool_search"}],
        }
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )
        assert result is None  # Pass through to server-side

    @pytest.mark.asyncio
    async def test_pre_call_deployment_hook_transforms_for_openai(self, hook):
        """Test hook transforms request for non-anthropic providers."""
        kwargs = {
            "model": "openai/gpt-4o",
            "tools": [
                {"type": "tool_search_tool_bm25_20251119", "name": "tool_search"},
                {"name": "deferred_tool", "defer_loading": True, "description": "A tool"},
                {"name": "always_available", "description": "Always available"},
            ],
        }
        result = await hook.async_pre_call_deployment_hook(
            kwargs, CallTypes.anthropic_messages
        )

        assert result is not None
        assert "_tool_search_config" in result
        assert "_deferred_tools" in result
        assert len(result["_deferred_tools"]) == 1

        # Check tools were transformed (all in Anthropic format)
        tool_names = [t.get("name") for t in result["tools"]]
        assert "always_available" in tool_names
        assert "tool_search" in tool_names  # Synthetic function in Anthropic format
        assert "deferred_tool" not in tool_names  # Removed
