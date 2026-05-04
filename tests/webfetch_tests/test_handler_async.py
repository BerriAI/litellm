"""Tests for async handler methods (agentic loop, init, etc.)

These cover the core async logic that drives the fetch interception.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from litellm.integrations.webfetch_interception.handler import WebFetchInterceptionLogger


class TestInitializeFromProxyConfig:
    """Test initialize_from_proxy_config."""

    def test_from_litellm_settings(self):
        """Initialize from litellm_settings with params."""
        litellm_settings = {
            "webfetch_interception_params": {
                "fetch_provider": "firecrawl",
                "api_key": "fc-test",
                "api_base": "https://api.firecrawl.dev",
            }
        }
        logger = WebFetchInterceptionLogger.initialize_from_proxy_config(
            litellm_settings=litellm_settings,
            callback_specific_params={},
        )
        assert isinstance(logger, WebFetchInterceptionLogger)

    def test_from_callback_specific_params(self):
        """Initialize from callback_specific_params."""
        callback_specific_params = {
            "webfetch_interception": {
                "fetch_provider": "firecrawl",
                "api_key": "fc-callback",
            }
        }
        logger = WebFetchInterceptionLogger.initialize_from_proxy_config(
            litellm_settings={},
            callback_specific_params=callback_specific_params,
        )
        assert isinstance(logger, WebFetchInterceptionLogger)

    def test_defaults(self):
        """Initialize with defaults."""
        logger = WebFetchInterceptionLogger.initialize_from_proxy_config(
            litellm_settings={},
            callback_specific_params={},
        )
        assert isinstance(logger, WebFetchInterceptionLogger)
        assert logger.fetch_provider == "firecrawl"


class TestFromConfigYaml:
    """Test from_config_yaml."""

    def test_basic(self):
        """Create from config dict."""
        config = {
            "fetch_provider": "firecrawl",
            "api_key": "fc-test",
            "api_base": "https://custom.firecrawl.dev",
        }
        logger = WebFetchInterceptionLogger.from_config_yaml(config)
        assert isinstance(logger, WebFetchInterceptionLogger)
        assert logger.fetch_provider == "firecrawl"

    def test_missing_api_key(self):
        """Missing API key raises ValueError."""
        config = {"fetch_provider": "firecrawl"}
        with pytest.raises(ValueError, match="api_key"):
            WebFetchInterceptionLogger.from_config_yaml(config)


class TestExecuteToolCallFetches:
    """Test _execute_tool_call_fetches."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        """Execute fetch for single tool call."""
        logger = WebFetchInterceptionLogger()
        logger._execute_fetch = AsyncMock(return_value="# Fetched content")

        tool_calls = [
            {
                "id": "call-1",
                "name": "litellm-web-fetch",
                "input": {"url": "https://example.com"},
            }
        ]

        result = await logger._execute_tool_call_fetches(tool_calls)

        assert len(result) == 1
        assert "Fetched content" in result[0]
        logger._execute_fetch.assert_awaited_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Execute fetch for multiple tool calls."""
        logger = WebFetchInterceptionLogger()
        logger._execute_fetch = AsyncMock(side_effect=["# Content 1", "# Content 2"])

        tool_calls = [
            {"id": "call-1", "name": "litellm-web-fetch", "input": {"url": "https://site1.com"}},
            {"id": "call-2", "name": "litellm-web-fetch", "input": {"url": "https://site2.com"}},
        ]

        result = await logger._execute_tool_call_fetches(tool_calls)

        assert len(result) == 2
        assert result[0] == "# Content 1"
        assert result[1] == "# Content 2"
        assert logger._execute_fetch.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_tool_calls(self):
        """Empty tool calls return empty results."""
        logger = WebFetchInterceptionLogger()
        logger._execute_fetch = AsyncMock()

        result = await logger._execute_tool_call_fetches([])

        assert result == []
        logger._execute_fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_error(self):
        """Failed fetch returns error message."""
        logger = WebFetchInterceptionLogger()
        logger._execute_fetch = AsyncMock(side_effect=Exception("Connection error"))

        tool_calls = [
            {"id": "call-1", "name": "litellm-web-fetch", "input": {"url": "https://fail.com"}}
        ]

        result = await logger._execute_tool_call_fetches(tool_calls)

        assert len(result) == 1
        assert "error" in result[0].lower() or "failed" in result[0].lower()


class TestAsyncPreRequestHook:
    """Test async_pre_request_hook interceptor."""

    @pytest.mark.asyncio
    async def test_no_tools_passes_through(self):
        """Without tools in request, pass through unchanged."""
        logger = WebFetchInterceptionLogger()
        kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        result = await logger.async_pre_request_hook({}, kwargs, {})
        assert result == kwargs

    @pytest.mark.asyncio
    async def test_with_native_anthropic_tools(self):
        """Convert native Anthropic tools to LiteLLM format."""
        logger = WebFetchInterceptionLogger()
        kwargs = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{"type": "web_fetch_20250305", "name": "web_fetch"}],
        }

        with patch(
            "litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger._convert_tools_to_litellm",
            return_value=[{"type": "function", "function": {"name": "litellm-web-fetch"}}],
        ):
            result = await logger.async_pre_request_hook({}, kwargs, {})

        assert "tools" in result
        assert result["tools"][0]["function"]["name"] == "litellm-web-fetch"


class TestAsyncShouldRunAgenticLoop:
    """Test async_should_run_agentic_loop."""

    @pytest.mark.asyncio
    async def test_no_tools(self):
        """Without tools, don't run."""
        logger = WebFetchInterceptionLogger()
        result = await logger.async_should_run_agentic_loop({})
        assert result is False

    @pytest.mark.asyncio
    async def test_no_webfetch_tools(self):
        """With non-fetch tools, don't run."""
        logger = WebFetchInterceptionLogger()
        kwargs = {"tools": [{"name": "calculator"}], "messages": []}
        result = await logger.async_should_run_agentic_loop(kwargs)
        assert result is False

    @pytest.mark.asyncio
    async def test_with_webfetch_tools(self):
        """With webfetch tools, run agentic loop."""
        logger = WebFetchInterceptionLogger()
        kwargs = {
            "tools": [{"name": "litellm-web-fetch"}],
            "messages": [],
        }
        result = await logger.async_should_run_agentic_loop(kwargs)
        assert result is True

    @pytest.mark.asyncio
    async def test_already_intercepted(self):
        """If already intercepted, don't run."""
        logger = WebFetchInterceptionLogger()
        kwargs = {
            "tools": [{"name": "litellm-web-fetch"}],
            "messages": [],
            "metadata": {"_webfetch_interception_completed": True},
        }
        result = await logger.async_should_run_agentic_loop(kwargs)
        assert result is False


class TestShouldRunChatCompletion:
    """Test async_should_run_chat_completion_agentic_loop."""

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        """Without tool_calls, don't run."""
        logger = WebFetchInterceptionLogger()
        kwargs = {"model": "gpt-4"}
        result = await logger.async_should_run_chat_completion_agentic_loop(kwargs)
        assert result is False

    @pytest.mark.asyncio
    async def test_with_webfetch_tool_calls(self):
        """With webfetch tool_calls, run."""
        logger = WebFetchInterceptionLogger()
        kwargs = {
            "model": "gpt-4",
            "response": {
                "tool_calls": [{"function": {"name": "litellm-web-fetch"}}]
            }
        }
        result = await logger.async_should_run_chat_completion_agentic_loop(kwargs)
        assert result is True


class TestAsyncRunAgenticLoop:
    """Test async_run_agentic_loop."""

    @pytest.mark.asyncio
    async def test_loop_execution(self):
        """Execute full agentic loop."""
        logger = WebFetchInterceptionLogger()
        logger._execute_agentic_loop = AsyncMock(return_value="Final response")

        kwargs = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await logger.async_run_agentic_loop(kwargs)
        assert result == "Final response"
        logger._execute_agentic_loop.assert_awaited_once()


class TestAsyncBuildAgenticLoopPlan:
    """Test async_build_agentic_loop_plan."""

    @pytest.mark.asyncio
    async def test_build_plan(self):
        """Build plan from kwargs."""
        logger = WebFetchInterceptionLogger()

        kwargs = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
            "optional_params": {"max_tokens": 4096, "temperature": 0.7},
        }

        result = await logger.async_build_agentic_loop_plan(kwargs)
        assert result["original_messages"] == kwargs["messages"]
        assert result["original_max_tokens"] == 4096


class TestAsyncBuildAnthropicRequestPatch:
    """Test _build_anthropic_request_patch."""

    @pytest.mark.asyncio
    async def test_builds_request(self):
        """Build Anthropic request from kwargs."""
        logger = WebFetchInterceptionLogger()
        logger._resolve_full_model_name = MagicMock(return_value="anthropic/claude-3")

        kwargs = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4096,
        }

        original_messages = ["original"]
        anthropic_headers = {"Authorization": "Bearer test"}

        result = await logger._build_anthropic_request_patch(
            kwargs=kwargs,
            original_anthropic_messages=original_messages,
            anthropic_headers=anthropic_headers,
        )

        assert "model" in result
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_preserves_thinking(self):
        """Preserve thinking blocks in request."""
        logger = WebFetchInterceptionLogger()
        logger._resolve_full_model_name = MagicMock(return_value="anthropic/claude-3")

        kwargs = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 1000},
        }

        result = await logger._build_anthropic_request_patch(
            kwargs=kwargs,
            original_anthropic_messages=[],
            anthropic_headers={},
        )

        assert "thinking" in result


class TestInitFromRouterFetchTools:
    """Test _init_from_router_fetch_tools."""

    def test_with_tools(self):
        """Initialize with router fetch tools."""
        logger = WebFetchInterceptionLogger()
        fetch_tools = [
            {
                "fetch_tool_name": "fetch-1",
                "litellm_params": {"provider": "firecrawl", "api_key": "fc-test"},
            }
        ]
        logger._init_from_router_fetch_tools(fetch_tools)
        assert logger.router_fetch_tools is not None
        assert len(logger.router_fetch_tools) == 1

    def test_none(self):
        """None tools doesn't crash."""
        logger = WebFetchInterceptionLogger()
        logger._init_from_router_fetch_tools(None)
        assert logger.router_fetch_tools is None


class TestAsyncExecuteFetch:
    """Test _execute_fetch."""

    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Successful fetch execution."""
        logger = WebFetchInterceptionLogger()
        logger.fetch_config = MagicMock()
        logger.fetch_config.afetch_url = AsyncMock(
            return_value=MagicMock(content="# Hello", title="Test")
        )

        result = await logger._execute_fetch("https://example.com")
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_fetch_error(self):
        """Fetch error raises exception."""
        logger = WebFetchInterceptionLogger()
        logger.fetch_config = MagicMock()
        logger.fetch_config.afetch_url = AsyncMock(side_effect=Exception("fail"))

        with pytest.raises(Exception, match="fail"):
            await logger._execute_fetch("https://example.com")


class TestAsyncRunChatCompletionAgenticLoop:
    """Test async_run_chat_completion_agentic_loop."""

    @pytest.mark.asyncio
    async def test_execution(self):
        """Execute chat completion agentic loop."""
        logger = WebFetchInterceptionLogger()
        
        mock_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call-1",
                        "function": {
                            "name": "litellm-web-fetch",
                            "arguments": '{"url": "https://example.com"}',
                        }
                    }]
                }
            }]
        }

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "response": mock_response,
        }

        with patch.object(
            logger,
            '_execute_chat_completion_agentic_loop',
            new_callable=AsyncMock,
            return_value={"choices": [{"message": {"content": "Final answer"}}]},
        ):
            result = await logger.async_run_chat_completion_agentic_loop(kwargs)
            assert result is not None
