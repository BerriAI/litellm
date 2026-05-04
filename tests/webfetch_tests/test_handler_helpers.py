"""Tests for handler.py uncovered methods.

Focus on static methods and internal helpers that can be tested
without mocking the entire litellm ecosystem.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from litellm.integrations.webfetch_interception.handler import WebFetchInterceptionLogger


class TestResolveMaxTokens:
    """Test _resolve_max_tokens helper."""

    def test_default_value(self):
        """Default max_tokens is 1024."""
        result = WebFetchInterceptionLogger._resolve_max_tokens({}, {})
        assert result == 1024

    def test_from_optional_params(self):
        """max_tokens from optional_params."""
        result = WebFetchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 2048}, {}
        )
        assert result == 2048

    def test_from_kwargs(self):
        """max_tokens from kwargs."""
        result = WebFetchInterceptionLogger._resolve_max_tokens(
            {}, {"max_tokens": 4096}
        )
        assert result == 4096

    def test_thinking_budget_adjustment(self):
        """max_tokens <= thinking.budget_tokens triggers adjustment."""
        result = WebFetchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 1000},
            {"max_tokens": 1000, "thinking": {"budget_tokens": 1500}},
        )
        # Should adjust to budget_tokens + 1024
        assert result == 2524

    def test_thinking_budget_no_adjustment(self):
        """max_tokens > thinking.budget_tokens no adjustment."""
        result = WebFetchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 5000},
            {"thinking": {"budget_tokens": 1000}},
        )
        assert result == 5000


class TestPrepareFollowupKwargs:
    """Test _prepare_followup_kwargs."""

    def test_basic(self):
        """Basic followup kwargs preparation."""
        kwargs = {"model": "claude-3", "messages": [], "max_tokens": 1000}
        result = WebFetchInterceptionLogger._prepare_followup_kwargs(kwargs)

        assert "model" in result
        assert "api_key" in result
        assert "base_url" in result

    def test_direct_params(self):
        """Direct params overrides."""
        kwargs = {
            "model": "gpt-4",
            "api_key": "sk-test",
            "base_url": "https://custom.example.com",
        }
        result = WebFetchInterceptionLogger._prepare_followup_kwargs(kwargs)

        assert result["api_key"] == "sk-test"
        assert result["base_url"] == "https://custom.example.com"


class TestBuildFollowUpMessages:
    """Test _build_follow_up_messages."""

    def test_basic(self):
        """Basic follow-up message building."""
        result = WebFetchInterceptionLogger._build_follow_up_messages(
            tool_call_id="call-1",
            function_name="web_fetch",
            fetch_result="Fetched content",
        )

        assert result["tool_call_id"] == "call-1"
        assert result["role"] == "tool"
        assert "Fetched content" in result["content"]

    def test_removes_thinking_blocks(self):
        """Thinking content is stripped."""
        result = WebFetchInterceptionLogger._build_follow_up_messages(
            tool_call_id="call-1",
            function_name="fetch",
            fetch_result="<antThinking>thinking</antThinking>\nActual content",
        )

        assert "<antThinking>" not in result["content"]
        assert "Actual content" in result["content"]


class TestBuildKwargsForFollowup:
    """Test _build_kwargs_for_followup."""

    def test_basic(self):
        """Build kwargs for followup request."""
        kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4096,
            "optional_params": {"temperature": 0.7},
        }

        result = WebFetchInterceptionLogger._build_kwargs_for_followup(kwargs)

        assert result["model"] == "claude-3-opus-20240229"
        assert result["max_tokens"] == 4096

    def test_resolves_model_name(self):
        """Model name is resolved from router."""
        with patch(
            "litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger._resolve_full_model_name",
            return_value="anthropic/claude-3-opus",
        ):
            kwargs = {
                "model": "claude-opus",
                "messages": [],
            }
            result = WebFetchInterceptionLogger._build_kwargs_for_followup(kwargs)
            assert result["model"] == "anthropic/claude-3-opus"


class TestResolveFullModelName:
    """Test _resolve_full_model_name."""

    def test_anthropic_prefix(self):
        """Anthropic models get full prefix."""
        result = WebFetchInterceptionLogger._resolve_full_model_name(
            "claude-3-opus",
            {"custom_llm_provider": "anthropic"},
        )
        assert result == "anthropic/claude-3-opus"

    def test_existing_prefix(self):
        """Already prefixed models unchanged."""
        result = WebFetchInterceptionLogger._resolve_full_model_name(
            "anthropic/claude-3-opus",
            {},
        )
        assert result == "anthropic/claude-3-opus"

    def test_no_provider(self):
        """No provider, return as-is."""
        result = WebFetchInterceptionLogger._resolve_full_model_name(
            "gpt-4",
            {},
        )
        assert result == "gpt-4"


class TestCleanOptionalParams:
    """Test _clean_optional_params."""

    def test_basic_cleaning(self):
        """Remove internal/chaining params."""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "user": "test-user",
        }
        result = WebFetchInterceptionLogger._clean_optional_params(params)

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000

    def test_removes_internal_keys(self):
        """Internal keys are removed."""
        params = {
            "temperature": 0.7,
            "litellm_id": "123",
            "call_type": "completion",
        }
        result = WebFetchInterceptionLogger._clean_optional_params(params)

        assert "litellm_id" not in result
        assert "call_type" not in result


class TestExtractUrlFromToolCall:
    """Test _extract_url_from_tool_call."""

    def test_from_input_url(self):
        """Extract URL from input.url."""
        tool_call = {
            "input": {"url": "https://example.com"},
        }
        result = WebFetchInterceptionLogger._extract_url_from_tool_call(tool_call)
        assert result == "https://example.com"

    def test_from_function_args(self):
        """Extract URL from function.arguments."""
        tool_call = {
            "function": {"arguments": {"url": "https://test.com"}},
        }
        result = WebFetchInterceptionLogger._extract_url_from_tool_call(tool_call)
        assert result == "https://test.com"

    def test_not_found(self):
        """URL not found returns None."""
        tool_call = {"other": "data"}
        result = WebFetchInterceptionLogger._extract_url_from_tool_call(tool_call)
        assert result is None


class TestCreateEmptyFetchResult:
    """Test _create_empty_fetch_result."""

    @pytest.mark.asyncio
    async def test_returns_string(self):
        """Returns error string."""
        result = await WebFetchInterceptionLogger._create_empty_fetch_result()
        assert isinstance(result, str)
        assert "Failed" in result or "error" in result.lower()


class TestAsyncBuildAnthropicRequestPatch:
    """Test async_build_anthropic_request_patch."""

    @pytest.mark.asyncio
    async def test_basic(self):
        """Build Anthropic request patch from kwargs."""
        kwargs = {
            "model": "claude-3-opus",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }

        with patch(
            "litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger._resolve_full_model_name",
            return_value="anthropic/claude-3-opus",
        ):
            result = await WebFetchInterceptionLogger.async_build_anthropic_request_patch(
                kwargs=kwargs,
                original_anthropic_messages=[],
                anthropic_headers={"Authorization": "Bearer test"},
            )

        assert "model" in result
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_with_tools(self):
        """Patch includes tools."""
        kwargs = {
            "model": "claude-3",
            "tools": [{"type": "web_fetch_20250305", "name": "web_fetch"}],
            "messages": [],
        }

        with patch(
            "litellm.integrations.webfetch_interception.handler.WebFetchInterceptionLogger._resolve_full_model_name",
            return_value="anthropic/claude-3",
        ):
            with patch(
                "litellm.integrations.webfetch_interception.transformation.WebFetchTransformation.convert_native_to_litellm",
                return_value={"name": "litellm-web-fetch", "input_schema": {}}),
            ):
                result = await WebFetchInterceptionLogger.async_build_anthropic_request_patch(
                    kwargs=kwargs,
                    original_anthropic_messages=[],
                    anthropic_headers={},
                )

        assert "tools" in result


class TestPreCallDeploymentHook:
    """Test async_pre_call_deployment_hook."""

    @pytest.mark.asyncio
    async def test_no_fetch_tools_no_action(self):
        """Without fetch_tools in router, pass through."""
        logger = WebFetchInterceptionLogger()
        kwargs = {"model": "gpt-4", "messages": []}

        result = await logger.async_pre_call_deployment_hook(kwargs)
        assert result == kwargs

    @pytest.mark.asyncio
    async def test_with_router_fetch_tools(self):
        """With router fetch_tools, intercept tools."""
        logger = WebFetchInterceptionLogger()
        logger.router_fetch_tools = [
            {"fetch_tool_name": "fetch-1", "litellm_params": {"provider": "firecrawl"}}
        ]

        kwargs = {
            "model": "claude-3",
            "messages": [],
            "tools": [{"type": "web_fetch_20250305", "name": "web_fetch"}],
        }

        result = await logger.async_pre_call_deployment_hook(kwargs)
        assert "tools" in result
