"""
Tests for tool_choice conversion when websearch interception renames tools.

Regression test for https://github.com/BerriAI/litellm/issues/30822:
websearch_interception doesn't convert tool_choice, so forced web_search
fails on Bedrock (400 "Tool 'web_search' not found").
"""

from typing import Any, Dict

import pytest

from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)


def _anthropic_web_search_tool():
    return {"type": "web_search_20250305", "name": "web_search"}


class TestToolChoiceConversionDeploymentHook:
    """async_pre_call_deployment_hook should rewrite tool_choice when it references a web search tool."""

    @pytest.mark.asyncio
    async def test_anthropic_tool_choice_type_tool(self):
        """tool_choice: {type: 'tool', name: 'web_search'} -> name becomes litellm_web_search."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs: Dict[str, Any] = {
            "model": "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            "tools": [_anthropic_web_search_tool()],
            "tool_choice": {"type": "tool", "name": "web_search"},
            "custom_llm_provider": "bedrock",
        }

        result = await logger.async_pre_call_deployment_hook(kwargs, None)

        assert result is not None
        assert result["tool_choice"]["name"] == LITELLM_WEB_SEARCH_TOOL_NAME
        assert result["tool_choice"]["type"] == "tool"

    @pytest.mark.asyncio
    async def test_non_web_tool_choice_unchanged(self):
        """tool_choice referencing a non-web-search tool should not be modified."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs: Dict[str, Any] = {
            "model": "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            "tools": [
                _anthropic_web_search_tool(),
                {"name": "calculator", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "tool", "name": "calculator"},
            "custom_llm_provider": "bedrock",
        }

        result = await logger.async_pre_call_deployment_hook(kwargs, None)

        assert result is not None
        assert result["tool_choice"]["name"] == "calculator"

    @pytest.mark.asyncio
    async def test_string_tool_choice_unchanged(self):
        """tool_choice as a string (e.g. 'auto', 'any') should not be modified."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs: Dict[str, Any] = {
            "model": "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            "tools": [_anthropic_web_search_tool()],
            "tool_choice": "auto",
            "custom_llm_provider": "bedrock",
        }

        result = await logger.async_pre_call_deployment_hook(kwargs, None)

        assert result is not None
        assert result["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_no_tool_choice_unchanged(self):
        """When no tool_choice is set, the hook should not add one."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs: Dict[str, Any] = {
            "model": "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            "tools": [_anthropic_web_search_tool()],
            "custom_llm_provider": "bedrock",
        }

        result = await logger.async_pre_call_deployment_hook(kwargs, None)

        assert result is not None
        assert "tool_choice" not in result


class TestToolChoiceConversionPreRequestHook:
    """async_pre_request_hook should also rewrite tool_choice."""

    @pytest.mark.asyncio
    async def test_anthropic_tool_choice_rewritten(self):
        """Pre-request hook converts tool_choice.name for Anthropic-style requests."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs: Dict[str, Any] = {
            "tools": [_anthropic_web_search_tool()],
            "tool_choice": {"type": "tool", "name": "web_search"},
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }

        result = await logger.async_pre_request_hook(
            model="bedrock/claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            kwargs=kwargs,
        )

        assert result is not None
        assert result["tool_choice"]["name"] == LITELLM_WEB_SEARCH_TOOL_NAME

    @pytest.mark.asyncio
    async def test_pre_request_non_web_tool_choice_unchanged(self):
        """Pre-request hook leaves non-web tool_choice alone."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs: Dict[str, Any] = {
            "tools": [
                _anthropic_web_search_tool(),
                {"name": "calculator", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "tool", "name": "calculator"},
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }

        result = await logger.async_pre_request_hook(
            model="bedrock/claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            kwargs=kwargs,
        )

        assert result is not None
        assert result["tool_choice"]["name"] == "calculator"

    @pytest.mark.asyncio
    async def test_pre_request_string_tool_choice_unchanged(self):
        """Pre-request hook leaves string tool_choice alone."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs: Dict[str, Any] = {
            "tools": [_anthropic_web_search_tool()],
            "tool_choice": "auto",
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }

        result = await logger.async_pre_request_hook(
            model="bedrock/claude-haiku-4-5",
            messages=[{"role": "user", "content": "test"}],
            kwargs=kwargs,
        )

        assert result is not None
        assert result["tool_choice"] == "auto"
