"""
Tests for max_tokens vs thinking.budget_tokens constraint validation
in the websearch interception agentic loop.

Covers:
  - M1-I1: max_tokens auto-adjustment when <= thinking.budget_tokens
  - M1-I3: Unit tests for thinking parameter validation
  - M2-I5/I8: litellm_logging_obj excluded from follow-up kwargs to prevent SpendLog dedup
"""
import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_calls() -> List[Dict]:
    return [
        {
            "id": "toolu_01",
            "type": "tool_use",
            "name": "web_search",
            "input": {"query": "litellm spend tracking"},
        }
    ]


def _make_logging_obj(model: str = "bedrock/us.anthropic.claude-opus-4-6-v1") -> MagicMock:
    obj = MagicMock()
    obj.model_call_details = {
        "agentic_loop_params": {"model": model, "custom_llm_provider": "bedrock"},
    }
    return obj


# ---------------------------------------------------------------------------
# M1-I1 / M1-I3: max_tokens validation against thinking.budget_tokens
# ---------------------------------------------------------------------------

class TestThinkingBudgetTokensConstraint:
    """Validate that _execute_agentic_loop adjusts max_tokens when <= thinking.budget_tokens."""

    @pytest.mark.asyncio
    async def test_max_tokens_adjusted_when_less_than_budget(self):
        """max_tokens < thinking.budget_tokens → auto-adjusted to budget_tokens + 1024."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()  # dummy response

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={
                    "max_tokens": 1024,
                    "thinking": {"type": "enabled", "budget_tokens": 5000},
                },
                logging_obj=_make_logging_obj(),
                stream=False,
                kwargs={},
            )

        assert captured_kwargs["max_tokens"] == 5000 + 1024

    @pytest.mark.asyncio
    async def test_max_tokens_adjusted_when_equal_to_budget(self):
        """max_tokens == thinking.budget_tokens → still adjusted (must be strictly greater)."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={
                    "max_tokens": 5000,
                    "thinking": {"type": "enabled", "budget_tokens": 5000},
                },
                logging_obj=_make_logging_obj(),
                stream=False,
                kwargs={},
            )

        assert captured_kwargs["max_tokens"] == 5000 + 1024

    @pytest.mark.asyncio
    async def test_max_tokens_unchanged_when_greater_than_budget(self):
        """max_tokens > thinking.budget_tokens → no adjustment needed."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={
                    "max_tokens": 10000,
                    "thinking": {"type": "enabled", "budget_tokens": 5000},
                },
                logging_obj=_make_logging_obj(),
                stream=False,
                kwargs={},
            )

        assert captured_kwargs["max_tokens"] == 10000

    @pytest.mark.asyncio
    async def test_no_thinking_param_no_adjustment(self):
        """No thinking parameter → max_tokens used as-is (default 1024)."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={},
                logging_obj=_make_logging_obj(),
                stream=False,
                kwargs={},
            )

        assert captured_kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_thinking_without_budget_tokens_no_adjustment(self):
        """thinking param exists but has no budget_tokens → max_tokens used as-is."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={
                    "max_tokens": 2048,
                    "thinking": {"type": "enabled"},
                },
                logging_obj=_make_logging_obj(),
                stream=False,
                kwargs={},
            )

        assert captured_kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# M2-I5 / M2-I8: litellm_logging_obj excluded from follow-up kwargs
# ---------------------------------------------------------------------------

class TestLoggingObjExcludedFromFollowUp:
    """Verify litellm_logging_obj is NOT forwarded to the follow-up acreate() call.

    Passing the same logging object to both initial and follow-up calls causes
    the has_logged_async_success dedup flag to fire, silently preventing the
    initial call's spend from being recorded in SpendLogs.
    """

    @pytest.mark.asyncio
    async def test_litellm_logging_obj_excluded_from_anthropic_followup(self):
        """The Anthropic messages follow-up must NOT receive litellm_logging_obj."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        fake_logging_obj = _make_logging_obj()

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={"max_tokens": 4096},
                logging_obj=fake_logging_obj,
                stream=False,
                kwargs={
                    "litellm_logging_obj": fake_logging_obj,
                    "metadata": {"user_api_key": "sk-test"},
                    "temperature": 0.5,
                },
            )

        # litellm_logging_obj must be absent from the follow-up call
        assert "litellm_logging_obj" not in captured_kwargs
        # But other kwargs (metadata, temperature) must be preserved
        assert captured_kwargs.get("metadata") == {"user_api_key": "sk-test"}
        assert captured_kwargs.get("temperature") == 0.5

    @pytest.mark.asyncio
    async def test_websearch_flags_also_excluded(self):
        """Both _websearch_interception flags and litellm_logging_obj must be excluded."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={"max_tokens": 4096},
                logging_obj=_make_logging_obj(),
                stream=False,
                kwargs={
                    "litellm_logging_obj": MagicMock(),
                    "_websearch_interception_converted_stream": True,
                    "_websearch_interception_other": "x",
                    "api_key": "fake",
                },
            )

        assert "litellm_logging_obj" not in captured_kwargs
        assert "_websearch_interception_converted_stream" not in captured_kwargs
        assert "_websearch_interception_other" not in captured_kwargs
        assert captured_kwargs.get("api_key") == "fake"
