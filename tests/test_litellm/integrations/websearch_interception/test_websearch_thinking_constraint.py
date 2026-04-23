"""
Tests for max_tokens vs thinking.budget_tokens constraint validation
in the websearch interception agentic loop.

Covers:
  - M1-I1: max_tokens auto-adjustment when <= thinking.budget_tokens
  - M1-I3: Unit tests for thinking parameter validation
  - M2-I5/I8: litellm_logging_obj excluded from follow-up kwargs to prevent SpendLog dedup
  - M3-I12: Regression tests for error scenarios
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

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


class TestResolveMaxTokensEdgeCases:
    """Edge cases for _resolve_max_tokens: infinity, negative, extreme values."""

    def test_infinity_budget_tokens_no_crash(self):
        """float('inf') budget_tokens must not crash with OverflowError."""
        result = WebSearchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 1024, "thinking": {"budget_tokens": float("inf")}}, {}
        )
        assert result == 1024  # no adjustment for non-finite values

    def test_negative_infinity_no_crash(self):
        result = WebSearchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 1024, "thinking": {"budget_tokens": float("-inf")}}, {}
        )
        assert result == 1024

    def test_nan_budget_tokens_no_crash(self):
        result = WebSearchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 1024, "thinking": {"budget_tokens": float("nan")}}, {}
        )
        assert result == 1024

    def test_negative_budget_tokens_no_adjustment(self):
        result = WebSearchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 1024, "thinking": {"budget_tokens": -100}}, {}
        )
        assert result == 1024

    def test_zero_budget_tokens_no_adjustment(self):
        result = WebSearchInterceptionLogger._resolve_max_tokens(
            {"max_tokens": 1024, "thinking": {"budget_tokens": 0}}, {}
        )
        assert result == 1024


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
                    "metadata": {"user_api_key": "test-key-hash"},
                    "temperature": 0.5,
                },
            )

        # litellm_logging_obj must be absent from the follow-up call
        assert "litellm_logging_obj" not in captured_kwargs
        # But other kwargs (metadata, temperature) must be preserved
        assert captured_kwargs.get("metadata") == {"user_api_key": "test-key-hash"}
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


# ---------------------------------------------------------------------------
# M3-I12: Regression tests for error scenarios
# ---------------------------------------------------------------------------

class TestFollowUpErrorScenarios:
    """Regression tests: the agentic loop must surface errors properly and
    not silently swallow them (except at the _call_agentic_completion_hooks
    level which intentionally catches to return the initial response)."""

    @pytest.mark.asyncio
    async def test_followup_400_raises(self):
        """A 400 error from the follow-up call must propagate out of _execute_agentic_loop."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

        async def _fail_acreate(**kw):
            raise Exception("max_tokens must be greater than thinking.budget_tokens")

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fail_acreate,
        ), patch.object(logger, "_execute_search", return_value="search result"):

            with pytest.raises(Exception, match="max_tokens must be greater"):
                await logger._execute_agentic_loop(
                    model="us.anthropic.claude-opus-4-6-v1",
                    messages=[{"role": "user", "content": "hi"}],
                    tool_calls=_make_tool_calls(),
                    thinking_blocks=[],
                    anthropic_messages_optional_request_params={"max_tokens": 4096},
                    logging_obj=_make_logging_obj(),
                    stream=False,
                    kwargs={},
                )

    @pytest.mark.asyncio
    async def test_search_failure_does_not_crash_loop(self):
        """If a search fails, the loop should still attempt the follow-up with error text."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        with patch(
            "litellm.integrations.websearch_interception.handler.anthropic_messages.acreate",
            side_effect=_fake_acreate,
        ), patch.object(
            logger, "_execute_search", side_effect=Exception("search API down")
        ):

            result = await logger._execute_agentic_loop(
                model="us.anthropic.claude-opus-4-6-v1",
                messages=[{"role": "user", "content": "hi"}],
                tool_calls=_make_tool_calls(),
                thinking_blocks=[],
                anthropic_messages_optional_request_params={"max_tokens": 4096},
                logging_obj=_make_logging_obj(),
                stream=False,
                kwargs={},
            )

        # The follow-up call should have been made (with error text in search results)
        assert result is not None
        # Messages should contain the error text
        follow_up_messages = captured_kwargs.get("messages", [])
        assert len(follow_up_messages) > 1  # original + assistant + tool_result

    @pytest.mark.asyncio
    async def test_metadata_preserved_after_logging_obj_exclusion(self):
        """Proxy metadata (user_api_key, team_id, etc.) must survive in follow-up kwargs
        even after litellm_logging_obj is excluded — so the new logging_obj from
        function_setup has access to proxy tracking metadata."""
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        captured_kwargs: Dict[str, Any] = {}

        async def _fake_acreate(**kw):
            captured_kwargs.update(kw)
            return MagicMock()

        proxy_metadata = {
            "user_api_key": "test-proxy-key-hash",
            "user_api_key_user_id": "user-123",
            "user_api_key_team_id": "team-456",
            "user_api_key_org_id": "org-789",
            "user_api_key_end_user_id": "end-user-001",
        }

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
                    "metadata": proxy_metadata,
                    "litellm_call_id": "call-abc-123",
                },
            )

        # litellm_logging_obj excluded
        assert "litellm_logging_obj" not in captured_kwargs
        # But ALL proxy metadata must be preserved
        assert captured_kwargs.get("metadata") == proxy_metadata
        assert captured_kwargs.get("litellm_call_id") == "call-abc-123"
