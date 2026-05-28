"""
Regression tests for LIT-3400: web search interception passes duplicate
``output_config`` to ``anthropic_messages.acreate``.

Bug: ``_build_anthropic_request_patch`` left ``output_config`` (and any other
key that appears in both ``anthropic_messages_optional_request_params`` and
the raw litellm ``kwargs``) inside ``patch.kwargs``. At the follow-up call
site the spread ``**optional_params, **patch.kwargs`` then raised
``TypeError: ... got multiple values for keyword argument 'output_config'``.

Same shape existed in ``_build_chat_completion_request_patch``.

Fix: strip from ``patch.kwargs`` any key already present in
``patch.optional_params`` so the two surfaces are disjoint.
"""

import asyncio
from unittest.mock import patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)


async def _noop_search(self, query):
    return ("search result text", None)


@pytest.mark.asyncio
async def test_anthropic_patch_kwargs_disjoint_from_optional_params():
    """patch.kwargs ∩ patch.optional_params must be empty after the fix."""
    logger = WebSearchInterceptionLogger(enabled_providers=["anthropic"])
    anthropic_optional = {
        "max_tokens": 1024,
        "temperature": 0.5,
        "output_config": {"format": "anthropic"},
        "thinking": {"type": "enabled"},
    }
    raw_kwargs = {
        "output_config": {"format": "anthropic"},
        "thinking": {"type": "enabled"},
        "litellm_call_id": "lit-3400-test",
        "custom_llm_provider": "anthropic",
    }
    tool_calls = [
        {"type": "tool_use", "id": "tool_1", "name": "web_search",
         "input": {"query": "what is litellm"}}
    ]
    with patch.object(
        WebSearchInterceptionLogger, "_execute_search", _noop_search
    ):
        patch_obj, _ = await logger._build_anthropic_request_patch(
            model="anthropic/claude-opus-4-6",
            messages=[{"role": "user", "content": "q"}],
            tool_calls=tool_calls,
            thinking_blocks=[],
            anthropic_messages_optional_request_params=anthropic_optional,
            logging_obj=None,
            kwargs=raw_kwargs,
        )

    overlap = set(patch_obj.optional_params) & set(patch_obj.kwargs)
    assert overlap == set(), (
        f"patch.optional_params and patch.kwargs must be disjoint; "
        f"overlap was {overlap}"
    )
    # output_config remains in optional_params (single source of truth)
    assert patch_obj.optional_params["output_config"] == {"format": "anthropic"}


@pytest.mark.asyncio
async def test_anthropic_execute_agentic_loop_no_duplicate_output_config():
    """End-to-end: legacy ``_execute_agentic_loop`` no longer raises
    ``TypeError: got multiple values for keyword argument 'output_config'``."""
    logger = WebSearchInterceptionLogger(enabled_providers=["anthropic"])
    anthropic_optional = {
        "max_tokens": 1024,
        "output_config": {"format": "anthropic"},
    }
    raw_kwargs = {
        "output_config": {"format": "anthropic"},
        "litellm_call_id": "lit-3400-e2e",
    }
    tool_calls = [
        {"type": "tool_use", "id": "tool_1", "name": "web_search",
         "input": {"query": "what is litellm"}}
    ]
    captured = {}

    async def fake_acreate(**kw):
        captured.update(kw)
        return {"id": "msg_ok"}

    with patch.object(
        WebSearchInterceptionLogger, "_execute_search", _noop_search
    ), patch(
        "litellm.anthropic_interface.messages.acreate", side_effect=fake_acreate
    ):
        # Must not raise.
        await logger._execute_agentic_loop(
            model="anthropic/claude-opus-4-6",
            messages=[{"role": "user", "content": "q"}],
            tool_calls=tool_calls,
            thinking_blocks=[],
            anthropic_messages_optional_request_params=anthropic_optional,
            logging_obj=None,
            stream=False,
            kwargs=raw_kwargs,
        )

    assert captured, "acreate was not called"
    # output_config arrives exactly once with the expected value.
    assert captured["output_config"] == {"format": "anthropic"}


@pytest.mark.asyncio
async def test_chat_completion_patch_kwargs_disjoint_from_optional_params():
    """Same invariant for the OpenAI chat-completion sibling builder."""
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])
    tool_calls = [
        {
            "type": "function",
            "id": "t",
            "name": "web_search",
            "input": {"query": "q"},
            "function": {"name": "web_search",
                         "arguments": '{"query":"q"}'},
        }
    ]
    with patch.object(
        WebSearchInterceptionLogger, "_execute_search", _noop_search
    ):
        patch_obj = await logger._build_chat_completion_request_patch(
            model="gpt-4o",
            messages=[{"role": "user", "content": "q"}],
            tool_calls=tool_calls,
            optional_params={
                "temperature": 0.5,
                "output_config": {"format": "openai"},
            },
            kwargs={
                "output_config": {"format": "openai"},
                "litellm_call_id": "lit-3400-chat",
            },
            response_format="openai",
        )

    overlap = set(patch_obj.optional_params) & set(patch_obj.kwargs)
    assert overlap == set(), (
        f"chat completion patch.optional_params and patch.kwargs "
        f"must be disjoint; overlap was {overlap}"
    )
    assert (
        patch_obj.optional_params["output_config"] == {"format": "openai"}
    )


@pytest.mark.asyncio
async def test_anthropic_patch_preserves_non_overlapping_kwargs():
    """The dedupe step must not drop kwargs that are NOT in optional_params."""
    logger = WebSearchInterceptionLogger(enabled_providers=["anthropic"])
    anthropic_optional = {
        "max_tokens": 1024,
        "output_config": {"format": "anthropic"},
    }
    raw_kwargs = {
        "output_config": {"format": "anthropic"},
        # These are unique to raw kwargs and must survive.
        "litellm_call_id": "keep-me",
        "custom_llm_provider": "anthropic",
        "api_base": "https://example.invalid",
    }
    tool_calls = [
        {"type": "tool_use", "id": "tool_1", "name": "web_search",
         "input": {"query": "q"}}
    ]
    with patch.object(
        WebSearchInterceptionLogger, "_execute_search", _noop_search
    ):
        patch_obj, _ = await logger._build_anthropic_request_patch(
            model="anthropic/claude-opus-4-6",
            messages=[{"role": "user", "content": "q"}],
            tool_calls=tool_calls,
            thinking_blocks=[],
            anthropic_messages_optional_request_params=anthropic_optional,
            logging_obj=None,
            kwargs=raw_kwargs,
        )

    # Non-overlapping kwargs are preserved.
    assert patch_obj.kwargs.get("litellm_call_id") == "keep-me"
    assert patch_obj.kwargs.get("custom_llm_provider") == "anthropic"
    assert patch_obj.kwargs.get("api_base") == "https://example.invalid"
    # Overlapping key removed from kwargs (lives in optional_params now).
    assert "output_config" not in patch_obj.kwargs
