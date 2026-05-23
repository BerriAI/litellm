"""
Tests for LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed.

Covers the chatgpt provider whitelist added alongside the encrypted-reasoning
round-trip fix (Phase 3.1): the chatgpt subscription/Codex backend must be
treated identically to openai for thinking-enabled requests so it gets routed
through the Responses API rather than chat completions.
"""

from unittest.mock import patch

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler as H,
)


def test_chatgpt_provider_routed_through_responses_api():
    """chatgpt + thinking enabled → model gets `responses/` prefix and
    reasoning_effort dict is built. This is the Phase 3.1 whitelist
    addition; before the fix, only openai got this treatment and chatgpt
    callers silently lost reasoning continuity."""
    kwargs = {
        "model": "gpt-5.5",
        "custom_llm_provider": "chatgpt",
        "reasoning_effort": "high",
    }
    H._route_openai_thinking_to_responses_api_if_needed(
        kwargs, thinking={"type": "enabled", "budget_tokens": 10000}
    )
    assert kwargs["model"] == "responses/gpt-5.5"
    assert kwargs["reasoning_effort"] == {"effort": "high"}


def test_openai_provider_still_routed_through_responses_api():
    """Regression guard: the chatgpt addition must not break the existing
    openai routing."""
    kwargs = {
        "model": "gpt-5",
        "custom_llm_provider": "openai",
        "reasoning_effort": "medium",
    }
    H._route_openai_thinking_to_responses_api_if_needed(
        kwargs, thinking={"type": "enabled", "budget_tokens": 5000}
    )
    assert kwargs["model"] == "responses/gpt-5"
    assert kwargs["reasoning_effort"] == {"effort": "medium"}


def test_anthropic_provider_early_returns_unchanged():
    """The whitelist guard at L74 must early-return for any provider not in
    {openai, chatgpt}. Native anthropic routing has its own thinking handling."""
    kwargs = {
        "model": "claude-opus-4-7",
        "custom_llm_provider": "anthropic",
        "reasoning_effort": "high",
    }
    H._route_openai_thinking_to_responses_api_if_needed(
        kwargs, thinking={"type": "enabled", "budget_tokens": 10000}
    )
    # Untouched: no model rewrite, reasoning_effort stays a bare string
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["reasoning_effort"] == "high"


def test_glm_provider_early_returns_unchanged():
    """zai / glm / any other non-{openai,chatgpt} provider is untouched."""
    kwargs = {
        "model": "glm-4.5",
        "custom_llm_provider": "zai",
        "reasoning_effort": "low",
    }
    H._route_openai_thinking_to_responses_api_if_needed(
        kwargs, thinking={"type": "enabled", "budget_tokens": 2000}
    )
    assert kwargs["model"] == "glm-4.5"
    assert kwargs["reasoning_effort"] == "low"


def test_chatgpt_thinking_disabled_does_not_rewrite_model():
    """Provider whitelist passes, but the next gate (`thinking.type == "enabled"`)
    must early-return when thinking is disabled."""
    kwargs = {
        "model": "gpt-5.5",
        "custom_llm_provider": "chatgpt",
        "reasoning_effort": "high",
    }
    H._route_openai_thinking_to_responses_api_if_needed(
        kwargs, thinking={"type": "disabled"}
    )
    assert kwargs["model"] == "gpt-5.5"
    # reasoning_effort untouched
    assert kwargs["reasoning_effort"] == "high"


def test_chatgpt_already_responses_prefixed_model_not_double_prefixed():
    kwargs = {
        "model": "responses/gpt-5.5",
        "custom_llm_provider": "chatgpt",
        "reasoning_effort": "high",
    }
    H._route_openai_thinking_to_responses_api_if_needed(
        kwargs, thinking={"type": "enabled", "budget_tokens": 10000}
    )
    assert kwargs["model"] == "responses/gpt-5.5"


def test_chatgpt_provider_inferred_from_model_when_missing():
    """custom_llm_provider may be absent; the function infers it via
    litellm.utils.get_llm_provider. Patch that to return chatgpt and verify
    routing still kicks in."""
    kwargs = {
        "model": "gpt-5.5",
        "reasoning_effort": "high",
    }
    with patch(
        "litellm.utils.get_llm_provider",
        return_value=("gpt-5.5", "chatgpt", None, None),
    ):
        H._route_openai_thinking_to_responses_api_if_needed(
            kwargs, thinking={"type": "enabled", "budget_tokens": 10000}
        )
    assert kwargs["model"] == "responses/gpt-5.5"
