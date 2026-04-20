"""
Tests for advisor tool model aliasing on the /v1/messages path.

Scenario: an operator remaps the advisor tool's model via
``model_group_alias`` (e.g. ``claude-opus-4-7 -> o3``) because the client
(Claude Code) hardcodes the advisor ``model`` field to ``claude-opus-4-7``.

Required behaviour:
  * ``AdvisorOrchestrationHandler.can_handle`` intercepts the request when
    the executor is direct Anthropic but the advisor tool resolves to a
    non-native advisor model.
  * ``handle()`` dispatches the advisor sub-call with the *resolved* model,
    but every client-visible surface (``iterations[].model``) keeps the
    original alias so the remap is opaque to the caller.
  * ``_normalize_anthropic_advisor_tool_models`` never forwards a
    non-Anthropic model to the native API — it leaves the alias untouched
    if the resolved model is not natively supported.
"""

from typing import Dict
from unittest.mock import AsyncMock, patch

import pytest

ADVISOR_TOOL_ALIAS = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-7",
}

MESSAGES = [
    {"role": "user", "content": "Write a Python function that checks if a number is prime."}
]


def _make_text_response(text: str, model: str = "openai/o3") -> Dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _make_advisor_tool_use_response(
    question: str = "How should I approach this?",
    tool_id: str = "toolu_advisor_01",
    model: str = "claude-opus-4-7",
) -> Dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "consult_advisor",
                "input": {"question": question},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }


# ---------------------------------------------------------------------------
# 1. can_handle: alias -> non-native forces interception even on anthropic
# ---------------------------------------------------------------------------


def test_can_handle_alias_to_non_native_intercepts_on_anthropic():
    """
    When the advisor tool's model (``claude-opus-4-7``) aliases to a
    non-Anthropic model (``o3``), the handler must intercept even though the
    executor provider is direct Anthropic — Anthropic's native advisor tool
    can't run ``o3`` for us.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.resolve_proxy_model_alias_to_litellm_model",
        return_value="openai/o3",
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.supports_native_advisor_tool",
        return_value=False,
    ):
        h = AdvisorOrchestrationHandler()
        assert h.can_handle([ADVISOR_TOOL_ALIAS], "anthropic") is True


def test_can_handle_alias_to_native_still_defers_to_anthropic():
    """
    When the advisor tool's model aliases to a still-native Anthropic model
    (e.g. someone maps ``claude-opus-4-7 -> claude-opus-4-6``), the native
    Anthropic server-side advisor can still handle it — we must not
    intercept.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.resolve_proxy_model_alias_to_litellm_model",
        return_value="anthropic/claude-opus-4-6",
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.supports_native_advisor_tool",
        return_value=True,
    ):
        h = AdvisorOrchestrationHandler()
        assert h.can_handle([ADVISOR_TOOL_ALIAS], "anthropic") is False


def test_can_handle_non_anthropic_executor_always_intercepts():
    """
    Non-Anthropic executors always need orchestration regardless of the
    advisor tool's resolved model — no behaviour change from before.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    h = AdvisorOrchestrationHandler()
    assert h.can_handle([ADVISOR_TOOL_ALIAS], "openai") is True
    assert h.can_handle([ADVISOR_TOOL_ALIAS], "bedrock") is True


# ---------------------------------------------------------------------------
# 2. handle(): alias vs resolved separation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_uses_resolved_model_for_subcall_and_alias_for_iterations():
    """
    The advisor sub-call receives the *resolved* model (``openai/o3``) so
    routing and cost lookup hit the real deployment, while every
    client-visible ``iterations[].model`` entry of type ``advisor_message``
    keeps the original alias (``claude-opus-4-7``). The alias must never
    leak to the sub-call, and the resolved name must never leak to the
    response.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    advisor_tool_use_resp = _make_advisor_tool_use_response(
        question="What algorithm should I use?",
        tool_id="toolu_01",
    )
    advisor_advice_resp = _make_text_response(
        "Use a sieve for large n, trial division for small n.",
        model="openai/o3",
    )
    final_resp = _make_text_response(
        "def is_prime(n): ...",
        model="claude-opus-4-7",
    )

    executor_call_count = 0

    async def mock_messages(model, messages, tools, stream, max_tokens, **kwargs):
        nonlocal executor_call_count
        executor_call_count += 1
        if executor_call_count == 1:
            return advisor_tool_use_resp
        return final_resp

    advisor_mock = AsyncMock(return_value=advisor_advice_resp)

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_messages,
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_advisor_with_router",
        advisor_mock,
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.resolve_proxy_model_alias_to_litellm_model",
        return_value="openai/o3",
    ):
        h = AdvisorOrchestrationHandler()
        result = await h.handle(
            model="claude-opus-4-6",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL_ALIAS],
            stream=False,
            max_tokens=512,
            custom_llm_provider="anthropic",
        )

    assert advisor_mock.await_count == 1
    advisor_call_kwargs = advisor_mock.await_args.kwargs
    assert advisor_call_kwargs["model"] == "openai/o3", (
        "Advisor sub-call must use the resolved router model, not the alias"
    )

    usage = result.get("usage", {})
    iterations = usage.get("iterations", [])
    advisor_iterations = [
        it for it in iterations if it.get("type") == "advisor_message"
    ]
    assert len(advisor_iterations) == 1
    assert advisor_iterations[0]["model"] == "claude-opus-4-7", (
        "iterations[].model must preserve the client-facing alias"
    )
    # Resolved model must never appear anywhere in the iterations surface.
    for it in iterations:
        assert it.get("model") != "openai/o3"


@pytest.mark.asyncio
async def test_handle_without_alias_still_works():
    """
    When ``resolve_proxy_model_alias_to_litellm_model`` returns ``""`` (no
    alias configured), ``handle()`` must fall back to using the tool's
    original model for both the sub-call and the iteration entry. Nothing
    regresses for users who don't configure an alias.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorOrchestrationHandler,
    )

    advisor_tool = {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": "openai/gpt-4o-mini",
    }

    advisor_tool_use_resp = _make_advisor_tool_use_response()
    advisor_advice_resp = _make_text_response("advice", model="openai/gpt-4o-mini")
    final_resp = _make_text_response("final")

    executor_call_count = 0

    async def mock_messages(model, messages, tools, stream, max_tokens, **kwargs):
        nonlocal executor_call_count
        executor_call_count += 1
        return advisor_tool_use_resp if executor_call_count == 1 else final_resp

    advisor_mock = AsyncMock(return_value=advisor_advice_resp)

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_messages,
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_advisor_with_router",
        advisor_mock,
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor.resolve_proxy_model_alias_to_litellm_model",
        return_value="",
    ):
        h = AdvisorOrchestrationHandler()
        result = await h.handle(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[advisor_tool],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )

    assert advisor_mock.await_args.kwargs["model"] == "openai/gpt-4o-mini"
    advisor_iterations = [
        it
        for it in result["usage"]["iterations"]
        if it.get("type") == "advisor_message"
    ]
    assert advisor_iterations[0]["model"] == "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# 3. _normalize_anthropic_advisor_tool_models defensive guard
# ---------------------------------------------------------------------------


def test_normalize_leaves_alias_when_resolved_model_is_non_native():
    """
    Defensive guard: if the alias resolves to a non-Anthropic advisor model,
    the normalizer must leave the caller's original alias in place rather
    than substituting the unsupported model into the Anthropic request body.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        _normalize_anthropic_advisor_tool_models,
    )

    tools = [
        {
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "claude-opus-4-7",
        }
    ]

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.transformation.resolve_proxy_model_alias_to_litellm_model",
        return_value="openai/o3",
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.transformation.supports_native_advisor_tool",
        return_value=False,
    ):
        normalized = _normalize_anthropic_advisor_tool_models(tools)

    assert normalized[0]["model"] == "claude-opus-4-7", (
        "Normalizer must not push the non-native resolved model to Anthropic"
    )


def test_normalize_strips_anthropic_prefix_when_resolved_model_is_native():
    """
    Regression: for the classic path (alias resolves to a native Anthropic
    model), the normalizer still strips the ``anthropic/`` prefix so the
    Anthropic API receives a bare model name.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
        _normalize_anthropic_advisor_tool_models,
    )

    tools = [
        {
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "claude_opus",
        }
    ]

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.transformation.resolve_proxy_model_alias_to_litellm_model",
        return_value="anthropic/claude-opus-4-6",
    ), patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.transformation.supports_native_advisor_tool",
        return_value=True,
    ):
        normalized = _normalize_anthropic_advisor_tool_models(tools)

    assert normalized[0]["model"] == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# 4. Advisor sub-call uses the same /v1/messages → completion translation path
# ---------------------------------------------------------------------------


def test_prepare_completion_kwargs_moves_thinking_out_of_content():
    """
    Advisor sub-calls must use ``LiteLLMMessagesToCompletionTransformationHandler``
    (same as non-Anthropic ``/v1/messages``), so interleaved ``thinking`` blocks
    become OpenAI-shaped messages — never raw ``content[].type == "thinking"``.
    """
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret reasoning", "signature": "s"},
                {"type": "redacted_thinking", "data": "redacted"},
                {"type": "text", "text": "hello"},
            ],
        },
    ]
    completion_kwargs, _ = (
        LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
            max_tokens=100,
            messages=messages,
            model="openai/gpt-5-nano",
            stream=False,
        )
    )
    for msg in completion_kwargs["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    assert part.get("type") != "thinking"
                    assert part.get("type") != "redacted_thinking"


def test_build_advisor_context_preserves_string_content_and_plain_messages():
    """
    Messages with plain string content or only supported block types must be
    passed through unchanged.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        _build_advisor_context,
    )

    messages = [
        {"role": "user", "content": "plain string user"},
        {"role": "assistant", "content": [{"type": "text", "text": "plain reply"}]},
    ]
    executor_response = {"content": [{"type": "text", "text": "draft"}]}
    advisor_use_block = {
        "type": "tool_use",
        "name": "advisor",
        "input": {"question": "advise"},
    }

    result = _build_advisor_context(messages, executor_response, advisor_use_block)

    assert result[0] == {"role": "user", "content": "plain string user"}
    assert result[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "plain reply"}],
    }
