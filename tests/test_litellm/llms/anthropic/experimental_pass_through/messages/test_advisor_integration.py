"""
Integration tests for advisor orchestration through the full /messages handler.

These tests exercise the real dispatch path:
  anthropic_messages() → interceptor registry → AdvisorOrchestrationHandler.handle()

The only thing mocked is _call_messages_handler (the outbound LLM call), so the
interceptor detection, loop logic, and message assembly all run for real.
"""

from typing import Dict
from unittest.mock import patch

import pytest

ADVISOR_TOOL = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-6",
}

MESSAGES = [
    {
        "role": "user",
        "content": "Write a Python function to check if a number is prime.",
    }
]


def _text_resp(text: str, model: str = "gpt-4o-mini") -> Dict:
    return {
        "id": "msg_int_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _advisor_call_resp(
    question: str = "How do I approach this?", tool_id: str = "tid_01"
) -> Dict:
    return {
        "id": "msg_int_test",
        "type": "message",
        "role": "assistant",
        "model": "gpt-4o-mini",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "advisor",
                "input": {"question": question},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }


# ---------------------------------------------------------------------------
# 1. Full dispatch: interceptor fires and orchestration loop completes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_dispatch_interceptor_fires_and_loop_completes():
    """
    Call anthropic_messages() with an openai model + advisor_20260301 tool.
    The interceptor must fire, run the loop (1 advisor call), and return a
    clean final response with no advisor tool_use blocks.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    call_count = 0

    async def mock_handler(model, messages, tools, stream, max_tokens, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _advisor_call_resp()  # executor: calls advisor
        if call_count == 2:
            return _text_resp("Use trial division.", model="claude-opus-4-6")  # advisor
        return _text_resp("def is_prime(n): ...")  # executor: final

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_handler,
    ):
        result = await anthropic_messages(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
        )

    # 3 internal calls: executor → advisor → executor-final
    assert call_count == 3

    assert isinstance(result, dict)
    content = result.get("content", [])
    text_blocks = [b for b in content if b.get("type") == "text"]
    advisor_uses = [
        b for b in content if b.get("type") == "tool_use" and b.get("name") == "advisor"
    ]

    assert len(text_blocks) >= 1, "Final response must have text"
    assert (
        len(advisor_uses) == 0
    ), "No advisor tool_use blocks must appear in final output"


# ---------------------------------------------------------------------------
# 2. max_uses enforced through the full handler path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_uses_enforced_through_full_handler():
    """
    AdvisorMaxIterationsError propagates out of anthropic_messages() when
    the executor keeps calling the advisor past max_uses.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor import (
        AdvisorMaxIterationsError,
    )

    advisor_tool_capped = {**ADVISOR_TOOL, "max_uses": 1}

    async def mock_handler(model, messages, tools, stream, max_tokens, **kwargs):
        # Advisor always returns text; executor always calls advisor
        if tools is None:
            return _text_resp("Some advice.", model="claude-opus-4-6")
        return _advisor_call_resp()

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_handler,
    ):
        with pytest.raises(AdvisorMaxIterationsError):
            await anthropic_messages(
                model="openai/gpt-4o-mini",
                messages=MESSAGES,
                tools=[advisor_tool_capped],
                stream=False,
                max_tokens=512,
                custom_llm_provider="openai",
            )


# ---------------------------------------------------------------------------
# 3. Anthropic provider bypasses interceptor — no orchestration loop runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_provider_bypasses_interceptor():
    """
    With custom_llm_provider='anthropic', the interceptor must NOT fire.
    The advisor_20260301 tool is forwarded as-is to the underlying handler.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    direct_response = _text_resp("Native anthropic response.")

    # Patch the non-interceptor code path — anthropic_messages_handler
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.handler.anthropic_messages_handler",
        return_value=direct_response,
    ) as mock_native:
        result = await anthropic_messages(
            model="claude-sonnet-4-6",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="anthropic",
        )

    # Native handler was called (not the orchestration loop)
    mock_native.assert_called_once()
    # Response passes through unmodified
    content = result.get("content", []) if isinstance(result, dict) else []
    text_blocks = [b for b in content if b.get("type") == "text"]
    assert any("Native anthropic" in b.get("text", "") for b in text_blocks)


# ---------------------------------------------------------------------------
# 4. Regression: top-level named params must be forwarded into executor sub-call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_named_params_forwarded_into_advisor_executor_subcall():
    """
    Regression test: ``thinking``, ``metadata``, ``system``, ``temperature``,
    ``stop_sequences``, ``tool_choice``, ``top_k``, ``top_p`` are bound as named
    parameters on ``anthropic_messages``. They must be forwarded to the
    interceptor handler so the advisor executor sub-call carries them through
    to the underlying provider.

    Without this forwarding, ``thinking={"type": "adaptive"}`` (and others)
    are silently dropped, causing 400s on providers whose validation depends on
    them, e.g. Vertex AI rejecting ``clear_thinking_20251015`` context_management
    edits with: ``strategy requires thinking to be enabled or adaptive``.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    captured_executor_kwargs: Dict = {}

    async def mock_handler(
        model, messages, tools, stream, max_tokens, custom_llm_provider, **kwargs
    ):
        # First call is the executor sub-call (returns advisor tool_use).
        # Capture its kwargs so we can assert the forwarded params.
        if not captured_executor_kwargs:
            captured_executor_kwargs.update(
                {
                    "thinking": kwargs.get("thinking"),
                    "metadata": kwargs.get("metadata"),
                    "system": kwargs.get("system"),
                    "temperature": kwargs.get("temperature"),
                    "stop_sequences": kwargs.get("stop_sequences"),
                    "tool_choice": kwargs.get("tool_choice"),
                    "top_k": kwargs.get("top_k"),
                    "top_p": kwargs.get("top_p"),
                }
            )
            return _advisor_call_resp()
        # Subsequent calls — terminate the loop.
        if tools is None:
            return _text_resp("Some advice.", model="claude-opus-4-6")
        return _text_resp("Final answer.")

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
        side_effect=mock_handler,
    ):
        await anthropic_messages(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
            thinking={"type": "adaptive"},
            metadata={"caller_field": "preserve_me"},
            system="You are a helpful assistant.",
            temperature=0.7,
            stop_sequences=["STOP"],
            tool_choice={"type": "auto"},
            top_k=40,
            top_p=0.9,
        )

    assert captured_executor_kwargs["thinking"] == {"type": "adaptive"}, (
        "thinking must be forwarded into executor sub-call — see "
        "anthropic_messages.handler interceptor invocation."
    )
    # The advisor enriches metadata with `advisor_sub_call` / `parent_request_id`,
    # but the original caller fields must survive into the executor sub-call.
    assert isinstance(captured_executor_kwargs["metadata"], dict)
    assert captured_executor_kwargs["metadata"].get("caller_field") == "preserve_me"
    assert captured_executor_kwargs["system"] == "You are a helpful assistant."
    assert captured_executor_kwargs["temperature"] == 0.7
    assert captured_executor_kwargs["stop_sequences"] == ["STOP"]
    assert captured_executor_kwargs["tool_choice"] == {"type": "auto"}
    assert captured_executor_kwargs["top_k"] == 40
    assert captured_executor_kwargs["top_p"] == 0.9


# ---------------------------------------------------------------------------
# 5. Regression: pre-request hook returning a named param must not cause
#    "got multiple values for keyword argument" at the interceptor dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_request_hook_override_does_not_collide_with_explicit_kwargs():
    """
    ``_execute_pre_request_hooks`` may return any subset of params. After
    extraction those values are also propagated as named kwargs into the
    interceptor, so the same key must not also appear in ``**kwargs`` (or the
    splat raises ``TypeError: got multiple values for keyword argument``).

    Regression for Greptile P2 on PR #27810.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    captured: Dict = {}

    async def mock_handler(
        model, messages, tools, stream, max_tokens, custom_llm_provider, **kwargs
    ):
        if not captured:
            captured.update(
                {
                    "thinking": kwargs.get("thinking"),
                    "system": kwargs.get("system"),
                    "temperature": kwargs.get("temperature"),
                }
            )
            return _advisor_call_resp()
        if tools is None:
            return _text_resp("Some advice.", model="claude-opus-4-6")
        return _text_resp("Final answer.")

    async def fake_pre_request_hooks(
        model, messages, tools, stream, custom_llm_provider, **hook_kwargs
    ):
        # Simulate a CustomLogger.async_pre_request_hook that overrides several
        # named params on its way through. Without the request_kwargs.pop()
        # extraction in handler.py, these would collide with the explicit
        # kwargs passed to interceptor.handle() (TypeError: got multiple
        # values for keyword argument).
        return {
            "tools": tools,
            "stream": stream,
            "litellm_params": {"custom_llm_provider": custom_llm_provider},
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "system": "Hook overrode the system prompt.",
            "temperature": 0.1,
        }

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.handler._execute_pre_request_hooks",
            side_effect=fake_pre_request_hooks,
        ),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.interceptors.advisor._call_messages_handler",
            side_effect=mock_handler,
        ),
    ):
        # Should not raise TypeError.
        await anthropic_messages(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            tools=[ADVISOR_TOOL],
            stream=False,
            max_tokens=512,
            custom_llm_provider="openai",
            thinking={"type": "adaptive"},
            system="Original system prompt.",
            temperature=0.9,
        )

    # Hook overrides win and reach the executor sub-call.
    assert captured["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert captured["system"] == "Hook overrode the system prompt."
    assert captured["temperature"] == 0.1
