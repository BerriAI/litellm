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
import litellm

ADVISOR_TOOL = {
    "type": "advisor_20260301",
    "name": "advisor",
    "model": "claude-opus-4-6",
}

MESSAGES = [{"role": "user", "content": "Write a Python function to check if a number is prime."}]


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
                "name": "consult_advisor",
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
            return _advisor_call_resp()   # executor: calls advisor
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

    # 2 calls to _call_messages_handler: executor -> executor-final.
    # Advisor subcall runs via _call_advisor_with_router (acompletion path).
    assert call_count == 2

    assert isinstance(result, dict)
    content = result.get("content", [])
    text_blocks = [b for b in content if b.get("type") == "text"]
    advisor_uses = [b for b in content if b.get("type") == "tool_use" and b.get("name") == "advisor"]

    assert len(text_blocks) >= 1, "Final response must have text"
    assert len(advisor_uses) == 0, "No advisor tool_use blocks must appear in final output"


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


@pytest.mark.asyncio
async def test_messages_interceptor_path_still_runs_non_advisor_pre_request_hooks():
    """
    Pre-request hooks should still run for intercepted /messages requests, while
    advisor tool conversion must be skipped so interceptors can still see the
    native advisor_20260301 tool.
    """
    from litellm.integrations.advisor_interception.handler import (
        AdvisorInterceptionLogger,
    )
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    class _NonAdvisorHookLogger(CustomLogger):
        async def async_pre_request_hook(self, model, messages, kwargs):
            kwargs["metadata"] = {"from_non_advisor_hook": True}
            return kwargs

    class _CaptureInterceptor:
        def __init__(self):
            self.seen_kwargs = {}
            self.seen_tools = None

        def can_handle(self, tools, custom_llm_provider):
            return True

        async def handle(self, **kwargs):
            self.seen_kwargs = kwargs
            self.seen_tools = kwargs.get("tools")
            return _text_resp("interceptor handled request")

    capture_interceptor = _CaptureInterceptor()
    original_callbacks = litellm.callbacks
    try:
        litellm.callbacks = [
            AdvisorInterceptionLogger(default_advisor_model="claude-opus-4-6"),
            _NonAdvisorHookLogger(),
        ]
        with patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.handler.get_messages_interceptors",
            return_value=[capture_interceptor],
        ):
            result = await anthropic_messages(
                model="openai/gpt-4o-mini",
                messages=MESSAGES,
                tools=[ADVISOR_TOOL],
                stream=False,
                max_tokens=256,
                custom_llm_provider="openai",
            )
    finally:
        litellm.callbacks = original_callbacks

    assert isinstance(result, dict)
    assert capture_interceptor.seen_kwargs.get("metadata") == {
        "from_non_advisor_hook": True
    }
    assert isinstance(capture_interceptor.seen_tools, list)
    assert capture_interceptor.seen_tools[0].get("type") == "advisor_20260301"
