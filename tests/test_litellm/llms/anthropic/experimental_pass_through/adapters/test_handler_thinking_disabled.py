"""Handler-level tests for ``thinking_disabled`` computation and threading.

Covers the boolean logic that decides whether thinking is disabled
(``thinking is None or thinking.type == "disabled"``) and verifies it is
threaded correctly to ``ANTHROPIC_ADAPTER`` output-translation calls for
both the async and sync handler entry points, in streaming and non-streaming
modes.

Mocks ``litellm.acompletion`` / ``litellm.completion`` and
``ANTHROPIC_ADAPTER`` directly, alongside the preparation helpers that run
before the ``thinking_disabled`` computation, so the tests are focused on
the computation and threading rather than the full request pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

THINKING_PARAMS = [
    (None, True),
    ({"type": "disabled"}, True),
    ({"type": "enabled", "budget_tokens": 1024}, False),
    ({"type": "adaptive"}, False),
]

MESSAGES = [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# Async handler — streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "thinking_param,expected_thinking_disabled",
    THINKING_PARAMS,
)
async def test_async_handler_streaming_threads_thinking_disabled(
    thinking_param, expected_thinking_disabled
):
    """Async handler, stream=True: ``thinking_disabled`` reaches the streaming
    adapter call."""
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler._prepare_context_managed_request",
            return_value=None,
        ),
        patch.object(
            LiteLLMMessagesToCompletionTransformationHandler,
            "_prepare_completion_kwargs",
            return_value=({}, {}),
        ),
        patch("litellm.acompletion", return_value=MagicMock()),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.ANTHROPIC_ADAPTER"
        ) as mock_adapter,
    ):
        mock_adapter.translate_completion_output_params_streaming.return_value = iter([])
        await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=100,
            messages=MESSAGES,
            model="gpt-4o",
            stream=True,
            thinking=thinking_param,
        )
        call_kwargs = (
            mock_adapter.translate_completion_output_params_streaming.call_args.kwargs
        )
        assert (
            call_kwargs.get("thinking_disabled") is expected_thinking_disabled
        ), f"thinking={thinking_param!r}: expected thinking_disabled={expected_thinking_disabled}"


# ---------------------------------------------------------------------------
# Async handler — non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "thinking_param,expected_thinking_disabled",
    THINKING_PARAMS,
)
async def test_async_handler_non_streaming_threads_thinking_disabled(
    thinking_param, expected_thinking_disabled
):
    """Async handler, stream=False: ``thinking_disabled`` reaches the
    non-streaming adapter call."""
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler._prepare_context_managed_request",
            return_value=None,
        ),
        patch.object(
            LiteLLMMessagesToCompletionTransformationHandler,
            "_prepare_completion_kwargs",
            return_value=({}, {}),
        ),
        patch("litellm.acompletion", return_value=MagicMock()),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.ANTHROPIC_ADAPTER"
        ) as mock_adapter,
    ):
        mock_adapter.translate_completion_output_params.return_value = MagicMock()
        await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=100,
            messages=MESSAGES,
            model="gpt-4o",
            stream=False,
            thinking=thinking_param,
        )
        call_kwargs = mock_adapter.translate_completion_output_params.call_args.kwargs
        assert (
            call_kwargs.get("thinking_disabled") is expected_thinking_disabled
        ), f"thinking={thinking_param!r}: expected thinking_disabled={expected_thinking_disabled}"


# ---------------------------------------------------------------------------
# Sync handler — streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "thinking_param,expected_thinking_disabled",
    THINKING_PARAMS,
)
def test_sync_handler_streaming_threads_thinking_disabled(
    thinking_param, expected_thinking_disabled
):
    """Sync handler, stream=True: ``thinking_disabled`` reaches the streaming
    adapter call.

    Uses the direct synchronous path (no ``context_management``, no compaction
    blocks) so ``run_async_function`` is never invoked.
    """
    with (
        patch.object(
            LiteLLMMessagesToCompletionTransformationHandler,
            "_prepare_completion_kwargs",
            return_value=({}, {}),
        ),
        patch("litellm.completion", return_value=MagicMock()),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.ANTHROPIC_ADAPTER"
        ) as mock_adapter,
    ):
        mock_adapter.translate_completion_output_params_streaming.return_value = iter([])
        LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
            max_tokens=100,
            messages=MESSAGES,
            model="gpt-4o",
            stream=True,
            thinking=thinking_param,
        )
        call_kwargs = (
            mock_adapter.translate_completion_output_params_streaming.call_args.kwargs
        )
        assert (
            call_kwargs.get("thinking_disabled") is expected_thinking_disabled
        ), f"thinking={thinking_param!r}: expected thinking_disabled={expected_thinking_disabled}"


# ---------------------------------------------------------------------------
# Sync handler — non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "thinking_param,expected_thinking_disabled",
    THINKING_PARAMS,
)
def test_sync_handler_non_streaming_threads_thinking_disabled(
    thinking_param, expected_thinking_disabled
):
    """Sync handler, stream=False: ``thinking_disabled`` reaches the
    non-streaming adapter call.

    Uses the direct synchronous path (no ``context_management``, no compaction
    blocks) so ``run_async_function`` is never invoked.
    """
    with (
        patch.object(
            LiteLLMMessagesToCompletionTransformationHandler,
            "_prepare_completion_kwargs",
            return_value=({}, {}),
        ),
        patch("litellm.completion", return_value=MagicMock()),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler.ANTHROPIC_ADAPTER"
        ) as mock_adapter,
    ):
        mock_adapter.translate_completion_output_params.return_value = MagicMock()
        LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
            max_tokens=100,
            messages=MESSAGES,
            model="gpt-4o",
            stream=False,
            thinking=thinking_param,
        )
        call_kwargs = mock_adapter.translate_completion_output_params.call_args.kwargs
        assert (
            call_kwargs.get("thinking_disabled") is expected_thinking_disabled
        ), f"thinking={thinking_param!r}: expected thinking_disabled={expected_thinking_disabled}"
