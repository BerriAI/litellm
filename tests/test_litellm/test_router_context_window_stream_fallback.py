"""
Tests for ContextWindowExceededError handling in streaming iterators.

When certain inference frameworks (e.g. SGLang) return HTTP 200 for the
initial request but embed a context-window-exceeded error *inside* the
SSE stream, the Router's streaming iterators must catch the resulting
``ContextWindowExceededError`` and route it through the fallback system
(``context_window_fallbacks``) instead of letting it bubble up to the
caller.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm


# ---------------------------------------------------------------------------
# Async streaming – ContextWindowExceededError triggers fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acompletion_streaming_context_window_error_triggers_fallback():
    """Async streaming: ContextWindowExceededError raised mid-stream triggers
    context_window_fallbacks via async_function_with_fallbacks_common_utils.

    This mirrors the pattern tested for MidStreamFallbackError in
    test_acompletion_streaming_iterator (test_router.py).
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "sglang-model",
                "litellm_params": {
                    "model": "openai/sglang-model",
                    "api_key": "fake-key-1",
                },
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/fallback-model",
                    "api_key": "fake-key-2",
                },
            },
        ],
        context_window_fallbacks=[{"sglang-model": ["fallback-model"]}],
    )

    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "sglang-model", "stream": True}

    ctx_error = litellm.ContextWindowExceededError(
        message="The input (85066 tokens) is longer than the model's context length (81920 tokens).",
        model="sglang-model",
        llm_provider="openai",
    )

    # Async iterator that raises ContextWindowExceededError immediately
    class AsyncIteratorWithContextError:
        def __init__(self):
            self.model = "sglang-model"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ctx_error

    mock_response = AsyncIteratorWithContextError()

    # Fallback yields two chunks
    fallback_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="fallback"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" response"))]),
    ]

    class AsyncFallbackIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    mock_fallback_response = AsyncFallbackIterator(fallback_chunks)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=mock_fallback_response,
    ) as mock_fallback_utils:
        result = await router._acompletion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        collected_chunks = []
        async for chunk in result:
            collected_chunks.append(chunk)

        # Verify fallback was called
        assert mock_fallback_utils.called
        call_args = mock_fallback_utils.call_args

        # Verify the exception was passed through
        assert call_args.kwargs["e"] is ctx_error

        # Verify disable_fallbacks is False (fallbacks should be enabled)
        assert call_args.kwargs["disable_fallbacks"] is False

        # Verify model_group
        assert call_args.kwargs["model_group"] == "sglang-model"

        # Verify original messages are used (no continuation prompt for
        # context window errors, unlike MidStreamFallbackError)
        fallback_kwargs = call_args.kwargs["kwargs"]
        assert fallback_kwargs["messages"] == messages

        # Verify original_function is _acompletion (async)
        assert fallback_kwargs["original_function"] == router._acompletion

        # Verify we received the fallback chunks
        assert len(collected_chunks) == 2
        assert collected_chunks == fallback_chunks


@pytest.mark.asyncio
async def test_acompletion_streaming_context_window_error_after_partial_chunks():
    """Async streaming: ContextWindowExceededError raised after yielding some
    chunks still triggers fallback.

    SGLang may return a few empty or partial chunks before the error appears
    in the SSE stream.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "sglang-model",
                "litellm_params": {
                    "model": "openai/sglang-model",
                    "api_key": "fake-key",
                },
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/fallback-model",
                    "api_key": "fake-key-2",
                },
            },
        ],
        context_window_fallbacks=[{"sglang-model": ["fallback-model"]}],
    )

    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "sglang-model", "stream": True}

    ctx_error = litellm.ContextWindowExceededError(
        message="Context length exceeded",
        model="sglang-model",
        llm_provider="openai",
    )

    initial_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content=""))]),
    ]

    class AsyncIteratorErrorAfterChunks:
        def __init__(self, items, error_after_index):
            self.items = items
            self.index = 0
            self.error_after_index = error_after_index
            self.model = "sglang-model"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= self.error_after_index:
                raise ctx_error
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    mock_response = AsyncIteratorErrorAfterChunks(initial_chunks, 1)

    fallback_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="ok"))]),
    ]

    class AsyncFallbackIterator:
        def __init__(self, items):
            self.items = items
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    mock_fallback_response = AsyncFallbackIterator(fallback_chunks)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=mock_fallback_response,
    ) as mock_fallback_utils:
        result = await router._acompletion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        collected_chunks = []
        async for chunk in result:
            collected_chunks.append(chunk)

        assert mock_fallback_utils.called
        # 1 initial chunk + 1 fallback chunk
        assert len(collected_chunks) == 2


@pytest.mark.asyncio
async def test_acompletion_streaming_context_window_fallback_failure_raises():
    """When context_window_fallback itself fails, the fallback error is raised."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "sglang-model",
                "litellm_params": {
                    "model": "openai/sglang-model",
                    "api_key": "fake-key",
                },
            },
        ],
    )

    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "sglang-model", "stream": True}

    ctx_error = litellm.ContextWindowExceededError(
        message="Context length exceeded",
        model="sglang-model",
        llm_provider="openai",
    )

    class AsyncIteratorImmediateError:
        def __init__(self):
            self.model = "sglang-model"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ctx_error

    mock_response = AsyncIteratorImmediateError()

    fallback_error = Exception("No fallback models available")

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        side_effect=fallback_error,
    ):
        result = await router._acompletion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        with pytest.raises(Exception, match="No fallback models available"):
            async for _ in result:
                pass


# ---------------------------------------------------------------------------
# Sync streaming – ContextWindowExceededError triggers fallback
# ---------------------------------------------------------------------------


def test_completion_streaming_context_window_error_triggers_fallback():
    """Sync streaming: ContextWindowExceededError raised mid-stream triggers
    context_window_fallbacks via function_with_fallbacks.

    This mirrors test_completion_streaming_iterator_fallback_on_429 in
    test_router.py.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "sglang-model",
                "litellm_params": {
                    "model": "openai/sglang-model",
                    "api_key": "fake-key-1",
                },
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/fallback-model",
                    "api_key": "fake-key-2",
                },
            },
        ],
        context_window_fallbacks=[{"sglang-model": ["fallback-model"]}],
    )

    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "sglang-model", "stream": True}

    ctx_error = litellm.ContextWindowExceededError(
        message="The input (85066 tokens) is longer than the model's context length (81920 tokens).",
        model="sglang-model",
        llm_provider="openai",
    )

    class SyncIteratorImmediateError:
        def __init__(self):
            self.model = "sglang-model"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()

        def __iter__(self):
            return self

        def __next__(self):
            raise ctx_error

    mock_response = SyncIteratorImmediateError()

    # Fallback returns a simple iterable
    fallback_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="fallback"))]),
    ]
    mock_fallback_response = MagicMock()
    mock_fallback_response.__iter__ = MagicMock(return_value=iter(fallback_chunks))

    with patch.object(
        router,
        "function_with_fallbacks",
        return_value=mock_fallback_response,
    ) as mock_fallback:
        result = router._completion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        list(result)

        # Verify fallback was called
        assert mock_fallback.called
        call_kwargs = mock_fallback.call_args

        # Verify original messages are used
        assert call_kwargs.kwargs.get("messages") == messages

        # Verify original_function is _completion (sync)
        assert call_kwargs.kwargs.get("original_function") == router._completion


def test_completion_streaming_context_window_fallback_failure_raises():
    """Sync streaming: when context_window_fallback itself fails, error is raised."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "sglang-model",
                "litellm_params": {
                    "model": "openai/sglang-model",
                    "api_key": "fake-key",
                },
            },
        ],
    )

    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "sglang-model", "stream": True}

    ctx_error = litellm.ContextWindowExceededError(
        message="Context length exceeded",
        model="sglang-model",
        llm_provider="openai",
    )

    class SyncIteratorImmediateError:
        def __init__(self):
            self.model = "sglang-model"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()

        def __iter__(self):
            return self

        def __next__(self):
            raise ctx_error

    mock_response = SyncIteratorImmediateError()

    fallback_error = Exception("No fallback models available")

    with patch.object(
        router,
        "function_with_fallbacks",
        side_effect=fallback_error,
    ):
        result = router._completion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        with pytest.raises(Exception, match="No fallback models available"):
            list(result)
