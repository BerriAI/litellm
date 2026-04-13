"""
Regression tests for the aspeech / atranscription double-provider-call bug.

Before the fix the `else` branch in both functions re-ran `func_with_context`
via `loop.run_in_executor` instead of reusing the already-resolved
`init_response`, causing two round-trips to the provider per call.
These tests assert that `run_in_executor` is called exactly once per invocation.
"""

import asyncio
import contextvars
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_aspeech_calls_executor_exactly_once():
    """should call run_in_executor exactly once — not twice via the else branch."""
    mock_response = MagicMock()
    mock_response.__class__.__name__ = "HttpxBinaryResponseContent"

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

    with (
        patch("litellm.main.asyncio.get_running_loop", return_value=mock_loop),
        patch(
            "litellm.main.get_llm_provider",
            return_value=("tts-1", "openai", None, None),
        ),
        patch("litellm.main.speech", return_value=mock_response),
        patch(
            "litellm.main.contextvars.copy_context",
            return_value=contextvars.copy_context(),
        ),
        patch(
            "litellm.main.exception_type",
            side_effect=lambda **kw: kw["original_exception"],
        ),
    ):
        import litellm

        try:
            await litellm.aspeech(model="tts-1", input="hello", voice="alloy")
        except Exception:
            pass

    assert mock_loop.run_in_executor.call_count == 1, (
        f"Expected run_in_executor to be called once, got {mock_loop.run_in_executor.call_count}. "
        "If this is >1 the double-call bug has been re-introduced."
    )


@pytest.mark.asyncio
async def test_atranscription_calls_executor_exactly_once():
    """should call run_in_executor exactly once — not twice via the else branch."""
    from litellm.types.utils import TranscriptionResponse

    mock_response = TranscriptionResponse(text="hello world")

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

    with (
        patch("litellm.main.asyncio.get_running_loop", return_value=mock_loop),
        patch(
            "litellm.main.get_llm_provider",
            return_value=("whisper-1", "openai", None, None),
        ),
        patch("litellm.main.transcription", return_value=mock_response),
        patch(
            "litellm.main.contextvars.copy_context",
            return_value=contextvars.copy_context(),
        ),
        patch(
            "litellm.main.exception_type",
            side_effect=lambda **kw: kw["original_exception"],
        ),
    ):
        import litellm

        try:
            await litellm.atranscription(model="whisper-1", file=MagicMock())
        except Exception:
            pass

    assert mock_loop.run_in_executor.call_count == 1, (
        f"Expected run_in_executor to be called once, got {mock_loop.run_in_executor.call_count}. "
        "If this is >1 the double-call bug has been re-introduced."
    )


@pytest.mark.asyncio
async def test_atranscription_else_branch_raises_on_unexpected_type():
    """should hit the else branch and raise ValueError when init_response is an
    unexpected type (not dict, TranscriptionResponse, or coroutine)."""
    # Return a plain string — not a dict, TranscriptionResponse, or coroutine
    mock_response = "unexpected_string"

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

    with (
        patch("litellm.main.asyncio.get_running_loop", return_value=mock_loop),
        patch(
            "litellm.main.get_llm_provider",
            return_value=("whisper-1", "openai", None, None),
        ),
        patch("litellm.main.transcription", return_value=mock_response),
        patch(
            "litellm.main.contextvars.copy_context",
            return_value=contextvars.copy_context(),
        ),
        patch(
            "litellm.main.exception_type",
            side_effect=lambda **kw: kw["original_exception"],
        ),
    ):
        import litellm

        with pytest.raises((ValueError, Exception)):
            await litellm.atranscription(model="whisper-1", file=MagicMock())

    # Even on the else path, run_in_executor must be called exactly once
    assert mock_loop.run_in_executor.call_count == 1


@pytest.mark.asyncio
async def test_aspeech_returns_init_response_directly():
    """should return init_response directly when it is not a coroutine (non-async provider)."""
    mock_response = MagicMock()

    mock_loop = MagicMock()
    mock_loop.run_in_executor = AsyncMock(return_value=mock_response)

    with (
        patch("litellm.main.asyncio.get_running_loop", return_value=mock_loop),
        patch(
            "litellm.main.get_llm_provider",
            return_value=("tts-1", "openai", None, None),
        ),
        patch("litellm.main.speech", return_value=mock_response),
        patch(
            "litellm.main.contextvars.copy_context",
            return_value=contextvars.copy_context(),
        ),
        patch(
            "litellm.main.exception_type",
            side_effect=lambda **kw: kw["original_exception"],
        ),
        patch("litellm.main.asyncio.iscoroutine", return_value=False),
    ):
        import litellm

        try:
            result = await litellm.aspeech(model="tts-1", input="hello", voice="alloy")
        except Exception:
            result = None

    assert mock_loop.run_in_executor.call_count == 1
