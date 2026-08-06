"""Regression tests for the responses -> completion fallback bridge guard.

When the Responses API falls back to chat completions (no native responses
config), it must tag the forwarded ``litellm.completion`` / ``litellm.acompletion``
call with ``_skip_responses_api_bridge=True`` so ``completion()`` does not bridge
the request straight back to the Responses API and mutually recurse forever.

Both fallback paths are covered: the sync ``response_api_handler`` (``_is_async``
False) and the async ``async_response_api_handler`` (``_is_async`` True). The
module-level ``litellm.completion`` / ``litellm.acompletion`` are patched to
capture the forwarded kwargs; if the flag-setting line is removed the captured
kwargs lack the flag and these tests fail.
"""

import os
import sys
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.streaming_iterator import SyntheticResponsesAPIStreamingIterator
from litellm.types.utils import ModelResponse


class _StopForwarding(Exception):
    """Raised by the mocked (a)completion once the forwarded kwargs are captured."""


def test_sync_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
            )

    assert captured.get("_skip_responses_api_bridge") is True


@pytest.mark.asyncio
async def test_async_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion):
        coro = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
        )
        with pytest.raises(_StopForwarding):
            await coro

    assert captured.get("_skip_responses_api_bridge") is True


def _make_model_response(model: str = "qwen3.8-max") -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        object="chat.completion",
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    )


def _make_logging_obj() -> MagicMock:
    logging_obj: Final = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.start_time = None
    logging_obj.model_call_details = {}
    return logging_obj


def test_sync_stream_true_with_model_response_returns_streaming_iterator():
    """Regression: when stream=True but the provider returns ModelResponse (non-streaming),
    the handler must return a SyntheticResponsesAPIStreamingIterator, not a bare
    ResponsesAPIResponse.  Without the fix, the proxy streaming hooks crash with:
      'async for' requires an object with __aiter__ method, got ResponsesAPIResponse
    """
    handler: Final = LiteLLMCompletionTransformationHandler()
    logging_obj: Final = _make_logging_obj()

    with patch("litellm.completion", return_value=_make_model_response()):
        result: Final = handler.response_api_handler(
            model="dashscope/qwen3.8-max",
            input="hello",
            responses_api_request={},
            custom_llm_provider="dashscope",
            _is_async=False,
            stream=True,
            litellm_logging_obj=logging_obj,
        )

    assert isinstance(
        result, SyntheticResponsesAPIStreamingIterator
    ), f"Expected SyntheticResponsesAPIStreamingIterator, got {type(result).__name__}"
    assert hasattr(result, "__aiter__"), "Result must be async-iterable"


@pytest.mark.asyncio
async def test_async_stream_true_with_model_response_returns_streaming_iterator():
    """Same regression check for the async path (aresponses / _is_async=True)."""
    handler: Final = LiteLLMCompletionTransformationHandler()
    logging_obj: Final = _make_logging_obj()

    async def fake_acompletion(**kwargs):
        return _make_model_response()

    with patch("litellm.acompletion", fake_acompletion):
        coro: Final = handler.response_api_handler(
            model="dashscope/qwen3.8-max",
            input="hello",
            responses_api_request={},
            custom_llm_provider="dashscope",
            _is_async=True,
            stream=True,
            litellm_logging_obj=logging_obj,
        )
        result: Final = await coro

    assert isinstance(
        result, SyntheticResponsesAPIStreamingIterator
    ), f"Expected SyntheticResponsesAPIStreamingIterator, got {type(result).__name__}"
    assert hasattr(result, "__aiter__"), "Result must be async-iterable"
