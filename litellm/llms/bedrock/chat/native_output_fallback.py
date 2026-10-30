from collections.abc import Callable, Coroutine
from functools import partial
from typing import Final, TypeAlias

from litellm.llms.anthropic.common_utils import is_claude_strict_tools_rejection
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.common_utils import (
    NativeOutputFallback,
    is_bedrock_native_output_rejection,
    warn_bedrock_native_output_rejected,
    warn_bedrock_strict_tools_dropped,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

NativeOutputResponse: TypeAlias = ModelResponse | CustomStreamWrapper
NativeOutputAttemptResult: TypeAlias = NativeOutputResponse | Coroutine[object, object, NativeOutputResponse]
NativeOutputAttempt: TypeAlias = Callable[[NativeOutputFallback | None], NativeOutputAttemptResult]
NativeOutputFallbackFactory: TypeAlias = Callable[[], NativeOutputFallback | None]
_RetryFor: TypeAlias = Callable[[BaseLLMException], NativeOutputFallback | None]


def _native_output_retry(
    error: BaseLLMException, fallback: NativeOutputFallbackFactory | None, model: str
) -> NativeOutputFallback | None:
    if fallback is None or not is_bedrock_native_output_rejection(error.status_code, error.message):
        return None
    retry: Final = fallback()
    if retry is not None:
        warn_bedrock_native_output_rejected(model, error.message)
    return retry


def _strict_tools_retry(
    error: BaseLLMException, fallback: NativeOutputFallbackFactory | None, model: str
) -> NativeOutputFallback | None:
    if fallback is None or not is_claude_strict_tools_rejection(error.status_code, error.message):
        return None
    retry: Final = fallback()
    if retry is not None:
        warn_bedrock_strict_tools_dropped(model)
    return retry


def _fallback_for(
    error: BaseLLMException,
    fallback: NativeOutputFallbackFactory | None,
    strict_tools_fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> NativeOutputFallback | None:
    native_output_retry: Final = _native_output_retry(error, fallback, model)
    if native_output_retry is not None:
        return native_output_retry
    return _strict_tools_retry(error, strict_tools_fallback, model)


async def _await_with_native_output_fallback(
    first_attempt: Coroutine[object, object, NativeOutputResponse],
    attempt: NativeOutputAttempt,
    retry_for: _RetryFor,
) -> NativeOutputResponse:
    try:
        return await first_attempt
    except BaseLLMException as error:
        retry: Final = retry_for(error)
        if retry is None:
            raise
        second_attempt: Final = attempt(retry)
    return await second_attempt if isinstance(second_attempt, Coroutine) else second_attempt


def call_with_native_output_fallback(
    attempt: NativeOutputAttempt,
    fallback: NativeOutputFallbackFactory | None,
    model: str,
    strict_tools_fallback: NativeOutputFallbackFactory | None = None,
) -> NativeOutputAttemptResult:
    if fallback is None and strict_tools_fallback is None:
        return attempt(None)
    retry_for: Final = partial(
        _fallback_for, fallback=fallback, strict_tools_fallback=strict_tools_fallback, model=model
    )
    try:
        first_attempt: Final = attempt(None)
    except BaseLLMException as error:
        retry: Final = retry_for(error)
        if retry is None:
            raise
        return attempt(retry)
    if isinstance(first_attempt, Coroutine):
        return _await_with_native_output_fallback(first_attempt, attempt, retry_for)
    return first_attempt
