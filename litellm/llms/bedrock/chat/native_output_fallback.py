from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import Final, TypeAlias

from litellm.llms.anthropic.common_utils import is_claude_strict_tools_rejection
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.common_utils import (
    NativeOutputFallback,
    bedrock_strict_tools_fallback,
    is_bedrock_native_output_rejection,
    is_bedrock_stream_permission_denial,
    warn_bedrock_native_output_rejected,
    warn_bedrock_native_output_stream_denied,
    warn_bedrock_strict_tools_dropped,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

NativeOutputResponse: TypeAlias = ModelResponse | CustomStreamWrapper
NativeOutputAttemptResult: TypeAlias = NativeOutputResponse | Coroutine[object, object, NativeOutputResponse]
NativeOutputAttempt: TypeAlias = Callable[[NativeOutputFallback | None], NativeOutputAttemptResult]
NativeOutputFallbackFactory: TypeAlias = Callable[[], NativeOutputFallback | None]


@dataclass(frozen=True, slots=True)
class _SelectedFallback:
    fallback: NativeOutputFallback
    native_output_retry: bool
    buffered_native_output: bool = False


_RetryFor: TypeAlias = Callable[[BaseLLMException], _SelectedFallback | None]
_NextRetryFor: TypeAlias = Callable[[BaseLLMException, _SelectedFallback], _SelectedFallback | None]


def _native_output_rejection_retry(
    error: BaseLLMException, fallback: NativeOutputFallbackFactory, model: str
) -> NativeOutputFallback | None:
    retry: Final = fallback()
    if retry is not None:
        warn_bedrock_native_output_rejected(model, error.message)
    return retry


def _buffered(retry: NativeOutputFallback) -> NativeOutputFallback:
    return NativeOutputFallback(
        optional_params={**retry.optional_params, "fake_stream": True}, litellm_params=retry.litellm_params
    )


def _buffered_stream_retry(fallback: NativeOutputFallbackFactory, model: str) -> NativeOutputFallback | None:
    retry: Final = fallback()
    if retry is None or retry.optional_params.get("fake_stream") is not True:
        return None
    warn_bedrock_native_output_stream_denied(model)
    return retry


def _native_output_retry(
    error: BaseLLMException,
    fallback: NativeOutputFallbackFactory | None,
    buffered_stream_fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> _SelectedFallback | None:
    if fallback is not None and is_bedrock_native_output_rejection(error.status_code, error.message):
        rejection_retry: Final = _native_output_rejection_retry(error, fallback, model)
        return None if rejection_retry is None else _SelectedFallback(rejection_retry, native_output_retry=True)
    if buffered_stream_fallback is not None and is_bedrock_stream_permission_denial(error.status_code, error.message):
        stream_retry: Final = _buffered_stream_retry(buffered_stream_fallback, model)
        return (
            None
            if stream_retry is None
            else _SelectedFallback(stream_retry, native_output_retry=True, buffered_native_output=True)
        )
    return None


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
    buffered_stream_fallback: NativeOutputFallbackFactory | None,
    strict_tools_fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> _SelectedFallback | None:
    native_output_retry: Final = _native_output_retry(error, fallback, buffered_stream_fallback, model)
    if native_output_retry is not None:
        return native_output_retry
    strict_tools_retry: Final = _strict_tools_retry(error, strict_tools_fallback, model)
    if strict_tools_retry is None:
        return None
    return _SelectedFallback(strict_tools_retry, native_output_retry=False)


def _strict_tools_retry_after_native_output_retry(
    error: BaseLLMException,
    retry: _SelectedFallback,
    strict_tools_fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> NativeOutputFallback | None:
    if (
        not retry.native_output_retry
        or strict_tools_fallback is None
        or not is_claude_strict_tools_rejection(error.status_code, error.message)
    ):
        return None
    composed_retry: Final = bedrock_strict_tools_fallback(
        model, retry.fallback.optional_params, retry.fallback.litellm_params
    )
    if composed_retry is not None:
        warn_bedrock_strict_tools_dropped(model)
    return composed_retry


def _next_retry(
    error: BaseLLMException,
    retry: _SelectedFallback,
    fallback: NativeOutputFallbackFactory | None,
    strict_tools_fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> _SelectedFallback | None:
    if (
        retry.buffered_native_output
        and fallback is not None
        and is_bedrock_native_output_rejection(error.status_code, error.message)
    ):
        rejection_retry: Final = _native_output_rejection_retry(error, fallback, model)
        return (
            None if rejection_retry is None else _SelectedFallback(_buffered(rejection_retry), native_output_retry=True)
        )
    composed_retry: Final = _strict_tools_retry_after_native_output_retry(error, retry, strict_tools_fallback, model)
    return None if composed_retry is None else _SelectedFallback(composed_retry, native_output_retry=False)


def _attempt_or_next_retry(
    attempt: NativeOutputAttempt, retry: _SelectedFallback, next_retry_for: _NextRetryFor
) -> NativeOutputAttemptResult | _SelectedFallback:
    try:
        return attempt(retry.fallback)
    except BaseLLMException as error:
        next_retry: Final = next_retry_for(error, retry)
        if next_retry is None:
            raise
        return next_retry


def _attempt_retries(
    attempt: NativeOutputAttempt, retry: _SelectedFallback, next_retry_for: _NextRetryFor
) -> NativeOutputAttemptResult:
    after_first_retry: Final = _attempt_or_next_retry(attempt, retry, next_retry_for)
    if not isinstance(after_first_retry, _SelectedFallback):
        return after_first_retry
    after_second_retry: Final = _attempt_or_next_retry(attempt, after_first_retry, next_retry_for)
    if not isinstance(after_second_retry, _SelectedFallback):
        return after_second_retry
    return attempt(after_second_retry.fallback)


async def _await_or_next_retry(
    attempt: NativeOutputAttempt, retry: _SelectedFallback, next_retry_for: _NextRetryFor
) -> NativeOutputResponse | _SelectedFallback:
    try:
        result: Final = attempt(retry.fallback)
        return await result if isinstance(result, Coroutine) else result
    except BaseLLMException as error:
        next_retry: Final = next_retry_for(error, retry)
        if next_retry is None:
            raise
        return next_retry


async def _await_retries(
    attempt: NativeOutputAttempt, retry: _SelectedFallback, next_retry_for: _NextRetryFor
) -> NativeOutputResponse:
    after_first_retry: Final = await _await_or_next_retry(attempt, retry, next_retry_for)
    if not isinstance(after_first_retry, _SelectedFallback):
        return after_first_retry
    after_second_retry: Final = await _await_or_next_retry(attempt, after_first_retry, next_retry_for)
    if not isinstance(after_second_retry, _SelectedFallback):
        return after_second_retry
    last_attempt: Final = attempt(after_second_retry.fallback)
    return await last_attempt if isinstance(last_attempt, Coroutine) else last_attempt


async def _await_with_native_output_fallback(
    first_attempt: Coroutine[object, object, NativeOutputResponse],
    attempt: NativeOutputAttempt,
    retry_for: _RetryFor,
    next_retry_for: _NextRetryFor,
) -> NativeOutputResponse:
    try:
        return await first_attempt
    except BaseLLMException as error:
        retry: Final = retry_for(error)
        if retry is None:
            raise
    return await _await_retries(attempt, retry, next_retry_for)


def call_with_native_output_fallback(
    attempt: NativeOutputAttempt,
    fallback: NativeOutputFallbackFactory | None,
    model: str,
    strict_tools_fallback: NativeOutputFallbackFactory | None = None,
    buffered_stream_fallback: NativeOutputFallbackFactory | None = None,
) -> NativeOutputAttemptResult:
    if fallback is None and strict_tools_fallback is None and buffered_stream_fallback is None:
        return attempt(None)
    retry_for: Final = partial(
        _fallback_for,
        fallback=fallback,
        buffered_stream_fallback=buffered_stream_fallback,
        strict_tools_fallback=strict_tools_fallback,
        model=model,
    )
    next_retry_for: Final = partial(
        _next_retry, fallback=fallback, strict_tools_fallback=strict_tools_fallback, model=model
    )
    try:
        first_attempt: Final = attempt(None)
    except BaseLLMException as error:
        retry: Final = retry_for(error)
        if retry is None:
            raise
        return _attempt_retries(attempt, retry, next_retry_for)
    if isinstance(first_attempt, Coroutine):
        return _await_with_native_output_fallback(first_attempt, attempt, retry_for, next_retry_for)
    return first_attempt
