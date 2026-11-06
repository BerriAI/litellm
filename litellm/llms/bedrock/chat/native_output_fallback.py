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


_RetryFor: TypeAlias = Callable[[BaseLLMException], _SelectedFallback | None]


def _native_output_rejection_retry(
    error: BaseLLMException, fallback: NativeOutputFallbackFactory, model: str
) -> NativeOutputFallback | None:
    retry: Final = fallback()
    if retry is not None:
        warn_bedrock_native_output_rejected(model, error.message)
    return retry


def _buffered_stream_retry(fallback: NativeOutputFallbackFactory, model: str) -> NativeOutputFallback | None:
    retry: Final = fallback()
    if retry is None or retry.optional_params.get("fake_stream") is not True:
        return None
    warn_bedrock_native_output_stream_denied(model)
    return retry


def _native_output_retry(
    error: BaseLLMException, fallback: NativeOutputFallbackFactory | None, model: str
) -> NativeOutputFallback | None:
    if fallback is None:
        return None
    if is_bedrock_native_output_rejection(error.status_code, error.message):
        return _native_output_rejection_retry(error, fallback, model)
    if is_bedrock_stream_permission_denial(error.status_code, error.message):
        return _buffered_stream_retry(fallback, model)
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
    strict_tools_fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> _SelectedFallback | None:
    native_output_retry: Final = _native_output_retry(error, fallback, model)
    if native_output_retry is not None:
        return _SelectedFallback(native_output_retry, native_output_retry=True)
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


async def _await_with_native_output_fallback(
    first_attempt: Coroutine[object, object, NativeOutputResponse],
    attempt: NativeOutputAttempt,
    retry_for: _RetryFor,
    strict_tools_fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> NativeOutputResponse:
    try:
        return await first_attempt
    except BaseLLMException as error:
        retry: Final = retry_for(error)
        if retry is None:
            raise
    try:
        second_attempt: Final = attempt(retry.fallback)
        return await second_attempt if isinstance(second_attempt, Coroutine) else second_attempt
    except BaseLLMException as error:
        composed_retry: Final = _strict_tools_retry_after_native_output_retry(
            error, retry, strict_tools_fallback, model
        )
        if composed_retry is None:
            raise
    third_attempt: Final = attempt(composed_retry)
    return await third_attempt if isinstance(third_attempt, Coroutine) else third_attempt


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
        try:
            return attempt(retry.fallback)
        except BaseLLMException as retry_error:
            composed_retry: Final = _strict_tools_retry_after_native_output_retry(
                retry_error, retry, strict_tools_fallback, model
            )
            if composed_retry is None:
                raise
            return attempt(composed_retry)
    if isinstance(first_attempt, Coroutine):
        return _await_with_native_output_fallback(first_attempt, attempt, retry_for, strict_tools_fallback, model)
    return first_attempt
