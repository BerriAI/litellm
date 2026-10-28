from collections.abc import Callable, Coroutine
from typing import Final, TypeAlias

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.common_utils import (
    NativeOutputFallback,
    is_bedrock_native_output_rejection,
    warn_bedrock_native_output_rejected,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

NativeOutputResponse: TypeAlias = ModelResponse | CustomStreamWrapper
NativeOutputAttemptResult: TypeAlias = NativeOutputResponse | Coroutine[object, object, NativeOutputResponse]
NativeOutputAttempt: TypeAlias = Callable[[NativeOutputFallback | None], NativeOutputAttemptResult]
NativeOutputFallbackFactory: TypeAlias = Callable[[], NativeOutputFallback | None]


def _fallback_for(
    error: BaseLLMException, fallback: NativeOutputFallbackFactory, model: str
) -> NativeOutputFallback | None:
    if not is_bedrock_native_output_rejection(error.status_code, error.message):
        return None
    retry: Final = fallback()
    if retry is not None:
        warn_bedrock_native_output_rejected(model, error.message)
    return retry


async def _await_with_native_output_fallback(
    first_attempt: Coroutine[object, object, NativeOutputResponse],
    attempt: NativeOutputAttempt,
    fallback: NativeOutputFallbackFactory,
    model: str,
) -> NativeOutputResponse:
    try:
        return await first_attempt
    except BaseLLMException as error:
        retry: Final = _fallback_for(error, fallback, model)
        if retry is None:
            raise
        second_attempt: Final = attempt(retry)
    return await second_attempt if isinstance(second_attempt, Coroutine) else second_attempt


def call_with_native_output_fallback(
    attempt: NativeOutputAttempt,
    fallback: NativeOutputFallbackFactory | None,
    model: str,
) -> NativeOutputAttemptResult:
    if fallback is None:
        return attempt(None)
    try:
        first_attempt: Final = attempt(None)
    except BaseLLMException as error:
        retry: Final = _fallback_for(error, fallback, model)
        if retry is None:
            raise
        return attempt(retry)
    if isinstance(first_attempt, Coroutine):
        return _await_with_native_output_fallback(first_attempt, attempt, fallback, model)
    return first_attempt
