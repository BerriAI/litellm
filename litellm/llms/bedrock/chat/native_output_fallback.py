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


def _is_native_output_rejection(error: BaseLLMException) -> bool:
    return is_bedrock_native_output_rejection(error.status_code, error.message)


async def _await_with_native_output_fallback(
    first_attempt: Coroutine[object, object, NativeOutputResponse],
    attempt: NativeOutputAttempt,
    fallback: NativeOutputFallback,
    model: str,
) -> NativeOutputResponse:
    try:
        return await first_attempt
    except BaseLLMException as error:
        if not _is_native_output_rejection(error):
            raise
    warn_bedrock_native_output_rejected(model)
    second_attempt: Final = attempt(fallback)
    return await second_attempt if isinstance(second_attempt, Coroutine) else second_attempt


def call_with_native_output_fallback(
    attempt: NativeOutputAttempt,
    fallback: NativeOutputFallback | None,
    model: str,
) -> NativeOutputAttemptResult:
    if fallback is None:
        return attempt(None)
    try:
        first_attempt: Final = attempt(None)
    except BaseLLMException as error:
        if not _is_native_output_rejection(error):
            raise
        warn_bedrock_native_output_rejected(model)
        return attempt(fallback)
    if isinstance(first_attempt, Coroutine):
        return _await_with_native_output_fallback(first_attempt, attempt, fallback, model)
    return first_attempt
