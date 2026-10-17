from collections.abc import Callable, Coroutine
from typing import Final, TypeAlias

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.common_utils import (
    NativeOutputFallback,
    error_response_text,
    is_bedrock_native_output_rejection,
    warn_bedrock_native_output_rejected,
)
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

NativeOutputResponse: TypeAlias = ModelResponse | CustomStreamWrapper
NativeOutputAttemptResult: TypeAlias = NativeOutputResponse | Coroutine[object, object, NativeOutputResponse]
NativeOutputAttempt: TypeAlias = Callable[[NativeOutputFallback | None], NativeOutputAttemptResult]
NativeOutputError: TypeAlias = BaseLLMException | httpx.HTTPStatusError


def _is_native_output_rejection(error: NativeOutputError) -> bool:
    match error:
        case httpx.HTTPStatusError(response=response):
            return is_bedrock_native_output_rejection(response.status_code, error_response_text(response))
        case BaseLLMException(status_code=status_code, message=message):
            return is_bedrock_native_output_rejection(status_code, message)


async def _await_with_native_output_fallback(
    first_attempt: Coroutine[object, object, NativeOutputResponse],
    attempt: NativeOutputAttempt,
    fallback: Callable[[], NativeOutputFallback],
    model: str,
) -> NativeOutputResponse:
    try:
        return await first_attempt
    except (BaseLLMException, httpx.HTTPStatusError) as error:
        if not _is_native_output_rejection(error):
            raise
    warn_bedrock_native_output_rejected(model)
    second_attempt: Final = attempt(fallback())
    return await second_attempt if isinstance(second_attempt, Coroutine) else second_attempt


def call_with_native_output_fallback(
    attempt: NativeOutputAttempt,
    fallback: Callable[[], NativeOutputFallback] | None,
    model: str,
) -> NativeOutputAttemptResult:
    if fallback is None:
        return attempt(None)
    try:
        first_attempt: Final = attempt(None)
    except (BaseLLMException, httpx.HTTPStatusError) as error:
        if not _is_native_output_rejection(error):
            raise
        warn_bedrock_native_output_rejected(model)
        return attempt(fallback())
    if isinstance(first_attempt, Coroutine):
        return _await_with_native_output_fallback(first_attempt, attempt, fallback, model)
    return first_attempt
