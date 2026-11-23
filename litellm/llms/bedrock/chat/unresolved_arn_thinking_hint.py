from collections.abc import Callable, Coroutine
from typing import Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.chat.converse_transformation import UNRESOLVED_ARN_BASE_MODEL_TARGET
from litellm.llms.bedrock.chat.native_output_fallback import NativeOutputAttemptResult, NativeOutputResponse
from litellm.llms.bedrock.common_utils import bedrock_capability_model, is_bedrock_application_inference_profile_arn

UNRESOLVED_ARN_THINKING_HINT: Final = (
    "An application inference profile ARN does not name its model, so LiteLLM cannot tell that this model needs "
    "adaptive thinking and sends budget thinking (`thinking.type: enabled`). Set `model_info.base_model` on the "
    "deployment, in config.yaml or in the model's Model Info JSON in the Admin UI, to "
    f"{UNRESOLVED_ARN_BASE_MODEL_TARGET}. LiteLLM then sends adaptive thinking and prices the call at that model."
)


def is_bedrock_enabled_thinking_rejection(error: BaseLLMException) -> bool:
    return error.status_code == 400 and "thinking.type.enabled" in error.message and "is not supported" in error.message


def _with_hint(error: BaseLLMException) -> BaseLLMException:
    return type(error)(
        status_code=error.status_code,
        message=f"{error.message}\n\n{UNRESOLVED_ARN_THINKING_HINT}",
        headers=error.headers,  # pyright: ignore[reportUnknownMemberType]  # BaseLLMException.headers is a bare dict
        request=error.request,
        response=error.response,
        body=error.body,  # pyright: ignore[reportUnknownMemberType]  # BaseLLMException.body is a bare dict
        status_code_is_synthesized=error.status_code_is_synthesized,
    )


async def _await_with_hint(attempt: Coroutine[object, object, NativeOutputResponse]) -> NativeOutputResponse:
    try:
        return await attempt
    except BaseLLMException as error:
        if not is_bedrock_enabled_thinking_rejection(error):
            raise
        raise _with_hint(error) from error


def _with_unresolved_arn_thinking_hint(attempt: Callable[[], NativeOutputAttemptResult]) -> NativeOutputAttemptResult:
    try:
        result: Final = attempt()
    except BaseLLMException as error:
        if not is_bedrock_enabled_thinking_rejection(error):
            raise
        raise _with_hint(error) from error
    return _await_with_hint(result) if isinstance(result, Coroutine) else result


def call_with_unresolved_arn_thinking_hint(
    model: str, base_model: object, attempt: Callable[[], NativeOutputAttemptResult]
) -> NativeOutputAttemptResult:
    if not is_bedrock_application_inference_profile_arn(bedrock_capability_model(model, base_model)):
        return attempt()
    return _with_unresolved_arn_thinking_hint(attempt)
