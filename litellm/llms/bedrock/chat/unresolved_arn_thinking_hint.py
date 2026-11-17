from collections.abc import Callable, Coroutine
from typing import Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.chat.native_output_fallback import NativeOutputAttemptResult, NativeOutputResponse

UNRESOLVED_ARN_THINKING_HINT: Final = (
    "LiteLLM sent budget thinking (`thinking.type: enabled`) because an application inference profile ARN does not "
    "name its model, so LiteLLM cannot tell that this model needs adaptive thinking. Set `base_model` on the "
    "deployment (for example `us.anthropic.claude-opus-5-5`) so LiteLLM sends adaptive thinking."
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


def with_unresolved_arn_thinking_hint(attempt: Callable[[], NativeOutputAttemptResult]) -> NativeOutputAttemptResult:
    try:
        result: Final = attempt()
    except BaseLLMException as error:
        if not is_bedrock_enabled_thinking_rejection(error):
            raise
        raise _with_hint(error) from error
    return _await_with_hint(result) if isinstance(result, Coroutine) else result
