"""
Translates from OpenAI's `/v1/chat/completions` to Together AI's `/v1/chat/completions`.

Docs: https://docs.together.ai/docs/chat-overview
"""

from collections.abc import Callable, Container, Coroutine
from typing import (
    Final,
    Literal,
    cast,  # noqa: TID251  # rebuilding a TypedDict minus keys has no checked spelling
    overload,
)

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_function_calling, supports_response_schema

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

TOOL_CALLING_PARAMS: Final = ("tools", "tool_choice", "function_call")
LITELLM_INTERNAL_ASSISTANT_FIELDS: Final = frozenset({"thinking_blocks", "provider_specific_fields"})
FUNCTION_CALLING_DOCS_URL: Final = "https://docs.together.ai/docs/function-calling"
STRUCTURED_OUTPUTS_DOCS_URL: Final = "https://docs.together.ai/docs/inference/chat/structured-outputs"


def _registry_verdict(model: str, flag: str, check: Callable[[str], bool]) -> bool | None:
    try:
        if check(model):
            return True
    except Exception as e:
        verbose_logger.debug("Error checking together_ai %s for %s: %s", flag, model, e)
    registry_entry: Final = litellm.model_cost.get(f"together_ai/{model}")
    if isinstance(registry_entry, dict) and registry_entry.get(flag) is False:
        return False
    return None


def _function_calling_verdict(model: str) -> bool | None:
    return _registry_verdict(
        model,
        "supports_function_calling",
        lambda checked_model: supports_function_calling(checked_model, custom_llm_provider="together_ai"),
    )


def _response_schema_verdict(model: str) -> bool | None:
    return _registry_verdict(
        model,
        "supports_response_schema",
        lambda checked_model: supports_response_schema(checked_model, custom_llm_provider="together_ai"),
    )


def _tool_params_to_drop(passed_params: Container[str], model: str, drop_params: bool) -> tuple[str, ...]:
    passed_tool_params: Final = tuple(param for param in TOOL_CALLING_PARAMS if param in passed_params)
    if not passed_tool_params:
        return ()
    verdict: Final = _function_calling_verdict(model)
    if verdict is True:
        return ()
    if verdict is None:
        verbose_logger.warning(
            "together_ai model %s has no function calling entry in the model registry; passing %s through for Together to validate. Docs - %s",
            model,
            ", ".join(passed_tool_params),
            FUNCTION_CALLING_DOCS_URL,
        )
        return ()
    if drop_params or litellm.drop_params:
        verbose_logger.warning(
            "together_ai model %s does not support function calling per the model registry; dropping %s. Docs - %s",
            model,
            ", ".join(passed_tool_params),
            FUNCTION_CALLING_DOCS_URL,
        )
        return passed_tool_params
    raise UnsupportedParamsError(
        status_code=500,
        message=f"together_ai does not support parameters: {', '.join(passed_tool_params)}, for model={model}. To drop it from the call, set `litellm.drop_params = True`.",
    )


def _drop_response_format(passed_params: Container[str], model: str, drop_params: bool) -> bool:
    if "response_format" not in passed_params:
        return False
    verdict: Final = _response_schema_verdict(model)
    if verdict is True:
        return False
    if verdict is None:
        verbose_logger.warning(
            "together_ai model %s has no structured outputs entry in the model registry; passing response_format through for Together to validate. Docs - %s",
            model,
            STRUCTURED_OUTPUTS_DOCS_URL,
        )
        return False
    if drop_params or litellm.drop_params:
        verbose_logger.warning(
            "together_ai model %s does not support structured outputs per the model registry; dropping response_format. Docs - %s",
            model,
            STRUCTURED_OUTPUTS_DOCS_URL,
        )
        return True
    raise UnsupportedParamsError(
        status_code=500,
        message=f"together_ai does not support parameters: response_format, for model={model}. To drop it from the call, set `litellm.drop_params = True`.",
    )


def _without_litellm_internal_fields(message: AllMessageValues) -> AllMessageValues:
    if message["role"] != "assistant" or LITELLM_INTERNAL_ASSISTANT_FIELDS.isdisjoint(message):
        return message
    return cast(  # cast-ok: rebuilding the same TypedDict minus internal keys loses the narrowed type
        "AllMessageValues",
        {  # mutable-ok: TypedDict rebuild minus internal keys
            key: value for key, value in message.items() if key not in LITELLM_INTERNAL_ASSISTANT_FIELDS
        },
    )


class TogetherAIChatConfig(OpenAIGPTConfig):
    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        model: str,
        is_async: Literal[True],
    ) -> Coroutine[object, object, list[AllMessageValues]]: ...  # mutable-ok: inherited contract

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...  # mutable-ok: inherited contract

    def _transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        model: str,
        is_async: bool = False,
    ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:  # mutable-ok: inherited contract
        """Together consumes replayed assistant `reasoning_content` (preserved thinking via
        `chat_template_kwargs: {"clear_thinking": false}`), so it must stay in the payload;
        only litellm-internal fields are stripped before sending."""
        stripped: Final = [  # mutable-ok: super() requires a list
            _without_litellm_internal_fields(message) for message in messages
        ]
        if is_async:
            return super()._transform_messages(stripped, model, is_async=True)
        return super()._transform_messages(stripped, model, is_async=False)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        for param in _tool_params_to_drop(mapped_openai_params, model, drop_params):
            mapped_openai_params.pop(param)
        if _drop_response_format(mapped_openai_params, model, drop_params):
            mapped_openai_params.pop("response_format")
        return mapped_openai_params
