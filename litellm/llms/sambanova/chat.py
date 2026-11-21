"""
Sambanova Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from collections.abc import Coroutine, Mapping
from typing import Final, Literal, cast, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues

DEFAULT_INTEGRATION_SOURCE: Final = "litellm"
INTEGRATION_SOURCE_HEADER: Final = "X-Integration-Source"


def _is_text_part(part: object) -> bool:
    match part:
        case {"type": "text"}:
            return True
        case _:
            return False


def _has_non_text_part(message: Mapping[str, object]) -> bool:
    content: Final = message.get("content")
    if not isinstance(content, list):
        return False
    parts: Final = cast("list[object]", content)  # cast-ok: isinstance narrows object to an unparameterized list
    return not all(_is_text_part(part) for part in parts)


class SambanovaConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.sambanova.ai/cloud/api-reference/

    Below are the parameters:
    """

    max_tokens: int | None = None
    temperature: int | None = None
    top_p: int | None = None
    top_k: int | None = None
    stop: str | list | None = None
    stream: bool | None = None
    stream_options: dict | None = None
    tool_choice: str | None = None
    response_format: dict | None = None
    tools: list | None = None

    def _flattens_content_lists(self) -> bool:
        return True

    def __init__(
        self,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        stop: str | None = None,
        stream: bool | None = None,
        stream_options: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        tool_choice: str | None = None,
        tools: list | None = None,
    ) -> None:
        locals_: Final[Mapping[str, object]] = dict(locals())
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model

        """
        from litellm.utils import supports_function_calling, supports_reasoning

        params: Final = (
            "max_completion_tokens",
            "max_tokens",
            "response_format",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "top_k",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "n",
            "logit_bias",
            "seed",
        )
        tool_params: Final = (
            ("tools", "tool_choice", "parallel_tool_calls")
            if supports_function_calling(model, custom_llm_provider="sambanova")
            else ()
        )
        reasoning_params: Final = (
            ("reasoning_effort",) if supports_reasoning(model, custom_llm_provider="sambanova") else ()
        )

        return [*params, *tool_params, *reasoning_params]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        map max_completion_tokens param to max_tokens
        """
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Send X-Integration-Source: caller header, then extra_body, then default"""
        extra_headers: Final[dict[str, str]] = dict(optional_params.get("extra_headers") or {})
        extra_body: Final[dict[str, object]] = dict(optional_params.get("extra_body") or {})
        body_source: Final = extra_body.pop("integration_source", None)
        header_source: Final = next(
            (value for key, value in extra_headers.items() if key.lower() == INTEGRATION_SOURCE_HEADER.lower()),
            None,
        )
        integration_source: Final = body_source or DEFAULT_INTEGRATION_SOURCE
        headers_with_source: Final = (
            extra_headers if header_source else {**extra_headers, INTEGRATION_SOURCE_HEADER: integration_source}
        )
        other_params: Final[dict[str, object]] = {k: v for k, v in optional_params.items() if k != "extra_body"}
        updated_params: Final[dict[str, object]] = {
            **other_params,
            "extra_headers": headers_with_source,
            **({"extra_body": extra_body} if extra_body else {}),
        }

        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=updated_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[object, object, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
        transformed: Final = [
            message
            if _has_non_text_part(message)
            else handle_messages_with_content_list_to_str_conversion([message])[0]
            for message in messages
        ]

        async def _async_transform() -> list[AllMessageValues]:
            return transformed

        return _async_transform() if is_async else transformed
