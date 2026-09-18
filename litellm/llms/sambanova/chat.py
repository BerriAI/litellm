"""
Sambanova Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from collections.abc import Coroutine
from typing import Any, Final, Literal, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues


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
        locals_: Final = locals().copy()
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
        from litellm.utils import supports_function_calling

        params: Final = [
            "max_completion_tokens",
            "max_tokens",
            "response_format",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "top_k",
        ]

        if supports_function_calling(model, custom_llm_provider="sambanova"):
            params.append("tools")
            params.append("tool_choice")
            params.append("parallel_tool_calls")

        return params

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

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[Any, Any, list[AllMessageValues]]:
        """
        Transform messages to handle content list conversion.

        SambaNova API doesn't support content as a list - only string content.
        This converts content lists like [{"type": "text", "text": "..."}] to strings.
        """

        async def _async_transform():
            return handle_messages_with_content_list_to_str_conversion(messages)

        if is_async:
            return _async_transform()
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        return messages
