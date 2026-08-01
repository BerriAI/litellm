"""
Heroku Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

import os
from collections.abc import Coroutine
from typing import Any, Literal, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues


# Base error class for Heroku
class HerokuError(Exception):
    pass


class HerokuChatConfig(OpenAIGPTConfig):
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
        Heroku does not support content in list format.
        See: https://devcenter.heroku.com/articles/heroku-inference-api-v1-chat-completions#content-object
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(messages=messages, model=model, is_async=True)
        else:
            return super()._transform_messages(messages=messages, model=model, is_async=False)

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or os.getenv("HEROKU_API_BASE")
        api_key = api_key or os.getenv("HEROKU_API_KEY")

        return api_base, api_key

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        api_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)

        if not api_base:
            raise HerokuError(
                "No api base was set. Please provide an api_base, or set the HEROKU_API_BASE environment variable."
            )

        if not api_base.endswith("/v1/chat/completions"):
            api_base = f"{api_base}/v1/chat/completions"

        return api_base
