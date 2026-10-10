"""
Translate from OpenAI's `/v1/chat/completions` to Friendliai's `/v1/chat/completions`
"""

from collections.abc import Coroutine
from typing import Literal, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import strip_litellm_internal_assistant_fields
from litellm.types.llms.openai import AllMessageValues

from ...openai_like.chat.handler import OpenAILikeChatConfig


class FriendliaiChatConfig(OpenAILikeChatConfig):
    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[object, object, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[False] = False
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
        return super()._transform_messages(
            messages=[strip_litellm_internal_assistant_fields(message) for message in messages],
            model=model,
            is_async=is_async,
        )
