"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

import types
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionAssistantMessage

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ...OpenAI.chat.gpt_transformation import OpenAIGPTConfig
from ...prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)


class DeepSeekChatConfig(OpenAIGPTConfig):

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        DeepSeek does not support content in list format.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        return super()._transform_messages(messages)

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/beta"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
        return api_base, dynamic_api_key
