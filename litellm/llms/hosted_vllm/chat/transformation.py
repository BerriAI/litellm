"""
Translate from OpenAI's `/v1/chat/completions` to VLLM's `/v1/chat/completions`
"""

import types
from typing import List, Optional, Union

from pydantic import BaseModel

import litellm
from litellm.types.llms.openai import AllMessageValues, ChatCompletionAssistantMessage

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ...OpenAI.chat.gpt_transformation import OpenAIGPTConfig


class HostedVLLMChatConfig(OpenAIGPTConfig):
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        _tools = non_default_params.pop("tools", None)
        if _tools is not None:
            # remove 'additionalProperties' from tools
            _tools = _remove_additional_properties(_tools)
            # remove 'strict' from tools
            _tools = _remove_strict_from_schema(_tools)
        non_default_params["tools"] = _tools
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
