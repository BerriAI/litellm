"""
Translates from OpenAI's `/v1/chat/completions` to Databricks' `/chat/completions`
"""

import types
from typing import List, Optional, Union

from pydantic import BaseModel

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderField

from ...OpenAI.chat.gpt_transformation import OpenAIGPTConfig
from ...prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    strip_name_from_messages,
)


class DatabricksConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    n: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        stop: Optional[Union[List[str], str]] = None,
        n: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="api_key",
                field_type="string",
                field_description="Your Databricks API Key.",
                field_value="dapi...",
            ),
            ProviderField(
                field_name="api_base",
                field_type="string",
                field_description="Your Databricks API Base.",
                field_value="https://adb-..",
            ),
        ]

    def get_supported_openai_params(self, model: Optional[str] = None) -> list:
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "response_format",
        ]

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Databricks doesn't support 'response_format' while streaming
        """
        if optional_params.get("response_format") is not None:
            return True

        return False

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ):
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "n":
                optional_params["n"] = value
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "response_format":
                optional_params["response_format"] = value
        return optional_params

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        Databricks does not support:
        - content in list format.
        - 'name' in user message.
        """
        new_messages = []
        for idx, message in enumerate(messages):
            if isinstance(message, BaseModel):
                _message = message.model_dump(exclude_none=True)
            else:
                _message = message
            new_messages.append(_message)
        new_messages = handle_messages_with_content_list_to_str_conversion(new_messages)
        new_messages = strip_name_from_messages(new_messages)
        return super()._transform_messages(new_messages)
