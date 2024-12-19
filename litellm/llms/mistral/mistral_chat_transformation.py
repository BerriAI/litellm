"""
Transformation logic from OpenAI /v1/chat/completion format to Mistral's /chat/completion format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.mistral.ai/api/
"""

from typing import List, Literal, Optional, Tuple, Union

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    strip_none_values_from_message,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class MistralConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.mistral.ai/api/

    The class `MistralConfig` provides configuration for the Mistral's Chat API interface. Below are the parameters:

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2. API Default - 0.7.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling. API Default - 1.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion. API Default - null.

    - `tools` (list or null): A list of available tools for the model. Use this to specify functions for which the model can generate JSON inputs.

    - `tool_choice` (string - 'auto'/'any'/'none' or null): Specifies if/how functions are called. If set to none the model won't call a function and will generate a message instead. If set to auto the model can choose to either generate a message or call a function. If set to any the model is forced to call a function. Default - 'auto'.

    - `stop` (string or array of strings): Stop generation if this token is detected. Or if one of these tokens is detected when providing an array

    - `random_seed` (integer or null): The seed to use for random sampling. If set, different calls will generate deterministic results.

    - `safe_prompt` (boolean): Whether to inject a safety prompt before all conversations. API Default - 'false'.

    - `response_format` (object or null): An object specifying the format that the model must output. Setting to { "type": "json_object" } enables JSON mode, which guarantees the message the model generates is in JSON. When using JSON mode you MUST also instruct the model to produce JSON yourself with a system or a user message.
    """

    temperature: Optional[int] = None
    top_p: Optional[int] = None
    max_tokens: Optional[int] = None
    tools: Optional[list] = None
    tool_choice: Optional[Literal["auto", "any", "none"]] = None
    random_seed: Optional[int] = None
    safe_prompt: Optional[bool] = None
    response_format: Optional[dict] = None
    stop: Optional[Union[str, list]] = None

    def __init__(
        self,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Literal["auto", "any", "none"]] = None,
        random_seed: Optional[int] = None,
        safe_prompt: Optional[bool] = None,
        response_format: Optional[dict] = None,
        stop: Optional[Union[str, list]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "tools",
            "tool_choice",
            "seed",
            "stop",
            "response_format",
        ]

    def _map_tool_choice(self, tool_choice: str) -> str:
        if tool_choice == "auto" or tool_choice == "none":
            return tool_choice
        elif tool_choice == "required":
            return "any"
        else:  # openai 'tool_choice' object param not supported by Mistral API
            return "any"

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "tool_choice" and isinstance(value, str):
                optional_params["tool_choice"] = self._map_tool_choice(
                    tool_choice=value
                )
            if param == "seed":
                optional_params["extra_body"] = {"random_seed": value}
            if param == "response_format":
                optional_params["response_format"] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # mistral is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.mistral.ai
        api_base = (
            api_base
            or get_secret_str("MISTRAL_AZURE_API_BASE")  # for Azure AI Mistral
            or "https://api.mistral.ai/v1"
        )  # type: ignore

        # if api_base does not end with /v1 we add it
        if api_base is not None and not api_base.endswith(
            "/v1"
        ):  # Mistral always needs a /v1 at the end
            api_base = api_base + "/v1"
        dynamic_api_key = (
            api_key
            or get_secret_str("MISTRAL_AZURE_API_KEY")  # for Azure AI Mistral
            or get_secret_str("MISTRAL_API_KEY")
        )
        return api_base, dynamic_api_key

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> List[AllMessageValues]:
        """
        - handles scenario where content is list and not string
        - content list is just text, and no images
        - if image passed in, then just return as is (user-intended)
        - if `name` is passed, then drop it for mistral API: https://github.com/BerriAI/litellm/issues/6696

        Motivation: mistral api doesn't support content as a list
        """
        ## 1. If 'image_url' in content, then return as is
        for m in messages:
            _content_block = m.get("content")
            if _content_block and isinstance(_content_block, list):
                for c in _content_block:
                    if c.get("type") == "image_url":
                        return messages

        ## 2. If content is list, then convert to string
        messages = handle_messages_with_content_list_to_str_conversion(messages)

        ## 3. Handle name in message
        new_messages: List[AllMessageValues] = []
        for m in messages:
            m = MistralConfig._handle_name_in_message(m)
            m = strip_none_values_from_message(m)  # prevents 'extra_forbidden' error
            new_messages.append(m)

        return new_messages

    @classmethod
    def _handle_name_in_message(cls, message: AllMessageValues) -> AllMessageValues:
        """
        Mistral API only supports `name` in tool messages

        If role == tool, then we keep `name`
        Otherwise, we drop `name`
        """
        _name = message.get("name")  # type: ignore
        if _name is not None and message["role"] != "tool":
            message.pop("name", None)  # type: ignore

        return message
