"""
OpenAI-like chat completion transformation
"""

import json
import types
from typing import List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel

import litellm
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import ModelResponse

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class OpenAILikeChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("OPENAI_LIKE_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("OPENAI_LIKE_API_KEY") or ""
        )  # vllm does not require an api key
        return api_base, dynamic_api_key

    def _create_json_tool_call_for_response_format(
        self,
        json_schema: Optional[dict] = None,
    ) -> ChatCompletionToolParam:
        """
        Handles creating a tool call for getting responses in JSON format.

        Args:
            json_schema (Optional[dict]): The JSON schema the response should be in

        Returns:
            ChatCompletionToolParam: The tool call to send to an OpenAI-like provider to get responses in JSON format
        """
        _input_schema: dict = {}

        if json_schema is None:
            # Anthropic raises a 400 BadRequest error if properties is passed as None
            # see usage with additionalProperties (Example 5) https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb
            _input_schema["additionalProperties"] = True
            _input_schema["properties"] = {}
        else:
            _input_schema["properties"] = {"values": json_schema}

        _tool = ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name=RESPONSE_FORMAT_TOOL_NAME,
                parameters=_input_schema,
            ),
        )
        return _tool

    def _add_tools_to_optional_params(
        self, optional_params: dict, tools: List[ChatCompletionToolParam]
    ) -> dict:
        if "tools" not in optional_params:
            optional_params["tools"] = tools
        else:
            optional_params["tools"] = [
                *optional_params["tools"],
                *tools,
            ]
        return optional_params

    @staticmethod
    def _convert_tool_response_to_message(
        message: ChatCompletionAssistantMessage, json_mode: bool
    ) -> ChatCompletionAssistantMessage:
        """
        if json_mode is true, convert the returned tool call response to a content with json str

        e.g. input:

        {"role": "assistant", "tool_calls": [{"id": "call_5ms4", "type": "function", "function": {"name": "json_tool_call", "arguments": "{\"key\": \"question\", \"value\": \"What is the capital of France?\"}"}}]}

        output:

        {"role": "assistant", "content": "{\"key\": \"question\", \"value\": \"What is the capital of France?\"}"}
        """
        if not json_mode:
            return message

        _tool_calls = message.get("tool_calls")

        if _tool_calls is None or len(_tool_calls) != 1:
            return message

        if (
            "name" in _tool_calls[0]["function"]
            and _tool_calls[0]["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME
        ):
            message["content"] = _tool_calls[0]["function"].get("arguments") or ""
            message["tool_calls"] = None

        return message

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        extra_body = optional_params.pop("extra_body", {})
        return {
            "model": model,
            "messages": messages,
            **optional_params,
            **extra_body,
        }

    @staticmethod
    def _transform_response(
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: bool,
        logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,  # type: ignore
        optional_params: dict,
        api_key: Optional[str],
        data: Union[dict, str],
        messages: List,
        print_verbose,
        encoding,
        json_mode: bool,
        custom_llm_provider: str,
        base_model: Optional[str],
    ) -> ModelResponse:
        response_json = response.json()
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": data},
        )

        if json_mode:
            for choice in response_json["choices"]:
                message = OpenAILikeChatConfig._convert_tool_response_to_message(
                    choice.get("message"), json_mode
                )
                choice["message"] = message

        returned_response = ModelResponse(**response_json)

        returned_response.model = (
            custom_llm_provider + "/" + (returned_response.model or "")
        )

        if base_model is not None:
            returned_response._hidden_params["model"] = base_model
        return returned_response

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        mapped_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        if (
            "max_completion_tokens" in non_default_params
            and replace_max_completion_tokens_with_max_tokens
        ):
            mapped_params["max_tokens"] = non_default_params[
                "max_completion_tokens"
            ]  # most openai-compatible providers support 'max_tokens' not 'max_completion_tokens'
            mapped_params.pop("max_completion_tokens", None)

        return mapped_params
