"""
OpenAI-like chat completion transformation
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ChatCompletionAssistantMessage
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


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

    @staticmethod
    def _json_mode_convert_tool_response_to_message(
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

        message["content"] = _tool_calls[0]["function"].get("arguments") or ""
        message["tool_calls"] = None

        return message

    @staticmethod
    def _transform_response(
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: bool,
        logging_obj: LiteLLMLoggingObj,
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
                message = (
                    OpenAILikeChatConfig._json_mode_convert_tool_response_to_message(
                        choice.get("message"), json_mode
                    )
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
