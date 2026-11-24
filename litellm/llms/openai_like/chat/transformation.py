"""
OpenAI-like chat completion transformation
"""

from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    iter_items,
    json_contains,
    keyword_arguments,
    length_of,
    require_get,
    set_provider_response_headers_in_hidden_params,
    subscript,
    unpackable_mapping,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_DECODED_JSON: Final = TypeAdapter(object)


class _ModelResponseFields(TypedDict, total=False, extra_items=object):
    pass


class OpenAILikeChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("OPENAI_LIKE_API_BASE")
        dynamic_api_key = api_key or get_secret_str("OPENAI_LIKE_API_KEY") or ""  # vllm does not require an api key
        return api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    @staticmethod
    def _json_mode_convert_tool_response_to_message(message: object, json_mode: bool) -> object:
        """
        if json_mode is true, convert the returned tool call response to a content with json str

        e.g. input:

        {"role": "assistant", "tool_calls": [{"id": "call_5ms4", "type": "function", "function": {"name": "json_tool_call", "arguments": "{\"key\": \"question\", \"value\": \"What is the capital of France?\"}"}}]}

        output:

        {"role": "assistant", "content": "{\"key\": \"question\", \"value\": \"What is the capital of France?\"}"}
        """
        if not json_mode:
            return message

        message_fields: Final = require_get(message)
        _tool_calls: Final = message_fields.get("tool_calls")

        if _tool_calls is None or length_of(_tool_calls) != 1:
            return message

        arguments: Final = require_get(subscript(subscript(_tool_calls, 0), "function")).get("arguments")
        return {**unpackable_mapping(message), "content": arguments or "", "tool_calls": None}

    @staticmethod
    def _json_mode_convert_choice(choice: object, json_mode: bool) -> object:
        message: Final = OpenAILikeChatConfig._json_mode_convert_tool_response_to_message(
            require_get(choice).get("message"), json_mode
        )
        return {**unpackable_mapping(choice), "message": message}

    @staticmethod
    def _json_mode_convert_choices(response_json: object, json_mode: bool) -> object:
        choices: Final = subscript(response_json, "choices")
        converted: Final = [
            OpenAILikeChatConfig._json_mode_convert_choice(choice, json_mode) for choice in iter_items(choices)
        ]
        return {**keyword_arguments(response_json, ModelResponse), "choices": converted or choices}

    @staticmethod
    def _sanitize_usage_obj(response_json: object) -> object:
        """
        Checks for a 'usage' object in the response and replaces any None token values with 0.
        This enforces OpenAI compatibility for providers that might return null.

        This method is future-proof and sanitizes any key ending in '_tokens'.
        """
        if not json_contains(response_json, "usage"):
            return response_json
        usage: Final = require_get(response_json).get("usage")
        usage_fields: Final = as_str_mapping(usage)
        if not isinstance(usage_fields, dict):
            return response_json
        return {
            **keyword_arguments(response_json, ModelResponse),
            "usage": {
                key: 0 if key.endswith("_tokens") and value is None else value for key, value in usage_fields.items()
            },
        }

    @staticmethod
    def _transform_response(
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: bool,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict[str, object],
        api_key: str | None,
        data: dict | str,
        messages: list,
        print_verbose: object,
        encoding: object,
        json_mode: bool | None,
        custom_llm_provider: str | None,
        base_model: str | None,
    ) -> ModelResponse:
        response_json: Final = _DECODED_JSON.validate_python(response.json())
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": data},
        )

        # Sanitize the usage object at the source
        sanitized_json: Final = OpenAILikeChatConfig._sanitize_usage_obj(response_json)
        converted_json: Final = (
            OpenAILikeChatConfig._json_mode_convert_choices(sanitized_json, json_mode) if json_mode else sanitized_json
        )

        returned_response: Final = ModelResponse(
            **_ModelResponseFields(**keyword_arguments(converted_json, ModelResponse))
        )
        set_provider_response_headers_in_hidden_params(returned_response, response.headers)

        if custom_llm_provider is not None:
            returned_response.model = custom_llm_provider + "/" + (returned_response.model or "")

        if base_model is not None:
            returned_response.hidden_params["model"] = base_model
        return returned_response

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict[str, object],
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        return OpenAILikeChatConfig._transform_response(
            model=model,
            response=raw_response,
            model_response=model_response,
            stream=optional_params.get("stream", False),
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
            data=request_data,
            messages=messages,
            print_verbose=None,
            encoding=None,
            json_mode=json_mode,
            custom_llm_provider=None,
            base_model=None,
        )

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        mapped_params: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        if "max_completion_tokens" in non_default_params and replace_max_completion_tokens_with_max_tokens:
            mapped_params["max_tokens"] = non_default_params[
                "max_completion_tokens"
            ]  # most openai-compatible providers support 'max_tokens' not 'max_completion_tokens'
            mapped_params.pop("max_completion_tokens", None)

        return mapped_params
