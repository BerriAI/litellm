"""
CompactifAI chat completion transformation
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ModelResponse
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.base_llm.chat.transformation import BaseLLMException

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CompactifAIChatConfig(OpenAIGPTConfig):
    """
    Configuration class for CompactifAI chat completions.
    Since CompactifAI is OpenAI-compatible, we extend OpenAIGPTConfig.
    """

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API base and key for CompactifAI provider.
        """
        api_base = api_base or "https://api.compactif.ai/v1"
        dynamic_api_key = api_key or get_secret_str("COMPACTIFAI_API_KEY") or ""
        return api_base, dynamic_api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform CompactifAI response to LiteLLM format.
        Since CompactifAI is OpenAI-compatible, we can use the standard OpenAI transformation.
        """
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        response_json = raw_response.json()

        # Handle JSON mode if needed
        if json_mode:
            for choice in response_json["choices"]:
                message = choice.get("message")
                if message and message.get("tool_calls"):
                    # Convert tool calls to content for JSON mode
                    tool_calls = message.get("tool_calls", [])
                    if len(tool_calls) == 1:
                        message["content"] = tool_calls[0]["function"].get("arguments", "")
                        message["tool_calls"] = None

        returned_response = ModelResponse(**response_json)

        # Set model name with provider prefix
        returned_response.model = f"compactifai/{model}"

        return returned_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get the appropriate error class for CompactifAI errors.
        Since CompactifAI is OpenAI-compatible, we use OpenAI error handling.
        """
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )