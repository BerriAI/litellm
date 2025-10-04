# MatterAI Chat Transformation for LiteLLM
import httpx
from typing import Any, List, Optional, Tuple, Union

from litellm.llms.openai import OpenAIGPTConfig
from litellm.types.utils import ModelResponse, BaseLLMException, OpenAIError
from litellm.utils import get_secret_str


class MatterAIChatConfig(OpenAIGPTConfig):
    """
    Configuration class for MatterAI chat completions.
    Since MatterAI is OpenAI-compatible, we extend OpenAIGPTConfig.
    """

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API base and key for MatterAI provider.
        """
        api_base = api_base or "https://api.matterai.so/v1"
        dynamic_api_key = api_key or get_secret_str("MATTERAI_API_KEY") or ""
        return api_base, dynamic_api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform MatterAI response to LiteLLM format.
        Since MatterAI is OpenAI-compatible, we can use the standard OpenAI transformation.
        """
        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        # RESPONSE OBJECT
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
        returned_response.model = f"matterai/{model}"

        return returned_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get the appropriate error class for MatterAI errors.
        Since MatterAI is OpenAI-compatible, we use OpenAI error handling.
        """
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
