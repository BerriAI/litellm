from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ModelResponse
from litellm.types.llms.openai import (
    AllMessageValues,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.base_llm.chat.transformation import BaseLLMException

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class ClarifaiConfig(OpenAIGPTConfig):
    """
    Configuration class for Clarifai chat completions.
    Since Clarifai is OpenAI-compatible, we extend OpenAIGPTConfig.
    """
    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model
        """
        return [
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stream",
            "temperature",
            "top_p",
            "tool_choice",
            "tools",
            "presence_penalty",
            "frequency_penalty",
            "stream_options",
        ]
    
    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
            api_key
            or get_secret_str("CLARIFAI_API_KEY")
        )
        
    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return api_base or "https://api.clarifai.com/v2/ext/openai/v1"
    
    @staticmethod
    def get_base_model(model: Optional[str] = None) -> Optional[str]:
        if model:
            user_id, app_id, model_id = model.split(".")
            return f"https://clarifai.com/{user_id}/{app_id}/models/{model_id}"
        return None

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API base and key for Clarifai provider.
        """
        api_base = api_base or "https://api.clarifai.com/v2/ext/openai/v1"
        dynamic_api_key = api_key or get_secret_str("CLARIFAI_API_KEY") or ""
        return api_base, dynamic_api_key
    
    def transform_request(self, model, messages, optional_params, litellm_params, headers):
        model = self.get_base_model(model) or model
        return super().transform_request(model, messages, optional_params, litellm_params, headers)
    
    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the Clarifai response to a standard ModelResponse.
        Since Clarifai is OpenAI-compatible, we use OpenAI response transformation.
        """
        ## Logging 
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        ## Reponse
        try:
            completion_response = raw_response.json()
        except Exception as e:
            raise OpenAIError(
                status_code=raw_response.status_code,
                message=f"Failed to parse Clarifai response: {str(e)}",
                headers=raw_response.headers,
            ) from e
        
        response = ModelResponse(**completion_response)
        
        if response.model is not None:
            response.model = "clarifai/" + model

        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get the appropriate error class for Clarifai errors.
        Since Clarifai is OpenAI-compatible, we use OpenAI error handling.
        """
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )