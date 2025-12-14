"""
Venice AI chat completion transformation
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai_like.chat.transformation import OpenAILikeChatConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VeniceAIChatConfig(OpenAILikeChatConfig):
    """
    Venice AI provider configuration.

    Venice AI is OpenAI-compatible but requires custom parameters to be nested
    in a `venice_parameters` object.
    """

    # Venice AI specific parameters that should be nested in venice_parameters
    VENICE_PARAMS = {
        "character_slug",
        "strip_thinking_response",
        "disable_thinking",
        "enable_web_search",
        "enable_web_scraping",
        "enable_web_citations",
        "include_search_results_in_stream",
        "return_search_results_as_documents",
        "include_venice_system_prompt",
    }

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get Venice AI API base URL and API key"""
        api_base = api_base or get_secret_str("VENICE_AI_API_BASE") or "https://api.venice.ai/api/v1"  # type: ignore
        dynamic_api_key = api_key or get_secret_str("VENICE_AI_API_KEY") or ""  # type: ignore
        return api_base, dynamic_api_key

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to include venice_parameters object.

        Collects Venice-specific parameters from optional_params and nests them
        in a venice_parameters object. Supports both:
        1. Direct parameter passing (e.g., enable_web_search="auto")
        2. Nested dict format (venice_parameters={...})
        """
        # Extract venice_parameters if provided as a dict
        venice_parameters = optional_params.pop("venice_parameters", {})

        # Check extra_body for Venice parameters (since Venice AI is OpenAI-compatible,
        # provider-specific params may be nested in extra_body)
        extra_body = optional_params.pop("extra_body", {})
        if isinstance(extra_body, dict):
            # Extract venice_parameters from extra_body if present
            if "venice_parameters" in extra_body:
                venice_parameters = {
                    **venice_parameters,
                    **extra_body.pop("venice_parameters"),
                }

            # Collect Venice-specific parameters from extra_body
            venice_params_from_extra = {}
            for param in list(extra_body.keys()):
                if param in self.VENICE_PARAMS:
                    venice_params_from_extra[param] = extra_body.pop(param)

            # Merge Venice params from extra_body
            if venice_params_from_extra:
                venice_parameters = {**venice_parameters, **venice_params_from_extra}

            # Put remaining extra_body back if it has other params
            if extra_body:
                optional_params["extra_body"] = extra_body

        # Collect Venice-specific parameters from optional_params
        venice_params_to_extract = {}
        for param in list(optional_params.keys()):
            if param in self.VENICE_PARAMS:
                venice_params_to_extract[param] = optional_params.pop(param)

        # Merge extracted params with any existing venice_parameters
        if venice_params_to_extract:
            venice_parameters = {**venice_parameters, **venice_params_to_extract}

        # Call parent transform_request to get the base OpenAI-compatible request
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Add venice_parameters to extra_body (OpenAI SDK supports extra_body parameter)
        # This ensures venice_parameters are properly passed to the API
        if venice_parameters:
            # Ensure extra_body exists in the data
            if "extra_body" not in data:
                data["extra_body"] = {}
            data["extra_body"]["venice_parameters"] = venice_parameters

        return data

    def transform_response(
        self,
        model: str,
        raw_response: Any,
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
        Transform Venice AI response to LiteLLM format.

        Since Venice AI is OpenAI-compatible, we can use the parent's
        transform_response method.
        """
        return super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )
