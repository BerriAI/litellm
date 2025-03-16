from typing import Any, Dict, cast, get_type_hints

import litellm
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponsesAPIOptionalRequestParams,
)
from litellm.types.utils import Usage


class ResponsesAPIRequestUtils:
    """Helper utils for constructing ResponseAPI requests"""

    @staticmethod
    def get_optional_params_responses_api(
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
    ) -> Dict:
        """
        Get optional parameters for the responses API.

        Args:
            params: Dictionary of all parameters
            model: The model name
            responses_api_provider_config: The provider configuration for responses API

        Returns:
            A dictionary of supported parameters for the responses API
        """
        # Remove None values and internal parameters

        # Get supported parameters for the model
        supported_params = responses_api_provider_config.get_supported_openai_params(
            model
        )

        # Check for unsupported parameters
        unsupported_params = [
            param
            for param in response_api_optional_params
            if param not in supported_params
        ]

        if unsupported_params:
            raise litellm.UnsupportedParamsError(
                model=model,
                message=f"The following parameters are not supported for model {model}: {', '.join(unsupported_params)}",
            )

        # Map parameters to provider-specific format
        mapped_params = responses_api_provider_config.map_openai_params(
            response_api_optional_params=response_api_optional_params,
            model=model,
            drop_params=litellm.drop_params,
        )

        return mapped_params

    @staticmethod
    def get_requested_response_api_optional_param(
        params: Dict[str, Any]
    ) -> ResponsesAPIOptionalRequestParams:
        """
        Filter parameters to only include those defined in ResponsesAPIOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            ResponsesAPIOptionalRequestParams instance with only the valid parameters
        """
        valid_keys = get_type_hints(ResponsesAPIOptionalRequestParams).keys()
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}
        return cast(ResponsesAPIOptionalRequestParams, filtered_params)


class ResponseAPILoggingUtils:
    @staticmethod
    def _is_response_api_usage(usage: dict) -> bool:
        """returns True if usage is from OpenAI Response API"""
        if "input_tokens" in usage and "output_tokens" in usage:
            return True
        return False

    @staticmethod
    def _transform_response_api_usage_to_chat_usage(usage: dict) -> Usage:
        """Tranforms the ResponseAPIUsage object to a Usage object"""
        response_api_usage: ResponseAPIUsage = ResponseAPIUsage(**usage)
        prompt_tokens: int = response_api_usage.input_tokens or 0
        completion_tokens: int = response_api_usage.output_tokens or 0
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
