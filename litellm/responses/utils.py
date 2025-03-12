import json
from typing import Any, Dict

import litellm
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIRequestParams,
)


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
    supported_params = responses_api_provider_config.get_supported_openai_params(model)

    # Check for unsupported parameters
    unsupported_params = [
        param for param in response_api_optional_params if param not in supported_params
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
