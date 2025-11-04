"""Support for OpenAI GPT-5 model family in Responses API."""

from typing import Dict, Optional

import litellm

from .transformation import OpenAIResponsesAPIConfig


class OpenAIGPT5ResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """Configuration for GPT-5 models in Responses API.
    
    Handles OpenAI API restrictions for the GPT-5 series:
    - Only temperature=1 is supported for GPT-5 reasoning models
    - Dropping unsupported temperature values when requested
    """

    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        """Check if the model string refers to a GPT-5 variant.
        
        Args:
            model: The model identifier string
            
        Returns:
            True if the model is a GPT-5 variant, False otherwise
        """
        return "gpt-5" in model

    @classmethod
    def is_model_gpt_5_codex_model(cls, model: str) -> bool:
        """Check if the model is specifically a GPT-5 Codex variant.
        
        Args:
            model: The model identifier string
            
        Returns:
            True if the model is a GPT-5 Codex variant, False otherwise
        """
        return "gpt-5-codex" in model

    def map_openai_params(
        self,
        response_api_optional_params: Dict,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """Map and validate parameters for GPT-5 models.
        
        GPT-5 models only support temperature=1. This method validates the
        temperature parameter and either keeps it (if =1), drops it (if
        drop_params=True), or raises an error.
        
        Args:
            response_api_optional_params: Optional parameters for the API call
            model: The model identifier
            drop_params: Whether to drop unsupported parameters
            
        Returns:
            Validated parameter dictionary
            
        Raises:
            UnsupportedParamsError: If temperature != 1 and drop_params=False
        """
        # Make a copy to avoid modifying the original
        params = dict(response_api_optional_params)
        
        if "temperature" in params:
            temperature_value: Optional[float] = params.pop("temperature")
            if temperature_value is not None:
                if temperature_value == 1:
                    # temperature=1 is the only supported value
                    params["temperature"] = temperature_value
                elif litellm.drop_params or drop_params:
                    # Drop the unsupported temperature value
                    pass
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "gpt-5 models (including gpt-5-codex) don't support temperature={}. "
                            "Only temperature=1 is supported. To drop unsupported params set "
                            "`litellm.drop_params = True`"
                        ).format(temperature_value),
                        status_code=400,
                    )
        
        # Call parent's method for any other transformations
        return super().map_openai_params(
            response_api_optional_params=params,
            model=model,
            drop_params=drop_params,
        )
