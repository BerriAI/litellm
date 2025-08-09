"""
Support for Azure OpenAI O-series models (o1, o3, etc.) in Responses API

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- temperature => drop param (if user opts in to dropping param)
- Other parameters follow base Azure OpenAI Responses API behavior
"""

from typing import TYPE_CHECKING, Any, Dict

from litellm._logging import verbose_logger
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.utils import supports_reasoning

from .transformation import AzureOpenAIResponsesAPIConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AzureOpenAIOSeriesResponsesAPIConfig(AzureOpenAIResponsesAPIConfig):
    """
    Configuration for Azure OpenAI O-series models in Responses API.
    
    O-series models (o1, o3, etc.) do not support the temperature parameter
    in the responses API, so we need to drop it when drop_params is enabled.
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported parameters for Azure OpenAI O-series Responses API.
        
        O-series models don't support temperature parameter in responses API.
        """
        # Get the base Azure supported params
        base_supported_params = super().get_supported_openai_params(model)
        
        # O-series models don't support temperature parameter in responses API
        o_series_unsupported_params = ["temperature"]
        
        # Filter out unsupported parameters for O-series models
        o_series_supported_params = [
            param for param in base_supported_params 
            if param not in o_series_unsupported_params
        ]
        
        return o_series_supported_params

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters for Azure OpenAI O-series Responses API.
        
        Drops temperature parameter if drop_params is True since O-series models
        don't support temperature in the responses API.
        """
        mapped_params = dict(response_api_optional_params)
        
        # If drop_params is enabled, remove temperature parameter for O-series models
        if drop_params and "temperature" in mapped_params:
            verbose_logger.debug(
                f"Dropping unsupported parameter 'temperature' for Azure OpenAI O-series responses API model {model}"
            )
            mapped_params.pop("temperature", None)
            
        return mapped_params

    def is_o_series_model(self, model: str) -> bool:
        """
        Check if the model is an O-series model.
        
        Args:
            model: The model name to check
            
        Returns:
            True if it's an O-series model, False otherwise
        """
        # Check if model name contains o_series or if it's a known O-series model
        if "o_series" in model.lower():
            return True
            
        # Check if the model supports reasoning (which is O-series specific)
        return supports_reasoning(model) 