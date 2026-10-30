"""
Support for Azure OpenAI O-series models (o1, o3, etc.) in Responses API

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- temperature => drop param (if user opts in to dropping param)
- Other parameters follow base Azure OpenAI Responses API behavior
"""

from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, Final

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
        base_supported_params: Final = super().get_supported_openai_params(model)

        # O-series models don't support temperature parameter in responses API
        o_series_unsupported_params: Final = ["temperature"]

        # Filter out unsupported parameters for O-series models
        o_series_supported_params: Final = [
            param for param in base_supported_params if param not in o_series_unsupported_params
        ]

        return o_series_supported_params

    def map_openai_params_for_deployment(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
        *,
        declared_reasoning_efforts: Collection[str] | None,
    ) -> Mapping[str, object]:
        """
        Map OpenAI parameters for Azure OpenAI O-series Responses API.

        Drops temperature and malformed values if drop_params is True since O-series models
        don't support temperature in the responses API.
        """
        if not drop_params:
            return dict(response_api_optional_params)
        kept: Final = self.params_dropped_for_deployment(
            response_api_optional_params, model, declared_reasoning_efforts=declared_reasoning_efforts
        )
        if "temperature" in kept:
            verbose_logger.debug(
                "Dropping unsupported parameter 'temperature' for Azure OpenAI O-series responses API model %s", model
            )
        return {name: value for name, value in kept.items() if name != "temperature"}

    def _reasoning_refusal(
        self, reasoning: object, model: str, declared_reasoning_efforts: Collection[str] | None
    ) -> None:
        return None

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
