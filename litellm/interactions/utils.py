"""
Utility functions for Interactions API.
"""

from typing import Any, Dict, Optional, cast

from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.types.interactions import InteractionsAPIOptionalRequestParams

# Valid optional parameter keys per OpenAPI spec
INTERACTIONS_API_OPTIONAL_PARAMS = {
    "tools",
    "system_instruction",
    "generation_config",
    "stream",
    "store",
    "background",
    "response_modalities",
    "response_format",
    "response_mime_type",
    "previous_interaction_id",
    "agent_config",
}


def get_provider_interactions_api_config(
    provider: str,
    model: Optional[str] = None,
) -> Optional[BaseInteractionsAPIConfig]:
    """
    Get the interactions API config for the given provider.
    
    Args:
        provider: The LLM provider name
        model: Optional model name
        
    Returns:
        The provider-specific interactions API config, or None if not supported
    """
    from litellm.types.utils import LlmProviders
    
    if provider == LlmProviders.GEMINI.value or provider == "gemini":
        from litellm.llms.gemini.interactions.transformation import (
            GoogleAIStudioInteractionsConfig,
        )
        return GoogleAIStudioInteractionsConfig()
    
    return None


class InteractionsAPIRequestUtils:
    """Helper utils for constructing Interactions API requests."""

    @staticmethod
    def get_requested_interactions_api_optional_params(
        params: Dict[str, Any],
    ) -> InteractionsAPIOptionalRequestParams:
        """
        Filter parameters to only include valid optional params per OpenAPI spec.

        Args:
            params: Dictionary of parameters to filter (typically from locals())

        Returns:
            Dict with only the valid optional parameters
        """
        from litellm.utils import PreProcessNonDefaultParams

        custom_llm_provider = params.pop("custom_llm_provider", None)
        special_params = params.pop("kwargs", {})
        additional_drop_params = params.pop("additional_drop_params", None)

        non_default_params = (
            PreProcessNonDefaultParams.base_pre_process_non_default_params(
                passed_params=params,
                special_params=special_params,
                custom_llm_provider=custom_llm_provider,
                additional_drop_params=additional_drop_params,
                default_param_values={k: None for k in INTERACTIONS_API_OPTIONAL_PARAMS},
                additional_endpoint_specific_params=["input", "model", "agent"],
            )
        )

        return cast(InteractionsAPIOptionalRequestParams, non_default_params)
