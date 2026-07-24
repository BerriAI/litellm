"""
Utility functions for Interactions API.
"""

from typing import Any, cast

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
    "environment",
    "response_modalities",
    "response_format",
    "response_mime_type",
    "previous_interaction_id",
    "agent_config",
}

_VERTEX_AI_INTERACTIONS_ENDPOINT = "/v1beta/interactions"


def get_provider_interactions_api_config(
    provider: str,
    model: str | None = None,
) -> BaseInteractionsAPIConfig | None:
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

    if provider == LlmProviders.VERTEX_AI.value or provider == "vertex_ai":
        if model is None or _supports_vertex_ai_interactions(model=model):
            from litellm.llms.vertex_ai.interactions.transformation import (
                VertexAIInteractionsConfig,
            )

            return VertexAIInteractionsConfig()

    return None


def _supports_vertex_ai_interactions(model: str | None) -> bool:
    if model is None:
        return False

    import litellm

    model_key = model if model.startswith("vertex_ai/") else f"vertex_ai/{model}"
    model_info = litellm.model_cost.get(model_key, {})
    supported_endpoints = model_info.get("supported_endpoints")
    return isinstance(supported_endpoints, list) and _VERTEX_AI_INTERACTIONS_ENDPOINT in supported_endpoints


class InteractionsAPIRequestUtils:
    """Helper utils for constructing Interactions API requests."""

    @staticmethod
    def get_requested_interactions_api_optional_params(
        params: dict[str, Any],
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

        non_default_params = PreProcessNonDefaultParams.base_pre_process_non_default_params(
            passed_params=params,
            special_params=special_params,
            custom_llm_provider=custom_llm_provider,
            additional_drop_params=additional_drop_params,
            default_param_values={k: None for k in INTERACTIONS_API_OPTIONAL_PARAMS},
            additional_endpoint_specific_params=["input", "model", "agent"],
        )

        return cast(InteractionsAPIOptionalRequestParams, non_default_params)
