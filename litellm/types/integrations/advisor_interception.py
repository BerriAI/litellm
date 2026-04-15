"""
Type definitions for Advisor Interception integration.
"""

from typing import List, Optional, TypedDict


class AdvisorInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for AdvisorInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          advisor_interception_params:
            default_advisor_model: "advisor-model"
            enabled_providers: ["openai", "vertex_ai"]
    """

    default_advisor_model: Optional[str]
    """
    The model_name (from model_list) or litellm model string to use as the
    advisor model. When running behind the proxy, this should match a
    model_name in model_list so the router resolves the correct deployment
    and credentials.
    """

    enabled_providers: List[str]
    """List of LLM provider names to enable interception for (e.g., ['openai', 'vertex_ai'])"""
