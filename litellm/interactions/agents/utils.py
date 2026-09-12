"""
Utility functions for the Agents API SDK.
"""

from collections.abc import Mapping
from typing import Final

from litellm.llms.base_llm.agents.transformation import BaseAgentsAPIConfig


def merge_agent_headers(
    *,
    dynamic_headers: Mapping[str, str] | None = None,
    static_headers: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Merge outbound HTTP headers for A2A agent calls.

    Merge rules:
    - Start with ``dynamic_headers`` (values extracted from the incoming client request).
    - Overlay ``static_headers`` (admin-configured per agent).
    - Comparison is case-insensitive (HTTP headers are case-insensitive), so a
      static ``Authorization`` strips any dynamic ``authorization`` before the
      static value is written. The static side's casing is preserved.

    If both contain the same header (case-insensitively), ``static_headers`` wins.
    """
    merged: dict[str, str] = {}

    if dynamic_headers:
        merged.update({str(k): str(v) for k, v in dynamic_headers.items()})

    if static_headers:
        static_lower: Final = {str(k).lower() for k in static_headers}
        merged = {k: v for k, v in merged.items() if k.lower() not in static_lower}
        merged.update({str(k): str(v) for k, v in static_headers.items()})

    return merged or None


def get_provider_agents_api_config(
    custom_llm_provider: str | None,
) -> BaseAgentsAPIConfig | None:
    """
    Return a provider-specific BaseAgentsAPIConfig if the provider has a
    native agent-creation API, or None otherwise.
    """
    from litellm.types.utils import LlmProviders

    if custom_llm_provider == LlmProviders.GEMINI.value:
        from litellm.llms.gemini.agents.transformation import GeminiAgentsConfig

        return GeminiAgentsConfig()
    return None
