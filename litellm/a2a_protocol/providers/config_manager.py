"""
A2A Provider Config Manager.

Manages provider-specific configurations for A2A protocol.
"""

from typing import Optional

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig


class A2AProviderConfigManager:
    """
    Manager for A2A provider configurations.

    Similar to ProviderConfigManager in litellm.utils but specifically for A2A providers.
    """

    @staticmethod
    def get_provider_config(
        custom_llm_provider: Optional[str],
        model: Optional[str] = None,
    ) -> Optional[BaseA2AProviderConfig]:
        """
        Get the provider configuration for a given custom_llm_provider.

        Args:
            custom_llm_provider: The provider identifier (e.g., "pydantic_ai_agents")
            model: The model string (used to distinguish sub-providers, e.g. agentcore vs other bedrock)

        Returns:
            Provider configuration instance or None if not found
        """
        if custom_llm_provider is None:
            return None

        if custom_llm_provider == "pydantic_ai_agents":
            from litellm.a2a_protocol.providers.pydantic_ai_agents.config import (
                PydanticAIProviderConfig,
            )

            return PydanticAIProviderConfig()

        if custom_llm_provider == "bedrock" and model and "agentcore" in model:
            from litellm.a2a_protocol.providers.bedrock_agentcore.config import (
                BedrockAgentCoreA2AConfig,
            )

            return BedrockAgentCoreA2AConfig()

        return None
