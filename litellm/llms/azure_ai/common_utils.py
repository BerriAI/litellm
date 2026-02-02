from typing import List, Literal, Optional

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues


class AzureFoundryModelInfo(BaseLLMModelInfo):
    """Model info for Azure AI / Azure Foundry models."""

    def __init__(self, model: Optional[str] = None):
        self._model = model

    @staticmethod
    def get_azure_ai_route(model: str) -> Literal["agents", "default"]:
        """
        Get the Azure AI route for the given model.

        Similar to BedrockModelInfo.get_bedrock_route().
        """
        if "agents/" in model:
            return "agents"
        return "default"

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return api_base or litellm.api_base or get_secret_str("AZURE_AI_API_BASE")

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("AZURE_AI_API_KEY")
        )

    @property
    def api_version(self, api_version: Optional[str] = None) -> Optional[str]:
        api_version = (
            api_version or litellm.api_version or get_secret_str("AZURE_API_VERSION")
        )
        return api_version

    def get_token_counter(self) -> Optional[BaseTokenCounter]:
        """
        Factory method to create a token counter for Azure AI.

        Returns:
            AzureAIAnthropicTokenCounter for Claude models, None otherwise.
        """
        # Only return token counter for Claude models
        if self._model and "claude" in self._model.lower():
            from litellm.llms.azure_ai.anthropic.count_tokens.token_counter import (
                AzureAIAnthropicTokenCounter,
            )

            return AzureAIAnthropicTokenCounter()
        return None

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Returns a list of models supported by Azure AI.
        
        Azure AI doesn't have a standard model listing endpoint,
        so this returns an empty list.
        """
        return []

    #########################################################
    # Not implemented methods
    #########################################################

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        raise NotImplementedError("Azure Foundry does not support base model")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Azure Foundry sends api key in query params"""
        raise NotImplementedError(
            "Azure Foundry does not support environment validation"
        )
