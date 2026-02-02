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
    def get_azure_ai_route(model: str) -> Literal["agents", "model_router", "default"]:
        """
        Get the Azure AI route for the given model.

        Similar to BedrockModelInfo.get_bedrock_route().
        
        Supported routes:
        - agents: azure_ai/agents/<agent_id>
        - model_router: azure_ai/model_router/<actual-model-name> or models with "model-router"/"model_router" in name
        - default: standard models
        """
        if "agents/" in model:
            return "agents"
        # Detect model router by prefix (model_router/<name>) or by name containing "model-router"/"model_router"
        model_lower = model.lower()
        if (
            "model_router/" in model_lower 
            or "model-router/" in model_lower
            or "model-router" in model_lower
            or "model_router" in model_lower
        ):
            return "model_router"
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
    def strip_model_router_prefix(model: str) -> str:
        """
        Strip the model_router prefix from model name.
        
        Examples:
        - "model_router/gpt-4o" -> "gpt-4o"
        - "model-router/gpt-4o" -> "gpt-4o"
        - "gpt-4o" -> "gpt-4o"
        
        Args:
            model: Model name potentially with model_router prefix
            
        Returns:
            Model name without the prefix
        """
        if "model_router/" in model:
            return model.split("model_router/", 1)[1]
        if "model-router/" in model:
            return model.split("model-router/", 1)[1]
        return model
    
    @staticmethod
    def get_base_model(model: str) -> str:
        """
        Get the base model name, stripping any Azure AI routing prefixes.
        
        Args:
            model: Model name potentially with routing prefixes
            
        Returns:
            Base model name
        """
        # Strip model_router prefix if present
        model = AzureFoundryModelInfo.strip_model_router_prefix(model)
        return model

    @staticmethod
    def get_azure_ai_config_for_model(model: str):
        """
        Get the appropriate Azure AI config class for the given model.
        
        Routes to specialized configs based on model type:
        - Model Router: AzureModelRouterConfig
        - Claude models: AzureAnthropicConfig  
        - Default: AzureAIStudioConfig
        
        Args:
            model: The model name
            
        Returns:
            The appropriate config instance
        """
        azure_ai_route = AzureFoundryModelInfo.get_azure_ai_route(model)
        
        if azure_ai_route == "model_router":
            from litellm.llms.azure_ai.azure_model_router.transformation import (
                AzureModelRouterConfig,
            )
            return AzureModelRouterConfig()
        elif "claude" in model.lower():
            from litellm.llms.azure_ai.anthropic.transformation import (
                AzureAnthropicConfig,
            )
            return AzureAnthropicConfig()
        else:
            from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig
            return AzureAIStudioConfig()

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
