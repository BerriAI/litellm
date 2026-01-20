from typing import Any, Dict, List, Literal, Optional

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import TokenCountResponse


class AzureAIAnthropicTokenCounter(BaseTokenCounter):
    """Token counter implementation for Azure AI Anthropic provider using the CountTokens API."""

    def should_use_token_counting_api(
        self,
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        from litellm.types.utils import LlmProviders

        return custom_llm_provider == LlmProviders.AZURE_AI.value

    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        contents: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
    ) -> Optional[TokenCountResponse]:
        """
        Count tokens using Azure AI Anthropic's CountTokens API.

        Args:
            model_to_use: The model identifier
            messages: The messages to count tokens for
            contents: Alternative content format (not used for Anthropic)
            deployment: Deployment configuration containing litellm_params
            request_model: The original request model name

        Returns:
            TokenCountResponse with token count, or None if counting fails
        """
        import os

        from litellm._logging import verbose_logger
        from litellm.llms.anthropic.common_utils import AnthropicError
        from litellm.llms.azure_ai.anthropic.count_tokens.handler import (
            AzureAIAnthropicCountTokensHandler,
        )

        if not messages:
            return None

        deployment = deployment or {}
        litellm_params = deployment.get("litellm_params", {})

        # Get Azure AI API key from deployment config or environment
        api_key = litellm_params.get("api_key")
        if not api_key:
            api_key = os.getenv("AZURE_AI_API_KEY")

        # Get API base from deployment config or environment
        api_base = litellm_params.get("api_base")
        if not api_base:
            api_base = os.getenv("AZURE_AI_API_BASE")

        if not api_key:
            verbose_logger.warning(
                "No Azure AI API key found for token counting"
            )
            return None

        if not api_base:
            verbose_logger.warning(
                "No Azure AI API base found for token counting"
            )
            return None

        try:
            handler = AzureAIAnthropicCountTokensHandler()
            result = await handler.handle_count_tokens_request(
                model=model_to_use,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                litellm_params=litellm_params,
            )

            if result is not None:
                return TokenCountResponse(
                    total_tokens=result.get("input_tokens", 0),
                    request_model=request_model,
                    model_used=model_to_use,
                    tokenizer_type="azure_ai_anthropic_api",
                    original_response=result,
                )
        except AnthropicError as e:
            verbose_logger.warning(
                f"Azure AI Anthropic CountTokens API error: status={e.status_code}, message={e.message}"
            )
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="azure_ai_anthropic_api",
                error=True,
                error_message=e.message,
                status_code=e.status_code,
            )
        except Exception as e:
            verbose_logger.warning(
                f"Error calling Azure AI Anthropic CountTokens API: {e}"
            )
            return TokenCountResponse(
                total_tokens=0,
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type="azure_ai_anthropic_api",
                error=True,
                error_message=str(e),
                status_code=500,
            )

        return None


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
        return (
            api_base or litellm.api_base or get_secret_str("AZURE_AI_API_BASE")
        )

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
            return AzureAIAnthropicTokenCounter()
        return None

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
