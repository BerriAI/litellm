"""
Support for OVHcloud AI Endpoints `/v1/responses` endpoint.

Our unified API follows the OpenAI standard.
More information on our website: https://oai.endpoints.kepler.ai.cloud.ovh.net/doc/gpt-oss-20b/openapi.json
"""
from typing import Optional
import litellm
from litellm._logging import verbose_logger
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import get_model_info

class OVHCloudResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for OVHCloud AI Endpoints Responses API.
    
    Inherits from OpenAIResponsesAPIConfig since OVHCloud's Responses API follows
    the OpenAI specification.
    
    Reference: https://oai.endpoints.kepler.ai.cloud.ovh.net/doc/gpt-oss-20b/openapi.json
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OVHCLOUD

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI params, filtering tool-related params for models
        that don't support function calling.
        
        Details about function calling support can be found here:
        https://help.ovhcloud.com/csm/en-gb-public-cloud-ai-endpoints-function-calling?id=kb_article_view&sysparm_article=KB0071907
        """
        supported_params = super().get_supported_openai_params(model)
        
        supports_function_calling: Optional[bool] = None
        try:
            model_info = get_model_info(model, custom_llm_provider="ovhcloud")
            supports_function_calling = model_info.get(
                "supports_function_calling", False
            )
        except Exception as e:
            verbose_logger.debug(f"Error getting supported OpenAI params: {e}")
            pass
        
        if supports_function_calling is not True:
            verbose_logger.debug(
                "You can see our models supporting function_calling in our catalog: https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/ "
            )
            # Remove tool-related params for models that don't support function calling
            for param in ("tools", "tool_choice"):
                if param in supported_params:
                    supported_params.remove(param)
        
        return supported_params

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate environment and set up headers for OVHCloud API.
        
        Uses OVHCLOUD_API_KEY from environment or litellm_params.
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or litellm.ovhcloud_key
            or get_secret_str("OVHCLOUD_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "OVHcloud AI Endpoints API key is required. Set OVHCLOUD_API_KEY environment variable or pass api_key parameter."
            )
        
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for OVHcloud AI Endpoints Responses API endpoint.
        
        Returns:
            str: The full URL for the OVHcloud AI Endpoints /v1/responses endpoint
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OVHCLOUD_API_BASE")
            or "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
        )
        
        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Avoid double-appending /responses
        if not api_base.endswith("/responses"):
            return f"{api_base}/responses"
        return api_base
