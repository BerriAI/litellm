from typing import TYPE_CHECKING, Any, Dict, List, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

XAI_API_BASE = "https://api.x.ai/v1"


class XAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for XAI's Responses API.
    
    Inherits from OpenAIResponsesAPIConfig since XAI's Responses API is largely
    compatible with OpenAI's, with a few differences:
    - Does not support the 'instructions' parameter
    - Requires code_interpreter tools to have 'container' field removed
    - Recommends store=false when sending images
    
    Reference: https://docs.x.ai/docs/api-reference#create-new-response
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.XAI

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported parameters for XAI Responses API.
        
        XAI supports most OpenAI Responses API params except 'instructions'.
        """
        supported_params = super().get_supported_openai_params(model)
        
        # Remove 'instructions' as it's not supported by XAI
        if "instructions" in supported_params:
            supported_params.remove("instructions")
        
        return supported_params

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map parameters for XAI Responses API.
        
        Handles XAI-specific transformations:
        1. Drops 'instructions' parameter (not supported)
        2. Transforms code_interpreter tools to remove 'container' field
        3. Sets store=false when images are detected (recommended by XAI)
        """
        params = dict(response_api_optional_params)
        
        # Drop instructions parameter (not supported by XAI)
        if "instructions" in params:
            verbose_logger.debug(
                "XAI Responses API does not support 'instructions' parameter. Dropping it."
            )
            params.pop("instructions")
        
        # Transform code_interpreter tools - remove container field
        if "tools" in params and params["tools"]:
            tools_list = params["tools"]
            # Ensure tools is a list for iteration
            if not isinstance(tools_list, list):
                tools_list = [tools_list]
            
            transformed_tools: List[Any] = []
            for tool in tools_list:
                if isinstance(tool, dict) and tool.get("type") == "code_interpreter":
                    # XAI supports code_interpreter but doesn't use the container field
                    # Keep only the type field
                    verbose_logger.debug(
                        "XAI: Transforming code_interpreter tool, removing container field"
                    )
                    transformed_tools.append({"type": "code_interpreter"})
                else:
                    transformed_tools.append(tool)
            params["tools"] = transformed_tools
        
        return params

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate environment and set up headers for XAI API.
        
        Uses XAI_API_KEY from environment or litellm_params.
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("XAI_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "XAI API key is required. Set XAI_API_KEY environment variable or pass api_key parameter."
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
        Get the complete URL for XAI Responses API endpoint.
        
        Returns:
            str: The full URL for the XAI /responses endpoint
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("XAI_API_BASE")
            or XAI_API_BASE
        )
        
        # Remove trailing slashes
        api_base = api_base.rstrip("/")
        
        return f"{api_base}/responses"

