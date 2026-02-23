from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.constants import XAI_API_BASE
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.llms.xai import XAIWebSearchTool, XAIXSearchTool
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


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

    def _transform_web_search_tool(self, tool: Dict[str, Any]) -> Union[XAIWebSearchTool, Dict[str, Any]]:
        """
        Transform web_search tool to XAI format.
        
        XAI supports web_search with specific filters:
        - allowed_domains (max 5)
        - excluded_domains (max 5)
        - enable_image_understanding
        
        XAI does NOT support search_context_size (OpenAI-specific).
        """
        xai_tool: Dict[str, Any] = {"type": "web_search"}
        
        # Remove search_context_size if present (not supported by XAI)
        if "search_context_size" in tool:
            verbose_logger.info(
                "XAI does not support 'search_context_size' parameter. Removing it from web_search tool."
            )
        
        # Handle filters (XAI-specific structure)
        filters = {}
        if "allowed_domains" in tool:
            allowed_domains = tool["allowed_domains"]
            filters["allowed_domains"] = allowed_domains
        
        if "excluded_domains" in tool:
            excluded_domains = tool["excluded_domains"]
            filters["excluded_domains"] = excluded_domains
        
        # Add filters if any were specified
        if filters:
            xai_tool["filters"] = filters
        
        # Handle enable_image_understanding (top-level in XAI format)
        if "enable_image_understanding" in tool:
            xai_tool["enable_image_understanding"] = tool["enable_image_understanding"]
        
        return xai_tool
    
    def _transform_x_search_tool(self, tool: Dict[str, Any]) -> Union[XAIXSearchTool, Dict[str, Any]]:
        """
        Transform x_search tool to XAI format.
        
        XAI supports x_search with specific parameters:
        - allowed_x_handles (max 10)
        - excluded_x_handles (max 10)
        - from_date (ISO8601: YYYY-MM-DD)
        - to_date (ISO8601: YYYY-MM-DD)
        - enable_image_understanding
        - enable_video_understanding
        """
        xai_tool: Dict[str, Any] = {"type": "x_search"}
        
        # Handle allowed_x_handles
        if "allowed_x_handles" in tool:
            allowed_handles = tool["allowed_x_handles"]
            xai_tool["allowed_x_handles"] = allowed_handles
        
        # Handle excluded_x_handles
        if "excluded_x_handles" in tool:
            excluded_handles = tool["excluded_x_handles"]
            xai_tool["excluded_x_handles"] = excluded_handles
        
        # Handle date range
        if "from_date" in tool:
            xai_tool["from_date"] = tool["from_date"]
        
        if "to_date" in tool:
            xai_tool["to_date"] = tool["to_date"]
        
        # Handle media understanding flags
        if "enable_image_understanding" in tool:
            xai_tool["enable_image_understanding"] = tool["enable_image_understanding"]
        
        if "enable_video_understanding" in tool:
            xai_tool["enable_video_understanding"] = tool["enable_video_understanding"]
        
        return xai_tool

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
        3. Transforms web_search tools to XAI format (removes search_context_size, adds filters)
        4. Transforms x_search tools to XAI format
        5. Sets store=false when images are detected (recommended by XAI)
        """
        params = dict(response_api_optional_params)
        
        # Drop instructions parameter (not supported by XAI)
        if "instructions" in params:
            verbose_logger.debug(
                "XAI Responses API does not support 'instructions' parameter. Dropping it."
            )
            params.pop("instructions")
        
        if "metadata" in params:
            verbose_logger.debug(
                "XAI Responses API does not support 'metadata' parameter. Dropping it."
            )
            params.pop("metadata")
        
        # Transform tools
        if "tools" in params and params["tools"]:
            tools_list = params["tools"]
            # Ensure tools is a list for iteration
            if not isinstance(tools_list, list):
                tools_list = [tools_list]
            
            transformed_tools: List[Any] = []
            for tool in tools_list:
                if isinstance(tool, dict):
                    tool_type = tool.get("type")
                    
                    if tool_type == "code_interpreter":
                        # XAI supports code_interpreter but doesn't use the container field
                        verbose_logger.debug(
                            "XAI: Transforming code_interpreter tool, removing container field"
                        )
                        transformed_tools.append({"type": "code_interpreter"})
                    
                    elif tool_type == "web_search":
                        # Transform web_search to XAI format
                        verbose_logger.debug(
                            "XAI: Transforming web_search tool to XAI format"
                        )
                        transformed_tools.append(self._transform_web_search_tool(tool))
                    
                    elif tool_type == "x_search":
                        # Transform x_search to XAI format
                        verbose_logger.debug(
                            "XAI: Transforming x_search tool to XAI format"
                        )
                        transformed_tools.append(self._transform_x_search_tool(tool))
                    
                    else:
                        # Keep other tools as-is
                        transformed_tools.append(tool)
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

