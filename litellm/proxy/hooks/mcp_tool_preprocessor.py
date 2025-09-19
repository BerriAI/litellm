from typing import Dict, List, Literal, Optional, Union

from litellm import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class _PROXY_MCPToolPreprocessor(CustomLogger):
    """
    Hook to preprocess MCP tools in requests.
    
    This hook normalizes MCP tools by:
    1. Setting missing server_url to "litellm_proxy" for internal execution
    2. Ensuring proper format for auto-execution based on require_approval settings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Preprocess MCP tools in the request data.
        
        This hook runs before the request is sent to the LLM provider and
        normalizes MCP tools to ensure proper auto-execution behavior.
        """
        if not data or "tools" not in data:
            return data

        tools = data.get("tools", [])
        if not isinstance(tools, list):
            return data

        # Track if we made any modifications
        modified = False

        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "mcp":
                server_url = tool.get("server_url", "")
                
                # Default missing server_url to internal - this enables auto-execution
                if not server_url:
                    tool["server_url"] = "litellm_proxy"
                    modified = True

        # Return modified data if we made changes
        if modified:
            return data
        
        return data