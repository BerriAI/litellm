"""
Semantic Tool Filter Hook

Pre-call hook that filters MCP tools semantically before LLM inference.
Reduces context window size and improves tool selection accuracy.
"""
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    from litellm.proxy._types import UserAPIKeyAuth


class SemanticToolFilterHook(CustomLogger):
    """
    Pre-call hook that filters MCP tools semantically.
    
    This hook:
    1. Extracts the user query from messages
    2. Filters tools based on semantic similarity to the query
    3. Returns only the top-k most relevant tools to the LLM
    """
    
    def __init__(self, semantic_filter: "SemanticMCPToolFilter"):
        """
        Initialize the hook.
        
        Args:
            semantic_filter: SemanticMCPToolFilter instance
        """
        super().__init__()
        self.filter = semantic_filter
        
        verbose_proxy_logger.debug(
            f"Initialized SemanticToolFilterHook with filter: "
            f"enabled={semantic_filter.enabled}, top_k={semantic_filter.top_k}"
        )
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Filter tools before LLM call based on user query.
        
        This hook is called before the LLM request is made. It filters the
        tools list to only include semantically relevant tools.
        
        Args:
            user_api_key_dict: User authentication
            cache: Cache instance
            data: Request data containing messages and tools
            call_type: Type of call (completion, acompletion, etc.)
            
        Returns:
            Modified data dict with filtered tools, or None if no changes
        """
        # Only filter chat completions with tools
        if call_type not in ("completion", "acompletion"):
            verbose_proxy_logger.debug(
                f"Skipping semantic filter for call_type={call_type}"
            )
            return None
        
        # Check if tools are present
        tools = data.get("tools")
        if not tools:
            verbose_proxy_logger.debug("No tools in request, skipping semantic filter")
            return None
        
        # Check if messages are present
        messages = data.get("messages", [])
        if not messages:
            verbose_proxy_logger.debug("No messages in request, skipping semantic filter")
            return None
        
        # Check if filter is enabled
        if not self.filter.enabled:
            verbose_proxy_logger.debug("Semantic filter disabled, skipping")
            return None
        
        try:
            # Extract user query from messages
            user_query = self.filter.extract_user_query(messages)
            if not user_query:
                verbose_proxy_logger.debug("No user query found, skipping semantic filter")
                return None
            
            verbose_proxy_logger.debug(
                f"Applying semantic filter to {len(tools)} tools "
                f"with query: '{user_query[:50]}...'"
            )
            
            # Filter tools semantically
            filtered_tools = await self.filter.filter_tools(
                query=user_query,
                available_tools=tools,
            )
            
            # Only modify data if filtering actually reduced the tool count
            if len(filtered_tools) < len(tools):
                data["tools"] = filtered_tools
                
                verbose_proxy_logger.info(
                    f"Semantic tool filter: {len(tools)} -> {len(filtered_tools)} tools"
                )
                
                return data
            else:
                verbose_proxy_logger.debug(
                    f"Semantic filter did not reduce tool count ({len(tools)}), "
                    "returning original"
                )
                return None
            
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Semantic tool filter hook failed: {e}. Proceeding with all tools."
            )
            return None
