"""
Semantic Tool Filter Hook

Pre-call hook that filters MCP tools semantically before LLM inference.
Reduces context window size and improves tool selection accuracy.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DEFAULT_MCP_SEMANTIC_FILTER_EMBEDDING_MODEL,
    DEFAULT_MCP_SEMANTIC_FILTER_SIMILARITY_THRESHOLD,
    DEFAULT_MCP_SEMANTIC_FILTER_TOP_K,
)
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.router import Router


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
    
    def _should_expand_mcp_tools(self, tools: List[Any]) -> bool:
        """
        Check if tools contain MCP references with server_url="litellm_proxy".
        
        Only expands MCP tools pointing to litellm proxy, not external MCP servers.
        """
        from litellm.responses.mcp.litellm_proxy_mcp_handler import (
            LiteLLM_Proxy_MCP_Handler,
        )
        
        return LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(tools)
    
    async def _expand_mcp_tools(
        self,
        tools: List[Any],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> List[Dict[str, Any]]:
        """
        Expand MCP references to actual tool definitions.
        
        Reuses LiteLLM_Proxy_MCP_Handler._process_mcp_tools_to_openai_format
        which internally does: parse -> fetch -> filter -> deduplicate -> transform
        """
        from litellm.responses.mcp.litellm_proxy_mcp_handler import (
            LiteLLM_Proxy_MCP_Handler,
        )

        # Parse to separate MCP tools from other tools
        mcp_tools, _ = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(tools)
        
        if not mcp_tools:
            return []
        
        # Use single combined method instead of 3 separate calls
        # This already handles: fetch -> filter by allowed_tools -> deduplicate -> transform
        openai_tools, _ = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_to_openai_format(
            user_api_key_auth=user_api_key_dict,
            mcp_tools_with_litellm_proxy=mcp_tools
        )
        
        # Convert Pydantic models to dicts for compatibility
        openai_tools_as_dicts = []
        for tool in openai_tools:
            if hasattr(tool, "model_dump"):
                tool_dict = tool.model_dump(exclude_none=True)
                verbose_proxy_logger.debug(f"Converted Pydantic tool to dict: {type(tool).__name__} -> dict with keys: {list(tool_dict.keys())}")
                openai_tools_as_dicts.append(tool_dict)
            elif hasattr(tool, "dict"):
                tool_dict = tool.dict(exclude_none=True)
                verbose_proxy_logger.debug(f"Converted Pydantic tool (v1) to dict: {type(tool).__name__} -> dict")
                openai_tools_as_dicts.append(tool_dict)
            elif isinstance(tool, dict):
                verbose_proxy_logger.debug(f"Tool is already a dict with keys: {list(tool.keys())}")
                openai_tools_as_dicts.append(tool)
            else:
                verbose_proxy_logger.warning(f"Tool is unknown type: {type(tool)}, passing as-is")
                openai_tools_as_dicts.append(tool)
        
        verbose_proxy_logger.debug(
            f"Expanded {len(mcp_tools)} MCP reference(s) to {len(openai_tools_as_dicts)} tools (all as dicts)"
        )
        
        return openai_tools_as_dicts
    
    def _get_metadata_variable_name(self, data: dict) -> str:
        if "litellm_metadata" in data:
            return "litellm_metadata"
        return "metadata"
    
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
        # Only filter endpoints that support tools
        if call_type not in ("completion", "acompletion", "aresponses"):
            verbose_proxy_logger.debug(
                f"Skipping semantic filter for call_type={call_type}"
            )
            return None
        
        # Check if tools are present
        tools = data.get("tools")
        if not tools:
            verbose_proxy_logger.debug("No tools in request, skipping semantic filter")
            return None
        
        original_tool_count = len(tools)
        
        # Check for MCP references (server_url="litellm_proxy") and expand them
        if self._should_expand_mcp_tools(tools):
            verbose_proxy_logger.debug(
                "Detected litellm_proxy MCP references, expanding before semantic filtering"
            )
            
            try:
                expanded_tools = await self._expand_mcp_tools(
                    tools, user_api_key_dict
                )
                
                if not expanded_tools:
                    verbose_proxy_logger.warning(
                        "No tools expanded from MCP references"
                    )
                    return None
                
                verbose_proxy_logger.info(
                    f"Expanded {len(tools)} MCP reference(s) to {len(expanded_tools)} tools"
                )
                
                # Update tools for filtering
                tools = expanded_tools
                original_tool_count = len(tools)
                
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Failed to expand MCP references: {e}", exc_info=True
                )
                return None
        
        # Check if messages are present (try both "messages" and "input" for responses API)
        messages = data.get("messages", [])
        if not messages:
            messages = data.get("input", [])
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
                available_tools=tools,  # type: ignore
            )
            
            # Always update tools and emit header (even if count unchanged)
            data["tools"] = filtered_tools
            
            # Store filter stats and tool names for response header
            filter_stats = f"{original_tool_count}->{len(filtered_tools)}"
            tool_names_csv = self._get_tool_names_csv(filtered_tools)
            
            _metadata_variable_name = self._get_metadata_variable_name(data)
            data[_metadata_variable_name]["litellm_semantic_filter_stats"] = filter_stats
            data[_metadata_variable_name]["litellm_semantic_filter_tools"] = tool_names_csv
            
            verbose_proxy_logger.info(
                f"Semantic tool filter: {filter_stats} tools"
            )
            
            return data
            
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Semantic tool filter hook failed: {e}. Proceeding with all tools."
            )
            return None
    
    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        """Add semantic filter stats and tool names to response headers."""
        from litellm.constants import MAX_MCP_SEMANTIC_FILTER_TOOLS_HEADER_LENGTH
        
        _metadata_variable_name = self._get_metadata_variable_name(data)
        metadata = data[_metadata_variable_name]
        
        filter_stats = metadata.get("litellm_semantic_filter_stats")
        if not filter_stats:
            return None
        
        headers = {"x-litellm-semantic-filter": filter_stats}
        
        # Add CSV of filtered tool names (nginx-safe length)
        tool_names_csv = metadata.get("litellm_semantic_filter_tools", "")
        if tool_names_csv:
            if len(tool_names_csv) > MAX_MCP_SEMANTIC_FILTER_TOOLS_HEADER_LENGTH:
                tool_names_csv = tool_names_csv[:MAX_MCP_SEMANTIC_FILTER_TOOLS_HEADER_LENGTH - 3] + "..."
            
            headers["x-litellm-semantic-filter-tools"] = tool_names_csv
        
        return headers
    
    def _get_tool_names_csv(self, tools: List[Any]) -> str:
        """Extract tool names and return as CSV string."""
        if not tools:
            return ""
        
        tool_names = []
        for tool in tools:
            name = tool.get("name", "") if isinstance(tool, dict) else getattr(tool, "name", "")
            if name:
                tool_names.append(name)
        
        return ",".join(tool_names)
    
    @staticmethod
    async def initialize_from_config(
        config: Optional[Dict[str, Any]],
        llm_router: Optional["Router"],
    ) -> Optional["SemanticToolFilterHook"]:
        """
        Initialize semantic tool filter from proxy config.
        
        Args:
            config: Proxy configuration dict (litellm_settings.mcp_semantic_tool_filter)
            llm_router: LiteLLM router instance for embeddings
            
        Returns:
            SemanticToolFilterHook instance if enabled, None otherwise
        """
        from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
            SemanticMCPToolFilter,
        )
        if not config or not config.get("enabled", False):
            verbose_proxy_logger.debug("Semantic tool filter not enabled in config")
            return None
        
        if llm_router is None:
            verbose_proxy_logger.warning(
                "Cannot initialize semantic filter: llm_router is None"
            )
            return None
        
        try:
            
            embedding_model = config.get(
                "embedding_model", DEFAULT_MCP_SEMANTIC_FILTER_EMBEDDING_MODEL
            )
            top_k = config.get("top_k", DEFAULT_MCP_SEMANTIC_FILTER_TOP_K)
            similarity_threshold = config.get(
                "similarity_threshold", DEFAULT_MCP_SEMANTIC_FILTER_SIMILARITY_THRESHOLD
            )
            
            semantic_filter = SemanticMCPToolFilter(
                embedding_model=embedding_model,
                litellm_router_instance=llm_router,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                enabled=True,
            )
            
            # Build router from MCP registry on startup
            await semantic_filter.build_router_from_mcp_registry()
            
            hook = SemanticToolFilterHook(semantic_filter)
            
            verbose_proxy_logger.info(
                f"✅ MCP Semantic Tool Filter enabled: "
                f"embedding_model={embedding_model}, top_k={top_k}, "
                f"similarity_threshold={similarity_threshold}"
            )
            
            return hook
            
        except ImportError as e:
            verbose_proxy_logger.warning(
                f"semantic-router not installed. Install with: "
                f"pip install 'litellm[semantic-router]'. Error: {e}"
            )
            return None
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Failed to initialize MCP semantic tool filter: {e}"
            )
            return None
