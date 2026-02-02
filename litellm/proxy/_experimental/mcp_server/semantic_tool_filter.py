"""
Semantic MCP Tool Filtering using semantic-router

This module provides semantic filtering for MCP tools to reduce context window size
and improve tool selection accuracy. It leverages the existing semantic-router library
and LiteLLMRouterEncoder to provide efficient tool filtering based on user queries.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
    from semantic_router.routers import SemanticRouter
    from semantic_router.schema import RouteChoice

    from litellm.router import Router


class SemanticMCPToolFilter:
    """
    Filters MCP tools using semantic-router library.
    
    Converts MCP tools to semantic-router Routes and uses SemanticRouter
    to find the most relevant tools for a given user query.
    """
    
    def __init__(
        self,
        embedding_model: str,
        litellm_router_instance: "Router",
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        enabled: bool = True,
    ):
        """
        Initialize the semantic tool filter.
        
        Args:
            embedding_model: Model to use for generating embeddings (e.g., "text-embedding-3-small")
            litellm_router_instance: Router instance for embedding generation
            top_k: Maximum number of tools to return
            similarity_threshold: Minimum similarity score for filtering
            enabled: Whether filtering is enabled
        """
        self.enabled = enabled
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.router_instance = litellm_router_instance
        self.tool_router: Optional["SemanticRouter"] = None
        self._tool_map: Dict[str, "MCPTool"] = {}  # name -> tool
        
        verbose_logger.debug(
            f"Initialized SemanticMCPToolFilter: enabled={enabled}, "
            f"top_k={top_k}, threshold={similarity_threshold}, "
            f"model={embedding_model}"
        )
    
    def _mcp_tools_to_routes(self, tools: List["MCPTool"]) -> List:
        """
        Convert MCP tools to semantic-router Routes.
        
        Args:
            tools: List of MCP tools
            
        Returns:
            List of Route objects
        """
        from semantic_router.routers.base import Route
        
        routes = []
        self._tool_map = {}
        
        for tool in tools:
            self._tool_map[tool.name] = tool
            
            # Use tool description as both description and utterance
            description = tool.description or tool.name
            utterances = [description] if description else []
            
            routes.append(
                Route(
                    name=tool.name,
                    description=description,
                    utterances=utterances,
                    score_threshold=self.similarity_threshold,
                )
            )
        
        verbose_logger.debug(f"Converted {len(tools)} MCP tools to Routes")
        return routes
    
    def rebuild_router(self, tools: List["MCPTool"]) -> None:
        """
        Rebuild semantic router with updated tools.
        
        This should be called whenever the tool list changes (server add/update/remove).
        
        Args:
            tools: Updated list of all available MCP tools
        """
        from semantic_router.routers import SemanticRouter

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )
        
        if not tools:
            self.tool_router = None
            verbose_logger.debug("No tools provided, semantic router set to None")
            return
        
        try:
            routes = self._mcp_tools_to_routes(tools)
            
            self.tool_router = SemanticRouter(
                routes=routes,
                encoder=LiteLLMRouterEncoder(
                    litellm_router_instance=self.router_instance,
                    model_name=self.embedding_model,
                    score_threshold=self.similarity_threshold,
                ),
                auto_sync="local",  # Build index immediately
            )
            
            verbose_logger.info(
                f"Rebuilt semantic router with {len(routes)} tool routes"
            )
            
        except Exception as e:
            verbose_logger.error(f"Failed to rebuild semantic router: {e}")
            self.tool_router = None
            raise
    
    async def filter_tools(
        self,
        query: str,
        available_tools: List["MCPTool"],
        top_k: Optional[int] = None,
    ) -> List["MCPTool"]:
        """
        Filter tools semantically based on query.
        
        Args:
            query: User query to match against tools
            available_tools: Full list of available tools
            top_k: Override default top_k (optional)
            
        Returns:
            Filtered and ordered list of tools (up to top_k)
        """
        if not self.enabled or not available_tools:
            return available_tools
        
        if not query or not query.strip():
            verbose_logger.debug("Empty query, returning all tools")
            return available_tools
        
        top_k = top_k or self.top_k
        
        try:
            # Rebuild router if needed (first time or tools changed)
            if self.tool_router is None:
                verbose_logger.debug("Router not initialized, rebuilding...")
                self.rebuild_router(available_tools)
            
            if self.tool_router is None:
                verbose_logger.warning("Router rebuild failed, returning all tools")
                return available_tools
            
            # Query semantic router
            from semantic_router.schema import RouteChoice
            
            verbose_logger.debug(f"Querying semantic router with: '{query[:50]}...'")
            matches = self.tool_router(text=query)
            
            if not matches:
                verbose_logger.warning(
                    f"No tools matched query. Returning all {len(available_tools)} tools."
                )
                return available_tools
            
            # Extract matched tool names
            matched_names: List[str] = []
            if isinstance(matches, RouteChoice):
                matched_names = [matches.name]
            elif isinstance(matches, list):
                # semantic-router returns list of RouteChoice, take top_k
                matched_names = [m.name for m in matches[:top_k] if hasattr(m, 'name')]
            
            if not matched_names:
                verbose_logger.warning("No matched tool names extracted, returning all tools")
                return available_tools
            
            # Filter available tools by matched names (preserve order from semantic router)
            matched_name_set = set(matched_names)
            filtered = [
                tool for tool in available_tools
                if tool.name in matched_name_set
            ]
            
            # Reorder based on semantic router's ordering
            name_to_tool = {tool.name: tool for tool in filtered}
            ordered_filtered = [
                name_to_tool[name] for name in matched_names
                if name in name_to_tool
            ]
            return ordered_filtered if ordered_filtered else available_tools
            
        except Exception as e:
            verbose_logger.error(
                f"Semantic tool filter failed: {e}. Returning all tools.",
                exc_info=True
            )
            return available_tools
    
    def extract_user_query(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract user query from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Extracted query string
        """
        # Get the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                
                # Handle string content
                if isinstance(content, str):
                    return content
                
                # Handle content blocks (list)
                elif isinstance(content, list):
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            texts.append(block)
                    return " ".join(texts)
        
        return ""
