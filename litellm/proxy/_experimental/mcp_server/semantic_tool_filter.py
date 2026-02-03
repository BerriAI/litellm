"""
Semantic MCP Tool Filtering using semantic-router

Filters MCP tools semantically for /chat/completions and /responses endpoints.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
    from semantic_router.routers import SemanticRouter

    from litellm.router import Router


class SemanticMCPToolFilter:
    """Filters MCP tools using semantic similarity to reduce context window size."""

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
            embedding_model: Model to use for embeddings (e.g., "text-embedding-3-small")
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
        self._tool_map: Dict[str, "MCPTool"] = {}

    async def build_router_from_mcp_registry(self) -> None:
        """Build semantic router from all MCP tools in the registry."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        try:
            # Fetch all MCP tools from the registry (no user auth = all servers)
            tools = await global_mcp_server_manager.list_tools(
                user_api_key_auth=None,
                mcp_auth_header=None,
            )

            if not tools:
                verbose_logger.warning("No MCP tools found in registry")
                self.tool_router = None
                return

            self._build_router(tools)

        except Exception as e:
            verbose_logger.error(f"Failed to build router from MCP registry: {e}")
            self.tool_router = None
            raise

    def _build_router(self, tools: List["MCPTool"]) -> None:
        """Build semantic router with tools."""
        from semantic_router.routers import SemanticRouter
        from semantic_router.routers.base import Route

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )

        if not tools:
            self.tool_router = None
            return

        try:
            # Convert tools to routes
            routes = []
            self._tool_map = {}

            for tool in tools:
                name = tool.name
                description = tool.description or tool.name
                self._tool_map[name] = tool

                routes.append(
                    Route(
                        name=name,
                        description=description,
                        utterances=[description],
                        score_threshold=self.similarity_threshold,
                    )
                )

            self.tool_router = SemanticRouter(
                routes=routes,
                encoder=LiteLLMRouterEncoder(
                    litellm_router_instance=self.router_instance,
                    model_name=self.embedding_model,
                    score_threshold=self.similarity_threshold,
                ),
                auto_sync="local",
            )

            verbose_logger.info(
                f"Built semantic router with {len(routes)} MCP tools from registry"
            )

        except Exception as e:
            verbose_logger.error(f"Failed to build semantic router: {e}")
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
            available_tools: Full list of available MCP tools
            top_k: Override default top_k (optional)

        Returns:
            Filtered and ordered list of tools (up to top_k)
        """
        # Early returns for cases where we can't/shouldn't filter
        if not self.enabled:
            return available_tools
            
        if not available_tools:
            return available_tools
            
        if not query or not query.strip():
            return available_tools

        if self.tool_router is None:
            verbose_logger.warning("Router not initialized, returning all tools")
            return available_tools

        # Run semantic filtering
        try:
            limit = top_k or self.top_k
            matches = self.tool_router(text=query, limit=limit)
            matched_tool_names = self._extract_tool_names_from_matches(matches)
            
            if not matched_tool_names:
                return available_tools
            
            return self._get_tools_by_names(matched_tool_names)

        except Exception as e:
            verbose_logger.error(f"Semantic tool filter failed: {e}", exc_info=True)
            return available_tools

    def _extract_tool_names_from_matches(self, matches) -> List[str]:
        """Extract tool names from semantic router match results."""
        if not matches:
            return []
        
        # Handle single match
        if hasattr(matches, "name") and matches.name:
            return [matches.name]
        
        # Handle list of matches
        if isinstance(matches, list):
            return [m.name for m in matches if hasattr(m, "name") and m.name]
        
        return []

    def _get_tools_by_names(self, tool_names: List[str]) -> List["MCPTool"]:
        """Get tools from tool map by their names, preserving order."""
        return [
            self._tool_map[name] 
            for name in tool_names 
            if name in self._tool_map
        ]

    def extract_user_query(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract user query from messages for /chat/completions or /responses.

        Args:
            messages: List of message dictionaries (from 'messages' or 'input' field)

        Returns:
            Extracted query string
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")

                if isinstance(content, str):
                    return content

                if isinstance(content, list):
                    texts = [
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                        if isinstance(block, (dict, str))
                    ]
                    return " ".join(texts)

        return ""
