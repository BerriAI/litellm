"""
Semantic MCP Tool Filtering using semantic-router

Filters MCP tools semantically for /chat/completions and /responses endpoints.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.utils import get_tool_name_and_description

if TYPE_CHECKING:
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
        self._tool_map: Dict[str, Any] = {}  # MCPTool objects or OpenAI function dicts

    async def build_router_from_mcp_registry(self) -> None:
        """Build semantic router from all MCP tools in the registry (no auth checks)."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        try:
            # Get all servers from registry without auth checks
            registry = global_mcp_server_manager.get_registry()
            if not registry:
                verbose_logger.warning("MCP registry is empty")
                self.tool_router = None
                return

            # Fetch tools from all servers in parallel
            all_tools = []
            for server_id, server in registry.items():
                try:
                    tools = await global_mcp_server_manager.get_tools_for_server(
                        server_id
                    )
                    all_tools.extend(tools)
                except Exception as e:
                    verbose_logger.warning(
                        f"Failed to fetch tools from server {server_id}: {e}"
                    )
                    continue

            if not all_tools:
                verbose_logger.warning("No MCP tools found in registry")
                self.tool_router = None
                return

            verbose_logger.info(
                f"Fetched {len(all_tools)} tools from {len(registry)} MCP servers"
            )
            self._build_router(all_tools)

        except Exception as e:
            verbose_logger.error(f"Failed to build router from MCP registry: {e}")
            self.tool_router = None
            raise

    def _extract_tool_info(self, tool) -> tuple[str, str]:
        """Name + description used as the semantic-router embedding input.

        Description falls back to the name (handled inside
        ``get_tool_name_and_description``) so the embedding text is never
        empty for tools that omit a description.
        """
        return get_tool_name_and_description(tool)

    def _build_router(self, tools: List) -> None:
        """Build semantic router with tools (MCPTool objects or OpenAI function dicts)."""
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
                name, description = self._extract_tool_info(tool)
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

            verbose_logger.info(f"Built semantic router with {len(routes)} tools")

        except Exception as e:
            verbose_logger.error(f"Failed to build semantic router: {e}")
            self.tool_router = None
            raise

    async def filter_tools(
        self,
        query: str,
        available_tools: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
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

        # Router should be built on startup - if not, something went wrong
        if self.tool_router is None:
            verbose_logger.warning(
                "Router not initialized - was build_router_from_mcp_registry() called on startup?"
            )
            return available_tools

        # Run semantic filtering
        try:
            limit = top_k or self.top_k
            matches = self.tool_router(text=query, limit=limit)
            matched_tool_names = self._extract_tool_names_from_matches(matches)

            if not matched_tool_names:
                return available_tools

            return self._get_tools_by_names(matched_tool_names, available_tools)

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

    def _get_tools_by_names(
        self, tool_names: List[str], available_tools: List[Any]
    ) -> List[Any]:
        """Get tools from ``available_tools`` by their names, preserving order.

        The semantic router emits canonical MCP names (e.g. ``<server>-<tool>``
        produced by ``add_server_prefix_to_name``). Clients, however, may
        send tools with their own extra prefix — for example opencode wraps
        every MCP tool as ``litellm_<server>-<tool>`` and Responses-API
        callers sometimes mirror the same pattern. Strict equality then
        drops every tool and the downstream request is shipped with
        ``tools=[]``, which upstream vLLM rejects when ``tool_choice`` is
        set.

        Matching rules (applied in order for every available tool):

        1. Exact canonical match wins.
        2. Otherwise, the longest canonical that appears as a suffix of the
           client name, preceded by ``-`` or ``_``, wins. The separator
           boundary avoids cross-server false matches such as
           ``server_b-a-search`` aliasing into canonical ``a-search``. The
           longest-wins property is emergent: we scan separator positions in
           the client name left-to-right, and the first matching tail is the
           longest canonical suffix by construction. Canonicals produced by
           ``add_server_prefix_to_name`` always carry a ``<server>-<tool>``
           shape, so the suffix relation combined with the separator check
           is specific enough in practice.

        Ordering from ``tool_names`` is preserved; unmatched canonicals are
        skipped rather than fabricated. When no canonical resolves at all we
        log a warning so the empty-``tools`` failure mode stays observable
        instead of silently degrading into an upstream HTTP 400.
        """
        if not tool_names or not available_tools:
            return []

        canonical_set = set(tool_names)

        tool_map: Dict[str, Any] = {}
        for tool in available_tools:
            tool_name, _ = get_tool_name_and_description(tool)
            if not tool_name:
                continue

            if tool_name in canonical_set:
                tool_map.setdefault(tool_name, tool)
                continue

            for i, ch in enumerate(tool_name):
                if ch in ("-", "_"):
                    candidate = tool_name[i + 1 :]
                    if candidate in canonical_set:
                        tool_map.setdefault(candidate, tool)
                        break

        if not tool_map:
            verbose_logger.warning(
                "Semantic filter matched 0 tools for canonicals=%s; check "
                "whether client-side tool names diverge from the registry.",
                tool_names,
            )

        return [tool_map[name] for name in tool_names if name in tool_map]

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
