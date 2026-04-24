"""
Semantic MCP Tool Filtering using semantic-router

Filters MCP tools semantically for /chat/completions and /responses endpoints.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.utils import MCP_TOOL_PREFIX_SEPARATOR

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
        """Extract name and description from MCP tool or OpenAI function dict."""
        name: str
        description: str

        if isinstance(tool, dict):
            # OpenAI function format
            name = tool.get("name", "")
            description = tool.get("description", name)
        else:
            # MCPTool object
            name = str(tool.name)
            description = str(tool.description) if tool.description else str(tool.name)

        return name, description

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

    @staticmethod
    def _name_matches_canonical(client_name: str, canonical: str) -> bool:
        """
        Return True if a client-side tool name refers to the given canonical
        MCP tool name.

        MCP clients (e.g. opencode) commonly wrap the proxy's canonical tool
        name with an additive namespace prefix of their own
        (``<client_alias><sep><canonical>``). The prefix can use either a
        dash or an underscore as separator regardless of what
        ``MCP_TOOL_PREFIX_SEPARATOR`` is set to on the proxy, because the
        client doesn't know the proxy's separator.

        The match is anchored: ``canonical`` must form the complete suffix
        of ``client_name`` and be preceded by a separator character, so
        ``rain_gear`` does not match canonical ``ear``.

        Suffix matching is additionally gated on ``canonical`` itself
        containing ``MCP_TOOL_PREFIX_SEPARATOR``. Server-registered MCP
        tools are always emitted as
        ``<server_name><MCP_TOOL_PREFIX_SEPARATOR><tool_name>`` (see
        ``add_server_prefix_to_name``), so a canonical without the
        separator is not a namespaced MCP tool and falling back to
        suffix matching would spuriously collide with unrelated local
        user functions whose names end in the same characters.
        """
        if client_name == canonical:
            return True
        if MCP_TOOL_PREFIX_SEPARATOR not in canonical:
            return False
        if len(client_name) <= len(canonical):
            return False
        if not client_name.endswith(canonical):
            return False
        separator = client_name[-len(canonical) - 1]
        return separator in ("_", "-")

    def _get_tools_by_names(
        self, tool_names: List[str], available_tools: List[Any]
    ) -> List[Any]:
        """
        Get tools from available_tools by their names, preserving the
        semantic router's ordering.

        Matching is tolerant of client-side namespace prefixes: if an
        incoming tool arrived as ``<client_alias>_<canonical>`` while the
        router returned ``<canonical>`` (see
        ``_name_matches_canonical``), that tool is still selected. The
        returned tool object is the original from ``available_tools``, so
        the client-facing name is preserved for tool-call round-trips.
        """
        # Build an index of incoming tools by their client-facing name.
        # Exact matches win over suffix matches when both are present, and
        # each incoming tool is returned at most once even if two canonical
        # names happen to be tail-compatible with the same incoming name.
        available_by_name: Dict[str, Any] = {}
        for tool in available_tools:
            client_name, _ = self._extract_tool_info(tool)
            if client_name and client_name not in available_by_name:
                available_by_name[client_name] = tool

        matched: List[Any] = []
        used_ids: set = set()
        for canonical in tool_names:
            tool = available_by_name.get(canonical)
            if tool is None:
                # Prefer the shortest qualifying name. When several
                # incoming tools suffix-match the same canonical (e.g.
                # "my_search" and "my_tag_search" both end in "search"),
                # the one closest in length to the canonical is the
                # least-wrapped and most likely the intended target.
                best_name: Optional[str] = None
                for client_name in available_by_name:
                    if not self._name_matches_canonical(client_name, canonical):
                        continue
                    if best_name is None or len(client_name) < len(best_name):
                        best_name = client_name
                if best_name is not None:
                    tool = available_by_name[best_name]
            if tool is not None and id(tool) not in used_ids:
                matched.append(tool)
                used_ids.add(id(tool))
        return matched

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
