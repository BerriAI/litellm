"""
Semantic MCP Tool Filtering using semantic-router

Filters MCP tools semantically for /chat/completions and /responses endpoints.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.exceptions import ContextWindowExceededError
from litellm.litellm_core_utils.exception_mapping_utils import ExceptionCheckers
from litellm.proxy._experimental.mcp_server.utils import MCP_TOOL_PREFIX_SEPARATOR

if TYPE_CHECKING:
    from semantic_router.routers import SemanticRouter

    from litellm.router import Router


class SemanticToolFilterContextWindowError(Exception):
    """Raised when the embedding model exceeds its context window, so semantic filtering cannot run."""

    def __init__(self, embedding_model: str, stage: str, original_error: str):
        self.embedding_model = embedding_model
        self.stage = stage
        self.original_error = original_error
        super().__init__(
            f"MCP semantic tool filtering could not run: embedding model '{embedding_model}' "
            f"exceeded its context window while embedding {stage}. "
            f"The request was blocked instead of silently passing all tools through. "
            f"Switch to an embedding model with a larger context window, or disable "
            f"semantic tool filtering."
        )


def _is_context_window_error(error: Optional[BaseException], max_depth: int = 5) -> bool:
    """Detect a context-window overflow anywhere in an exception's cause chain."""
    current = error
    for _ in range(max_depth):
        if current is None:
            return False
        if isinstance(current, ContextWindowExceededError):
            return True
        if ExceptionCheckers.is_error_str_context_window_exceeded(str(current)):
            return True
        current = current.__cause__ or current.__context__
    return False


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
        self.context_window_error: Optional[str] = None
        self._tool_map: Dict[str, Any] = {}  # MCPTool objects or OpenAI function dicts
        self._index_sync_lock = asyncio.Lock()

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
                    tools = await global_mcp_server_manager.get_tools_for_server(server_id)
                    all_tools.extend(tools)
                except Exception as e:
                    verbose_logger.warning(f"Failed to fetch tools from server {server_id}: {e}")
                    continue

            if not all_tools:
                verbose_logger.warning("No MCP tools found in registry")
                self.tool_router = None
                return

            verbose_logger.info(f"Fetched {len(all_tools)} tools from {len(registry)} MCP servers")
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
            function_spec = tool.get("function")
            if isinstance(function_spec, dict):
                # Chat Completions nested format:
                # {"type": "function", "function": {"name": ..., "description": ...}}
                name = function_spec.get("name", "")
                description = function_spec.get("description", name)
            else:
                # Flat format (legacy OpenAI functions / Responses API):
                # {"type": "function", "name": ..., "description": ...}
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
            self.context_window_error = None
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
            if _is_context_window_error(e):
                self.context_window_error = str(e)
                return
            raise

    def _has_tools_missing_from_index(self, tools: list[Any]) -> bool:
        """Allocation-free check for any named tool not yet in the semantic index."""
        return any(name and name not in self._tool_map for name in (self._extract_tool_info(t)[0] for t in tools))

    def _tools_missing_from_index(self, tools: list[Any]) -> dict[str, Any]:
        """Map name -> tool for every named tool not yet in the semantic index."""
        return {
            name: tool
            for name, tool in ((self._extract_tool_info(t)[0], t) for t in tools)
            if name and name not in self._tool_map
        }

    async def _ensure_tools_indexed(self, available_tools: list[Any]) -> None:
        """
        Index request-time tools the startup build never saw.

        The startup index lists every registered MCP server WITHOUT per-user
        credentials, so servers requiring per-user auth (interactive OAuth
        tokens, user-scoped env vars) contribute zero routes. Tools reaching
        the filter came through an authenticated expansion; without indexing
        them here they can never be selected, so requests either bypass
        filtering entirely (N->N) or lose every tool to unrelated matches.

        Runs async-only (no synchronous embedding on the request path) and
        never writes shared error state: an embedding failure here raises and
        is scoped to the requesting call, so one request's oversized tool
        description cannot poison the filter for other users on the worker.
        """
        from semantic_router.routers import SemanticRouter
        from semantic_router.routers.base import Route

        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )

        if not self._has_tools_missing_from_index(available_tools):
            return

        async with self._index_sync_lock:
            missing = self._tools_missing_from_index(available_tools)
            if not missing:
                return

            descriptions = {name: self._extract_tool_info(tool)[1] for name, tool in missing.items()}
            routes = [
                Route(
                    name=name,
                    description=description,
                    utterances=[description],
                    score_threshold=self.similarity_threshold,
                )
                for name, description in descriptions.items()
            ]

            if self.tool_router is None:
                router = SemanticRouter(
                    routes=[],
                    encoder=LiteLLMRouterEncoder(
                        litellm_router_instance=self.router_instance,
                        model_name=self.embedding_model,
                        score_threshold=self.similarity_threshold,
                    ),
                    auto_sync="local",
                    top_k=self.top_k,
                )
                await router.aadd(routes)
                self.tool_router = router
            else:
                await self.tool_router.aadd(routes)

            self._tool_map.update(missing)
            verbose_logger.info(
                f"Semantic tool filter indexed {len(routes)} request-time tools missing from the startup index"
            )

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

        if self.context_window_error is not None:
            raise SemanticToolFilterContextWindowError(
                embedding_model=self.embedding_model,
                stage="the MCP tool descriptions during semantic router build",
                original_error=self.context_window_error,
            )

        if not query or not query.strip():
            return available_tools

        # Run semantic filtering
        try:
            await self._ensure_tools_indexed(available_tools)

            if self.tool_router is None:
                verbose_logger.warning("Semantic router could not be built from the request's tools")
                return available_tools

            available_names = [name for name in (self._extract_tool_info(t)[0] for t in available_tools) if name]
            if not available_names:
                return available_tools

            limit = top_k or self.top_k
            if self.tool_router.top_k < limit:
                self.tool_router.top_k = limit
            matches = self.tool_router(text=query, limit=limit, route_filter=available_names)
            matched_tool_names = self._extract_tool_names_from_matches(matches)

            if not matched_tool_names:
                return available_tools

            filtered_tools = self._get_tools_by_names(matched_tool_names, available_tools)
            if not filtered_tools:
                return available_tools
            return filtered_tools

        except SemanticToolFilterContextWindowError:
            raise
        except Exception as e:
            if _is_context_window_error(e):
                verbose_logger.error(
                    f"Semantic tool filter embedding exceeded its context window: {e}",
                    exc_info=True,
                )
                raise SemanticToolFilterContextWindowError(
                    embedding_model=self.embedding_model,
                    stage="the user query or the MCP tool descriptions being indexed",
                    original_error=str(e),
                ) from e
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

    def _get_tools_by_names(self, tool_names: List[str], available_tools: List[Any]) -> List[Any]:
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

        if available_tools and not available_by_name:
            # Couldn't extract a usable name from any tool in the list, so
            # there's nothing to safely map the router's matches back onto.
            # Fail open instead of silently returning zero tools.
            verbose_logger.warning(
                f"Semantic tool filter: could not extract names from any of "
                f"{len(available_tools)} available tool(s); returning tools unfiltered"
            )
            return available_tools

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
