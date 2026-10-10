"""
Semantic MCP Tool Filtering using semantic-router

Filters MCP tools semantically for /chat/completions and /responses endpoints.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from mcp.types import Tool as MCPTool
from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.exceptions import ContextWindowExceededError
from litellm.litellm_core_utils.exception_mapping_utils import ExceptionCheckers
from litellm.proxy._experimental.mcp_server.catalog import catalog_operation, global_manager
from litellm.proxy._experimental.mcp_server.faults import iter_exception_tree
from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
    CachingToolEncoder,
    ToolVectorStore,
    tool_vector_key,
)
from litellm.proxy._experimental.mcp_server.utils import MCP_TOOL_PREFIX_SEPARATOR

if TYPE_CHECKING:
    from semantic_router.routers import SemanticRouter

    from litellm.router import Router
    from litellm.router_strategy.auto_router.litellm_encoder import LiteLLMRouterEncoder


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


def _is_context_window_error(error: BaseException | None) -> bool:
    """Detect a context-window overflow anywhere in an exception's tree."""
    if error is None:
        return False
    return any(
        isinstance(current, ContextWindowExceededError)
        or ExceptionCheckers.is_error_str_context_window_exceeded(str(current))
        for current in iter_exception_tree(error)
    )


class SemanticMCPToolFilter:
    """Filters MCP tools using semantic similarity to reduce context window size."""

    def __init__(
        self,
        embedding_model: str,
        litellm_router_instance: "Router",
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        enabled: bool = True,
        vector_store: ToolVectorStore | None = None,
        lock_poll_interval_s: float = 5.0,
    ):
        """
        Initialize the semantic tool filter.

        Args:
            embedding_model: Model to use for embeddings (e.g., "text-embedding-3-small")
            litellm_router_instance: Router instance for embedding generation
            top_k: Maximum number of tools to return
            similarity_threshold: Minimum similarity score for filtering
            enabled: Whether filtering is enabled
            vector_store: Shared embedding store; when set the index builds
                cross-pod in the background and requests pass through unfiltered
                until startup_index_ready flips
            lock_poll_interval_s: How often to re-check the store while another
                pod holds the build lock
        """
        self.enabled = enabled
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.router_instance = litellm_router_instance
        self.tool_router: SemanticRouter | None = None
        self.context_window_error: str | None = None
        self._tool_map: dict[str, object] = {}  # MCPTool objects or OpenAI function dicts
        self._index_sync_lock = asyncio.Lock()
        self._vector_store = vector_store
        self._lock_poll_interval_s = lock_poll_interval_s
        self.embedding_identity = self._compute_embedding_identity()
        self.startup_index_ready = vector_store is None

    def _compute_embedding_identity(self) -> str:
        """embedding_model plus every deployment model it routes to, so repointing an alias invalidates the shared vectors."""
        try:
            deployments: Final = self.router_instance.get_model_list(model_name=self.embedding_model) or ()
            deployment_maps: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(deployments)
            litellm_params: Final = tuple(
                TypeAdapter(Mapping[str, object]).validate_python(deployment.get("litellm_params") or {})
                for deployment in deployment_maps
            )
            deployment_models: Final = sorted(
                {str(model) for model in (p.get("model") for p in litellm_params) if model}
            )
        except (AttributeError, TypeError, KeyError, ValueError):
            return self.embedding_model
        if not deployment_models:
            return self.embedding_model
        return self.embedding_model + "|" + "|".join(deployment_models)

    def _new_encoder(self) -> "LiteLLMRouterEncoder":
        """Encoder used for every index build; cached in Redis when a store is set."""
        from litellm.router_strategy.auto_router.litellm_encoder import (
            LiteLLMRouterEncoder,
        )

        if self._vector_store is not None:
            return CachingToolEncoder(
                litellm_router_instance=self.router_instance,
                model_name=self.embedding_model,
                score_threshold=self.similarity_threshold,
                store=self._vector_store,
                embedding_identity=self.embedding_identity,
            )
        return LiteLLMRouterEncoder(
            litellm_router_instance=self.router_instance,
            model_name=self.embedding_model,
            score_threshold=self.similarity_threshold,
        )

    @catalog_operation(global_manager)
    async def build_router_from_mcp_registry(self, *, async_index: bool = False) -> None:
        """Build semantic router from all MCP tools in the registry (no auth checks).

        With a vector store the shared Redis path runs: acquire the build lock
        or wait for the holding pod's vectors, then embed only the misses.
        Without one, the sync build blocks startup exactly as before; the
        keyword arg lets a background task run the async (aadd) path instead.
        """
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        try:
            # Get all servers from registry without auth checks
            registry: Final = global_mcp_server_manager.get_registry()
            if not registry:
                verbose_logger.warning("MCP registry is empty")
                self.tool_router = None
                return

            # Fetch tools from all servers in parallel
            all_tools: Final[Sequence[object]] = []
            for server_id, server in registry.items():
                try:
                    tools = await global_mcp_server_manager.get_tools_for_server(server_id)
                    all_tools.extend(tools)
                except Exception as e:
                    verbose_logger.warning("Failed to fetch tools from server %s: %s", server_id, e)
                    continue

            if not all_tools:
                verbose_logger.warning("No MCP tools found in registry")
                self.tool_router = None
                return

            verbose_logger.info("Fetched %s tools from %s MCP servers", len(all_tools), len(registry))
            if self._vector_store is not None or async_index:
                await self._abuild_router(all_tools)
            else:
                self._build_router(all_tools)

        except Exception as e:
            verbose_logger.error("Failed to build router from MCP registry: %s", e)
            self.tool_router = None
            raise
        finally:
            self.startup_index_ready = True

    def extract_tool_info(self, tool: object) -> tuple[str, str]:
        """Extract name and description from MCP tool or OpenAI function dict."""
        name: str
        description: str

        if isinstance(tool, dict):
            # OpenAI function format
            tool_dict: Final = TypeAdapter(Mapping[str, object]).validate_python(tool)
            name = str(tool_dict.get("name", ""))
            description = str(tool_dict.get("description", name))
        else:
            # MCPTool object
            mcp_tool: Final = MCPTool.model_validate(tool, from_attributes=True)
            name = str(mcp_tool.name)
            description = str(mcp_tool.description) if mcp_tool.description else str(mcp_tool.name)

        return name, description

    _extract_tool_info = extract_tool_info

    def _tools_to_routes_and_map(self, tools: Sequence[object]) -> tuple[Sequence[object], Mapping[str, object]]:
        from semantic_router.routers.base import Route

        extracted: Final = tuple((self.extract_tool_info(tool), tool) for tool in tools)
        tool_map: Final = {name: tool for (name, _), tool in extracted}
        routes: Final[Sequence[object]] = [
            Route(
                name=name,
                description=description,
                utterances=[description],
                score_threshold=self.similarity_threshold,
            )
            for (name, description), _ in extracted
        ]
        return routes, tool_map

    def _handle_build_error(self, error: Exception) -> bool:
        """True when the failure was recorded as a context-window overflow; False when the caller should re-raise."""
        verbose_logger.error("Failed to build semantic router: %s", error)
        self.tool_router = None
        if _is_context_window_error(error):
            self.context_window_error = str(error)
            return True
        return False

    def _build_router(self, tools: Sequence[object]) -> None:
        """Build semantic router with tools (MCPTool objects or OpenAI function dicts)."""
        from semantic_router.routers import SemanticRouter

        if not tools:
            self.tool_router = None
            return

        try:
            self.context_window_error = None
            routes, tool_map = self._tools_to_routes_and_map(tools)
            self.tool_router = SemanticRouter(
                routes=list(routes),
                encoder=self._new_encoder(),
                auto_sync="local",
            )
            self._tool_map = dict(tool_map)

            verbose_logger.info("Built semantic router with %s tools", len(routes))

        except Exception as e:
            if self._handle_build_error(e):
                return
            raise

    async def _abuild_router(self, tools: Sequence[object]) -> None:
        """
        Async index build for the background path.

        When a store is set, this first waits on the cross-pod build lock so
        only one pod embeds; the caching encoder then fills any vectors the
        holder did not write.
        """
        from semantic_router.routers import SemanticRouter

        if not tools:
            self.tool_router = None
            return

        lock_held: Final = await self._await_store_or_lock(tools) if self._vector_store is not None else False
        try:
            self.context_window_error = None
            routes, tool_map = self._tools_to_routes_and_map(tools)
            encoder: Final = self._new_encoder()
            router: Final = SemanticRouter(  # pyright: ignore[reportUnknownVariableType]  # semantic_router ships no type stubs
                routes=[],
                encoder=encoder,
                init_async_index=True,
                top_k=self.top_k,
            )

            # aadd's lazy init probes index dimensions through the sync encoder,
            # so dimensions are seeded with an async probe to keep the build async.
            dims_probe: Final = await encoder.aencode_queries(["test"])  # pyright: ignore[reportUnknownMemberType]  # semantic_router ships no type stubs
            router.index.dimensions = len(dims_probe[0])  # pyright: ignore[reportUnknownMemberType]  # semantic_router ships no type stubs
            await router.aadd(list(routes))  # pyright: ignore[reportUnknownMemberType]  # semantic_router ships no type stubs

            self._tool_map = dict(tool_map)
            self.tool_router = router

            verbose_logger.info("Built semantic router with %s tools via async index build", len(routes))

        except Exception as e:
            if self._handle_build_error(e):
                return
            raise
        finally:
            if lock_held and self._vector_store is not None:
                await self._vector_store.release_build_lock()

    async def _await_store_or_lock(self, tools: Sequence[object]) -> bool:
        """True when this pod holds the build lock after the wait; False when every vector arrived without it."""
        store: Final = self._vector_store
        if store is None:
            return False

        descriptions: Final = tuple(self.extract_tool_info(tool)[1] for tool in tools)
        keys: Final = tuple(tool_vector_key(self.embedding_identity, text) for text in set(descriptions))

        while {key for key in keys if key not in await store.get_many(keys)}:
            if await store.try_acquire_build_lock():
                return True
            await asyncio.sleep(self._lock_poll_interval_s)
        return False

    def _has_tools_missing_from_index(self, tools: Sequence[object]) -> bool:
        """Allocation-free check for any named tool not yet in the semantic index."""
        return any(
            name and (name not in self._tool_map or self.extract_tool_info(self._tool_map[name])[1] != description)
            for name, description in (self.extract_tool_info(t) for t in tools)
        )

    def _tools_missing_from_index(self, tools: Sequence[object]) -> Mapping[str, object]:
        """Map name -> tool for every named tool not yet in the semantic index."""
        infos: Final = ((self.extract_tool_info(t), t) for t in tools)
        return {
            name: tool
            for (name, description), tool in infos
            if name and (name not in self._tool_map or self.extract_tool_info(self._tool_map[name])[1] != description)
        }

    async def _ensure_tools_indexed(self, available_tools: Sequence[object]) -> None:
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

        if not self._has_tools_missing_from_index(available_tools):
            return

        async with self._index_sync_lock:
            missing: Final = self._tools_missing_from_index(available_tools)
            if not missing:
                return

            routes, _ = self._tools_to_routes_and_map(tuple(missing.values()))

            if self.tool_router is None:
                router: Final = SemanticRouter(
                    routes=[],
                    encoder=self._new_encoder(),
                    auto_sync="local",
                    top_k=self.top_k,
                )
                await router.aadd(list(routes))
                self.tool_router = router
            else:
                for name in (name for name in missing if name in self._tool_map):
                    await self.tool_router.adelete(name)  # pyright: ignore[reportUnknownMemberType]  # semantic_router ships no type stubs
                await self.tool_router.aadd(list(routes))

            self._tool_map.update(missing)
            verbose_logger.info(
                "Semantic tool filter indexed %s request-time tools missing from the startup index", len(routes)
            )

    async def filter_tools(
        self,
        query: str,
        available_tools: list[Any],
        top_k: int | None = None,
    ) -> list[Any]:
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

            available_names: Final = [name for name in (self.extract_tool_info(t)[0] for t in available_tools) if name]
            if not available_names:
                return available_tools

            limit: Final = top_k or self.top_k
            self.tool_router.top_k = max(self.tool_router.top_k, limit)
            matches: Final = self.tool_router(text=query, limit=limit, route_filter=available_names)
            matched_tool_names: Final = self._extract_tool_names_from_matches(matches)

            if not matched_tool_names:
                return available_tools

            filtered_tools: Final = self._get_tools_by_names(matched_tool_names, available_tools)
            if not filtered_tools:
                return available_tools
            return filtered_tools

        except SemanticToolFilterContextWindowError:
            raise
        except Exception as e:
            if _is_context_window_error(e):
                verbose_logger.error(
                    "Semantic tool filter embedding exceeded its context window: %s",
                    e,
                    exc_info=True,
                )
                raise SemanticToolFilterContextWindowError(
                    embedding_model=self.embedding_model,
                    stage="the user query or the MCP tool descriptions being indexed",
                    original_error=str(e),
                ) from e
            verbose_logger.error("Semantic tool filter failed: %s", e, exc_info=True)
            return available_tools

    def _extract_tool_names_from_matches(self, matches) -> list[str]:
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
        separator: Final = client_name[-len(canonical) - 1]
        return separator in ("_", "-")

    def _get_tools_by_names(self, tool_names: Sequence[str], available_tools: Sequence[object]) -> list[object]:
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
        available_by_name: Final[dict[str, object]] = {}
        for tool in available_tools:
            client_name, _ = self.extract_tool_info(tool)
            if client_name and client_name not in available_by_name:
                available_by_name[client_name] = tool

        matched: Final[list[object]] = []
        used_ids: Final[set[int]] = set()
        for canonical in tool_names:
            tool = available_by_name.get(canonical)
            if tool is None:
                # Prefer the shortest qualifying name. When several
                # incoming tools suffix-match the same canonical (e.g.
                # "my_search" and "my_tag_search" both end in "search"),
                # the one closest in length to the canonical is the
                # least-wrapped and most likely the intended target.
                best_name: str | None = None
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

    def extract_user_query(self, messages: Sequence[Mapping[str, object]]) -> str:
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
