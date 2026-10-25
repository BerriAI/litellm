import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Final, Protocol

from fastapi import APIRouter, HTTPException, Request
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_blog_posts import (
    BlogPostsResponse,
    GetBlogPosts,
    get_blog_posts,
)
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.proxy._types import (
    CommonProxyErrors,
)
from litellm.proxy.db.health_check_latest import latest_by_model_name
from litellm.proxy.utils import get_custom_url
from litellm.repositories.table_repositories import ClaudeCodePluginRepository
from litellm.router_strategy.complexity_router.fuse_presets import FusePresetCatalog, get_fuse_presets
from litellm.types.agents import AgentCard
from litellm.types.mcp import MCPPublicServer
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)
from litellm.types.proxy.public_endpoints.public_endpoints import (
    AgentCreateInfo,
    AgentCredentialField,
    AutoRouterPresetRecord,
    ComplexityScorerDefaults,
    ProviderCreateInfo,
    PublicModelHubInfo,
    SupportedEndpointsResponse,
)
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from datetime import datetime

router: Final = APIRouter()


class _ProviderSupportEntry(TypedDict, total=False):
    display_name: ReadOnly[str]
    endpoints: ReadOnly[Mapping[str, bool]]


class _ProvidersFile(TypedDict, total=False):
    providers: ReadOnly[Mapping[str, _ProviderSupportEntry]]


class _EndpointProviderEntry(TypedDict):
    slug: ReadOnly[str]
    display_name: ReadOnly[str]


class _EndpointEntry(TypedDict):
    key: ReadOnly[str]
    label: ReadOnly[str]
    endpoint: ReadOnly[str]
    providers: ReadOnly[Sequence[_EndpointProviderEntry]]


class _PluginRow(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    @property
    def created_at(self) -> "datetime | None": ...

    @property
    def updated_at(self) -> "datetime | None": ...

    @property
    def manifest_json(self) -> str | None: ...


class _PluginTableActions(Protocol):
    def find_many(self, *, where: Mapping[str, bool]) -> Awaitable[Sequence[_PluginRow]]: ...


def _plugin_table(prisma_client: object) -> _PluginTableActions:
    return ClaudeCodePluginRepository(prisma_client).table


# ---------------------------------------------------------------------------
# /public/endpoints — helpers
# ---------------------------------------------------------------------------

_ENDPOINT_METADATA: Final[Mapping[str, Mapping[str, str]]] = {
    "chat_completions": {"label": "Chat Completions", "endpoint": "/chat/completions"},
    "messages": {"label": "Messages", "endpoint": "/messages"},
    "responses": {"label": "Responses", "endpoint": "/responses"},
    "embeddings": {"label": "Embeddings", "endpoint": "/embeddings"},
    "image_generations": {
        "label": "Image Generations",
        "endpoint": "/images/generations",
    },
    "audio_transcriptions": {
        "label": "Audio Transcriptions",
        "endpoint": "/audio/transcriptions",
    },
    "audio_speech": {"label": "Audio Speech", "endpoint": "/audio/speech"},
    "moderations": {"label": "Moderations", "endpoint": "/moderations"},
    "batches": {"label": "Batches", "endpoint": "/batches"},
    "rerank": {"label": "Rerank", "endpoint": "/rerank"},
    "ocr": {"label": "OCR", "endpoint": "/ocr"},
    "search": {"label": "Search", "endpoint": "/search"},
    "skills": {"label": "Skills", "endpoint": "/skills"},
    "interactions": {"label": "Interactions", "endpoint": "/interactions"},
    "a2a": {"label": "A2A (Agent Gateway)", "endpoint": "/a2a/{agent}/message/send"},
    "container": {"label": "Containers", "endpoint": "/containers"},
    "container_files": {
        "label": "Container Files",
        "endpoint": "/containers/{id}/files",
    },
    "compact": {"label": "Compact", "endpoint": "/responses/compact"},
    "files": {"label": "Files", "endpoint": "/files"},
    "image_edits": {"label": "Image Edits", "endpoint": "/images/edits"},
    "vector_stores_create": {
        "label": "Vector Stores (Create)",
        "endpoint": "/vector_stores",
    },
    "vector_stores_search": {
        "label": "Vector Stores (Search)",
        "endpoint": "/vector_stores/{id}/search",
    },
    "vector_store_files": {
        "label": "Vector Store Files",
        "endpoint": "/vector_stores/{id}/files",
    },
    "video_generations": {
        "label": "Video Generations",
        "endpoint": "/videos/generations",
    },
    "assistants": {"label": "Assistants", "endpoint": "/assistants"},
    "fine_tuning": {"label": "Fine Tuning", "endpoint": "/fine_tuning/jobs"},
    "text_completion": {"label": "Text Completion", "endpoint": "/completions"},
    "realtime": {"label": "Realtime", "endpoint": "/realtime"},
    "count_tokens": {"label": "Count Tokens", "endpoint": "/utils/token_counter"},
    "image_variations": {"label": "Image Variations", "endpoint": "/images/variations"},
    "generateContent": {"label": "Generate Content", "endpoint": "/generateContent"},
    "bedrock_invoke": {"label": "Bedrock Invoke", "endpoint": "/bedrock/invoke"},
    "bedrock_converse": {"label": "Bedrock Converse", "endpoint": "/bedrock/converse"},
    "rag_ingest": {"label": "RAG Ingest", "endpoint": "/rag/ingest"},
    "rag_query": {"label": "RAG Query", "endpoint": "/rag/query"},
}

_SLUG_SUFFIX_RE: Final = re.compile(r"\s*\(`[^`]+`\)\s*$")

# Loaded once on first request; never invalidated (local file, no TTL needed).
_cached_endpoints: SupportedEndpointsResponse | None = None


def _clean_display_name(raw: str) -> str:
    return _SLUG_SUFFIX_RE.sub("", raw).strip()


def _build_endpoints(raw: _ProvidersFile) -> list[_EndpointEntry]:
    """Transform raw provider_endpoints_support_backup.json into the response shape."""
    providers: Final = raw.get("providers", {})

    # Collect endpoint keys in insertion order (union across all providers).
    seen: Final[set[str]] = set()
    all_keys: Final[list[str]] = []
    for provider_data in providers.values():
        for key in provider_data.get("endpoints", {}):
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    result: Final[list[_EndpointEntry]] = []
    for key in all_keys:
        meta = _ENDPOINT_METADATA.get(key)
        label = meta["label"] if meta else key.replace("_", " ").title()
        path = meta["endpoint"] if meta else "/" + key.replace("_", "/")

        supporting: list[_EndpointProviderEntry] = [
            {
                "slug": slug,
                "display_name": _clean_display_name(pd.get("display_name", slug)),
            }
            for slug, pd in providers.items()
            if pd.get("endpoints", {}).get(key)
        ]
        result.append({"key": key, "label": label, "endpoint": path, "providers": supporting})

    return result


_PROVIDERS_FILE_ADAPTER: Final = TypeAdapter(_ProvidersFile)
_PROVIDER_CREATE_FIELDS_ADAPTER: Final = TypeAdapter(list[ProviderCreateInfo])


def _load_endpoints() -> list[_EndpointEntry]:
    raw: Final = _PROVIDERS_FILE_ADAPTER.validate_python(
        json.loads(files("litellm").joinpath("provider_endpoints_support_backup.json").read_text(encoding="utf-8"))
    )
    return _build_endpoints(raw)


class _BundledAgentCreateInfo(AgentCreateInfo):
    model_config = ConfigDict(frozen=True)

    inherit_credentials_from_provider: str | None = None


_BUNDLED_AGENTS_ADAPTER: Final = TypeAdapter(list[_BundledAgentCreateInfo])


def _read_bundled_file(filename: str) -> bytes:
    return files("litellm.proxy.public_endpoints").joinpath(filename).read_bytes()


@cache
def _provider_create_fields() -> tuple[ProviderCreateInfo, ...]:
    return tuple(_PROVIDER_CREATE_FIELDS_ADAPTER.validate_json(_read_bundled_file("provider_create_fields.json")))


def _inherited_credential_fields(provider: ProviderCreateInfo) -> tuple[AgentCredentialField, ...]:
    return tuple(
        AgentCredentialField.model_validate({**field.model_dump(), "include_in_litellm_params": True})
        for field in provider.credential_fields
    )


def _with_inherited_credentials(
    agent: _BundledAgentCreateInfo, providers: Mapping[str, ProviderCreateInfo]
) -> AgentCreateInfo:
    provider_name: Final = agent.inherit_credentials_from_provider
    provider: Final = providers.get(provider_name) if provider_name else None
    inherited: Final = _inherited_credential_fields(provider) if provider else ()
    return AgentCreateInfo.model_validate(
        {
            **agent.model_dump(exclude={"inherit_credentials_from_provider"}),
            "credential_fields": [*agent.credential_fields, *inherited],
        }
    )


@cache
def _agent_create_fields() -> tuple[AgentCreateInfo, ...]:
    providers: Final = {provider.provider: provider for provider in _provider_create_fields()}
    return tuple(
        _with_inherited_credentials(agent, providers)
        for agent in _BUNDLED_AGENTS_ADAPTER.validate_json(_read_bundled_file("agent_create_fields.json"))
    )


# ---------------------------------------------------------------------------


@router.get(
    "/public/model_hub",
    tags=["public", "model management"],
    response_model=list[ModelGroupInfoProxy],
)
async def public_model_hub():
    import litellm
    from litellm.proxy.health_endpoints._health_endpoints import (
        convert_health_check_to_dict,
    )
    from litellm.proxy.proxy_server import (
        get_model_group_info,
        llm_router,
        prisma_client,
    )

    if llm_router is None:
        raise HTTPException(status_code=400, detail=CommonProxyErrors.no_llm_router.value)

    model_groups: list[ModelGroupInfoProxy] = []
    if litellm.public_model_groups is not None:
        model_groups = get_model_group_info(  # rebind-ok: pre-existing rebinding on a rename-only line
            llm_router=llm_router,
            all_models_str=litellm.public_model_groups,
            model_group=None,
        )

    # Fetch health check information if available
    health_checks_map: Final = {}
    if prisma_client is not None:
        try:
            latest_checks: Final = await prisma_client.get_all_latest_health_checks()
            for model_name, check in latest_by_model_name(latest_checks).items():
                health_checks_map[model_name] = convert_health_check_to_dict(check)
        except Exception:
            pass

    for model_group in model_groups:
        health_info = health_checks_map.get(model_group.model_group)
        if health_info:
            model_group.health_status = health_info.get("status")
            model_group.health_response_time = health_info.get("response_time_ms")
            model_group.health_checked_at = health_info.get("checked_at")

    return model_groups


@router.get(
    "/public/agent_hub",
    tags=["[beta] Agents", "public"],
    response_model=list[AgentCard],
)
async def get_agents(request: Request):
    import litellm
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    agents: Final = global_agent_registry.get_public_agent_list()

    if litellm.public_agent_groups is None:
        return []

    return [
        {
            **(agent.agent_card_params or {}),
            "url": get_custom_url(str(request.base_url), route=f"a2a/{agent.agent_id}"),
        }
        for agent in agents
        if not global_agent_registry.ids_for_agent(agent.agent_id).isdisjoint(litellm.public_agent_groups)
    ]


@router.get(
    "/public/mcp_hub",
    tags=["[beta] MCP", "public"],
    response_model=list[MCPPublicServer],
)
async def get_mcp_servers():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    public_mcp_servers: Final = global_mcp_server_manager.get_public_mcp_servers()
    return [
        MCPPublicServer.model_validate(
            {
                **server.model_dump(),
                "mcp_info": {
                    **(server.mcp_info or {}),
                    "is_public": True,
                    "is_public_explicit": server.server_id in (litellm.public_mcp_servers or ()),
                },
            }
        )
        for server in public_mcp_servers
    ]


@router.get(
    "/public/skill_hub",
    tags=["public", "Claude Code Marketplace"],
)
async def public_skill_hub():
    """Return enabled (public) Claude Code skills — no auth required."""
    from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
        MANIFEST_FIELDS,
        get_installation_preference,
        get_prisma_client,
    )
    from litellm.types.proxy.claude_code_endpoints import (
        ListPluginsResponse,
        PluginListItem,
    )

    try:
        prisma_client: Final = await get_prisma_client()
        plugins: Final = await _plugin_table(prisma_client).find_many(where={"enabled": True})
        items: Final = []
        for plugin in plugins:
            raw = plugin.manifest_json or {}
            manifest = json.loads(raw) if isinstance(raw, str) else raw
            items.append(
                PluginListItem(
                    id=plugin.id,
                    name=plugin.name,
                    enabled=plugin.enabled,
                    created_at=str(plugin.created_at) if plugin.created_at else None,
                    updated_at=str(plugin.updated_at) if plugin.updated_at else None,
                    source=manifest.get("source", {}),
                    description=manifest.get("description"),
                    version=manifest.get("version"),
                    category=manifest.get("category"),
                    keywords=manifest.get("keywords"),
                    author=manifest.get("author"),
                    homepage=manifest.get("homepage"),
                    domain=manifest.get("domain"),
                    namespace=manifest.get("namespace"),
                    installation_preference=get_installation_preference(MANIFEST_FIELDS.validate_python(manifest)),
                )
            )
        return ListPluginsResponse(plugins=items, count=len(items))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/public/model_hub/info",
    tags=["public", "model management"],
    response_model=PublicModelHubInfo,
)
async def public_model_hub_info():
    import litellm
    from litellm.proxy.proxy_server import title, version

    try:
        from litellm_enterprise.proxy.proxy_server import EnterpriseProxyConfig

        custom_docs_description = EnterpriseProxyConfig.get_custom_docs_description()
    except Exception:
        custom_docs_description = None

    return PublicModelHubInfo(
        docs_title=title,
        custom_docs_description=custom_docs_description,
        litellm_version=version,
        useful_links=litellm.public_model_groups_links,
    )


@router.get(
    "/public/providers",
    tags=["public", "providers"],
    response_model=list[str],
)
async def get_supported_providers() -> list[str]:
    """
    Return a sorted list of all providers supported by LiteLLM.
    """

    return sorted(provider.value for provider in LlmProviders)


@router.get(
    "/public/providers/fields",
    tags=["public", "providers"],
    response_model=list[ProviderCreateInfo],
)
async def get_provider_fields() -> tuple[ProviderCreateInfo, ...]:
    """
    Return provider metadata required by the dashboard create-model flow.

    Reads from the bundled local file. Result is cached in-process for the
    lifetime of the server process.
    """

    return _provider_create_fields()


@router.get(
    "/public/complexity_router/scorer_defaults",
    tags=["public", "auto router"],
    response_model=ComplexityScorerDefaults,
)
async def get_complexity_scorer_defaults() -> ComplexityScorerDefaults:
    """
    Return the complexity router's shipped heuristic scorer defaults, for the dashboard to prefill with.
    """
    from litellm.router_strategy.complexity_router.config import (
        DEFAULT_DIMENSION_WEIGHTS,
        DEFAULT_TIER_BOUNDARIES,
        DEFAULT_TOKEN_THRESHOLDS,
    )

    return ComplexityScorerDefaults(
        tier_boundaries=DEFAULT_TIER_BOUNDARIES,
        token_thresholds=DEFAULT_TOKEN_THRESHOLDS,
        dimension_weights=DEFAULT_DIMENSION_WEIGHTS,
    )


@router.get(
    "/public/complexity_router/fuse_presets",
    response_model=FusePresetCatalog,
)
async def get_public_fuse_presets() -> FusePresetCatalog:
    return get_fuse_presets()


@router.get(
    "/public/litellm_model_cost_map",
    tags=["public", "model management"],
)
async def get_litellm_model_cost_map(catalog_only: bool = False):
    """
    Public endpoint to get the LiteLLM model cost map.
    Returns pricing information for all supported models.
    With catalog_only=true, returns the catalog as loaded, without entries registered at runtime for proxy deployments.
    """
    import litellm

    try:
        _model_cost_map: Final = GetModelCostMap.loaded_model_cost_map() if catalog_only else litellm.model_cost
        return _model_cost_map
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error ({e})",
        )


@router.get(
    "/public/litellm_blog_posts",
    tags=["public"],
    response_model=BlogPostsResponse,
)
async def get_litellm_blog_posts() -> BlogPostsResponse:
    """
    Public endpoint to get the latest LiteLLM blog posts, each with its RSS categories as ``tags``.

    Fetches the docs RSS feed with a 1-hour in-process cache.
    Falls back to the bundled local backup on any failure.
    """
    try:
        return BlogPostsResponse(posts=get_blog_posts(url=litellm.blog_posts_url))
    except Exception as e:
        verbose_logger.warning("LiteLLM: get_litellm_blog_posts endpoint fallback triggered: %s", str(e))
        return BlogPostsResponse(posts=GetBlogPosts.load_local_blog_posts())


_AUTOROUTER_PRESETS_ADAPTER: Final = TypeAdapter(dict[str, AutoRouterPresetRecord])


def _load_bundled_autorouter_presets() -> Mapping[str, AutoRouterPresetRecord]:
    return _AUTOROUTER_PRESETS_ADAPTER.validate_python(
        json.loads(
            files("litellm.proxy.public_endpoints").joinpath("autorouter_presets.json").read_text(encoding="utf-8")
        )
    )


async def _fetch_remote_bytes(url: str) -> bytes:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.UI)
    response: Final = await client.get(url, timeout=5.0)
    response.raise_for_status()
    return response.content


async def _fetch_remote_autorouter_presets(url: str) -> Mapping[str, AutoRouterPresetRecord]:
    presets: Final = _AUTOROUTER_PRESETS_ADAPTER.validate_json(await _fetch_remote_bytes(url))
    if not presets:
        raise ValueError("remote auto-router preset catalog is empty")
    return presets


async def _resolve_autorouter_presets(
    url: str,
    fetch: Callable[[str], Awaitable[Mapping[str, AutoRouterPresetRecord]]],
) -> Mapping[str, AutoRouterPresetRecord]:
    if is_offline_mode() or os.getenv("LITELLM_LOCAL_AUTOROUTER_PRESETS", "").lower() == "true":
        return _load_bundled_autorouter_presets()
    try:
        return await fetch(url)
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: failed to fetch auto-router presets from %s: %s. Serving the bundled catalog for the life of this process.",
            url,
            str(e),
        )
        return _load_bundled_autorouter_presets()


class _AutoRouterPresetsCache:
    presets: Mapping[str, AutoRouterPresetRecord] | None = None
    lock: asyncio.Lock | None = None


async def get_autorouter_presets(
    url: str,
    fetch: Callable[[str], Awaitable[Mapping[str, AutoRouterPresetRecord]]] = _fetch_remote_autorouter_presets,
) -> Mapping[str, AutoRouterPresetRecord]:
    cached: Final = _AutoRouterPresetsCache.presets
    if cached is not None:
        return cached
    if _AutoRouterPresetsCache.lock is None:
        _AutoRouterPresetsCache.lock = asyncio.Lock()
    async with _AutoRouterPresetsCache.lock:
        held: Final = _AutoRouterPresetsCache.presets
        if held is not None:
            return held
        resolved: Final = await _resolve_autorouter_presets(url=url, fetch=fetch)
        _AutoRouterPresetsCache.presets = resolved
        return resolved


@router.get(
    "/public/autorouter_presets",
    tags=["public", "auto router"],
    response_model=dict[str, AutoRouterPresetRecord],
)
async def get_public_autorouter_presets() -> Mapping[str, AutoRouterPresetRecord]:
    """
    Return the auto-router preset catalog the dashboard's template picker renders.

    Resolved once per process, like the model cost map: fetched from ``litellm.autorouter_presets_url``
    (override with ``LITELLM_AUTOROUTER_PRESETS_URL``) on the first request, falling back to the
    catalog bundled with the package on any failure. Set ``LITELLM_LOCAL_AUTOROUTER_PRESETS=True``
    to serve the bundled catalog only. A restart picks up a newly published catalog.
    """
    return await get_autorouter_presets(url=litellm.autorouter_presets_url)


@router.get(
    "/public/endpoints",
    tags=["public"],
    response_model=SupportedEndpointsResponse,
)
async def get_supported_endpoints() -> SupportedEndpointsResponse:
    """
    Return the list of LiteLLM proxy endpoints and which providers support each one.

    Reads from the bundled local backup file. Result is cached in-process for
    the lifetime of the server process.
    """
    global _cached_endpoints
    if _cached_endpoints is None:
        _cached_endpoints = SupportedEndpointsResponse(endpoints=_load_endpoints())
    return _cached_endpoints


@router.get(
    "/public/agents/fields",
    tags=["public", "[beta] Agents"],
    response_model=list[AgentCreateInfo],
)
async def get_agent_fields() -> tuple[AgentCreateInfo, ...]:
    """
    Return agent type metadata required by the dashboard create-agent flow.

    If an agent has `inherit_credentials_from_provider`, the provider's credential
    fields are automatically appended to the agent's credential_fields.

    Reads from the bundled local files. Result is cached in-process for the
    lifetime of the server process.
    """
    return _agent_create_fields()
