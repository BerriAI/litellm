import json
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from typing import Any, Final

from fastapi import APIRouter, HTTPException, Request

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_blog_posts import (
    BlogPost,
    BlogPostsResponse,
    GetBlogPosts,
    get_blog_posts,
)
from litellm.proxy._types import (
    CommonProxyErrors,
)
from litellm.proxy.utils import get_custom_url
from litellm.repositories.table_repositories import ClaudeCodePluginRepository
from litellm.types.agents import AgentCard
from litellm.types.mcp import MCPPublicServer
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupInfoProxy,
)
from litellm.types.proxy.public_endpoints.public_endpoints import (
    AgentCreateInfo,
    ProviderCreateInfo,
    PublicModelHubInfo,
    SupportedEndpointsResponse,
)
from litellm.types.utils import LlmProviders

router: Final = APIRouter()


# ---------------------------------------------------------------------------
# /public/endpoints — helpers
# ---------------------------------------------------------------------------

_ENDPOINT_METADATA: Final[dict[str, dict[str, str]]] = {
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


def _build_endpoints(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Transform raw provider_endpoints_support_backup.json into the response shape."""
    providers: Final[dict[str, Any]] = raw.get("providers", {})

    # Collect endpoint keys in insertion order (union across all providers).
    seen: Final[set] = set()
    all_keys: Final[list[str]] = []
    for provider_data in providers.values():
        for key in provider_data.get("endpoints", {}):
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    result: Final[list[dict[str, Any]]] = []
    for key in all_keys:
        meta = _ENDPOINT_METADATA.get(key)
        label = meta["label"] if meta else key.replace("_", " ").title()
        path = meta["endpoint"] if meta else "/" + key.replace("_", "/")

        supporting: list[dict[str, str]] = [
            {
                "slug": slug,
                "display_name": _clean_display_name(pd.get("display_name", slug)),
            }
            for slug, pd in providers.items()
            if pd.get("endpoints", {}).get(key)
        ]
        result.append({"key": key, "label": label, "endpoint": path, "providers": supporting})

    return result


def _load_endpoints() -> list[dict[str, Any]]:
    raw = json.loads(files("litellm").joinpath("provider_endpoints_support_backup.json").read_text(encoding="utf-8"))
    return _build_endpoints(raw)


def _read_bundled_json(filename: str) -> tuple[Mapping[str, Any], ...]:
    """Read one of the JSON data files bundled alongside this module."""
    with open(os.path.join(os.path.dirname(__file__), filename), "r") as f:
        return tuple(json.load(f))


@lru_cache(maxsize=1)
def _get_provider_create_fields() -> tuple[Mapping[str, Any], ...]:
    """Provider metadata for the dashboard create-model flow, read from disk once per process."""
    return _read_bundled_json("provider_create_fields.json")


def _agent_with_inherited_credentials(
    agent: Mapping[str, Any], provider_map: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    """One agent entry with its provider's credential fields appended after its own.

    ``inherit_credentials_from_provider`` is dropped from the result; the frontend
    does not consume it. Inherited fields are marked ``include_in_litellm_params``.
    """
    inherit_from: Final = agent.get("inherit_credentials_from_provider")
    provider: Final = provider_map.get(inherit_from) if inherit_from else None
    merged: Final = {key: value for key, value in agent.items() if key != "inherit_credentials_from_provider"}

    if provider is not None:
        inherited: Final = tuple(
            {**field, "include_in_litellm_params": True} for field in provider.get("credential_fields") or ()
        )
        merged["credential_fields"] = tuple(agent.get("credential_fields") or ()) + inherited

    return merged


@lru_cache(maxsize=1)
def _get_agent_create_fields() -> tuple[Mapping[str, Any], ...]:
    """Agent metadata for the dashboard create-agent flow, built once per process."""
    provider_map: Final = {provider["provider"]: provider for provider in _get_provider_create_fields()}
    return tuple(
        _agent_with_inherited_credentials(agent, provider_map)
        for agent in _read_bundled_json("agent_create_fields.json")
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
        _convert_health_check_to_dict,
    )
    from litellm.proxy.proxy_server import (
        _get_model_group_info,
        llm_router,
        prisma_client,
    )

    if llm_router is None:
        raise HTTPException(status_code=400, detail=CommonProxyErrors.no_llm_router.value)

    model_groups: list[ModelGroupInfoProxy] = []
    if litellm.public_model_groups is not None:
        model_groups = _get_model_group_info(
            llm_router=llm_router,
            all_models_str=litellm.public_model_groups,
            model_group=None,
        )

    # Fetch health check information if available
    health_checks_map: Final = {}
    if prisma_client is not None:
        try:
            latest_checks: Final = await prisma_client.get_all_latest_health_checks()
            for check in latest_checks:
                key = check.model_id if check.model_id else check.model_name
                if key:
                    health_check_dict = _convert_health_check_to_dict(check)
                    health_checks_map[key] = health_check_dict
                    if check.model_name:
                        health_checks_map[check.model_name] = health_check_dict
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
        MCPPublicServer(
            **server.model_dump(),
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
        _get_prisma_client,
    )
    from litellm.types.proxy.claude_code_endpoints import (
        ListPluginsResponse,
        PluginListItem,
    )

    try:
        prisma_client: Final = await _get_prisma_client()
        plugins: Final = await ClaudeCodePluginRepository(prisma_client).table.find_many(where={"enabled": True})
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
    from litellm.proxy.proxy_server import _title, version

    try:
        from litellm_enterprise.proxy.proxy_server import EnterpriseProxyConfig

        custom_docs_description = EnterpriseProxyConfig.get_custom_docs_description()
    except Exception:
        custom_docs_description = None

    return PublicModelHubInfo(
        docs_title=_title,
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
async def get_provider_fields() -> list[ProviderCreateInfo]:
    """
    Return provider metadata required by the dashboard create-model flow.

    Reads from the bundled local file. Result is cached in-process for the
    lifetime of the server process.
    """

    return _get_provider_create_fields()  # pyright: ignore[reportReturnType]  # response_model validates the dicts


@router.get(
    "/public/litellm_model_cost_map",
    tags=["public", "model management"],
)
async def get_litellm_model_cost_map():
    """
    Public endpoint to get the LiteLLM model cost map.
    Returns pricing information for all supported models.
    """
    import litellm

    try:
        _model_cost_map: Final = litellm.model_cost
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
async def get_litellm_blog_posts():
    """
    Public endpoint to get the latest LiteLLM blog posts.

    Fetches from GitHub with a 1-hour in-process cache.
    Falls back to the bundled local backup on any failure.
    """
    try:
        posts_data = get_blog_posts(url=litellm.blog_posts_url)
    except Exception as e:
        verbose_logger.warning("LiteLLM: get_litellm_blog_posts endpoint fallback triggered: %s", str(e))
        posts_data = GetBlogPosts.load_local_blog_posts()

    posts: Final = [BlogPost(**p) for p in posts_data[:5]]
    return BlogPostsResponse(posts=posts)


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
async def get_agent_fields() -> list[AgentCreateInfo]:
    """
    Return agent type metadata required by the dashboard create-agent flow.

    If an agent has `inherit_credentials_from_provider`, the provider's credential
    fields are automatically appended to the agent's credential_fields.

    Reads from the bundled local files. Result is cached in-process for the
    lifetime of the server process.
    """
    return _get_agent_create_fields()  # pyright: ignore[reportReturnType]  # response_model validates the dicts
