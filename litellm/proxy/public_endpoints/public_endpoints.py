import json
import os
import re
from importlib.resources import files
from typing import Any, Dict, List, Optional

import litellm
from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_blog_posts import (
    BlogPost,
    BlogPostsResponse,
    GetBlogPosts,
    get_blog_posts,
)
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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

router = APIRouter()

# ---------------------------------------------------------------------------
# /public/endpoints — helpers
# ---------------------------------------------------------------------------

_ENDPOINT_METADATA: Dict[str, Dict[str, str]] = {
    "chat_completions": {"label": "Chat Completions", "endpoint": "/chat/completions"},
    "messages": {"label": "Messages", "endpoint": "/messages"},
    "responses": {"label": "Responses", "endpoint": "/responses"},
    "embeddings": {"label": "Embeddings", "endpoint": "/embeddings"},
    "image_generations": {"label": "Image Generations", "endpoint": "/images/generations"},
    "audio_transcriptions": {"label": "Audio Transcriptions", "endpoint": "/audio/transcriptions"},
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
    "container_files": {"label": "Container Files", "endpoint": "/containers/{id}/files"},
    "compact": {"label": "Compact", "endpoint": "/responses/compact"},
    "files": {"label": "Files", "endpoint": "/files"},
    "image_edits": {"label": "Image Edits", "endpoint": "/images/edits"},
    "vector_stores_create": {"label": "Vector Stores (Create)", "endpoint": "/vector_stores"},
    "vector_stores_search": {"label": "Vector Stores (Search)", "endpoint": "/vector_stores/{id}/search"},
    "vector_store_files": {"label": "Vector Store Files", "endpoint": "/vector_stores/{id}/files"},
    "video_generations": {"label": "Video Generations", "endpoint": "/videos/generations"},
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

_SLUG_SUFFIX_RE = re.compile(r"\s*\(`[^`]+`\)\s*$")

# Loaded once on first request; never invalidated (local file, no TTL needed).
_cached_endpoints: Optional[List[Dict[str, Any]]] = None


def _clean_display_name(raw: str) -> str:
    return _SLUG_SUFFIX_RE.sub("", raw).strip()


def _build_endpoints(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transform raw provider_endpoints_support_backup.json into the response shape."""
    providers: Dict[str, Any] = raw.get("providers", {})

    # Collect endpoint keys in insertion order (union across all providers).
    seen: set = set()
    all_keys: List[str] = []
    for provider_data in providers.values():
        for key in provider_data.get("endpoints", {}):
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    result: List[Dict[str, Any]] = []
    for key in all_keys:
        meta = _ENDPOINT_METADATA.get(key)
        label = meta["label"] if meta else key.replace("_", " ").title()
        path = meta["endpoint"] if meta else "/" + key.replace("_", "/")

        supporting: List[Dict[str, str]] = [
            {
                "slug": slug,
                "display_name": _clean_display_name(pd.get("display_name", slug)),
            }
            for slug, pd in providers.items()
            if pd.get("endpoints", {}).get(key)
        ]
        result.append({"key": key, "label": label, "endpoint": path, "providers": supporting})

    return result


def _load_endpoints() -> List[Dict[str, Any]]:
    raw = json.loads(
        files("litellm")
        .joinpath("provider_endpoints_support_backup.json")
        .read_text(encoding="utf-8")
    )
    return _build_endpoints(raw)


# ---------------------------------------------------------------------------


@router.get(
    "/public/model_hub",
    tags=["public", "model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[ModelGroupInfoProxy],
)
async def public_model_hub():
    import litellm
    from litellm.proxy.proxy_server import _get_model_group_info, llm_router, prisma_client
    from litellm.proxy.health_endpoints._health_endpoints import _convert_health_check_to_dict

    if llm_router is None:
        raise HTTPException(
            status_code=400, detail=CommonProxyErrors.no_llm_router.value
        )

    model_groups: List[ModelGroupInfoProxy] = []
    if litellm.public_model_groups is not None:
        model_groups = _get_model_group_info(
            llm_router=llm_router,
            all_models_str=litellm.public_model_groups,
            model_group=None,
        )

    # Fetch health check information if available
    health_checks_map = {}
    if prisma_client is not None:
        try:
            latest_checks = await prisma_client.get_all_latest_health_checks()
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
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[AgentCard],
)
async def get_agents():
    import litellm
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    agents = global_agent_registry.get_public_agent_list()

    if litellm.public_agent_groups is None:
        return []
    agent_card_list = [
        agent.agent_card_params
        for agent in agents
        if agent.agent_id in litellm.public_agent_groups
    ]
    return agent_card_list


@router.get(
    "/public/mcp_hub",
    tags=["[beta] MCP", "public"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[MCPPublicServer],
)
async def get_mcp_servers():
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    public_mcp_servers = global_mcp_server_manager.get_public_mcp_servers()
    return [
        MCPPublicServer(
            **server.model_dump(),
        )
        for server in public_mcp_servers
    ]


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
    response_model=List[str],
)
async def get_supported_providers() -> List[str]:
    """
    Return a sorted list of all providers supported by LiteLLM.
    """

    return sorted(provider.value for provider in LlmProviders)


@router.get(
    "/public/providers/fields",
    tags=["public", "providers"],
    response_model=List[ProviderCreateInfo],
)
async def get_provider_fields() -> List[ProviderCreateInfo]:
    """
    Return provider metadata required by the dashboard create-model flow.
    """

    provider_create_fields_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "proxy",
        "public_endpoints",
        "provider_create_fields.json"
    )

    with open(provider_create_fields_path, "r") as f:
        provider_create_fields = json.load(f)

    return provider_create_fields


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
        _model_cost_map = litellm.model_cost
        return _model_cost_map
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error ({str(e)})",
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
        verbose_logger.warning(
            "LiteLLM: get_litellm_blog_posts endpoint fallback triggered: %s", str(e)
        )
        posts_data = GetBlogPosts.load_local_blog_posts()

    posts = [BlogPost(**p) for p in posts_data[:5]]
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
    response_model=List[AgentCreateInfo],
)
async def get_agent_fields() -> List[AgentCreateInfo]:
    """
    Return agent type metadata required by the dashboard create-agent flow.

    If an agent has `inherit_credentials_from_provider`, the provider's credential
    fields are automatically appended to the agent's credential_fields.
    """
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "proxy",
        "public_endpoints",
    )

    agent_create_fields_path = os.path.join(base_path, "agent_create_fields.json")
    provider_create_fields_path = os.path.join(base_path, "provider_create_fields.json")

    with open(agent_create_fields_path, "r") as f:
        agent_create_fields = json.load(f)

    with open(provider_create_fields_path, "r") as f:
        provider_create_fields = json.load(f)

    # Build a lookup map for providers by name
    provider_map = {p["provider"]: p for p in provider_create_fields}

    # Merge inherited credential fields
    for agent in agent_create_fields:
        inherit_from = agent.get("inherit_credentials_from_provider")
        if inherit_from and inherit_from in provider_map:
            provider = provider_map[inherit_from]
            # Copy provider fields and mark them for inclusion in litellm_params
            inherited_fields = []
            for field in provider.get("credential_fields", []):
                field_copy = field.copy()
                field_copy["include_in_litellm_params"] = True
                inherited_fields.append(field_copy)
            # Append provider credential fields after agent's own fields
            agent["credential_fields"] = agent.get("credential_fields", []) + inherited_fields
        # Remove the inherit field from response (not needed by frontend)
        agent.pop("inherit_credentials_from_provider", None)

    return agent_create_fields
