import json
import os
import re
from typing import List

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
    SupportedEndpointInfo,
    SupportedEndpointsResponse,
    SupportedProviderInfo,
)
from litellm.types.utils import LlmProviders

router = APIRouter()

_supported_endpoints_cache: SupportedEndpointsResponse | None = None


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
    "/public/supported_endpoints",
    tags=["public", "providers"],
    response_model=SupportedEndpointsResponse,
)
async def get_provider_supported_endpoints() -> SupportedEndpointsResponse:
    """
    Return all supported endpoints and which providers support them.

    Reads from provider_endpoints_support.json at the repo root.
    Result is cached for the lifetime of the process.
    """
    global _supported_endpoints_cache
    if _supported_endpoints_cache is not None:
        return _supported_endpoints_cache

    provider_endpoints_support_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "proxy",
        "public_endpoints",
        "provider_endpoints_support.json",
    )

    with open(provider_endpoints_support_path, "r") as f:
        data = json.load(f)

    schema_endpoints = data["_schema"]["provider_slug"]["endpoints"]

    endpoints = []
    for key, description in schema_endpoints.items():
        path_match = re.search(r"(/[\w/{}.()*-]+)", description)
        endpoint_path = path_match.group(1) if path_match else f"/{key}"
        display_name = key.replace("_", " ").title()
        endpoints.append(
            SupportedEndpointInfo(
                key=key,
                display_name=display_name,
                endpoint=endpoint_path,
            )
        )

    providers = []
    for slug, provider_data in data["providers"].items():
        supported = [
            endpoint_key
            for endpoint_key, supported in provider_data["endpoints"].items()
            if supported
        ]
        providers.append(
            SupportedProviderInfo(
                slug=slug,
                display_name=provider_data["display_name"],
                supported=supported,
            )
        )

    _supported_endpoints_cache = SupportedEndpointsResponse(
        endpoints=endpoints, providers=providers
    )
    return _supported_endpoints_cache


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
