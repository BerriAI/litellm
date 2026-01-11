import json
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException

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
)
from litellm.types.utils import LlmProviders

router = APIRouter()


@router.get(
    "/public/model_hub",
    tags=["public", "model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[ModelGroupInfoProxy],
)
async def public_model_hub():
    import litellm
    from litellm.proxy.proxy_server import _get_model_group_info, llm_router

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
    Includes both hardcoded providers from JSON file and dynamically loaded JSON-configured providers.
    """

    provider_create_fields_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "proxy",
        "public_endpoints",
        "provider_create_fields.json"
    )

    with open(provider_create_fields_path, "r") as f:
        provider_create_fields = json.load(f)

        # Add providers from providers.json (OpenAI-like providers)
    try:
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        existing_provider_names = {p["provider"] for p in provider_create_fields}

        for slug in JSONProviderRegistry.list_providers():
            # Skip if already in provider_create_fields.json
            if slug in existing_provider_names:
                continue

            provider_config = JSONProviderRegistry.get(slug)
            if not provider_config:
                continue

            # Generate credential fields based on provider configuration
            credential_fields = []
            
            # Add api_base field if api_base_env is configured or base_url is provided
            if provider_config.api_base_env or provider_config.base_url:
                credential_fields.append({
                    "key": "api_base",
                    "label": "API Base",
                    "placeholder": provider_config.base_url,
                    "tooltip": f"Base URL for {slug.title()}. Default: {provider_config.base_url}" + (
                        f". You can also set the environment variable {provider_config.api_base_env}." 
                        if provider_config.api_base_env else ""
                    ),
                    "required": False,
                    "field_type": "text",
                    "options": None,
                    "default_value": provider_config.base_url if provider_config.base_url else None
                })

            # Add api_key field if api_key_env is configured
            if provider_config.api_key_env:
                credential_fields.append({
                    "key": "api_key",
                    "label": "API Key",
                    "placeholder": None,
                    "tooltip": f"API key for {slug.title()}. You can set the environment variable {provider_config.api_key_env} or provide the key directly.",
                    "required": True,
                    "field_type": "password",
                    "options": None,
                    "default_value": None
                })
            


            provider_create_fields.append({
                "provider": slug.title(),
                "provider_display_name": slug.title(),
                "litellm_provider": slug,
                "credential_fields": credential_fields,
                "default_model_placeholder": "{model}",
            })
    except Exception as e:
        # Log error but don't fail the endpoint
        import logging
        logging.warning(f"Failed to load providers from providers.json: {e}")

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
