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
async def public_model_hub(provider: str = None):
    """
    Get public model hub information.
    
    If provider="databricks" and Databricks credentials are configured,
    this will return actual Databricks serving endpoints instead of
    models from the cost map.
    
    Args:
        provider: Optional provider filter (e.g., "databricks")
    
    Returns:
        List of model group information
    """
    import litellm
    from litellm.proxy.proxy_server import _get_model_group_info, llm_router

    if llm_router is None:
        raise HTTPException(
            status_code=400, detail=CommonProxyErrors.no_llm_router.value
        )

    # If requesting Databricks models and credentials are available, fetch from Databricks API
    if provider and provider.lower() == "databricks":
        try:
            databricks_endpoints = await get_databricks_serving_endpoints()
            
            # Convert Databricks endpoints to ModelGroupInfoProxy format
            model_groups: List[ModelGroupInfoProxy] = []
            for endpoint_data in databricks_endpoints.get("endpoints", []):
                endpoint_name = endpoint_data.get("name")
                if endpoint_name:
                    # Create a model group entry for each Databricks endpoint
                    model_group = ModelGroupInfoProxy(
                        model_group=f"databricks/{endpoint_name}",
                        providers=["databricks"],
                        max_input_tokens=None,
                        max_output_tokens=None,
                        input_cost_per_token=None,
                        output_cost_per_token=None,
                        mode="chat",
                        supports_parallel_function_calling=False,
                        supports_vision=False,
                        supports_function_calling=True,
                        supported_openai_params=[],
                        is_public_model_group=True,
                    )
                    model_groups.append(model_group)
            
            return model_groups
        except HTTPException as e:
            # If Databricks API call fails (e.g., no credentials), fall back to default behavior
            if e.status_code != 400:  # Only fall back on missing credentials, not other errors
                raise

    # Default behavior: use configured public model groups
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


@router.get(
    "/public/databricks/serving_endpoints",
    tags=["public", "providers"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_databricks_serving_endpoints(
    api_key: str = None,
    api_base: str = None,
    client_id: str = None,
    client_secret: str = None,
):
    """
    Fetch available serving endpoints from Databricks.
    
    This endpoint queries the Databricks API to retrieve the list of deployed
    serving endpoints, allowing the model hub to display actual Databricks models
    instead of relying on the LiteLLM model cost map.
    
    Authentication can be provided via:
    1. Query parameters (api_key, api_base) or (client_id, client_secret, api_base)
    2. Environment variables (DATABRICKS_API_KEY, DATABRICKS_API_BASE) or
       (DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, DATABRICKS_API_BASE)
    
    Returns:
        List of serving endpoint information including name and model details
    """
    import httpx
    from litellm.llms.databricks.common_utils import DatabricksBase
    
    try:
        # Check environment variables if not provided as parameters
        api_key = api_key or os.getenv("DATABRICKS_API_KEY")
        api_base = api_base or os.getenv("DATABRICKS_API_BASE")
        client_id = client_id or os.getenv("DATABRICKS_CLIENT_ID")
        client_secret = client_secret or os.getenv("DATABRICKS_CLIENT_SECRET")
        
        databricks_base = DatabricksBase()
        
        # Validate we have necessary credentials
        if not api_base and not (client_id and client_secret):
            raise HTTPException(
                status_code=400,
                detail="Missing Databricks credentials. Provide either api_key and api_base, or client_id, client_secret, and api_base",
            )
        
        # Get workspace URL from api_base
        if api_base:
            workspace_url = api_base.rstrip("/")
            if "/serving-endpoints" in workspace_url:
                workspace_url = workspace_url.replace("/serving-endpoints", "")
        else:
            # Try to get from Databricks SDK
            try:
                from databricks.sdk import WorkspaceClient
                databricks_client = WorkspaceClient()
                workspace_url = databricks_client.config.host
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to determine Databricks workspace URL",
                )
        
        # Prepare authentication headers
        headers = {"Content-Type": "application/json"}
        
        # Use OAuth M2M if credentials are available
        if client_id and client_secret:
            access_token = databricks_base._get_oauth_m2m_token(
                api_base=workspace_url,
                client_id=client_id,
                client_secret=client_secret,
            )
            headers["Authorization"] = f"Bearer {access_token}"
        elif api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            # Try Databricks SDK authentication
            try:
                from databricks.sdk import WorkspaceClient
                databricks_client = WorkspaceClient()
                auth_headers = databricks_client.config.authenticate()
                headers.update(auth_headers)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to authenticate with Databricks",
                )
        
        # Call Databricks API to list serving endpoints
        list_url = f"{workspace_url}/api/2.0/serving-endpoints"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(list_url, headers=headers, timeout=30.0)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch Databricks endpoints: {response.text}",
                )
            
            data = response.json()
            endpoints = data.get("endpoints", [])
            
            # Format the response for the model hub
            formatted_endpoints = []
            for endpoint in endpoints:
                endpoint_info = {
                    "name": endpoint.get("name"),
                    "creator": endpoint.get("creator"),
                    "creation_timestamp": endpoint.get("creation_timestamp"),
                    "last_updated_timestamp": endpoint.get("last_updated_timestamp"),
                    "state": endpoint.get("state", {}).get("ready"),
                    "config": {
                        "served_models": endpoint.get("config", {}).get("served_models", []),
                        "served_entities": endpoint.get("config", {}).get("served_entities", []),
                    },
                    "endpoint_url": f"{workspace_url}/serving-endpoints/{endpoint.get('name')}",
                }
                formatted_endpoints.append(endpoint_info)
            
            return {
                "endpoints": formatted_endpoints,
                "workspace_url": workspace_url,
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )
