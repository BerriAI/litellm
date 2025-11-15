"""
Agent endpoints for registering + discovering agents via LiteLLM.

Follows the A2A Spec.

1. Register an agent via POST `/v1/agents`
2. Discover agents via GET `/v1/agents`
3. Get specific agent via GET `/v1/agents/{agent_id}`
"""

from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Request

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.agents import (
    AgentConfig,
    AgentMakePublicResponse,
    AgentResponse,
    MakeAgentsPublicRequest,
    PatchAgentRequest,
)

router = APIRouter()


@router.get(
    "/v1/agents",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[AgentResponse],
)
async def get_agents(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # Used for auth
):
    """
    Example usage:
    ```
    curl -X GET "http://localhost:4000/v1/agents" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer your-key" \
    ```

    Returns: List[AgentResponse]

    """
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    try:
        returned_agents: List[AgentResponse] = []
        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
            or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
        ):
            returned_agents = global_agent_registry.get_agent_list()
        key_agents = user_api_key_dict.metadata.get("agents")
        _team_metadata = user_api_key_dict.team_metadata or {}
        team_agents = _team_metadata.get("agents")
        if key_agents is not None:
            returned_agents = global_agent_registry.get_agent_list(
                agent_names=key_agents
            )
        if team_agents is not None:
            returned_agents = global_agent_registry.get_agent_list(
                agent_names=team_agents
            )

        # add is_public field to each agent - we do it this way, to allow setting config agents as public
        for agent in returned_agents:
            if agent.litellm_params is None:
                agent.litellm_params = {}
            agent.litellm_params["is_public"] = (
                litellm.public_agent_groups is not None
                and (agent.agent_id in litellm.public_agent_groups)
            )

        return returned_agents
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(
                str(e)
            )
        )
        raise HTTPException(
            status_code=500, detail={"error": f"Internal server error: {str(e)}"}
        )


#### CRUD ENDPOINTS FOR AGENTS ####

from litellm.proxy.agent_endpoints.agent_registry import (
    global_agent_registry as AGENT_REGISTRY,
)


@router.post(
    "/v1/agents",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentResponse,
)
async def create_agent(
    request: AgentConfig,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new agent

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/agents" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "agent": {
                "agent_name": "my-custom-agent",
                "agent_card_params": {
                    "protocolVersion": "1.0",
                    "name": "Hello World Agent",
                    "description": "Just a hello world agent",
                    "url": "http://localhost:9999/",
                    "version": "1.0.0",
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "capabilities": {
                        "streaming": true
                    },
                    "skills": [
                        {
                            "id": "hello_world",
                            "name": "Returns hello world",
                            "description": "just returns hello world",
                            "tags": ["hello world"],
                            "examples": ["hi", "hello world"]
                        }
                    ]
                },
                "litellm_params": {
                    "make_public": true
                }
            }
        }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        # Get the user ID from the API key auth
        created_by = user_api_key_dict.user_id or "unknown"

        # check for naming conflicts
        existing_agent = AGENT_REGISTRY.get_agent_by_name(
            agent_name=request.get("agent_name")  # type: ignore
        )
        if existing_agent is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Agent with name {request.get('agent_name')} already exists",
            )

        result = await AGENT_REGISTRY.add_agent_to_db(
            agent=request, prisma_client=prisma_client, created_by=created_by
        )

        agent_name = result.agent_name
        agent_id = result.agent_id

        # Also register in memory
        try:
            AGENT_REGISTRY.register_agent(agent_config=result)
            verbose_proxy_logger.info(
                f"Successfully registered agent '{agent_name}' (ID: {agent_id}) in memory"
            )
        except Exception as reg_error:
            verbose_proxy_logger.warning(
                f"Failed to register agent '{agent_name}' (ID: {agent_id}) in memory: {reg_error}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error adding agent to db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/agents/{agent_id}",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentResponse,
)
async def get_agent_by_id(agent_id: str):
    """
    Get a specific agent by ID

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        agent = AGENT_REGISTRY.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            agent = await prisma_client.db.litellm_agentstable.find_unique(
                where={"agent_id": agent_id}
            )
            if agent is not None:
                agent = AgentResponse(**agent.model_dump())  # type: ignore

        if agent is None:
            raise HTTPException(
                status_code=404, detail=f"Agent with ID {agent_id} not found"
            )

        return agent
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting agent from db: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/v1/agents/{agent_id}",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentResponse,
)
async def update_agent(
    agent_id: str,
    request: AgentConfig,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing agent

    Example Request:
    ```bash
    curl -X PUT "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "agent": {
                "agent_name": "updated-agent",
                "agent_card_params": {
                    "protocolVersion": "1.0",
                    "name": "Updated Agent",
                    "description": "Updated description",
                    "url": "http://localhost:9999/",
                    "version": "1.1.0",
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "capabilities": {
                        "streaming": true
                    },
                    "skills": []
                },
                "litellm_params": {
                    "make_public": false
                }
            }
        }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Check if agent exists
        existing_agent = await prisma_client.db.litellm_agentstable.find_unique(
            where={"agent_id": agent_id}
        )
        if existing_agent is not None:
            existing_agent = dict(existing_agent)

        if existing_agent is None:
            raise HTTPException(
                status_code=404, detail=f"Agent with ID {agent_id} not found"
            )

        # Get the user ID from the API key auth
        updated_by = user_api_key_dict.user_id or "unknown"

        result = await AGENT_REGISTRY.update_agent_in_db(
            agent_id=agent_id,
            agent=request,
            prisma_client=prisma_client,
            updated_by=updated_by,
        )

        # deregister in memory
        AGENT_REGISTRY.deregister_agent(agent_name=existing_agent.get("agent_name"))  # type: ignore
        # register in memory
        AGENT_REGISTRY.register_agent(agent_config=result)

        verbose_proxy_logger.info(
            f"Successfully updated agent '{existing_agent.get('agent_name')}' (ID: {agent_id}) in memory"
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/v1/agents/{agent_id}",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentResponse,
)
async def patch_agent(
    agent_id: str,
    request: PatchAgentRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing agent

    Example Request:
    ```bash
    curl -X PUT "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "agent": {
                "agent_name": "updated-agent",
                "agent_card_params": {
                    "protocolVersion": "1.0",
                    "name": "Updated Agent",
                    "description": "Updated description",
                    "url": "http://localhost:9999/",
                    "version": "1.1.0",
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "capabilities": {
                        "streaming": true
                    },
                    "skills": []
                },
                "litellm_params": {
                    "make_public": false
                }
            }
        }'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Check if agent exists
        existing_agent = await prisma_client.db.litellm_agentstable.find_unique(
            where={"agent_id": agent_id}
        )
        if existing_agent is not None:
            existing_agent = dict(existing_agent)

        if existing_agent is None:
            raise HTTPException(
                status_code=404, detail=f"Agent with ID {agent_id} not found"
            )

        # Get the user ID from the API key auth
        updated_by = user_api_key_dict.user_id or "unknown"

        result = await AGENT_REGISTRY.patch_agent_in_db(
            agent_id=agent_id,
            agent=request,
            prisma_client=prisma_client,
            updated_by=updated_by,
        )

        # deregister in memory
        AGENT_REGISTRY.deregister_agent(agent_name=existing_agent.get("agent_name"))  # type: ignore
        # register in memory
        AGENT_REGISTRY.register_agent(agent_config=result)

        verbose_proxy_logger.info(
            f"Successfully updated agent '{existing_agent.get('agent_name')}' (ID: {agent_id}) in memory"
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/v1/agents/{agent_id}",
    tags=["Agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_agent(agent_id: str):
    """
    Delete an agent

    Example Request:
    ```bash
    curl -X DELETE "http://localhost:4000/agents/123e4567-e89b-12d3-a456-426614174000" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "message": "Agent 123e4567-e89b-12d3-a456-426614174000 deleted successfully"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Prisma client not initialized")

    try:
        # Check if agent exists
        existing_agent = await prisma_client.db.litellm_agentstable.find_unique(
            where={"agent_id": agent_id}
        )
        if existing_agent is not None:
            existing_agent = dict[Any, Any](existing_agent)

        if existing_agent is None:
            raise HTTPException(
                status_code=404, detail=f"Agent with ID {agent_id} not found in DB."
            )

        await AGENT_REGISTRY.delete_agent_from_db(
            agent_id=agent_id, prisma_client=prisma_client
        )

        AGENT_REGISTRY.deregister_agent(agent_name=existing_agent.get("agent_name"))  # type: ignore

        return {"message": f"Agent {agent_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/v1/agents/{agent_id}/make_public",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentMakePublicResponse,
)
async def make_agent_public(
    agent_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Make an agent publicly discoverable

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/v1/agents/123e4567-e89b-12d3-a456-426614174000/make_public" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json"
    ```

    Example Response:
    ```json
    {
        "agent_id": "123e4567-e89b-12d3-a456-426614174000",
        "agent_name": "my-custom-agent",
        "litellm_params": {
            "make_public": true
        },
        "agent_card_params": {...},
        "created_at": "2025-11-15T10:30:00Z",
        "updated_at": "2025-11-15T10:35:00Z",
        "created_by": "user123",
        "updated_by": "user123"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Update the public model groups
        import litellm
        from litellm.proxy.agent_endpoints.agent_registry import (
            global_agent_registry as AGENT_REGISTRY,
        )
        from litellm.proxy.proxy_server import proxy_config

        # Check if user has admin permissions
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only proxy admins can update public model groups. Your role={}".format(
                        user_api_key_dict.user_role
                    )
                },
            )

        agent = AGENT_REGISTRY.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            # check if agent exists in DB
            agent = await prisma_client.db.litellm_agentstable.find_unique(
                where={"agent_id": agent_id}
            )
            if agent is not None:
                agent = AgentResponse(**agent.model_dump())  # type: ignore

            if agent is None:
                raise HTTPException(
                    status_code=404, detail=f"Agent with ID {agent_id} not found"
                )

        if litellm.public_agent_groups is None:
            litellm.public_agent_groups = []
        # handle duplicates
        if agent.agent_id in litellm.public_agent_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Agent with name {agent.agent_name} already in public agent groups",
            )
        litellm.public_agent_groups.append(agent.agent_id)

        # Load existing config
        config = await proxy_config.get_config()

        # Update config with new settings
        if "litellm_settings" not in config or config["litellm_settings"] is None:
            config["litellm_settings"] = {}

        config["litellm_settings"]["public_agent_groups"] = litellm.public_agent_groups

        # Save the updated config
        await proxy_config.save_config(new_config=config)

        verbose_proxy_logger.debug(
            f"Updated public agent groups to: {litellm.public_agent_groups} by user: {user_api_key_dict.user_id}"
        )

        return {
            "message": "Successfully updated public agent groups",
            "public_agent_groups": litellm.public_agent_groups,
            "updated_by": user_api_key_dict.user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error making agent public: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/v1/agents/make_public",
    tags=["[beta] Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentMakePublicResponse,
)
async def make_agents_public(
    request: MakeAgentsPublicRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Make multiple agents publicly discoverable

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/v1/agents/make_public" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "agent_ids": ["123e4567-e89b-12d3-a456-426614174000", "123e4567-e89b-12d3-a456-426614174001"]
        }'
    ```

    Example Response:
    ```json
    {
        "agent_id": "123e4567-e89b-12d3-a456-426614174000",
        "agent_name": "my-custom-agent",
        "litellm_params": {
            "make_public": true
        },
        "agent_card_params": {...},
        "created_at": "2025-11-15T10:30:00Z",
        "updated_at": "2025-11-15T10:35:00Z",
        "created_by": "user123",
        "updated_by": "user123"
    }
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Update the public model groups
        import litellm
        from litellm.proxy.agent_endpoints.agent_registry import (
            global_agent_registry as AGENT_REGISTRY,
        )
        from litellm.proxy.proxy_server import proxy_config

        # Load existing config
        config = await proxy_config.get_config()
        # Check if user has admin permissions
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only proxy admins can update public model groups. Your role={}".format(
                        user_api_key_dict.user_role
                    )
                },
            )

        if litellm.public_agent_groups is None:
            litellm.public_agent_groups = []

        for agent_id in request.agent_ids:
            agent = AGENT_REGISTRY.get_agent_by_id(agent_id=agent_id)
            if agent is None:
                # check if agent exists in DB
                agent = await prisma_client.db.litellm_agentstable.find_unique(
                    where={"agent_id": agent_id}
                )
                if agent is not None:
                    agent = AgentResponse(**agent.model_dump())  # type: ignore

                if agent is None:
                    raise HTTPException(
                        status_code=404, detail=f"Agent with ID {agent_id} not found"
                    )

        litellm.public_agent_groups = request.agent_ids

        # Update config with new settings
        if "litellm_settings" not in config or config["litellm_settings"] is None:
            config["litellm_settings"] = {}

        config["litellm_settings"]["public_agent_groups"] = litellm.public_agent_groups

        # Save the updated config
        await proxy_config.save_config(new_config=config)

        verbose_proxy_logger.debug(
            f"Updated public agent groups to: {litellm.public_agent_groups} by user: {user_api_key_dict.user_id}"
        )

        return {
            "message": "Successfully updated public agent groups",
            "public_agent_groups": litellm.public_agent_groups,
            "updated_by": user_api_key_dict.user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error making agent public: {e}")
        raise HTTPException(status_code=500, detail=str(e))
