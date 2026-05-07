"""FastAPI router for ``POST /v2/agents`` (LIT-2922).

Spec: ``.claude/v2_api_contract.md`` §6.1.

Behavior:
1. Auth via ``user_api_key_auth``; ``created_by`` is scoped to the caller.
2. Names are unique per ``created_by`` — duplicate returns 409.
3. The ``litellm_api_key`` is masked in the response (``masking.mask_litellm_api_key``).
   The raw key is stored in the DB row; encryption-at-rest is handled by the
   proxy's standard config-secret pipeline (Wave 1 owns that contract).
4. No router registration here — Wave 3 wires this into ``proxy_server.py``.
"""

from fastapi import APIRouter, Depends, HTTPException

from litellm.managed_agents.db import get_agent_by_name, insert_agent
from litellm.managed_agents.id_utils import new_agent_id
from litellm.managed_agents.masking import mask_litellm_api_key
from litellm.managed_agents.types import (
    AgentConfig,
    AgentRow,
    CreateAgentRequest,
)
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# Fallback owner when an unauthenticated/master-key call is permitted by auth
# but no user_id is attached to the verification token. Mirrors
# ``team_endpoints.py:1050`` — see also ``litellm.constants.LITELLM_PROXY_ADMIN_NAME``.
_DEFAULT_CREATED_BY = "default_user"


@router.post(
    "/v2/agents",
    response_model=AgentRow,
    tags=["managed-agents-v2"],
)
async def create_agent(
    request: CreateAgentRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentRow:
    """Create a new managed agent.

    Returns ``200`` with the agent row, ``config.litellm_api_key`` masked.

    Errors:
      - ``409`` if ``(name, created_by)`` already exists for this caller.
      - ``422`` for Pydantic validation failures (FastAPI default).
      - ``500`` if the proxy DB is not connected.
    """
    # Lazy import to avoid circular dependency between proxy_server and this
    # router at module-load time. ``proxy_server`` imports a lot of things;
    # importing it here defers resolution until request time.
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    created_by = user_api_key_dict.user_id or _DEFAULT_CREATED_BY

    # Names are unique per (created_by) — pre-check before insert so we can
    # return a clean 409 instead of relying on a DB constraint that the
    # current schema does not declare.
    existing = await get_agent_by_name(
        prisma_client,
        name=request.name,
        created_by=created_by,
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Agent with name '{request.name}' already exists for this user.",
        )

    agent_id = new_agent_id()
    config_dict = request.config.model_dump()

    inserted = await insert_agent(
        prisma_client,
        agent_id=agent_id,
        name=request.name,
        config=config_dict,
        created_by=created_by,
    )

    # Mask on read. The original key was persisted by ``insert_agent`` above;
    # the masked copy is what we return to the caller.
    masked_config = {
        **config_dict,
        "litellm_api_key": mask_litellm_api_key(config_dict["litellm_api_key"]),
    }

    return AgentRow(
        id=inserted.get("id", agent_id),
        name=inserted.get("name", request.name),
        config=AgentConfig(**masked_config),
        created_by=inserted.get("created_by", created_by),
        created_at=inserted["created_at"],
        updated_at=inserted["updated_at"],
    )
