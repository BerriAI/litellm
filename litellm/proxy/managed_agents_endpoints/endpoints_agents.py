import asyncio
import json
from typing import List

from fastapi import Depends, HTTPException

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.managed_agents_endpoints.endpoints import (
    _assert_owner_or_admin,
    _is_admin,
    _template_visible_to,
    router,
)
from litellm.proxy.managed_agents_endpoints.git_validation import (
    decrypt_git_token,
    validate_repo_branch,
)
from litellm.proxy.managed_agents_endpoints.types import AgentCreate, AgentOut
from litellm.proxy.utils import jsonify_object


def _agent_row_to_out(row) -> AgentOut:
    return AgentOut(
        id=row.agent_id,
        name=row.agent_name,
        model=row.model,
        template_id=row.template_id,
        branch=row.branch,
        created_at=getattr(row, "created_at", None),
    )


@router.post("/agents", response_model=AgentOut)
async def create_agent(
    body: AgentCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentOut:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    template = (
        await prisma_client.db.litellm_managedagentsandboxtemplatetable.find_unique(
            where={"template_id": body.template_id}
        )
    )
    if template is None or not _template_visible_to(template, user_api_key_dict):
        raise HTTPException(
            status_code=404, detail=f"template '{body.template_id}' not found"
        )

    branch = body.branch or template.default_branch

    git_token = await decrypt_git_token(prisma_client, template.git_credential_id)
    await asyncio.to_thread(validate_repo_branch, template.repo_url, branch, git_token)

    create_data = jsonify_object(
        {
            "agent_name": body.name,
            "model": body.model,
            "prompt": body.prompt,
            "tools": json.dumps(body.tools),
            "branch": branch,
            "metadata": {
                "litellm_api_key": body.litellm_api_key,
                "litellm_api_base": body.litellm_api_base,
            },
            "created_by": user_api_key_dict.user_id,
            "updated_by": user_api_key_dict.user_id,
        }
    )
    create_data["template"] = {"connect": {"template_id": body.template_id}}

    row = await prisma_client.db.litellm_managedagenttable.create(data=create_data)

    return _agent_row_to_out(row)


@router.get("/agents", response_model=List[AgentOut])
async def list_agents(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[AgentOut]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    where: dict = {}
    if not _is_admin(user_api_key_dict) and user_api_key_dict.user_id is not None:
        where["created_by"] = user_api_key_dict.user_id

    rows = await prisma_client.db.litellm_managedagenttable.find_many(
        where=where, order={"created_at": "desc"}
    )
    return [_agent_row_to_out(row) for row in rows]


@router.get("/agents/{agent_id}", response_model=AgentOut)
async def get_agent(
    agent_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentOut:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagenttable.find_unique(
        where={"agent_id": agent_id}
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

    _assert_owner_or_admin(user_api_key_dict, row.created_by, "agent", agent_id)

    return _agent_row_to_out(row)
