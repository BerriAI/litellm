import asyncio
import json
from typing import Any, Dict, List

from fastapi import Depends, HTTPException

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
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
from litellm.proxy.managed_agents_endpoints.types import (
    AgentCreate,
    AgentOut,
    AgentUpdate,
)
from litellm.proxy.utils import jsonify_object


def _coerce_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_string_list(raw: Any) -> List[str]:
    """Best-effort: accept a list, drop non-strings."""
    if not isinstance(raw, list):
        return []
    return [v for v in raw if isinstance(v, str)]


def _agent_row_to_out(row) -> AgentOut:
    metadata = _coerce_metadata(getattr(row, "metadata", None))
    pfp_url = metadata.get("pfp_url")
    return AgentOut(
        id=row.agent_id,
        name=row.agent_name,
        model=row.model,
        prompt=getattr(row, "prompt", None),
        template_id=row.template_id,
        branch=row.branch,
        pfp_url=pfp_url if isinstance(pfp_url, str) else None,
        mcp_servers=_coerce_string_list(metadata.get("mcp_servers")),
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

    encrypted_key = (
        encrypt_value_helper(body.litellm_api_key) if body.litellm_api_key else None
    )

    metadata: Dict[str, Any] = {
        "litellm_api_key_encrypted": encrypted_key,
        "litellm_api_base": body.litellm_api_base,
    }
    if body.pfp_url is not None:
        metadata["pfp_url"] = body.pfp_url
    if body.mcp_servers:
        metadata["mcp_servers"] = list(body.mcp_servers)

    create_data = jsonify_object(
        {
            "agent_name": body.name,
            "model": body.model,
            "prompt": body.prompt,
            "tools": json.dumps(body.tools),
            "branch": branch,
            "metadata": metadata,
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
    if not _is_admin(user_api_key_dict):
        # Non-admin callers see only their own rows. If the API key has no
        # user_id, treat it as "no rows" rather than exposing every agent.
        if user_api_key_dict.user_id is None:
            return []
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


@router.patch("/agents/{agent_id}", response_model=AgentOut)
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
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

    update_data: Dict[str, Any] = {"updated_by": user_api_key_dict.user_id}
    fields = body.model_dump(exclude_unset=True)

    if "name" in fields:
        update_data["agent_name"] = fields["name"] or None

    if "pfp_url" in fields or "mcp_servers" in fields:
        # Pull existing metadata once and apply both edits to the same dict so
        # we don't issue two updates that race each other.
        metadata = _coerce_metadata(getattr(row, "metadata", None))
        if "pfp_url" in fields:
            new_pfp = fields["pfp_url"]
            if new_pfp:
                metadata["pfp_url"] = new_pfp
            else:
                metadata.pop("pfp_url", None)
        if "mcp_servers" in fields:
            new_list = _coerce_string_list(fields["mcp_servers"])
            if new_list:
                metadata["mcp_servers"] = new_list
            else:
                metadata.pop("mcp_servers", None)
        update_data["metadata"] = metadata

    if len(update_data) == 1:  # only updated_by
        return _agent_row_to_out(row)

    updated = await prisma_client.db.litellm_managedagenttable.update(
        where={"agent_id": agent_id},
        data=jsonify_object(update_data),
    )
    return _agent_row_to_out(updated)
