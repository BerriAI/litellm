"""
Agent CRUD endpoints — POST/GET/PATCH/DELETE /v2/agents{,/<id>}.

Note: /v1/agents is the existing A2A registry (litellm/proxy/agent_endpoints/);
this module mounts under /v2/ to avoid collision.

Agents are pure definitions: model, system prompt, default repos, tools.
Creating an agent does NOT spin up a VM — that happens at session create.

DELETE cascades to sessions: every non-terminal session under the agent
is torn down (run cancellation + provider.terminate) before the agent
row is removed.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import ORJSONResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_session_endpoints.constants import SESSION_TERMINAL_STATUSES
from litellm.proxy.agent_session_endpoints.ids import new_agent_id
from litellm.proxy.agent_session_endpoints.ownership import (
    assert_caller_can_mutate,
    assert_caller_owns_agent,
    caller_api_key_hash,
    owner_filter_for_caller,
)
from litellm.proxy.agent_session_endpoints.pagination import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    build_page_response,
    cursor_where_clause,
    normalize_limit,
)
from litellm.proxy.agent_session_endpoints.schemas import (
    AgentCreate,
    AgentResponse,
    AgentUpdate,
)
from litellm.proxy.agent_session_endpoints.serialization import (
    agent_row_to_response,
)
from litellm.proxy.agent_session_endpoints.session_endpoints import (
    _terminate_session_internal,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _agent_create_payload(body: AgentCreate, user_api_key_dict: UserAPIKeyAuth) -> dict:
    """Build the Prisma create payload for an agent.

    Pydantic models are converted to plain dicts so Prisma can serialize them
    into Json columns. ``model_dump(exclude_none=True)`` keeps the row tight —
    null Json columns stay null instead of becoming the string "null".
    """
    return {
        "id": new_agent_id(),
        "name": body.name,
        "user_api_key_hash": caller_api_key_hash(user_api_key_dict),
        "team_id": user_api_key_dict.team_id,
        "model": body.model,
        "system_prompt": body.system_prompt,
        "default_repos": (
            [r.model_dump(exclude_none=True) for r in body.default_repos]
            if body.default_repos
            else None
        ),
        "default_env_vars": body.default_env_vars,
        "tools_config": body.tools_config,
        "metadata": body.metadata,
        "updated_at": _now(),
    }


def _agent_update_payload(body: AgentUpdate) -> dict:
    """Build the Prisma update payload — only fields the caller actually
    set. Pydantic ``exclude_unset=True`` is the canonical way to do this."""
    raw = body.model_dump(exclude_unset=True)
    if "default_repos" in raw and raw["default_repos"] is not None:
        # Re-dump nested RepoSpec models as plain dicts.
        raw["default_repos"] = [
            r.model_dump(exclude_none=True) if hasattr(r, "model_dump") else r
            for r in body.default_repos or []
        ]
    raw["updated_at"] = _now()
    return raw


async def _get_prisma_client_or_503():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return prisma_client


@router.post(
    "/v2/agents",
    response_class=ORJSONResponse,
    response_model=AgentResponse,
    tags=["agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_agent(
    body: AgentCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()
    payload = _agent_create_payload(body, user_api_key_dict)
    row = await prisma_client.db.litellm_agent.create(data=payload)
    verbose_proxy_logger.info(
        "agent.create id=%s name=%s model=%s", row.id, row.name, row.model
    )
    return agent_row_to_response(row)


@router.get(
    "/v2/agents/{agent_id}",
    response_class=ORJSONResponse,
    response_model=AgentResponse,
    tags=["agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_agent(
    agent_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    prisma_client = await _get_prisma_client_or_503()
    row = await prisma_client.db.litellm_agent.find_unique(where={"id": agent_id})
    assert_caller_owns_agent(user_api_key_dict, row)
    return agent_row_to_response(row)


@router.patch(
    "/v2/agents/{agent_id}",
    response_class=ORJSONResponse,
    response_model=AgentResponse,
    tags=["agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()
    existing = await prisma_client.db.litellm_agent.find_unique(where={"id": agent_id})
    assert_caller_owns_agent(user_api_key_dict, existing)
    payload = _agent_update_payload(body)
    if not any(k for k in payload if k != "updated_at"):
        # No-op patch — return existing row unchanged (still 200, idempotent).
        return agent_row_to_response(existing)
    updated = await prisma_client.db.litellm_agent.update(
        where={"id": agent_id}, data=payload
    )
    return agent_row_to_response(updated)


@router.get(
    "/v2/agents",
    response_class=ORJSONResponse,
    tags=["agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_agents(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
):
    """List agents owned by the caller (or all, for proxy admins).

    Cursor-paginated. Pass ``?cursor=<next_cursor>`` from the previous
    response to fetch the next page.
    """
    prisma_client = await _get_prisma_client_or_503()
    page_limit = normalize_limit(limit)
    rows = await prisma_client.db.litellm_agent.find_many(
        where=cursor_where_clause(owner_filter_for_caller(user_api_key_dict), cursor),
        order=[{"created_at": "desc"}, {"id": "desc"}],
        take=page_limit + 1,
    )
    return build_page_response(
        rows,
        page_limit,
        lambda r: agent_row_to_response(r).model_dump(),
    )


@router.delete(
    "/v2/agents/{agent_id}",
    response_class=ORJSONResponse,
    tags=["agents"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_agent(
    agent_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()
    existing = await prisma_client.db.litellm_agent.find_unique(where={"id": agent_id})
    assert_caller_owns_agent(user_api_key_dict, existing)

    # Cascade: terminate every non-terminal session under this agent
    # first. We gather them in parallel because each call hits the VM
    # provider. ``status not in SESSION_TERMINAL_STATUSES`` matches the
    # rest of the module — ``terminated_at is None`` was a near-equivalent
    # but broke if a session had its status flipped without
    # ``terminated_at`` being set (Greptile P3).
    sessions = await prisma_client.db.litellm_agentsession.find_many(
        where={"agent_id": agent_id}
    )
    active = [s for s in sessions if s.status not in SESSION_TERMINAL_STATUSES]
    if active:
        await asyncio.gather(
            *[
                _terminate_session_internal(s.id, reason="agent_deleted")
                for s in active
            ],
            return_exceptions=True,
        )

    # Now drop the agent row. Prisma cascade handles FK chains
    # (sessions -> runs -> events) for any session still on disk.
    await prisma_client.db.litellm_agent.delete(where={"id": agent_id})
    return {"id": agent_id, "deleted": True}
