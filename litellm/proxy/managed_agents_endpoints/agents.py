"""FastAPI router for ``POST /v2/agents`` (LIT-2922) and ``GET /v2/agents``.

Spec: ``.claude/v2_api_contract.md`` §6.1.

Behavior:
1. Auth via ``user_api_key_auth``; ``created_by`` is scoped to the caller.
2. Names are unique per ``created_by`` — duplicate returns 409. The DB also
   carries a ``UNIQUE(name, created_by)`` constraint as a backstop against
   the read-then-write race.
3. The ``litellm_api_key`` is encrypted at rest via the proxy's standard
   secret-encryption helper (``encrypt_value_helper``) before insert and
   decrypted only for masking on read. The response always carries the
   masked form (``masking.mask_litellm_api_key``).
4. Router registration lives in ``proxy_server.py`` via the composite
   ``managed_agents_router``.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm.managed_agents.db import (
    get_agent_by_name,
    insert_agent,
    list_agents,
)
from litellm.managed_agents.id_utils import new_agent_id
from litellm.managed_agents.masking import mask_litellm_api_key
from litellm.managed_agents.types import (
    AgentConfig,
    AgentList,
    AgentRow,
    CreateAgentRequest,
)
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)

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
    plaintext_api_key = config_dict["litellm_api_key"]

    # Encrypt secrets before persistence — same pipeline used by
    # ``model_management_endpoints`` and ``mcp_management_endpoints``.
    # The masked form returned to the caller is computed from the
    # plaintext, not the ciphertext, so the response prefix still
    # reflects the real key.
    config_to_persist = {
        **config_dict,
        "litellm_api_key": encrypt_value_helper(plaintext_api_key),
    }

    try:
        inserted = await insert_agent(
            prisma_client,
            agent_id=agent_id,
            name=request.name,
            config=config_to_persist,
            created_by=created_by,
        )
    except Exception as e:
        # Backstop for the DB-level UNIQUE(name, created_by) constraint —
        # if a concurrent insert beat us between the pre-check and here,
        # surface a 409 instead of leaking the underlying integrity error.
        msg = str(e)
        if (
            "Unique constraint" in msg
            or "duplicate key" in msg
            or "UNIQUE constraint failed" in msg
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Agent with name '{request.name}' already exists for this user.",
            ) from e
        raise

    masked_config = {
        **config_dict,
        "litellm_api_key": mask_litellm_api_key(plaintext_api_key),
    }

    return AgentRow(
        id=inserted.get("id", agent_id),
        name=inserted.get("name", request.name),
        config=AgentConfig(**masked_config),
        created_by=inserted.get("created_by", created_by),
        created_at=inserted["created_at"],
        updated_at=inserted["updated_at"],
    )


def _row_to_agent_response(row: Dict[str, Any]) -> AgentRow:
    """Map a DB row to a public AgentRow with masked ``litellm_api_key``.

    Handles both raw-dict configs (Prisma JSON column) and `prisma.Json`-wrapped
    configs (mock-style with a ``.data`` attribute). The stored
    ``litellm_api_key`` is encrypted; we decrypt it before masking so the
    response prefix still reflects the original key. If decryption fails
    (e.g. a legacy plaintext row from a prior schema version, or a salt-key
    mismatch) we fall back to masking the raw stored value.
    """
    raw_config: Any = row.get("config") or {}
    # prisma.Json wrapper used by some test fakes exposes the dict via ``.data``;
    # use ``getattr`` so static type checkers don't flag attribute access on the
    # ``dict`` branch of the union.
    wrapped = getattr(raw_config, "data", None)
    if isinstance(wrapped, dict):
        config_dict: Dict[str, Any] = dict(wrapped)
    elif isinstance(raw_config, dict):
        config_dict = dict(raw_config)
    else:
        config_dict = {}

    stored_key = config_dict.get("litellm_api_key", "") or ""
    decrypted = decrypt_value_helper(
        value=stored_key,
        key="litellm_api_key",
        exception_type="debug",
        return_original_value=True,
    )
    plaintext_for_mask = decrypted if isinstance(decrypted, str) else stored_key

    masked_config = {
        **config_dict,
        "litellm_api_key": mask_litellm_api_key(plaintext_for_mask),
    }

    return AgentRow(
        id=row["id"],
        name=row["name"],
        config=AgentConfig(**masked_config),
        created_by=row.get("created_by"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get(
    "/v2/agents",
    response_model=AgentList,
    tags=["managed-agents-v2"],
)
async def list_agents_endpoint(
    limit: int = 50,
    cursor: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentList:
    """List agents created by the caller, newest first.

    Pagination: simple offset-based. ``cursor`` is the next offset (as a
    decimal string). When omitted, starts at offset 0.

    Returns ``AgentList`` with ``litellm_api_key`` masked on every row.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    created_by = user_api_key_dict.user_id or _DEFAULT_CREATED_BY

    # Parse cursor → offset. Invalid cursors are a 422 (callers should treat
    # the cursor as opaque and round-trip it).
    if cursor is None or cursor == "":
        skip = 0
    else:
        try:
            skip = int(cursor)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid cursor: {cursor!r}")

    # Fetch one extra row to detect ``has_more`` without a separate count call.
    rows = await list_agents(
        prisma_client,
        created_by=created_by,
        limit=limit + 1,
        skip=skip,
    )
    has_more = len(rows) > limit
    page_rows = rows[:limit]

    next_cursor = str(skip + limit) if has_more else None

    return AgentList(
        data=[_row_to_agent_response(r) for r in page_rows],
        next_cursor=next_cursor,
        has_more=has_more,
    )
