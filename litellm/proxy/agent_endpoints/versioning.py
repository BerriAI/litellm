"""
Agent version history helpers (S3-03 + S3-04).

LiteLLM_AgentVersionTable is append-only. snapshot_agent_version() writes
one row capturing the pre-mutation state of an agent; the caller invokes
it before persisting changes. rollback_agent() copies an older snapshot
back onto the live row and also appends a *new* version marking that the
rollback happened (so the history stays linear).
"""

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger


async def _next_version_number(prisma_client, agent_id: str) -> int:
    """Return next version_number for an agent (1 if none yet)."""
    rows = await prisma_client.db.litellm_agentversiontable.find_many(
        where={"agent_id": agent_id},
        order={"version_number": "desc"},
        take=1,
    )
    return (rows[0].version_number + 1) if rows else 1


async def snapshot_agent_version(
    *,
    prisma_client,
    agent_id: str,
    agent_card_params: Dict[str, Any],
    litellm_params: Optional[Dict[str, Any]] = None,
    static_headers: Optional[Dict[str, Any]] = None,
    created_by: Optional[str] = None,
    is_rollback: bool = False,
    rolled_back_from: Optional[int] = None,
) -> int:
    """Persist one version row and return its version_number.

    Best-effort: failures are logged but never propagated — versioning is
    nice-to-have, the primary mutation must still succeed.
    """
    try:
        version_number = await _next_version_number(prisma_client, agent_id)
        await prisma_client.db.litellm_agentversiontable.create(
            data={
                "agent_id": agent_id,
                "version_number": version_number,
                "agent_card_params": agent_card_params,
                "litellm_params": litellm_params,
                "static_headers": static_headers,
                "created_by": created_by,
                "is_rollback": is_rollback,
                "rolled_back_from": rolled_back_from,
            }
        )
        return version_number
    except Exception as e:
        verbose_proxy_logger.warning(
            "snapshot_agent_version(%s) failed: %s", agent_id, e
        )
        return -1


async def snapshot_existing_agent(
    *, prisma_client, existing_row: Dict[str, Any], created_by: Optional[str]
) -> int:
    """Snapshot the row's current state before a PUT/PATCH overwrites it."""
    return await snapshot_agent_version(
        prisma_client=prisma_client,
        agent_id=existing_row.get("agent_id"),
        agent_card_params=existing_row.get("agent_card_params") or {},
        litellm_params=existing_row.get("litellm_params"),
        static_headers=existing_row.get("static_headers"),
        created_by=created_by,
    )


async def list_agent_versions(
    *, prisma_client, agent_id: str, limit: int = 20, cursor: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return version history sorted newest-first."""
    find_args: Dict[str, Any] = {
        "where": {"agent_id": agent_id},
        "order": {"version_number": "desc"},
        "take": limit,
    }
    if cursor is not None:
        find_args["cursor"] = {"version_id": cursor}
        find_args["skip"] = 1
    rows = await prisma_client.db.litellm_agentversiontable.find_many(**find_args)
    return [_row_to_dict(r) for r in rows]


async def rollback_agent_to_version(
    *,
    prisma_client,
    agent_id: str,
    target_version_number: int,
    created_by: Optional[str],
):
    """Copy an older version's content back onto the live agent row.

    Also appends a NEW version row marking the rollback (so history stays
    append-only and you can see when/who rolled back).
    """
    target = await prisma_client.db.litellm_agentversiontable.find_unique(
        where={
            "agent_id_version_number": {
                "agent_id": agent_id,
                "version_number": target_version_number,
            }
        }
    )
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent version {target_version_number} not found for {agent_id}.",
        )

    live = await prisma_client.db.litellm_agentstable.find_unique(
        where={"agent_id": agent_id}
    )
    if live is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found.")

    # Snapshot the *current* live row first so the rollback action itself
    # is recoverable.
    await snapshot_existing_agent(
        prisma_client=prisma_client,
        existing_row=dict(live),
        created_by=created_by,
    )

    # Apply the target version's content onto the live row.
    updated = await prisma_client.db.litellm_agentstable.update(
        where={"agent_id": agent_id},
        data={
            "agent_card_params": target.agent_card_params,
            "litellm_params": target.litellm_params,
            "static_headers": target.static_headers,
            "updated_by": created_by or "unknown",
        },
    )

    # Append the rollback marker.
    await snapshot_agent_version(
        prisma_client=prisma_client,
        agent_id=agent_id,
        agent_card_params=target.agent_card_params,
        litellm_params=target.litellm_params,
        static_headers=target.static_headers,
        created_by=created_by,
        is_rollback=True,
        rolled_back_from=target_version_number,
    )
    return updated


def _row_to_dict(row) -> Dict[str, Any]:
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if isinstance(row, dict):
        return row
    return vars(row)
