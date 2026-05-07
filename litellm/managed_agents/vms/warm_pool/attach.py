"""Race-safe warm-pool attach (LIT-2890).

``attach_warm_vm()`` is the hot path called from ``POST /v2/sessions``:

  1. Picks the oldest ``state='warm'`` row for the team (non-locking
     read used as a candidate filter).
  2. Performs an atomic flip ``warm -> hydrating`` using ``update_many``
     scoped to the candidate id + ``state='warm'``. Postgres's MVCC + the
     extra state predicate makes the write race-safe: if another worker
     already grabbed the row, our ``count`` is 0 and we loop to the next
     candidate.
  3. Builds the hydrate payload, pushes via the configured transport.
  4. Flips ``hydrating -> attached`` and returns.

We deliberately do NOT use ``SELECT FOR UPDATE SKIP LOCKED`` here even
though LIT-2890 mentions it. Reason: Prisma's Python client doesn't expose
``FOR UPDATE`` and ``execute_raw`` would couple us to one DB dialect. The
"update_many with state predicate" pattern is functionally equivalent for
a single-row CAS and works on Postgres + SQLite without leaning on
dialect-specific syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.managed_agents.vms.team_config import (
    get_team_vm_config,
)
from litellm.managed_agents.vms.warm_pool.hydrate import (
    build_hydrate_payload,
)
from litellm.managed_agents.vms.warm_pool.transports.ssm import (
    HydrateTransportError,
    get_default_transport,
)
from litellm.managed_agents.vms.warm_pool.types import HydratePayload

# How many candidate rows we'll try before giving up and falling back to
# cold-boot. Higher values trade attach latency for pool-empty resilience —
# 5 is plenty since the pool size in practice is 2-10 per team.
MAX_ATTACH_CANDIDATES = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AttachResult:
    """Returned by ``attach_warm_vm`` on success."""

    vm_id: str
    region: Optional[str]
    payload: HydratePayload


class WarmPoolEmptyError(RuntimeError):
    """Raised when no warm VM is available — caller falls back to cold boot."""


async def _claim_one_warm(
    prisma_client: Any, team_id: str, session_id: str
) -> Optional[Any]:
    """Atomic warm -> hydrating flip on the oldest warm row for ``team_id``.

    Returns the (re-fetched) row on success, ``None`` if all candidates were
    snatched by other workers between the read and the CAS.
    """
    candidates: List[Any] = await prisma_client.db.litellm_agentvm.find_many(
        where={"team_id": team_id, "state": "warm"},
        order={"warmed_at": "asc"},
        take=MAX_ATTACH_CANDIDATES,
    )
    if not candidates:
        return None

    for candidate in candidates:
        # Atomic CAS: flip warm -> hydrating only if still warm. Prisma's
        # update_many returns the count of rows it modified. If 0, another
        # worker already claimed this row and we move to the next candidate.
        result = await prisma_client.db.litellm_agentvm.update_many(
            where={"id": candidate.id, "state": "warm"},
            data={
                "state": "hydrating",
                "attached_session_id": session_id,
                "last_hydrate_at": _now(),
            },
        )
        # Prisma's update_many returns either an int count or a wrapper object
        # exposing ``.count`` depending on version — handle both.
        affected = result if isinstance(result, int) else getattr(result, "count", 0)
        if affected and affected > 0:
            # Refetch the row so we have the latest fields (incl. region/metadata).
            return await prisma_client.db.litellm_agentvm.find_unique(
                where={"id": candidate.id}
            )
    return None


async def _mark_attached(prisma_client: Any, vm_id: str, session_id: str) -> None:
    await prisma_client.db.litellm_agentvm.update_many(
        where={"id": vm_id, "state": "hydrating"},
        data={"state": "attached", "attached_session_id": session_id},
    )


async def _release_back_to_warm(prisma_client: Any, vm_id: str) -> None:
    """If hydrate failed mid-flight, mark the VM terminating so the
    maintenance loop replaces it. Never put it back to ``warm`` — the daemon
    may have started consuming the partial payload."""
    try:
        await prisma_client.db.litellm_agentvm.update_many(
            where={"id": vm_id, "state": "hydrating"},
            data={"state": "terminating", "attached_session_id": None},
        )
    except Exception as exc:
        verbose_proxy_logger.warning(
            "warm_pool.attach: failed to release vm=%s: %s", vm_id, exc
        )


async def attach_warm_vm(
    *,
    prisma_client: Any,
    team_id: str,
    session_id: str,
    agent_id: str,
    jwt: str,
    jwt_expires_at: datetime,
    repos: List[Any],
    env_vars: Optional[dict],
    agent_row: Any = None,
    transport: Any = None,
) -> AttachResult:
    """Attempt to attach a warm VM for the given session.

    Raises ``WarmPoolEmptyError`` if no warm VMs are available. Caller
    should fall back to the cold-boot path in that case.

    Raises ``HydrateTransportError`` if the SSM push failed. The VM has
    been moved to ``terminating`` so the maintenance loop will replace it.
    """
    if not team_id:
        raise WarmPoolEmptyError("team_id required to attach warm VM")

    row = await _claim_one_warm(prisma_client, team_id, session_id)
    if row is None:
        raise WarmPoolEmptyError(f"no warm VM available for team={team_id}")

    payload = await build_hydrate_payload(
        prisma_client=prisma_client,
        session_id=session_id,
        agent_id=agent_id,
        team_id=team_id,
        jwt=jwt,
        jwt_expires_at=jwt_expires_at,
        repos=list(repos or []),
        env_vars=env_vars,
        agent_row=agent_row,
    )

    try:
        team_resolved = await get_team_vm_config(team_id, prisma_client)
        aws_creds = team_resolved.aws_creds
        region = row.region or team_resolved.ec2_config.region
    except Exception as exc:
        await _release_back_to_warm(prisma_client, row.id)
        raise HydrateTransportError(
            f"warm_pool.attach: BYOC creds resolve failed for team={team_id}: "
            f"{type(exc).__name__}"
        ) from exc

    active_transport = transport or get_default_transport()
    try:
        await active_transport.push(
            vm_id=row.id,
            region=region,
            aws_creds=aws_creds,
            payload=payload,
        )
    except Exception:
        await _release_back_to_warm(prisma_client, row.id)
        raise

    await _mark_attached(prisma_client, row.id, session_id)
    verbose_proxy_logger.info(
        "warm_pool.attach: attached vm=%s session=%s team=%s",
        row.id,
        session_id,
        team_id,
    )
    return AttachResult(vm_id=row.id, region=row.region, payload=payload)


__all__ = [
    "AttachResult",
    "MAX_ATTACH_CANDIDATES",
    "WarmPoolEmptyError",
    "attach_warm_vm",
]
