"""
`GET /v2/agent-vm-pool/status` — live VM pool status.

This endpoint returns the warm-pool / hydrating / attached counts the dashboard
shows under "View live pool status" on the Warm Pool screen.

The real implementation is owned by Epic B2 (LIT-2890), which actually tracks
warm VMs, hydrate state, and session attachments. To unblock the UI ahead of
B2, we ship a deterministic stub here:

* When `LITELLM_AGENT_POOL_STATUS_MOCK=1` (the default while B2 is open), the
  endpoint returns `{warm: 0, hydrating: 0, attached: 0}`.
* When B2 lands, that flag flips to `0` (or the env var goes away entirely)
  and the implementation reads from B2's session/VM tracking tables.

Keeping the route here means the UI ships against a stable contract today —
no front-end changes required when B2 swaps the body.
"""

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class PoolStatusResponse(BaseModel):
    """Match the shape B2 will return so the UI doesn't have to change."""

    warm: int
    hydrating: int
    attached: int
    # Optional for the v1 stub; B2 will populate.
    avg_create_latency_ms: Optional[int] = None
    oldest_warm_seconds: Optional[int] = None


def _pool_status_mock_enabled() -> bool:
    """B2 owns the real status path; default to mock until then."""
    return os.getenv("LITELLM_AGENT_POOL_STATUS_MOCK", "1") == "1"


def _resolve_team_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    team_id = user_api_key_dict.team_id or (user_api_key_dict.metadata or {}).get(
        "team_id"
    )
    if not team_id:
        raise HTTPException(
            status_code=400,
            detail="Cloud Agent pool status is scoped to a team.",
        )
    return team_id


@router.get(
    "/v2/agent-vm-pool/status",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PoolStatusResponse,
    tags=["cloud agents"],
)
async def get_agent_vm_pool_status(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PoolStatusResponse:
    """Return live VM pool counts. Stubbed until B2 (LIT-2890) lands."""
    # Validate team scoping even in the mock so we surface 400s consistently
    # with the rest of the cloud-agents endpoints — saves a UI bug later.
    _resolve_team_id(user_api_key_dict)

    if _pool_status_mock_enabled():
        return PoolStatusResponse(warm=0, hydrating=0, attached=0)

    # Real path — B2 will fill this in. Until then, refusing loudly is better
    # than silently returning zeros and pretending everything is fine.
    raise HTTPException(
        status_code=501,
        detail={
            "error": (
                "Live pool status is owned by LIT-2890 (Epic B2). Set "
                "LITELLM_AGENT_POOL_STATUS_MOCK=1 to use the stub."
            ),
            "code": "not_implemented",
        },
    )
