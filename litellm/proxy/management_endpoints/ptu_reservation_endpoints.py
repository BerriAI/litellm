"""
PTU RESERVATION MANAGEMENT

All /ptu_reservation management endpoints.

/ptu_reservation/new
/ptu_reservation/list
/ptu_reservation/info
/ptu_reservation/close
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.ptu_reservation_repository import PTUReservationRepository
from litellm.types.proxy.management_endpoints.ptu_reservation import (
    PTUReservationCloseRequest,
    PTUReservationNewRequest,
)

router = APIRouter()

CurrentUser = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )


def _require_feature_enabled() -> None:
    from litellm.proxy.proxy_server import general_settings

    if not general_settings.get("enable_ptu_cost_attribution", False):
        raise HTTPException(
            status_code=403,
            detail={
                "error": (
                    "PTU cost attribution is not enabled. Set 'enable_ptu_cost_attribution: true' in general_settings."
                )
            },
        )


def _require_db() -> "object":
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return prisma_client


@router.post(
    "/ptu_reservation/new",
    tags=["ptu reservation management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_ptu_reservation(
    body: PTUReservationNewRequest,
    user_api_key_dict: CurrentUser,
):
    """Create a new PTU reservation for a (team, model) pair.

    Parameters:
    - team_id (str, required)
    - model (str, required)
    - cost_source (str): "manual" (default). "azure_billing" is reserved.
    - ptu_count (int): required for cost_source="manual", positive
    - cost_per_ptu (float): required for cost_source="manual", non-negative USD/month
    - azure_resource_id (str): reserved; must be null for manual
    - effective_from (datetime, required): inclusive UTC start
    - effective_to (datetime, optional): exclusive UTC end; null = still active
    """
    _require_feature_enabled()
    _require_proxy_admin(user_api_key_dict)
    prisma_client = _require_db()

    if body.cost_source == "azure_billing" and not body.azure_resource_id:
        raise HTTPException(
            status_code=400,
            detail={"error": "azure_resource_id is required when cost_source='azure_billing'"},
        )

    try:
        validated = body.model_dump(exclude_none=False)
        validated_reservation = _validated_domain_model(validated)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    repo = PTUReservationRepository(prisma_client)
    overlapping = await repo.find_overlapping(
        team_id=validated_reservation["team_id"],
        model=validated_reservation["model"],
        effective_from=validated_reservation["effective_from"],
        effective_to=validated_reservation["effective_to"],
    )
    if overlapping:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "reservation overlaps existing active reservation(s) for the same (team, model)",
                "overlapping_ids": [r.id for r in overlapping],
            },
        )

    actor = user_api_key_dict.user_id or "admin"
    create_data = {
        **{k: v for k, v in validated_reservation.items() if k != "id" and v is not None},
        "created_by": actor,
        "updated_by": actor,
    }
    return await repo.table.create(data=create_data)


@router.get(
    "/ptu_reservation/list",
    tags=["ptu reservation management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_ptu_reservations(
    user_api_key_dict: CurrentUser,
    team_id: str | None = None,
    model: str | None = None,
    active_only: bool = False,
):
    """List PTU reservations.

    Query parameters:
    - team_id (optional): filter by team
    - model (optional): filter by model
    - active_only (optional, default false): only reservations live right now
    """
    _require_feature_enabled()
    _require_proxy_admin(user_api_key_dict)
    prisma_client = _require_db()

    repo = PTUReservationRepository(prisma_client)
    if active_only:
        return await repo.find_active(as_of=datetime.now(timezone.utc), team_id=team_id, model=model)

    where: dict = {}
    if team_id is not None:
        where["team_id"] = team_id
    if model is not None:
        where["model"] = model
    return await repo.table.find_many(where=where)


@router.get(
    "/ptu_reservation/info",
    tags=["ptu reservation management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_ptu_reservation(
    id: str,
    user_api_key_dict: CurrentUser,
):
    """Get a single reservation by id.

    Query parameter:
    - id (str, required)
    """
    _require_feature_enabled()
    _require_proxy_admin(user_api_key_dict)
    prisma_client = _require_db()

    repo = PTUReservationRepository(prisma_client)
    row = await repo.table.find_unique(where={"id": id})
    if row is None:
        raise HTTPException(status_code=404, detail={"error": f"reservation '{id}' not found"})
    return row


@router.post(
    "/ptu_reservation/close",
    tags=["ptu reservation management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def close_ptu_reservation(
    body: PTUReservationCloseRequest,
    user_api_key_dict: CurrentUser,
):
    """Close a reservation by setting effective_to.

    Parameters:
    - id (str, required)
    - effective_to (datetime, optional): defaults to now UTC
    """
    _require_feature_enabled()
    _require_proxy_admin(user_api_key_dict)
    prisma_client = _require_db()

    repo = PTUReservationRepository(prisma_client)
    row = await repo.table.find_unique(where={"id": body.id})
    if row is None:
        raise HTTPException(status_code=404, detail={"error": f"reservation '{body.id}' not found"})
    if row.effective_to is not None:
        raise HTTPException(
            status_code=400,
            detail={"error": f"reservation '{body.id}' is already closed at {row.effective_to.isoformat()}"},
        )

    close_at = body.effective_to or datetime.now(timezone.utc)
    if close_at <= row.effective_from:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    f"effective_to ({close_at.isoformat()}) must be strictly after "
                    f"effective_from ({row.effective_from.isoformat()})"
                )
            },
        )

    actor = user_api_key_dict.user_id or "admin"
    return await repo.table.update(
        where={"id": body.id},
        data={"effective_to": close_at, "updated_by": actor},
    )


def _validated_domain_model(payload: dict) -> dict:
    """Validate a reservation payload against the domain model and return its dict form."""
    from litellm.models.ptu_reservation import LiteLLM_PTUReservation

    reservation = LiteLLM_PTUReservation(**payload)
    return reservation.model_dump(exclude_none=False)
