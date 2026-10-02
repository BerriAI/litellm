"""
ORACLE ROUTER MANAGEMENT ENDPOINTS

POST /oracle_router/feedback - Report a finished program so its verifier runs and the router learns
GET  /oracle_router/state    - Live decision-maker state, bindings and recent feedback per router
"""

from itertools import chain
from typing import TYPE_CHECKING, Annotated, Final

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.management_endpoints.oracle_router_endpoints import (
    OracleRouterFeedbackRequest,
    OracleRouterFeedbackResponse,
    OracleRouterStateResponse,
)

if TYPE_CHECKING:
    from fastapi import APIRouter, Depends, HTTPException

    from litellm.router_strategy.oracle_router.oracle_router import OracleRouter
else:
    try:
        from fastapi import APIRouter, Depends, HTTPException
    except ImportError:
        pass

router: Final = APIRouter()


def _is_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)


def _oracle_routers() -> tuple["OracleRouter", ...]:
    from litellm.proxy.proxy_server import llm_router

    registry: Final = llm_router.oracle_routers if llm_router is not None else {}
    tagged_entries: Final = tuple(chain.from_iterable(registry.values()))
    return tuple(tagged.strategy for tagged in tagged_entries)


@router.post(
    "/oracle_router/feedback",
    tags=["oracle_router"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OracleRouterFeedbackResponse,
)
async def submit_oracle_router_feedback(
    data: OracleRouterFeedbackRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> OracleRouterFeedbackResponse:
    """Complete a program: its model slot is released now, its verifier runs in the background.

    The key that started the program (or an admin) may complete it. 404 when no ORACLE router holds a
    program with that id, which also covers programs completed earlier.
    """
    routers: Final = _oracle_routers()
    if not routers:
        raise HTTPException(status_code=404, detail={"error": "No oracle_router is configured on this proxy."})
    owner: Final = next((candidate for candidate in routers if candidate.binding(data.program_id) is not None), None)
    binding: Final = owner.binding(data.program_id) if owner is not None else None
    if owner is None or binding is None:
        raise HTTPException(
            status_code=404, detail={"error": f"Unknown or already completed program {data.program_id!r}."}
        )
    if binding.api_key_hash and binding.api_key_hash != user_api_key_dict.api_key and not _is_admin(user_api_key_dict):
        raise HTTPException(status_code=403, detail={"error": CommonProxyErrors.not_allowed_access.value})
    owner.complete(data.program_id, score=data.score, cost=data.cost, payload=data.payload)
    return OracleRouterFeedbackResponse(
        program_id=data.program_id,
        router_name=owner.router_name,
        model=binding.model,
        pending_verifications=owner.pending_verifications,
    )


@router.get(
    "/oracle_router/state",
    tags=["oracle_router"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OracleRouterStateResponse,
)
async def get_oracle_router_state(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> OracleRouterStateResponse:
    """Decision-maker state, active bindings and recent verified feedback for every ORACLE router. Admin only."""
    if not _is_admin(user_api_key_dict):
        raise HTTPException(status_code=403, detail={"error": CommonProxyErrors.not_allowed_access.value})
    routers: Final = _oracle_routers()
    if not routers:
        raise HTTPException(status_code=404, detail={"error": "No oracle_router is configured on this proxy."})
    return OracleRouterStateResponse(routers=[await candidate.get_state_snapshot() for candidate in routers])
