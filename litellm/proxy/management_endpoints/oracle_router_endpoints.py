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

    from litellm.router_strategy.oracle_router.oracle_router import OracleRouter, ProgramBinding
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


def _find_program(
    routers: tuple["OracleRouter", ...], program_id: str, user_api_key_dict: UserAPIKeyAuth
) -> tuple["OracleRouter", "ProgramBinding"] | None:
    """The caller's program with this id. An admin may also reach another key's program of that id."""
    for candidate in routers:
        owned = candidate.binding(program_id, user_api_key_dict.api_key)
        if owned is not None:
            return candidate, owned
    if not _is_admin(user_api_key_dict):
        return None
    for candidate in routers:
        for any_owner in candidate.bindings_named(program_id):
            return candidate, any_owner
    return None


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

    Programs are private to the key that started them; an admin may complete any key's program. 404 when
    the caller holds no program with that id, which also covers programs completed earlier. A program none
    of whose responses has been observed yet is not completed: ``deferred`` is true and its next observed
    response completes it with this feedback.
    """
    routers: Final = _oracle_routers()
    if not routers:
        raise HTTPException(status_code=404, detail={"error": "No oracle_router is configured on this proxy."})
    found: Final = _find_program(routers, data.program_id, user_api_key_dict)
    if found is None:
        raise HTTPException(
            status_code=404, detail={"error": f"Unknown or already completed program {data.program_id!r}."}
        )
    holder, binding = found
    holder.complete(data.program_id, score=data.score, cost=data.cost, payload=data.payload, owner=binding.api_key_hash)
    return OracleRouterFeedbackResponse(
        program_id=data.program_id,
        router_name=holder.router_name,
        model=binding.model,
        pending_verifications=holder.pending_verifications,
        deferred=binding.requests == 0,
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
