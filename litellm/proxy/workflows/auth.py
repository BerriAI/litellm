from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Final

from fastapi import Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.resolvers.store import IdentityStore


@dataclass(frozen=True, slots=True)
class WorkflowChildGrant:
    principal: str
    team_id: str
    snapshot_id: str
    workflow_model: str
    node_id: str
    model: str
    run_id: str


workflow_child_grant: Final[ContextVar[WorkflowChildGrant | None]] = ContextVar("workflow_child_grant", default=None)


def workflow_parent(model: str) -> str | None:
    grant: Final = workflow_child_grant.get()
    return grant.workflow_model if grant is not None and grant.model == model else None


async def workflow_child_identity(request: Request, request_data: Mapping[str, object]) -> UserAPIKeyAuth | None:
    grant: Final = workflow_child_grant.get()
    if grant is None:
        return None
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if (
        request.method != "POST"
        or not request.url.path.endswith("/v1/chat/completions")
        or request_data.get("model") != grant.model
    ):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Invalid internal workflow child scope")
    identity: Final = IdentityStore.key_from_principal(
        await IdentityStore(
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        ).resolve(hashed_token=grant.principal)
    )
    if (
        identity.team_id != grant.team_id
        or identity.is_session_token
        or identity.user_role
        in {
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        }
    ):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Workflow requires its original team virtual key")
    return identity.model_copy(update={"via_virtual_key": True})
