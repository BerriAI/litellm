from typing import TYPE_CHECKING, Annotated, Final

from fastapi import APIRouter, Depends

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.management.users.exceptions import current_user_not_found
from litellm.proxy.management.users.schemas import CurrentUserResponse
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.repositories.user_repository import UserRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


async def _lookup_user_alias(prisma_client: "PrismaClient", user_id: str) -> str | None:
    try:
        user: Final = await bounded_db_lookup(
            UserRepository(prisma_client).table.find_unique(where={"user_id": user_id}),
            name="current user alias",
        )
    except Exception as exc:
        if not PrismaDBExceptionHandler.is_database_service_unavailable_error_in_chain(exc):
            raise
        verbose_proxy_logger.warning("Database unavailable during current user alias lookup: %s", type(exc).__name__)
        return None
    return user.user_alias if user is not None else None


@router.get("/users/me", include_in_schema=False)
async def get_current_user(
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CurrentUserResponse:
    from litellm.proxy.proxy_server import prisma_client

    if caller.user_id is None:
        raise current_user_not_found()

    user_alias: Final = (
        await _lookup_user_alias(prisma_client, caller.user_id)
        if prisma_client is not None and not (caller.user_email or "").strip()
        else None
    )

    return CurrentUserResponse(
        user_id=caller.user_id, user_role=caller.user_role, user_email=caller.user_email, user_alias=user_alias
    )
