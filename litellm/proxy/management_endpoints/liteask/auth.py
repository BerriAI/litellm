from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Protocol

from fastapi import Depends, HTTPException
from pydantic import BaseModel

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import get_user_object
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.auth.auth_checks import UserNotFoundError


class _UserMetadata(BaseModel):
    metadata: Mapping[str, object] | None = None


class UserLookup(Protocol):
    def __call__(self, user_id: str) -> Awaitable[LiteLLM_UserTable | None]: ...


@dataclass(frozen=True, slots=True)
class AdminRefusal:
    status_code: Literal[403, 503]
    message: str


async def load_live_user(user_id: str) -> LiteLLM_UserTable | None:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    return await get_user_object(
        user_id=user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        user_id_upsert=False,
        check_db_only=True,
        proxy_logging_obj=proxy_logging_obj,
    )


async def check_fresh_admin(caller: UserAPIKeyAuth, lookup: UserLookup) -> UserAPIKeyAuth | AdminRefusal:
    if caller.user_role != LitellmUserRoles.PROXY_ADMIN or not caller.user_id:
        return AdminRefusal(403, "LiteAsk requires a signed-in proxy administrator.")
    try:
        user: Final = await lookup(caller.user_id)
    except UserNotFoundError:
        return AdminRefusal(403, "LiteAsk requires a current proxy administrator account.")
    except Exception:  # noqa: BLE001  # failed identity lookup must deny without exposing database errors
        return AdminRefusal(503, "Administrator access could not be verified. Try again shortly.")
    if user is None or user.user_id != caller.user_id or user.user_role != LitellmUserRoles.PROXY_ADMIN:
        return AdminRefusal(403, "LiteAsk requires a current proxy administrator account.")
    metadata: Final = _UserMetadata.model_validate(user.model_dump()).metadata
    if metadata is not None and metadata.get("scim_active") is False:
        return AdminRefusal(403, "LiteAsk requires a current proxy administrator account.")
    return caller


async def assert_fresh_admin(caller: UserAPIKeyAuth, *, lookup: UserLookup = load_live_user) -> UserAPIKeyAuth:
    result: Final = await check_fresh_admin(caller, lookup)
    if isinstance(result, AdminRefusal):
        raise HTTPException(status_code=result.status_code, detail=result.message)
    return result


async def fresh_admin(caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> UserAPIKeyAuth:
    return await assert_fresh_admin(caller)
