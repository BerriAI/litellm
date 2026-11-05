from typing import Annotated, Final

from fastapi import APIRouter, Depends

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management.users.exceptions import current_user_not_found
from litellm.proxy.management.users.schemas import CurrentUserResponse
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)


@router.get("/users/me", include_in_schema=False)
async def get_current_user(
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> CurrentUserResponse:
    if caller.user_id is None:
        raise current_user_not_found()
    return CurrentUserResponse(user_id=caller.user_id, user_role=caller.user_role, user_email=caller.user_email)
