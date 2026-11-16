from typing import Annotated, Final

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import ConfigDict

from litellm._uuid import uuid4
from litellm.models.user_banner import UserBanner, UserBannerUpdate
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import UserBannerRepositoryDep
from litellm.proxy.ui_crud_endpoints.user_banner_dependencies import (
    UserBannerAuditDep,
)
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter()


class UpdateUserBannerResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    banner: UserBanner


@router.get(
    "/get/user_banner",
    tags=["UI Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=UserBanner,
)
async def get_user_banner(banners: UserBannerRepositoryDep) -> UserBanner:
    """
    Get the admin-published dashboard banner.
    Readable by any authenticated user; rendered on every dashboard page.
    """
    if banners is None:
        return UserBanner()

    return await banners.get_banner()


@router.patch(
    "/update/user_banner",
    tags=["UI Settings"],
    response_model=UpdateUserBannerResponse,
)
async def update_user_banner(
    banner_update: UserBannerUpdate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    banners: UserBannerRepositoryDep,
    audit: UserBannerAuditDep,
    background_tasks: BackgroundTasks,
) -> UpdateUserBannerResponse:
    """
    Publish, edit, or unpublish the dashboard banner.
    Only proxy admins are allowed to modify it.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can update the user banner.")

    if banners is None:
        raise HTTPException(status_code=500, detail="Database not connected. Please connect a database.")

    before: Final = await banners.get_banner()
    banner: Final = UserBanner(
        enabled=banner_update.enabled,
        message=banner_update.message,
        severity=banner_update.severity,
        revision=uuid4().hex,
    )

    await banners.save_banner(banner)

    background_tasks.add_task(audit, before, banner, user_api_key_dict)

    return UpdateUserBannerResponse(message="User banner updated successfully", banner=banner)
