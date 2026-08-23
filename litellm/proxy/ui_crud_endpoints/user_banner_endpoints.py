import asyncio
import json
from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator

from litellm._uuid import uuid4
from litellm.proxy._types import LitellmTableNames, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.user_banner_repository import USER_BANNER_ROW_ID, UserBannerRepository

router: Final = APIRouter()

USER_BANNER_MAX_MESSAGE_LENGTH: Final = 4000

UserBannerSeverity = Literal["info", "warning", "error"]


class UserBannerUpdate(BaseModel):
    enabled: bool = Field(
        default=False,
        description="If true, the banner is shown to all authenticated dashboard users.",
    )
    message: str = Field(
        default="",
        max_length=USER_BANNER_MAX_MESSAGE_LENGTH,
        description="Banner text shown to dashboard users. Markdown is supported.",
    )
    severity: UserBannerSeverity = Field(
        default="info",
        description="Visual style of the banner.",
    )

    @model_validator(mode="after")
    def _require_message_when_enabled(self) -> "UserBannerUpdate":
        if self.enabled and not self.message.strip():
            raise ValueError("message must be non-empty when the banner is enabled")
        return self


class UserBanner(UserBannerUpdate):
    revision: str = Field(
        default="",
        description=(
            "Server-stamped opaque publish identity; a fresh value is generated on every "
            "update so clients re-surface dismissed banners on republish."
        ),
    )


class UpdateUserBannerResponse(BaseModel):
    message: str
    banner: UserBanner


def parse_user_banner(raw_settings: object) -> UserBanner:
    if raw_settings is None:
        return UserBanner()
    try:
        parsed: Final = json.loads(raw_settings) if isinstance(raw_settings, str) else raw_settings
        return UserBanner.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError):
        return UserBanner()


@router.get(
    "/get/user_banner",
    tags=["UI Settings"],  # mutable-ok: FastAPI's route decorator only accepts a list
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI's route decorator only accepts a list
    response_model=UserBanner,
)
async def get_user_banner() -> UserBanner:
    """
    Get the admin-published dashboard banner.
    Readable by any authenticated user; rendered on every dashboard page.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return UserBanner()

    raw_settings: Final = await UserBannerRepository(prisma_client).get_raw_settings()
    return parse_user_banner(raw_settings)


@router.patch(
    "/update/user_banner",
    tags=["UI Settings"],  # mutable-ok: FastAPI's route decorator only accepts a list
    response_model=UpdateUserBannerResponse,
)
async def update_user_banner(
    banner_update: UserBannerUpdate,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UpdateUserBannerResponse:
    """
    Publish, edit, or unpublish the dashboard banner.
    Only proxy admins are allowed to modify it.
    """
    from litellm.proxy.proxy_server import create_config_audit_log, prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can update the user banner.")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected. Please connect a database.")

    repository: Final = UserBannerRepository(prisma_client)
    before: Final = parse_user_banner(await repository.get_raw_settings())
    banner: Final = UserBanner(
        enabled=banner_update.enabled,
        message=banner_update.message,
        severity=banner_update.severity,
        revision=uuid4().hex,
    )

    await repository.upsert_settings(json.dumps(banner.model_dump()))

    asyncio.create_task(
        create_config_audit_log(
            param_name=USER_BANNER_ROW_ID,
            action="updated",
            before_value=before.model_dump(),
            after_value=banner.model_dump(),
            user_api_key_dict=user_api_key_dict,
            table_name=LitellmTableNames.UI_SETTINGS_TABLE_NAME,
        )
    )

    return UpdateUserBannerResponse(message="User banner updated successfully", banner=banner)
