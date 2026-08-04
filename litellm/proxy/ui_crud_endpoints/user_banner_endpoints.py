import asyncio
import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError, model_validator

from litellm.proxy._types import LitellmTableNames, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.table_repositories import UISettingsRepository

router = APIRouter()

USER_BANNER_ROW_ID = "user_banner"
USER_BANNER_CACHE_KEY = "user_banner:settings_dict"
USER_BANNER_CACHE_TTL = 60
USER_BANNER_MAX_MESSAGE_LENGTH = 4000

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
    revision: int = Field(
        default=0,
        description="Server-stamped publish revision; increments on every update so clients re-surface dismissed banners on republish.",
    )


class UpdateUserBannerResponse(BaseModel):
    message: str
    banner: UserBanner


def parse_user_banner(raw_settings: object) -> UserBanner:
    if raw_settings is None:
        return UserBanner()
    try:
        parsed = json.loads(raw_settings) if isinstance(raw_settings, str) else raw_settings
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
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    cached = await user_api_key_cache.async_get_cache(key=USER_BANNER_CACHE_KEY)
    if cached is not None:
        return parse_user_banner(cached)

    if prisma_client is None:
        return UserBanner()

    db_record = await UISettingsRepository(prisma_client).table.find_unique(
        where={"id": USER_BANNER_ROW_ID}  # mutable-ok: prisma filters are plain dicts
    )
    banner = parse_user_banner(db_record.ui_settings if db_record is not None else None)

    await user_api_key_cache.async_set_cache(
        key=USER_BANNER_CACHE_KEY,
        value=banner.model_dump(),
        ttl=USER_BANNER_CACHE_TTL,
    )
    return banner


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
    from litellm.proxy.proxy_server import (
        create_config_audit_log,
        prisma_client,
        store_model_in_db,
        user_api_key_cache,
    )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can update the user banner.")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected. Please connect a database.")

    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail="Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature.",
        )

    db_record = await UISettingsRepository(prisma_client).table.find_unique(
        where={"id": USER_BANNER_ROW_ID}  # mutable-ok: prisma filters are plain dicts
    )
    before = parse_user_banner(db_record.ui_settings if db_record is not None else None)
    banner = UserBanner(
        enabled=banner_update.enabled,
        message=banner_update.message,
        severity=banner_update.severity,
        revision=before.revision + 1,
    )

    payload = json.dumps(banner.model_dump())
    banner_row = {"id": USER_BANNER_ROW_ID, "ui_settings": payload}  # mutable-ok: prisma rows are plain dicts
    await UISettingsRepository(prisma_client).table.upsert(
        where={"id": USER_BANNER_ROW_ID},  # mutable-ok: prisma filters are plain dicts
        data={  # mutable-ok: prisma upsert payloads are plain dicts
            "create": banner_row,
            "update": {"ui_settings": payload},  # mutable-ok: prisma upsert payloads are plain dicts
        },
    )

    await user_api_key_cache.async_set_cache(
        key=USER_BANNER_CACHE_KEY,
        value=banner.model_dump(),
        ttl=USER_BANNER_CACHE_TTL,
    )

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
