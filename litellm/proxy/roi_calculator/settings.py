from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Final, Literal

from fastapi import Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.roi_calculator import DEFAULT_PROMPT, ROISettings

_SETTINGS_KEY: Final = "roi_calculator_settings"


class StoredROISettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    revision: int = 0
    source_provider: Literal["github", "gitlab"] = "github"
    connection_type: Literal["token", "app"] = "token"
    oauth_refresh_token: str = ""
    oauth_expires_at: datetime | None = None
    ignored_logins: tuple[str, ...] = ()
    gitlab_api_url: str = "https://gitlab.com/api/v4"
    gitlab_token: str = ""
    github_api_url: str = "https://api.github.com"
    github_token: str = ""
    estimator_key: str = ""
    repos: tuple[str, ...] = ()
    estimator_model: str = ""
    estimator_prompt: str = DEFAULT_PROMPT
    backfill_days: int = Field(default=7, ge=1, le=3650)
    update_interval_minutes: float = Field(default=1440, ge=0, le=43200)
    identity_map: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}))


async def read_admin(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(status_code=403, detail="Only proxy admins can access the ROI Calculator.")
    return user_api_key_dict


async def write_admin(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can change ROI Calculator settings.")
    return user_api_key_dict


async def get_roi_config_repository(
    _user: Annotated[UserAPIKeyAuth, Depends(read_admin)],
) -> ConfigRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )
    return ConfigRepository(prisma_client, use_writer=True)


async def load_stored_settings(repository: ConfigRepository) -> StoredROISettings:
    parameter: Final = await repository.get_param(_SETTINGS_KEY)
    if parameter is None:
        return StoredROISettings()
    try:
        return StoredROISettings.model_validate(parameter.param_value)
    except ValidationError:
        raise HTTPException(status_code=500, detail="Stored ROI Calculator settings are invalid.") from None


async def load_settings(repository: ConfigRepository, value: StoredROISettings | None = None) -> ROISettings:
    stored: Final = value if value is not None else await load_stored_settings(repository)
    token: Final = decrypt_value_helper(stored.github_token, _SETTINGS_KEY) if stored.github_token else ""
    try:
        return ROISettings(
            source_provider=stored.source_provider,
            connection_type=stored.connection_type,
            oauth_refresh_token=SecretStr(decrypt_value_helper(stored.oauth_refresh_token, _SETTINGS_KEY) or "")
            if stored.oauth_refresh_token
            else SecretStr(""),
            oauth_expires_at=stored.oauth_expires_at,
            ignored_logins=stored.ignored_logins,
            gitlab_api_url=stored.gitlab_api_url,
            gitlab_token=SecretStr(decrypt_value_helper(stored.gitlab_token, _SETTINGS_KEY) or "")
            if stored.gitlab_token
            else SecretStr(""),
            github_api_url=stored.github_api_url,
            github_token=SecretStr(token or ""),
            estimator_key=SecretStr(decrypt_value_helper(stored.estimator_key, _SETTINGS_KEY) or "")
            if stored.estimator_key
            else SecretStr(""),
            update_interval_minutes=stored.update_interval_minutes,
            repos=stored.repos,
            estimator_model=stored.estimator_model,
            estimator_prompt=stored.estimator_prompt,
            backfill_days=stored.backfill_days,
            identity_map=stored.identity_map,
        )
    except ValidationError:
        raise HTTPException(status_code=500, detail="Stored ROI Calculator settings are invalid.") from None


async def save_settings(
    repository: ConfigRepository,
    settings: ROISettings,
    encrypted_token: str,
    encrypted_estimator_key: str,
    encrypted_gitlab_token: str = "",
    revision: int = 0,
) -> None:
    stored: Final = StoredROISettings(
        revision=revision + 1,
        source_provider=settings.source_provider,
        connection_type=settings.connection_type,
        oauth_refresh_token=TypeAdapter(str).validate_python(
            encrypt_value_helper(settings.oauth_refresh_token.get_secret_value())
        )
        if settings.oauth_refresh_token.get_secret_value()
        else "",
        oauth_expires_at=settings.oauth_expires_at,
        ignored_logins=settings.ignored_logins,
        gitlab_api_url=settings.gitlab_api_url,
        gitlab_token=encrypted_gitlab_token,
        github_api_url=settings.github_api_url,
        github_token=encrypted_token,
        estimator_key=encrypted_estimator_key,
        update_interval_minutes=settings.update_interval_minutes,
        repos=settings.repos,
        estimator_model=settings.estimator_model,
        estimator_prompt=settings.estimator_prompt,
        backfill_days=settings.backfill_days,
        identity_map=settings.identity_map,
    )
    if not await repository.set_param_if_revision(_SETTINGS_KEY, stored.model_dump(mode="json"), revision):
        raise HTTPException(409, "Settings changed while you were editing. Reload and try again.")
