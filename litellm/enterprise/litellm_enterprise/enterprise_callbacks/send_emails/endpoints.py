"""
Endpoints for managing email alerts on litellm
"""

import json
from collections.abc import Mapping
from typing import Final

from fastapi import APIRouter, Depends, HTTPException
from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    DefaultEmailSettings,
    EmailEvent,
    EmailEventSettings,
    EmailEventSettingsResponse,
    EmailEventSettingsUpdateRequest,
)
from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.db.db_span import db_span
from litellm.proxy.utils import PrismaClient
from litellm.repositories.config_repository import CONFIG_PARAM_LOCK_SQL, ConfigRepository
from litellm.types.integrations.model_deprecation_email import MODEL_DEPRECATION_EMAIL_RECIPIENT

router = APIRouter()


_GENERAL_SETTINGS: Final = "general_settings"
_CONFIG_VALUES: Final = TypeAdapter(dict[str, JsonValue])
_EMAIL_SETTINGS: Final = TypeAdapter(dict[str, bool])


def _general_settings(value: object) -> dict[str, JsonValue]:
    if value is None:
        return {}
    if isinstance(value, str):
        return _CONFIG_VALUES.validate_json(value)
    return _CONFIG_VALUES.validate_python(value)


def _event_flags(general_settings: Mapping[str, JsonValue]) -> dict[str, bool]:
    stored: Final = _EMAIL_SETTINGS.validate_python(general_settings.get("email_settings") or {})
    return {**DefaultEmailSettings.get_defaults(), **stored}


async def _read_general_settings(prisma_client: PrismaClient) -> dict[str, JsonValue]:
    record: Final = await ConfigRepository(prisma_client, use_writer=True).get_param(_GENERAL_SETTINGS)
    return _general_settings(record.param_value if record else None)


async def get_email_settings(prisma_client: PrismaClient) -> dict[str, bool]:
    try:
        return _event_flags(await _read_general_settings(prisma_client))
    except Exception:
        verbose_proxy_logger.exception("Error reading email event settings")
        return DefaultEmailSettings.get_defaults()


async def _save_email_settings(
    prisma_client: PrismaClient,
    settings: Mapping[str, bool],
    *,
    merge: bool = False,
    recipient_change: Mapping[str, JsonValue] | None = None,
) -> None:
    from litellm.proxy.proxy_server import proxy_config

    async with prisma_client.tx() as transaction:
        await transaction.query_raw(CONFIG_PARAM_LOCK_SQL, _GENERAL_SETTINGS)
        table: Final = transaction.litellm_config
        entry: Final = await table.find_unique(where={"param_name": _GENERAL_SETTINGS})
        current: Final = _general_settings(entry.param_value if entry else None)
        flags: Final = {**_event_flags(current), **settings} if merge else dict(settings)
        changes: Final[dict[str, JsonValue]] = {
            "email_settings": {event: enabled for event, enabled in flags.items()},
            **(recipient_change or {}),
        }
        proxy_config.reject_config_owned_writes(section_name=_GENERAL_SETTINGS, changed_keys=changes)
        serialized: Final = json.dumps({**current, **changes})
        async with db_span("save_email_settings", "LiteLLM_Config"):
            await table.upsert(
                where={"param_name": _GENERAL_SETTINGS},
                data={
                    "create": {"param_name": _GENERAL_SETTINGS, "param_value": serialized},
                    "update": {"param_value": serialized},
                },
            )


@router.get(
    "/email/event_settings",
    response_model=EmailEventSettingsResponse,
    tags=["email management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_email_event_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get all email event settings
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        general_settings: Final = await _read_general_settings(prisma_client)
        flags: Final = _event_flags(general_settings)
        return EmailEventSettingsResponse(
            settings=[EmailEventSettings(event=event, enabled=flags[event.value]) for event in EmailEvent],
            model_deprecation_email_recipient=general_settings.get(MODEL_DEPRECATION_EMAIL_RECIPIENT),
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting email settings: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/email/event_settings",
    tags=["email management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_event_settings(
    request: EmailEventSettingsUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update the settings for email events
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        changes_recipient: Final = MODEL_DEPRECATION_EMAIL_RECIPIENT in request.model_fields_set
        if changes_recipient and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(status_code=403, detail="Only proxy admins can change the model deprecation recipient")

        recipient: Final = request.model_deprecation_email_recipient
        await _save_email_settings(
            prisma_client,
            {setting.event.value: setting.enabled for setting in request.settings},
            merge=True,
            recipient_change={MODEL_DEPRECATION_EMAIL_RECIPIENT: str(recipient) if recipient else None}
            if changes_recipient
            else None,
        )

        return {"message": "Email event settings updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating email settings: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/email/event_settings/reset",
    tags=["email management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def reset_event_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Restore the default email events. Proxy admins also clear the model deprecation recipient override."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        await _save_email_settings(
            prisma_client,
            DefaultEmailSettings.get_defaults(),
            recipient_change={MODEL_DEPRECATION_EMAIL_RECIPIENT: None}
            if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
            else None,
        )

        return {"message": "Email event settings reset to defaults"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error resetting email settings: {e!s}")
        raise HTTPException(status_code=500, detail=str(e))
