"""
Endpoints for managing email alerts on litellm
"""

import json
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from litellm_enterprise.types.enterprise_callbacks.send_emails import (
    DefaultEmailSettings,
    EmailEvent,
    EmailEventSettings,
    EmailEventSettingsResponse,
    EmailEventSettingsUpdateRequest,
)

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


async def _get_email_settings(prisma_client) -> Dict[str, bool]:
    """Helper function to get email settings from general_settings in db"""
    try:
        # Get general settings from db
        general_settings_entry = await prisma_client.db.litellm_config.find_unique(
            where={"param_name": "general_settings"}
        )

        # Initialize with default email settings
        settings_dict = DefaultEmailSettings.get_defaults()

        if (
            general_settings_entry is not None
            and general_settings_entry.param_value is not None
        ):
            # Get general settings value
            if isinstance(general_settings_entry.param_value, str):
                general_settings = json.loads(general_settings_entry.param_value)
            else:
                general_settings = general_settings_entry.param_value

            # Extract email_settings from general settings if it exists
            if general_settings and "email_settings" in general_settings:
                email_settings = general_settings["email_settings"]
                # Update settings_dict with values from general_settings
                for event_name, enabled in email_settings.items():
                    settings_dict[event_name] = enabled

        return settings_dict
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error getting email settings from general_settings: {str(e)}"
        )
        # Return default settings in case of error
        return DefaultEmailSettings.get_defaults()


async def _save_email_settings(prisma_client, settings: Dict[str, bool]):
    """Helper function to save email settings to general_settings in db"""
    try:
        verbose_proxy_logger.debug(
            f"Saving email settings to general_settings: {settings}"
        )

        # Get current general settings
        general_settings_entry = await prisma_client.db.litellm_config.find_unique(
            where={"param_name": "general_settings"}
        )

        # Initialize general settings dict
        if (
            general_settings_entry is not None
            and general_settings_entry.param_value is not None
        ):
            if isinstance(general_settings_entry.param_value, str):
                general_settings = json.loads(general_settings_entry.param_value)
            else:
                general_settings = dict(general_settings_entry.param_value)
        else:
            general_settings = {}

        # Update email_settings in general_settings
        general_settings["email_settings"] = settings

        # Convert to JSON for storage
        json_settings = json.dumps(general_settings, default=str)

        # Save updated general settings
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "general_settings"},
            data={
                "create": {
                    "param_name": "general_settings",
                    "param_value": json_settings,
                },
                "update": {"param_value": json_settings},
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving email settings to general_settings: {str(e)}",
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
        # Get existing settings
        settings_dict = await _get_email_settings(prisma_client)

        # Create a response with all events (enabled or disabled)
        response_settings = []
        for event in EmailEvent:
            enabled = settings_dict.get(event.value, False)
            response_settings.append(EmailEventSettings(event=event, enabled=enabled))

        return EmailEventSettingsResponse(settings=response_settings)
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting email settings: {str(e)}")
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
        # Get existing settings
        settings_dict = await _get_email_settings(prisma_client)

        # Update with new settings
        for setting in request.settings:
            settings_dict[setting.event.value] = setting.enabled

        # Save updated settings
        await _save_email_settings(prisma_client, settings_dict)

        return {"message": "Email event settings updated successfully"}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating email settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/email/event_settings/reset",
    tags=["email management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def reset_event_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Reset all email event settings to default (new user invitations on, virtual key creation off)
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Reset to default settings using the Pydantic model
        default_settings = DefaultEmailSettings.get_defaults()

        # Save default settings
        await _save_email_settings(prisma_client, default_settings)

        return {"message": "Email event settings reset to defaults"}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error resetting email settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
