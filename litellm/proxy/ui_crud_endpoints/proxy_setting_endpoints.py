#### CRUD ENDPOINTS for UI Settings #####
from typing import List

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class IPAddress(BaseModel):
    ip: str


@router.get(
    "/get/allowed_ips",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def get_allowed_ips():
    from litellm.proxy.proxy_server import general_settings

    _allowed_ip = general_settings.get("allowed_ips")
    return {"data": _allowed_ip}


@router.post(
    "/add/allowed_ip",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def add_allowed_ip(ip_address: IPAddress):
    from litellm.proxy.proxy_server import (
        general_settings,
        prisma_client,
        proxy_config,
        store_model_in_db,
    )

    _allowed_ips: List = general_settings.get("allowed_ips", [])
    if ip_address.ip not in _allowed_ips:
        _allowed_ips.append(ip_address.ip)
        general_settings["allowed_ips"] = _allowed_ips
    else:
        raise HTTPException(status_code=400, detail="IP address already exists")

    if prisma_client is None:
        raise Exception("No DB Connected")

    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )

    # Load existing config
    config = await proxy_config.get_config()
    verbose_proxy_logger.debug("Loaded config: %s", config)
    if "general_settings" not in config:
        config["general_settings"] = {}

    if "allowed_ips" not in config["general_settings"]:
        config["general_settings"]["allowed_ips"] = []

    if ip_address.ip not in config["general_settings"]["allowed_ips"]:
        config["general_settings"]["allowed_ips"].append(ip_address.ip)

    await proxy_config.save_config(new_config=config)

    return {
        "message": f"IP {ip_address.ip} address added successfully",
        "status": "success",
    }


@router.post(
    "/delete/allowed_ip",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_allowed_ip(ip_address: IPAddress):
    from litellm.proxy.proxy_server import general_settings, proxy_config

    _allowed_ips: List = general_settings.get("allowed_ips", [])
    if ip_address.ip in _allowed_ips:
        _allowed_ips.remove(ip_address.ip)
        general_settings["allowed_ips"] = _allowed_ips
    else:
        raise HTTPException(status_code=404, detail="IP address not found")

    # Load existing config
    config = await proxy_config.get_config()
    verbose_proxy_logger.debug("Loaded config: %s", config)
    if "general_settings" not in config:
        config["general_settings"] = {}

    if "allowed_ips" not in config["general_settings"]:
        config["general_settings"]["allowed_ips"] = []

    if ip_address.ip in config["general_settings"]["allowed_ips"]:
        config["general_settings"]["allowed_ips"].remove(ip_address.ip)

    await proxy_config.save_config(new_config=config)

    return {"message": f"IP {ip_address.ip} deleted successfully", "status": "success"}


@router.get(
    "/get/internal_user_settings",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_sso_settings():
    """
    Get all SSO settings from the litellm_settings configuration.
    Returns a structured object with values and descriptions for UI display.
    """
    from pydantic import TypeAdapter

    from litellm.proxy.proxy_server import proxy_config

    # Load existing config
    config = await proxy_config.get_config()
    litellm_settings = config.get("litellm_settings", {}) or {}
    default_internal_user_params = (
        litellm_settings.get("default_internal_user_params", {}) or {}
    )

    # Create the settings object first
    sso_settings = DefaultInternalUserParams(**(default_internal_user_params))
    # Get the schema for UISSOSettings
    schema = TypeAdapter(DefaultInternalUserParams).json_schema(by_alias=True)

    # Convert to dict for response
    settings_dict = sso_settings.model_dump()

    # Add descriptions to the response
    result = {
        "values": settings_dict,
        "schema": {"description": schema.get("description", ""), "properties": {}},
    }

    # Add property descriptions
    for field_name, field_info in schema["properties"].items():
        result["schema"]["properties"][field_name] = {
            "description": field_info.get("description", ""),
            "type": field_info.get("type", "string"),
        }

    # Add nested object descriptions
    for def_name, def_schema in schema.get("definitions", {}).items():
        result["schema"][def_name] = {
            "description": def_schema.get("description", ""),
            "properties": {
                prop_name: {"description": prop_info.get("description", "")}
                for prop_name, prop_info in def_schema.get("properties", {}).items()
            },
        }

    return result


@router.patch(
    "/update/internal_user_settings",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_internal_user_settings(settings: DefaultInternalUserParams):
    """
    Update the default internal user parameters for SSO users.
    These settings will be applied to new users who sign in via SSO.
    """
    from litellm.proxy.proxy_server import proxy_config

    # Update the in-memory settings
    litellm.default_internal_user_params = settings.model_dump(exclude_none=True)

    # Load existing config
    config = await proxy_config.get_config()

    # Update config with new settings
    if "litellm_settings" not in config:
        config["litellm_settings"] = {}

    config["litellm_settings"]["default_internal_user_params"] = settings.model_dump(
        exclude_none=True
    )

    # Save the updated config
    await proxy_config.save_config(new_config=config)

    return {
        "message": "Internal user settings updated successfully",
        "status": "success",
        "settings": litellm.default_internal_user_params,
    }
