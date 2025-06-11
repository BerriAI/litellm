#### CRUD ENDPOINTS for UI Settings #####
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams, SSOConfig

router = APIRouter()


class IPAddress(BaseModel):
    ip: str


class SettingsResponse(BaseModel):
    """Base response model for settings with values and schema information"""
    
    values: Dict[str, Any]
    """The current configuration values"""
    
    field_schema: Dict[str, Any]
    """Schema information including descriptions and property types for UI display"""


class SSOSettingsResponse(SettingsResponse):
    """Response model for SSO settings"""
    pass


class InternalUserSettingsResponse(SettingsResponse):
    """Response model for internal user settings"""
    pass


class DefaultTeamSettingsResponse(SettingsResponse):
    """Response model for default team settings"""
    pass


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


async def _get_settings_with_schema(
    settings_key: str,
    settings_class: Any,
    config: dict,
) -> dict:
    """
    Common utility function to get settings with schema information.

    Args:
        settings_key: The key in litellm_settings to get
        settings_class: The Pydantic class to use for schema
        config: The config dictionary
    """
    from pydantic import TypeAdapter

    litellm_settings = config.get("litellm_settings", {}) or {}
    settings_data = litellm_settings.get(settings_key, {}) or {}

    # Create the settings object
    settings = settings_class(**(settings_data))
    # Get the schema
    schema = TypeAdapter(settings_class).json_schema(by_alias=True)

    # Convert to dict for response
    settings_dict = settings.model_dump()

    # Add descriptions to the response
    result = {
        "values": settings_dict,
        "field_schema": {"description": schema.get("description", ""), "properties": {}},
    }

    # Add property descriptions
    for field_name, field_info in schema["properties"].items():
        result["field_schema"]["properties"][field_name] = {
            "description": field_info.get("description", ""),
            "type": field_info.get("type", "string"),
        }

    # Add nested object descriptions
    for def_name, def_schema in schema.get("definitions", {}).items():
        result["field_schema"][def_name] = {
            "description": def_schema.get("description", ""),
            "properties": {
                prop_name: {"description": prop_info.get("description", "")}
                for prop_name, prop_info in def_schema.get("properties", {}).items()
            },
        }

    return result


@router.get(
    "/get/internal_user_settings",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=InternalUserSettingsResponse,
)
async def get_internal_user_settings():
    """
    Get all SSO settings from the litellm_settings configuration.
    Returns a structured object with values and descriptions for UI display.
    """
    from litellm.proxy.proxy_server import proxy_config

    # Load existing config
    config = await proxy_config.get_config()

    return await _get_settings_with_schema(
        settings_key="default_internal_user_params",
        settings_class=DefaultInternalUserParams,
        config=config,
    )


@router.get(
    "/get/default_team_settings",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=DefaultTeamSettingsResponse,
)
async def get_default_team_settings():
    """
    Get all SSO settings from the litellm_settings configuration.
    Returns a structured object with values and descriptions for UI display.
    """
    from litellm.proxy.proxy_server import proxy_config

    # Load existing config
    config = await proxy_config.get_config()

    return await _get_settings_with_schema(
        settings_key="default_team_params",
        settings_class=DefaultTeamSSOParams,
        config=config,
    )


async def _update_litellm_setting(
    settings: Union[DefaultInternalUserParams, DefaultTeamSSOParams],
    settings_key: str,
    in_memory_var: Any,
    success_message: str,
):
    """
    Common utility function to update `litellm_settings` in both memory and config.

    Args:
        settings: The settings object to update
        settings_key: The key in litellm_settings to update
        in_memory_var: The in-memory variable to update
        success_message: Message to return on success
    """
    from litellm.proxy.proxy_server import proxy_config

    # Update the in-memory settings
    in_memory_var = settings.model_dump(exclude_none=True)

    # Load existing config
    config = await proxy_config.get_config()

    # Update config with new settings
    if "litellm_settings" not in config:
        config["litellm_settings"] = {}

    config["litellm_settings"][settings_key] = settings.model_dump(exclude_none=True)

    # Save the updated config
    await proxy_config.save_config(new_config=config)

    return {
        "message": success_message,
        "status": "success",
        "settings": in_memory_var,
    }


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
    return await _update_litellm_setting(
        settings=settings,
        settings_key="default_internal_user_params",
        in_memory_var=litellm.default_internal_user_params,
        success_message="Internal user settings updated successfully",
    )


@router.patch(
    "/update/default_team_settings",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_default_team_settings(settings: DefaultTeamSSOParams):
    """
    Update the default team parameters for SSO users.
    These settings will be applied to new teams created from SSO.
    """
    return await _update_litellm_setting(
        settings=settings,
        settings_key="default_team_params",
        in_memory_var=litellm.default_team_params,
        success_message="Default team settings updated successfully",
    )


@router.get(
    "/get/sso_settings",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SSOSettingsResponse,
)
async def get_sso_settings():
    """
    Get all SSO configuration settings from the environment variables.
    Returns a structured object with values and descriptions for UI display.
    """
    import os
    from litellm.proxy.proxy_server import proxy_config
    
    # Load existing config to get both environment variables and general settings
    config = await proxy_config.get_config()
    general_settings = config.get("general_settings", {}) or {}
    environment_variables = config.get("environment_variables", {}) or {}
    
    # Get user_email from general_settings
    proxy_admin_email = general_settings.get("proxy_admin_email", None)
    
    # Helper function to get env var value (first from config, then from environment)
    def get_env_value(env_var_name: str):
        return environment_variables.get(env_var_name) or os.getenv(env_var_name)
    
    # Get current environment variables for SSO
    sso_config = SSOConfig(
        google_client_id=get_env_value("GOOGLE_CLIENT_ID"),
        google_client_secret=get_env_value("GOOGLE_CLIENT_SECRET"),
        microsoft_client_id=get_env_value("MICROSOFT_CLIENT_ID"),
        microsoft_client_secret=get_env_value("MICROSOFT_CLIENT_SECRET"),
        microsoft_tenant=get_env_value("MICROSOFT_TENANT"),
        generic_client_id=get_env_value("GENERIC_CLIENT_ID"),
        generic_client_secret=get_env_value("GENERIC_CLIENT_SECRET"),
        generic_authorization_endpoint=get_env_value("GENERIC_AUTHORIZATION_ENDPOINT"),
        generic_token_endpoint=get_env_value("GENERIC_TOKEN_ENDPOINT"),
        generic_userinfo_endpoint=get_env_value("GENERIC_USERINFO_ENDPOINT"),
        proxy_base_url=get_env_value("PROXY_BASE_URL"),
        user_email=proxy_admin_email,  # Get from config instead of environment
    )
    
    # Get the schema for UI display
    from pydantic import TypeAdapter
    schema = TypeAdapter(SSOConfig).json_schema(by_alias=True)
    
    # Convert to dict for response
    sso_dict = sso_config.model_dump()
    
    # Add descriptions to the response
    result = {
        "values": sso_dict,
        "field_schema": {"description": schema.get("description", ""), "properties": {}},
    }
    
    # Add property descriptions
    for field_name, field_info in schema["properties"].items():
        result["field_schema"]["properties"][field_name] = {
            "description": field_info.get("description", ""),
            "type": field_info.get("type", "string"),
        }
    
    return result


@router.patch(
    "/update/sso_settings",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_sso_settings(sso_config: SSOConfig):
    """
    Update SSO configuration by saving to both environment variables and config file.
    """
    from litellm.proxy.proxy_server import proxy_config
    import os
    
    # Update environment variables
    env_var_mapping = {
        'google_client_id': 'GOOGLE_CLIENT_ID',
        'google_client_secret': 'GOOGLE_CLIENT_SECRET',
        'microsoft_client_id': 'MICROSOFT_CLIENT_ID',
        'microsoft_client_secret': 'MICROSOFT_CLIENT_SECRET',
        'microsoft_tenant': 'MICROSOFT_TENANT',
        'generic_client_id': 'GENERIC_CLIENT_ID',
        'generic_client_secret': 'GENERIC_CLIENT_SECRET',
        'generic_authorization_endpoint': 'GENERIC_AUTHORIZATION_ENDPOINT',
        'generic_token_endpoint': 'GENERIC_TOKEN_ENDPOINT',
        'generic_userinfo_endpoint': 'GENERIC_USERINFO_ENDPOINT',
        'proxy_base_url': 'PROXY_BASE_URL',
    }
    
    # Load existing config
    config = await proxy_config.get_config()
    
    # Update config with new environment variables
    if "environment_variables" not in config:
        config["environment_variables"] = {}
    
    # Update general_settings for user_email (admin email)
    if "general_settings" not in config:
        config["general_settings"] = {}
    
    # Update environment variables in config and in memory
    sso_data = sso_config.model_dump(exclude_none=True)
    for field_name, value in sso_data.items():
        if field_name == 'user_email' and value is not None:
            # Store user_email in general_settings instead of environment variables
            config["general_settings"]["proxy_admin_email"] = value
        elif field_name in env_var_mapping and value is not None:
            env_var_name = env_var_mapping[field_name]
            # Update in config
            config["environment_variables"][env_var_name] = value
            # Update in runtime environment
            os.environ[env_var_name] = value
    
    # Save the updated config
    await proxy_config.save_config(new_config=config)
    
    return {
        "message": "SSO settings updated successfully",
        "status": "success",
        "settings": sso_data,
    }
