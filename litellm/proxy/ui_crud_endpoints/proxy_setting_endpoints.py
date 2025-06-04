#### CRUD ENDPOINTS for UI Settings #####
from typing import Any, List, Union, Optional

from fastapi import APIRouter, Depends, HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.management_endpoints.ui_sso import DefaultTeamSSOParams
from pydantic import BaseModel, Field

router = APIRouter()


class IPAddress(BaseModel):
    ip: str


class GoogleSSOConfig(BaseModel):
    """Configuration for Google SSO provider"""
    google_client_id: Optional[str] = Field(None, description="Google OAuth Client ID")
    google_client_secret: Optional[str] = Field(None, description="Google OAuth Client Secret")


class MicrosoftSSOConfig(BaseModel):
    """Configuration for Microsoft SSO provider"""
    microsoft_client_id: Optional[str] = Field(None, description="Microsoft OAuth Client ID")
    microsoft_client_secret: Optional[str] = Field(None, description="Microsoft OAuth Client Secret")
    microsoft_tenant: Optional[str] = Field(None, description="Microsoft Tenant ID")


class GenericSSOConfig(BaseModel):
    """Configuration for Generic SSO provider (including Okta)"""
    generic_client_id: Optional[str] = Field(None, description="Generic OAuth Client ID")
    generic_client_secret: Optional[str] = Field(None, description="Generic OAuth Client Secret")
    generic_authorization_endpoint: Optional[str] = Field(None, description="OAuth Authorization Endpoint URL")
    generic_token_endpoint: Optional[str] = Field(None, description="OAuth Token Endpoint URL")
    generic_userinfo_endpoint: Optional[str] = Field(None, description="OAuth UserInfo Endpoint URL")
    generic_scope: Optional[str] = Field("openid email profile", description="OAuth Scope")


class SSOProviderConfig(BaseModel):
    """Complete SSO provider configuration"""
    sso_provider: Optional[str] = Field(None, description="SSO Provider type (google, microsoft, generic, okta)")
    google: Optional[GoogleSSOConfig] = Field(None, description="Google SSO configuration")
    microsoft: Optional[MicrosoftSSOConfig] = Field(None, description="Microsoft SSO configuration") 
    generic: Optional[GenericSSOConfig] = Field(None, description="Generic SSO configuration")
    proxy_base_url: Optional[str] = Field(None, description="Proxy base URL for SSO redirects")
    user_email: Optional[str] = Field(None, description="Admin user email")


class SSOConfigRequest(BaseModel):
    """Request model for SSO configuration"""
    sso_provider: str = Field(..., description="SSO Provider type")
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    microsoft_client_id: Optional[str] = None
    microsoft_client_secret: Optional[str] = None
    microsoft_tenant: Optional[str] = None
    generic_client_id: Optional[str] = None
    generic_client_secret: Optional[str] = None
    generic_authorization_endpoint: Optional[str] = None
    generic_token_endpoint: Optional[str] = None
    generic_userinfo_endpoint: Optional[str] = None
    generic_scope: Optional[str] = "openid email profile"
    proxy_base_url: str = Field(..., description="Proxy base URL")
    user_email: str = Field(..., description="Admin user email")


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
    "/get/sso_provider_config",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_sso_provider_config():
    """
    Get current SSO provider configuration including client IDs and endpoints.
    Returns the current SSO configuration from environment variables and general settings.
    """
    import os
    from litellm.proxy.proxy_server import general_settings
    
    try:
        # Helper function to mask secrets consistently
        def mask_secret(value):
            return "***" if value and value.strip() else None
        
        # Get SSO configuration from environment variables
        config = SSOProviderConfig(
            sso_provider=general_settings.get("sso_provider"),
            google=GoogleSSOConfig(
                google_client_id=os.getenv("GOOGLE_CLIENT_ID"),
                google_client_secret=mask_secret(os.getenv("GOOGLE_CLIENT_SECRET")),
            ),
            microsoft=MicrosoftSSOConfig(
                microsoft_client_id=os.getenv("MICROSOFT_CLIENT_ID"),
                microsoft_client_secret=mask_secret(os.getenv("MICROSOFT_CLIENT_SECRET")),
                microsoft_tenant=os.getenv("MICROSOFT_TENANT"),
            ),
            generic=GenericSSOConfig(
                generic_client_id=os.getenv("GENERIC_CLIENT_ID"),
                generic_client_secret=mask_secret(os.getenv("GENERIC_CLIENT_SECRET")),
                generic_authorization_endpoint=os.getenv("GENERIC_AUTHORIZATION_ENDPOINT"),
                generic_token_endpoint=os.getenv("GENERIC_TOKEN_ENDPOINT"),
                generic_userinfo_endpoint=os.getenv("GENERIC_USERINFO_ENDPOINT"),
                generic_scope=os.getenv("GENERIC_SCOPE", "openid email profile"),
            ),
            proxy_base_url=os.getenv("PROXY_BASE_URL") or general_settings.get("proxy_base_url"),
            user_email=general_settings.get("admin_user_email"),
        )
        
        # Convert to dict and remove None values for cleaner response
        config_dict = config.model_dump(exclude_none=False)
        
        # Clean up provider configs - remove if all values are None
        def is_provider_configured(provider_config):
            if not provider_config:
                return False
            return any(v is not None for v in provider_config.values())
        
        if not is_provider_configured(config_dict.get("google")):
            config_dict["google"] = None
        if not is_provider_configured(config_dict.get("microsoft")):
            config_dict["microsoft"] = None
        if not is_provider_configured(config_dict.get("generic")):
            config_dict["generic"] = None
        
        return {
            "config": config_dict,
            "status": "success"
        }
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting SSO provider config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get SSO provider configuration: {str(e)}"
        )


async def _update_general_settings(config: SSOConfigRequest, general_settings: dict):
    """Update general settings with SSO configuration."""
    general_settings["sso_provider"] = config.sso_provider
    general_settings["proxy_base_url"] = config.proxy_base_url
    general_settings["admin_user_email"] = config.user_email


async def _get_env_updates_for_google(config: SSOConfigRequest, existing_env: dict) -> dict:
    """Get environment variable updates for Google SSO provider."""
    env_updates = {}
    if config.google_client_id:
        env_updates["GOOGLE_CLIENT_ID"] = config.google_client_id
    if config.google_client_secret:
        env_updates["GOOGLE_CLIENT_SECRET"] = config.google_client_secret
    elif "GOOGLE_CLIENT_SECRET" in existing_env:
        env_updates["GOOGLE_CLIENT_SECRET"] = existing_env["GOOGLE_CLIENT_SECRET"]
    return env_updates


async def _get_env_updates_for_microsoft(config: SSOConfigRequest, existing_env: dict) -> dict:
    """Get environment variable updates for Microsoft SSO provider."""
    env_updates = {}
    if config.microsoft_client_id:
        env_updates["MICROSOFT_CLIENT_ID"] = config.microsoft_client_id
    if config.microsoft_client_secret:
        env_updates["MICROSOFT_CLIENT_SECRET"] = config.microsoft_client_secret
    elif "MICROSOFT_CLIENT_SECRET" in existing_env:
        env_updates["MICROSOFT_CLIENT_SECRET"] = existing_env["MICROSOFT_CLIENT_SECRET"]
    if config.microsoft_tenant:
        env_updates["MICROSOFT_TENANT"] = config.microsoft_tenant
    return env_updates


async def _get_env_updates_for_generic(config: SSOConfigRequest, existing_env: dict) -> dict:
    """Get environment variable updates for Generic/Okta SSO provider."""
    env_updates = {}
    if config.generic_client_id:
        env_updates["GENERIC_CLIENT_ID"] = config.generic_client_id
    if config.generic_client_secret:
        env_updates["GENERIC_CLIENT_SECRET"] = config.generic_client_secret
    elif "GENERIC_CLIENT_SECRET" in existing_env:
        env_updates["GENERIC_CLIENT_SECRET"] = existing_env["GENERIC_CLIENT_SECRET"]
    if config.generic_authorization_endpoint:
        env_updates["GENERIC_AUTHORIZATION_ENDPOINT"] = config.generic_authorization_endpoint
    if config.generic_token_endpoint:
        env_updates["GENERIC_TOKEN_ENDPOINT"] = config.generic_token_endpoint
    if config.generic_userinfo_endpoint:
        env_updates["GENERIC_USERINFO_ENDPOINT"] = config.generic_userinfo_endpoint
    if config.generic_scope:
        env_updates["GENERIC_SCOPE"] = config.generic_scope
    return env_updates


async def _get_sso_env_updates(config: SSOConfigRequest, existing_env: dict) -> dict:
    """Get environment variable updates based on SSO provider type."""
    env_updates = {}
    
    if config.sso_provider == "google":
        env_updates.update(await _get_env_updates_for_google(config, existing_env))
    elif config.sso_provider == "microsoft":
        env_updates.update(await _get_env_updates_for_microsoft(config, existing_env))
    elif config.sso_provider in ["generic", "okta"]:
        env_updates.update(await _get_env_updates_for_generic(config, existing_env))
    
    # Update proxy base URL
    if config.proxy_base_url:
        env_updates["PROXY_BASE_URL"] = config.proxy_base_url
    
    return env_updates


async def _update_environment_variables(env_updates: dict):
    """Update environment variables in memory."""
    import os
    for key, value in env_updates.items():
        os.environ[key] = value


async def _save_sso_config_to_file(config: SSOConfigRequest, env_updates: dict, proxy_config):
    """Save SSO configuration to config file."""
    proxy_config_data = await proxy_config.get_config()
    if "general_settings" not in proxy_config_data:
        proxy_config_data["general_settings"] = {}
    
    proxy_config_data["general_settings"]["sso_provider"] = config.sso_provider
    proxy_config_data["general_settings"]["proxy_base_url"] = config.proxy_base_url
    proxy_config_data["general_settings"]["admin_user_email"] = config.user_email
    
    # Store environment variables in config for persistence
    if "environment_variables" not in proxy_config_data:
        proxy_config_data["environment_variables"] = {}
    
    for key, value in env_updates.items():
        proxy_config_data["environment_variables"][key] = value
    
    await proxy_config.save_config(new_config=proxy_config_data)


@router.post(
    "/update/sso_provider_config",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_sso_provider_config(config: SSOConfigRequest):
    """
    Update SSO provider configuration. This will update environment variables
    and general settings based on the selected provider.
    
    Note: Some changes may require a server restart to take effect.
    """
    from litellm.proxy.proxy_server import general_settings, proxy_config
    
    try:
        # Update general settings
        await _update_general_settings(config, general_settings)
        
        # Get existing configuration to preserve secrets that weren't updated
        existing_config = await proxy_config.get_config()
        existing_env = existing_config.get("environment_variables", {})
        
        # Get environment variable updates based on provider
        env_updates = await _get_sso_env_updates(config, existing_env)
        
        # Update environment variables in memory
        await _update_environment_variables(env_updates)
        
        # Save configuration to config file
        await _save_sso_config_to_file(config, env_updates, proxy_config)
        
        return {
            "message": "SSO provider configuration updated successfully",
            "status": "success",
            "provider": config.sso_provider,
            "note": "Some changes may require a server restart to take full effect"
        }
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error updating SSO provider config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update SSO provider configuration: {str(e)}"
        )


@router.delete(
    "/delete/sso_provider_config",
    tags=["SSO Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_sso_provider_config():
    """
    Delete/reset SSO provider configuration. This will remove SSO environment
    variables and reset the configuration to defaults.
    """
    import os
    from litellm.proxy.proxy_server import general_settings, proxy_config
    
    try:
        # List of SSO environment variables to remove
        sso_env_vars = [
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CLIENT_SECRET", 
            "MICROSOFT_CLIENT_ID",
            "MICROSOFT_CLIENT_SECRET",
            "MICROSOFT_TENANT",
            "GENERIC_CLIENT_ID",
            "GENERIC_CLIENT_SECRET",
            "GENERIC_AUTHORIZATION_ENDPOINT",
            "GENERIC_TOKEN_ENDPOINT",
            "GENERIC_USERINFO_ENDPOINT",
            "GENERIC_SCOPE",
        ]
        
        # Remove from environment
        for var in sso_env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Reset general settings
        general_settings.pop("sso_provider", None)
        general_settings.pop("admin_user_email", None)
        
        # Update config file
        proxy_config_data = await proxy_config.get_config()
        
        # Remove from general_settings
        if "general_settings" in proxy_config_data:
            proxy_config_data["general_settings"].pop("sso_provider", None)
            proxy_config_data["general_settings"].pop("admin_user_email", None)
        
        # Remove from environment section
        if "environment_variables" in proxy_config_data:
            for var in sso_env_vars:
                proxy_config_data["environment_variables"].pop(var, None)
        
        await proxy_config.save_config(new_config=proxy_config_data)
        
        return {
            "message": "SSO provider configuration deleted successfully",
            "status": "success",
            "note": "Server restart may be required for changes to take full effect"
        }
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error deleting SSO provider config: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete SSO provider configuration: {str(e)}"
        )
