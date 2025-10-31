"""
CACHE SETTINGS MANAGEMENT

Endpoints for managing cache configuration

GET /cache/settings - Get cache configuration including available settings
POST /cache/settings/test - Test cache connection with provided credentials
POST /cache/settings - Save cache settings to database
"""

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.management_endpoints import (
    CACHE_SETTINGS_FIELDS,
    REDIS_TYPE_DESCRIPTIONS,
    CacheSettingsField,
)

router = APIRouter()


class CacheSettingsResponse(BaseModel):
    fields: List[CacheSettingsField] = Field(
        description="List of all configurable cache settings with metadata"
    )
    current_values: Dict[str, Any] = Field(
        description="Current values of cache settings"
    )
    redis_type_descriptions: Dict[str, str] = Field(
        description="Descriptions for each Redis type option"
    )


class CacheTestRequest(BaseModel):
    cache_settings: Dict[str, Any] = Field(
        description="Cache settings to test connection with"
    )


class CacheTestResponse(BaseModel):
    status: str = Field(description="Connection status: 'success' or 'failed'")
    message: str = Field(description="Connection result message")
    error: Optional[str] = Field(default=None, description="Error message if connection failed")


class CacheSettingsUpdateRequest(BaseModel):
    cache_settings: Dict[str, Any] = Field(
        description="Cache settings to save"
    )


@router.get(
    "/cache/settings",
    tags=["Cache Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CacheSettingsResponse,
)
async def get_cache_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get cache configuration and available settings.
    
    Returns:
    - fields: List of all configurable cache settings with their metadata (type, description, default, options)
    - current_values: Current values of cache settings from database
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_config
    
    try:
        # Get cache settings fields from types file
        cache_fields = [field.model_copy(deep=True) for field in CACHE_SETTINGS_FIELDS]
        
        # Try to get cache settings from database
        current_values = {}
        if prisma_client is not None:
            cache_config = await prisma_client.db.litellm_cacheconfig.find_unique(
                where={"id": "cache_config"}
            )
            if cache_config is not None and cache_config.cache_settings:
                # Decrypt cache settings
                cache_settings_json = cache_config.cache_settings
                if isinstance(cache_settings_json, str):
                    cache_settings_dict = json.loads(cache_settings_json)
                else:
                    cache_settings_dict = cache_settings_json
                
                # Decrypt environment variables
                decrypted_settings = proxy_config._decrypt_and_set_db_env_variables(
                    environment_variables=cache_settings_dict,
                    return_original_value=True
                )
                current_values = decrypted_settings
        
        # Update field values with current values
        for field in cache_fields:
            if field.field_name in current_values:
                field.field_value = current_values[field.field_name]
        
        return CacheSettingsResponse(
            fields=cache_fields,
            current_values=current_values,
            redis_type_descriptions=REDIS_TYPE_DESCRIPTIONS,
        )
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error fetching cache settings: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching cache settings: {str(e)}"
        )


@router.post(
    "/cache/settings/test",
    tags=["Cache Settings"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CacheTestResponse,
)
async def test_cache_connection(
    request: CacheTestRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Test cache connection with provided credentials.
    
    This endpoint creates a temporary cache instance, tests the connection,
    and returns the result without persisting the settings.
    """
    from litellm import Cache
    
    try:
        cache_settings = request.cache_settings.copy()
        
        # Ensure type is set to redis
        cache_settings["type"] = "redis"
        
        # Create temporary cache instance
        temp_cache = Cache(**cache_settings)
        
        # Test connection by pinging
        ping_result = await temp_cache.ping()
        
        if ping_result:
            return CacheTestResponse(
                status="success",
                message="Cache connection test successful",
            )
        else:
            return CacheTestResponse(
                status="failed",
                message="Cache ping returned False",
            )
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error testing cache connection: {str(e)}"
        )
        return CacheTestResponse(
            status="failed",
            message=f"Cache connection test failed: {str(e)}",
            error=str(e),
        )


@router.post(
    "/cache/settings",
    tags=["Cache Settings"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_cache_settings(
    request: CacheSettingsUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Save cache settings to database and initialize cache.
    
    This endpoint:
    1. Encrypts sensitive fields (passwords, etc.)
    2. Saves to LiteLLM_CacheConfig table
    3. Reinitializes cache with new settings
    """
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_config,
        store_model_in_db,
    )
    
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Database not connected. Please connect a database."},
        )
    
    if store_model_in_db is not True:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
            },
        )
    
    try:
        cache_settings = request.cache_settings.copy()
        
        # Ensure type is set to redis
        cache_settings["type"] = "redis"
        
        # Encrypt sensitive fields
        encrypted_settings = proxy_config._encrypt_env_variables(
            environment_variables=cache_settings
        )
        
        # Save to database
        await prisma_client.db.litellm_cacheconfig.upsert(
            where={"id": "cache_config"},
            data={
                "create": {
                    "id": "cache_config",
                    "cache_settings": json.dumps(encrypted_settings),
                },
                "update": {
                    "cache_settings": json.dumps(encrypted_settings),
                },
            },
        )
        
        # Reinitialize cache with new settings
        # Decrypt for initialization (proxy_config._init_cache handles decryption internally)
        decrypted_settings = proxy_config._decrypt_and_set_db_env_variables(
            environment_variables=encrypted_settings,
            return_original_value=True
        )
        
        # Ensure type is set to redis
        decrypted_settings["type"] = "redis"
        
        # Initialize cache
        proxy_config._init_cache(cache_params=decrypted_settings)
        
        return {
            "message": "Cache settings updated successfully",
            "status": "success",
            "settings": cache_settings,
        }
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error updating cache settings: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error updating cache settings: {str(e)}"
        )

