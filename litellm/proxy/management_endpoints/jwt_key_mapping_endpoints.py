import asyncio
from typing import List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Request
import litellm
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.auth.auth_checks import _delete_cache_key_object

router = APIRouter()

@router.post("/jwt/key/mapping/new", tags=["JWT Key Mapping"])
async def create_jwt_key_mapping(
    data: CreateJWTKeyMappingRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can create JWT key mappings")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        new_mapping = await prisma_client.db.litellm_jwtkeymapping.create(
            data={
                "jwt_claim_name": data.jwt_claim_name,
                "jwt_claim_value": data.jwt_claim_value,
                "token": data.token,
                "is_active": data.is_active,
            }
        )
        
        # Invalidate cache
        cache_key = f"jwt_key_mapping:{data.jwt_claim_name}:{data.jwt_claim_value}"
        await user_api_key_cache.async_delete_cache(cache_key)

        return new_mapping
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jwt/key/mapping/update", tags=["JWT Key Mapping"])
async def update_jwt_key_mapping(
    data: UpdateJWTKeyMappingRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can update JWT key mappings")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    update_data = data.model_dump(exclude_unset=True, exclude={"mapping_id"})
    
    try:
        # Get old mapping for cache invalidation
        old_mapping = await prisma_client.db.litellm_jwtkeymapping.find_unique(
            where={"mapping_id": data.mapping_id}
        )
        
        if old_mapping:
            cache_key = f"jwt_key_mapping:{old_mapping.jwt_claim_name}:{old_mapping.jwt_claim_value}"
            await user_api_key_cache.async_delete_cache(cache_key)

        updated_mapping = await prisma_client.db.litellm_jwtkeymapping.update(
            where={"mapping_id": data.mapping_id},
            data=update_data
        )
        
        # Invalidate new cache key if claim fields changed
        cache_key = f"jwt_key_mapping:{updated_mapping.jwt_claim_name}:{updated_mapping.jwt_claim_value}"
        await user_api_key_cache.async_delete_cache(cache_key)

        return updated_mapping
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jwt/key/mapping/delete", tags=["JWT Key Mapping"])
async def delete_jwt_key_mapping(
    data: DeleteJWTKeyMappingRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can delete JWT key mappings")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Get old mapping for cache invalidation
        old_mapping = await prisma_client.db.litellm_jwtkeymapping.find_unique(
            where={"mapping_id": data.mapping_id}
        )
        
        if old_mapping:
            cache_key = f"jwt_key_mapping:{old_mapping.jwt_claim_name}:{old_mapping.jwt_claim_value}"
            await user_api_key_cache.async_delete_cache(cache_key)

        await prisma_client.db.litellm_jwtkeymapping.delete(
            where={"mapping_id": data.mapping_id}
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jwt/key/mapping/list", tags=["JWT Key Mapping"])
async def list_jwt_key_mappings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can list JWT key mappings")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        mappings = await prisma_client.db.litellm_jwtkeymapping.find_many()
        return mappings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jwt/key/mapping/info", tags=["JWT Key Mapping"])
async def info_jwt_key_mapping(
    mapping_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can get JWT key mapping info")

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        mapping = await prisma_client.db.litellm_jwtkeymapping.find_unique(
            where={"mapping_id": mapping_id}
        )
        if mapping is None:
            raise HTTPException(status_code=404, detail="Mapping not found")
        return mapping
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
