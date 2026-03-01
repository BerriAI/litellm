from fastapi import APIRouter, Depends, HTTPException, Query
from litellm.proxy._types import *
from litellm.proxy._types import hash_token
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.post("/jwt/key/mapping/new", tags=["JWT Key Mapping"])
async def create_jwt_key_mapping(
    data: CreateJWTKeyMappingRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403, detail="Only proxy admins can create JWT key mappings"
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        hashed_key = hash_token(data.key)
        create_data = {
            "jwt_claim_name": data.jwt_claim_name,
            "jwt_claim_value": data.jwt_claim_value,
            "token": hashed_key,
            "created_by": user_api_key_dict.user_id,
            "updated_by": user_api_key_dict.user_id,
        }
        if data.description is not None:
            create_data["description"] = data.description

        new_mapping = await prisma_client.db.litellm_jwtkeymapping.create(
            data=create_data
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
        raise HTTPException(
            status_code=403, detail="Only proxy admins can update JWT key mappings"
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    update_data = data.model_dump(exclude_unset=True, exclude={"id", "key"})
    if data.key is not None:
        update_data["token"] = hash_token(data.key)
    update_data["updated_by"] = user_api_key_dict.user_id

    try:
        # Get old mapping for cache invalidation
        old_mapping = await prisma_client.db.litellm_jwtkeymapping.find_unique(
            where={"id": data.id}
        )

        if old_mapping:
            cache_key = f"jwt_key_mapping:{old_mapping.jwt_claim_name}:{old_mapping.jwt_claim_value}"
            await user_api_key_cache.async_delete_cache(cache_key)

        updated_mapping = await prisma_client.db.litellm_jwtkeymapping.update(
            where={"id": data.id}, data=update_data
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
        raise HTTPException(
            status_code=403, detail="Only proxy admins can delete JWT key mappings"
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Get old mapping for cache invalidation
        old_mapping = await prisma_client.db.litellm_jwtkeymapping.find_unique(
            where={"id": data.id}
        )

        if old_mapping:
            cache_key = f"jwt_key_mapping:{old_mapping.jwt_claim_name}:{old_mapping.jwt_claim_value}"
            await user_api_key_cache.async_delete_cache(cache_key)

        await prisma_client.db.litellm_jwtkeymapping.delete(
            where={"id": data.id}
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jwt/key/mapping/list", tags=["JWT Key Mapping"])
async def list_jwt_key_mappings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    page: int = Query(1, description="Page number", ge=1),
    size: int = Query(50, description="Page size", ge=1, le=100),
):
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403, detail="Only proxy admins can list JWT key mappings"
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        skip = (page - 1) * size
        mappings = await prisma_client.db.litellm_jwtkeymapping.find_many(
            skip=skip,
            take=size,
            order={"created_at": "desc"},
        )
        total_count = await prisma_client.db.litellm_jwtkeymapping.count()
        return {
            "mappings": mappings,
            "total_count": total_count,
            "current_page": page,
            "total_pages": -(-total_count // size),  # ceiling division
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jwt/key/mapping/info", tags=["JWT Key Mapping"])
async def info_jwt_key_mapping(
    id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403, detail="Only proxy admins can get JWT key mapping info"
        )

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        mapping = await prisma_client.db.litellm_jwtkeymapping.find_unique(
            where={"id": id}
        )
        if mapping is None:
            raise HTTPException(status_code=404, detail="Mapping not found")
        return mapping
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
