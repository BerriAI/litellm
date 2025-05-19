import asyncio
import copy

from fastapi import APIRouter, Depends, HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
from litellm.proxy.proxy_server import (
    LITELLM_PROXY_ADMIN_NAME,
    LitellmTableNames,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    prisma_client,
    store_model_in_db,
)
from litellm.proxy.types import UserAPIKeyAuth
from litellm.types.proxy.management_endpoints.router_management_endpoints import (
    GetRouterSettingsResponse,
    PatchRouterSettingsRequest,
)

router = APIRouter()


@router.patch(
    "/router_settings",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_router_settings(
    router_settings: PatchRouterSettingsRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update router settings in the database.
    Only accessible by proxy admin users.
    """
    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )

        if store_model_in_db is not True:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

        # Only allow proxy admin to update router settings
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=403,
                detail={"error": "Only proxy admin users can update router settings"},
            )

        existing_router_settings = (
            await prisma_client.db.litellm_config.find_unique(
                where={"param_name": "router_settings"}
            )
            or {}
        )

        # new router settings
        new_router_settings = copy.deepcopy(existing_router_settings)

        # update new router settings with request body
        new_router_settings.update(router_settings.model_dump(exclude_none=True))

        # Update router settings in DB
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "router_settings"},
            data={
                "create": {
                    "param_name": "router_settings",
                    "param_value": new_router_settings,
                    "updated_by": user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
                },
                "update": {
                    "param_value": new_router_settings,
                    "updated_by": user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
                },
            },
        )

        ## CREATE AUDIT LOG ##
        asyncio.create_task(
            create_object_audit_log(
                object_id="router_settings",
                action="updated",
                user_api_key_dict=user_api_key_dict,
                table_name=LitellmTableNames.CONFIG_TABLE_NAME,
                before_value=safe_dumps(existing_router_settings),
                after_value=safe_dumps(new_router_settings),
                litellm_changed_by=user_api_key_dict.user_id,
                litellm_proxy_admin_name=LITELLM_PROXY_ADMIN_NAME,
            )
        )

        return {"message": "Router settings updated successfully"}

    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating router settings: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise ProxyException(
            message=f"Error updating router settings: {str(e)}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )


@router.get(
    "/router_settings",
    tags=["model management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=GetRouterSettingsResponse,
)
async def get_router_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get current router settings from the database.
    Only accessible by proxy admin users.
    """
    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "No DB Connected. Here's how to do it - https://docs.litellm.ai/docs/proxy/virtual_keys"
                },
            )

        if store_model_in_db is not True:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature."
                },
            )

        # Only allow proxy admin to view router settings
        if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
            raise HTTPException(
                status_code=403,
                detail={"error": "Only proxy admin users can view router settings"},
            )

        # Get router settings from DB
        router_settings = await prisma_client.db.litellm_config.find_unique(
            where={"param_name": "router_settings"}
        )

        if router_settings is None:
            return {"router_settings": {}}

        return GetRouterSettingsResponse(**router_settings.param_value)

    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting router settings: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise ProxyException(
            message=f"Error getting router settings: {str(e)}",
            type=ProxyErrorTypes.internal_server_error,
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            param=None,
        )
