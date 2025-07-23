import json

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.cloudzero_endpoints import (
    CloudZeroExportRequest,
    CloudZeroExportResponse,
    CloudZeroInitRequest,
    CloudZeroInitResponse,
)

router = APIRouter()


@router.post(
    "/cloudzero/init",
    tags=["CloudZero"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CloudZeroInitResponse,
)
async def init_cloudzero_settings(
    request: CloudZeroInitRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Initialize CloudZero settings and store in the database.
    
    This endpoint stores the CloudZero API key, connection ID, and timezone configuration
    in the proxy database for use by the CloudZero logger.
    
    Parameters:
    - api_key: CloudZero API key for authentication
    - connection_id: CloudZero connection ID for data submission  
    - timezone: Timezone for date handling (default: UTC)
    
    Only admin users can configure CloudZero settings.
    """
    from litellm.proxy.proxy_server import prisma_client

    # Validation
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )
    
    try:
        # Prepare CloudZero settings
        cloudzero_settings = {
            "api_key": request.api_key,
            "connection_id": request.connection_id,
            "timezone": request.timezone,
        }
        
        # Store in database  
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "cloudzero_settings"},
            data={
                "create": {
                    "param_name": "cloudzero_settings",
                    "param_value": json.dumps(cloudzero_settings),
                },
                "update": {"param_value": json.dumps(cloudzero_settings)},
            },
        )
        
        verbose_proxy_logger.info("CloudZero settings initialized successfully")
        
        return CloudZeroInitResponse(
            message="CloudZero settings initialized successfully",
            status="success"
        )
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error initializing CloudZero settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to initialize CloudZero settings: {str(e)}"}
        )


@router.post(
    "/cloudzero/dry-run",
    tags=["CloudZero"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CloudZeroExportResponse,
)
async def cloudzero_dry_run_export(
    request: CloudZeroExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Perform a dry run export using the CloudZero logger.
    
    This endpoint uses the CloudZero logger to perform a dry run export,
    which displays the data that would be exported without actually sending it to CloudZero.
    
    Parameters:
    - limit: Optional limit on number of records to process (default: 10000)
    
    Only admin users can perform CloudZero exports.
    """
    from litellm.proxy.proxy_server import prisma_client

    # Validation
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )
    
    try:
        # Import and initialize CloudZero logger with credentials
        from litellm.integrations.cloudzero.ll2cz.cloudzero import CloudZeroLogger

        # Initialize logger with credentials directly
        logger = CloudZeroLogger()
        await logger.dry_run_export_usage_data(limit=request.limit)
        
        verbose_proxy_logger.info("CloudZero dry run export completed successfully")
        
        return CloudZeroExportResponse(
            message="CloudZero dry run export completed successfully. Check logs for output.",
            status="success"
        )
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error performing CloudZero dry run export: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform CloudZero dry run export: {str(e)}"}
        )


@router.post(
    "/cloudzero/export",
    tags=["CloudZero"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CloudZeroExportResponse,
)
async def cloudzero_export(
    request: CloudZeroExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Perform an actual export using the CloudZero logger.
    
    This endpoint uses the CloudZero logger to export usage data to CloudZero AnyCost API.
    
    Parameters:
    - limit: Optional limit on number of records to export
    - operation: CloudZero operation type ("replace_hourly" or "sum", default: "replace_hourly")
    
    Only admin users can perform CloudZero exports.
    """
    from litellm.proxy.proxy_server import prisma_client

    # Validation
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )
    
    try:
        # Get CloudZero settings from database
        cloudzero_config = await prisma_client.db.litellm_config.find_first(
            where={"param_name": "cloudzero_settings"}
        )
        
        if not cloudzero_config or not cloudzero_config.param_value:
            raise HTTPException(
                status_code=400,
                detail={"error": "CloudZero settings not configured. Please run /cloudzero/init first."}
            )
        
        settings = dict(cloudzero_config.param_value)
        
        # Import and initialize CloudZero logger with credentials
        from litellm.integrations.cloudzero.ll2cz.cloudzero import CloudZeroLogger

        # Initialize logger with credentials directly
        logger = CloudZeroLogger(
            api_key=settings["api_key"],
            connection_id=settings["connection_id"],
            timezone=settings["timezone"]
        )
        await logger.export_usage_data(limit=request.limit, operation=request.operation)
        
        verbose_proxy_logger.info("CloudZero export completed successfully")
        
        return CloudZeroExportResponse(
            message="CloudZero export completed successfully",
            status="success"
        )
        
    except Exception as e:
        verbose_proxy_logger.error(f"Error performing CloudZero export: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform CloudZero export: {str(e)}"}
        )


