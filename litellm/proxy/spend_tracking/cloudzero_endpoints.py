import json

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.proxy.cloudzero_endpoints import (
    CloudZeroExportRequest,
    CloudZeroExportResponse,
    CloudZeroInitRequest,
    CloudZeroInitResponse,
    CloudZeroSettingsUpdate,
    CloudZeroSettingsView,
)

router = APIRouter()


# Initialize the sensitive data masker for API key masking
_sensitive_masker = SensitiveDataMasker()


async def _set_cloudzero_settings(api_key: str, connection_id: str, timezone: str):
    """
    Store CloudZero settings in the database with encrypted API key.

    Args:
        api_key: CloudZero API key to encrypt and store
        connection_id: CloudZero connection ID
        timezone: Timezone for date handling
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Encrypt the API key before storing
    encrypted_api_key = encrypt_value_helper(api_key)

    cloudzero_settings = {
        "api_key": encrypted_api_key,
        "connection_id": connection_id,
        "timezone": timezone,
    }

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


async def _get_cloudzero_settings():
    """
    Retrieve CloudZero settings from the database with decrypted API key.

    Returns:
        dict: CloudZero settings with decrypted API key
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    cloudzero_config = await prisma_client.db.litellm_config.find_first(
        where={"param_name": "cloudzero_settings"}
    )
    if cloudzero_config is None:
        return {}

    settings = dict(cloudzero_config.param_value)

    # Decrypt the API key
    encrypted_api_key = settings.get("api_key")
    if encrypted_api_key:
        decrypted_api_key = decrypt_value_helper(
            encrypted_api_key, key="cloudzero_api_key", exception_type="error"
        )
        if decrypted_api_key is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to decrypt CloudZero API key. Check your salt key configuration."
                },
            )
        settings["api_key"] = decrypted_api_key

    return settings


@router.get(
    "/cloudzero/settings",
    tags=["CloudZero"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CloudZeroSettingsView,
)
async def get_cloudzero_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    View current CloudZero settings.

    Returns the current CloudZero configuration with the API key masked for security.
    Only the first 4 and last 4 characters of the API key are shown.

    Only admin users can view CloudZero settings.
    """
    # Validation
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        # Get CloudZero settings using the accessor method
        settings = await _get_cloudzero_settings()

        # Use SensitiveDataMasker to mask the API key
        masked_settings = _sensitive_masker.mask_dict(settings)

        return CloudZeroSettingsView(
            api_key_masked=masked_settings["api_key"],
            connection_id=settings["connection_id"],
            timezone=settings["timezone"],
            status="configured",
        )

    except HTTPException as e:
        if e.status_code == 400:
            # Settings not configured
            raise HTTPException(
                status_code=404, detail={"error": "CloudZero settings not configured"}
            )
        raise e
    except Exception as e:
        verbose_proxy_logger.error(f"Error retrieving CloudZero settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to retrieve CloudZero settings: {str(e)}"},
        )


@router.put(
    "/cloudzero/settings",
    tags=["CloudZero"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CloudZeroInitResponse,
)
async def update_cloudzero_settings(
    request: CloudZeroSettingsUpdate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update existing CloudZero settings.

    Allows updating individual CloudZero configuration fields without requiring all fields.
    Only provided fields will be updated; others will remain unchanged.

    Parameters:
    - api_key: (Optional) New CloudZero API key for authentication
    - connection_id: (Optional) New CloudZero connection ID for data submission
    - timezone: (Optional) New timezone for date handling

    Only admin users can update CloudZero settings.
    """
    # Validation
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    # Check if at least one field is provided
    if not any([request.api_key, request.connection_id, request.timezone]):
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one field must be provided for update"},
        )

    try:
        # Get current settings
        current_settings = await _get_cloudzero_settings()

        # Update only provided fields
        updated_api_key = (
            request.api_key
            if request.api_key is not None
            else current_settings["api_key"]
        )
        updated_connection_id = (
            request.connection_id
            if request.connection_id is not None
            else current_settings["connection_id"]
        )
        updated_timezone = (
            request.timezone
            if request.timezone is not None
            else current_settings["timezone"]
        )

        # Store updated settings using the setter method with encryption
        await _set_cloudzero_settings(
            api_key=updated_api_key,
            connection_id=updated_connection_id,
            timezone=updated_timezone,
        )

        verbose_proxy_logger.info("CloudZero settings updated successfully")

        return CloudZeroInitResponse(
            message="CloudZero settings updated successfully", status="success"
        )

    except HTTPException as e:
        if e.status_code == 400:
            # Settings not configured yet
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CloudZero settings not found. Please initialize settings first using /cloudzero/init"
                },
            )
        raise e
    except Exception as e:
        verbose_proxy_logger.error(f"Error updating CloudZero settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update CloudZero settings: {str(e)}"},
        )


# Global variable to track if CloudZero background job has been initialized
_cloudzero_background_job_initialized = False


async def is_cloudzero_setup_in_db() -> bool:
    """
    Check if CloudZero is setup in the database.

    CloudZero is considered setup in the database if:
    - CloudZero settings exist in the database
    - The settings have a non-None value

    Returns:
        bool: True if CloudZero is active, False otherwise
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return False

        # Check for CloudZero settings in database
        cloudzero_config = await prisma_client.db.litellm_config.find_first(
            where={"param_name": "cloudzero_settings"}
        )

        # CloudZero is setup in the database if config exists and has non-None value
        return cloudzero_config is not None and cloudzero_config.param_value is not None

    except Exception as e:
        verbose_proxy_logger.error(f"Error checking CloudZero status: {str(e)}")
        return False


def is_cloudzero_setup_in_config() -> bool:
    """
    Check if CloudZero is setup in config.yaml or environment variables.

    CloudZero is considered setup in config if:
    - "cloudzero" is in the callbacks list in config.yaml, OR
    Returns:
        bool: True if CloudZero is configured, False otherwise
    """
    import litellm
    return "cloudzero" in litellm.callbacks


async def is_cloudzero_setup() -> bool:
    """
    Check if CloudZero is setup in either config.yaml/env vars OR database.

    CloudZero is considered setup if:
    - CloudZero is configured in config.yaml callbacks, OR
    - CloudZero environment variables are set, OR  
    - CloudZero settings exist in the database

    Returns:
        bool: True if CloudZero is configured anywhere, False otherwise
    """
    try:
        # Check config.yaml/environment variables first
        if is_cloudzero_setup_in_config():
            return True
            
        # Check database as fallback
        if await is_cloudzero_setup_in_db():
            return True
            
        return False

    except Exception as e:
        verbose_proxy_logger.error(f"Error checking CloudZero setup: {str(e)}")
        return False


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
    # Validation
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        # Store settings using the setter method with encryption
        await _set_cloudzero_settings(
            api_key=request.api_key,
            connection_id=request.connection_id,
            timezone=request.timezone,
        )

        verbose_proxy_logger.info("CloudZero settings initialized successfully")

        return CloudZeroInitResponse(
            message="CloudZero settings initialized successfully", status="success"
        )

    except Exception as e:
        verbose_proxy_logger.error(f"Error initializing CloudZero settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to initialize CloudZero settings: {str(e)}"},
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
    which returns the data that would be exported without actually sending it to CloudZero.

    Parameters:
    - limit: Optional limit on number of records to process (default: 10000)

    Returns:
    - usage_data: Sample of the raw usage data (first 50 records)
    - cbf_data: CloudZero CBF formatted data ready for export
    - summary: Statistics including total cost, tokens, and record counts

    Only admin users can perform CloudZero exports.
    """
    # Validation
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        # Import and initialize CloudZero logger with credentials
        from litellm.integrations.cloudzero.cloudzero import CloudZeroLogger

        # Initialize logger with credentials directly
        logger = CloudZeroLogger()
        dry_run_result = await logger.dry_run_export_usage_data(
            limit=request.limit
        )

        verbose_proxy_logger.info("CloudZero dry run export completed successfully")

        return CloudZeroExportResponse(
            message="CloudZero dry run export completed successfully.",
            status="success",
            dry_run_data=dry_run_result,
            summary=dry_run_result.get("summary") if dry_run_result else None,
        )

    except Exception as e:
        verbose_proxy_logger.error(
            f"Error performing CloudZero dry run export: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform CloudZero dry run export: {str(e)}"},
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


    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        # Get CloudZero settings using the accessor method with decryption
        settings = await _get_cloudzero_settings()

        # Import and initialize CloudZero logger with credentials
        from litellm.integrations.cloudzero.cloudzero import CloudZeroLogger

        # Initialize logger with credentials directly
        logger = CloudZeroLogger(
            api_key=settings.get("api_key"),
            connection_id=settings.get("connection_id"),
            timezone=settings.get("timezone"),
        )
        await logger.export_usage_data(
            limit=request.limit,
            operation=request.operation,
            start_time_utc=request.start_time_utc,
            end_time_utc=request.end_time_utc,
        )

        verbose_proxy_logger.info("CloudZero export completed successfully")

        return CloudZeroExportResponse(
            message="CloudZero export completed successfully", 
            status="success",
            dry_run_data=None,
            summary=None
        )

    except Exception as e:
        verbose_proxy_logger.error(f"Error performing CloudZero export: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform CloudZero export: {str(e)}"},
        )
