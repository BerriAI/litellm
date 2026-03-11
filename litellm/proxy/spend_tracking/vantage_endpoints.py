import json

import litellm
from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.proxy.vantage_endpoints import (
    VantageDryRunRequest,
    VantageExportRequest,
    VantageExportResponse,
    VantageInitRequest,
    VantageInitResponse,
    VantageSettingsUpdate,
    VantageSettingsView,
)

router = APIRouter()

_sensitive_masker = SensitiveDataMasker()

VANTAGE_SETTINGS_PARAM_NAME = "vantage_settings"


def _get_registered_vantage_logger():
    """Return the VantageLogger already registered in litellm.callbacks, if any."""
    from litellm.integrations.vantage.vantage_logger import VantageLogger

    vantage_loggers = litellm.logging_callback_manager.get_custom_loggers_for_type(
        callback_type=VantageLogger
    )
    if vantage_loggers:
        return vantage_loggers[0]
    return None


async def _set_vantage_settings(
    api_key: str, integration_token: str, base_url: str
):
    """Store Vantage settings in the database with encrypted API key."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    encrypted_api_key = encrypt_value_helper(api_key)
    encrypted_integration_token = encrypt_value_helper(integration_token)

    vantage_settings = {
        "api_key": encrypted_api_key,
        "integration_token": encrypted_integration_token,
        "base_url": base_url,
    }

    await prisma_client.db.litellm_config.upsert(
        where={"param_name": VANTAGE_SETTINGS_PARAM_NAME},
        data={
            "create": {
                "param_name": VANTAGE_SETTINGS_PARAM_NAME,
                "param_value": json.dumps(vantage_settings),
            },
            "update": {"param_value": json.dumps(vantage_settings)},
        },
    )


async def _get_vantage_settings():
    """Retrieve Vantage settings from the database with decrypted API key."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    vantage_config = await prisma_client.db.litellm_config.find_first(
        where={"param_name": VANTAGE_SETTINGS_PARAM_NAME}
    )
    if vantage_config is None or vantage_config.param_value is None:
        return {}

    if isinstance(vantage_config.param_value, dict):
        settings = vantage_config.param_value
    elif isinstance(vantage_config.param_value, str):
        settings = json.loads(vantage_config.param_value)
    else:
        settings = dict(vantage_config.param_value)

    encrypted_api_key = settings.get("api_key")
    if encrypted_api_key:
        decrypted_api_key = decrypt_value_helper(
            encrypted_api_key, key="vantage_api_key", exception_type="error"
        )
        if decrypted_api_key is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to decrypt Vantage API key. Check your salt key configuration."
                },
            )
        settings["api_key"] = decrypted_api_key

    encrypted_integration_token = settings.get("integration_token")
    if encrypted_integration_token:
        decrypted_integration_token = decrypt_value_helper(
            encrypted_integration_token,
            key="vantage_integration_token",
            exception_type="error",
        )
        if decrypted_integration_token is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to decrypt Vantage integration token. Check your salt key configuration."
                },
            )
        settings["integration_token"] = decrypted_integration_token

    return settings


@router.get(
    "/vantage/settings",
    tags=["Vantage"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=VantageSettingsView,
)
async def get_vantage_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    View current Vantage settings.

    Returns the current Vantage configuration with the API key masked for security.
    Only admin users can view Vantage settings.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        settings = await _get_vantage_settings()

        if not settings:
            return VantageSettingsView(
                api_key_masked=None,
                integration_token_masked=None,
                base_url=None,
                status=None,
            )

        masked_settings = _sensitive_masker.mask_dict(settings)

        return VantageSettingsView(
            api_key_masked=masked_settings.get("api_key"),
            integration_token_masked=masked_settings.get("integration_token"),
            base_url=settings.get("base_url"),
            status="configured",
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error retrieving Vantage settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to retrieve Vantage settings: {str(e)}"},
        )


@router.put(
    "/vantage/settings",
    tags=["Vantage"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=VantageInitResponse,
)
async def update_vantage_settings(
    request: VantageSettingsUpdate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update existing Vantage settings.

    Allows updating individual Vantage configuration fields without requiring all fields.
    Only admin users can update Vantage settings.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    if not any([request.api_key, request.integration_token, request.base_url]):
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one field must be provided for update"},
        )

    try:
        current_settings = await _get_vantage_settings()

        if not current_settings:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Vantage settings not found. Please initialize settings first using /vantage/init"
                },
            )

        updated_api_key = (
            request.api_key
            if request.api_key is not None
            else current_settings.get("api_key", "")
        )
        updated_token = (
            request.integration_token
            if request.integration_token is not None
            else current_settings.get("integration_token", "")
        )
        updated_base_url = (
            request.base_url
            if request.base_url is not None
            else current_settings.get("base_url", "https://api.vantage.sh")
        )

        await _set_vantage_settings(
            api_key=updated_api_key,
            integration_token=updated_token,
            base_url=updated_base_url,
        )

        verbose_proxy_logger.info("Vantage settings updated successfully")

        return VantageInitResponse(
            message="Vantage settings updated successfully", status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error updating Vantage settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update Vantage settings: {str(e)}"},
        )


async def is_vantage_setup_in_db() -> bool:
    """Check if Vantage is setup in the database."""
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return False

        vantage_config = await prisma_client.db.litellm_config.find_first(
            where={"param_name": VANTAGE_SETTINGS_PARAM_NAME}
        )

        return vantage_config is not None and vantage_config.param_value is not None

    except Exception as e:
        verbose_proxy_logger.error(f"Error checking Vantage status: {str(e)}")
        return False


def is_vantage_setup_in_config() -> bool:
    """Check if Vantage is setup in config.yaml, environment variables, or programmatically."""
    from litellm.integrations.vantage.vantage_logger import VantageLogger

    for cb in litellm.callbacks:
        if cb == "vantage" or isinstance(cb, VantageLogger):
            return True
    return False


async def is_vantage_setup() -> bool:
    """Check if Vantage is setup in either config or database."""
    try:
        if is_vantage_setup_in_config():
            return True
        if await is_vantage_setup_in_db():
            return True
        return False
    except Exception as e:
        verbose_proxy_logger.error(f"Error checking Vantage setup: {str(e)}")
        return False


@router.post(
    "/vantage/init",
    tags=["Vantage"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=VantageInitResponse,
)
async def init_vantage_settings(
    request: VantageInitRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Initialize Vantage settings and store in the database.

    Parameters:
    - api_key: Vantage API key for authentication
    - integration_token: Vantage integration token for the cost-import endpoint
    - base_url: Vantage API base URL (default: https://api.vantage.sh)

    Only admin users can configure Vantage settings.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        await _set_vantage_settings(
            api_key=request.api_key,
            integration_token=request.integration_token,
            base_url=request.base_url,
        )

        verbose_proxy_logger.info("Vantage settings initialized successfully")

        return VantageInitResponse(
            message="Vantage settings initialized successfully", status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error initializing Vantage settings: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to initialize Vantage settings: {str(e)}"},
        )


@router.post(
    "/vantage/dry-run",
    tags=["Vantage"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=VantageExportResponse,
)
async def vantage_dry_run_export(
    request: VantageDryRunRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Perform a dry run export using the Vantage logger.

    Returns the data that would be exported without actually sending it to Vantage.

    Parameters:
    - limit: Limit on number of records to preview (default: 500)

    Only admin users can perform Vantage exports.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        # Dry-run uses the FOCUS database + transformer directly,
        # bypassing the destination so no Vantage credentials are required.
        from litellm.integrations.focus.database import FocusLiteLLMDatabase
        from litellm.integrations.focus.export_engine import FocusExportEngine
        from litellm.integrations.focus.transformer import FocusTransformer

        database = FocusLiteLLMDatabase()
        transformer = FocusTransformer()

        data = await database.get_usage_data(limit=request.limit)
        normalized = transformer.transform(data)

        usage_sample = data.head(min(50, len(data))).to_dicts() if not data.is_empty() else []
        normalized_sample = normalized.head(min(50, len(normalized))).to_dicts() if not normalized.is_empty() else []

        # Use the same pre-transform column names as
        # FocusExportEngine.dry_run_export_usage_data for consistency.
        summary = {
            "total_records": len(normalized),
            "total_spend": FocusExportEngine._sum_column(data, "spend"),
            "total_tokens": FocusExportEngine._sum_column(data, "total_tokens"),
            "unique_teams": FocusExportEngine._count_unique(data, "team_id"),
            "unique_models": FocusExportEngine._count_unique(data, "model"),
        }

        dry_run_result = {
            "usage_data": usage_sample,
            "normalized_data": normalized_sample,
            "summary": summary,
        }

        verbose_proxy_logger.info("Vantage dry run export completed successfully")

        return VantageExportResponse(
            message="Vantage dry run export completed successfully.",
            status="success",
            dry_run_data=dry_run_result,
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error performing Vantage dry run export: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to perform Vantage dry run export: {str(e)}"
            },
        )


@router.post(
    "/vantage/export",
    tags=["Vantage"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=VantageExportResponse,
)
async def vantage_export(
    request: VantageExportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Perform an actual export using the Vantage logger.

    Exports usage data in FOCUS CSV format to the Vantage API.

    Parameters:
    - limit: Optional limit on number of records to export
    - start_time_utc: Optional start time for data export
    - end_time_utc: Optional end time for data export

    Only admin users can perform Vantage exports.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        from litellm.integrations.vantage.vantage_logger import VantageLogger

        # Prefer the already-registered logger to avoid recreating HTTP clients
        # on every export call.
        logger = _get_registered_vantage_logger()
        if logger is None:
            settings = await _get_vantage_settings()
            if not settings:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Vantage settings not found. Please initialize settings first using /vantage/init"
                    },
                )
            logger = VantageLogger(
                api_key=settings.get("api_key"),
                integration_token=settings.get("integration_token"),
                base_url=settings.get("base_url"),
            )
        await logger.export_usage_data(
            limit=request.limit,
            start_time_utc=request.start_time_utc,
            end_time_utc=request.end_time_utc,
        )

        verbose_proxy_logger.info("Vantage export completed successfully")

        return VantageExportResponse(
            message="Vantage export completed successfully",
            status="success",
            dry_run_data=None,
            summary=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error performing Vantage export: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to perform Vantage export: {str(e)}"},
        )


@router.delete(
    "/vantage/delete",
    tags=["Vantage"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=VantageInitResponse,
)
async def delete_vantage_settings(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete Vantage settings from the database.

    Only admin users can delete Vantage settings.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": CommonProxyErrors.not_allowed_access.value},
        )

    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        vantage_config = await prisma_client.db.litellm_config.find_first(
            where={"param_name": VANTAGE_SETTINGS_PARAM_NAME}
        )

        if vantage_config is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "Vantage settings not found"},
            )

        await prisma_client.db.litellm_config.delete(
            where={"param_name": VANTAGE_SETTINGS_PARAM_NAME}
        )

        # Deregister in-memory VantageLogger so the scheduler stops firing
        from litellm.integrations.vantage.vantage_logger import VantageLogger

        litellm.logging_callback_manager.remove_callbacks_by_type(
            litellm.callbacks, VantageLogger
        )

        verbose_proxy_logger.info("Vantage settings deleted successfully")

        return VantageInitResponse(
            message="Vantage settings deleted successfully", status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.error(f"Error deleting Vantage settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to delete Vantage settings: {str(e)}"},
        )
