"""
License-related endpoints for viewing enterprise license information.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from fastapi import HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from litellm.proxy.utils import (
    handle_exception_on_proxy,
)
router = APIRouter()


class LicenseInfoResponse(BaseModel):
    """Response model for license info endpoint"""
    license_configured: bool = Field(
        False,
        description="Has the proxy been configured with a license key",
    )
    license_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="License details from EnterpriseLicenseData",
    )

@router.get(
    "/license/info",
    tags=["license management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LicenseInfoResponse,
)
async def get_license_info(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> LicenseInfoResponse:
    """
    Get license information.
    
    Returns the license configuration status and details for the deployment.
    """

    try:
        # RBAC check - only PROXY_ADMIN and PROXY_ADMIN_VIEW_ONLY can access
        if user_api_key_dict.user_role not in [
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        ]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": f"Only proxy admin users can access license information. Your role: {user_api_key_dict.user_role}"
                },
            )

        from litellm.proxy.proxy_server import premium_user_data

        verbose_proxy_logger.debug(
            "litellm.proxy.management_endpoints.license_management_endpoints.get_license_info() - "
            "Retrieving license information"
        )

        if premium_user_data is None:
            verbose_proxy_logger.debug(
                "No license configured - returning empty license_details"
            )
            return LicenseInfoResponse(
                license_configured=False,
                license_details={},
                error=None,
            )

        license_details_dict = dict(premium_user_data)
        verbose_proxy_logger.debug(
            f"License configured - returning license details: "
            f"license_id={premium_user_data.get('user_id')}"
        )

        return LicenseInfoResponse(
            license_configured=True,
            license_details=license_details_dict,
            error=None,
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like RBAC checks)
        raise
    except Exception as e:
        handle_exception_on_proxy(e)
