import os
from typing import Optional

from fastapi import Request

from litellm.proxy._types import LitellmUserRoles


def check_is_admin_only_access(ui_access_mode: str) -> bool:
    """Checks ui access mode is admin_only"""
    return ui_access_mode == "admin_only"


def has_admin_ui_access(user_role: str) -> bool:
    """
    Check if the user has admin access to the UI.

    Returns:
        bool: True if user is 'proxy_admin' or 'proxy_admin_view_only', False otherwise.
    """

    if (
        user_role != LitellmUserRoles.PROXY_ADMIN.value
        and user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value
    ):
        return False
    return True


class LitellmUILoginUtils:
    """
    Helper utils for LiteLLM UI Login

    - Decider method for using SSO or regular UI login
    - Get redirect URL for SSO
    """

    @staticmethod
    def should_use_sso_handler(
        google_client_id: Optional[str] = None,
        microsoft_client_id: Optional[str] = None,
        generic_client_id: Optional[str] = None,
    ) -> bool:
        if (
            google_client_id is not None
            or microsoft_client_id is not None
            or generic_client_id is not None
        ):
            return True
        return False

    @staticmethod
    def get_redirect_url_for_sso(
        request: Request,
        sso_callback_route: str,
    ) -> str:
        """
        Get the redirect URL for SSO
        """
        redirect_url = os.getenv("PROXY_BASE_URL", str(request.base_url))
        if redirect_url.endswith("/"):
            redirect_url += sso_callback_route
        else:
            redirect_url += "/" + sso_callback_route
        return redirect_url
