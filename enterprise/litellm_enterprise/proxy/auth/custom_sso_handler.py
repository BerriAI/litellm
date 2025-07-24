"""
Enterprise Custom SSO Handler for LiteLLM Proxy

This module contains enterprise-specific custom SSO authentication functionality
that allows users to implement their own SSO handling logic by providing custom
handlers that process incoming request headers and return OpenID objects.

Use this when you have an OAuth proxy in front of LiteLLM (where the OAuth proxy
has already authenticated the user) and you need to extract user information from
custom headers or other request attributes.
"""

from typing import TYPE_CHECKING, Dict, Optional, Union, cast

from fastapi import Request
from fastapi.responses import RedirectResponse

if TYPE_CHECKING:
    from fastapi_sso.sso.base import OpenID
else:
    from typing import Any as OpenID

from litellm.proxy.management_endpoints.types import CustomOpenID


class EnterpriseCustomSSOHandler:
    """
    Enterprise Custom SSO Handler for LiteLLM Proxy
    
    This class provides methods for handling custom SSO authentication flows
    where users can implement their own authentication logic by processing
    request headers and returning user information in OpenID format.
    """
    
    @staticmethod
    async def handle_custom_ui_sso_sign_in(
        request: Request,
    ) -> RedirectResponse:
        """
        Allow a user to execute their custom code to parse incoming request headers and return a OpenID object

        Use this when you have an OAuth proxy in front of LiteLLM (where the OAuth proxy has already authenticated the user)
        
        Args:
            request: The FastAPI request object containing headers and other request data
            
        Returns:
            RedirectResponse: Redirect response that sends the user to the LiteLLM UI with authentication token
            
        Raises:
            ValueError: If custom_ui_sso_sign_in_handler is not configured
            
        Example:
            This method is typically called when a user has already been authenticated by an
            external OAuth proxy and the proxy has added custom headers containing user information.
            The custom handler extracts this information and converts it to an OpenID object.
        """
        from fastapi_sso.sso.base import OpenID

        from litellm.integrations.custom_sso_handler import CustomSSOLoginHandler
        from litellm.proxy.proxy_server import (
            CommonProxyErrors,
            premium_user,
            user_custom_ui_sso_sign_in_handler,
        )
        if premium_user is not True:
            raise ValueError(CommonProxyErrors.not_premium_user.value)
        
        if user_custom_ui_sso_sign_in_handler is None:
            raise ValueError("custom_ui_sso_sign_in_handler is not configured. Please set it in general_settings.")
        
        custom_sso_login_handler = cast(CustomSSOLoginHandler, user_custom_ui_sso_sign_in_handler)
        openid_response: OpenID = await custom_sso_login_handler.handle_custom_ui_sso_sign_in(
            request=request,
        )
        
        # Import here to avoid circular imports
        from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler
        
        return await SSOAuthenticationHandler.get_redirect_response_from_openid(
            result=openid_response,
            request=request,
            received_response=None,
            generic_client_id=None,
            ui_access_mode=None,
        ) 