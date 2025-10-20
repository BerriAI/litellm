"""
SAML 2.0 Authentication Handler

This module provides SAML 2.0 SSO authentication capabilities for LiteLLM Proxy.
It follows the same patterns as the existing OAuth/OIDC handlers.

Flow:
1. User lands on Admin UI
2. LiteLLM redirects user to SAML IdP for authentication
3. IdP authenticates user and redirects back with SAML assertion
4. LiteLLM validates the SAML assertion and extracts user information
5. User is signed in to UI with JWT token

"""

import os
from typing import Dict, Optional

from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.management_endpoints.types import CustomOpenID


class SAMLAuthenticationHandler:
    """
    Handler for SAML 2.0 Authentication

    Provides methods to:
    - Build SAML settings from environment variables
    - Generate authentication requests
    - Process SAML assertions
    - Extract user attributes
    """

    @staticmethod
    def _get_saml_settings(request_data: Dict) -> Dict:
        """
        Build SAML settings dictionary from environment variables.

        Args:
            request_data: Dictionary containing HTTP request information (host, scheme, etc.)

        Returns:
            Dictionary with SAML settings for python3-saml library

        Raises:
            ProxyException: If required SAML configuration is missing
        """

        # required env variables for SAML
        saml_idp_entity_id = os.getenv("SAML_IDP_ENTITY_ID")
        saml_idp_sso_url = os.getenv("SAML_IDP_SSO_URL")
        saml_idp_x509_cert = os.getenv("SAML_IDP_X509_CERT")
        proxy_base_url = os.getenv("PROXY_BASE_URL")

        # Validate required settings
        if not saml_idp_entity_id:
            raise ProxyException(
                message="SAML_IDP_ENTITY_ID not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="SAML_IDP_ENTITY_ID",
                code=500,
            )
        if not saml_idp_sso_url:
            raise ProxyException(
                message="SAML_IDP_SSO_URL not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="SAML_IDP_SSO_URL",
                code=500,
            )
        if not saml_idp_x509_cert:
            raise ProxyException(
                message="SAML_IDP_X509_CERT not set. Set it in .env file",
                type=ProxyErrorTypes.auth_error,
                param="SAML_IDP_X509_CERT",
                code=500,
            )
        if not proxy_base_url:
            raise ProxyException(
                message="PROXY_BASE_URL not set. Required for SAML SSO redirects",
                type=ProxyErrorTypes.auth_error,
                param="PROXY_BASE_URL",
                code=500,
            )

        # Optional settings with defaults
        saml_entity_id = os.getenv(
            "SAML_ENTITY_ID", f"{proxy_base_url}/sso/saml/metadata"
        )
        saml_name_id_format = os.getenv(
            "SAML_NAME_ID_FORMAT",
            "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
        )

        # optional service provider (SP) certificate and key for signing/encryption
        saml_sp_x509_cert = os.getenv("SAML_SP_X509_CERT", "")
        saml_sp_private_key = os.getenv("SAML_SP_PRIVATE_KEY", "")

        # Build callback URL - allow customization via env var
        acs_path = os.getenv("SAML_ACS_PATH", "/sso/saml/acs")
        acs_url = f"{proxy_base_url}{acs_path}"

        # Build SAML settings dictionary
        settings = {
            "strict": True,
            "debug": os.getenv("SAML_DEBUG", "false").lower() == "true",
            "sp": {
                "entityId": saml_entity_id,
                "assertionConsumerService": {
                    "url": acs_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
                "singleLogoutService": {
                    "url": f"{proxy_base_url}/sso/saml/sls",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "NameIDFormat": saml_name_id_format,
                "x509cert": saml_sp_x509_cert,
                "privateKey": saml_sp_private_key,
            },
            "idp": {
                "entityId": saml_idp_entity_id,
                "singleSignOnService": {
                    "url": saml_idp_sso_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "singleLogoutService": {
                    "url": os.getenv("SAML_IDP_SLO_URL", ""),
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "x509cert": saml_idp_x509_cert,
            },
            "security": {
                "nameIdEncrypted": False,
                "authnRequestsSigned": bool(saml_sp_private_key),
                "logoutRequestSigned": bool(saml_sp_private_key),
                "logoutResponseSigned": bool(saml_sp_private_key),
                "signMetadata": bool(saml_sp_private_key),
                "wantMessagesSigned": False,
                "wantAssertionsSigned": True,
                "wantAssertionsEncrypted": False,
                "wantNameId": True,
                "wantNameIdEncrypted": False,
                "wantAttributeStatement": False,
                "requestedAuthnContext": True,
                "requestedAuthnContextComparison": "exact",
                "signatureAlgorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
                "digestAlgorithm": "http://www.w3.org/2001/04/xmlenc#sha256",
            },
        }

        verbose_proxy_logger.debug(
            f"SAML settings built - SP Entity ID: {saml_entity_id}, IdP SSO URL: {saml_idp_sso_url}"
        )

        return settings

    @staticmethod
    def _prepare_request_from_fastapi(request) -> Dict:
        """
        Prepare request data dictionary from FastAPI request object.
        Args:
            request: FastAPI Request object
        Returns:
            Dictionary with request data for python3-saml
        """
        return {
            "https": "on" if request.url.scheme == "https" else "off",
            "http_host": request.url.hostname,
            "server_port": request.url.port
            or (443 if request.url.scheme == "https" else 80),
            "script_name": request.url.path,
            "get_data": dict(request.query_params),
            "post_data": {},
        }

    @staticmethod
    def get_login_url(request, state: Optional[str] = None) -> str:
        """
        Generate SAML authentication request and return the IdP SSO URL.
        Args:
            request: FastAPI Request object
            state: Optional state parameter for OAuth compatibility
        Returns:
            URL to redirect user to for SAML authentication
        """
        request_data = SAMLAuthenticationHandler._prepare_request_from_fastapi(request)
        saml_settings = SAMLAuthenticationHandler._get_saml_settings(request_data)

        auth = OneLogin_Saml2_Auth(request_data, saml_settings)

        # Build authentication request with optional relay state
        sso_url = auth.login(return_to=state)

        verbose_proxy_logger.debug(f"Generated SAML login URL: {sso_url}")

        return sso_url

    @staticmethod
    async def process_saml_response(request) -> CustomOpenID:
        """
        Process SAML response from IdP and extract user information.
        Args:
            request: FastAPI Request object containing SAML response
        Returns:
            CustomOpenID object with user information
        """
        request_data = SAMLAuthenticationHandler._prepare_request_from_fastapi(request)

        form_data = await request.form()
        request_data["post_data"] = dict(form_data)

        saml_settings = SAMLAuthenticationHandler._get_saml_settings(request_data)

        auth = OneLogin_Saml2_Auth(request_data, saml_settings)

        # Process the SAML response
        auth.process_response()

        errors = auth.get_errors()
        if errors:
            error_reason = auth.get_last_error_reason()
            verbose_proxy_logger.error(
                f"SAML authentication failed - Errors: {errors}, Reason: {error_reason}"
            )
            raise ProxyException(
                message=f"SAML authentication failed: {error_reason}",
                type=ProxyErrorTypes.auth_error,
                param="saml_response",
                code=401,
            )

        if not auth.is_authenticated():
            verbose_proxy_logger.error(
                "SAML authentication failed - User not authenticated"
            )
            raise ProxyException(
                message="SAML authentication failed: User not authenticated",
                type=ProxyErrorTypes.auth_error,
                param="saml_response",
                code=401,
            )

        # Extract user attributes from SAML assertion
        attributes = auth.get_attributes()
        name_id = auth.get_nameid()

        # Log what we received
        if attributes:
            verbose_proxy_logger.debug(
                f"SAML authentication successful - NameID: {name_id}, Attributes: {list(attributes.keys())}"
            )
        else:
            verbose_proxy_logger.warning(
                f"SAML authentication successful but no attributes received - NameID: {name_id}. "
                "Will use NameID as fallback for user information."
            )

        # Get attribute names from environment or use defaults
        user_id_attr = os.getenv("SAML_USER_ID_ATTRIBUTE", "email")
        user_email_attr = os.getenv("SAML_USER_EMAIL_ATTRIBUTE", "email")
        user_first_name_attr = os.getenv("SAML_USER_FIRST_NAME_ATTRIBUTE", "firstName")
        user_last_name_attr = os.getenv("SAML_USER_LAST_NAME_ATTRIBUTE", "lastName")
        user_display_name_attr = os.getenv(
            "SAML_USER_DISPLAY_NAME_ATTRIBUTE", "displayName"
        )

        def get_attr_value(attr_name: str) -> Optional[str]:
            """Get first value from SAML attribute list, or None if not present"""
            if not attributes:
                return None
            attr_values = attributes.get(attr_name, [])
            return attr_values[0] if attr_values else None

        # Extract user information - fallback to NameID if attributes not present
        user_id = get_attr_value(user_id_attr) or name_id
        user_email = get_attr_value(user_email_attr) or name_id
        user_first_name = get_attr_value(user_first_name_attr)
        user_last_name = get_attr_value(user_last_name_attr)
        user_display_name = get_attr_value(user_display_name_attr)

        # If display name not provided, construct from first/last name
        if not user_display_name and user_first_name and user_last_name:
            user_display_name = f"{user_first_name} {user_last_name}"
        elif not user_display_name:
            user_display_name = user_email

        # Build CustomOpenID response (compatible with existing SSO flow)
        result = CustomOpenID(
            id=user_id,
            email=user_email,
            first_name=user_first_name,
            last_name=user_last_name,
            display_name=user_display_name,
            provider="saml",
            team_ids=[],  # Teams managed via SCIM provisioning
        )

        verbose_proxy_logger.info(
            f"SAML user authenticated - ID: {user_id}, Email: {user_email}"
        )

        return result

    @staticmethod
    def get_metadata() -> str:
        """
        Generate SAML Service Provider metadata XML.

        Returns:
            XML string containing SP metadata
        """
        # Create a dummy request for settings generation
        request_data = {
            "https": "on"
            if os.getenv("PROXY_BASE_URL", "").startswith("https")
            else "off",
            "http_host": "localhost",
            "server_port": 443,
            "script_name": "/",
            "get_data": {},
            "post_data": {},
        }

        saml_settings = SAMLAuthenticationHandler._get_saml_settings(request_data)
        settings = OneLogin_Saml2_Settings(settings=saml_settings)

        metadata = settings.get_sp_metadata()
        errors = settings.validate_metadata(metadata)

        if errors:
            verbose_proxy_logger.error(f"SAML metadata validation errors: {errors}")
            raise ProxyException(
                message=f"SAML metadata validation failed: {', '.join(errors)}",
                type=ProxyErrorTypes.auth_error,
                param="saml_metadata",
                code=500,
            )

        return metadata

    @staticmethod
    def should_use_saml_handler() -> bool:
        """
        Check if SAML SSO should be used based on environment variables.

        Returns:
            True if SAML is configured, False otherwise
        """
        return bool(
            os.getenv("SAML_IDP_ENTITY_ID")
            and os.getenv("SAML_IDP_SSO_URL")
            and os.getenv("SAML_IDP_X509_CERT")
        )
