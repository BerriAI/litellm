"""
SSO (Single Sign-On) related modules for LiteLLM Proxy.

This package contains custom SSO implementations and utilities.
"""

from litellm.proxy.management_endpoints.sso.custom_microsoft_sso import (
    CustomMicrosoftSSO,
)
from litellm.proxy.management_endpoints.sso.litellm_http_client_sso import (
    LiteLLMGoogleSSO,
    LiteLLMHttpClientSSO,
    create_generic_sso_provider,
)

__all__ = ["CustomMicrosoftSSO", "LiteLLMGoogleSSO", "LiteLLMHttpClientSSO", "create_generic_sso_provider"]
