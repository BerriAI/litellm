"""
SSO (Single Sign-On) related modules for LiteLLM Proxy.

This package contains custom SSO implementations and utilities.
"""

from litellm.proxy.management_endpoints.sso.custom_microsoft_sso import (
    CustomMicrosoftSSO,
)

__all__ = ["CustomMicrosoftSSO"]

