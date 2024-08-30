import json
import os
from typing import Callable, Optional, Tuple

import httpx

from litellm import verbose_logger
from litellm.caching import DualCache
from litellm.proxy.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
)
from litellm.utils import get_secret

azure_ad_cache = DualCache()


class AzureOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


def get_azure_ad_token_from_oidc(azure_ad_token: str):
    azure_client_id = os.getenv("AZURE_CLIENT_ID", None)
    azure_tenant_id = os.getenv("AZURE_TENANT_ID", None)
    azure_authority_host = os.getenv(
        "AZURE_AUTHORITY_HOST", "https://login.microsoftonline.com"
    )

    if azure_client_id is None or azure_tenant_id is None:
        raise AzureOpenAIError(
            status_code=422,
            message="AZURE_CLIENT_ID and AZURE_TENANT_ID must be set",
        )

    oidc_token = get_secret(azure_ad_token)

    if oidc_token is None:
        raise AzureOpenAIError(
            status_code=401,
            message="OIDC token could not be retrieved from secret manager.",
        )

    azure_ad_token_cache_key = json.dumps(
        {
            "azure_client_id": azure_client_id,
            "azure_tenant_id": azure_tenant_id,
            "azure_authority_host": azure_authority_host,
            "oidc_token": oidc_token,
        }
    )

    azure_ad_token_access_token = azure_ad_cache.get_cache(azure_ad_token_cache_key)
    if azure_ad_token_access_token is not None:
        return azure_ad_token_access_token

    req_token = httpx.post(
        f"{azure_authority_host}/{azure_tenant_id}/oauth2/v2.0/token",
        data={
            "client_id": azure_client_id,
            "grant_type": "client_credentials",
            "scope": "https://cognitiveservices.azure.com/.default",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": oidc_token,
        },
    )

    if req_token.status_code != 200:
        raise AzureOpenAIError(
            status_code=req_token.status_code,
            message=req_token.text,
        )

    azure_ad_token_json = req_token.json()
    azure_ad_token_access_token = azure_ad_token_json.get("access_token", None)
    azure_ad_token_expires_in = azure_ad_token_json.get("expires_in", None)

    if azure_ad_token_access_token is None:
        raise AzureOpenAIError(
            status_code=422, message="Azure AD Token access_token not returned"
        )

    if azure_ad_token_expires_in is None:
        raise AzureOpenAIError(
            status_code=422, message="Azure AD Token expires_in not returned"
        )

    azure_ad_cache.set_cache(
        key=azure_ad_token_cache_key,
        value=azure_ad_token_access_token,
        ttl=azure_ad_token_expires_in,
    )

    return azure_ad_token_access_token


def get_azure_ad_token_and_provider(
    azure_ad_token: Optional[str],
    azure_ad_token_provider: Optional[Callable[[], str]] = None,
) -> Tuple[Optional[str], Optional[Callable[[], str]]]:
    """
    Utility function to simplify oidc + common azure auth logic

    Returns
    - azure_ad_token - Optional[str]
    - azure_ad_token_provider - Optional[str]
    """
    if azure_ad_token is not None:
        if azure_ad_token.startswith("oidc/"):
            azure_ad_token = get_azure_ad_token_from_oidc(azure_ad_token)
        else:
            try:
                azure_ad_token_provider = get_azure_ad_token_provider()
            except ValueError:
                verbose_logger.debug("Azure AD Token Provider could not be used.")

    return azure_ad_token, azure_ad_token_provider
