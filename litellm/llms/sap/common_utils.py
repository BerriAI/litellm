"""
SAP OpenAI-like utilities with OAuth authentication
File: openai_like/sap_utils.py
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Literal, Optional, Tuple

import httpx
from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.secret_managers.main import get_secret

from ..openai_like.common_utils import OpenAILikeBase, OpenAILikeError


class SAPOAuthToken(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    expires_at: datetime


class SAPOpenAILikeBase(OpenAILikeBase):
    """
    OpenAI-like base class with SAP OAuth authentication support
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token_cache = DualCache()

    def _get_cache_key(self, client_id: str, client_secret: str, xsuaa_url: str) -> str:
        """Generate a unique cache key based on credentials"""
        credential_str = json.dumps({
            "client_id": client_id,
            "client_secret": client_secret,
            "xsuaa_url": xsuaa_url
        }, sort_keys=True)
        return f"sap_oauth_{hashlib.sha256(credential_str.encode()).hexdigest()}"

    def _get_sap_oauth_token(
            self,
            client_id: str,
            client_secret: str,
            xsuaa_url: str,
    ) -> SAPOAuthToken:
        """
        Exchange client credentials for an OAuth token via SAP xsuaa.
        """
        verbose_logger.debug(
            f"Exchanging SAP credentials for OAuth token at {xsuaa_url}"
        )

        token_endpoint = f"{xsuaa_url}/oauth/token"

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

        try:
            response = httpx.post(
                token_endpoint,
                headers=headers,
                data=data,
                timeout=30.0,
            )
            response.raise_for_status()

            token_data = response.json()

            # Calculate token expiration time (subtract 60 seconds for safety margin)
            expires_in = token_data.get("expires_in", 3600)
            expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

            return SAPOAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=expires_in,
                expires_at=expires_at,
            )

        except httpx.HTTPStatusError as e:
            raise OpenAILikeError(
                status_code=e.response.status_code,
                message=f"Failed to get SAP OAuth token: {e.response.text}",
            )
        except Exception as e:
            raise OpenAILikeError(
                status_code=500,
                message=f"Error getting SAP OAuth token: {str(e)}",
            )

    def _get_sap_token_from_params(
            self,
            optional_params: dict,
            headers: Optional[dict] = None,
    ) -> Optional[SAPOAuthToken]:
        """
        Extract SAP credentials from optional_params and get OAuth token.
        Returns None if SAP credentials are not provided.
        """
        # Check if SAP credentials are provided
        sap_client_id = optional_params.pop("sap_client_id", None) or get_secret("UAA_CLIENT_ID")
        sap_client_secret = optional_params.pop("sap_client_secret", None) or get_secret("UAA_CLIENT_SECRET")
        sap_xsuaa_url = optional_params.pop("sap_xsuaa_url", None) or get_secret("UAA_URL")

        # If no SAP credentials, return None (fallback to regular auth)
        if not all([sap_client_id, sap_client_secret, sap_xsuaa_url]):
            return None

        # Check cache
        cache_key = self._get_cache_key(sap_client_id, sap_client_secret, sap_xsuaa_url)
        cached_token = self.token_cache.get_cache(cache_key)

        if cached_token and isinstance(cached_token, SAPOAuthToken):
            if cached_token.expires_at > datetime.now():
                verbose_logger.debug("Using cached SAP OAuth token")
                return cached_token

        # Get new token
        token = self._get_sap_oauth_token(
            client_id=sap_client_id,
            client_secret=sap_client_secret,
            xsuaa_url=sap_xsuaa_url,
        )

        # Cache the token
        ttl = (token.expires_at - datetime.now()).total_seconds()
        self.token_cache.set_cache(cache_key, token, ttl=int(ttl))

        return token

    def _validate_environment(
            self,
            api_key: Optional[str],
            api_base: Optional[str],
            endpoint_type: Literal["chat_completions", "embeddings"],
            headers: Optional[dict],
            custom_endpoint: Optional[bool],
            optional_params: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        """
        Override to add SAP OAuth token support.
        If SAP credentials are provided, use OAuth token instead of API key.
        """
        # Try to get SAP token first
        sap_token = None
        if optional_params:
            sap_token = self._get_sap_token_from_params(optional_params, headers)

        # If we have SAP token, use it
        if sap_token:
            if api_base is None:
                # Get SAP AI Core base URL
                sap_base_url = get_secret("SAP_AI_CORE_BASE_URL")
                if sap_base_url:
                    api_base = sap_base_url
                else:
                    raise OpenAILikeError(
                        status_code=400,
                        message="Missing SAP AI Core API Base URL - Please set SAP_AI_CORE_BASE_URL",
                    )

            # Extract deployment ID if provided
            deployment_id = optional_params.pop("sap_deployment_id")
            if deployment_id:
                # For SAP AI Core, the deployment ID is part of the URL path
                api_base = f"{api_base}/v2/inference/deployments/{deployment_id}"

            # Set up headers with OAuth token
            if headers is None:
                headers = {}

            headers.update({
                "Content-Type": "application/json",
                "Authorization": f"{sap_token.token_type} {sap_token.access_token}",
            })

            # Don't append /chat/completions for SAP endpoints if custom_endpoint is True
            if not custom_endpoint:
                if endpoint_type == "chat_completions":
                    api_base = f"{api_base}/chat/completions?api-version=2025-03-01-preview"
                elif endpoint_type == "embeddings":
                    api_base = f"{api_base}/embeddings?api-version=2025-03-01-preview"

            return api_base, headers

        # If no SAP token, fall back to regular validation
        return super()._validate_environment(
            api_key=api_key,
            api_base=api_base,
            endpoint_type=endpoint_type,
            headers=headers,
            custom_endpoint=custom_endpoint,
        )