import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.secret_managers.main import get_secret

from .base_aws_llm import BaseAWSLLM, AwsAuthError


class SAPOAuthToken(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    expires_at: datetime


class SAPAWSLLM(BaseAWSLLM):
    """
    Base class for SAP AI Core LLMs that overrides AWS authentication
    with SAP's OAuth-based authentication.
    """

    def __init__(self) -> None:
        super().__init__()
        self.token_cache = DualCache()
        self.sap_authentication_params = [
            "sap_client_id",
            "sap_client_secret",
            "sap_xsuaa_url",
            "sap_ai_core_base_url",
            "sap_deployment_id",
        ]

    def get_cache_key(self, credential_args: Dict[str, Optional[str]]) -> str:
        """
        Generate a unique cache key based on the SAP credential arguments.
        """
        # Filter out only SAP-related credentials
        sap_credentials = {k: v for k, v in credential_args.items() if k.startswith("sap_")}
        credential_str = json.dumps(sap_credentials, sort_keys=True)
        return f"sap_oauth_{hashlib.sha256(credential_str.encode()).hexdigest()}"

    @tracer.wrap()
    def _get_sap_oauth_token(
            self,
            client_id: str,
            client_secret: str,
            xsuaa_url: str,
    ) -> SAPOAuthToken:
        """
        Exchange client credentials for an OAuth token via SAP xsuaa.

        Args:
            client_id: SAP AI Core client ID
            client_secret: SAP AI Core client secret
            xsuaa_url: xsuaa OAuth endpoint URL

        Returns:
            SAPOAuthToken object with access token and expiration
        """
        verbose_logger.debug(
            f"Exchanging SAP credentials for OAuth token at {xsuaa_url}"
        )

        token_endpoint = f"{xsuaa_url}/oauth/token"

        # Prepare the request
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
            raise AwsAuthError(
                status_code=e.response.status_code,
                message=f"Failed to get SAP OAuth token: {e.response.text}",
            )
        except Exception as e:
            raise AwsAuthError(
                status_code=500,
                message=f"Error getting SAP OAuth token: {str(e)}",
            )

    def _extract_deployment_id_from_model(self, model: str) -> Optional[str]:
        """
        Extract deployment ID from model string.
        Examples:
        - "sap/deployment/d50f75fdf3352f02" -> "d50f75fdf3352f02"
        - "deployment/d50f75fdf3352f02" -> "d50f75fdf3352f02"
        - "d50f75fdf3352f02" -> "d50f75fdf3352f02"
        """
        if "/" in model:
            parts = model.split("/")
            # Return the last part as deployment ID
            return parts[-1]
        return model

    def _get_sap_params_from_env(self) -> Dict[str, Optional[str]]:
        """
        Get SAP parameters from environment variables.
        """
        return {
            "sap_client_id": get_secret("SAP_AI_CORE_CLIENT_ID"),
            "sap_client_secret": get_secret("SAP_AI_CORE_CLIENT_SECRET"),
            "sap_xsuaa_url": get_secret("SAP_AI_CORE_XSUAA_URL"),
            "sap_ai_core_base_url": get_secret("SAP_AI_CORE_BASE_URL"),
        }

    @tracer.wrap()
    def get_credentials(
            self,
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None,
            aws_session_token: Optional[str] = None,
            aws_region_name: Optional[str] = None,
            aws_session_name: Optional[str] = None,
            aws_profile_name: Optional[str] = None,
            aws_role_name: Optional[str] = None,
            aws_web_identity_token: Optional[str] = None,
            aws_sts_endpoint: Optional[str] = None,
            sap_client_id: Optional[str] = None,
            sap_client_secret: Optional[str] = None,
            sap_xsuaa_url: Optional[str] = None,
            **kwargs,
    ) -> SAPOAuthToken:
        """
        Override to return SAP OAuth token instead of AWS credentials.
        """
        # Get SAP credentials from params or environment
        env_params = self._get_sap_params_from_env()

        client_id = sap_client_id or env_params.get("sap_client_id")
        client_secret = sap_client_secret or env_params.get("sap_client_secret")
        xsuaa_url = sap_xsuaa_url or env_params.get("sap_xsuaa_url")

        if not all([client_id, client_secret, xsuaa_url]):
            missing = []
            if not client_id:
                missing.append("sap_client_id")
            if not client_secret:
                missing.append("sap_client_secret")
            if not xsuaa_url:
                missing.append("sap_xsuaa_url")

            raise AwsAuthError(
                status_code=401,
                message=f"Missing required SAP credentials: {', '.join(missing)}",
            )

        # Create cache key
        cache_key = self.get_cache_key({
            "sap_client_id": client_id,
            "sap_client_secret": client_secret,
            "sap_xsuaa_url": xsuaa_url,
        })

        # Check cache
        cached_token = self.token_cache.get_cache(cache_key)
        if cached_token and isinstance(cached_token, SAPOAuthToken):
            # Check if token is still valid
            if cached_token.expires_at > datetime.now():
                verbose_logger.debug("Using cached SAP OAuth token")
                return cached_token

        # Get new token
        token = self._get_sap_oauth_token(
            client_id=client_id,
            client_secret=client_secret,
            xsuaa_url=xsuaa_url,
        )

        # Cache the token
        ttl = (token.expires_at - datetime.now()).total_seconds()
        self.token_cache.set_cache(cache_key, token, ttl=int(ttl))

        return token

    @tracer.wrap()
    def get_request_headers(
            self,
            credentials: Any,  # Will be SAPOAuthToken
            aws_region_name: str,
            extra_headers: Optional[dict],
            endpoint_url: str,
            data: str,
            headers: dict,
    ) -> Dict[str, str]:
        """
        Override to use Bearer token instead of AWS SigV4 signing.
        """
        if not isinstance(credentials, SAPOAuthToken):
            # Try to get SAP token if AWS credentials were passed
            credentials = self.get_credentials()

        # Start with base headers
        request_headers = headers.copy() if headers else {}

        # Add Bearer token
        request_headers["Authorization"] = f"{credentials.token_type} {credentials.access_token}"

        # Add extra headers if provided
        if extra_headers:
            request_headers.update(extra_headers)

        # Ensure Content-Type is set
        if "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"

        verbose_logger.debug(f"SAP request headers prepared for {endpoint_url}")

        # Return a simple dict instead of AWSPreparedRequest
        return request_headers

    def get_runtime_endpoint(
            self,
            api_base: Optional[str],
            aws_bedrock_runtime_endpoint: Optional[str],
            aws_region_name: str,
            model: Optional[str] = None,
            deployment_id: Optional[str] = None,
            stream: bool = False,
    ) -> Tuple[str, str]:
        """
        Override to return SAP AI Core endpoints.
        """
        # Get SAP base URL from environment or params
        sap_base_url = get_secret("SAP_AI_CORE_BASE_URL")

        if api_base:
            base_url = api_base
        elif aws_bedrock_runtime_endpoint:
            # Allow using aws_bedrock_runtime_endpoint for SAP URL
            base_url = aws_bedrock_runtime_endpoint
        elif sap_base_url:
            base_url = sap_base_url
        else:
            # Default SAP AI Core URL
            base_url = "https://api.ai.internalprod.eu-central-aws.ml.hana.ondemand.com"

        # Extract deployment ID
        if deployment_id:
            deploy_id = deployment_id
        elif model:
            deploy_id = self._extract_deployment_id_from_model(model)
        else:
            raise AwsAuthError(
                status_code=400,
                message="No deployment ID provided",
            )

        # Construct endpoint URL
        endpoint_path = "converse-stream" if stream else "converse"
        endpoint_url = f"{base_url}/v2/inference/deployments/{deploy_id}/{endpoint_path}"

        verbose_logger.debug(f"SAP AI Core endpoint: {endpoint_url}")

        # Return same URL for both (no proxy distinction in SAP)
        return endpoint_url, endpoint_url

    def _get_boto_credentials_from_optional_params(
            self, optional_params: dict, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Override to extract SAP-specific parameters.
        """
        # Extract SAP parameters
        sap_params = {}
        for param in self.sap_authentication_params:
            if param in optional_params:
                sap_params[param] = optional_params.pop(param)

        # Get token
        token = self.get_credentials(**sap_params)

        # Get deployment ID
        deployment_id = sap_params.get("sap_deployment_id")
        if not deployment_id and model:
            deployment_id = self._extract_deployment_id_from_model(model)

        # Get base URL
        sap_base_url = sap_params.get("sap_ai_core_base_url") or get_secret("SAP_AI_CORE_BASE_URL")

        return {
            "credentials": token,
            "deployment_id": deployment_id,
            "sap_base_url": sap_base_url,
            "aws_region_name": "sap-ai-core",  # Dummy value for compatibility
        }

    def _sign_request(self, *args, **kwargs) -> Tuple[dict, Optional[bytes]]:
        """
        Override to skip AWS signing - SAP uses Bearer tokens.
        """
        # Extract headers from kwargs
        headers = kwargs.get("headers", {})
        request_data = kwargs.get("request_data", {})

        # Return headers as-is (Bearer token should already be added)
        return headers, json.dumps(request_data).encode() if request_data else None