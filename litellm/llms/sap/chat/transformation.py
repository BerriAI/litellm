"""
SAP OpenAI-like chat completion transformation
File: sap/chat/transformation.py
"""
import hashlib
import json
from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload
import httpx

from litellm import DualCache, verbose_logger
from litellm.secret_managers.main import get_secret_str, get_secret
from litellm.types.llms.openai import AllMessageValues
from datetime import datetime, timedelta

from ..common_utils import SAPOAuthToken
from ...openai.chat.gpt_transformation import OpenAIGPTConfig

from ...openai_like.chat.transformation import OpenAILikeChatConfig
from ...openai_like.common_utils import OpenAILikeError


class SAPChatConfig(OpenAIGPTConfig):
    """
    SAP-specific chat configuration that extends OpenAI-like configuration.
    Handles SAP AI Core specific parameters and transformations.
    """

    frequency_penalty: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

    # SAP-specific parameters
    sap_deployment_id: Optional[str] = None
    sap_resource_group: Optional[str] = None
    sap_api_version: Optional[str] = None

    def __init__(
            self,
            frequency_penalty: Optional[int] = None,
            max_tokens: Optional[int] = None,
            n: Optional[int] = None,
            presence_penalty: Optional[int] = None,
            stop: Optional[Union[str, list]] = None,
            temperature: Optional[int] = None,
            top_p: Optional[int] = None,
            response_format: Optional[dict] = None,
            tools: Optional[list] = None,
            tool_choice: Optional[Union[str, dict]] = None,
            sap_deployment_id: Optional[str] = None,
            sap_resource_group: Optional[str] = None,
            sap_api_version: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        self.token_cache = DualCache()
    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for SAP AI Core.
        SAP AI Core supports most standard OpenAI parameters.
        """
        base_params = super().get_supported_openai_params(model)

        # Remove parameters not supported by SAP AI Core
        unsupported_params = ["max_retries", "logit_bias", "functions", "function_call"]
        for param in unsupported_params:
            if param in base_params:
                base_params.remove(param)

        # Add SAP-specific parameters
        sap_params = ["sap_deployment_id", "sap_resource_group", "sap_api_version"]
        base_params.extend(sap_params)

        return base_params

    @overload
    def _transform_messages(
            self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
            self,
            messages: List[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
            self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Transform messages for SAP AI Core compatibility.
        SAP AI Core generally follows OpenAI format but may need specific adjustments.
        """
        # SAP AI Core specific message transformations can go here
        # For now, we'll use the parent class transformation

        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
            self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get SAP AI Core provider information.
        Note: Authentication is handled by SAPOpenAILikeBase via OAuth,
        so this mainly handles the API base URL.
        """
        # Get SAP AI Core base URL from environment if not provided
        api_base = (
                api_base
                or get_secret_str("SAP_AI_CORE_BASE_URL")
                or get_secret_str("SAP_AI_CORE_API_BASE")
        )

        # For SAP, the API key might be None if using OAuth
        # The actual auth is handled by the SAPOpenAILikeBase class
        dynamic_api_key = api_key or get_secret_str("SAP_AI_CORE_API_KEY")

        return api_base, dynamic_api_key

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Determine if we should fake streaming for SAP AI Core.
        Some SAP deployments might not support streaming with certain features.
        """
        # SAP AI Core might not support streaming with response_format
        if optional_params.get("response_format") is not None:
            return True

        # SAP AI Core might not support streaming with certain tool configurations
        if optional_params.get("tools") is not None and len(optional_params.get("tools", [])) > 5:
            # Fake stream if using many tools (arbitrary limit for example)
            return True

        return False

    def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool = False
    ) -> dict:
        """
        Map OpenAI parameters to SAP AI Core compatible parameters.
        """
        # Handle SAP-specific parameters
        sap_deployment_id = non_default_params.pop("sap_deployment_id", None)
        sap_resource_group = non_default_params.pop("sap_resource_group", None)
        sap_api_version = non_default_params.pop("sap_api_version", None)

        # Store SAP parameters separately as they're not part of the request body
        if sap_deployment_id:
            optional_params["_sap_deployment_id"] = sap_deployment_id
        if sap_resource_group:
            optional_params["_sap_resource_group"] = sap_resource_group
        if sap_api_version:
            optional_params["_sap_api_version"] = sap_api_version

        # Check if we need to fake streaming
        if self._should_fake_stream(non_default_params):
            optional_params["fake_stream"] = True

        # Handle response_format for SAP AI Core
        # SAP might have different requirements for structured output
        response_format = non_default_params.get("response_format")
        if response_format is not None and isinstance(response_format, dict):
            # SAP AI Core might handle JSON mode differently
            # For now, we'll keep the standard OpenAI approach
            pass

        # Call parent method to handle standard OpenAI parameter mapping
        mapped_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # Additional SAP-specific parameter transformations
        # SAP AI Core might use different parameter names
        if "max_tokens" in mapped_params and mapped_params.get("max_tokens") is not None:
            # Ensure max_tokens is within SAP AI Core limits (example: 4096)
            mapped_params["max_tokens"] = min(mapped_params["max_tokens"], 4096)

        return mapped_params

    def validate_sap_params(self, optional_params: dict) -> dict:
        """
        Validate SAP-specific parameters before sending the request.
        """
        errors = []

        # Check for required SAP parameters based on the deployment
        sap_deployment_id = optional_params.get("_sap_deployment_id")
        if not sap_deployment_id and not optional_params.get("sap_deployment_id"):
            # Deployment ID might be required for SAP AI Core
            pass  # Not always required, depends on the setup

        # Validate temperature range for SAP AI Core
        temperature = optional_params.get("temperature")
        if temperature is not None and (temperature < 0 or temperature > 2):
            errors.append("Temperature must be between 0 and 2 for SAP AI Core")

        # Validate top_p range
        top_p = optional_params.get("top_p")
        if top_p is not None and (top_p < 0 or top_p > 1):
            errors.append("top_p must be between 0 and 1 for SAP AI Core")

        if errors:
            raise ValueError(f"SAP parameter validation failed: {'; '.join(errors)}")

        return optional_params

    def validate_environment(
            self,
            headers: dict,
            model: str,
            messages: List[AllMessageValues],
            optional_params: dict,
            litellm_params: dict,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Override to add SAP OAuth token support.
        If SAP credentials are provided, use OAuth token instead of API key.
        """
        # Try to get SAP token first
        sap_token = None
        if optional_params:
            sap_token = self._get_sap_token_from_params(optional_params, headers)

            # Set up headers with OAuth token
            if headers is None:
                headers = {}

            headers.update({
                "Content-Type": "application/json",
                "Authorization": f"{sap_token.token_type} {sap_token.access_token}",
            })

            # TODO: Check embedding here
            # elif endpoint_type == "embeddings":
            #     api_base = f"{api_base}/embeddings?api-version=2025-03-01-preview"


            return headers
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

    def get_complete_url( self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None):
        # Extract deployment ID if provided
        deployment_id = optional_params.pop("sap_deployment_id")
        if deployment_id:
            # For SAP AI Core, the deployment ID is part of the URL path
            api_base = f"{api_base}/v2/inference/deployments/{deployment_id}/chat/completions?api-version=2025-03-01-preview"
        return api_base
