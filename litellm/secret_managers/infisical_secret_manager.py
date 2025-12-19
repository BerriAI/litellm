"""
Infisical Secret Manager Integration for LiteLLM

This module provides integration with Infisical (https://infisical.com/) for secret management.

Infisical is an open-source secret management platform that helps teams
manage secrets across multiple environments.

Required Environment Variables:
- INFISICAL_URL: Base URL for Infisical API (default: https://app.infisical.com)
- INFISICAL_CLIENT_ID: Machine Identity Client ID for authentication
- INFISICAL_CLIENT_SECRET: Machine Identity Client Secret for authentication

Optional Environment Variables:
- INFISICAL_PROJECT_ID: Default project ID for secrets
- INFISICAL_ENVIRONMENT: Default environment slug (e.g., 'dev', 'staging', 'prod')
- INFISICAL_SECRET_PATH: Default path for secrets (default: '/')
- INFISICAL_REFRESH_INTERVAL: Token/cache refresh interval in seconds (default: 86400)

Usage:
    In your LiteLLM config.yaml:
    
    general_settings:
      key_management_system: infisical
      key_management_settings:
        project_id: "your-project-id"
        environment: "prod"
        secret_path: "/"

Authentication:
    Infisical uses Machine Identity authentication (Universal Auth).
    See: https://infisical.com/docs/documentation/platform/identities/universal-auth
"""

import os
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.caching import InMemoryCache
from litellm.constants import SECRET_MANAGER_REFRESH_INTERVAL
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import KeyManagementSystem

from .base_secret_manager import BaseSecretManager


class InfisicalSecretManager(BaseSecretManager):
    """
    Secret manager implementation for Infisical.
    
    Infisical API Reference:
    - Authentication: https://infisical.com/docs/api-reference/endpoints/universal-auth/login
    - Get Secret: https://infisical.com/docs/api-reference/endpoints/secrets/read-one
    - Create Secret: https://infisical.com/docs/api-reference/endpoints/secrets/create
    - Update Secret: https://infisical.com/docs/api-reference/endpoints/secrets/update
    - Delete Secret: https://infisical.com/docs/api-reference/endpoints/secrets/delete
    """

    def __init__(
        self,
        site_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        secret_path: Optional[str] = None,
    ):
        """
        Initialize the Infisical Secret Manager.
        
        Args:
            site_url: Infisical API base URL (default: https://app.infisical.com)
            client_id: Machine Identity Client ID
            client_secret: Machine Identity Client Secret
            project_id: Default project ID for secrets
            environment: Default environment slug (e.g., 'dev', 'staging', 'prod')
            secret_path: Default path for secrets (default: '/')
        """
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        # Configuration from parameters or environment
        self.site_url = (
            site_url 
            or os.getenv("INFISICAL_URL", "https://app.infisical.com")
        ).rstrip("/")
        
        self.client_id = client_id or os.getenv("INFISICAL_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("INFISICAL_CLIENT_SECRET", "")
        self.project_id = project_id or os.getenv("INFISICAL_PROJECT_ID", "")
        self.environment = environment or os.getenv("INFISICAL_ENVIRONMENT", "dev")
        self.secret_path = secret_path or os.getenv("INFISICAL_SECRET_PATH", "/")

        # Validate required credentials
        self._validate_credentials()

        # Set up cache for tokens and secrets
        _refresh_interval = int(
            os.environ.get("INFISICAL_REFRESH_INTERVAL", SECRET_MANAGER_REFRESH_INTERVAL)
        )
        self.cache = InMemoryCache(default_ttl=_refresh_interval)

        # Register as the secret manager
        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.INFISICAL

        if premium_user is not True:
            raise ValueError(
                f"Infisical secret manager is only available for premium users. {CommonProxyErrors.not_premium_user.value}"
            )

        verbose_logger.debug(
            f"Infisical Secret Manager initialized. Site URL: {self.site_url}, "
            f"Project ID: {self.project_id}, Environment: {self.environment}"
        )

    def _validate_credentials(self) -> None:
        """
        Validate that required credentials are present.
        
        Raises:
            ValueError: If required credentials are missing
        """
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Missing Infisical credentials. Please set INFISICAL_CLIENT_ID and "
                "INFISICAL_CLIENT_SECRET environment variables, or pass them as parameters."
            )

    def _authenticate(self) -> str:
        """
        Authenticate with Infisical using Machine Identity (Universal Auth).
        
        Returns:
            str: Access token for API requests
            
        Raises:
            RuntimeError: If authentication fails
        """
        # Check cache first
        cached_token = self.cache.get_cache("infisical_access_token")
        if cached_token is not None:
            verbose_logger.debug("Using cached Infisical access token")
            return cached_token

        verbose_logger.debug("Authenticating with Infisical...")
        
        auth_url = f"{self.site_url}/api/v1/auth/universal-auth/login"
        
        try:
            client = _get_httpx_client()
            response = client.post(
                url=auth_url,
                json={
                    "clientId": self.client_id,
                    "clientSecret": self.client_secret,
                },
            )
            response.raise_for_status()
            
            auth_data = response.json()
            access_token = auth_data.get("accessToken")
            expires_in = auth_data.get("expiresIn", 7200)  # Default 2 hours
            
            if not access_token:
                raise RuntimeError("No access token in Infisical auth response")
            
            verbose_logger.debug(
                f"Successfully authenticated with Infisical. Token expires in {expires_in}s"
            )
            
            # Cache the token (with some buffer before expiry)
            cache_ttl = max(expires_in - 60, 60)  # At least 60 seconds, 60 second buffer
            self.cache.set_cache(
                key="infisical_access_token",
                value=access_token,
                ttl=cache_ttl,
            )
            
            return access_token
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = e.response.text
            raise RuntimeError(
                f"Failed to authenticate with Infisical: {e.response.status_code} - {error_detail}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to authenticate with Infisical: {e}")

    async def _async_authenticate(self) -> str:
        """
        Asynchronously authenticate with Infisical using Machine Identity (Universal Auth).
        
        Returns:
            str: Access token for API requests
            
        Raises:
            RuntimeError: If authentication fails
        """
        # Check cache first
        cached_token = self.cache.get_cache("infisical_access_token")
        if cached_token is not None:
            verbose_logger.debug("Using cached Infisical access token")
            return cached_token

        verbose_logger.debug("Authenticating with Infisical (async)...")
        
        auth_url = f"{self.site_url}/api/v1/auth/universal-auth/login"
        
        try:
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.SecretManager,
            )
            response = await async_client.post(
                url=auth_url,
                json={
                    "clientId": self.client_id,
                    "clientSecret": self.client_secret,
                },
            )
            response.raise_for_status()
            
            auth_data = response.json()
            access_token = auth_data.get("accessToken")
            expires_in = auth_data.get("expiresIn", 7200)
            
            if not access_token:
                raise RuntimeError("No access token in Infisical auth response")
            
            verbose_logger.debug(
                f"Successfully authenticated with Infisical. Token expires in {expires_in}s"
            )
            
            cache_ttl = max(expires_in - 60, 60)
            self.cache.set_cache(
                key="infisical_access_token",
                value=access_token,
                ttl=cache_ttl,
            )
            
            return access_token
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = e.response.text
            raise RuntimeError(
                f"Failed to authenticate with Infisical: {e.response.status_code} - {error_detail}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to authenticate with Infisical: {e}")

    def _get_request_headers(self) -> Dict[str, str]:
        """
        Get headers for Infisical API requests including authentication.
        
        Returns:
            dict: Headers with Bearer token
        """
        token = self._authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def _get_async_request_headers(self) -> Dict[str, str]:
        """
        Get headers for async Infisical API requests including authentication.
        
        Returns:
            dict: Headers with Bearer token
        """
        token = await self._async_authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _get_secret_url(
        self,
        secret_name: str,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        secret_path: Optional[str] = None,
    ) -> str:
        """
        Build the URL for accessing a secret in Infisical.
        
        Args:
            secret_name: Name of the secret
            project_id: Project ID (uses default if not provided)
            environment: Environment slug (uses default if not provided)
            secret_path: Secret path (uses default if not provided)
            
        Returns:
            str: Full URL for the secret endpoint
        """
        resolved_project_id = project_id or self.project_id
        resolved_environment = environment or self.environment
        resolved_path = secret_path or self.secret_path
        
        # Ensure path starts with /
        if not resolved_path.startswith("/"):
            resolved_path = "/" + resolved_path
            
        return (
            f"{self.site_url}/api/v3/secrets/raw/{secret_name}"
            f"?workspaceId={resolved_project_id}"
            f"&environment={resolved_environment}"
            f"&secretPath={resolved_path}"
        )

    def _get_secrets_batch_url(
        self,
        project_id: Optional[str] = None,
        environment: Optional[str] = None,
        secret_path: Optional[str] = None,
    ) -> str:
        """
        Build the URL for fetching multiple secrets from Infisical.
        
        Args:
            project_id: Project ID (uses default if not provided)
            environment: Environment slug (uses default if not provided)
            secret_path: Secret path (uses default if not provided)
            
        Returns:
            str: Full URL for the secrets batch endpoint
        """
        resolved_project_id = project_id or self.project_id
        resolved_environment = environment or self.environment
        resolved_path = secret_path or self.secret_path
        
        if not resolved_path.startswith("/"):
            resolved_path = "/" + resolved_path
            
        return (
            f"{self.site_url}/api/v3/secrets/raw"
            f"?workspaceId={resolved_project_id}"
            f"&environment={resolved_environment}"
            f"&secretPath={resolved_path}"
        )

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Asynchronously read a secret from Infisical.
        
        Args:
            secret_name: Name of the secret to read
            optional_params: Additional parameters:
                - project_id: Override default project ID
                - environment: Override default environment
                - secret_path: Override default secret path
            timeout: Request timeout
            
        Returns:
            Optional[str]: The secret value if found, None otherwise
        """
        # Check cache first
        cache_key = f"infisical_secret_{secret_name}"
        cached_value = self.cache.get_cache(cache_key)
        if cached_value is not None:
            verbose_logger.debug(f"Using cached value for secret: {secret_name}")
            return cached_value

        optional_params = optional_params or {}
        
        try:
            headers = await self._get_async_request_headers()
            url = self._get_secret_url(
                secret_name=secret_name,
                project_id=optional_params.get("project_id"),
                environment=optional_params.get("environment"),
                secret_path=optional_params.get("secret_path"),
            )
            
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.SecretManager,
                params={"timeout": timeout},
            )
            
            response = await async_client.get(url=url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            secret_value = data.get("secret", {}).get("secretValue")
            
            if secret_value is not None:
                self.cache.set_cache(cache_key, secret_value)
                verbose_logger.debug(f"Successfully read secret: {secret_name}")
                
            return secret_value
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                verbose_logger.debug(f"Secret {secret_name} not found in Infisical")
            else:
                verbose_logger.warning(
                    f"Error reading secret from Infisical: {e.response.status_code} - {e.response.text}"
                )
            return None
        except Exception as e:
            verbose_logger.exception(f"Error reading secret from Infisical: {e}")
            return None

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Synchronously read a secret from Infisical.
        
        Args:
            secret_name: Name of the secret to read
            optional_params: Additional parameters:
                - project_id: Override default project ID
                - environment: Override default environment
                - secret_path: Override default secret path
            timeout: Request timeout
            
        Returns:
            Optional[str]: The secret value if found, None otherwise
        """
        # Check cache first
        cache_key = f"infisical_secret_{secret_name}"
        cached_value = self.cache.get_cache(cache_key)
        if cached_value is not None:
            verbose_logger.debug(f"Using cached value for secret: {secret_name}")
            return cached_value

        optional_params = optional_params or {}
        
        try:
            headers = self._get_request_headers()
            url = self._get_secret_url(
                secret_name=secret_name,
                project_id=optional_params.get("project_id"),
                environment=optional_params.get("environment"),
                secret_path=optional_params.get("secret_path"),
            )
            
            sync_client = _get_httpx_client(params={"timeout": timeout})
            response = sync_client.get(url=url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            secret_value = data.get("secret", {}).get("secretValue")
            
            if secret_value is not None:
                self.cache.set_cache(cache_key, secret_value)
                verbose_logger.debug(f"Successfully read secret: {secret_name}")
                
            return secret_value
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                verbose_logger.debug(f"Secret {secret_name} not found in Infisical")
            else:
                verbose_logger.warning(
                    f"Error reading secret from Infisical: {e.response.status_code} - {e.response.text}"
                )
            return None
        except Exception as e:
            verbose_logger.exception(f"Error reading secret from Infisical: {e}")
            return None

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tags: Optional[Union[dict, list]] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously write a secret to Infisical.
        
        First tries to create the secret, if it already exists, updates it.
        
        Args:
            secret_name: Name of the secret to write
            secret_value: Value to store
            description: Optional description/comment for the secret
            optional_params: Additional parameters:
                - project_id: Override default project ID
                - environment: Override default environment
                - secret_path: Override default secret path
            timeout: Request timeout
            tags: Optional tags (Infisical uses tag IDs)
            
        Returns:
            dict: Response containing status and details
        """
        optional_params = optional_params or {}
        
        try:
            headers = await self._get_async_request_headers()
            
            resolved_project_id = optional_params.get("project_id") or self.project_id
            resolved_environment = optional_params.get("environment") or self.environment
            resolved_path = optional_params.get("secret_path") or self.secret_path
            
            if not resolved_path.startswith("/"):
                resolved_path = "/" + resolved_path
            
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.SecretManager,
                params={"timeout": timeout},
            )
            
            # Build the request body
            create_body: Dict[str, Any] = {
                "workspaceId": resolved_project_id,
                "environment": resolved_environment,
                "secretPath": resolved_path,
                "secretKey": secret_name,
                "secretValue": secret_value,
            }
            
            if description:
                create_body["secretComment"] = description
                
            if tags:
                # Infisical expects tag IDs
                if isinstance(tags, list):
                    create_body["tagIds"] = tags
                elif isinstance(tags, dict):
                    # If dict, try to extract tag IDs from values
                    create_body["tagIds"] = list(tags.values())
            
            # Try to create the secret first
            create_url = f"{self.site_url}/api/v3/secrets/raw/{secret_name}"
            
            try:
                response = await async_client.post(
                    url=create_url,
                    headers=headers,
                    json=create_body,
                )
                response.raise_for_status()
                
                # Update cache
                cache_key = f"infisical_secret_{secret_name}"
                self.cache.set_cache(cache_key, secret_value)
                
                verbose_logger.debug(f"Successfully created secret: {secret_name}")
                return {
                    "status": "success",
                    "operation": "create",
                    "secret_name": secret_name,
                    "data": response.json(),
                }
                
            except httpx.HTTPStatusError as create_error:
                # If secret already exists (400 or 409), try to update it
                if create_error.response.status_code in [400, 409]:
                    verbose_logger.debug(
                        f"Secret {secret_name} already exists, updating..."
                    )
                    
                    update_body: Dict[str, Any] = {
                        "workspaceId": resolved_project_id,
                        "environment": resolved_environment,
                        "secretPath": resolved_path,
                        "secretValue": secret_value,
                    }
                    
                    if description:
                        update_body["secretComment"] = description
                        
                    if tags:
                        if isinstance(tags, list):
                            update_body["tagIds"] = tags
                        elif isinstance(tags, dict):
                            update_body["tagIds"] = list(tags.values())
                    
                    update_response = await async_client.patch(
                        url=create_url,
                        headers=headers,
                        json=update_body,
                    )
                    update_response.raise_for_status()
                    
                    # Update cache
                    cache_key = f"infisical_secret_{secret_name}"
                    self.cache.set_cache(cache_key, secret_value)
                    
                    verbose_logger.debug(f"Successfully updated secret: {secret_name}")
                    return {
                        "status": "success",
                        "operation": "update",
                        "secret_name": secret_name,
                        "data": update_response.json(),
                    }
                else:
                    raise create_error
                    
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = e.response.text
            verbose_logger.exception(
                f"Error writing secret to Infisical: {e.response.status_code} - {error_detail}"
            )
            return {
                "status": "error",
                "message": f"HTTP {e.response.status_code}: {error_detail}",
            }
        except Exception as e:
            verbose_logger.exception(f"Error writing secret to Infisical: {e}")
            return {"status": "error", "message": str(e)}

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Asynchronously delete a secret from Infisical.
        
        Note: Infisical does not support soft-delete/recovery window.
        
        Args:
            secret_name: Name of the secret to delete
            recovery_window_in_days: Not used by Infisical
            optional_params: Additional parameters:
                - project_id: Override default project ID
                - environment: Override default environment
                - secret_path: Override default secret path
            timeout: Request timeout
            
        Returns:
            dict: Response containing status and details
        """
        optional_params = optional_params or {}
        
        try:
            headers = await self._get_async_request_headers()
            
            resolved_project_id = optional_params.get("project_id") or self.project_id
            resolved_environment = optional_params.get("environment") or self.environment
            resolved_path = optional_params.get("secret_path") or self.secret_path
            
            if not resolved_path.startswith("/"):
                resolved_path = "/" + resolved_path
            
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.SecretManager,
                params={"timeout": timeout},
            )
            
            delete_url = f"{self.site_url}/api/v3/secrets/raw/{secret_name}"
            
            delete_body = {
                "workspaceId": resolved_project_id,
                "environment": resolved_environment,
                "secretPath": resolved_path,
            }
            
            response = await async_client.request(
                method="DELETE",
                url=delete_url,
                headers=headers,
                json=delete_body,
            )
            response.raise_for_status()
            
            # Clear cache
            cache_key = f"infisical_secret_{secret_name}"
            self.cache.delete_cache(cache_key)
            
            verbose_logger.debug(f"Successfully deleted secret: {secret_name}")
            return {
                "status": "success",
                "message": f"Secret {secret_name} deleted successfully",
                "data": response.json() if response.text else {},
            }
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.json()
            except Exception:
                error_detail = e.response.text
            verbose_logger.exception(
                f"Error deleting secret from Infisical: {e.response.status_code} - {error_detail}"
            )
            return {
                "status": "error",
                "message": f"HTTP {e.response.status_code}: {error_detail}",
            }
        except Exception as e:
            verbose_logger.exception(f"Error deleting secret from Infisical: {e}")
            return {"status": "error", "message": str(e)}

    async def async_list_secrets(
        self,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously list all secrets in the specified path.
        
        Args:
            optional_params: Additional parameters:
                - project_id: Override default project ID
                - environment: Override default environment
                - secret_path: Override default secret path
            timeout: Request timeout
            
        Returns:
            List of secret metadata (without values)
        """
        optional_params = optional_params or {}
        
        try:
            headers = await self._get_async_request_headers()
            url = self._get_secrets_batch_url(
                project_id=optional_params.get("project_id"),
                environment=optional_params.get("environment"),
                secret_path=optional_params.get("secret_path"),
            )
            
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.SecretManager,
                params={"timeout": timeout},
            )
            
            response = await async_client.get(url=url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            secrets = data.get("secrets", [])
            
            verbose_logger.debug(f"Listed {len(secrets)} secrets from Infisical")
            return secrets
            
        except Exception as e:
            verbose_logger.exception(f"Error listing secrets from Infisical: {e}")
            return []
