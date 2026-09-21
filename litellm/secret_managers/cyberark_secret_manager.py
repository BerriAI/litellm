import base64
import os
from typing import Any, Final
from urllib.parse import quote

import httpx
import yaml

import litellm
from litellm._logging import verbose_logger
from litellm.caching import InMemoryCache
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import KeyManagementSystem

from .base_secret_manager import BaseSecretManager, raise_if_unsafe_secret_name
from .main import str_to_bool


class CyberArkSecretManager(BaseSecretManager):
    def __init__(self):
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        # CyberArk Conjur-specific config
        self.conjur_addr = os.getenv("CYBERARK_API_BASE", "http://127.0.0.1:8080")
        self.conjur_account = os.getenv("CYBERARK_ACCOUNT", "default")
        self.conjur_username = os.getenv("CYBERARK_USERNAME", "admin")
        self.conjur_api_key = os.getenv("CYBERARK_API_KEY", "")

        # Optional config for certificate-based auth
        self.tls_cert_path = os.getenv("CYBERARK_CLIENT_CERT", "")
        self.tls_key_path = os.getenv("CYBERARK_CLIENT_KEY", "")

        # SSL verification - can be disabled for self-signed certificates
        # Set CYBERARK_SSL_VERIFY=false to disable SSL verification
        ssl_verify_env: Final = str_to_bool(os.getenv("CYBERARK_SSL_VERIFY"))
        self.ssl_verify: bool = ssl_verify_env if ssl_verify_env is not None else True

        # Validate environment
        if not self.conjur_api_key and not (self.tls_cert_path and self.tls_key_path):
            raise ValueError(
                "Missing CyberArk credentials. Please set CYBERARK_API_KEY or both CYBERARK_CLIENT_CERT and CYBERARK_CLIENT_KEY in your environment."
            )

        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.CYBERARK

        # Tokens expire after ~8 minutes, so we cache for 5 minutes to be safe
        _refresh_interval: Final = int(os.environ.get("CYBERARK_REFRESH_INTERVAL", "300"))
        self.cache = InMemoryCache(default_ttl=_refresh_interval)

        if premium_user is not True:
            raise ValueError(
                f"CyberArk secret manager is only available for premium users. {CommonProxyErrors.not_premium_user.value}"
            )

        if not self.ssl_verify:
            verbose_logger.warning(
                "CyberArk SSL verification is disabled. This is insecure and should only be used for testing with self-signed certificates."
            )

    def _authenticate(self) -> str:
        """
        Authenticate with CyberArk Conjur and get a session token.

        The token is a JSON object that must be base64-encoded for use in subsequent requests.

        Returns:
            str: Base64-encoded session token
        """
        # Check if we have a cached token
        cached_token: Final = self.cache.get_cache("cyberark_auth_token")
        if cached_token is not None:
            return cached_token

        verbose_logger.debug("Authenticating with CyberArk Conjur...")
        auth_url: Final = f"{self.conjur_addr}/authn/{self.conjur_account}/{self.conjur_username}/authenticate"

        try:
            if self.tls_cert_path and self.tls_key_path:
                # Certificate-based authentication - need custom client for cert
                http_client: Final = httpx.Client(
                    cert=(self.tls_cert_path, self.tls_key_path),
                    verify=self.ssl_verify,
                )
                resp = http_client.post(auth_url, content=self.conjur_api_key)
            else:
                # API key authentication
                http_handler: Final = _get_httpx_client(params={"ssl_verify": self.ssl_verify})
                resp = http_handler.client.post(auth_url, content=self.conjur_api_key)

            resp.raise_for_status()

            # The response is a JSON token that needs to be base64-encoded
            token_json: Final = resp.text
            token_b64: Final = base64.b64encode(token_json.encode()).decode()

            verbose_logger.debug("Successfully authenticated with CyberArk Conjur.")

            # Cache the token for the refresh interval
            self.cache.set_cache(key="cyberark_auth_token", value=token_b64)

            return token_b64
        except Exception as e:
            raise RuntimeError(f"Could not authenticate to CyberArk Conjur: {e}")

    def _get_request_headers(self) -> dict:
        """
        Get headers for CyberArk API requests including authentication.

        Returns:
            dict: Headers with authentication token
        """
        token: Final = self._authenticate()
        return {"Authorization": f'Token token="{token}"'}

    def _ensure_variable_exists(self, secret_name: str) -> None:
        """
        Ensure a variable exists in CyberArk Conjur by creating a policy entry if needed.

        Args:
            secret_name: Name of the variable to ensure exists
        """
        # In production, we'd check if the variable exists first
        # For now, we'll attempt to create it and ignore if it already exists
        raise_if_unsafe_secret_name(secret_name)
        policy_url: Final = f"{self.conjur_addr}/policies/{self.conjur_account}/policy/root"
        # Use a real YAML serializer to build the scalar safely.
        quoted_name: Final = yaml.safe_dump(secret_name, default_style='"').strip()
        policy_yaml: Final = f"- !variable {quoted_name}\n"

        try:
            client: Final = _get_httpx_client(params={"ssl_verify": self.ssl_verify})
            resp: Final = client.client.post(
                policy_url,
                headers={
                    **self._get_request_headers(),
                    "Content-Type": "application/x-yaml",
                },
                content=policy_yaml,
            )
            resp.raise_for_status()
            verbose_logger.debug("Created policy entry for variable: %s", secret_name)
        except httpx.HTTPStatusError as e:
            # Variable might already exist, which is fine
            if e.response.status_code in [409, 422]:
                verbose_logger.debug("Variable %s already exists or policy conflict (expected)", secret_name)
            else:
                verbose_logger.warning(
                    "Could not ensure variable exists: %s - %s", e.response.status_code, e.response.text
                )
        except Exception as e:
            verbose_logger.warning("Error ensuring variable exists: %s", e)

    def get_url(self, secret_name: str) -> str:
        """
        Build the URL for accessing a secret in CyberArk Conjur.

        Args:
            secret_name: Name of the secret (will be URL-encoded)

        Returns:
            str: Full URL for the secret
        """
        # URL-encode the secret name to handle slashes and special characters
        encoded_name: Final = quote(secret_name, safe="")
        return f"{self.conjur_addr}/secrets/{self.conjur_account}/variable/{encoded_name}"

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        """
        Reads a secret from CyberArk Conjur using an async HTTPX client.

        Args:
            secret_name: Name/path of the secret to read
            optional_params: Additional parameters (not used for Conjur)
            timeout: Request timeout

        Returns:
            Optional[str]: The secret value if found, None otherwise
        """
        # Check cache first
        if self.cache.get_cache(secret_name) is not None:
            return self.cache.get_cache(secret_name)

        async_client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"ssl_verify": self.ssl_verify},
        )

        try:
            url: Final = self.get_url(secret_name)
            response: Final = await async_client.get(url, headers=self._get_request_headers())
            response.raise_for_status()

            # CyberArk Conjur returns the raw secret value as text
            secret_value: Final = response.text
            self.cache.set_cache(secret_name, secret_value)
            return secret_value

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                verbose_logger.debug("Secret %s not found in CyberArk Conjur", secret_name)
            else:
                verbose_logger.exception("Error reading secret from CyberArk Conjur: %s", e)
            return None
        except Exception as e:
            verbose_logger.exception("Error reading secret from CyberArk Conjur: %s", e)
            return None

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        """
        Reads a secret from CyberArk Conjur using a sync HTTPX client.

        Args:
            secret_name: Name/path of the secret to read
            optional_params: Additional parameters (not used for Conjur)
            timeout: Request timeout

        Returns:
            Optional[str]: The secret value if found, None otherwise
        """
        # Check cache first
        if self.cache.get_cache(secret_name) is not None:
            return self.cache.get_cache(secret_name)

        sync_client: Final = _get_httpx_client(params={"ssl_verify": self.ssl_verify})

        try:
            url: Final = self.get_url(secret_name)
            response: Final = sync_client.client.get(url, headers=self._get_request_headers())
            response.raise_for_status()

            # CyberArk Conjur returns the raw secret value as text
            secret_value: Final = response.text
            self.cache.set_cache(secret_name, secret_value)
            return secret_value

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                verbose_logger.debug("Secret %s not found in CyberArk Conjur", secret_name)
            else:
                verbose_logger.exception("Error reading secret from CyberArk Conjur: %s", e)
            return None
        except Exception as e:
            verbose_logger.exception("Error reading secret from CyberArk Conjur: %s", e)
            return None

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
        tags: dict | list | None = None,
    ) -> dict[str, Any]:
        """
        Writes a secret to CyberArk Conjur using an async HTTPX client.

        Args:
            secret_name: Name/path of the secret to write
            secret_value: Value to store
            description: Optional description (not used by Conjur)
            optional_params: Additional parameters
            timeout: Request timeout
            tags: Optional tags (not used by Conjur)

        Returns:
            dict: Response containing status and details of the operation
        """
        async_client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"ssl_verify": self.ssl_verify},
        )

        try:
            # Ensure the variable exists in the policy first
            self._ensure_variable_exists(secret_name)

            # Now set the secret value
            url: Final = self.get_url(secret_name)
            response = await async_client.post(url=url, headers=self._get_request_headers(), content=secret_value)
            response.raise_for_status()

            # Update cache
            self.cache.set_cache(secret_name, secret_value)

            return {
                "status": "success",
                "message": f"Secret {secret_name} written successfully",
            }
        except Exception as e:
            verbose_logger.exception("Error writing secret to CyberArk Conjur: %s", e)
            return {"status": "error", "message": str(e)}

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: int | None = 7,
        optional_params: dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> dict:
        """
        CyberArk Conjur does not support direct secret deletion via API.
        Secrets can only be removed through policy updates.

        Args:
            secret_name: Name of the secret
            recovery_window_in_days: Not used
            optional_params: Additional parameters
            timeout: Request timeout

        Returns:
            dict: Response indicating operation not supported
        """
        verbose_logger.warning(
            "CyberArk Conjur does not support direct secret deletion. Secrets must be removed through policy updates."
        )

        # Clear from cache
        self.cache.delete_cache(secret_name)

        return {
            "status": "not_supported",
            "message": "CyberArk Conjur does not support direct secret deletion. Use policy updates to remove variables.",
        }
