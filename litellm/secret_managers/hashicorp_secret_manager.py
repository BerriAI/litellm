import os
from typing import Any, Dict, Optional, Union

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


class HashicorpSecretManager(BaseSecretManager):
    def __init__(self):
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        # Vault-specific config
        self.vault_addr = os.getenv("HCP_VAULT_ADDR", "http://127.0.0.1:8200")
        self.vault_token = os.getenv("HCP_VAULT_TOKEN", "")
        # If your KV engine is mounted somewhere other than "secret", adjust here:
        self.vault_namespace = os.getenv("HCP_VAULT_NAMESPACE", None)

        # Optional config for TLS cert auth
        self.tls_cert_path = os.getenv("HCP_VAULT_CLIENT_CERT", "")
        self.tls_key_path = os.getenv("HCP_VAULT_CLIENT_KEY", "")
        self.vault_cert_role = os.getenv("HCP_VAULT_CERT_ROLE", None)

        # Validate environment
        if not self.vault_token:
            raise ValueError(
                "Missing Vault token. Please set HCP_VAULT_TOKEN in your environment."
            )

        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
        _refresh_interval = os.environ.get(
            "HCP_VAULT_REFRESH_INTERVAL", SECRET_MANAGER_REFRESH_INTERVAL
        )
        _refresh_interval = (
            int(_refresh_interval)
            if _refresh_interval
            else SECRET_MANAGER_REFRESH_INTERVAL
        )
        self.cache = InMemoryCache(
            default_ttl=_refresh_interval
        )  # store in memory for 1 day

        if premium_user is not True:
            raise ValueError(
                f"Hashicorp secret manager is only available for premium users. {CommonProxyErrors.not_premium_user.value}"
            )

    def _auth_via_tls_cert(self) -> str:
        """
        Ref: https://developer.hashicorp.com/vault/api-docs/auth/cert

        Request:
        ```
        curl \
            --request POST \
            --cacert vault-ca.pem \
            --cert cert.pem \
            --key key.pem \
            --header "X-Vault-Namespace: mynamespace/" \
            --data '{"name": "my-cert-role"}' \
            https://127.0.0.1:8200/v1/auth/cert/login
        ```

        Response:
        ```
        {
            "auth": {
                "client_token": "cf95f87d-f95b-47ff-b1f5-ba7bff850425",
                "policies": ["web", "stage"],
                "lease_duration": 3600,
                "renewable": true
            }
        }
        ```
        """
        verbose_logger.debug("Using TLS cert auth for Hashicorp Vault")
        # Vault endpoint for cert-based login, e.g. '/v1/auth/cert/login'
        login_url = f"{self.vault_addr}/v1/auth/cert/login"

        # Include your Vault namespace in the header if you're using namespaces.
        # E.g. self.vault_namespace = 'mynamespace/'
        # If you only have root namespace, you can omit this header entirely.
        headers = {}
        if hasattr(self, "vault_namespace") and self.vault_namespace:
            headers["X-Vault-Namespace"] = self.vault_namespace
        try:
            # We use the client cert and key for mutual TLS
            client = httpx.Client(cert=(self.tls_cert_path, self.tls_key_path))
            resp = client.post(
                login_url,
                headers=headers,
                json=self._get_tls_cert_auth_body(),
            )
            resp.raise_for_status()
            token = resp.json()["auth"]["client_token"]
            _lease_duration = resp.json()["auth"]["lease_duration"]
            verbose_logger.info("Successfully obtained Vault token via TLS cert auth.")
            self.cache.set_cache(
                key="hcp_vault_token", value=token, ttl=_lease_duration
            )
            return token
        except Exception as e:
            raise RuntimeError(f"Could not authenticate to Vault via TLS cert: {e}")

    def _get_tls_cert_auth_body(self) -> dict:
        return {"name": self.vault_cert_role}

    def get_url(self, secret_name: str) -> str:
        _url = f"{self.vault_addr}/v1/"
        if self.vault_namespace:
            _url += f"{self.vault_namespace}/"
        _url += f"secret/data/{secret_name}"
        return _url

    def _get_request_headers(self) -> dict:
        if self.tls_cert_path and self.tls_key_path:
            return {"X-Vault-Token": self._auth_via_tls_cert()}
        return {"X-Vault-Token": self.vault_token}

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Reads a secret from Vault KV v2 using an async HTTPX client.
        secret_name is just the path inside the KV mount (e.g., 'myapp/config').
        Returns the entire data dict from data.data, or None on failure.
        """
        if self.cache.get_cache(secret_name) is not None:
            return self.cache.get_cache(secret_name)
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
        )
        try:
            # For KV v2: /v1/<mount>/data/<path>
            # Example: http://127.0.0.1:8200/v1/secret/data/myapp/config
            _url = self.get_url(secret_name)
            url = _url

            response = await async_client.get(url, headers=self._get_request_headers())
            response.raise_for_status()

            # For KV v2, the secret is in response.json()["data"]["data"]
            json_resp = response.json()
            _value = self._get_secret_value_from_json_response(json_resp)
            self.cache.set_cache(secret_name, _value)
            return _value

        except Exception as e:
            verbose_logger.exception(f"Error reading secret from Hashicorp Vault: {e}")
            return None

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Reads a secret from Vault KV v2 using a sync HTTPX client.
        secret_name is just the path inside the KV mount (e.g., 'myapp/config').
        Returns the entire data dict from data.data, or None on failure.
        """
        if self.cache.get_cache(secret_name) is not None:
            return self.cache.get_cache(secret_name)
        sync_client = _get_httpx_client()
        try:
            # For KV v2: /v1/<mount>/data/<path>
            url = self.get_url(secret_name)

            response = sync_client.get(url, headers=self._get_request_headers())
            response.raise_for_status()

            # For KV v2, the secret is in response.json()["data"]["data"]
            json_resp = response.json()
            _value = self._get_secret_value_from_json_response(json_resp)
            self.cache.set_cache(secret_name, _value)
            return _value

        except Exception as e:
            verbose_logger.exception(f"Error reading secret from Hashicorp Vault: {e}")
            return None

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Dict[str, Any]:
        """
        Writes a secret to Vault KV v2 using an async HTTPX client.

        Args:
            secret_name: Path inside the KV mount (e.g., 'myapp/config')
            secret_value: Value to store
            description: Optional description for the secret
            optional_params: Additional parameters to include in the secret data
            timeout: Request timeout

        Returns:
            dict: Response containing status and details of the operation
        """
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"timeout": timeout},
        )

        try:
            url = self.get_url(secret_name)

            # Prepare the secret data
            data = {"data": {"key": secret_value}}

            if description:
                data["data"]["description"] = description

            response = await async_client.post(
                url=url, headers=self._get_request_headers(), json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            verbose_logger.exception(f"Error writing secret to Hashicorp Vault: {e}")
            return {"status": "error", "message": str(e)}

    async def async_rotate_secret(
        self,
        current_secret_name: str,
        new_secret_name: str,
        new_secret_value: str,
        optional_params: Dict | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Dict:
        raise NotImplementedError("Hashicorp does not support secret rotation")

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Async function to delete a secret from Hashicorp Vault.
        In KV v2, this marks the latest version of the secret as deleted.

        Args:
            secret_name: Name of the secret to delete
            recovery_window_in_days: Not used for Vault (Vault handles this internally)
            optional_params: Additional parameters specific to the secret manager
            timeout: Request timeout

        Returns:
            dict: Response containing status and details of the operation
        """
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"timeout": timeout},
        )

        try:
            # For KV v2 delete: /v1/<mount>/data/<path>
            url = self.get_url(secret_name)

            response = await async_client.delete(
                url=url, headers=self._get_request_headers()
            )
            response.raise_for_status()

            # Clear the cache for this secret
            self.cache.delete_cache(secret_name)

            return {
                "status": "success",
                "message": f"Secret {secret_name} deleted successfully",
            }
        except Exception as e:
            verbose_logger.exception(f"Error deleting secret from Hashicorp Vault: {e}")
            return {"status": "error", "message": str(e)}

    def _get_secret_value_from_json_response(
        self, json_resp: Optional[dict]
    ) -> Optional[str]:
        """
        Get the secret value from the JSON response

        Json response from hashicorp vault is of the form:

        {
            "request_id":"036ba77c-018b-31dd-047b-323bcd0cd332",
            "lease_id":"",
            "renewable":false,
            "lease_duration":0,
            "data":
                {"data":
                    {"key":"Vault Is The Way"},
                    "metadata":{"created_time":"2025-01-01T22:13:50.93942388Z","custom_metadata":null,"deletion_time":"","destroyed":false,"version":1}
                },
            "wrap_info":null,
            "warnings":null,
            "auth":null,
            "mount_type":"kv"
        }

        Note: LiteLLM assumes that all secrets are stored as under the key "key"
        """
        if json_resp is None:
            return None
        return json_resp.get("data", {}).get("data", {}).get("key", None)
