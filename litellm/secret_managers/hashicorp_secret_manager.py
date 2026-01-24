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
        # Vault namespace (for X-Vault-Namespace header)
        self.vault_namespace = os.getenv("HCP_VAULT_NAMESPACE", None)
        # KV engine mount name (default: "secret")
        # If your KV engine is mounted somewhere other than "secret", set HCP_VAULT_MOUNT_NAME
        self.vault_mount_name = os.getenv("HCP_VAULT_MOUNT_NAME", "secret")
        # Optional path prefix for secrets (e.g., "myapp" -> secret/data/myapp/{secret_name})
        self.vault_path_prefix = os.getenv("HCP_VAULT_PATH_PREFIX", None)

        # Optional config for TLS cert auth
        self.tls_cert_path = os.getenv("HCP_VAULT_CLIENT_CERT", "")
        self.tls_key_path = os.getenv("HCP_VAULT_CLIENT_KEY", "")
        self.vault_cert_role = os.getenv("HCP_VAULT_CERT_ROLE", None)

        # Optional config for AppRole auth
        self.approle_role_id = os.getenv("HCP_VAULT_APPROLE_ROLE_ID", "")
        self.approle_secret_id = os.getenv("HCP_VAULT_APPROLE_SECRET_ID", "")
        self.approle_mount_path = os.getenv("HCP_VAULT_APPROLE_MOUNT_PATH", "approle")

        self._verify_required_credentials_exist()

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

    def _verify_required_credentials_exist(self) -> None:
        """
        Validate that at least one authentication method is configured.

        Raises:
            ValueError: If no valid authentication credentials are provided
        """
        if not self.vault_token and not (
            self.approle_role_id and self.approle_secret_id
        ):
            raise ValueError(
                "Missing Vault authentication credentials. Please set either:\n"
                "  - HCP_VAULT_TOKEN for token-based auth, or\n"
                "  - HCP_VAULT_APPROLE_ROLE_ID and HCP_VAULT_APPROLE_SECRET_ID for AppRole auth"
            )

    def _auth_via_approle(self) -> str:
        """
        Authenticate to Vault using AppRole auth method.
        
        Ref: https://developer.hashicorp.com/vault/api-docs/auth/approle

        Request:
        ```
        curl \
            --request POST \
            --header "X-Vault-Namespace: mynamespace/" \
            --data '{"role_id": "...", "secret_id": "..."}' \
            http://127.0.0.1:8200/v1/auth/approle/login
        ```

        Response:
        ```
        {
            "auth": {
                "client_token": "hvs.CAESI...",
                "accessor": "hmac-sha256...",
                "policies": ["default", "dev-policy"],
                "token_policies": ["default", "dev-policy"],
                "lease_duration": 2764800,
                "renewable": true
            }
        }
        ```
        """
        verbose_logger.debug("Using AppRole auth for Hashicorp Vault")

        # Check cache first
        cached_token = self.cache.get_cache(key="hcp_vault_approle_token")
        if cached_token:
            verbose_logger.debug("Using cached Vault token from AppRole auth")
            return cached_token

        # Vault endpoint for AppRole login
        login_url = f"{self.vault_addr}/v1/auth/{self.approle_mount_path}/login"

        headers = {}
        if hasattr(self, "vault_namespace") and self.vault_namespace:
            headers["X-Vault-Namespace"] = self.vault_namespace

        try:
            client = _get_httpx_client()
            resp = client.post(
                url=login_url,
                headers=headers,
                json={
                    "role_id": self.approle_role_id,
                    "secret_id": self.approle_secret_id,
                },
            )
            resp.raise_for_status()

            auth_data = resp.json()["auth"]
            token = auth_data["client_token"]
            _lease_duration = auth_data["lease_duration"]

            verbose_logger.debug(
                f"Successfully obtained Vault token via AppRole auth. Lease duration: {_lease_duration}s"
            )

            # Cache the token with its lease duration
            self.cache.set_cache(
                key="hcp_vault_approle_token", value=token, ttl=_lease_duration
            )
            return token
        except Exception as e:
            raise RuntimeError(f"Could not authenticate to Vault via AppRole: {e}")

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
            verbose_logger.debug("Successfully obtained Vault token via TLS cert auth.")
            self.cache.set_cache(
                key="hcp_vault_token", value=token, ttl=_lease_duration
            )
            return token
        except Exception as e:
            raise RuntimeError(f"Could not authenticate to Vault via TLS cert: {e}")

    def _get_tls_cert_auth_body(self) -> dict:
        return {"name": self.vault_cert_role}

    def get_url(
        self,
        secret_name: str,
        namespace: Optional[str] = None,
        mount_name: Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> str:
        """
        Constructs the Vault URL for KV v2 secrets.

        Format: {VAULT_ADDR}/v1/{NAMESPACE}/{MOUNT_NAME}/data/{PATH_PREFIX}/{SECRET_NAME}

        Examples:
        - Default: http://127.0.0.1:8200/v1/secret/data/mykey
        - With namespace: http://127.0.0.1:8200/v1/mynamespace/secret/data/mykey
        - With custom mount: http://127.0.0.1:8200/v1/kv/data/mykey
        - With path prefix: http://127.0.0.1:8200/v1/secret/data/myapp/mykey
        """
        resolved_namespace = self._sanitize_path_component(
            namespace if namespace is not None else self.vault_namespace
        )
        resolved_mount = self._sanitize_path_component(
            mount_name if mount_name is not None else self.vault_mount_name
        )
        if resolved_mount is None:
            resolved_mount = "secret"
        resolved_path_prefix = self._sanitize_path_component(
            path_prefix if path_prefix is not None else self.vault_path_prefix
        )

        _url = f"{self.vault_addr}/v1/"
        if resolved_namespace:
            _url += f"{resolved_namespace}/"
        _url += f"{resolved_mount}/data/"
        if resolved_path_prefix:
            _url += f"{resolved_path_prefix}/"
        _url += secret_name
        return _url

    def _sanitize_plain_value(self, value: Optional[Union[str, int]]) -> Optional[str]:
        if value is None:
            return None
        value_str = str(value).strip()
        if value_str == "":
            return None
        return value_str

    def _sanitize_path_component(
        self, value: Optional[Union[str, int]]
    ) -> Optional[str]:
        sanitized_value = self._sanitize_plain_value(value)
        if sanitized_value is None:
            return None
        sanitized_value = sanitized_value.strip("/")
        return sanitized_value or None

    def _extract_secret_manager_settings(
        self, optional_params: Optional[dict]
    ) -> Dict[str, Any]:
        if not isinstance(optional_params, dict):
            return {}

        candidate = optional_params.get("secret_manager_settings")
        source = candidate if isinstance(candidate, dict) else optional_params
        allowed_keys = {"namespace", "mount", "path_prefix", "data"}
        return {k: source[k] for k in allowed_keys if k in source}

    def _build_secret_target(
        self, secret_name: str, optional_params: Optional[dict]
    ) -> Dict[str, Any]:
        settings = self._extract_secret_manager_settings(optional_params)

        namespace = settings.get("namespace", self.vault_namespace)
        mount = settings.get("mount", self.vault_mount_name)
        path_prefix = settings.get("path_prefix", self.vault_path_prefix)
        data_key_override = settings.get("data")

        data_key = self._sanitize_plain_value(data_key_override) or "key"

        url = self.get_url(
            secret_name=secret_name,
            namespace=namespace,
            mount_name=mount,
            path_prefix=path_prefix,
        )

        return {
            "url": url,
            "data_key": data_key,
            "secret_name": secret_name,
        }

    def _get_request_headers(self) -> dict:
        """
        Get the headers for Vault API requests.

        Authentication priority:
        1. AppRole (if role_id and secret_id are configured)
        2. TLS Certificate (if cert paths are configured)
        3. Direct token (if HCP_VAULT_TOKEN is set)
        """
        # Priority 1: AppRole auth
        if self.approle_role_id and self.approle_secret_id:
            return {"X-Vault-Token": self._auth_via_approle()}

        # Priority 2: TLS cert auth
        if self.tls_cert_path and self.tls_key_path:
            return {"X-Vault-Token": self._auth_via_tls_cert()}

        # Priority 3: Direct token
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
        tags: Optional[Union[dict, list]] = None,
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
            target = self._build_secret_target(secret_name, optional_params)

            # Prepare the secret data
            data = {"data": {target["data_key"]: secret_value}}

            if description:
                data["data"]["description"] = description

            response = await async_client.post(
                url=target["url"],
                headers=self._get_request_headers(),
                json=data,
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
        """
        Rotates a secret by creating a new one and deleting the old one.
        Uses _build_secret_target to handle optional_params for namespace, mount, path_prefix customization.

        Args:
            current_secret_name: Current name of the secret
            new_secret_name: New name for the secret
            new_secret_value: New value for the secret
            optional_params: Additional parameters (namespace, mount, path_prefix, data)
            timeout: Request timeout

        Returns:
            dict: Response containing status and details of the operation.
                  On success, returns the response from async_write_secret.
                  On error, returns {"status": "error", "message": "error message"}
        """
        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.SecretManager,
            params={"timeout": timeout},
        )

        try:
            # First verify the old secret exists using _build_secret_target
            current_target = self._build_secret_target(current_secret_name, optional_params)
            try:
                response = await async_client.get(
                    url=current_target["url"],
                    headers=self._get_request_headers(),
                )
                response.raise_for_status()
                # Secret exists, we can proceed
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    verbose_logger.exception(f"Current secret {current_secret_name} not found")
                    return {"status": "error", "message": f"Current secret {current_secret_name} not found"}
                verbose_logger.exception(
                    f"Error checking current secret existence: {e.response.text if hasattr(e, 'response') else str(e)}"
                )
                return {
                    "status": "error",
                    "message": f"HTTP error occurred while checking current secret: {e.response.text if hasattr(e, 'response') else str(e)}",
                }
            except Exception as e:
                verbose_logger.exception(f"Error checking current secret existence: {e}")
                return {"status": "error", "message": f"Error checking current secret: {e}"}

            # Create new secret with new name and value
            # Use _build_secret_target to handle optional_params
            create_response = await self.async_write_secret(
                secret_name=new_secret_name,
                secret_value=new_secret_value,
                description=f"Rotated from {current_secret_name}",
                optional_params=optional_params,
                timeout=timeout,
            )

            # Check if async_write_secret returned an error
            if isinstance(create_response, dict) and create_response.get("status") == "error":
                return create_response

            # Verify new secret was created successfully using _build_secret_target
            new_target = self._build_secret_target(new_secret_name, optional_params)
            try:
                response = await async_client.get(
                    url=new_target["url"],
                    headers=self._get_request_headers(),
                )
                response.raise_for_status()
                json_resp = response.json()
                # Use data_key from target to get the correct value
                data_key = new_target["data_key"]
                new_secret_value_from_vault = json_resp.get("data", {}).get("data", {}).get(data_key, None)
                if new_secret_value_from_vault != new_secret_value:
                    verbose_logger.exception(
                        f"New secret value mismatch. Expected: {new_secret_value}, Got: {new_secret_value_from_vault}"
                    )
                    return {
                        "status": "error",
                        "message": f"New secret value mismatch. Expected: {new_secret_value}, Got: {new_secret_value_from_vault}",
                    }
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    verbose_logger.exception(f"Failed to verify new secret {new_secret_name}")
                    return {"status": "error", "message": f"Failed to verify new secret {new_secret_name}"}
                verbose_logger.exception(
                    f"Error verifying new secret: {e.response.text if hasattr(e, 'response') else str(e)}"
                )
                return {
                    "status": "error",
                    "message": f"HTTP error occurred while verifying new secret: {e.response.text if hasattr(e, 'response') else str(e)}",
                }
            except Exception as e:
                verbose_logger.exception(f"Error verifying new secret: {e}")
                return {"status": "error", "message": f"Error verifying new secret: {e}"}

            # If everything is successful, delete the old secret
            # Only delete if the names are different (same name means we're just updating the value)
            if current_secret_name != new_secret_name:
                delete_response = await self.async_delete_secret(
                    secret_name=current_secret_name,
                    recovery_window_in_days=7,  # Keep for recovery if needed
                    optional_params=optional_params,
                    timeout=timeout,
                )
                # Check if async_delete_secret returned an error
                if isinstance(delete_response, dict) and delete_response.get("status") == "error":
                    # Log the error but don't fail the rotation since new secret was created successfully
                    verbose_logger.warning(
                        f"Failed to delete old secret {current_secret_name} after rotation: {delete_response.get('message')}"
                    )
                else:
                    # Clear cache for the old secret only if deletion was successful
                    self.cache.delete_cache(current_secret_name)

            # Clear cache for the new secret (or updated secret if names are the same)
            self.cache.delete_cache(new_secret_name)

            return create_response

        except httpx.TimeoutException:
            verbose_logger.exception("Timeout error occurred during secret rotation")
            return {"status": "error", "message": "Timeout error occurred"}
        except Exception as e:
            verbose_logger.exception(f"Error rotating secret in Hashicorp Vault: {e}")
            return {"status": "error", "message": str(e)}

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
            target = self._build_secret_target(secret_name, optional_params)
            response = await async_client.delete(
                url=target["url"], headers=self._get_request_headers()
            )
            response.raise_for_status()

            # Clear the cache for this secret
            self.cache.delete_cache(secret_name)
            if target["secret_name"] != secret_name:
                self.cache.delete_cache(target["secret_name"])

            return {
                "status": "success",
                "message": f"Secret {target['secret_name']} deleted successfully",
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
