import os
from typing import Optional

import litellm
from litellm._logging import verbose_logger
from litellm.caching import InMemoryCache
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import KeyManagementSystem


class HashicorpSecretManager:
    def __init__(self):
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        # Vault-specific config
        self.vault_addr = os.getenv("HCP_VAULT_ADDR", "http://127.0.0.1:8200")
        self.vault_token = os.getenv("HCP_VAULT_TOKEN", "")
        # If your KV engine is mounted somewhere other than "secret", adjust here:
        self.vault_namespace = os.getenv("HCP_VAULT_NAMESPACE", None)

        # Validate environment
        if not self.vault_token:
            raise ValueError(
                "Missing Vault token. Please set VAULT_TOKEN in your environment."
            )

        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
        _refresh_interval = os.environ.get("HCP_VAULT_REFRESH_INTERVAL", 86400)
        _refresh_interval = int(_refresh_interval) if _refresh_interval else 86400
        self.cache = InMemoryCache(
            default_ttl=_refresh_interval
        )  # store in memory for 1 day

        if premium_user is not True:
            raise ValueError(
                f"Hashicorp secret manager is only available for premium users. {CommonProxyErrors.not_premium_user.value}"
            )

    def get_url(self, secret_name: str) -> str:
        _url = f"{self.vault_addr}/v1/"
        if self.vault_namespace:
            _url += f"{self.vault_namespace}/"
        _url += f"secret/data/{secret_name}"
        return _url

    async def async_read_secret(self, secret_name: str) -> Optional[str]:
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

            response = await async_client.get(
                url, headers={"X-Vault-Token": self.vault_token}
            )
            response.raise_for_status()

            # For KV v2, the secret is in response.json()["data"]["data"]
            json_resp = response.json()
            _value = self._get_secret_value_from_json_response(json_resp)
            self.cache.set_cache(secret_name, _value)
            return _value

        except Exception as e:
            verbose_logger.exception(f"Error reading secret from Hashicorp Vault: {e}")
            return None

    def read_secret(self, secret_name: str) -> Optional[str]:
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

            response = sync_client.get(url, headers={"X-Vault-Token": self.vault_token})
            response.raise_for_status()

            # For KV v2, the secret is in response.json()["data"]["data"]
            json_resp = response.json()
            verbose_logger.debug(f"Hashicorp secret manager response: {json_resp}")
            _value = self._get_secret_value_from_json_response(json_resp)
            self.cache.set_cache(secret_name, _value)
            return _value

        except Exception as e:
            verbose_logger.exception(f"Error reading secret from Hashicorp Vault: {e}")
            return None

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
