import os
from typing import Optional

import litellm
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import KeyManagementSystem


class HashicorpSecretManager:
    def __init__(self):
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

    def get_url(self, secret_name: str) -> str:
        _url = f"{self.vault_addr}/v1/"
        if self.vault_namespace:
            _url += f"{self.vault_namespace}/"
        _url += f"secret/data/{secret_name}"
        return _url

    async def async_read_secret(self, secret_name: str) -> Optional[dict]:
        """
        Reads a secret from Vault KV v2 using an async HTTPX client.
        secret_name is just the path inside the KV mount (e.g., 'myapp/config').
        Returns the entire data dict from data.data, or None on failure.
        """
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
            return json_resp["data"]["data"]

        except Exception as e:
            return None

    def read_secret(self, secret_name: str) -> Optional[dict]:
        """
        Reads a secret from Vault KV v2 using a sync HTTPX client.
        secret_name is just the path inside the KV mount (e.g., 'myapp/config').
        Returns the entire data dict from data.data, or None on failure.
        """
        sync_client = _get_httpx_client()
        try:
            # For KV v2: /v1/<mount>/data/<path>
            url = self.get_url(secret_name)

            response = sync_client.get(url, headers={"X-Vault-Token": self.vault_token})
            response.raise_for_status()

            # For KV v2, the secret is in response.json()["data"]["data"]
            json_resp = response.json()
            return json_resp["data"]["data"]

        except Exception as e:
            return None
