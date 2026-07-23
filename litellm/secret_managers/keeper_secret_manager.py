"""
Keeper Secrets Manager (KSM) integration.

Read-only support for resolving provider credentials (e.g. an OpenAI or Gemini
api_key) that live in a Keeper vault. Secrets are addressed with Keeper Notation
(https://docs.keeper.io/en/keeperpam/secrets-manager/about/keeper-notation),
for example ``<record_uid>/field/password``.

Configuration (environment variables):
* ``KSM_CONFIG``   base64 KSM device configuration (preferred, persistent)
* ``KSM_TOKEN``    one-time access token used to bind a new device
* ``KSM_HOSTNAME`` optional region hostname (e.g. ``keepersecurity.eu``)
"""

import asyncio
import os
from typing import Protocol, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.caching import InMemoryCache
from litellm.constants import SECRET_MANAGER_REFRESH_INTERVAL
from litellm.proxy._types import KeyManagementSystem

from .base_secret_manager import BaseSecretManager

_READ_ONLY_RESPONSE: dict[str, str] = {
    "status": "not_supported",
    "message": "Keeper Secrets Manager integration is read-only.",
}


class _KeeperClient(Protocol):
    def get_notation_results(self, notation: str) -> list[str]: ...


class KeeperSecretManager(BaseSecretManager):
    def __init__(self, client: _KeeperClient | None = None):
        self.config_b64 = os.getenv("KSM_CONFIG")
        self.token = os.getenv("KSM_TOKEN")
        self.hostname = os.getenv("KSM_HOSTNAME")

        self._client: _KeeperClient = client if client is not None else self._build_client()

        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.KEEPER

        _refresh_interval = os.environ.get("KSM_REFRESH_INTERVAL", SECRET_MANAGER_REFRESH_INTERVAL)
        _refresh_interval = int(_refresh_interval) if _refresh_interval else SECRET_MANAGER_REFRESH_INTERVAL
        self.cache = InMemoryCache(default_ttl=_refresh_interval)

    def _build_client(self) -> _KeeperClient:
        if not self.config_b64 and not self.token:
            raise ValueError(
                "Missing Keeper Secrets Manager credentials. Set either:\n"
                "  - KSM_CONFIG (base64 device configuration), or\n"
                "  - KSM_TOKEN (one-time access token)"
            )

        try:
            from keeper_secrets_manager_core import SecretsManager
            from keeper_secrets_manager_core.storage import InMemoryKeyValueStorage
        except ImportError as e:
            raise ImportError(
                "keeper-secrets-manager-core is not installed. Run `pip install keeper-secrets-manager-core`."
            ) from e

        storage = InMemoryKeyValueStorage(self.config_b64) if self.config_b64 else InMemoryKeyValueStorage()
        client = SecretsManager(token=self.token, hostname=self.hostname, config=storage)
        return cast(_KeeperClient, client)  # cast-ok: KSM SDK has no stubs; _KeeperClient pins the one method used

    def _read_notation(self, secret_name: str) -> str | None:
        cached: str | None = self.cache.get_cache(secret_name)
        if cached is not None:
            return cached
        try:
            results = self._client.get_notation_results(secret_name)
        except Exception as e:  # noqa: BLE001  # KSM SDK raises varied errors; a failed lookup is a miss
            verbose_logger.exception(f"Error reading secret from Keeper Secrets Manager: {e}")
            return None
        if not results:
            verbose_logger.debug(f"Secret {secret_name} not found in Keeper Secrets Manager")
            return None
        value = results[0]
        self.cache.set_cache(secret_name, value)
        return value

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return await asyncio.to_thread(self._read_notation, secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> str | None:
        return self._read_notation(secret_name)

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        tags: dict[str, object] | list[object] | None = None,
    ) -> dict[str, str]:
        return dict(_READ_ONLY_RESPONSE)

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: int | None = 7,
        optional_params: dict[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> dict[str, str]:
        return dict(_READ_ONLY_RESPONSE)
