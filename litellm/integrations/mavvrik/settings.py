"""Settings management for the Mavvrik integration.

Consolidates all configuration concerns:
  - Config detection (env vars or database)
  - Persistence (load/save/delete via LiteLLM_Config table)
  - Encryption/decryption of the API key

The export marker (cursor) is owned exclusively by the Mavvrik API.
On each scheduled run, MavvrikOrchestrator calls client.register() to
retrieve the current metricsMarker from Mavvrik — no local marker
storage is needed.
"""

import json
import os
from typing import Optional

from litellm._logging import verbose_logger

_CONFIG_KEY = "mavvrik_settings"

_ENV_VARS = (
    "MAVVRIK_API_KEY",
    "MAVVRIK_API_ENDPOINT",
    "MAVVRIK_CONNECTION_ID",
)


class MavvrikSettings:
    """Manages Mavvrik configuration: detection, persistence, and encryption.

    Usage::

        settings = MavvrikSettings()
        if await settings.is_setup():
            data = await settings.load()  # api_key already decrypted
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config_key(self) -> str:
        """The LiteLLM_Config row key used to store Mavvrik settings."""
        return _CONFIG_KEY

    @property
    def has_env_vars(self) -> bool:
        """Return True when all three required env vars are non-empty."""
        return all(os.getenv(v, "").strip() for v in _ENV_VARS)

    @property
    def _prisma_client(self):
        """Lazy import of the prisma_client singleton.

        Returns None when the proxy database is not connected (e.g. in tests
        or when running without a database backend).
        """
        try:
            from litellm.proxy.proxy_server import prisma_client

            return prisma_client
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Setup detection
    # ------------------------------------------------------------------

    async def is_setup(self) -> bool:
        """Return True if Mavvrik credentials exist in env vars or the database."""
        if self.has_env_vars:
            return True

        client = self._prisma_client
        if client is None:
            return False

        try:
            row = await client.db.litellm_config.find_first(
                where={"param_name": _CONFIG_KEY}
            )
            return row is not None and row.param_value is not None
        except Exception as exc:
            verbose_logger.debug("MavvrikSettings.is_setup: DB check failed — %s", exc)
            return False

    # ------------------------------------------------------------------
    # Load / Save / Delete
    # ------------------------------------------------------------------

    async def load(self) -> dict:
        """Load and decrypt Mavvrik settings from the database.

        Returns an empty dict when no row exists.
        The ``api_key`` field is returned in plaintext (decrypted).
        """
        client = self._ensure_prisma_client()

        row = await client.db.litellm_config.find_first(
            where={"param_name": _CONFIG_KEY}
        )
        if row is None or row.param_value is None:
            return {}

        value = row.param_value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}

        if not isinstance(value, dict):
            return {}

        encrypted_key: Optional[str] = value.get("api_key")
        if encrypted_key:
            decrypted = self.decrypt_value_helper(encrypted_key, key="mavvrik_api_key")
            if decrypted is not None:
                value["api_key"] = decrypted

        return value

    async def save(
        self,
        api_key: str,
        api_endpoint: str,
        connection_id: str,
    ) -> None:
        """Encrypt the API key and persist credentials to LiteLLM_Config.

        The export marker (cursor) is owned exclusively by the Mavvrik API
        and is NOT stored locally — it is retrieved via client.register()
        at the start of each scheduled run.
        """
        encrypted_api_key: str = self.encrypt_value_helper(api_key)
        settings: dict = {
            "api_key": encrypted_api_key,
            "api_endpoint": api_endpoint,
            "connection_id": connection_id,
        }
        await self._upsert(settings)

    async def delete(self) -> None:
        """Remove the Mavvrik settings row from LiteLLM_Config.

        Raises:
            LookupError: When no Mavvrik settings row exists in the database.
        """
        client = self._ensure_prisma_client()

        row = await client.db.litellm_config.find_first(
            where={"param_name": _CONFIG_KEY}
        )
        if row is None or row.param_value is None:
            raise LookupError("Mavvrik settings not found — nothing to delete.")

        await client.db.litellm_config.delete(where={"param_name": _CONFIG_KEY})
        verbose_logger.info("MavvrikSettings: settings row deleted")

    # ------------------------------------------------------------------
    # Encryption helpers (owned here so callers never touch utils directly)
    # ------------------------------------------------------------------

    def encrypt_value_helper(self, value: str) -> str:
        """Encrypt a plaintext string using the LiteLLM salt key."""
        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            encrypt_value_helper as _encrypt,
        )

        return _encrypt(value)

    def decrypt_value_helper(
        self, value: str, key: str = "mavvrik_api_key"
    ) -> Optional[str]:
        """Decrypt an encrypted string using the LiteLLM salt key.

        Returns None when decryption fails (e.g. salt key mismatch).
        """
        from litellm.proxy.common_utils.encrypt_decrypt_utils import (
            decrypt_value_helper as _decrypt,
        )

        return _decrypt(value, key=key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_prisma_client(self):
        """Return the prisma_client or raise if the database is not connected."""
        client = self._prisma_client
        if client is None:
            raise Exception(
                "Database not connected. Connect a database to your proxy — "
                "https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        return client

    async def _upsert(self, settings: dict) -> None:
        """Write (create or update) the settings row in LiteLLM_Config."""
        client = self._ensure_prisma_client()
        payload = json.dumps(settings)
        await client.db.litellm_config.upsert(
            where={"param_name": _CONFIG_KEY},
            data={
                "create": {"param_name": _CONFIG_KEY, "param_value": payload},
                "update": {"param_value": payload},
            },
        )
