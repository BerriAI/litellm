"""
Docker Secrets Manager

Reads secrets from files mounted at `/run/secrets/<secret_name>` by Docker
Swarm or `docker run --secret`. The file contents are returned as-is after
stripping surrounding whitespace (like the trailing newline Docker adds)
"""

import os
from typing import Any, Dict, Optional, Union
import httpx
import litellm
from litellm._logging import verbose_logger
from litellm.proxy._types import KeyManagementSystem
from .base_secret_manager import BaseSecretManager


class DockerSecretsManager(BaseSecretManager):
    """
    secrets are mounted by under these default directories
    linux: '/run/secrets/<SECRET_NAME>'
    windows: 'C:\\ProgramData\\Docker\\secrets'
    
    `docker run --secret` mount each secret as a file whose
    name equals the secret name and whose content is the secret value. This
    manager simply reads those files, so no external SDK or credentials are
    required
    """

    @staticmethod
    def _default_secrets_dir() -> str:
        if os.name == "nt":
            return r"C:\ProgramData\Docker\secrets"
        return "/run/secrets"

    def __init__(self, secrets_dir: Optional[str] = None) -> None:
        self.secrets_dir = (
            secrets_dir
            if secrets_dir is not None
            else DockerSecretsManager._default_secrets_dir()
        )

        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.DOCKER_SECRET_MANAGER

    def _secret_path(self, secret_name: str) -> str:
        """Return the absolute filesystem path for *secret_name*."""
        return os.path.join(self.secrets_dir, secret_name)

    def _read_file(self, secret_name: str) -> Optional[str]:
        """
        Read the content of the secret file, stripping surrounding whitespace.

        Returns None when the file does not exist.  Logs a warning and returns
        None on permission errors so the caller can fall back to env vars
        """
        path = self._secret_path(secret_name)
        try:
            with open(path, "r") as fh:
                return fh.read().strip()
        except FileNotFoundError:
            # base case: secret simply isn't present as a Docker secret
            return None
        except PermissionError:
            verbose_logger.warning(
                "DockerSecretsManager: permission denied reading '%s'. "
                "Ensure the process has read access to the secrets directory.",
                path,
            )
            return None

    # ------------------------------------------------------------------
    # BaseSecretManager interface
    # ------------------------------------------------------------------

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        async read a docker secret
        Docker secrets are plain files this method delegates directly to the synchronous file read.
        """
        return self._read_file(secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Synchronously read a Docker secret."""
        return self._read_file(secret_name)

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
        Docker secrets are managed by the Docker daemon and cannot be written
        at runtime via the filesystem.  This operation is intentionally
        unsupported.
        """
        verbose_logger.warning(
            "DockerSecretsManager: write operations are not supported. "
            "Docker secrets are managed by the Docker daemon."
        )
        return {
            "status": "not_supported",
            "message": (
                "DockerSecretsManager does not support write operations. "
                "Manage secrets through the Docker CLI or Swarm."
            ),
        }

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Docker secrets cannot be deleted at runtime via the filesystem.  This
        operation is intentionally unsupported.
        """
        verbose_logger.warning(
            "DockerSecretsManager: delete operations are not supported. "
            "Docker secrets are managed by the Docker daemon."
        )
        return {
            "status": "not_supported",
            "message": (
                "DockerSecretsManager does not support delete operations. "
                "Manage secrets through the Docker CLI or Swarm."
            ),
        }
