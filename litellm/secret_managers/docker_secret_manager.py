"""
Docker Secret Manager

Reads secrets from Docker secrets mounted on the filesystem (default: /run/secrets/).

Supports both Docker Swarm (encrypted at rest + in transit, production-grade) and
Docker Compose (bind-mounted files, development / small self-hosted deployments).

Security properties:
- Path traversal protection: realpath() + prefix check prevents escape from secrets dir
- No secret values are ever logged — only secret names appear in log messages
- Write / delete operations are rejected; Docker secrets are managed by the daemon
- Trailing whitespace is stripped (common "echo vs printf" footgun in secret creation)
- Startup validates the secrets directory exists and warns if it does not
"""

import os
from typing import Any, Dict, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.secret_managers.base_secret_manager import BaseSecretManager
from litellm.types.secret_managers.main import KeyManagementSystem


_DOCKER_SECRETS_DEFAULT_PATH = "/run/secrets"


class DockerSecretManager(BaseSecretManager):
    """
    Secret manager that reads secrets from Docker-mounted secret files.

    Each secret is a plain file at ``<secrets_path>/<secret_name>``.
    The class is intentionally read-only: Docker secrets are managed by the
    Docker daemon and must not be written or deleted by the application.

    Usage (config.yaml):

    .. code-block:: yaml

        general_settings:
          key_management_system: "docker"
          key_management_settings:
            secrets_path: "/run/secrets"   # optional, this is the default
            hosted_keys:
              - OPENAI_API_KEY
              - ANTHROPIC_API_KEY

    Secret names in Docker are lowercase (e.g. ``openai_api_key``), but LiteLLM
    looks up environment-variable-style names in UPPERCASE.  The manager tries
    the exact name first, then a lowercase fallback, so both conventions work.
    """

    def __init__(self, secrets_path: str = _DOCKER_SECRETS_DEFAULT_PATH) -> None:
        self.secrets_path: str = os.path.realpath(secrets_path)
        self._validate_secrets_dir()

        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.DOCKER

        verbose_logger.debug(
            "DockerSecretManager initialised. secrets_path=%s", self.secrets_path
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_secrets_dir(self) -> None:
        """Warn (but do not raise) if the secrets directory does not exist.

        Raising here would crash the proxy on startup for operators who set
        ``key_management_system: docker`` in a non-Docker environment (e.g.
        local dev with env-var fallback).  A warning is sufficient.
        """
        if not os.path.isdir(self.secrets_path):
            verbose_logger.warning(
                "DockerSecretManager: secrets directory %r does not exist. "
                "All secret lookups will return None until the directory is available.",
                self.secrets_path,
            )

    def _resolve_secret_path(self, secret_name: str) -> Optional[str]:
        """Return the absolute path for *secret_name*, or None if not found.

        Tries the exact name first (e.g. ``OPENAI_API_KEY``), then a lowercase
        variant (e.g. ``openai_api_key``).

        Raises ``ValueError`` if the resolved path escapes the secrets directory
        (path traversal attempt).
        """
        # Reject names containing path separators — Docker secret names never
        # include '/' and allowing them would silently enable subdirectory reads.
        if "/" in secret_name or os.sep in secret_name:
            raise ValueError(
                f"DockerSecretManager: secret name {secret_name!r} contains a "
                f"path separator and is not a valid Docker secret name."
            )

        for candidate in _name_candidates(secret_name):
            path = os.path.realpath(os.path.join(self.secrets_path, candidate))
            # Security: ensure the resolved path stays inside the secrets dir
            if not path.startswith(self.secrets_path + os.sep) and path != self.secrets_path:
                raise ValueError(
                    f"DockerSecretManager: secret name {secret_name!r} resolves "
                    f"to a path outside the secrets directory — possible path "
                    f"traversal attempt."
                )
            if os.path.isfile(path):
                return path
        return None

    def _read_file(self, path: str) -> str:
        """Read a secret file and return its value with trailing whitespace stripped.

        Stripping trailing whitespace / newlines prevents the common footgun
        where ``echo "value" | docker secret create name -`` adds a ``\\n``.

        Tries UTF-8 first; falls back to Latin-1 for secrets whose bytes are
        encoded in ISO-8859-1 or Windows-1252 (e.g. passwords with é, ñ, ü, ß).
        Latin-1 maps bytes 0–255 directly so it never raises ``UnicodeDecodeError``.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().rstrip()
        except UnicodeDecodeError:
            verbose_logger.debug(
                "DockerSecretManager: secret at %r is not valid UTF-8; "
                "retrying with Latin-1 (ISO-8859-1)",
                path,
            )
            with open(path, "r", encoding="latin-1") as fh:
                return fh.read().rstrip()

    # ------------------------------------------------------------------
    # BaseSecretManager interface
    # ------------------------------------------------------------------

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Asynchronously read a Docker secret by name."""
        return self.sync_read_secret(secret_name=secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Synchronously read a Docker secret by name.

        Returns the secret value (trailing whitespace stripped), or ``None``
        if the secret file does not exist.  Never raises for a missing secret
        so that ``get_secret()`` can fall back to environment variables.
        """
        try:
            path = self._resolve_secret_path(secret_name)
        except ValueError:
            verbose_logger.exception(
                "DockerSecretManager: path traversal detected for secret %r",
                secret_name,
            )
            raise

        if path is None:
            verbose_logger.debug(
                "DockerSecretManager: secret %r not found in %s",
                secret_name,
                self.secrets_path,
            )
            return None

        try:
            value = self._read_file(path)
            verbose_logger.debug(
                "DockerSecretManager: successfully read secret %r", secret_name
            )
            return value
        except OSError as exc:
            # Permission denied, file disappeared between isfile() and open(), etc.
            verbose_logger.error(
                "DockerSecretManager: failed to read secret %r from %s: %s",
                secret_name,
                path,
                exc,
            )
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
        raise NotImplementedError(
            "DockerSecretManager is read-only. "
            "Docker secrets are managed by the Docker daemon, not the application. "
            "Use `docker secret create` or your compose/stack file to create secrets."
        )

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        raise NotImplementedError(
            "DockerSecretManager is read-only. "
            "Docker secrets are managed by the Docker daemon, not the application. "
            "Use `docker secret rm` to delete secrets."
        )

    @classmethod
    def load_docker_secret_manager(
        cls,
        key_management_settings: Optional[Any] = None,
    ) -> "DockerSecretManager":
        """Factory used by ``initialize_secret_manager`` in the proxy server."""
        secrets_path = _DOCKER_SECRETS_DEFAULT_PATH
        if key_management_settings is not None:
            # `or` guard handles the case where the field exists but is None
            # (KeyManagementSettings.secrets_path is Optional[str] defaulting to None,
            # so getattr returns None rather than the fallback when the attr exists)
            secrets_path = getattr(key_management_settings, "secrets_path", secrets_path) or _DOCKER_SECRETS_DEFAULT_PATH
        return cls(secrets_path=secrets_path)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _name_candidates(secret_name: str):
    """Yield candidate filenames for *secret_name*.

    Docker secret names must be lowercase, but LiteLLM typically uses
    UPPER_CASE env-var style names.  Try exact match first so that
    operators who name their secrets in uppercase (e.g. on Compose) still
    work, then fall back to lowercase.
    """
    yield secret_name
    lower = secret_name.lower()
    if lower != secret_name:
        yield lower
