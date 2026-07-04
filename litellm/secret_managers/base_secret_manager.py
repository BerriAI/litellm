import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import httpx

from litellm import verbose_logger

# U+0085 (NEL), U+2028 (LINE SEPARATOR), and U+2029 (PARAGRAPH SEPARATOR) are line
# breaks under the YAML spec, same as backslash-n or backslash-r, so a denylist
# restricted to ASCII control characters is bypassable for YAML-based sinks.
_UNSAFE_SECRET_NAME_PATTERN = re.compile(r"\.\.|[\x00-\x1f\x7f-\x9f  ]")


def raise_if_unsafe_secret_name(secret_name: str) -> None:
    """
    Reject secret names that could path-traverse a secret manager's API (e.g. HashiCorp
    Vault, which builds its request URL by string concatenation) or inject control
    characters into a policy document (e.g. CyberArk Conjur, which embeds the secret
    name directly into a YAML policy body).

    Rejects any ".." substring rather than only a ".." path segment, since that is
    the only traversal-safe answer for Vault's raw string-concatenated URL; a
    version-like name such as "release-1.0..2" is rejected too.
    """
    if _UNSAFE_SECRET_NAME_PATTERN.search(secret_name):
        raise ValueError(f"Unsafe secret_name {secret_name!r}: must not contain '..' or control characters")


class BaseSecretManager(ABC):
    """
    Abstract base class for secret management implementations.
    """

    @abstractmethod
    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Asynchronously read a secret from the secret manager.

        Args:
            secret_name (str): Name/path of the secret to read
            optional_params (Optional[dict]): Additional parameters specific to the secret manager
            timeout (Optional[Union[float, httpx.Timeout]]): Request timeout

        Returns:
            Optional[str]: The secret value if found, None otherwise
        """
        pass

    @abstractmethod
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Synchronously read a secret from the secret manager.

        Args:
            secret_name (str): Name/path of the secret to read
            optional_params (Optional[dict]): Additional parameters specific to the secret manager
            timeout (Optional[Union[float, httpx.Timeout]]): Request timeout

        Returns:
            Optional[str]: The secret value if found, None otherwise
        """
        pass

    @abstractmethod
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
        Asynchronously write a secret to the secret manager.

        Args:
            secret_name (str): Name/path of the secret to write
            secret_value (str): Value to store
            description (Optional[str]): Description of the secret. Some secret managers allow storing a description with the secret.
            optional_params (Optional[dict]): Additional parameters specific to the secret manager
            timeout (Optional[Union[float, httpx.Timeout]]): Request timeout
            tags: Optional dict or list of tags to apply, e.g.
                  {"Environment": "Prod", "Owner": "AI-Platform"} or
                  [{"Key": "Environment", "Value": "Prod"}]
        Returns:
            Dict[str, Any]: Response from the secret manager containing write operation details
        """
        pass

    @abstractmethod
    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Async function to delete a secret from the secret manager

        Args:
            secret_name: Name of the secret to delete
            recovery_window_in_days: Number of days before permanent deletion (default: 7)
            optional_params: Additional parameters specific to the secret manager
            timeout: Request timeout

        Returns:
            dict: Response from the secret manager containing deletion details
        """
        pass

    async def async_rotate_secret(
        self,
        current_secret_name: str,
        new_secret_name: str,
        new_secret_value: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Async function to rotate a secret by creating a new one and deleting the old one.
        This allows for both value and name changes during rotation.

        Args:
            current_secret_name: Current name of the secret
            new_secret_name: New name for the secret
            new_secret_value: New value for the secret
            optional_params: Additional AWS parameters
            timeout: Request timeout

        Returns:
            dict: Response containing the new secret details

        Raises:
            ValueError: If the secret doesn't exist or if there's an HTTP error
        """
        try:
            # First verify the old secret exists
            old_secret = await self.async_read_secret(
                secret_name=current_secret_name,
                optional_params=optional_params,
                timeout=timeout,
            )

            if old_secret is None:
                raise ValueError(f"Current secret {current_secret_name} not found")

            # Create new secret with new name and value
            create_response = await self.async_write_secret(
                secret_name=new_secret_name,
                secret_value=new_secret_value,
                description=f"Rotated from {current_secret_name}",
                optional_params=optional_params,
                timeout=timeout,
            )

            # Verify new secret was created successfully
            new_secret = await self.async_read_secret(
                secret_name=new_secret_name,
                optional_params=optional_params,
                timeout=timeout,
            )

            if new_secret is None:
                raise ValueError(f"Failed to verify new secret {new_secret_name}")

            # If everything is successful, delete the old secret
            await self.async_delete_secret(
                secret_name=current_secret_name,
                recovery_window_in_days=7,  # Keep for recovery if needed
                optional_params=optional_params,
                timeout=timeout,
            )

            return create_response

        except httpx.HTTPStatusError as err:
            verbose_logger.exception(
                "Error rotating secret in AWS Secrets Manager: %s",
                str(err.response.text),
            )
            raise ValueError(f"HTTP error occurred: {err.response.text}")
        except httpx.TimeoutException:
            raise ValueError("Timeout error occurred")
        except Exception as e:
            verbose_logger.exception("Error rotating secret in AWS Secrets Manager: %s", str(e))
            raise
