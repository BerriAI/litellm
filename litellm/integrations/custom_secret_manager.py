"""
Custom Secret Manager Integration

This module provides a base class for implementing custom secret managers in LiteLLM.

Usage:
    from litellm.integrations.custom_secret_manager import CustomSecretManager

    class MySecretManager(CustomSecretManager):
        def __init__(self):
            super().__init__(secret_manager_name="my_secret_manager")

        async def async_read_secret(
            self,
            secret_name: str,
            optional_params=None,
            timeout=None,
        ):
            # Your implementation here
            return await self._fetch_secret_from_service(secret_name)

        def sync_read_secret(
            self,
            secret_name: str,
            optional_params=None,
            timeout=None,
        ):
            # Your implementation here
            return self._fetch_secret_from_service_sync(secret_name)

    # Set your custom secret manager
    import litellm
    from litellm.types.secret_managers.main import KeyManagementSystem

    litellm.secret_manager_client = MySecretManager()
    litellm._key_management_system = KeyManagementSystem.CUSTOM
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.secret_managers.base_secret_manager import BaseSecretManager


class CustomSecretManager(BaseSecretManager):
    """
    Base class for implementing custom secret managers.

    This class provides a standard interface for implementing custom secret management
    integrations in LiteLLM. Users can extend this class to integrate their own secret
    management systems.

    Example:
        ```python
        from litellm.integrations.custom_secret_manager import CustomSecretManager

        class MyVaultSecretManager(CustomSecretManager):
            def __init__(self, vault_url: str, token: str):
                super().__init__(secret_manager_name="my_vault")
                self.vault_url = vault_url
                self.token = token

            async def async_read_secret(self, secret_name: str, optional_params=None, timeout=None):
                # Implementation for reading secrets from your vault
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.vault_url}/v1/secret/{secret_name}",
                        headers={"X-Vault-Token": self.token},
                        timeout=timeout
                    )
                    return response.json()["data"]["value"]

            def sync_read_secret(self, secret_name: str, optional_params=None, timeout=None):
                # Sync implementation
                with httpx.Client() as client:
                    response = client.get(
                        f"{self.vault_url}/v1/secret/{secret_name}",
                        headers={"X-Vault-Token": self.token},
                        timeout=timeout
                    )
                    return response.json()["data"]["value"]
        ```
    """

    def __init__(
        self,
        secret_manager_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the CustomSecretManager.

        Args:
            secret_manager_name: A descriptive name for your secret manager.
                                This is used for logging and debugging purposes.
            **kwargs: Additional keyword arguments to pass to your secret manager.
        """
        super().__init__()
        self.secret_manager_name = secret_manager_name or "custom_secret_manager"
        verbose_logger.info(
            "Initialized custom secret manager"
        )

    @abstractmethod
    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Asynchronously read a secret from your custom secret manager.

        Args:
            secret_name: Name/path of the secret to read
            optional_params: Additional parameters specific to your secret manager
            timeout: Request timeout

        Returns:
            The secret value if found, None otherwise

        Raises:
            Exception: If there's an error reading the secret
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
        Synchronously read a secret from your custom secret manager.

        Args:
            secret_name: Name/path of the secret to read
            optional_params: Additional parameters specific to your secret manager
            timeout: Request timeout

        Returns:
            The secret value if found, None otherwise

        Raises:
            Exception: If there's an error reading the secret
        """
        pass

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
        Asynchronously write a secret to your custom secret manager.

        This is optional to implement. If your secret manager supports writing secrets,
        you can override this method.

        Args:
            secret_name: Name/path of the secret to write
            secret_value: Value to store
            description: Description of the secret
            optional_params: Additional parameters specific to your secret manager
            timeout: Request timeout
            tags: Optional tags to apply to the secret

        Returns:
            Response from the secret manager containing write operation details

        Raises:
            NotImplementedError: If write operations are not supported
        """
        raise NotImplementedError(
            f"Write operations are not implemented for {self.secret_manager_name}. "
            "Override async_write_secret() to add write support."
        )

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Asynchronously delete a secret from your custom secret manager.

        This is optional to implement. If your secret manager supports deleting secrets,
        you can override this method.

        Args:
            secret_name: Name of the secret to delete
            recovery_window_in_days: Number of days before permanent deletion (if supported)
            optional_params: Additional parameters specific to your secret manager
            timeout: Request timeout

        Returns:
            Response from the secret manager containing deletion details

        Raises:
            NotImplementedError: If delete operations are not supported
        """
        raise NotImplementedError(
            f"Delete operations are not implemented for {self.secret_manager_name}. "
            "Override async_delete_secret() to add delete support."
        )

    def validate_environment(self) -> bool:
        """
        Validate that all required environment variables and configuration are present.

        Override this method to validate your secret manager's configuration.

        Returns:
            True if the environment is valid

        Raises:
            ValueError: If required configuration is missing
        """
        verbose_logger.debug(
            "No environment validation configured for custom secret manager"
        )
        return True

    async def async_health_check(
        self, timeout: Optional[Union[float, httpx.Timeout]] = None
    ) -> bool:
        """
        Perform a health check on your secret manager.

        This is optional to implement. Override this method to add health check support.

        Args:
            timeout: Request timeout

        Returns:
            True if the secret manager is healthy, False otherwise
        """
        verbose_logger.debug(
            f"Health check not implemented for {self.secret_manager_name}"
        )
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.secret_manager_name})>"
