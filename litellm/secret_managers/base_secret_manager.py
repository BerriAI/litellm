from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import httpx


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
    ) -> Dict[str, Any]:
        """
        Asynchronously write a secret to the secret manager.

        Args:
            secret_name (str): Name/path of the secret to write
            secret_value (str): Value to store
            description (Optional[str]): Description of the secret. Some secret managers allow storing a description with the secret.
            optional_params (Optional[dict]): Additional parameters specific to the secret manager
            timeout (Optional[Union[float, httpx.Timeout]]): Request timeout
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
