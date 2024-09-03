import base64
import os
from typing import Optional

import litellm
from litellm._logging import verbose_logger
from litellm.caching import InMemoryCache
from litellm.integrations.gcs_bucket_base import GCSBucketBase
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.proxy._types import KeyManagementSystem


class GoogleSecretManager(GCSBucketBase):
    def __init__(
        self,
        refresh_interval: Optional[int] = 86400,
        always_read_secret_manager: Optional[bool] = False,
    ) -> None:
        """
        Args:
            refresh_interval (int, optional): The refresh interval in seconds. Defaults to 86400. (24 hours)
            always_read_secret_manager (bool, optional): Whether to always read from the secret manager. Defaults to False. Since we do want to cache values
        """
        super().__init__()
        self.PROJECT_ID = "adroit-crow-413218"
        self.sync_httpx_client = _get_httpx_client()
        litellm.secret_manager_client = self
        litellm._key_management_system = KeyManagementSystem.GOOGLE_SECRET_MANAGER
        self.cache = InMemoryCache(
            default_ttl=refresh_interval
        )  # store in memory for 1 day
        self.always_read_secret_manager = False

    def get_secret_from_google_secret_manager(self, secret_name: str) -> Optional[str]:
        """
        Retrieve a secret from Google Secret Manager or cache.

        Args:
            secret_name (str): The name of the secret.

        Returns:
            str: The secret value if successful, None otherwise.
        """
        if self.always_read_secret_manager is not True:
            cached_secret = self.cache.get_cache(secret_name)
            if cached_secret is not None:
                return cached_secret
            if secret_name in self.cache.cache_dict:
                return cached_secret

        _secret_name = (
            f"projects/{self.PROJECT_ID}/secrets/{secret_name}/versions/latest"
        )
        headers = self.sync_construct_request_headers()
        url = f"https://secretmanager.googleapis.com/v1/{_secret_name}:access"

        # Send the GET request to retrieve the secret
        response = self.sync_httpx_client.get(url=url, headers=headers)

        if response.status_code != 200:
            verbose_logger.error(
                "Google Secret Manager retrieval error: %s", str(response.text)
            )
            self.cache.set_cache(
                secret_name, None
            )  # Cache that the secret was not found
            raise ValueError(
                f"secret {secret_name} not found in Google Secret Manager. Error: {response.text}"
            )

        verbose_logger.debug(
            "Google Secret Manager retrieval response status code: %s",
            response.status_code,
        )

        # Parse the JSON response and return the secret value
        secret_data = response.json()
        _base64_encoded_value = secret_data.get("payload", {}).get("data")

        # decode the base64 encoded value
        if _base64_encoded_value is not None:
            _decoded_value = base64.b64decode(_base64_encoded_value).decode("utf-8")
            self.cache.set_cache(
                secret_name, _decoded_value
            )  # Cache the retrieved secret
            return _decoded_value

        self.cache.set_cache(secret_name, None)  # Cache that the secret was not found
        raise ValueError(f"secret {secret_name} not found in Google Secret Manager")
