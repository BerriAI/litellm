"""
Azure Blob Cache implementation

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

import asyncio
import json
from contextlib import suppress
from typing import Final

from litellm._logging import print_verbose, verbose_logger

from .base_cache import BaseCache


class AzureBlobCache(BaseCache):
    def __init__(self, account_url, container) -> None:
        from azure.core.exceptions import ResourceExistsError
        from azure.identity import DefaultAzureCredential
        from azure.identity.aio import (
            DefaultAzureCredential as AsyncDefaultAzureCredential,
        )
        from azure.storage.blob import BlobServiceClient
        from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient

        self.container_client = BlobServiceClient(
            account_url=account_url,
            credential=DefaultAzureCredential(),
        ).get_container_client(container)
        self.async_container_client = AsyncBlobServiceClient(
            account_url=account_url,
            credential=AsyncDefaultAzureCredential(),
        ).get_container_client(container)

        with suppress(ResourceExistsError):
            self.container_client.create_container()

    def set_cache(self, key, value, **kwargs) -> None:
        print_verbose(f"LiteLLM SET Cache - Azure Blob. Key={key}. Value={value}")
        serialized_value: Final = json.dumps(value)
        try:
            self.container_client.upload_blob(key, serialized_value)
        except Exception as e:
            # NON blocking - notify users Azure Blob is throwing an exception
            print_verbose(f"LiteLLM set_cache() - Got exception from Azure Blob: {e}")

    async def async_set_cache(self, key, value, **kwargs) -> None:
        print_verbose(f"LiteLLM SET Cache - Azure Blob. Key={key}. Value={value}")
        serialized_value: Final = json.dumps(value)
        try:
            await self.async_container_client.upload_blob(key, serialized_value, overwrite=True)
        except Exception as e:
            # NON blocking - notify users Azure Blob is throwing an exception
            print_verbose(f"LiteLLM set_cache() - Got exception from Azure Blob: {e}")

    def get_cache(self, key, **kwargs):
        from azure.core.exceptions import ResourceNotFoundError

        try:
            print_verbose(f"Get Azure Blob Cache: key: {key}")
            as_bytes: Final = self.container_client.download_blob(key).readall()
            as_str: Final = as_bytes.decode("utf-8")
            cached_response: Final = json.loads(as_str)

            verbose_logger.debug(
                "Got Azure Blob Cache: key: %s, cached_response %s. Type Response %s",
                key,
                cached_response,
                type(cached_response),
            )

            return cached_response
        except ResourceNotFoundError:
            return None

    async def async_get_cache(self, key, **kwargs):
        from azure.core.exceptions import ResourceNotFoundError

        try:
            print_verbose(f"Get Azure Blob Cache: key: {key}")
            blob: Final = await self.async_container_client.download_blob(key)
            as_bytes: Final = await blob.readall()
            as_str: Final = as_bytes.decode("utf-8")
            cached_response: Final = json.loads(as_str)
            verbose_logger.debug(
                "Got Azure Blob Cache: key: %s, cached_response %s. Type Response %s",
                key,
                cached_response,
                type(cached_response),
            )
            return cached_response
        except ResourceNotFoundError:
            return None

    def flush_cache(self) -> None:
        for blob in self.container_client.walk_blobs():
            self.container_client.delete_blob(blob.name)

    async def disconnect(self) -> None:
        self.container_client.close()
        await self.async_container_client.close()

    async def async_set_cache_pipeline(self, cache_list, **kwargs) -> None:
        tasks: Final = []
        for val in cache_list:
            tasks.append(self.async_set_cache(val[0], val[1], **kwargs))
        await asyncio.gather(*tasks)
