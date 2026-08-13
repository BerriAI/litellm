"""GCS Cache implementation
Supports syncing responses to Google Cloud Storage Buckets using HTTP requests.
"""

import asyncio
import json
from typing import Final
from urllib.parse import quote

from litellm._logging import print_verbose, verbose_logger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base_cache import BaseCache


class GCSCache(BaseCache):
    def __init__(
        self,
        bucket_name: str | None = None,
        path_service_account: str | None = None,
        gcs_path: str | None = None,
    ) -> None:
        super().__init__()
        self.bucket_name = bucket_name or GCSBucketBase(bucket_name=None).BUCKET_NAME
        self.path_service_account = path_service_account or GCSBucketBase(bucket_name=None).path_service_account_json
        self.key_prefix = gcs_path.rstrip("/") + "/" if gcs_path else ""
        # create httpx clients
        self.async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        self.sync_client = _get_httpx_client()

    def _construct_headers(self) -> dict:
        base: Final = GCSBucketBase(bucket_name=self.bucket_name)
        base.path_service_account_json = self.path_service_account
        base.BUCKET_NAME = self.bucket_name
        return base.sync_construct_request_headers()

    def set_cache(self, key, value, **kwargs):
        try:
            print_verbose(f"LiteLLM SET Cache - GCS. Key={key}. Value={value}")
            headers: Final = self._construct_headers()
            object_name: Final = self.key_prefix + key
            bucket_name: Final = self.bucket_name
            url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={quote(object_name, safe='')}"
            data: Final = json.dumps(value)
            self.sync_client.post(url=url, data=data, headers=headers)
        except Exception as e:
            print_verbose(f"GCS Caching: set_cache() - Got exception from GCS: {e}")

    async def async_set_cache(self, key, value, **kwargs):
        try:
            headers: Final = self._construct_headers()
            object_name: Final = self.key_prefix + key
            bucket_name: Final = self.bucket_name
            url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={quote(object_name, safe='')}"
            data: Final = json.dumps(value)
            await self.async_client.post(url=url, data=data, headers=headers)
        except Exception as e:
            print_verbose(f"GCS Caching: async_set_cache() - Got exception from GCS: {e}")

    def get_cache(self, key, **kwargs):
        try:
            headers: Final = self._construct_headers()
            object_name: Final = self.key_prefix + key
            bucket_name: Final = self.bucket_name
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{quote(object_name, safe='')}?alt=media"
            response: Final = self.sync_client.get(url=url, headers=headers)
            if response.status_code == 200:
                cached_response: Final = json.loads(response.text)
                verbose_logger.debug(
                    "Got GCS Cache: key: %s, cached_response %s. Type Response %s",
                    key,
                    cached_response,
                    type(cached_response),
                )
                return cached_response
            return None
        except Exception as e:
            verbose_logger.error("GCS Caching: get_cache() - Got exception from GCS: %s", e)

    async def async_get_cache(self, key, **kwargs):
        try:
            headers: Final = self._construct_headers()
            object_name: Final = self.key_prefix + key
            bucket_name: Final = self.bucket_name
            url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{quote(object_name, safe='')}?alt=media"
            response: Final = await self.async_client.get(url=url, headers=headers)
            if response.status_code == 200:
                return json.loads(response.text)
            return None
        except Exception as e:
            verbose_logger.error("GCS Caching: async_get_cache() - Got exception from GCS: %s", e)

    def flush_cache(self):
        pass

    async def disconnect(self):
        pass

    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        tasks: Final = []
        for val in cache_list:
            tasks.append(self.async_set_cache(val[0], val[1], **kwargs))
        await asyncio.gather(*tasks)
