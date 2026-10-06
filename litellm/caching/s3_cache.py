"""
S3 Cache implementation

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache (uses run_in_executor)
    - async_get_cache (uses run_in_executor)
"""

import ast
import asyncio
import json
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Final

from litellm._logging import print_verbose, verbose_logger

from .base_cache import BaseCache


class S3Cache(BaseCache):
    def __init__(
        self,
        s3_bucket_name,
        s3_region_name=None,
        s3_api_version=None,
        s3_use_ssl: bool | None = True,
        s3_verify=None,
        s3_endpoint_url=None,
        s3_aws_access_key_id=None,
        s3_aws_secret_access_key=None,
        s3_aws_session_token=None,
        s3_config=None,
        s3_path=None,
        **kwargs,
    ):
        import boto3

        self.bucket_name = s3_bucket_name
        self.key_prefix = s3_path.rstrip("/") + "/" if s3_path else ""
        # Create an S3 client with custom endpoint URL

        self.s3_client = boto3.client(
            "s3",
            region_name=s3_region_name,
            endpoint_url=s3_endpoint_url,
            api_version=s3_api_version,
            use_ssl=s3_use_ssl,
            verify=s3_verify,
            aws_access_key_id=s3_aws_access_key_id,
            aws_secret_access_key=s3_aws_secret_access_key,
            aws_session_token=s3_aws_session_token,
            config=s3_config,
            **kwargs,
        )

    def _to_s3_key(self, key: str) -> str:
        """Convert cache key to S3 key"""
        return self.key_prefix + key.replace(":", "/")

    def set_cache(self, key, value, **kwargs):
        try:
            print_verbose(f"LiteLLM SET Cache - S3. Key={key}. Value={value}")
            ttl: Final = kwargs.get("ttl", None)
            # Convert value to JSON before storing in S3
            serialized_value: Final = json.dumps(value)
            key = self._to_s3_key(key)

            if ttl is not None:
                cache_control = f"immutable, max-age={ttl}, s-maxage={ttl}"

                # Calculate expiration time
                expiration_time: Final = datetime.now(timezone.utc) + timedelta(seconds=ttl)
                # Upload the data to S3 with the calculated expiration time
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=serialized_value,
                    Expires=expiration_time,
                    CacheControl=cache_control,
                    ContentType="application/json",
                    ContentLanguage="en",
                    ContentDisposition=f'inline; filename="{key}.json"',
                )
            else:
                cache_control = "immutable, max-age=31536000, s-maxage=31536000"
                # Upload the data to S3 without specifying Expires
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=serialized_value,
                    CacheControl=cache_control,
                    ContentType="application/json",
                    ContentLanguage="en",
                    ContentDisposition=f'inline; filename="{key}.json"',
                )
        except Exception as e:
            print_verbose(f"S3 Caching: set_cache() - Got exception from S3: {e}")

    async def async_set_cache(self, key, value, **kwargs):
        """
        Asynchronously set cache using run_in_executor to avoid blocking the event loop.
        Compatible with Python 3.8+.
        """
        try:
            verbose_logger.debug("Set ASYNC S3 Cache: Key=%s. Value=%s", key, value)
            loop: Final = asyncio.get_event_loop()
            func: Final = partial(self.set_cache, key, value, **kwargs)
            await loop.run_in_executor(None, func)
        except Exception as e:
            verbose_logger.error("S3 Caching: async_set_cache() - Got exception from S3: %s", e)

    def get_cache(self, key, **kwargs):
        import botocore

        try:
            key = self._to_s3_key(key)

            print_verbose(f"Get S3 Cache: key: {key}")
            # Download the data from S3
            cached_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)

            if cached_response is not None:
                if "Expires" in cached_response:
                    expires_time: Final = cached_response["Expires"]
                    current_time: Final = datetime.now(expires_time.tzinfo)

                    if current_time > expires_time:
                        return None

                # cached_response is in `b{} convert it to ModelResponse
                cached_response = cached_response["Body"].read().decode("utf-8")  # Convert bytes to string
                try:
                    cached_response = json.loads(cached_response)  # Convert string to dictionary
                except Exception:
                    cached_response = ast.literal_eval(cached_response)
            if not isinstance(cached_response, dict):
                cached_response = dict(cached_response)
            verbose_logger.debug(
                "Got S3 Cache: key: %s, cached_response %s. Type Response %s",
                key,
                cached_response,
                type(cached_response),
            )

            return cached_response
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                verbose_logger.debug("S3 Cache: The specified key '%s' does not exist in the S3 bucket.", key)
                return None

        except Exception as e:
            verbose_logger.error("S3 Caching: get_cache() - Got exception from S3: %s", e)

    async def async_get_cache(self, key, **kwargs):
        """
        Asynchronously get cache using run_in_executor to avoid blocking the event loop.
        Compatible with Python 3.8+.
        """
        try:
            verbose_logger.debug("Get ASYNC S3 Cache: key: %s", key)
            loop: Final = asyncio.get_event_loop()
            func: Final = partial(self.get_cache, key, **kwargs)
            result: Final = await loop.run_in_executor(None, func)
            return result
        except Exception as e:
            verbose_logger.error("S3 Caching: async_get_cache() - Got exception from S3: %s", e)
            return None

    def flush_cache(self):
        pass

    async def disconnect(self):
        pass

    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        tasks: Final = []
        for val in cache_list:
            tasks.append(self.async_set_cache(val[0], val[1], **kwargs))
        await asyncio.gather(*tasks)
