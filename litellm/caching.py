# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import ast
import hashlib
import json
import logging
import time
import traceback
from typing import Any, List, Literal, Optional, Union

from openai._models import BaseModel as OpenAIObject

import litellm


def print_verbose(print_statement):
    try:
        if litellm.set_verbose:
            print(print_statement)  # noqa
    except:
        pass


class BaseCache:
    def set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    def get_cache(self, key, **kwargs):
        raise NotImplementedError


class InMemoryCache(BaseCache):
    def __init__(self):
        # if users don't provider one, use the default litellm cache
        self.cache_dict = {}
        self.ttl_dict = {}

    def set_cache(self, key, value, **kwargs):
        self.cache_dict[key] = value
        if "ttl" in kwargs:
            self.ttl_dict[key] = time.time() + kwargs["ttl"]

    def get_cache(self, key, **kwargs):
        if key in self.cache_dict:
            if key in self.ttl_dict:
                if time.time() > self.ttl_dict[key]:
                    self.cache_dict.pop(key, None)
                    return None
            original_cached_response = self.cache_dict[key]
            try:
                cached_response = json.loads(original_cached_response)
            except:
                cached_response = original_cached_response
            return cached_response
        return None

    def flush_cache(self):
        self.cache_dict.clear()
        self.ttl_dict.clear()


class RedisCache(BaseCache):
    def __init__(self, host=None, port=None, password=None, **kwargs):
        import redis

        # if users don't provider one, use the default litellm cache
        from ._redis import get_redis_client

        redis_kwargs = {}
        if host is not None:
            redis_kwargs["host"] = host
        if port is not None:
            redis_kwargs["port"] = port
        if password is not None:
            redis_kwargs["password"] = password

        redis_kwargs.update(kwargs)

        self.redis_client = get_redis_client(**redis_kwargs)

    def set_cache(self, key, value, **kwargs):
        ttl = kwargs.get("ttl", None)
        print_verbose(f"Set Redis Cache: key: {key}\nValue {value}")
        try:
            self.redis_client.set(name=key, value=str(value), ex=ttl)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            logging.debug("LiteLLM Caching: set() - Got exception from REDIS : ", e)

    def get_cache(self, key, **kwargs):
        try:
            print_verbose(f"Get Redis Cache: key: {key}")
            cached_response = self.redis_client.get(key)
            print_verbose(
                f"Got Redis Cache: key: {key}, cached_response {cached_response}"
            )
            if cached_response != None:
                # cached_response is in `b{} convert it to ModelResponse
                cached_response = cached_response.decode(
                    "utf-8"
                )  # Convert bytes to string
                try:
                    cached_response = json.loads(
                        cached_response
                    )  # Convert string to dictionary
                except:
                    cached_response = ast.literal_eval(cached_response)
                return cached_response
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            traceback.print_exc()
            logging.debug("LiteLLM Caching: get() - Got exception from REDIS: ", e)

    def flush_cache(self):
        self.redis_client.flushall()


class S3Cache(BaseCache):
    def __init__(
        self,
        s3_bucket_name,
        s3_region_name=None,
        s3_api_version=None,
        s3_use_ssl=True,
        s3_verify=None,
        s3_endpoint_url=None,
        s3_aws_access_key_id=None,
        s3_aws_secret_access_key=None,
        s3_aws_session_token=None,
        s3_config=None,
        **kwargs,
    ):
        import boto3

        self.bucket_name = s3_bucket_name
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

    def set_cache(self, key, value, **kwargs):
        try:
            print_verbose(f"LiteLLM SET Cache - S3. Key={key}. Value={value}")
            ttl = kwargs.get("ttl", None)
            # Convert value to JSON before storing in S3
            serialized_value = json.dumps(value)
            if ttl is not None:
                cache_control = f"immutable, max-age={ttl}, s-maxage={ttl}"
                import datetime

                # Calculate expiration time
                expiration_time = datetime.datetime.now() + ttl

                # Upload the data to S3 with the calculated expiration time
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=serialized_value,
                    Expires=expiration_time,
                    CacheControl=cache_control,
                    ContentType="application/json",
                    ContentLanguage="en",
                    ContentDisposition=f"inline; filename=\"{key}.json\""
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
                    ContentDisposition=f"inline; filename=\"{key}.json\""
                )
        except Exception as e:
            # NON blocking - notify users S3 is throwing an exception
            print_verbose(f"S3 Caching: set_cache() - Got exception from S3: {e}")

    def get_cache(self, key, **kwargs):
        import boto3
        import botocore

        try:
            print_verbose(f"Get S3 Cache: key: {key}")
            # Download the data from S3
            cached_response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=key
            )

            if cached_response != None:
                # cached_response is in `b{} convert it to ModelResponse
                cached_response = (
                    cached_response["Body"].read().decode("utf-8")
                )  # Convert bytes to string
                try:
                    cached_response = json.loads(
                        cached_response
                    )  # Convert string to dictionary
                except Exception as e:
                    cached_response = ast.literal_eval(cached_response)
            if type(cached_response) is not dict:
                cached_response = dict(cached_response)
            print_verbose(
                f"Got S3 Cache: key: {key}, cached_response {cached_response}. Type Response {type(cached_response)}"
            )

            return cached_response
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print_verbose(
                    f"S3 Cache: The specified key '{key}' does not exist in the S3 bucket."
                )
                return None

        except Exception as e:
            # NON blocking - notify users S3 is throwing an exception
            traceback.print_exc()
            print_verbose(f"S3 Caching: get_cache() - Got exception from S3: {e}")

    def flush_cache(self):
        pass


class DualCache(BaseCache):
    """
    This updates both Redis and an in-memory cache simultaneously.
    When data is updated or inserted, it is written to both the in-memory cache + Redis.
    This ensures that even if Redis hasn't been updated yet, the in-memory cache reflects the most recent data.
    """

    def __init__(
        self,
        in_memory_cache: Optional[InMemoryCache] = None,
        redis_cache: Optional[RedisCache] = None,
    ) -> None:
        super().__init__()
        # If in_memory_cache is not provided, use the default InMemoryCache
        self.in_memory_cache = in_memory_cache or InMemoryCache()
        # If redis_cache is not provided, use the default RedisCache
        self.redis_cache = redis_cache

    def set_cache(self, key, value, local_only: bool = False, **kwargs):
        # Update both Redis and in-memory cache
        try:
            print_verbose(f"set cache: key: {key}; value: {value}")
            if self.in_memory_cache is not None:
                self.in_memory_cache.set_cache(key, value, **kwargs)

            if self.redis_cache is not None and local_only == False:
                self.redis_cache.set_cache(key, value, **kwargs)
        except Exception as e:
            print_verbose(e)

    def get_cache(self, key, local_only: bool = False, **kwargs):
        # Try to fetch from in-memory cache first
        try:
            print_verbose(f"get cache: cache key: {key}; local_only: {local_only}")
            result = None
            if self.in_memory_cache is not None:
                in_memory_result = self.in_memory_cache.get_cache(key, **kwargs)

                print_verbose(f"in_memory_result: {in_memory_result}")
                if in_memory_result is not None:
                    result = in_memory_result

            if result is None and self.redis_cache is not None and local_only == False:
                # If not found in in-memory cache, try fetching from Redis
                redis_result = self.redis_cache.get_cache(key, **kwargs)

                if redis_result is not None:
                    # Update in-memory cache with the value from Redis
                    self.in_memory_cache.set_cache(key, redis_result, **kwargs)

                result = redis_result

            print_verbose(f"get cache: cache result: {result}")
            return result
        except Exception as e:
            traceback.print_exc()

    def flush_cache(self):
        if self.in_memory_cache is not None:
            self.in_memory_cache.flush_cache()
        if self.redis_cache is not None:
            self.redis_cache.flush_cache()


#### LiteLLM.Completion / Embedding Cache ####
class Cache:
    def __init__(
        self,
        type: Optional[Literal["local", "redis", "s3"]] = "local",
        host: Optional[str] = None,
        port: Optional[str] = None,
        password: Optional[str] = None,
        supported_call_types: Optional[
            List[Literal["completion", "acompletion", "embedding", "aembedding"]]
        ] = ["completion", "acompletion", "embedding", "aembedding"],
        # s3 Bucket, boto3 configuration
        s3_bucket_name: Optional[str] = None,
        s3_region_name: Optional[str] = None,
        s3_api_version: Optional[str] = None,
        s3_use_ssl: Optional[bool] = True,
        s3_verify: Optional[Union[bool, str]] = None,
        s3_endpoint_url: Optional[str] = None,
        s3_aws_access_key_id: Optional[str] = None,
        s3_aws_secret_access_key: Optional[str] = None,
        s3_aws_session_token: Optional[str] = None,
        s3_config: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initializes the cache based on the given type.

        Args:
            type (str, optional): The type of cache to initialize. Can be "local" or "redis". Defaults to "local".
            host (str, optional): The host address for the Redis cache. Required if type is "redis".
            port (int, optional): The port number for the Redis cache. Required if type is "redis".
            password (str, optional): The password for the Redis cache. Required if type is "redis".
            supported_call_types (list, optional): List of call types to cache for. Defaults to cache == on for all call types.
            **kwargs: Additional keyword arguments for redis.Redis() cache

        Raises:
            ValueError: If an invalid cache type is provided.

        Returns:
            None. Cache is set as a litellm param
        """
        if type == "redis":
            self.cache: BaseCache = RedisCache(host, port, password, **kwargs)
        if type == "local":
            self.cache = InMemoryCache()
        if type == "s3":
            self.cache = S3Cache(
                s3_bucket_name=s3_bucket_name,
                s3_region_name=s3_region_name,
                s3_api_version=s3_api_version,
                s3_use_ssl=s3_use_ssl,
                s3_verify=s3_verify,
                s3_endpoint_url=s3_endpoint_url,
                s3_aws_access_key_id=s3_aws_access_key_id,
                s3_aws_secret_access_key=s3_aws_secret_access_key,
                s3_aws_session_token=s3_aws_session_token,
                s3_config=s3_config,
                **kwargs,
            )
        if "cache" not in litellm.input_callback:
            litellm.input_callback.append("cache")
        if "cache" not in litellm.success_callback:
            litellm.success_callback.append("cache")
        if "cache" not in litellm._async_success_callback:
            litellm._async_success_callback.append("cache")
        self.supported_call_types = supported_call_types  # default to ["completion", "acompletion", "embedding", "aembedding"]
        self.type = type

    def get_cache_key(self, *args, **kwargs):
        """
        Get the cache key for the given arguments.

        Args:
            *args: args to litellm.completion() or embedding()
            **kwargs: kwargs to litellm.completion() or embedding()

        Returns:
            str: The cache key generated from the arguments, or None if no cache key could be generated.
        """
        cache_key = ""
        print_verbose(f"\nGetting Cache key. Kwargs: {kwargs}")

        # for streaming, we use preset_cache_key. It's created in wrapper(), we do this because optional params like max_tokens, get transformed for bedrock -> max_new_tokens
        if kwargs.get("litellm_params", {}).get("preset_cache_key", None) is not None:
            print_verbose(f"\nReturning preset cache key: {cache_key}")
            return kwargs.get("litellm_params", {}).get("preset_cache_key", None)

        # sort kwargs by keys, since model: [gpt-4, temperature: 0.2, max_tokens: 200] == [temperature: 0.2, max_tokens: 200, model: gpt-4]
        completion_kwargs = [
            "model",
            "messages",
            "temperature",
            "top_p",
            "n",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
        ]
        embedding_only_kwargs = [
            "input",
            "encoding_format",
        ]  # embedding kwargs = model, input, user, encoding_format. Model, user are checked in completion_kwargs

        # combined_kwargs - NEEDS to be ordered across get_cache_key(). Do not use a set()
        combined_kwargs = completion_kwargs + embedding_only_kwargs
        for param in combined_kwargs:
            # ignore litellm params here
            if param in kwargs:
                # check if param == model and model_group is passed in, then override model with model_group
                if param == "model":
                    model_group = None
                    caching_group = None
                    metadata = kwargs.get("metadata", None)
                    litellm_params = kwargs.get("litellm_params", {})
                    if metadata is not None:
                        model_group = metadata.get("model_group")
                        model_group = metadata.get("model_group", None)
                        caching_groups = metadata.get("caching_groups", None)
                        if caching_groups:
                            for group in caching_groups:
                                if model_group in group:
                                    caching_group = group
                                    break
                    if litellm_params is not None:
                        metadata = litellm_params.get("metadata", None)
                        if metadata is not None:
                            model_group = metadata.get("model_group", None)
                            caching_groups = metadata.get("caching_groups", None)
                            if caching_groups:
                                for group in caching_groups:
                                    if model_group in group:
                                        caching_group = group
                                        break
                    param_value = (
                        caching_group or model_group or kwargs[param]
                    )  # use caching_group, if set then model_group if it exists, else use kwargs["model"]
                else:
                    if kwargs[param] is None:
                        continue  # ignore None params
                    param_value = kwargs[param]
                cache_key += f"{str(param)}: {str(param_value)}"
        print_verbose(f"\nCreated cache key: {cache_key}")
        # Use hashlib to create a sha256 hash of the cache key
        hash_object = hashlib.sha256(cache_key.encode())
        # Hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()
        print_verbose(f"Hashed cache key (SHA-256): {hash_hex}")
        return hash_hex

    def generate_streaming_content(self, content):
        chunk_size = 5  # Adjust the chunk size as needed
        for i in range(0, len(content), chunk_size):
            yield {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": content[i : i + chunk_size],
                        }
                    }
                ]
            }
            time.sleep(0.02)

    def get_cache(self, *args, **kwargs):
        """
        Retrieves the cached result for the given arguments.

        Args:
            *args: args to litellm.completion() or embedding()
            **kwargs: kwargs to litellm.completion() or embedding()

        Returns:
            The cached result if it exists, otherwise None.
        """
        try:  # never block execution
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = self.get_cache_key(*args, **kwargs)
            if cache_key is not None:
                cache_control_args = kwargs.get("cache", {})
                max_age = cache_control_args.get(
                    "s-max-age", cache_control_args.get("s-maxage", float("inf"))
                )
                cached_result = self.cache.get_cache(cache_key)
                # Check if a timestamp was stored with the cached response
                if (
                    cached_result is not None
                    and isinstance(cached_result, dict)
                    and "timestamp" in cached_result
                    and max_age is not None
                ):
                    timestamp = cached_result["timestamp"]
                    current_time = time.time()

                    # Calculate age of the cached response
                    response_age = current_time - timestamp

                    # Check if the cached response is older than the max-age
                    if response_age > max_age:
                        print_verbose(
                            f"Cached response for key {cache_key} is too old. Max-age: {max_age}s, Age: {response_age}s"
                        )
                        return None  # Cached response is too old

                    # If the response is fresh, or there's no max-age requirement, return the cached response
                    # cached_response is in `b{} convert it to ModelResponse
                    cached_response = cached_result.get("response")
                    try:
                        if isinstance(cached_response, dict):
                            pass
                        else:
                            cached_response = json.loads(
                                cached_response
                            )  # Convert string to dictionary
                    except:
                        cached_response = ast.literal_eval(cached_response)
                    return cached_response
                return cached_result
        except Exception as e:
            print_verbose(f"An exception occurred: {traceback.format_exc()}")
            return None

    def add_cache(self, result, *args, **kwargs):
        """
        Adds a result to the cache.

        Args:
            *args: args to litellm.completion() or embedding()
            **kwargs: kwargs to litellm.completion() or embedding()

        Returns:
            None
        """
        try:
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = self.get_cache_key(*args, **kwargs)
            if cache_key is not None:
                if isinstance(result, OpenAIObject):
                    result = result.model_dump_json()

                ## Get Cache-Controls ##
                if kwargs.get("cache", None) is not None and isinstance(
                    kwargs.get("cache"), dict
                ):
                    for k, v in kwargs.get("cache").items():
                        if k == "ttl":
                            kwargs["ttl"] = v
                cached_data = {"timestamp": time.time(), "response": result}
                self.cache.set_cache(cache_key, cached_data, **kwargs)
        except Exception as e:
            print_verbose(f"LiteLLM Cache: Excepton add_cache: {str(e)}")
            traceback.print_exc()
            pass

    async def _async_add_cache(self, result, *args, **kwargs):
        self.add_cache(result, *args, **kwargs)


def enable_cache(
    type: Optional[Literal["local", "redis", "s3"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[Literal["completion", "acompletion", "embedding", "aembedding"]]
    ] = ["completion", "acompletion", "embedding", "aembedding"],
    **kwargs,
):
    """
    Enable cache with the specified configuration.

    Args:
        type (Optional[Literal["local", "redis"]]): The type of cache to enable. Defaults to "local".
        host (Optional[str]): The host address of the cache server. Defaults to None.
        port (Optional[str]): The port number of the cache server. Defaults to None.
        password (Optional[str]): The password for the cache server. Defaults to None.
        supported_call_types (Optional[List[Literal["completion", "acompletion", "embedding", "aembedding"]]]):
            The supported call types for the cache. Defaults to ["completion", "acompletion", "embedding", "aembedding"].
        **kwargs: Additional keyword arguments.

    Returns:
        None

    Raises:
        None
    """
    print_verbose("LiteLLM: Enabling Cache")
    if "cache" not in litellm.input_callback:
        litellm.input_callback.append("cache")
    if "cache" not in litellm.success_callback:
        litellm.success_callback.append("cache")
    if "cache" not in litellm._async_success_callback:
        litellm._async_success_callback.append("cache")

    if litellm.cache == None:
        litellm.cache = Cache(
            type=type,
            host=host,
            port=port,
            password=password,
            supported_call_types=supported_call_types,
            **kwargs,
        )
    print_verbose(f"LiteLLM: Cache enabled, litellm.cache={litellm.cache}")
    print_verbose(f"LiteLLM Cache: {vars(litellm.cache)}")


def update_cache(
    type: Optional[Literal["local", "redis"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[Literal["completion", "acompletion", "embedding", "aembedding"]]
    ] = ["completion", "acompletion", "embedding", "aembedding"],
    **kwargs,
):
    """
    Update the cache for LiteLLM.

    Args:
        type (Optional[Literal["local", "redis"]]): The type of cache. Defaults to "local".
        host (Optional[str]): The host of the cache. Defaults to None.
        port (Optional[str]): The port of the cache. Defaults to None.
        password (Optional[str]): The password for the cache. Defaults to None.
        supported_call_types (Optional[List[Literal["completion", "acompletion", "embedding", "aembedding"]]]):
            The supported call types for the cache. Defaults to ["completion", "acompletion", "embedding", "aembedding"].
        **kwargs: Additional keyword arguments for the cache.

    Returns:
        None

    """
    print_verbose("LiteLLM: Updating Cache")
    litellm.cache = Cache(
        type=type,
        host=host,
        port=port,
        password=password,
        supported_call_types=supported_call_types,
        **kwargs,
    )
    print_verbose(f"LiteLLM: Cache Updated, litellm.cache={litellm.cache}")
    print_verbose(f"LiteLLM Cache: {vars(litellm.cache)}")


def disable_cache():
    """
    Disable the cache used by LiteLLM.

    This function disables the cache used by the LiteLLM module. It removes the cache-related callbacks from the input_callback, success_callback, and _async_success_callback lists. It also sets the litellm.cache attribute to None.

    Parameters:
    None

    Returns:
    None
    """
    from contextlib import suppress

    print_verbose("LiteLLM: Disabling Cache")
    with suppress(ValueError):
        litellm.input_callback.remove("cache")
        litellm.success_callback.remove("cache")
        litellm._async_success_callback.remove("cache")

    litellm.cache = None
    print_verbose(f"LiteLLM: Cache disabled, litellm.cache={litellm.cache}")
