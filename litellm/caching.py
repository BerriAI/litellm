# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import litellm
import time, logging
import json, traceback, ast
from typing import Optional, Literal

def print_verbose(print_statement):
    try:
        if litellm.set_verbose:
            print(print_statement) # noqa
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
            print_verbose(f"Got Redis Cache: key: {key}, cached_response {cached_response}")
            if cached_response != None:
                # cached_response is in `b{} convert it to ModelResponse
                cached_response = cached_response.decode("utf-8")  # Convert bytes to string
                try: 
                    cached_response = json.loads(cached_response)  # Convert string to dictionary
                except: 
                    cached_response = ast.literal_eval(cached_response)
                return cached_response
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            traceback.print_exc()
            logging.debug("LiteLLM Caching: get() - Got exception from REDIS: ", e)

    def flush_cache(self):
        self.redis_client.flushall()

class DualCache(BaseCache): 
    """
    This updates both Redis and an in-memory cache simultaneously. 
    When data is updated or inserted, it is written to both the in-memory cache + Redis. 
    This ensures that even if Redis hasn't been updated yet, the in-memory cache reflects the most recent data.
    """
    def __init__(self, in_memory_cache: Optional[InMemoryCache] =None, redis_cache: Optional[RedisCache] =None) -> None:
        super().__init__()
        # If in_memory_cache is not provided, use the default InMemoryCache
        self.in_memory_cache = in_memory_cache or InMemoryCache()
        # If redis_cache is not provided, use the default RedisCache
        self.redis_cache = redis_cache
    
    def set_cache(self, key, value, **kwargs):
        # Update both Redis and in-memory cache
        try: 
            print_verbose(f"set cache: key: {key}; value: {value}")
            if self.in_memory_cache is not None:
                self.in_memory_cache.set_cache(key, value, **kwargs)

            if self.redis_cache is not None:
                self.redis_cache.set_cache(key, value, **kwargs)
        except Exception as e: 
            print_verbose(e)

    def get_cache(self, key, **kwargs):
        # Try to fetch from in-memory cache first
        try: 
            print_verbose(f"get cache: cache key: {key}")
            result = None
            if self.in_memory_cache is not None:
                in_memory_result = self.in_memory_cache.get_cache(key, **kwargs)

                if in_memory_result is not None:
                    result = in_memory_result

            if self.redis_cache is not None: 
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
            type: Optional[Literal["local", "redis"]] = "local",
            host: Optional[str] = None,
            port: Optional[str] = None,
            password: Optional[str] = None,
            supported_call_types: Optional[list[Literal["completion", "acompletion", "embedding", "aembedding"]]] = ["completion", "acompletion", "embedding", "aembedding"],
            **kwargs
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
        if "cache" not in litellm.input_callback:
            litellm.input_callback.append("cache")
        if "cache" not in litellm.success_callback:
            litellm.success_callback.append("cache")
        if "cache" not in litellm._async_success_callback:
            litellm._async_success_callback.append("cache")
        self.supported_call_types = supported_call_types # default to ["completion", "acompletion", "embedding", "aembedding"]

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
        completion_kwargs = ["model", "messages", "temperature", "top_p", "n", "stop", "max_tokens", "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format", "seed", "tools", "tool_choice"]
        embedding_only_kwargs = ["input", "encoding_format"] # embedding kwargs = model, input, user, encoding_format. Model, user are checked in completion_kwargs
        
        # combined_kwargs - NEEDS to be ordered across get_cache_key(). Do not use a set()
        combined_kwargs = completion_kwargs + embedding_only_kwargs 
        for param in combined_kwargs:
            # ignore litellm params here
            if param in kwargs:
                # check if param == model and model_group is passed in, then override model with model_group
                if param == "model":
                    model_group = None
                    metadata = kwargs.get("metadata", None)
                    litellm_params = kwargs.get("litellm_params", {})
                    if metadata is not None:
                        model_group = metadata.get("model_group")
                    if litellm_params is not None:
                        metadata = litellm_params.get("metadata", None)
                        if metadata is not None:
                            model_group = metadata.get("model_group", None)
                    param_value = model_group or kwargs[param] # use model_group if it exists, else use kwargs["model"]
                else:
                    if kwargs[param] is None:
                        continue # ignore None params
                    param_value = kwargs[param]
                cache_key+= f"{str(param)}: {str(param_value)}"
        print_verbose(f"\nCreated cache key: {cache_key}")
        return cache_key

    def generate_streaming_content(self, content):
        chunk_size = 5  # Adjust the chunk size as needed
        for i in range(0, len(content), chunk_size):
            yield {'choices': [{'delta': {'role': 'assistant', 'content': content[i:i + chunk_size]}}]}
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
                cached_result = self.cache.get_cache(cache_key)
                return cached_result
        except Exception as e:
            logging.debug(f"An exception occurred: {traceback.format_exc()}")
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
                if isinstance(result, litellm.ModelResponse):
                    result = result.model_dump_json()
                self.cache.set_cache(cache_key, result, **kwargs)
        except Exception as e:
            print_verbose(f"LiteLLM Cache: Excepton add_cache: {str(e)}")
            traceback.print_exc()
            pass

    async def _async_add_cache(self, result, *args, **kwargs):
        self.add_cache(result, *args, **kwargs)