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
from typing import Optional

def get_prompt(*args, **kwargs):
    # make this safe checks, it should not throw any exceptions
    if len(args) > 1:
        messages = args[1]
        prompt = " ".join(message["content"] for message in messages)
        return prompt
    if "messages" in kwargs:
        messages = kwargs["messages"]
        prompt = " ".join(message["content"] for message in messages)
        return prompt
    return None

def print_verbose(print_statement):
    if litellm.set_verbose:
        print(print_statement) # noqa

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
            if isinstance(cached_response, dict): 
                    cached_response['cache'] = True  # set cache-hit flag to True
            return cached_response
        return None
    
    def flush_cache(self):
        self.cache_dict.clear()
        self.ttl_dict.clear()


class RedisCache(BaseCache):
    def __init__(self, host, port, password):
        import redis
        # if users don't provider one, use the default litellm cache
        self.redis_client = redis.Redis(host=host, port=port, password=password)

    def set_cache(self, key, value, **kwargs):
        ttl = kwargs.get("ttl", None)
        try:
            self.redis_client.set(name=key, value=str(value), ex=ttl)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            logging.debug("LiteLLM Caching: set() - Got exception from REDIS : ", e)

    def get_cache(self, key, **kwargs):
        try:
            # TODO convert this to a ModelResponse object
            cached_response = self.redis_client.get(key)
            if cached_response != None:
                # cached_response is in `b{} convert it to ModelResponse
                cached_response = cached_response.decode("utf-8")  # Convert bytes to string
                try: 
                    cached_response = json.loads(cached_response)  # Convert string to dictionary
                except: 
                    cached_response = ast.literal_eval(cached_response)
                if isinstance(cached_response, dict): 
                    cached_response['cache'] = True  # set cache-hit flag to True
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

#### LiteLLM.Completion Cache ####
class Cache:
    def __init__(
            self,
            type="local",
            host=None,
            port=None,
            password=None
    ):
        """
        Initializes the cache based on the given type.

        Args:
            type (str, optional): The type of cache to initialize. Defaults to "local".
            host (str, optional): The host address for the Redis cache. Required if type is "redis".
            port (int, optional): The port number for the Redis cache. Required if type is "redis".
            password (str, optional): The password for the Redis cache. Required if type is "redis".

        Raises:
            ValueError: If an invalid cache type is provided.

        Returns:
            None
        """
        if type == "redis":
            self.cache = RedisCache(host, port, password)
        if type == "local":
            self.cache = InMemoryCache()
        if "cache" not in litellm.input_callback:
            litellm.input_callback.append("cache")
        if "cache" not in litellm.success_callback:
            litellm.success_callback.append("cache")

    def get_cache_key(self, *args, **kwargs):
        """
        Get the cache key for the given arguments.

        Args:
            *args: args to litellm.completion() or embedding()
            **kwargs: kwargs to litellm.completion() or embedding()

        Returns:
            str: The cache key generated from the arguments, or None if no cache key could be generated.
        """
        cache_key =""
        for param in kwargs:
            # ignore litellm params here
            if param in set(["model", "messages", "temperature", "top_p", "n", "stop", "max_tokens", "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format", "seed", "tools", "tool_choice"]):
                # check if param == model and model_group is passed in, then override model with model_group
                if param == "model" and kwargs.get("metadata", None) is not None and kwargs["metadata"].get("model_group", None) is not None:
                    param_value = kwargs["metadata"].get("model_group", None) # for litellm.Router use model_group for caching over `model`
                else:
                    param_value = kwargs[param]
                cache_key+= f"{str(param)}: {str(param_value)}"
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
                if cached_result != None and 'stream' in kwargs and kwargs['stream'] == True:
                    # if streaming is true and we got a cache hit, return a generator
                    return self.generate_streaming_content(cached_result["choices"][0]['message']['content'])
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
            pass
