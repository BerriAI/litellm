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
import json, traceback

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


class BaseCache:
    def set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    def get_cache(self, key, **kwargs):
        raise NotImplementedError


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
                cached_response = json.loads(cached_response)  # Convert string to dictionary
                cached_response['cache'] = True  # set cache-hit flag to True
                return cached_response
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            traceback.print_exc()
            logging.debug("LiteLLM Caching: get() - Got exception from REDIS: ", e)


class HostedCache(BaseCache):
    def set_cache(self, key, value, **kwargs):
        if "ttl" in kwargs:
            logging.debug("LiteLLM Caching: TTL is not supported for hosted cache!")
        # make a post request to api.litellm.ai/set_cache
        import requests
        url = f"https://api.litellm.ai/set_cache?key={key}&value={str(value)}"
        requests.request("POST", url)  # post request to set this in the hosted litellm cache

    def get_cache(self, key, **kwargs):
        import requests
        url = f"https://api.litellm.ai/get_cache?key={key}"
        cached_response = requests.request("GET", url)
        cached_response = cached_response.text
        if cached_response == "NONE":  # api.litellm.ai returns "NONE" if it's not a cache hit
            return None
        if cached_response != None:
            try:
                cached_response = json.loads(cached_response)  # Convert string to dictionary
                cached_response['cache'] = True  # set cache-hit flag to True
                return cached_response
            except:
                return cached_response


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
            cached_response['cache'] = True  # set cache-hit flag to True
            return cached_response
        return None


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
        if type == "hosted":
            self.cache = HostedCache()
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
        prompt = get_prompt(*args, **kwargs)
        if prompt is not None:
            cache_key = prompt
            if "model" in kwargs:
                cache_key += kwargs["model"]
        elif "input" in kwargs:
            cache_key = " ".join(kwargs["input"])
            if "model" in kwargs:
                cache_key += kwargs["model"]
        else:
            return None
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
        except:
            pass
