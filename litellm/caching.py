import redis
import litellm, openai

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

class RedisCache():
    import redis
    def __init__(self, host, port, password):
        # if users don't provider one, use the default litellm cache
        self.redis_client = redis.Redis(host=host, port=port, password=password)

    def set_cache(self, key, value):
        self.redis_client.set(key, str(value))

    def get_cache(self, key):
        # TODO convert this to a ModelResponse object 
        return self.redis_client.get(key)

class InMemoryCache():
    def __init__(self):
        # if users don't provider one, use the default litellm cache
        self.cache_dict = {}

    def set_cache(self, key, value):
        self.cache_dict[key] = value

    def get_cache(self, key):
        if key in self.cache_dict:
            return self.cache_dict[key]
        return None

class Cache():
    def __init__(self, type="local", host="", port="", password=""):
        if type == "redis":
            self.cache = RedisCache(type, host, port, password)
        if type == "local":
            self.cache = InMemoryCache()

    def check_cache(self, *args, **kwargs):
        try:  # never block execution
            prompt = get_prompt(*args, **kwargs)
            if prompt != None:  # check if messages / prompt exists
                if "model" in kwargs: # default to caching with `model + prompt` as key
                    cache_key = prompt + kwargs["model"]
                    return self.cache.get_cache(cache_key)
            else:
                return self.cache.get_cache(prompt)
        except:
            return None

    def add_cache(self, result, *args, **kwargs):
        try:
            prompt = get_prompt(*args, **kwargs)
            if "model" in kwargs: # default to caching with `model + prompt` as key
                cache_key = prompt + kwargs["model"]
                self.cache.set_cache(cache_key, result)
            else:
                self.cache.set_cache(prompt, result)
        except:
            pass











