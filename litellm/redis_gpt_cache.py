import time
import ast
import json
import inspect
from typing import List, Optional, Any, Dict, Callable, Tuple
import os
import hashlib
import cachetools
from gptcache import Cache, Config
from gptcache.manager import manager_factory
from gptcache.manager.eviction.memory_cache import MemoryCacheEviction, popitem_wrapper
from gptcache.adapter.api import init_similar_cache, get, put
from gptcache.embedding import LangChain, Onnx
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from langchain_community.cache import GPTCache
from langchain.embeddings.base import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from .caching import BaseCache, print_verbose
import uuid

class BudServeMemoryCacheEviction(MemoryCacheEviction):
    """This class `BudServeMemoryCacheEviction` is a subclass of `MemoryCacheEviction`
    that implements memory cache eviction policies such as LRU and TTL with customizable
    parameters."""

    def __init__(
        self,
        policy: str = "LRU",
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Callable[[List[Any]], None] = None,
        ttl: Optional[int] = None,
        **kwargs,
    ):
        try:
            super().__init__(policy, maxsize, clean_size, on_evict, **kwargs)
        except ValueError:
            self._policy = policy.upper()
            if self._policy == "TTL":
                if not ttl:
                    raise ValueError("TTL policy requires ttl parameter")
                self._cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl, **kwargs)
            else:
                raise ValueError(f"Unknown policy {policy}")
            self._cache.popitem = popitem_wrapper(
                self._cache.popitem, on_evict, clean_size
            )


def init_gptcache_redis(
    cache_obj: Cache,
    hashed_llm: str,
    cache_config: dict,
    embedding_model: str,
    similarity_threshold: float,
    **redis_params,
):
    """Initialise the GPT cache object."""
    print_verbose(
        f"Initialise the GPT cache object: init_gptcache_redis.{cache_config}"
    )
    endpoint_id = cache_config.get("endpoint_id", "1234")
    embedding_model: str = cache_config.get("embedding_model", embedding_model)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
        )
        embeddings = LangChain(embeddings=embeddings)
    except Exception as exc:
        print_verbose(f"gptcache redis semantic-cache embeddings error: {str(exc)}")
        print_verbose(f"gptcache redis semantic-cache using Onnx embeddings")
        embeddings = Onnx()
    eviction_policy = cache_config.get("eviction_policy", {})
    host = redis_params["host"]
    port = redis_params["port"]
    password = redis_params["password"]
    data_manager = manager_factory(
        "redis,redis",
        scalar_params={
            "redis_host": host,
            "redis_port": port,
            "password": password,
            "global_key_prefix": f"cache_{endpoint_id}_{hashed_llm}",
        },
        vector_params={
            "host": host,
            "port": port,
            "password": password,
            "dimension": embeddings.dimension,
            "top_k": 1,
            "collection_name": f"index_{endpoint_id}_{hashed_llm}",
            "namespace": f"namespace_{endpoint_id}_{hashed_llm}",
        },
        eviction_manager="no_op_eviction",
    )

    eviction_params = {
        "maxsize": eviction_policy.get("max_size", 100),
        "policy": eviction_policy.get("policy", "LRU"),
        "clean_size": int(eviction_policy.get("max_size", 100) * 0.2) or 1,
        "ttl": eviction_policy.get("ttl"),
        "on_evict": data_manager._clear,
    }
    data_manager.eviction_base = BudServeMemoryCacheEviction(**eviction_params)

    ids = data_manager.s.get_ids(deleted=False)
    data_manager.eviction_base.put(ids)
    init_similar_cache(
        cache_obj=cache_obj,
        embedding=embeddings,
        data_manager=data_manager,
        evaluation=SbertCrossencoderEvaluation(),
        config=Config(
            similarity_threshold=cache_config.get(
                "score_threshold", similarity_threshold
            ),
            auto_flush=1,
        ),
    )


class RedisGPTCache(BaseCache, GPTCache):
    def __init__(
        self,
        host=None,
        port=None,
        password=None,
        redis_url=None,
        similarity_threshold=None,
        use_async=False,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        **kwargs,
    ):
        print_verbose(f"gpt cache redis initializing...{kwargs}")
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.eviction_policy=kwargs.pop("eviction_policy",None)
        self.cache_config=None
        if redis_url is None:
            # if no url passed, check if host, port and password are passed, if not raise an Exception
            if host is None or port is None or password is None:
                # try checking env for host, port and password
                import os

                host = os.getenv("REDIS_HOST")
                port = os.getenv("REDIS_PORT")
                password = os.getenv("REDIS_PASSWORD")
                if host is None or port is None or password is None:
                    raise Exception("Redis host, port, and password must be provided")

            redis_url = "redis://:" + password + "@" + host + ":" + port
        self.redis_params = {"host": host, "port": port, "password": password}
        GPTCache.__init__(self, init_gptcache_redis)
        print_verbose(f"gptcache redis semantic-cache redis_url: {redis_url}")
        if use_async == False:
            print_verbose("gptcache redis semantic-cache using sync redis client")

    def _create_cache_config(self, **kwargs):
        """ Generate cache config from request user_config """
        cache_config = None
        user_config = kwargs["proxy_server_request"]["body"].get("user_config")
        endpoint_cache_settings = user_config.get("endpoint_cache_settings")
        if user_config and endpoint_cache_settings:
            cache_params = endpoint_cache_settings.get("cache_params", {})
            cache_config = {
                "embedding_model": cache_params.get(
                    "redis_semantic_cache_embedding_model", self.embedding_model
                ),
                "eviction_policy": cache_params.get("eviction_policy", {}),
                "score_threshold": cache_params.get(
                    "similarity_threshold", self.similarity_threshold
                ),
                "metric_config": cache_params.get("metric_config", {}),
            }
        self.cache_config=cache_config
        return cache_config

    def _new_gptcache(self, llm_string: str, cache_config: dict) -> Any:
        """New gptcache object"""
        _gptcache = Cache()
        if self.init_gptcache_func is not None:
            sig = inspect.signature(self.init_gptcache_func)
            if len(sig.parameters) == 6:
                self.init_gptcache_func(
                    _gptcache,
                    llm_string,
                    cache_config,
                    self.embedding_model,
                    self.similarity_threshold,
                    **self.redis_params,
                )
            elif len(sig.parameters) == 3:
                self.init_gptcache_func(_gptcache, llm_string, cache_config)
            elif len(sig.parameters) == 2:
                self.init_gptcache_func(_gptcache, llm_string)  # type: ignore[call-arg]
            else:
                self.init_gptcache_func(_gptcache)  # type: ignore[call-arg]
        else:
            raise ValueError("init_gptcache_func is not defined.")

        self.gptcache_dict[llm_string] = _gptcache
        return _gptcache

    def _get_gptcache(self, llm_string: str, cache_config: dict) -> Any:
        """Get a cache object."""
        _gptcache = self.gptcache_dict.get(llm_string, None)
        if not _gptcache:
            _gptcache = self._new_gptcache(llm_string, cache_config)
        return _gptcache

    def set_cache(self, key: str, value: Any, **kwargs):
        """Set cache for the given key."""
        cache_config = self.cache_config
        cache_config["metric_config"] = {"request_start_time": time.time()}
        llm_cache = self._get_gptcache(key, cache_config)
        # get the prompt
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)
        result = (value if isinstance(value, str) else json.dumps(value),)
        cache_metric = put(
            prompt,
            result,
            cache_obj=llm_cache,
            cache_metric_config=cache_config.get("metric_config", {}),
        )

        return

    def _get_cache_logic(self, cached_response: Any):
        """
        Common 'get_cache_logic' across sync + async redis client implementations
        """
        if cached_response is None:
            return cached_response

        # check if cached_response is bytes
        if isinstance(cached_response, bytes):
            cached_response = cached_response.decode("utf-8")

        try:
            cached_response = json.loads(
                cached_response
            )  # Convert string to dictionary
        except:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def get_cache(self, key: str, **kwargs):
        # send endpoint_cache_config in kwargs
        import time
        from gptcache.adapter.api import get
        print_verbose(f"kwargs GET, {kwargs}")
        cache_config = self._create_cache_config(**kwargs) or {}
        cache_config["metric_config"] = {"request_start_time": time.time()}

        llm_cache = self._get_gptcache(key, cache_config)
        # query
        # get the messages
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)
        print_verbose(f"getting inside GET, {prompt, llm_cache, cache_config}")
        results, cache_metric = get(
            prompt,
            cache_obj=llm_cache,
            cache_metric_config=cache_config.get("metric_config", {}),
        )
        print_verbose(f"getting outside GET, results {results}")
        results = [json.loads(results)] if results is not None else None

        if results == None:
            return None

        if isinstance(results, list):
            if len(results) == 0:
                return None

        cached_value = json.dumps(results[0]["response"])
        return self._get_cache_logic(cached_response=cached_value)

    async def async_get_cache(self, key: str, **kwargs):
        """Asynchronous cache retrieval."""
        # Directly call get_cache asynchronously
        return self.get_cache(key, **kwargs)

    async def async_set_cache(self, key: str, value: Any, **kwargs):
        """Asynchronous cache insertion."""
        # Directly call set_cache asynchronously
        return self.set_cache(key, value, **kwargs)

    async def batch_cache_write(self, result: List[Tuple[str, Any]], *args, **kwargs):
        """Batch write results to cache."""
        for key, value in result:
            self.async_set_cache(key, value, **kwargs)

    async def disconnect(self):
        """Perform any necessary cleanup on disconnect."""
        print_verbose("Disconnecting from cache system...")
