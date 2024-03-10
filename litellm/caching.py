# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import litellm
import time, logging, asyncio
import json, traceback, ast, hashlib
from typing import Optional, Literal, List, Union, Any, BinaryIO
from openai._models import BaseModel as OpenAIObject
from litellm._logging import verbose_logger


def print_verbose(print_statement):
    try:
        verbose_logger.debug(print_statement)
        if litellm.set_verbose:
            print(print_statement)  # noqa
    except:
        pass


class BaseCache:
    def set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    async def async_set_cache(self, key, value, **kwargs):
        raise NotImplementedError

    def get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def async_get_cache(self, key, **kwargs):
        raise NotImplementedError

    async def disconnect(self):
        raise NotImplementedError


class InMemoryCache(BaseCache):
    def __init__(self):
        # if users don't provider one, use the default litellm cache
        self.cache_dict = {}
        self.ttl_dict = {}

    def set_cache(self, key, value, **kwargs):
        print_verbose("InMemoryCache: set_cache")
        self.cache_dict[key] = value
        if "ttl" in kwargs:
            self.ttl_dict[key] = time.time() + kwargs["ttl"]

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key=key, value=value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, ttl=None):
        for cache_key, cache_value in cache_list:
            if ttl is not None:
                self.set_cache(key=cache_key, value=cache_value, ttl=ttl)
            else:
                self.set_cache(key=cache_key, value=cache_value)

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

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key=key, **kwargs)

    def flush_cache(self):
        self.cache_dict.clear()
        self.ttl_dict.clear()

    async def disconnect(self):
        pass

    def delete_cache(self, key):
        self.cache_dict.pop(key, None)
        self.ttl_dict.pop(key, None)


class RedisCache(BaseCache):
    # if users don't provider one, use the default litellm cache

    def __init__(self, host=None, port=None, password=None, **kwargs):
        from ._redis import get_redis_client, get_redis_connection_pool

        redis_kwargs = {}
        if host is not None:
            redis_kwargs["host"] = host
        if port is not None:
            redis_kwargs["port"] = port
        if password is not None:
            redis_kwargs["password"] = password

        redis_kwargs.update(kwargs)
        self.redis_client = get_redis_client(**redis_kwargs)
        self.redis_kwargs = redis_kwargs
        self.async_redis_conn_pool = get_redis_connection_pool()

    def init_async_client(self):
        from ._redis import get_redis_async_client

        return get_redis_async_client(
            connection_pool=self.async_redis_conn_pool, **self.redis_kwargs
        )

    def set_cache(self, key, value, **kwargs):
        ttl = kwargs.get("ttl", None)
        print_verbose(f"Set Redis Cache: key: {key}\nValue {value}\nttl={ttl}")
        try:
            self.redis_client.set(name=key, value=str(value), ex=ttl)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            print_verbose(
                f"LiteLLM Caching: set() - Got exception from REDIS : {str(e)}"
            )

    async def async_set_cache(self, key, value, **kwargs):
        _redis_client = self.init_async_client()
        async with _redis_client as redis_client:
            ttl = kwargs.get("ttl", None)
            print_verbose(
                f"Set ASYNC Redis Cache: key: {key}\nValue {value}\nttl={ttl}"
            )
            try:
                await redis_client.set(
                    name=key, value=json.dumps(value), ex=ttl, get=True
                )
            except Exception as e:
                # NON blocking - notify users Redis is throwing an exception
                print_verbose("LiteLLM Caching: set() - Got exception from REDIS : ", e)

    async def async_set_cache_pipeline(self, cache_list, ttl=None):
        """
        Use Redis Pipelines for bulk write operations
        """
        _redis_client = self.init_async_client()
        try:
            async with _redis_client as redis_client:
                async with redis_client.pipeline(transaction=True) as pipe:
                    # Iterate through each key-value pair in the cache_list and set them in the pipeline.
                    for cache_key, cache_value in cache_list:
                        print_verbose(
                            f"Set ASYNC Redis Cache PIPELINE: key: {cache_key}\nValue {cache_value}\nttl={ttl}"
                        )
                        # Set the value with a TTL if it's provided.
                        if ttl is not None:
                            pipe.setex(cache_key, ttl, json.dumps(cache_value))
                        else:
                            pipe.set(cache_key, json.dumps(cache_value))
                    # Execute the pipeline and return the results.
                    results = await pipe.execute()

            print_verbose(f"pipeline results: {results}")
            # Optionally, you could process 'results' to make sure that all set operations were successful.
            return results
        except Exception as e:
            print_verbose(f"Error occurred in pipeline write - {str(e)}")
            # NON blocking - notify users Redis is throwing an exception
            logging.debug("LiteLLM Caching: set() - Got exception from REDIS : ", e)

    def _get_cache_logic(self, cached_response: Any):
        """
        Common 'get_cache_logic' across sync + async redis client implementations
        """
        if cached_response is None:
            return cached_response
        # cached_response is in `b{} convert it to ModelResponse
        cached_response = cached_response.decode("utf-8")  # Convert bytes to string
        try:
            cached_response = json.loads(
                cached_response
            )  # Convert string to dictionary
        except:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def get_cache(self, key, **kwargs):
        try:
            print_verbose(f"Get Redis Cache: key: {key}")
            cached_response = self.redis_client.get(key)
            print_verbose(
                f"Got Redis Cache: key: {key}, cached_response {cached_response}"
            )
            return self._get_cache_logic(cached_response=cached_response)
        except Exception as e:
            # NON blocking - notify users Redis is throwing an exception
            traceback.print_exc()
            logging.debug("LiteLLM Caching: get() - Got exception from REDIS: ", e)

    async def async_get_cache(self, key, **kwargs):
        _redis_client = self.init_async_client()
        async with _redis_client as redis_client:
            try:
                print_verbose(f"Get Redis Cache: key: {key}")
                cached_response = await redis_client.get(key)
                print_verbose(
                    f"Got Async Redis Cache: key: {key}, cached_response {cached_response}"
                )
                response = self._get_cache_logic(cached_response=cached_response)
                return response
            except Exception as e:
                # NON blocking - notify users Redis is throwing an exception
                traceback.print_exc()
                logging.debug("LiteLLM Caching: get() - Got exception from REDIS: ", e)

    def flush_cache(self):
        self.redis_client.flushall()

    async def disconnect(self):
        pass

    def delete_cache(self, key):
        self.redis_client.delete(key)


class RedisSemanticCache(BaseCache):
    def __init__(
        self,
        host=None,
        port=None,
        password=None,
        redis_url=None,
        similarity_threshold=None,
        use_async=False,
        embedding_model="text-embedding-ada-002",
        **kwargs,
    ):
        from redisvl.index import SearchIndex
        from redisvl.query import VectorQuery

        print_verbose(
            "redis semantic-cache initializing INDEX - litellm_semantic_cache_index"
        )
        if similarity_threshold is None:
            raise Exception("similarity_threshold must be provided, passed None")
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        schema = {
            "index": {
                "name": "litellm_semantic_cache_index",
                "prefix": "litellm",
                "storage_type": "hash",
            },
            "fields": {
                "text": [{"name": "response"}],
                "text": [{"name": "prompt"}],
                "vector": [
                    {
                        "name": "litellm_embedding",
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    }
                ],
            },
        }
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
        print_verbose(f"redis semantic-cache redis_url: {redis_url}")
        if use_async == False:
            self.index = SearchIndex.from_dict(schema)
            self.index.connect(redis_url=redis_url)
            try:
                self.index.create(overwrite=False)  # don't overwrite existing index
            except Exception as e:
                print_verbose(f"Got exception creating semantic cache index: {str(e)}")
        elif use_async == True:
            schema["index"]["name"] = "litellm_semantic_cache_index_async"
            self.index = SearchIndex.from_dict(schema)
            self.index.connect(redis_url=redis_url, use_async=True)

    #
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

    def set_cache(self, key, value, **kwargs):
        import numpy as np

        print_verbose(f"redis semantic-cache set_cache, kwargs: {kwargs}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]

        # create an embedding for prompt
        embedding_response = litellm.embedding(
            model=self.embedding_model,
            input=prompt,
            cache={"no-store": True, "no-cache": True},
        )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        # make the embedding a numpy array, convert to bytes
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        value = str(value)
        assert isinstance(value, str)

        new_data = [
            {"response": value, "prompt": prompt, "litellm_embedding": embedding_bytes}
        ]

        # Add more data
        keys = self.index.load(new_data)

        return

    def get_cache(self, key, **kwargs):
        print_verbose(f"sync redis semantic-cache get_cache, kwargs: {kwargs}")
        from redisvl.query import VectorQuery
        import numpy as np

        # query

        # get the messages
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]

        # convert to embedding
        embedding_response = litellm.embedding(
            model=self.embedding_model,
            input=prompt,
            cache={"no-store": True, "no-cache": True},
        )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        query = VectorQuery(
            vector=embedding,
            vector_field_name="litellm_embedding",
            return_fields=["response", "prompt", "vector_distance"],
            num_results=1,
        )

        results = self.index.query(query)
        if results == None:
            return None
        if isinstance(results, list):
            if len(results) == 0:
                return None

        vector_distance = results[0]["vector_distance"]
        vector_distance = float(vector_distance)
        similarity = 1 - vector_distance
        cached_prompt = results[0]["prompt"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )
        if similarity > self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None

        pass

    async def async_set_cache(self, key, value, **kwargs):
        import numpy as np
        from litellm.proxy.proxy_server import llm_router, llm_model_list

        try:
            await self.index.acreate(overwrite=False)  # don't overwrite existing index
        except Exception as e:
            print_verbose(f"Got exception creating semantic cache index: {str(e)}")
        print_verbose(f"async redis semantic-cache set_cache, kwargs: {kwargs}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]
        # create an embedding for prompt
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        if llm_router is not None and self.embedding_model in router_model_names:
            user_api_key = kwargs.get("metadata", {}).get("user_api_key", "")
            embedding_response = await llm_router.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
                metadata={
                    "user_api_key": user_api_key,
                    "semantic-cache-embedding": True,
                    "trace_id": kwargs.get("metadata", {}).get("trace_id", None),
                },
            )
        else:
            # convert to embedding
            embedding_response = await litellm.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
            )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        # make the embedding a numpy array, convert to bytes
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        value = str(value)
        assert isinstance(value, str)

        new_data = [
            {"response": value, "prompt": prompt, "litellm_embedding": embedding_bytes}
        ]

        # Add more data
        keys = await self.index.aload(new_data)
        return

    async def async_get_cache(self, key, **kwargs):
        print_verbose(f"async redis semantic-cache get_cache, kwargs: {kwargs}")
        from redisvl.query import VectorQuery
        import numpy as np
        from litellm.proxy.proxy_server import llm_router, llm_model_list

        # query

        # get the messages
        messages = kwargs["messages"]
        prompt = ""
        for message in messages:
            prompt += message["content"]

        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )
        if llm_router is not None and self.embedding_model in router_model_names:
            user_api_key = kwargs.get("metadata", {}).get("user_api_key", "")
            embedding_response = await llm_router.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
                metadata={
                    "user_api_key": user_api_key,
                    "semantic-cache-embedding": True,
                    "trace_id": kwargs.get("metadata", {}).get("trace_id", None),
                },
            )
        else:
            # convert to embedding
            embedding_response = await litellm.aembedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
            )

        # get the embedding
        embedding = embedding_response["data"][0]["embedding"]

        query = VectorQuery(
            vector=embedding,
            vector_field_name="litellm_embedding",
            return_fields=["response", "prompt", "vector_distance"],
        )
        results = await self.index.aquery(query)
        if results == None:
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
            return None
        if isinstance(results, list):
            if len(results) == 0:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

        vector_distance = results[0]["vector_distance"]
        vector_distance = float(vector_distance)
        similarity = 1 - vector_distance
        cached_prompt = results[0]["prompt"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )

        # update kwargs["metadata"] with similarity, don't rewrite the original metadata
        kwargs.setdefault("metadata", {})["semantic-similarity"] = similarity

        if similarity > self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None
        pass

    async def _index_info(self):
        return await self.index.ainfo()


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

    def set_cache(self, key, value, **kwargs):
        try:
            print_verbose(f"LiteLLM SET Cache - S3. Key={key}. Value={value}")
            ttl = kwargs.get("ttl", None)
            # Convert value to JSON before storing in S3
            serialized_value = json.dumps(value)
            key = self.key_prefix + key

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
            # NON blocking - notify users S3 is throwing an exception
            print_verbose(f"S3 Caching: set_cache() - Got exception from S3: {e}")

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key=key, value=value, **kwargs)

    def get_cache(self, key, **kwargs):
        import boto3, botocore

        try:
            key = self.key_prefix + key

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

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key=key, **kwargs)

    def flush_cache(self):
        pass

    async def disconnect(self):
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

    def delete_cache(self, key):
        if self.in_memory_cache is not None:
            self.in_memory_cache.delete_cache(key)
        if self.redis_cache is not None:
            self.redis_cache.delete_cache(key)


#### LiteLLM.Completion / Embedding Cache ####
class Cache:
    def __init__(
        self,
        type: Optional[Literal["local", "redis", "redis-semantic", "s3"]] = "local",
        host: Optional[str] = None,
        port: Optional[str] = None,
        password: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        supported_call_types: Optional[
            List[
                Literal[
                    "completion",
                    "acompletion",
                    "embedding",
                    "aembedding",
                    "atranscription",
                    "transcription",
                ]
            ]
        ] = [
            "completion",
            "acompletion",
            "embedding",
            "aembedding",
            "atranscription",
            "transcription",
        ],
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
        s3_path: Optional[str] = None,
        redis_semantic_cache_use_async=False,
        redis_semantic_cache_embedding_model="text-embedding-ada-002",
        **kwargs,
    ):
        """
        Initializes the cache based on the given type.

        Args:
            type (str, optional): The type of cache to initialize. Can be "local", "redis", "redis-semantic", or "s3". Defaults to "local".
            host (str, optional): The host address for the Redis cache. Required if type is "redis".
            port (int, optional): The port number for the Redis cache. Required if type is "redis".
            password (str, optional): The password for the Redis cache. Required if type is "redis".
            similarity_threshold (float, optional): The similarity threshold for semantic-caching, Required if type is "redis-semantic"

            supported_call_types (list, optional): List of call types to cache for. Defaults to cache == on for all call types.
            **kwargs: Additional keyword arguments for redis.Redis() cache

        Raises:
            ValueError: If an invalid cache type is provided.

        Returns:
            None. Cache is set as a litellm param
        """
        if type == "redis":
            self.cache: BaseCache = RedisCache(host, port, password, **kwargs)
        elif type == "redis-semantic":
            self.cache = RedisSemanticCache(
                host,
                port,
                password,
                similarity_threshold=similarity_threshold,
                use_async=redis_semantic_cache_use_async,
                embedding_model=redis_semantic_cache_embedding_model,
                **kwargs,
            )
        elif type == "local":
            self.cache = InMemoryCache()
        elif type == "s3":
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
                s3_path=s3_path,
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
        transcription_only_kwargs = [
            "file",
            "language",
        ]
        # combined_kwargs - NEEDS to be ordered across get_cache_key(). Do not use a set()
        combined_kwargs = (
            completion_kwargs + embedding_only_kwargs + transcription_only_kwargs
        )
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
                elif param == "file":
                    metadata_file_name = kwargs.get("metadata", {}).get(
                        "file_name", None
                    )
                    litellm_params_file_name = kwargs.get("litellm_params", {}).get(
                        "file_name", None
                    )
                    if metadata_file_name is not None:
                        param_value = metadata_file_name
                    elif litellm_params_file_name is not None:
                        param_value = litellm_params_file_name
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

    def _get_cache_logic(
        self,
        cached_result: Optional[Any],
        max_age: Optional[float],
    ):
        """
        Common get cache logic across sync + async implementations
        """
        # Check if a timestamp was stored with the cached response
        if (
            cached_result is not None
            and isinstance(cached_result, dict)
            and "timestamp" in cached_result
        ):
            timestamp = cached_result["timestamp"]
            current_time = time.time()

            # Calculate age of the cached response
            response_age = current_time - timestamp

            # Check if the cached response is older than the max-age
            if max_age is not None and response_age > max_age:
                return None  # Cached response is too old

            # If the response is fresh, or there's no max-age requirement, return the cached response
            # cached_response is in `b{} convert it to ModelResponse
            cached_response = cached_result.get("response")
            try:
                if isinstance(cached_response, dict):
                    pass
                else:
                    cached_response = json.loads(
                        cached_response  # type: ignore
                    )  # Convert string to dictionary
            except:
                cached_response = ast.literal_eval(cached_response)  # type: ignore
            return cached_response
        return cached_result

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
            messages = kwargs.get("messages", [])
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = self.get_cache_key(*args, **kwargs)
            if cache_key is not None:
                cache_control_args = kwargs.get("cache", {})
                max_age = cache_control_args.get(
                    "s-max-age", cache_control_args.get("s-maxage", float("inf"))
                )
                cached_result = self.cache.get_cache(cache_key, messages=messages)
                return self._get_cache_logic(
                    cached_result=cached_result, max_age=max_age
                )
        except Exception as e:
            print_verbose(f"An exception occurred: {traceback.format_exc()}")
            return None

    async def async_get_cache(self, *args, **kwargs):
        """
        Async get cache implementation.

        Used for embedding calls in async wrapper
        """
        try:  # never block execution
            messages = kwargs.get("messages", [])
            if "cache_key" in kwargs:
                cache_key = kwargs["cache_key"]
            else:
                cache_key = self.get_cache_key(*args, **kwargs)
            if cache_key is not None:
                cache_control_args = kwargs.get("cache", {})
                max_age = cache_control_args.get(
                    "s-max-age", cache_control_args.get("s-maxage", float("inf"))
                )
                cached_result = await self.cache.async_get_cache(
                    cache_key, *args, **kwargs
                )
                return self._get_cache_logic(
                    cached_result=cached_result, max_age=max_age
                )
        except Exception as e:
            print_verbose(f"An exception occurred: {traceback.format_exc()}")
            return None

    def _add_cache_logic(self, result, *args, **kwargs):
        """
        Common implementation across sync + async add_cache functions
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
                return cache_key, cached_data, kwargs
            else:
                raise Exception("cache key is None")
        except Exception as e:
            raise e

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
            cache_key, cached_data, kwargs = self._add_cache_logic(
                result=result, *args, **kwargs
            )
            self.cache.set_cache(cache_key, cached_data, **kwargs)
        except Exception as e:
            print_verbose(f"LiteLLM Cache: Excepton add_cache: {str(e)}")
            traceback.print_exc()
            pass

    async def async_add_cache(self, result, *args, **kwargs):
        """
        Async implementation of add_cache
        """
        try:
            cache_key, cached_data, kwargs = self._add_cache_logic(
                result=result, *args, **kwargs
            )
            await self.cache.async_set_cache(cache_key, cached_data, **kwargs)
        except Exception as e:
            print_verbose(f"LiteLLM Cache: Excepton add_cache: {str(e)}")
            traceback.print_exc()

    async def async_add_cache_pipeline(self, result, *args, **kwargs):
        """
        Async implementation of add_cache for Embedding calls

        Does a bulk write, to prevent using too many clients
        """
        try:
            cache_list = []
            for idx, i in enumerate(kwargs["input"]):
                preset_cache_key = litellm.cache.get_cache_key(
                    *args, **{**kwargs, "input": i}
                )
                kwargs["cache_key"] = preset_cache_key
                embedding_response = result.data[idx]
                cache_key, cached_data, kwargs = self._add_cache_logic(
                    result=embedding_response,
                    *args,
                    **kwargs,
                )
                cache_list.append((cache_key, cached_data))
            if hasattr(self.cache, "async_set_cache_pipeline"):
                await self.cache.async_set_cache_pipeline(cache_list=cache_list)
            else:
                tasks = []
                for val in cache_list:
                    tasks.append(
                        self.cache.async_set_cache(cache_key, cached_data, **kwargs)
                    )
                await asyncio.gather(*tasks)
        except Exception as e:
            print_verbose(f"LiteLLM Cache: Excepton add_cache: {str(e)}")
            traceback.print_exc()

    async def disconnect(self):
        if hasattr(self.cache, "disconnect"):
            await self.cache.disconnect()


def enable_cache(
    type: Optional[Literal["local", "redis", "s3"]] = "local",
    host: Optional[str] = None,
    port: Optional[str] = None,
    password: Optional[str] = None,
    supported_call_types: Optional[
        List[
            Literal[
                "completion",
                "acompletion",
                "embedding",
                "aembedding",
                "atranscription",
                "transcription",
            ]
        ]
    ] = [
        "completion",
        "acompletion",
        "embedding",
        "aembedding",
        "atranscription",
        "transcription",
    ],
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
        List[
            Literal[
                "completion",
                "acompletion",
                "embedding",
                "aembedding",
                "atranscription",
                "transcription",
            ]
        ]
    ] = [
        "completion",
        "acompletion",
        "embedding",
        "aembedding",
        "atranscription",
        "transcription",
    ],
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
