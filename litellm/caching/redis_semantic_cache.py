"""
Redis Semantic Cache implementation

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

import ast
import json
from typing import Any

import litellm
from litellm._logging import print_verbose

from .base_cache import BaseCache


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
        if use_async is False:
            self.index = SearchIndex.from_dict(schema)
            self.index.connect(redis_url=redis_url)
            try:
                self.index.create(overwrite=False)  # don't overwrite existing index
            except Exception as e:
                print_verbose(f"Got exception creating semantic cache index: {str(e)}")
        elif use_async is True:
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
        except Exception:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def set_cache(self, key, value, **kwargs):
        import numpy as np

        print_verbose(f"redis semantic-cache set_cache, kwargs: {kwargs}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)

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
        self.index.load(new_data)

        return

    def get_cache(self, key, **kwargs):
        print_verbose(f"sync redis semantic-cache get_cache, kwargs: {kwargs}")
        import numpy as np
        from redisvl.query import VectorQuery

        # query
        # get the messages
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)

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
        if results is None:
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

        from litellm.proxy.proxy_server import llm_model_list, llm_router

        try:
            await self.index.acreate(overwrite=False)  # don't overwrite existing index
        except Exception as e:
            print_verbose(f"Got exception creating semantic cache index: {str(e)}")
        print_verbose(f"async redis semantic-cache set_cache, kwargs: {kwargs}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)
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
        await self.index.aload(new_data)
        return

    async def async_get_cache(self, key, **kwargs):
        print_verbose(f"async redis semantic-cache get_cache, kwargs: {kwargs}")
        import numpy as np
        from redisvl.query import VectorQuery

        from litellm.proxy.proxy_server import llm_model_list, llm_router

        # query
        # get the messages
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)

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
        if results is None:
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
