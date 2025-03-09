"""
Qdrant Semantic Cache implementation

Has 4 methods:
    - set_cache
    - get_cache
    - async_set_cache
    - async_get_cache
"""

import ast
import asyncio
import json
from typing import Any

import litellm
from litellm._logging import print_verbose

from .base_cache import BaseCache


class QdrantSemanticCache(BaseCache):
    def __init__(  # noqa: PLR0915
        self,
        qdrant_api_base=None,
        qdrant_api_key=None,
        collection_name=None,
        similarity_threshold=None,
        quantization_config=None,
        embedding_model="text-embedding-ada-002",
        host_type=None,
    ):
        import os

        from litellm.llms.custom_httpx.http_handler import (
            _get_httpx_client,
            get_async_httpx_client,
            httpxSpecialProvider,
        )
        from litellm.secret_managers.main import get_secret_str

        if collection_name is None:
            raise Exception("collection_name must be provided, passed None")

        self.collection_name = collection_name
        print_verbose(
            f"qdrant semantic-cache initializing COLLECTION - {self.collection_name}"
        )

        if similarity_threshold is None:
            raise Exception("similarity_threshold must be provided, passed None")
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        headers = {}

        # check if defined as os.environ/ variable
        if qdrant_api_base:
            if isinstance(qdrant_api_base, str) and qdrant_api_base.startswith(
                "os.environ/"
            ):
                qdrant_api_base = get_secret_str(qdrant_api_base)
        if qdrant_api_key:
            if isinstance(qdrant_api_key, str) and qdrant_api_key.startswith(
                "os.environ/"
            ):
                qdrant_api_key = get_secret_str(qdrant_api_key)

        qdrant_api_base = (
            qdrant_api_base or os.getenv("QDRANT_URL") or os.getenv("QDRANT_API_BASE")
        )
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        headers = {"Content-Type": "application/json"}
        if qdrant_api_key:
            headers["api-key"] = qdrant_api_key

        if qdrant_api_base is None:
            raise ValueError("Qdrant url must be provided")

        self.qdrant_api_base = qdrant_api_base
        self.qdrant_api_key = qdrant_api_key
        print_verbose(f"qdrant semantic-cache qdrant_api_base: {self.qdrant_api_base}")

        self.headers = headers

        self.sync_client = _get_httpx_client()
        self.async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.Caching
        )

        if quantization_config is None:
            print_verbose(
                "Quantization config is not provided. Default binary quantization will be used."
            )
        collection_exists = self.sync_client.get(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/exists",
            headers=self.headers,
        )
        if collection_exists.status_code != 200:
            raise ValueError(
                f"Error from qdrant checking if /collections exist {collection_exists.text}"
            )

        if collection_exists.json()["result"]["exists"]:
            collection_details = self.sync_client.get(
                url=f"{self.qdrant_api_base}/collections/{self.collection_name}",
                headers=self.headers,
            )
            self.collection_info = collection_details.json()
            print_verbose(
                f"Collection already exists.\nCollection details:{self.collection_info}"
            )
        else:
            if quantization_config is None or quantization_config == "binary":
                quantization_params = {
                    "binary": {
                        "always_ram": False,
                    }
                }
            elif quantization_config == "scalar":
                quantization_params = {
                    "scalar": {"type": "int8", "quantile": 0.99, "always_ram": False}
                }
            elif quantization_config == "product":
                quantization_params = {
                    "product": {"compression": "x16", "always_ram": False}
                }
            else:
                raise Exception(
                    "Quantization config must be one of 'scalar', 'binary' or 'product'"
                )

            new_collection_status = self.sync_client.put(
                url=f"{self.qdrant_api_base}/collections/{self.collection_name}",
                json={
                    "vectors": {"size": 1536, "distance": "Cosine"},
                    "quantization_config": quantization_params,
                },
                headers=self.headers,
            )
            if new_collection_status.json()["result"]:
                collection_details = self.sync_client.get(
                    url=f"{self.qdrant_api_base}/collections/{self.collection_name}",
                    headers=self.headers,
                )
                self.collection_info = collection_details.json()
                print_verbose(
                    f"New collection created.\nCollection details:{self.collection_info}"
                )
            else:
                raise Exception("Error while creating new collection")

    def _get_cache_logic(self, cached_response: Any):
        if cached_response is None:
            return cached_response
        try:
            cached_response = json.loads(
                cached_response
            )  # Convert string to dictionary
        except Exception:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def set_cache(self, key, value, **kwargs):
        print_verbose(f"qdrant semantic-cache set_cache, kwargs: {kwargs}")
        import uuid

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

        value = str(value)
        assert isinstance(value, str)

        data = {
            "points": [
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "text": prompt,
                        "response": value,
                    },
                },
            ]
        }
        self.sync_client.put(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points",
            headers=self.headers,
            json=data,
        )
        return

    def get_cache(self, key, **kwargs):
        print_verbose(f"sync qdrant semantic-cache get_cache, kwargs: {kwargs}")

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

        data = {
            "vector": embedding,
            "params": {
                "quantization": {
                    "ignore": False,
                    "rescore": True,
                    "oversampling": 3.0,
                }
            },
            "limit": 1,
            "with_payload": True,
        }

        search_response = self.sync_client.post(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points/search",
            headers=self.headers,
            json=data,
        )
        results = search_response.json()["result"]

        if results is None:
            return None
        if isinstance(results, list):
            if len(results) == 0:
                return None

        similarity = results[0]["score"]
        cached_prompt = results[0]["payload"]["text"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )
        if similarity >= self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["payload"]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None
        pass

    async def async_set_cache(self, key, value, **kwargs):
        import uuid

        from litellm.proxy.proxy_server import llm_model_list, llm_router

        print_verbose(f"async qdrant semantic-cache set_cache, kwargs: {kwargs}")

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

        value = str(value)
        assert isinstance(value, str)

        data = {
            "points": [
                {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "text": prompt,
                        "response": value,
                    },
                },
            ]
        }

        await self.async_client.put(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points",
            headers=self.headers,
            json=data,
        )
        return

    async def async_get_cache(self, key, **kwargs):
        print_verbose(f"async qdrant semantic-cache get_cache, kwargs: {kwargs}")
        from litellm.proxy.proxy_server import llm_model_list, llm_router

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

        data = {
            "vector": embedding,
            "params": {
                "quantization": {
                    "ignore": False,
                    "rescore": True,
                    "oversampling": 3.0,
                }
            },
            "limit": 1,
            "with_payload": True,
        }

        search_response = await self.async_client.post(
            url=f"{self.qdrant_api_base}/collections/{self.collection_name}/points/search",
            headers=self.headers,
            json=data,
        )

        results = search_response.json()["result"]

        if results is None:
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
            return None
        if isinstance(results, list):
            if len(results) == 0:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

        similarity = results[0]["score"]
        cached_prompt = results[0]["payload"]["text"]

        # check similarity, if more than self.similarity_threshold, return results
        print_verbose(
            f"semantic cache: similarity threshold: {self.similarity_threshold}, similarity: {similarity}, prompt: {prompt}, closest_cached_prompt: {cached_prompt}"
        )

        # update kwargs["metadata"] with similarity, don't rewrite the original metadata
        kwargs.setdefault("metadata", {})["semantic-similarity"] = similarity

        if similarity >= self.similarity_threshold:
            # cache hit !
            cached_value = results[0]["payload"]["response"]
            print_verbose(
                f"got a cache hit, similarity: {similarity}, Current prompt: {prompt}, cached_prompt: {cached_prompt}"
            )
            return self._get_cache_logic(cached_response=cached_value)
        else:
            # cache miss !
            return None
        pass

    async def _collection_info(self):
        return self.collection_info

    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        tasks = []
        for val in cache_list:
            tasks.append(self.async_set_cache(val[0], val[1], **kwargs))
        await asyncio.gather(*tasks)
