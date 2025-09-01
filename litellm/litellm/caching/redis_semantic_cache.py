"""
Redis Semantic Cache implementation for LiteLLM

The RedisSemanticCache provides semantic caching functionality using Redis as a backend.
This cache stores responses based on the semantic similarity of prompts rather than
exact matching, allowing for more flexible caching of LLM responses.

This implementation uses RedisVL's SemanticCache to find semantically similar prompts
and their cached responses.
"""

import ast
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm._logging import print_verbose
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.types.utils import EmbeddingResponse

from .base_cache import BaseCache


class RedisSemanticCache(BaseCache):
    """
    Redis-backed semantic cache for LLM responses.

    This cache uses vector similarity to find semantically similar prompts that have been
    previously sent to the LLM, allowing for cache hits even when prompts are not identical
    but carry similar meaning.
    """

    DEFAULT_REDIS_INDEX_NAME: str = "litellm_semantic_cache_index"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        password: Optional[str] = None,
        redis_url: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        embedding_model: str = "text-embedding-ada-002",
        index_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Redis Semantic Cache.

        Args:
            host: Redis host address
            port: Redis port
            password: Redis password
            redis_url: Full Redis URL (alternative to separate host/port/password)
            similarity_threshold: Threshold for semantic similarity (0.0 to 1.0)
                where 1.0 requires exact matches and 0.0 accepts any match
            embedding_model: Model to use for generating embeddings
            index_name: Name for the Redis index
            ttl: Default time-to-live for cache entries in seconds
            **kwargs: Additional arguments passed to the Redis client

        Raises:
            Exception: If similarity_threshold is not provided or required Redis
                connection information is missing
        """
        from redisvl.extensions.llmcache import SemanticCache
        from redisvl.utils.vectorize import CustomTextVectorizer

        if index_name is None:
            index_name = self.DEFAULT_REDIS_INDEX_NAME

        print_verbose(f"Redis semantic-cache initializing index - {index_name}")

        # Validate similarity threshold
        if similarity_threshold is None:
            raise ValueError("similarity_threshold must be provided, passed None")

        # Store configuration
        self.similarity_threshold = similarity_threshold

        # Convert similarity threshold [0,1] to distance threshold [0,2]
        # For cosine distance: 0 = most similar, 2 = least similar
        # While similarity: 1 = most similar, 0 = least similar
        self.distance_threshold = 1 - similarity_threshold
        self.embedding_model = embedding_model

        # Set up Redis connection
        if redis_url is None:
            try:
                # Attempt to use provided parameters or fallback to environment variables
                host = host or os.environ["REDIS_HOST"]
                port = port or os.environ["REDIS_PORT"]
                password = password or os.environ["REDIS_PASSWORD"]
            except KeyError as e:
                # Raise a more informative exception if any of the required keys are missing
                missing_var = e.args[0]
                raise ValueError(
                    f"Missing required Redis configuration: {missing_var}. "
                    f"Provide {missing_var} or redis_url."
                ) from e

            redis_url = f"redis://:{password}@{host}:{port}"

        print_verbose(f"Redis semantic-cache redis_url: {redis_url}")

        # Initialize the Redis vectorizer and cache
        cache_vectorizer = CustomTextVectorizer(self._get_embedding)

        self.llmcache = SemanticCache(
            name=index_name,
            redis_url=redis_url,
            vectorizer=cache_vectorizer,
            distance_threshold=self.distance_threshold,
            overwrite=False,
        )

    def _get_ttl(self, **kwargs) -> Optional[int]:
        """
        Get the TTL (time-to-live) value for cache entries.

        Args:
            **kwargs: Keyword arguments that may contain a custom TTL

        Returns:
            Optional[int]: The TTL value in seconds, or None if no TTL should be applied
        """
        ttl = kwargs.get("ttl")
        if ttl is not None:
            ttl = int(ttl)
        return ttl

    def _get_embedding(self, prompt: str) -> List[float]:
        """
        Generate an embedding vector for the given prompt using the configured embedding model.

        Args:
            prompt: The text to generate an embedding for

        Returns:
            List[float]: The embedding vector
        """
        # Create an embedding from prompt
        embedding_response = cast(
            EmbeddingResponse,
            litellm.embedding(
                model=self.embedding_model,
                input=prompt,
                cache={"no-store": True, "no-cache": True},
            ),
        )
        embedding = embedding_response["data"][0]["embedding"]
        return embedding

    def _get_cache_logic(self, cached_response: Any) -> Any:
        """
        Process the cached response to prepare it for use.

        Args:
            cached_response: The raw cached response

        Returns:
            The processed cache response, or None if input was None
        """
        if cached_response is None:
            return cached_response

        # Convert bytes to string if needed
        if isinstance(cached_response, bytes):
            cached_response = cached_response.decode("utf-8")

        # Convert string representation to Python object
        try:
            cached_response = json.loads(cached_response)
        except json.JSONDecodeError:
            try:
                cached_response = ast.literal_eval(cached_response)
            except (ValueError, SyntaxError) as e:
                print_verbose(f"Error parsing cached response: {str(e)}")
                return None

        return cached_response

    def set_cache(self, key: str, value: Any, **kwargs) -> None:
        """
        Store a value in the semantic cache.

        Args:
            key: The cache key (not directly used in semantic caching)
            value: The response value to cache
            **kwargs: Additional arguments including 'messages' for the prompt
                and optional 'ttl' for time-to-live
        """
        print_verbose(f"Redis semantic-cache set_cache, kwargs: {kwargs}")

        value_str: Optional[str] = None
        try:
            # Extract the prompt from messages
            messages = kwargs.get("messages", [])
            if not messages:
                print_verbose("No messages provided for semantic caching")
                return

            prompt = get_str_from_messages(messages)
            value_str = str(value)

            # Get TTL and store in Redis semantic cache
            ttl = self._get_ttl(**kwargs)
            if ttl is not None:
                self.llmcache.store(prompt, value_str, ttl=int(ttl))
            else:
                self.llmcache.store(prompt, value_str)
        except Exception as e:
            print_verbose(
                f"Error setting {value_str or value} in the Redis semantic cache: {str(e)}"
            )

    def get_cache(self, key: str, **kwargs) -> Any:
        """
        Retrieve a semantically similar cached response.

        Args:
            key: The cache key (not directly used in semantic caching)
            **kwargs: Additional arguments including 'messages' for the prompt

        Returns:
            The cached response if a semantically similar prompt is found, else None
        """
        print_verbose(f"Redis semantic-cache get_cache, kwargs: {kwargs}")

        try:
            # Extract the prompt from messages
            messages = kwargs.get("messages", [])
            if not messages:
                print_verbose("No messages provided for semantic cache lookup")
                return None

            prompt = get_str_from_messages(messages)
            # Check the cache for semantically similar prompts
            results = self.llmcache.check(prompt=prompt)

            # Return None if no similar prompts found
            if not results:
                return None

            # Process the best matching result
            cache_hit = results[0]
            vector_distance = float(cache_hit["vector_distance"])

            # Convert vector distance back to similarity score
            # For cosine distance: 0 = most similar, 2 = least similar
            # While similarity: 1 = most similar, 0 = least similar
            similarity = 1 - vector_distance

            cached_prompt = cache_hit["prompt"]
            cached_response = cache_hit["response"]

            print_verbose(
                f"Cache hit: similarity threshold: {self.similarity_threshold}, "
                f"actual similarity: {similarity}, "
                f"current prompt: {prompt}, "
                f"cached prompt: {cached_prompt}"
            )

            return self._get_cache_logic(cached_response=cached_response)
        except Exception as e:
            print_verbose(f"Error retrieving from Redis semantic cache: {str(e)}")

    async def _get_async_embedding(self, prompt: str, **kwargs) -> List[float]:
        """
        Asynchronously generate an embedding for the given prompt.

        Args:
            prompt: The text to generate an embedding for
            **kwargs: Additional arguments that may contain metadata

        Returns:
            List[float]: The embedding vector
        """
        from litellm.proxy.proxy_server import llm_model_list, llm_router

        # Route the embedding request through the proxy if appropriate
        router_model_names = (
            [m["model_name"] for m in llm_model_list]
            if llm_model_list is not None
            else []
        )

        try:
            if llm_router is not None and self.embedding_model in router_model_names:
                # Use the router for embedding generation
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
                # Generate embedding directly
                embedding_response = await litellm.aembedding(
                    model=self.embedding_model,
                    input=prompt,
                    cache={"no-store": True, "no-cache": True},
                )

            # Extract and return the embedding vector
            return embedding_response["data"][0]["embedding"]
        except Exception as e:
            print_verbose(f"Error generating async embedding: {str(e)}")
            raise ValueError(f"Failed to generate embedding: {str(e)}") from e

    async def async_set_cache(self, key: str, value: Any, **kwargs) -> None:
        """
        Asynchronously store a value in the semantic cache.

        Args:
            key: The cache key (not directly used in semantic caching)
            value: The response value to cache
            **kwargs: Additional arguments including 'messages' for the prompt
                and optional 'ttl' for time-to-live
        """
        print_verbose(f"Async Redis semantic-cache set_cache, kwargs: {kwargs}")

        try:
            # Extract the prompt from messages
            messages = kwargs.get("messages", [])
            if not messages:
                print_verbose("No messages provided for semantic caching")
                return

            prompt = get_str_from_messages(messages)
            value_str = str(value)

            # Generate embedding for the value (response) to cache
            prompt_embedding = await self._get_async_embedding(prompt, **kwargs)

            # Get TTL and store in Redis semantic cache
            ttl = self._get_ttl(**kwargs)
            if ttl is not None:
                await self.llmcache.astore(
                    prompt,
                    value_str,
                    vector=prompt_embedding,  # Pass through custom embedding
                    ttl=ttl,
                )
            else:
                await self.llmcache.astore(
                    prompt,
                    value_str,
                    vector=prompt_embedding,  # Pass through custom embedding
                )
        except Exception as e:
            print_verbose(f"Error in async_set_cache: {str(e)}")

    async def async_get_cache(self, key: str, **kwargs) -> Any:
        """
        Asynchronously retrieve a semantically similar cached response.

        Args:
            key: The cache key (not directly used in semantic caching)
            **kwargs: Additional arguments including 'messages' for the prompt

        Returns:
            The cached response if a semantically similar prompt is found, else None
        """
        print_verbose(f"Async Redis semantic-cache get_cache, kwargs: {kwargs}")

        try:
            # Extract the prompt from messages
            messages = kwargs.get("messages", [])
            if not messages:
                print_verbose("No messages provided for semantic cache lookup")
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

            prompt = get_str_from_messages(messages)

            # Generate embedding for the prompt
            prompt_embedding = await self._get_async_embedding(prompt, **kwargs)

            # Check the cache for semantically similar prompts
            results = await self.llmcache.acheck(prompt=prompt, vector=prompt_embedding)

            # handle results / cache hit
            if not results:
                kwargs.setdefault("metadata", {})[
                    "semantic-similarity"
                ] = 0.0  # TODO why here but not above??
                return None

            cache_hit = results[0]
            vector_distance = float(cache_hit["vector_distance"])

            # Convert vector distance back to similarity
            # For cosine distance: 0 = most similar, 2 = least similar
            # While similarity: 1 = most similar, 0 = least similar
            similarity = 1 - vector_distance

            cached_prompt = cache_hit["prompt"]
            cached_response = cache_hit["response"]

            # update kwargs["metadata"] with similarity, don't rewrite the original metadata
            kwargs.setdefault("metadata", {})["semantic-similarity"] = similarity

            print_verbose(
                f"Cache hit: similarity threshold: {self.similarity_threshold}, "
                f"actual similarity: {similarity}, "
                f"current prompt: {prompt}, "
                f"cached prompt: {cached_prompt}"
            )

            return self._get_cache_logic(cached_response=cached_response)
        except Exception as e:
            print_verbose(f"Error in async_get_cache: {str(e)}")
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0

    async def _index_info(self) -> Dict[str, Any]:
        """
        Get information about the Redis index.

        Returns:
            Dict[str, Any]: Information about the Redis index
        """
        aindex = await self.llmcache._get_async_index()
        return await aindex.info()

    async def async_set_cache_pipeline(
        self, cache_list: List[Tuple[str, Any]], **kwargs
    ) -> None:
        """
        Asynchronously store multiple values in the semantic cache.

        Args:
            cache_list: List of (key, value) tuples to cache
            **kwargs: Additional arguments
        """
        try:
            tasks = []
            for val in cache_list:
                tasks.append(self.async_set_cache(val[0], val[1], **kwargs))
            await asyncio.gather(*tasks)
        except Exception as e:
            print_verbose(f"Error in async_set_cache_pipeline: {str(e)}")
