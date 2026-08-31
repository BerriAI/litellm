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
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final, cast

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.constants import SEMANTIC_CACHE_EMBEDDING_TIMEOUT_SECONDS
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.types.utils import EmbeddingResponse

from ._embedding_router import (
    build_router_embedding_metadata,
    resolve_embedding_max_input_tokens,
    resolve_embedding_router,
    resolve_embedding_timeout,
    truncate_embedding_input,
)
from .base_cache import BaseCache

if TYPE_CHECKING:
    from litellm.router import Router


class RedisSemanticCache(BaseCache):
    """
    Redis-backed semantic cache for LLM responses.

    This cache uses vector similarity to find semantically similar prompts that have been
    previously sent to the LLM, allowing for cache hits even when prompts are not identical
    but carry similar meaning.
    """

    DEFAULT_REDIS_INDEX_NAME: str = "litellm_semantic_cache_index"
    CACHE_KEY_FIELD_NAME: str = "litellm_cache_key"
    embedding_max_input_tokens: int | None = None
    embedding_timeout: float = SEMANTIC_CACHE_EMBEDDING_TIMEOUT_SECONDS

    def __init__(
        self,
        host: str | None = None,
        port: str | None = None,
        password: str | None = None,
        redis_url: str | None = None,
        similarity_threshold: float | None = None,
        embedding_model: str = "text-embedding-ada-002",
        index_name: str | None = None,
        embedding_max_input_tokens: int | None = None,
        embedding_timeout: float | None = None,
        **kwargs: object,
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
            embedding_max_input_tokens: Truncate prompts to this many tokens before
                embedding; defaults to the Router deployment's configured max_input_tokens
            embedding_timeout: Seconds a cache lookup may spend embedding the prompt before it
                gives up and lets the request continue to the LLM
            ttl: Default time-to-live for cache entries in seconds
            **kwargs: Additional arguments passed to the Redis client

        Raises:
            Exception: If similarity_threshold is not provided or required Redis
                connection information is missing
        """
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
        self.embedding_max_input_tokens = embedding_max_input_tokens
        self.embedding_timeout = resolve_embedding_timeout(embedding_timeout)

        # Set up Redis connection
        if redis_url is None:
            try:
                # Attempt to use provided parameters or fallback to environment variables
                host = host or os.environ["REDIS_HOST"]
                port = port or os.environ["REDIS_PORT"]
                password = password or os.environ["REDIS_PASSWORD"]
            except KeyError as e:
                # Raise a more informative exception if any of the required keys are missing
                missing_var: Final = e.args[0]
                raise ValueError(
                    f"Missing required Redis configuration: {missing_var}. Provide {missing_var} or redis_url."
                ) from e

            redis_url = f"redis://:{password}@{host}:{port}"

        print_verbose(f"Redis semantic-cache redis_url: {redis_url}")

        # Defer redisvl index construction until first use. redisvl's
        # CustomTextVectorizer eagerly embeds a probe string at construction;
        # building lazily ensures that probe runs after llm_router is wired so
        # per-deployment auth (e.g. Bedrock aws_role_name) is applied.
        self._index_name = index_name
        self._redis_url = redis_url
        self._llmcache = None

    @property
    def llmcache(self) -> object:
        if getattr(self, "_llmcache", None) is None:
            self._llmcache = self._build_llmcache()
        return self._llmcache

    @llmcache.setter
    def llmcache(self, value: object) -> None:
        self._llmcache = value

    def _build_llmcache(self) -> object:
        # CustomTextVectorizer probes its embedding dimension at construction by
        # embedding "dimension test", so the first cache request issues one extra
        # billable embedding on top of the request's own.
        from redisvl.extensions.llmcache import SemanticCache
        from redisvl.utils.vectorize import CustomTextVectorizer

        try:
            cache_vectorizer: Final = CustomTextVectorizer(self._get_embedding)
            return self._init_semantic_cache(
                semantic_cache_cls=SemanticCache,
                index_name=self._index_name,
                redis_url=self._redis_url,
                cache_vectorizer=cache_vectorizer,
            )
        except Exception as e:
            verbose_logger.error("Redis semantic-cache index build failed: %s", e)
            raise

    @classmethod
    def _cache_key_filterable_field(cls) -> dict[str, str]:
        return {
            "name": cls.CACHE_KEY_FIELD_NAME,
            "type": "tag",
        }

    def _init_semantic_cache(
        self,
        semantic_cache_cls: Callable[..., object],
        index_name: str,
        redis_url: str,
        cache_vectorizer: object,
    ) -> object:
        def _is_schema_mismatch(exc: ValueError) -> bool:
            error_message: Final = str(exc).lower()
            return any(phrase in error_message for phrase in ("schema does not match", "index schema"))

        try:
            return semantic_cache_cls(
                name=index_name,
                redis_url=redis_url,
                vectorizer=cache_vectorizer,
                distance_threshold=self.distance_threshold,
                filterable_fields=[self._cache_key_filterable_field()],
                overwrite=False,
            )
        except ValueError as exc:
            if not _is_schema_mismatch(exc):
                raise

            isolated_index_name: Final = f"{index_name}_isolated"
            print_verbose(
                "Redis semantic-cache existing index schema is not isolated; "
                f"using isolated index - {isolated_index_name}"
            )
            try:
                return semantic_cache_cls(
                    name=isolated_index_name,
                    redis_url=redis_url,
                    vectorizer=cache_vectorizer,
                    distance_threshold=self.distance_threshold,
                    filterable_fields=[self._cache_key_filterable_field()],
                    overwrite=False,
                )
            except ValueError as isolated_exc:
                if not _is_schema_mismatch(isolated_exc):
                    raise

                print_verbose(
                    "Redis semantic-cache isolated index schema is stale; "
                    f"recreating isolated index - {isolated_index_name}"
                )
                return semantic_cache_cls(
                    name=isolated_index_name,
                    redis_url=redis_url,
                    vectorizer=cache_vectorizer,
                    distance_threshold=self.distance_threshold,
                    filterable_fields=[self._cache_key_filterable_field()],
                    overwrite=True,
                )

    def _get_cache_filters(self, key: str) -> dict[str, str]:
        return {self.CACHE_KEY_FIELD_NAME: str(key)}

    def _get_cache_key_filter_expression(self, key: str) -> object:
        from redisvl.query.filter import Tag

        return Tag(self.CACHE_KEY_FIELD_NAME) == str(key)

    def _cache_hit_matches_key(self, cache_hit: Mapping[str, object], key: str) -> bool:
        # Pre-isolation entries with no ``litellm_cache_key`` field cannot be
        # safely reassigned to a caller's scope and are treated as misses.
        cached_key = cache_hit.get(self.CACHE_KEY_FIELD_NAME)
        if isinstance(cached_key, bytes):
            cached_key = cached_key.decode("utf-8")
        return cached_key is not None and str(cached_key) == str(key)

    def _get_ttl(self, **kwargs) -> int | None:
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

    @classmethod
    def _get_prompt_from_kwargs(cls, **kwargs) -> str | None:
        """
        Extract a semantic-cache prompt from chat or Responses API request kwargs.
        """
        messages: Final = kwargs.get("messages")
        if messages:
            return get_str_from_messages(messages)

        if "input" not in kwargs:
            return None

        prompt_parts: Final[list[str]] = []
        cls._collect_responses_input_text(kwargs.get("input"), prompt_parts)
        prompt: Final = "\n".join(prompt_parts).strip()
        return prompt or None

    @classmethod
    def _collect_responses_input_text(cls, value: Any, prompt_parts: list[str]) -> None:
        value = cls._coerce_response_input_value(value)
        if value is None:
            return

        if isinstance(value, str):
            stripped_value: Final = value.strip()
            if stripped_value:
                prompt_parts.append(stripped_value)
            return

        if isinstance(value, (list, tuple)):
            for item in value:
                cls._collect_responses_input_text(item, prompt_parts)
            return

        if isinstance(value, dict):
            content = value.get("content")
            if content is not None:
                cls._collect_responses_input_text(content, prompt_parts)
                return

            for text_key in ("text", "output", "input_text", "output_text"):
                text_value = value.get(text_key)
                if isinstance(text_value, str):
                    stripped_text = text_value.strip()
                    if stripped_text:
                        prompt_parts.append(stripped_text)
                        return
            return

        content = getattr(value, "content", None)
        if content is not None:
            cls._collect_responses_input_text(content, prompt_parts)
            return

        for text_key in ("text", "output", "input_text", "output_text"):
            text_value = getattr(value, text_key, None)
            if isinstance(text_value, str):
                stripped_text = text_value.strip()
                if stripped_text:
                    prompt_parts.append(stripped_text)
                    return

    @staticmethod
    def _coerce_response_input_value(value: object) -> object:
        model_dump: Final = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        dict_method: Final = getattr(value, "dict", None)
        if callable(dict_method):
            return dict_method()
        return value

    def _embedding_input(self, prompt: str, router: "Router | None") -> str:
        return truncate_embedding_input(
            prompt,
            self.embedding_model,
            resolve_embedding_max_input_tokens(self.embedding_max_input_tokens, self.embedding_model, router),
        )

    def _get_embedding(self, prompt: str, metadata: dict[str, Any] | None = None) -> list[float]:
        """
        Routes through the proxy Router when the embedding model is a Router
        deployment so per-deployment auth (e.g. Bedrock aws_role_name) applies,
        mirroring ``_get_async_embedding``; otherwise embeds directly.
        """
        try:
            from litellm.proxy.proxy_server import llm_model_list, llm_router
        except ImportError:
            llm_model_list = None
            llm_router = None

        router: Final = resolve_embedding_router(self.embedding_model, llm_router, llm_model_list)
        embedding_input: Final = self._embedding_input(prompt, router)
        if router is not None:
            embedding_response = cast(
                EmbeddingResponse,
                router.embedding(
                    model=self.embedding_model,
                    input=embedding_input,
                    cache={"no-store": True, "no-cache": True},
                    metadata=build_router_embedding_metadata(metadata),
                    timeout=self.embedding_timeout,
                    num_retries=0,
                ),
            )
        else:
            embedding_response = cast(
                EmbeddingResponse,
                litellm.embedding(
                    model=self.embedding_model,
                    input=embedding_input,
                    cache={"no-store": True, "no-cache": True},
                    timeout=self.embedding_timeout,
                    num_retries=0,
                ),
            )
        return embedding_response["data"][0]["embedding"]

    def _get_cache_logic(self, cached_response: Any) -> object:
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
                print_verbose(f"Error parsing cached response: {e}")
                return None

        return cached_response

    def set_cache(self, key: str, value: object, **kwargs) -> None:
        """
        Store a value in the semantic cache.

        Args:
            key: The cache key used to isolate semantic cache entries
            value: The response value to cache
            **kwargs: Additional arguments including 'messages' for the prompt
                and optional 'ttl' for time-to-live
        """
        print_verbose(f"Redis semantic-cache set_cache, kwargs: {kwargs}")

        value_str: str | None = None
        try:
            prompt: Final = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                print_verbose("No prompt provided for semantic caching")
                return

            value_str = str(value)

            prompt_embedding: Final = self._get_embedding(prompt, metadata=kwargs.get("metadata"))

            store_kwargs: Final[dict[str, Any]] = {
                "vector": prompt_embedding,
                "filters": self._get_cache_filters(key),
            }

            # Get TTL and store in Redis semantic cache
            ttl: Final = self._get_ttl(**kwargs)
            if ttl is not None:
                store_kwargs["ttl"] = int(ttl)
            self.llmcache.store(prompt, value_str, **store_kwargs)
        except Exception as e:
            print_verbose(f"Error setting {value_str or value} in the Redis semantic cache: {e}")

    def get_cache(self, key: str, **kwargs) -> object:
        """
        Retrieve a semantically similar cached response.

        Args:
            key: The cache key used to isolate semantic cache entries
            **kwargs: Additional arguments including 'messages' for the prompt

        Returns:
            The cached response if a semantically similar prompt is found, else None
        """
        print_verbose(f"Redis semantic-cache get_cache, kwargs: {kwargs}")

        try:
            prompt: Final = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                print_verbose("No prompt provided for semantic cache lookup")
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

            # Check the cache for semantically similar prompts in this exact
            # LiteLLM cache-key scope.
            prompt_embedding: Final = self._get_embedding(prompt, metadata=kwargs.get("metadata"))
            check_kwargs: Final[Mapping[str, object]] = {
                "prompt": prompt,
                "vector": prompt_embedding,
                "filter_expression": self._get_cache_key_filter_expression(key),
            }
            results: Final = self.llmcache.check(**check_kwargs)

            # Return None if no similar prompts found
            if not results:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

            # Process the best matching result
            cache_hit: Final = results[0]
            if not self._cache_hit_matches_key(cache_hit=cache_hit, key=key):
                print_verbose("Redis semantic-cache hit did not match cache key scope")
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None
            vector_distance: Final = float(cache_hit["vector_distance"])

            # Convert vector distance back to similarity score
            # For cosine distance: 0 = most similar, 2 = least similar
            # While similarity: 1 = most similar, 0 = least similar
            similarity: Final = 1 - vector_distance

            cached_prompt: Final = cache_hit["prompt"]
            cached_response: Final = cache_hit["response"]

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
            print_verbose(f"Error retrieving from Redis semantic cache: {e}")
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0

    async def _get_async_embedding(self, prompt: str, metadata: dict[str, Any] | None = None) -> list[float]:
        """
        Asynchronously generate an embedding for the given prompt.

        Args:
            prompt: The text to generate an embedding for
            metadata: Request metadata forwarded to the Router embedding call

        Returns:
            List[float]: The embedding vector
        """
        try:
            from litellm.proxy.proxy_server import llm_model_list, llm_router
        except ImportError:
            llm_model_list = None
            llm_router = None

        router: Final = resolve_embedding_router(self.embedding_model, llm_router, llm_model_list)
        embedding_input: Final = self._embedding_input(prompt, router)
        embedding_call: Final = (
            router.aembedding(
                model=self.embedding_model,
                input=embedding_input,
                cache={"no-store": True, "no-cache": True},
                metadata=build_router_embedding_metadata(metadata),
                timeout=self.embedding_timeout,
                num_retries=0,
            )
            if router is not None
            else litellm.aembedding(
                model=self.embedding_model,
                input=embedding_input,
                cache={"no-store": True, "no-cache": True},
                timeout=self.embedding_timeout,
                num_retries=0,
            )
        )
        try:
            embedding_response: Final = await asyncio.wait_for(embedding_call, self.embedding_timeout)
            return embedding_response["data"][0]["embedding"]
        except Exception as e:
            print_verbose(f"Error generating async embedding: {e}")
            raise ValueError(f"Failed to generate embedding: {e}") from e

    async def async_set_cache(self, key: str, value: object, **kwargs) -> None:
        """
        Asynchronously store a value in the semantic cache.

        Args:
            key: The cache key used to isolate semantic cache entries
            value: The response value to cache
            **kwargs: Additional arguments including 'messages' for the prompt
                and optional 'ttl' for time-to-live
        """
        print_verbose(f"Async Redis semantic-cache set_cache, kwargs: {kwargs}")

        try:
            prompt: Final = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                print_verbose("No prompt provided for semantic caching")
                return

            value_str: Final = str(value)

            # Generate embedding for the value (response) to cache
            prompt_embedding: Final = await self._get_async_embedding(prompt, metadata=kwargs.get("metadata"))

            store_kwargs: Final[dict[str, Any]] = {
                "vector": prompt_embedding,
                "filters": self._get_cache_filters(key),
            }

            # Get TTL and store in Redis semantic cache
            ttl: Final = self._get_ttl(**kwargs)
            if ttl is not None:
                store_kwargs["ttl"] = ttl
            await self.llmcache.astore(
                prompt,
                value_str,
                **store_kwargs,
            )
        except Exception as e:
            print_verbose(f"Error in async_set_cache: {e}")

    async def async_get_cache(self, key: str, **kwargs) -> object:
        """
        Asynchronously retrieve a semantically similar cached response.

        Args:
            key: The cache key used to isolate semantic cache entries
            **kwargs: Additional arguments including 'messages' for the prompt

        Returns:
            The cached response if a semantically similar prompt is found, else None
        """
        print_verbose(f"Async Redis semantic-cache get_cache, kwargs: {kwargs}")

        try:
            prompt: Final = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                print_verbose("No prompt provided for semantic cache lookup")
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

            # Generate embedding for the prompt
            prompt_embedding: Final = await self._get_async_embedding(prompt, metadata=kwargs.get("metadata"))

            # Check the cache for semantically similar prompts in this exact
            # LiteLLM cache-key scope.
            check_kwargs: Final[Mapping[str, object]] = {
                "prompt": prompt,
                "vector": prompt_embedding,
                "filter_expression": self._get_cache_key_filter_expression(key),
            }
            results: Final = await self.llmcache.acheck(**check_kwargs)

            # handle results / cache hit
            if not results:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

            cache_hit: Final = results[0]
            if not self._cache_hit_matches_key(cache_hit=cache_hit, key=key):
                print_verbose("Redis semantic-cache hit did not match cache key scope")
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None
            vector_distance: Final = float(cache_hit["vector_distance"])

            # Convert vector distance back to similarity
            # For cosine distance: 0 = most similar, 2 = least similar
            # While similarity: 1 = most similar, 0 = least similar
            similarity: Final = 1 - vector_distance

            cached_prompt: Final = cache_hit["prompt"]
            cached_response: Final = cache_hit["response"]

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
            print_verbose(f"Error in async_get_cache: {e}")
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0

    async def _index_info(self) -> Mapping[str, object]:
        """
        Get information about the Redis index.

        Returns:
            Dict[str, Any]: Information about the Redis index
        """
        aindex: Final = await self.llmcache._get_async_index()
        return await aindex.info()

    async def async_set_cache_pipeline(self, cache_list: list[tuple[str, Any]], **kwargs: object) -> None:
        """
        Asynchronously store multiple values in the semantic cache.

        Args:
            cache_list: List of (key, value) tuples to cache
            **kwargs: Additional arguments
        """
        try:
            tasks: Final = []
            for val in cache_list:
                tasks.append(self.async_set_cache(val[0], val[1], **kwargs))
            await asyncio.gather(*tasks)
        except Exception as e:
            print_verbose(f"Error in async_set_cache_pipeline: {e}")
