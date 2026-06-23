"""
Valkey Semantic Cache implementation for LiteLLM

Backs semantic caching with Valkey (for example AWS ElastiCache for Valkey)
running the valkey-search module.

RedisVL cannot drive valkey-search: it gates on a RediSearch module version
that valkey-search does not report, and its SemanticCache index uses a TEXT
field that valkey-search does not implement. This backend therefore talks to
valkey-search directly over redis-py, building a vector index from the field
types valkey-search does support (TAG for cache-key isolation and VECTOR for
the prompt embedding) and running KNN queries for retrieval. Prompt extraction,
embedding generation, and cached-response parsing are reused from
RedisSemanticCache since those are backend agnostic.
"""

import asyncio
import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Any

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from litellm._logging import print_verbose
from litellm._uuid import uuid

from .redis_semantic_cache import RedisSemanticCache


@dataclass(frozen=True, slots=True)
class _ValkeyCacheHit:
    response: str
    distance: float


class ValkeySemanticCache(RedisSemanticCache):
    """Valkey-backed semantic cache for LLM responses."""

    DEFAULT_VALKEY_INDEX_NAME: str = "litellm_semantic_cache_index"
    EMBEDDING_FIELD_NAME: str = "embedding"
    PROMPT_FIELD_NAME: str = "prompt"
    RESPONSE_FIELD_NAME: str = "response"
    DISTANCE_FIELD_NAME: str = "vector_distance"

    def __init__(
        self,
        host: str | None = None,
        port: str | None = None,
        password: str | None = None,
        redis_url: str | None = None,
        similarity_threshold: float | None = None,
        embedding_model: str = "text-embedding-ada-002",
        index_name: str | None = None,
        ssl: bool = False,
        startup_nodes: list | None = None,
        sync_client: Redis | None = None,
        async_client: AsyncRedis | None = None,
        **kwargs: Any,
    ):
        if similarity_threshold is None:
            raise ValueError("similarity_threshold must be provided, passed None")

        if startup_nodes:
            raise ValueError(
                "valkey-semantic does not support cluster-mode-enabled (multi-shard) "
                "endpoints. The async cluster client cannot route the FT.* search "
                "commands reliably. Point it at a cluster-mode-disabled endpoint "
                "instead (a primary with replicas is fine; only horizontal sharding "
                "is unsupported), or pass a single redis_url. On AWS, vector search "
                "needs ElastiCache for Valkey 8.2+ on a node-based cluster."
            )

        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.index_name = index_name or self.DEFAULT_VALKEY_INDEX_NAME
        self.key_prefix = f"{self.index_name}:"
        self._index_dim: int | None = None

        resolved_url = None
        if sync_client is None or async_client is None:
            resolved_url = redis_url or self._build_valkey_url(
                host, port, password, ssl
            )
        self.sync_client = (
            sync_client if sync_client is not None else Redis.from_url(resolved_url)  # type: ignore[arg-type]
        )
        self.async_client = (
            async_client
            if async_client is not None
            else AsyncRedis.from_url(resolved_url)  # type: ignore[arg-type]
        )

        print_verbose(f"Valkey semantic-cache initializing index - {self.index_name}")

    @staticmethod
    def _build_valkey_url(
        host: str | None, port: str | None, password: str | None, ssl: bool = False
    ) -> str:
        host = host or os.environ.get("VALKEY_HOST") or os.environ.get("REDIS_HOST")
        port = port or os.environ.get("VALKEY_PORT") or os.environ.get("REDIS_PORT")
        password = (
            password
            or os.environ.get("VALKEY_PASSWORD")
            or os.environ.get("REDIS_PASSWORD")
        )

        if not host or not port:
            raise ValueError(
                "Missing required Valkey configuration. Provide host and port "
                "(or VALKEY_HOST/VALKEY_PORT), or pass redis_url."
            )

        credentials = f":{password}@" if password else ""
        scheme = "rediss" if ssl else "redis"
        return f"{scheme}://{credentials}{host}:{port}"

    @classmethod
    def _scope_tag(cls, key: str) -> str:
        # valkey-search TAG fields tokenize on punctuation and do not honour
        # backslash escaping, so an arbitrary cache key cannot be matched
        # verbatim. Hashing to hex yields a token that is always exact-match
        # safe and still uniquely isolates a caller's scope.
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

    @staticmethod
    def _embedding_to_bytes(embedding: list[float]) -> bytes:
        return struct.pack(f"<{len(embedding)}f", *embedding)

    def _index_schema(self, dim: int) -> tuple[TagField, VectorField]:
        return (
            TagField(self.CACHE_KEY_FIELD_NAME),
            VectorField(
                self.EMBEDDING_FIELD_NAME,
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
            ),
        )

    def _index_definition(self) -> IndexDefinition:
        return IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH)

    @staticmethod
    def _is_index_exists_error(exc: Exception) -> bool:
        return "already exists" in str(exc).lower()

    @staticmethod
    def _extract_index_dim(info: dict) -> int | None:
        # FT.INFO nests the vector field's "dimensions" one level inside its
        # "index" block, so flatten each field descriptor a single level and
        # scan for the dimensions marker.
        for field in info.get("attributes") or []:
            if not isinstance(field, (list, tuple)):
                continue
            flat = [
                sub
                for item in field
                for sub in (item if isinstance(item, (list, tuple)) else [item])
            ]
            for i, marker in enumerate(flat):
                if marker in (b"dimensions", "dimensions") and i + 1 < len(flat):
                    return int(flat[i + 1])
        return None

    def _assert_dim_matches(self, info: dict, dim: int) -> None:
        existing_dim = self._extract_index_dim(info)
        if existing_dim is not None and existing_dim != dim:
            raise ValueError(
                f"Valkey semantic-cache index '{self.index_name}' already exists with "
                f"embedding dimension {existing_dim}, but the configured embedding "
                f"model produced dimension {dim}. Use a different "
                f"valkey_semantic_cache_index_name or drop the existing index."
            )

    def _ensure_index_sync(self, dim: int) -> None:
        if self._index_dim == dim:
            return
        try:
            self.sync_client.ft(self.index_name).create_index(
                self._index_schema(dim), definition=self._index_definition()
            )
        except Exception as exc:
            if not self._is_index_exists_error(exc):
                raise
            self._assert_dim_matches(self.sync_client.ft(self.index_name).info(), dim)
        self._index_dim = dim

    async def _ensure_index_async(self, dim: int) -> None:
        if self._index_dim == dim:
            return
        try:
            await self.async_client.ft(self.index_name).create_index(
                self._index_schema(dim), definition=self._index_definition()
            )
        except Exception as exc:
            if not self._is_index_exists_error(exc):
                raise
            info = await self.async_client.ft(self.index_name).info()
            self._assert_dim_matches(info, dim)
        self._index_dim = dim

    def _doc_key(self, key: str) -> str:
        return f"{self.key_prefix}{self._scope_tag(key)}:{uuid.uuid4()}"

    def _doc_mapping(
        self, key: str, prompt: str, value_str: str, embedding: list[float]
    ) -> dict:
        return {
            self.CACHE_KEY_FIELD_NAME: self._scope_tag(key),
            self.PROMPT_FIELD_NAME: prompt,
            self.RESPONSE_FIELD_NAME: value_str,
            self.EMBEDDING_FIELD_NAME: self._embedding_to_bytes(embedding),
        }

    def _knn_query(self, key: str) -> Query:
        scope = self._scope_tag(key)
        query_string = (
            f"(@{self.CACHE_KEY_FIELD_NAME}:{{{scope}}})"
            f"=>[KNN 1 @{self.EMBEDDING_FIELD_NAME} $vec AS {self.DISTANCE_FIELD_NAME}]"
        )
        return (
            Query(query_string)
            .return_fields(self.RESPONSE_FIELD_NAME, self.DISTANCE_FIELD_NAME)
            .dialect(2)
        )

    @classmethod
    def _first_hit(cls, search_result: Any) -> _ValkeyCacheHit | None:
        docs = getattr(search_result, "docs", [])
        if not docs:
            return None
        doc = docs[0]
        return _ValkeyCacheHit(
            response=str(getattr(doc, cls.RESPONSE_FIELD_NAME)),
            distance=float(getattr(doc, cls.DISTANCE_FIELD_NAME)),
        )

    def _resolve_hit(self, hit: _ValkeyCacheHit | None, key: str, **kwargs: Any) -> Any:
        if hit is None:
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
            return None

        similarity = 1 - hit.distance
        kwargs.setdefault("metadata", {})["semantic-similarity"] = similarity

        if similarity < self.similarity_threshold:
            return None
        return self._get_cache_logic(cached_response=hit.response)

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        print_verbose(f"Valkey semantic-cache set_cache, kwargs: {kwargs}")
        try:
            prompt = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                print_verbose("No prompt provided for semantic caching")
                return

            embedding = self._get_embedding(prompt)
            self._ensure_index_sync(len(embedding))

            doc_key = self._doc_key(key)
            self.sync_client.hset(
                doc_key, mapping=self._doc_mapping(key, prompt, str(value), embedding)
            )
            ttl = self._get_ttl(**kwargs)
            if ttl is not None:
                self.sync_client.expire(doc_key, ttl)
        except Exception as e:
            print_verbose(f"Error in Valkey semantic-cache set_cache: {str(e)}")

    def get_cache(self, key: str, **kwargs: Any) -> Any:
        print_verbose(f"Valkey semantic-cache get_cache, kwargs: {kwargs}")
        try:
            prompt = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

            embedding = self._get_embedding(prompt)
            self._ensure_index_sync(len(embedding))

            search_result = self.sync_client.ft(self.index_name).search(
                self._knn_query(key),
                query_params={"vec": self._embedding_to_bytes(embedding)},
            )
            return self._resolve_hit(self._first_hit(search_result), key, **kwargs)
        except Exception as e:
            print_verbose(f"Error in Valkey semantic-cache get_cache: {str(e)}")
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        print_verbose(f"Async Valkey semantic-cache set_cache, kwargs: {kwargs}")
        try:
            prompt = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                print_verbose("No prompt provided for semantic caching")
                return

            embedding = await self._get_async_embedding(prompt, **kwargs)
            await self._ensure_index_async(len(embedding))

            doc_key = self._doc_key(key)
            await self.async_client.hset(
                doc_key, mapping=self._doc_mapping(key, prompt, str(value), embedding)
            )
            ttl = self._get_ttl(**kwargs)
            if ttl is not None:
                await self.async_client.expire(doc_key, ttl)
        except Exception as e:
            print_verbose(f"Error in async Valkey semantic-cache set_cache: {str(e)}")

    async def async_get_cache(self, key: str, **kwargs: Any) -> Any:
        print_verbose(f"Async Valkey semantic-cache get_cache, kwargs: {kwargs}")
        try:
            prompt = self._get_prompt_from_kwargs(**kwargs)
            if prompt is None:
                kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0
                return None

            embedding = await self._get_async_embedding(prompt, **kwargs)
            await self._ensure_index_async(len(embedding))

            search_result = await self.async_client.ft(self.index_name).search(
                self._knn_query(key),
                query_params={"vec": self._embedding_to_bytes(embedding)},
            )
            return self._resolve_hit(self._first_hit(search_result), key, **kwargs)
        except Exception as e:
            print_verbose(f"Error in async Valkey semantic-cache get_cache: {str(e)}")
            kwargs.setdefault("metadata", {})["semantic-similarity"] = 0.0

    async def async_set_cache_pipeline(
        self, cache_list: list[tuple[str, Any]], **kwargs: Any
    ) -> None:
        try:
            await asyncio.gather(
                *[
                    self.async_set_cache(key, value, **kwargs)
                    for key, value in cache_list
                ]
            )
        except Exception as e:
            print_verbose(
                f"Error in Valkey semantic-cache async_set_cache_pipeline: {str(e)}"
            )

    async def _index_info(self) -> dict:
        return await self.async_client.ft(self.index_name).info()
