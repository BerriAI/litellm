"""
Valkey vector store provider.

Valkey's vector search (the valkey-search module) speaks RESP only, no HTTP
API, so this config extends BaseDirectVectorStoreConfig and executes the
FT.SEARCH KNN query itself via redis-py instead of shaping an httpx request.
Documents are HASHes indexed by an FT index named after the vector_store_id.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn

import httpx
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.llms.base_llm.vector_store.transformation import BaseDirectVectorStoreConfig
from litellm.llms.valkey.common_utils import build_valkey_url, pack_vector
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis
    from redis.commands.search.document import Document
    from redis.commands.search.query import Query
    from redis.commands.search.result import Result

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_VALKEY_PORT: Final = 6379
DEFAULT_SOCKET_CONNECT_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_SOCKET_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_MAX_NUM_RESULTS: Final = 10
MIN_MAX_NUM_RESULTS: Final = 1
MAX_MAX_NUM_RESULTS: Final = 50
DEFAULT_EMBEDDING_FIELD_NAME: Final = "embedding"
DEFAULT_TEXT_FIELD_NAME: Final = "text"
DISTANCE_FIELD_NAME: Final = "vector_distance"

_EMPTY_EMBEDDING_CONFIG: Final = MappingProxyType({})
_REDIS_INSTALL_HINT: Final = (
    "The Valkey vector store requires the 'redis' package. Run 'pip install redis' to install it."
)
_SEARCH_ONLY_MESSAGE: Final = "Valkey vector store is search-only; create indexes with FT.CREATE directly"


def _import_sync_redis() -> "type[Redis]":
    try:
        from redis import Redis as SyncRedisClient
    except ImportError as e:
        raise ValueError(_REDIS_INSTALL_HINT) from e
    return SyncRedisClient


def _import_async_redis() -> "type[AsyncRedis]":
    try:
        from redis.asyncio import Redis as AsyncRedisClient
    except ImportError as e:
        raise ValueError(_REDIS_INSTALL_HINT) from e
    return AsyncRedisClient


def _import_query() -> "type[Query]":
    try:
        from redis.commands.search.query import Query as RedisQuery
    except ImportError as e:
        raise ValueError(_REDIS_INSTALL_HINT) from e
    return RedisQuery


class _ValkeySearchParams(BaseModel):
    """Typed view over the vector store's litellm_params; unrelated keys are ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_embedding_model: str | None = None
    litellm_embedding_config: Mapping[str, object] | None = None
    valkey_host: str | None = None
    valkey_port: int | None = None
    valkey_password: str | None = None
    valkey_ssl: bool | None = None
    valkey_text_field: str | None = None
    valkey_embedding_field: str | None = None

    @property
    def text_field(self) -> str:
        return self.valkey_text_field or DEFAULT_TEXT_FIELD_NAME

    @property
    def embedding_field(self) -> str:
        return self.valkey_embedding_field or DEFAULT_EMBEDDING_FIELD_NAME

    def require_embedding_model(self) -> str:
        if not self.litellm_embedding_model:
            raise ValueError(
                "litellm_embedding_model is required in litellm_params for the Valkey vector store. "
                "Example: litellm_params['litellm_embedding_model'] = 'openai/text-embedding-3-small'"
            )
        return self.litellm_embedding_model

    def connection_url(self) -> str:
        if not self.valkey_host:
            raise ValueError(
                "valkey_host is required in litellm_params for the Valkey vector store. "
                "Set it on the vector store's litellm_params, e.g. valkey_host: my-valkey.example.com"
            )
        return build_valkey_url(
            host=self.valkey_host,
            port=str(self.valkey_port or DEFAULT_VALKEY_PORT),
            password=self.valkey_password,
            ssl=bool(self.valkey_ssl),
        )


class ValkeyVectorStoreConfig(BaseDirectVectorStoreConfig):
    def __init__(
        self,
        sync_client: "Redis | None" = None,
        async_client: "AsyncRedis | None" = None,
        embedding_fn: Callable[..., EmbeddingResponse] | None = None,
        aembedding_fn: Callable[..., Awaitable[EmbeddingResponse]] | None = None,
    ) -> None:
        super().__init__()
        self.sync_client = sync_client
        self.async_client = async_client
        self.embedding_fn = embedding_fn if embedding_fn is not None else litellm.embedding
        self.aembedding_fn = aembedding_fn if aembedding_fn is not None else litellm.aembedding

    @staticmethod
    def _query_text(query: str | Sequence[str]) -> str:
        if isinstance(query, str):
            return query
        if not query:
            raise ValueError("query must not be empty")
        return " ".join(query)

    @staticmethod
    def _socket_timeouts(timeout: float | httpx.Timeout | None) -> tuple[float, float]:
        if isinstance(timeout, httpx.Timeout):
            return (
                timeout.connect or DEFAULT_SOCKET_CONNECT_TIMEOUT_SECONDS,
                timeout.read or DEFAULT_SOCKET_TIMEOUT_SECONDS,
            )
        if timeout is not None:
            return (min(float(timeout), DEFAULT_SOCKET_CONNECT_TIMEOUT_SECONDS), float(timeout))
        return (DEFAULT_SOCKET_CONNECT_TIMEOUT_SECONDS, DEFAULT_SOCKET_TIMEOUT_SECONDS)

    @staticmethod
    def _knn_limit(vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams) -> int:
        requested: Final = vector_store_search_optional_params.get("max_num_results")
        if requested is None:
            return DEFAULT_MAX_NUM_RESULTS
        if not MIN_MAX_NUM_RESULTS <= requested <= MAX_MAX_NUM_RESULTS:
            raise ValueError(
                f"max_num_results must be between {MIN_MAX_NUM_RESULTS} and {MAX_MAX_NUM_RESULTS}, got {requested}"
            )
        return requested

    @classmethod
    def _knn_query(
        cls,
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        embedding_field: str,
        text_field: str,
    ) -> "Query":
        if vector_store_search_optional_params.get("filters") is not None:
            raise ValueError("Valkey vector store does not support the filters parameter yet")
        k: Final = cls._knn_limit(vector_store_search_optional_params)
        query_cls: Final = _import_query()
        knn_expr: Final = f"*=>[KNN {k} @{embedding_field} $vec AS {DISTANCE_FIELD_NAME}]"
        # valkey-search rejects SORTBY on the KNN distance alias, so results are
        # re-ordered client-side in _to_response instead.
        return query_cls(knn_expr).return_fields(text_field, DISTANCE_FIELD_NAME).paging(0, k).dialect(2)

    @staticmethod
    def _to_result(doc: "Document", text_field: str) -> VectorStoreSearchResult:
        content: Final = [  # mutable-ok: VectorStoreSearchResult declares a list of content parts
            VectorStoreResultContent(text=str(getattr(doc, text_field, "")), type="text")
        ]
        return VectorStoreSearchResult(
            score=1.0 - float(getattr(doc, DISTANCE_FIELD_NAME)),
            content=content,
            file_id=getattr(doc, "id", None),
            filename=getattr(doc, "id", None),
        )

    @classmethod
    def _to_response(cls, search_result: "Result", query_text: str, text_field: str) -> VectorStoreSearchResponse:
        docs: Final = getattr(search_result, "docs", None) or ()
        data: Final = sorted(
            (cls._to_result(doc, text_field) for doc in docs),
            key=lambda result: result.get("score") or 0.0,
            reverse=True,
        )
        return VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query=query_text,
            data=data,
        )

    def execute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        params: Final = _ValkeySearchParams.model_validate(litellm_params)
        query_text: Final = self._query_text(query)
        knn: Final = self._knn_query(
            vector_store_search_optional_params,
            embedding_field=params.embedding_field,
            text_field=params.text_field,
        )
        embedding_response: Final = self.embedding_fn(
            model=params.require_embedding_model(),
            input=[query_text],  # mutable-ok: litellm.embedding's input contract is a list
            **(params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG),
        )
        vec_params: Final = {"vec": pack_vector(embedding_response.data[0]["embedding"])}  # mutable-ok: redis-py API

        if self.sync_client is not None:
            raw: Final = self.sync_client.ft(vector_store_id).search(knn, query_params=vec_params)
            return self._to_response(raw, query_text, params.text_field)

        connect_timeout, op_timeout = self._socket_timeouts(timeout)
        client: Final = _import_sync_redis().from_url(
            params.connection_url(),
            socket_connect_timeout=connect_timeout,
            socket_timeout=op_timeout,
        )
        try:
            raw_result: Final = client.ft(vector_store_id).search(knn, query_params=vec_params)
            return self._to_response(raw_result, query_text, params.text_field)
        finally:
            client.close()

    async def aexecute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        params: Final = _ValkeySearchParams.model_validate(litellm_params)
        query_text: Final = self._query_text(query)
        knn: Final = self._knn_query(
            vector_store_search_optional_params,
            embedding_field=params.embedding_field,
            text_field=params.text_field,
        )
        embedding_response: Final = await self.aembedding_fn(
            model=params.require_embedding_model(),
            input=[query_text],  # mutable-ok: litellm.embedding's input contract is a list
            **(params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG),
        )
        vec_params: Final = {"vec": pack_vector(embedding_response.data[0]["embedding"])}  # mutable-ok: redis-py API

        if self.async_client is not None:
            raw: Final = await self.async_client.ft(vector_store_id).search(  # pyright: ignore[reportGeneralTypeIssues]  # types-redis 4.6 stubs shadow redis 5.3.1 and type the async client's ft() as the sync Search, so search() returns a non-awaitable Result; it is a coroutine at runtime
                knn, query_params=vec_params
            )
            return self._to_response(raw, query_text, params.text_field)

        connect_timeout, op_timeout = self._socket_timeouts(timeout)
        client: Final = _import_async_redis().from_url(
            params.connection_url(),
            socket_connect_timeout=connect_timeout,
            socket_timeout=op_timeout,
        )
        try:
            raw_result: Final = await client.ft(vector_store_id).search(  # pyright: ignore[reportGeneralTypeIssues]  # types-redis 4.6 stubs shadow redis 5.3.1 and type the async client's ft() as the sync Search, so search() returns a non-awaitable Result; it is a coroutine at runtime
                knn, query_params=vec_params
            )
            return self._to_response(raw_result, query_text, params.text_field)
        finally:
            await client.aclose()

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> NoReturn:
        raise NotImplementedError(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_response(self, response: httpx.Response) -> NoReturn:
        raise NotImplementedError(_SEARCH_ONLY_MESSAGE)
