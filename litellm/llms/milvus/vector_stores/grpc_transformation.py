import typing
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import Protocol, ReadOnly, TypedDict

import litellm
from litellm.llms.base_llm.vector_store.transformation import (
    BaseDirectVectorStoreConfig,
    LiteLLMVectorStoreEmbeddingExecutor,
    VectorStoreEmbeddingExecutor,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.vector_stores import (
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

from .transformation import MILVUS_OPTIONAL_PARAMS

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_LIMIT: Final = 10
DEFAULT_TEXT_FIELD: Final = "text"
_EMPTY_EMBEDDING_CONFIG: Final[Mapping[str, object]] = MappingProxyType({})
_PYMILVUS_INSTALL_HINT: Final = (
    "Milvus gRPC transport requires the 'pymilvus' package. Install it with 'pip install litellm[milvus]'."
)
_MILVUS_CONNECT_FAILURE_CODE: Final = 2
_MILVUS_CONNECTION_HINT: Final = (
    "Milvus gRPC connection failed. Check that api_base points at a reachable gRPC endpoint "
    "and that api_key holds a valid 'user:password' token."
)
_MILVUS_ENTITY_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_STRING_KEYS_ADAPTER: Final = TypeAdapter(tuple[str, ...])


class _MilvusSearchArguments(TypedDict):
    collection_name: ReadOnly[str]
    data: ReadOnly[list[list[float]]]  # mutable-ok: PyMilvus requires nested list search data
    anns_field: ReadOnly[str | None]
    limit: ReadOnly[int]
    filter: ReadOnly[str]
    offset: ReadOnly[int | None]
    group_by_field: ReadOnly[str | None]
    output_fields: ReadOnly[list[str]]  # mutable-ok: PyMilvus requires list output fields
    search_params: ReadOnly[dict[str, object] | None]  # mutable-ok: PyMilvus requires dict search params
    consistency_level: ReadOnly[str | None]
    partition_names: ReadOnly[list[str] | None]  # mutable-ok: PyMilvus requires list partition names
    timeout: ReadOnly[float | None]


class _SyncMilvusClient(Protocol):
    def search(
        self,
        collection_name: str,
        data: list[list[float]],  # mutable-ok: PyMilvus requires nested list search data
        anns_field: str | None,
        limit: int,
        filter: str,
        offset: int | None,
        group_by_field: str | None,
        output_fields: list[str] | None,  # mutable-ok: PyMilvus requires list output fields
        search_params: dict[str, object] | None,  # mutable-ok: PyMilvus requires dict search params
        consistency_level: str | None,
        partition_names: list[str] | None,  # mutable-ok: PyMilvus requires list partition names
        timeout: float | None,
    ) -> object: ...

    def close(self) -> None: ...


class _AsyncMilvusClient(Protocol):
    async def search(
        self,
        collection_name: str,
        data: list[list[float]],  # mutable-ok: PyMilvus requires nested list search data
        anns_field: str | None,
        limit: int,
        filter: str,
        offset: int | None,
        group_by_field: str | None,
        output_fields: list[str] | None,  # mutable-ok: PyMilvus requires list output fields
        search_params: dict[str, object] | None,  # mutable-ok: PyMilvus requires dict search params
        consistency_level: str | None,
        partition_names: list[str] | None,  # mutable-ok: PyMilvus requires list partition names
        timeout: float | None,
    ) -> object: ...

    async def close(self) -> None: ...


class _MilvusErrorLike(Protocol):
    @property
    def code(self) -> int: ...


class _NeverRaised(Exception): ...


def _milvus_error_type() -> type[Exception]:
    try:
        from pymilvus import (  # pyright: ignore[reportMissingTypeStubs]  # pymilvus does not publish typing metadata
            MilvusException,
        )
    except ImportError:
        return _NeverRaised
    return MilvusException


def _is_connect_failure(cause: Exception) -> bool:
    error: Final = typing.cast(  # noqa: TID251  # cast-ok: pymilvus lacks typing metadata; MilvusException always carries code
        _MilvusErrorLike, cause
    )
    return error.code == _MILVUS_CONNECT_FAILURE_CODE


@contextmanager
def _milvus_connection_errors_mapped() -> Generator[None, None, None]:
    try:
        yield
    except _milvus_error_type() as e:
        if not _is_connect_failure(e):
            raise
        raise litellm.APIConnectionError(
            message=f"{_MILVUS_CONNECTION_HINT} {e}",
            model="milvus",
            llm_provider="milvus",
        ) from e


def _new_sync_client(uri: str, token: str, db_name: str, timeout: float | None) -> _SyncMilvusClient:
    try:
        from pymilvus import (  # pyright: ignore[reportMissingTypeStubs]  # pymilvus does not publish typing metadata
            MilvusClient,
        )
    except ImportError as e:
        raise litellm.BadRequestError(
            message=_PYMILVUS_INSTALL_HINT,
            model="milvus",
            llm_provider="milvus",
        ) from e
    with _milvus_connection_errors_mapped():
        return typing.cast(  # noqa: TID251  # cast-ok: pymilvus lacks typing metadata; protocol defines the used surface
            _SyncMilvusClient,
            MilvusClient(uri=uri, token=token, db_name=db_name, timeout=timeout, dedicated=True),
        )


def _new_async_client(uri: str, token: str, db_name: str, timeout: float | None) -> _AsyncMilvusClient:
    try:
        from pymilvus import (  # pyright: ignore[reportMissingTypeStubs]  # pymilvus does not publish typing metadata
            AsyncMilvusClient,
        )
    except ImportError as e:
        raise litellm.BadRequestError(
            message=_PYMILVUS_INSTALL_HINT,
            model="milvus",
            llm_provider="milvus",
        ) from e
    with _milvus_connection_errors_mapped():
        return typing.cast(  # noqa: TID251  # cast-ok: pymilvus lacks typing metadata; protocol defines the used surface
            _AsyncMilvusClient,
            AsyncMilvusClient(uri=uri, token=token, db_name=db_name, timeout=timeout, dedicated=True),
        )


class _MilvusSearchParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    api_base: str | None = None
    api_key: str | None = None
    litellm_embedding_model: str | None = None
    litellm_embedding_config: Mapping[str, object] | None = None
    milvus_db_name: str | None = None
    milvus_partition_names: tuple[str, ...] | None = None
    milvus_text_field: str | None = None

    @property
    def uri(self) -> str:
        uri: Final = self.api_base or get_secret_str("MILVUS_API_BASE")
        if not uri:
            raise litellm.BadRequestError(
                message="Milvus API base URL is required. Set MILVUS_API_BASE or pass api_base in litellm_params.",
                model="milvus",
                llm_provider="milvus",
            )
        return uri.rstrip("/")

    @property
    def token(self) -> str:
        return self.api_key or get_secret_str("MILVUS_API_KEY") or ""

    @property
    def db_name(self) -> str:
        return self.milvus_db_name or ""

    @property
    def text_field(self) -> str:
        return self.milvus_text_field or DEFAULT_TEXT_FIELD

    def require_embedding_model(self) -> str:
        if not self.litellm_embedding_model:
            raise litellm.BadRequestError(
                message=(
                    "litellm_embedding_model is required in litellm_params for Milvus. "
                    "Example: litellm_params['litellm_embedding_model'] = 'openai/text-embedding-3-small'"
                ),
                model="milvus",
                llm_provider="milvus",
            )
        return self.litellm_embedding_model


class _MilvusSearchOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    anns_field: str | None = Field(default=None, alias="annsField")
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=50)
    max_num_results: int | None = Field(default=None, ge=1, le=50)
    filters: Mapping[str, object] | None = None
    ranking_options: Mapping[str, object] | None = None
    rewrite_query: bool | None = None
    filter_expression: str = Field(default="", alias="filter")
    offset: int | None = None
    grouping_field: str | None = Field(default=None, alias="groupingField")
    output_fields: tuple[str, ...] | None = Field(default=None, alias="outputFields")
    search_params: Mapping[str, object] | None = Field(default=None, alias="searchParams")
    consistency_level: str | None = Field(default=None, alias="consistencyLevel")

    @property
    def result_limit(self) -> int:
        return self.max_num_results or self.limit

    def output_fields_with_text(self, text_field: str) -> list[str]:  # mutable-ok: PyMilvus requires a list
        output_fields: Final = self.output_fields or ()
        if "*" in output_fields or text_field in output_fields:
            return list(output_fields)  # mutable-ok: PyMilvus requires output_fields as a list
        return [*output_fields, text_field]  # mutable-ok: PyMilvus requires output_fields as a list


class _EmbeddingItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    embedding: tuple[float, ...]


class _EmbeddingPayload(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    data: tuple[_EmbeddingItem, ...]

    def vector(self) -> tuple[float, ...]:
        if not self.data:
            raise ValueError("The embedding response did not contain an embedding")
        return self.data[0].embedding


class MilvusGRPCVectorStoreConfig(BaseDirectVectorStoreConfig):
    def __init__(
        self,
        sync_client: _SyncMilvusClient | None = None,
        async_client: _AsyncMilvusClient | None = None,
    ) -> None:
        super().__init__()
        self.sync_client = sync_client
        self.async_client = async_client

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: BaseVectorStoreConfig requires dict parameters
        optional_params: dict[str, object],  # mutable-ok: BaseVectorStoreConfig requires dict parameters
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVectorStoreConfig requires a dict result
        mapped_params: Final = {  # mutable-ok: BaseVectorStoreConfig requires a dict result
            key: value for key, value in non_default_params.items() if key in MILVUS_OPTIONAL_PARAMS
        }
        return {**optional_params, **mapped_params}  # mutable-ok: BaseVectorStoreConfig requires a dict result

    @staticmethod
    def _search_options(
        optional_params: VectorStoreSearchOptionalRequestParams,
    ) -> _MilvusSearchOptions:
        for parameter in ("filters", "ranking_options", "rewrite_query"):
            if optional_params.get(parameter) is not None:
                raise litellm.BadRequestError(
                    message=f"Milvus gRPC search does not support the {parameter} parameter",
                    model="milvus",
                    llm_provider="milvus",
                )
        try:
            return _MilvusSearchOptions.model_validate(optional_params)
        except ValidationError as exc:
            raise litellm.BadRequestError(
                message=f"Invalid Milvus gRPC search options: {exc}",
                model="milvus",
                llm_provider="milvus",
            ) from exc

    @staticmethod
    def _query_text(query: str | Sequence[str]) -> str:
        query_text: Final = query if isinstance(query, str) else " ".join(query)
        if not query_text.strip():
            raise litellm.BadRequestError(
                message="query must not be empty",
                model="milvus",
                llm_provider="milvus",
            )
        return query_text

    @staticmethod
    def _timeouts(timeout: float | httpx.Timeout | None) -> tuple[float | None, float | None]:
        if isinstance(timeout, httpx.Timeout):
            return timeout.connect, timeout.read
        timeout_seconds: Final = float(timeout) if timeout is not None else None
        return timeout_seconds, timeout_seconds

    @staticmethod
    def _is_hit(value: object) -> typing.TypeGuard[Mapping[str, object]]:  # noqa: TID251  # guard-ok: validates Mapping and string keys
        if not isinstance(value, Mapping):
            return False
        try:
            _STRING_KEYS_ADAPTER.validate_python(
                tuple(value.keys())  # pyright: ignore[reportUnknownArgumentType]  # runtime Mapping keys are untyped
            )
        except ValidationError:
            return False
        return True

    @staticmethod
    def _to_result(raw_hit: object, text_field: str) -> VectorStoreSearchResult:
        if not MilvusGRPCVectorStoreConfig._is_hit(raw_hit):
            raise TypeError(f"Milvus returned an invalid search hit: {type(raw_hit).__name__}")
        hit: Final = raw_hit
        entity_value: Final = hit.get("entity")
        entity: Final = _MILVUS_ENTITY_ADAPTER.validate_python(entity_value or _EMPTY_EMBEDDING_CONFIG)
        text_value: Final = entity.get(text_field, hit.get(text_field, ""))
        attributes: Final[dict[str, object]] = {  # mutable-ok: VectorStoreSearchResult requires dict attributes
            **{  # mutable-ok: VectorStoreSearchResult requires dict attributes
                key: value for key, value in entity.items() if key != text_field
            },
            **{  # mutable-ok: VectorStoreSearchResult requires dict attributes
                key: value
                for key, value in hit.items()
                if key not in frozenset(("id", "distance", "entity", text_field))
            },
        }
        score_value: Final = hit.get("distance", 0.0)
        score: Final = float(score_value) if isinstance(score_value, int | float) else 0.0
        content: Final[list[VectorStoreResultContent]] = [  # mutable-ok: VectorStoreSearchResult requires list content
            VectorStoreResultContent(text="" if text_value is None else str(text_value), type="text")
        ]
        return VectorStoreSearchResult(
            score=score,
            content=content,
            file_id=None,
            filename=None,
            attributes=attributes,
        )

    @staticmethod
    def _is_result_sequence(value: object) -> typing.TypeGuard[Sequence[object]]:  # noqa: TID251  # guard-ok: validates non-text Sequence
        return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)

    @classmethod
    def _result_sequence(cls, value: object) -> Sequence[object]:
        if cls._is_result_sequence(value):
            return value
        raise TypeError(f"Milvus returned an invalid search result: {type(value).__name__}")

    @classmethod
    def _to_response(cls, raw_result: object, query_text: str, text_field: str) -> VectorStoreSearchResponse:
        result_sets: Final = cls._result_sequence(raw_result)
        hits: Final = cls._result_sequence(result_sets[0]) if result_sets else ()
        data: Final[list[VectorStoreSearchResult]] = [  # mutable-ok: VectorStoreSearchResponse requires list data
            cls._to_result(hit, text_field) for hit in hits
        ]
        return VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query=query_text,
            data=data,
        )

    @staticmethod
    def _search_arguments(
        vector_store_id: str,
        query_vector: Sequence[float],
        options: _MilvusSearchOptions,
        params: _MilvusSearchParams,
        timeout: float | None,
    ) -> _MilvusSearchArguments:
        return _MilvusSearchArguments(
            collection_name=vector_store_id,
            data=[list(query_vector)],  # mutable-ok: PyMilvus requires nested list search data
            anns_field=options.anns_field,
            limit=options.result_limit,
            filter=options.filter_expression,
            offset=options.offset,
            group_by_field=options.grouping_field,
            output_fields=options.output_fields_with_text(params.text_field),
            search_params=dict(options.search_params)  # mutable-ok: PyMilvus requires dict search params
            if options.search_params is not None
            else None,
            consistency_level=options.consistency_level,
            partition_names=list(params.milvus_partition_names)  # mutable-ok: PyMilvus requires list partition names
            if params.milvus_partition_names is not None
            else None,
            timeout=timeout,
        )

    def execute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        params: Final = _MilvusSearchParams.model_validate(litellm_params)
        options: Final = self._search_options(vector_store_search_optional_params)
        query_text: Final = self._query_text(query)
        executor: Final = embedding_executor or LiteLLMVectorStoreEmbeddingExecutor()
        embedding_response: Final = executor.embed(
            params.require_embedding_model(),
            query_text,
            params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG,
        )
        query_vector: Final = _EmbeddingPayload.model_validate(embedding_response).vector()
        connection_timeout, search_timeout = self._timeouts(timeout)
        arguments: Final = self._search_arguments(vector_store_id, query_vector, options, params, search_timeout)
        client: Final = (
            self.sync_client
            if self.sync_client is not None
            else _new_sync_client(params.uri, params.token, params.db_name, connection_timeout)
        )
        try:
            with _milvus_connection_errors_mapped():
                result: Final = client.search(**arguments)
                return self._to_response(result, query_text, params.text_field)
        finally:
            if self.sync_client is None:
                client.close()

    async def aexecute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        params: Final = _MilvusSearchParams.model_validate(litellm_params)
        options: Final = self._search_options(vector_store_search_optional_params)
        query_text: Final = self._query_text(query)
        executor: Final = embedding_executor or LiteLLMVectorStoreEmbeddingExecutor()
        embedding_response: Final = await executor.aembed(
            params.require_embedding_model(),
            query_text,
            params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG,
        )
        query_vector: Final = _EmbeddingPayload.model_validate(embedding_response).vector()
        connection_timeout, search_timeout = self._timeouts(timeout)
        arguments: Final = self._search_arguments(vector_store_id, query_vector, options, params, search_timeout)
        client: Final = (
            self.async_client
            if self.async_client is not None
            else _new_async_client(params.uri, params.token, params.db_name, connection_timeout)
        )
        try:
            with _milvus_connection_errors_mapped():
                result: Final = await client.search(**arguments)
                return self._to_response(result, query_text, params.text_field)
        finally:
            if self.async_client is None:
                await client.close()
