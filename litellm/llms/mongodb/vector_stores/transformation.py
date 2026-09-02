"""MongoDB Atlas vector store provider.

Atlas Vector Search has no HTTP query API (the Data API and HTTPS Endpoints are
end-of-life), so this config extends BaseDirectVectorStoreConfig and runs the
``$vectorSearch`` aggregation itself through pymongo instead of shaping an httpx
request.

``vector_store_id`` is the Atlas Search index name, matching the Valkey provider
where the id names the index; the database and collection it covers come from
litellm_params.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NoReturn

import httpx
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.llms.base_llm.vector_store.transformation import BaseDirectVectorStoreConfig
from litellm.llms.mongodb.common_utils import (
    DEFAULT_CONNECT_TIMEOUT_MS,
    DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    DEFAULT_SOCKET_TIMEOUT_MS,
    MongoClientKey,
    config_error,
    get_async_client,
    get_sync_client,
    index_not_ready_error,
    missing_index_error,
    translate_mongo_error,
)
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_EMBEDDING_FIELD_NAME: Final = "embedding"
DEFAULT_TEXT_FIELD_NAME: Final = "text"
SCORE_FIELD_NAME: Final = "score"

DEFAULT_MAX_NUM_RESULTS: Final = 10
MIN_MAX_NUM_RESULTS: Final = 1
MAX_MAX_NUM_RESULTS: Final = 50

NUM_CANDIDATES_MULTIPLIER: Final = 10
MIN_NUM_CANDIDATES: Final = 100
MAX_NUM_CANDIDATES: Final = 10_000

MAX_QUERY_CHARACTERS: Final = 32_000

_EMPTY_EMBEDDING_CONFIG: Final = MappingProxyType({})

_SEARCH_ONLY_MESSAGE: Final = (
    "MongoDB vector store is search-only. Create the collection and its Atlas Vector Search "
    "index in MongoDB directly, then register it here by index name."
)


class _MongoDBSearchParams(BaseModel):
    """Typed view over the vector store's litellm_params; unrelated keys are ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_embedding_model: str | None = None
    litellm_embedding_config: Mapping[str, object] | None = None
    mongodb_connection_string: str | None = None
    mongodb_database: str | None = None
    mongodb_collection: str | None = None
    mongodb_text_field: str | None = None
    mongodb_embedding_field: str | None = None
    mongodb_num_candidates: int | None = None

    @property
    def text_field(self) -> str:
        return self.mongodb_text_field or DEFAULT_TEXT_FIELD_NAME

    @property
    def embedding_field(self) -> str:
        return self.mongodb_embedding_field or DEFAULT_EMBEDDING_FIELD_NAME

    def require_embedding_model(self) -> str:
        if not self.litellm_embedding_model:
            raise config_error(
                "litellm_embedding_model is required in litellm_params for the MongoDB vector store. "
                "It must be the same model that produced the vectors stored in "
                f"'{self.mongodb_collection or '<collection>'}.{self.embedding_field}', or search results "
                "will be meaningless. Example: litellm_embedding_model: openai/text-embedding-3-small"
            )
        return self.litellm_embedding_model

    def require_connection_string(self) -> str:
        if not self.mongodb_connection_string:
            raise config_error(
                "mongodb_connection_string is required in litellm_params for the MongoDB vector store. "
                "Example: mongodb+srv://<user>:<password>@<cluster>.mongodb.net"
            )
        scheme: Final = self.mongodb_connection_string.split("://", 1)[0].lower()
        if scheme not in ("mongodb", "mongodb+srv"):
            raise config_error(
                "mongodb_connection_string must start with 'mongodb://' or 'mongodb+srv://', "
                f"got '{self.mongodb_connection_string.split('://', 1)[0]}://'"
            )
        return self.mongodb_connection_string

    def require_database(self) -> str:
        if not self.mongodb_database:
            raise config_error(
                "mongodb_database is required in litellm_params for the MongoDB vector store. "
                "Example: mongodb_database: sample_mflix"
            )
        return self.mongodb_database

    def require_collection(self) -> str:
        if not self.mongodb_collection:
            raise config_error(
                "mongodb_collection is required in litellm_params for the MongoDB vector store. "
                "Example: mongodb_collection: embedded_movies"
            )
        return self.mongodb_collection


_MONGODB_PARAM_PREFIX: Final = "mongodb_"
_KNOWN_MONGODB_PARAMS: Final = frozenset(
    name for name in _MongoDBSearchParams.model_fields if name.startswith(_MONGODB_PARAM_PREFIX)
)


class MongoDBVectorStoreConfig(BaseDirectVectorStoreConfig):
    def __init__(
        self,
        embedding_fn: Callable[..., EmbeddingResponse] | None = None,
        aembedding_fn: Callable[..., Awaitable[EmbeddingResponse]] | None = None,
        sync_client_factory: Callable[[MongoClientKey], object] | None = None,
        async_client_factory: Callable[[MongoClientKey], object] | None = None,
    ) -> None:
        super().__init__()
        self.embedding_fn = embedding_fn if embedding_fn is not None else litellm.embedding
        self.aembedding_fn = aembedding_fn if aembedding_fn is not None else litellm.aembedding
        self.sync_client_factory = sync_client_factory if sync_client_factory is not None else get_sync_client
        self.async_client_factory = async_client_factory if async_client_factory is not None else get_async_client

    @staticmethod
    def _reject_unknown_params(litellm_params: Mapping[str, object]) -> None:
        """The params model ignores unrelated keys because litellm_params carries plenty of them,
        which would otherwise turn a mistyped mongodb_collection into 'mongodb_collection is
        required' pointing at a key the reader can see they have set."""
        unknown: Final = sorted(
            key
            for key in litellm_params
            if key.startswith(_MONGODB_PARAM_PREFIX) and key not in _KNOWN_MONGODB_PARAMS
        )
        if unknown:
            raise config_error(
                f"Unrecognised MongoDB vector store parameter(s): {', '.join(unknown)}. "
                f"Supported: {', '.join(sorted(_KNOWN_MONGODB_PARAMS))}."
            )

    @staticmethod
    def _query_text(query: str | Sequence[str]) -> str:
        text: Final = query if isinstance(query, str) else " ".join(query)
        if not text.strip():
            raise config_error("query must not be empty")
        if len(text) > MAX_QUERY_CHARACTERS:
            raise config_error(f"query must be at most {MAX_QUERY_CHARACTERS} characters, got {len(text)}")
        return text

    @staticmethod
    def _limit(vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams) -> int:
        requested: Final = vector_store_search_optional_params.get("max_num_results")
        if requested is None:
            return DEFAULT_MAX_NUM_RESULTS
        if not MIN_MAX_NUM_RESULTS <= requested <= MAX_MAX_NUM_RESULTS:
            raise config_error(
                f"max_num_results must be between {MIN_MAX_NUM_RESULTS} and {MAX_MAX_NUM_RESULTS}, got {requested}"
            )
        return requested

    @staticmethod
    def _num_candidates(limit: int, configured: int | None) -> int:
        if configured is not None:
            if not limit <= configured <= MAX_NUM_CANDIDATES:
                raise config_error(
                    f"mongodb_num_candidates must be between max_num_results ({limit}) and "
                    f"{MAX_NUM_CANDIDATES}, got {configured}"
                )
            return configured
        return min(max(limit * NUM_CANDIDATES_MULTIPLIER, MIN_NUM_CANDIDATES), MAX_NUM_CANDIDATES)

    @staticmethod
    def _client_key(params: _MongoDBSearchParams, timeout: float | httpx.Timeout | None) -> MongoClientKey:
        if isinstance(timeout, httpx.Timeout):
            connect_ms: Final = int((timeout.connect or DEFAULT_CONNECT_TIMEOUT_MS / 1000) * 1000)
            socket_ms: Final = int((timeout.read or DEFAULT_SOCKET_TIMEOUT_MS / 1000) * 1000)
        elif timeout is not None:
            connect_ms = min(int(float(timeout) * 1000), DEFAULT_CONNECT_TIMEOUT_MS)
            socket_ms = int(float(timeout) * 1000)
        else:
            connect_ms = DEFAULT_CONNECT_TIMEOUT_MS
            socket_ms = DEFAULT_SOCKET_TIMEOUT_MS
        return MongoClientKey(
            connection_string=params.require_connection_string(),
            connect_timeout_ms=connect_ms,
            socket_timeout_ms=socket_ms,
            server_selection_timeout_ms=min(connect_ms, DEFAULT_SERVER_SELECTION_TIMEOUT_MS),
        )

    @classmethod
    def _pipeline(
        cls,
        vector_store_id: str,
        query_vector: Sequence[float],
        params: _MongoDBSearchParams,
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
    ) -> list[dict[str, object]]:
        if vector_store_search_optional_params.get("filters") is not None:
            raise config_error(
                "MongoDB vector store does not support the filters parameter yet. "
                "Restrict the collection or the Atlas Vector Search index definition instead."
            )
        limit: Final = cls._limit(vector_store_search_optional_params)
        return [  # mutable-ok: pymongo's aggregate contract is a list of stage dicts
            {
                "$vectorSearch": {
                    "index": vector_store_id,
                    "path": params.embedding_field,
                    "queryVector": list(query_vector),
                    "numCandidates": cls._num_candidates(limit, params.mongodb_num_candidates),
                    "limit": limit,
                }
            },
            {"$project": {params.text_field: 1, SCORE_FIELD_NAME: {"$meta": "vectorSearchScore"}}},
        ]

    @staticmethod
    def _field_value(document: Mapping[str, object], dotted_path: str) -> str | None:
        """None means the path is absent from the document, which is what separates a
        mistyped mongodb_text_field from a document whose text is genuinely empty."""
        current: object = document
        for segment in dotted_path.split("."):
            if not isinstance(current, Mapping) or segment not in current:
                return None
            current = current[segment]
        return None if current is None else str(current)

    @classmethod
    def _to_result(cls, document: Mapping[str, object], text_field: str) -> VectorStoreSearchResult:
        document_id: Final = document.get("_id")
        identifier: Final = None if document_id is None else str(document_id)
        content: Final = [  # mutable-ok: VectorStoreSearchResult declares a list of content parts
            VectorStoreResultContent(text=cls._field_value(document, text_field) or "", type="text")
        ]
        raw_score: Final = document.get(SCORE_FIELD_NAME)
        return VectorStoreSearchResult(
            score=float(raw_score) if isinstance(raw_score, (int, float)) else None,
            content=content,
            file_id=identifier,
            filename=identifier,
        )

    @classmethod
    def _raise_for_missing_text_field(
        cls, documents: Sequence[Mapping[str, object]], text_field: str, database: str, collection: str
    ) -> None:
        """Atlas happily matches vectors in documents that carry no text at all, so a mistyped
        mongodb_text_field returns well-scored results whose content is empty and feeds an empty
        context to the model. Every matched document lacking the field is the misconfiguration."""
        if documents and all(cls._field_value(document, text_field) is None for document in documents):
            raise config_error(
                f"None of the {len(documents)} matched documents in '{database}.{collection}' has a "
                f"'{text_field}' field, so every result would carry empty text. Set mongodb_text_field "
                "to the field holding the readable text; it accepts a dotted path such as metadata.body."
            )

    @classmethod
    def _to_response(
        cls, documents: Sequence[Mapping[str, object]], query_text: str, text_field: str
    ) -> VectorStoreSearchResponse:
        return VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query=query_text,
            data=[cls._to_result(document, text_field) for document in documents],
        )

    @staticmethod
    def _raise_for_unusable_index(
        catalogue: Sequence[Mapping[str, object]], index_name: str, database: str, collection: str
    ) -> None:
        """An empty result set is ambiguous: Atlas returns zero documents both for a query that
        genuinely matched nothing and for a missing database, collection or index. Only the second
        is a misconfiguration, so the index catalogue decides which one happened."""
        if not catalogue:
            raise missing_index_error(index_name, database, collection)
        entry: Final = catalogue[0]
        if not entry.get("queryable"):
            raise index_not_ready_error(index_name, database, collection, str(entry.get("status") or "unknown"))

    @staticmethod
    def _embedding_vector(embedding_response: EmbeddingResponse) -> Sequence[float]:
        data: Final = embedding_response.data
        if not data:
            raise config_error(
                "The embedding model returned no embedding for the search query, so there is nothing "
                "to search MongoDB with. Check the embedding deployment named by litellm_embedding_model."
            )
        return data[0]["embedding"]

    def execute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        self._reject_unknown_params(litellm_params)
        params: Final = _MongoDBSearchParams.model_validate(litellm_params)
        query_text: Final = self._query_text(query)
        key: Final = self._client_key(params, timeout)
        database: Final = params.require_database()
        collection: Final = params.require_collection()

        embedding_response: Final = self.embedding_fn(
            model=params.require_embedding_model(),
            input=[query_text],  # mutable-ok: litellm.embedding's input contract is a list
            **(params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG),
        )
        pipeline: Final = self._pipeline(
            vector_store_id, self._embedding_vector(embedding_response), params, vector_store_search_optional_params
        )

        client: Final = self.sync_client_factory(key)
        target: Final = client[database][collection]  # pyright: ignore[reportIndexIssue]  # factory is typed as returning object so injected doubles are accepted
        try:
            documents: Final = list(target.aggregate(pipeline))
        except Exception as e:
            raise translate_mongo_error(
                e, index_name=vector_store_id, database=database, collection=collection
            ) from e
        if not documents:
            try:
                catalogue: Final = list(target.list_search_indexes(vector_store_id))
            except Exception as e:
                raise translate_mongo_error(
                    e, index_name=vector_store_id, database=database, collection=collection
                ) from e
            self._raise_for_unusable_index(catalogue, vector_store_id, database, collection)
        self._raise_for_missing_text_field(documents, params.text_field, database, collection)
        return self._to_response(documents, query_text, params.text_field)

    async def aexecute_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        timeout: float | httpx.Timeout | None = None,
    ) -> VectorStoreSearchResponse:
        self._reject_unknown_params(litellm_params)
        params: Final = _MongoDBSearchParams.model_validate(litellm_params)
        query_text: Final = self._query_text(query)
        key: Final = self._client_key(params, timeout)
        database: Final = params.require_database()
        collection: Final = params.require_collection()

        embedding_response: Final = await self.aembedding_fn(
            model=params.require_embedding_model(),
            input=[query_text],  # mutable-ok: litellm.embedding's input contract is a list
            **(params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG),
        )
        pipeline: Final = self._pipeline(
            vector_store_id, self._embedding_vector(embedding_response), params, vector_store_search_optional_params
        )

        client: Final = self.async_client_factory(key)
        target: Final = client[database][collection]  # pyright: ignore[reportIndexIssue]  # factory is typed as returning object so injected doubles are accepted
        try:
            cursor: Final = await target.aggregate(pipeline)
            documents: Final = [document async for document in cursor]
        except Exception as e:
            raise translate_mongo_error(
                e, index_name=vector_store_id, database=database, collection=collection
            ) from e
        if not documents:
            try:
                index_cursor: Final = await target.list_search_indexes(vector_store_id)
                catalogue: Final = [entry async for entry in index_cursor]
            except Exception as e:
                raise translate_mongo_error(
                    e, index_name=vector_store_id, database=database, collection=collection
                ) from e
            self._raise_for_unusable_index(catalogue, vector_store_id, database, collection)
        self._raise_for_missing_text_field(documents, params.text_field, database, collection)
        return self._to_response(documents, query_text, params.text_field)

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> NoReturn:
        raise NotImplementedError(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_response(self, response: httpx.Response) -> NoReturn:
        raise NotImplementedError(_SEARCH_ONLY_MESSAGE)
