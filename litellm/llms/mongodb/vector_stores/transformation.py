from collections.abc import Mapping, Sequence
from ipaddress import ip_address
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.exceptions import AuthenticationError, BadRequestError, ServiceUnavailableError, Timeout
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    LiteLLMVectorStoreEmbeddingExecutor,
    VectorStoreEmbeddingExecutor,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import EmbeddingResponse
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreIndexEndpoints,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_EMBEDDING_FIELD_NAME: Final = "embedding"
DEFAULT_TEXT_FIELD_NAME: Final = "text"
DEFAULT_MAX_NUM_RESULTS: Final = 10
MIN_MAX_NUM_RESULTS: Final = 1
MAX_MAX_NUM_RESULTS: Final = 50
NUM_CANDIDATES_MULTIPLIER: Final = 10
MIN_NUM_CANDIDATES: Final = 100
MAX_NUM_CANDIDATES: Final = 10_000
MAX_QUERY_CHARACTERS: Final = 32_000
_EMPTY_EMBEDDING_CONFIG: Final = MappingProxyType({})
_SEARCH_ONLY_MESSAGE: Final = (
    "MongoDB vector store is search-only. Create the collection and its MongoDB Vector Search "
    "index in MongoDB directly, then register it here by index name."
)


def config_error(message: str) -> BadRequestError:
    return BadRequestError(message=message, model=None, llm_provider="mongodb")


class _Content(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    type: Literal["text"]
    text: str


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)
    score: float | None
    content: list[_Content]
    file_id: str | None
    filename: str | None


class _SearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    object: Literal["vector_store.search_results.page"]
    search_query: str
    data: list[_Result]


class _MongoDBSearchParams(BaseModel):
    """Typed view over the vector store's litellm_params; unrelated keys are ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_embedding_model: str | None = None
    litellm_embedding_config: Mapping[str, object] | None = None
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
_RESPONSE_ADAPTER: Final = TypeAdapter(VectorStoreSearchResponse)


class MongoDBVectorStoreConfig(BaseQueryEmbeddingVectorStoreConfig):
    def __init__(self, embedding_executor: VectorStoreEmbeddingExecutor | None = None) -> None:
        self.embedding_executor: Final = embedding_executor or LiteLLMVectorStoreEmbeddingExecutor()

    def get_auth_credentials(self, litellm_params: Mapping[str, object]) -> BaseVectorStoreAuthCredentials:
        return BaseVectorStoreAuthCredentials()

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return VectorStoreIndexEndpoints(read=[], write=[])  # mutable-ok: the TypedDict declares list fields

    @staticmethod
    def _reject_unknown_params(litellm_params: Mapping[str, object]) -> None:
        """Without this a mistyped mongodb_collection reads as 'mongodb_collection is required',
        naming a key the reader can see they have set."""
        if litellm_params.get("mongodb_connection_string") is not None:
            raise config_error(
                "MongoDB vector stores now use the BETA sidecar. Move mongodb_connection_string to "
                "MONGODB_CONNECTION_STRING in the sidecar, remove it from LiteLLM, and configure api_base and api_key."
            )
        unknown: Final = sorted(
            key for key in litellm_params if key.startswith(_MONGODB_PARAM_PREFIX) and key not in _KNOWN_MONGODB_PARAMS
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

    def validate_environment(
        self, headers: dict[str, object], litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, object]:
        if litellm_params is None:
            raise config_error("Configure api_base and api_key for the MongoDB BETA sidecar.")
        self._reject_unknown_params(dict(litellm_params))
        api_key: Final = litellm_params.api_key or get_secret_str("MONGODB_SIDECAR_API_KEY")
        if not api_key:
            raise config_error("MongoDB sidecar api_key is required. Set api_key or MONGODB_SIDECAR_API_KEY.")
        return {**headers, "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def get_complete_url(self, api_base: str | None, litellm_params: dict[str, object]) -> str:
        if not api_base:
            raise config_error("MongoDB sidecar api_base is required, for example http://127.0.0.1:8080.")
        try:
            parsed: Final = urlsplit(api_base)
            valid: Final = parsed.scheme in ("http", "https") and bool(parsed.hostname) and parsed.port != 0
        except ValueError:
            raise config_error("MongoDB sidecar api_base must be a valid HTTP or HTTPS URL.") from None
        if not valid or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise config_error(
                "MongoDB sidecar api_base must be an HTTP or HTTPS URL without credentials, query, or fragment."
            )
        if parsed.scheme == "http":
            try:
                loopback: Final = ip_address(parsed.hostname or "").is_loopback
            except ValueError:
                raise config_error(
                    "MongoDB sidecar requires HTTPS. HTTP is supported only for a loopback IP such as 127.0.0.1."
                ) from None
            if not loopback:
                raise config_error(
                    "MongoDB sidecar requires HTTPS. HTTP is supported only for a loopback IP such as 127.0.0.1."
                )
        return api_base.rstrip("/")

    @staticmethod
    def _timeout_ms(value: object) -> int:
        seconds: Final = value.read if isinstance(value, httpx.Timeout) else value
        if seconds is None:
            return 30_000
        if not isinstance(seconds, (int, float)) or not isfinite(seconds) or seconds <= 0:
            raise config_error("MongoDB search timeout must be a positive finite number.")
        try:
            return max(1, int(seconds * 1000))
        except (ValueError, OverflowError):
            raise config_error("MongoDB search timeout must be a positive finite number.") from None

    @classmethod
    def _params(
        cls,
        litellm_params: Mapping[str, object],
        optional_params: VectorStoreSearchOptionalRequestParams,
        extra_body: Mapping[str, object] | None,
    ) -> _MongoDBSearchParams:
        cls._reject_unknown_params(litellm_params)
        if extra_body:
            raise config_error("MongoDB vector store does not support extra_body overrides.")
        for unsupported in ("filters", "ranking_options", "rewrite_query"):
            if optional_params.get(unsupported) is not None:
                raise config_error(f"MongoDB vector store does not support the {unsupported} parameter.")
        try:
            params: Final = _MongoDBSearchParams.model_validate(litellm_params)
        except ValidationError:
            raise config_error(
                "Invalid MongoDB vector-store configuration. Check the database, collection, fields, and candidate count."
            ) from None
        params.require_database()
        params.require_collection()
        params.require_embedding_model()
        cls._num_candidates(cls._limit(optional_params), params.mongodb_num_candidates)
        cls._timeout_ms(litellm_params.get("timeout"))
        return params

    @classmethod
    def _request(
        cls,
        vector_store_id: str,
        query_text: str,
        params: _MongoDBSearchParams,
        optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        embedding_response: EmbeddingResponse,
        timeout: object,
    ) -> tuple[str, dict[str, object]]:
        if not embedding_response.data:
            raise config_error(
                "The embedding model returned no embedding for the search query. Check litellm_embedding_model."
            )
        vector: Final = embedding_response.data[0]["embedding"]
        if not vector or any(not isinstance(value, (float, int)) or not isfinite(value) for value in vector):
            raise config_error("The embedding model must return a non-empty, finite query vector.")
        limit: Final = cls._limit(optional_params)
        return f"{api_base}/v1/vector_stores/{quote(vector_store_id, safe='')}/search", {
            "query": query_text,
            "query_vector": tuple(vector),
            "mongodb_database": params.require_database(),
            "mongodb_collection": params.require_collection(),
            "mongodb_embedding_field": params.embedding_field,
            "mongodb_text_field": params.text_field,
            "mongodb_num_candidates": cls._num_candidates(limit, params.mongodb_num_candidates),
            "max_num_results": limit,
            "timeout_ms": cls._timeout_ms(timeout),
        }

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:
        params: Final = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        query_text: Final = self._query_text(query)
        response: Final = (embedding_executor or self.embedding_executor).embed(
            params.require_embedding_model(), query_text, params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG
        )
        return self._request(
            vector_store_id,
            query_text,
            params,
            vector_store_search_optional_params,
            api_base,
            response,
            litellm_params.get("timeout"),
        )

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: "LiteLLMLoggingObj",
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:
        params: Final = self._params(litellm_params, vector_store_search_optional_params, extra_body)
        query_text: Final = self._query_text(query)
        response: Final = await (embedding_executor or self.embedding_executor).aembed(
            params.require_embedding_model(), query_text, params.litellm_embedding_config or _EMPTY_EMBEDDING_CONFIG
        )
        return self._request(
            vector_store_id,
            query_text,
            params,
            vector_store_search_optional_params,
            api_base,
            response,
            litellm_params.get("timeout"),
        )

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: "LiteLLMLoggingObj"
    ) -> VectorStoreSearchResponse:
        try:
            validated: Final = _SearchResponse.model_validate_json(response.content)
            return _RESPONSE_ADAPTER.validate_python(validated.model_dump())
        except ValidationError:
            raise ServiceUnavailableError(
                message="MongoDB sidecar returned an invalid search response. Check the sidecar version and deployment.",
                model=None,
                llm_provider="mongodb",
            ) from None

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers
    ) -> BaseLLMException:
        if status_code == 400:
            raise config_error(error_message)
        if status_code == 401:
            raise AuthenticationError(message="MongoDB sidecar rejected api_key.", model=None, llm_provider="mongodb")
        if status_code == 408:
            raise Timeout(message=error_message, model=None, llm_provider="mongodb")
        raise ServiceUnavailableError(
            message="MongoDB sidecar is unavailable. Check its address, health, and logs.",
            model=None,
            llm_provider="mongodb",
        )

    def validate_create_vector_store(self) -> NoReturn:
        raise config_error(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_request(
        self, vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams, api_base: str
    ) -> NoReturn:
        raise config_error(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_response(self, response: httpx.Response) -> NoReturn:
        raise config_error(_SEARCH_ONLY_MESSAGE)
