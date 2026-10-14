from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Final, NoReturn
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    VectorStoreEmbeddingExecutor,
)
from litellm.llms.qdrant.vector_stores.filters import configuration_error, translate_filter
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VECTOR_STORE_OPENAI_PARAMS,
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SEARCH_ONLY_MESSAGE: Final = (
    "Qdrant vector stores support search only. Create and populate a collection in Qdrant, "
    "then register its collection name as vector_store_id in LiteLLM."
)
_OBJECT: Final = TypeAdapter(dict[str, object])
_STRING: Final = TypeAdapter(str)


class _SearchParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    qdrant_text_field: str | None = None
    qdrant_vector_name: str | None = None

    @property
    def text_field(self) -> str:
        return self.qdrant_text_field or "text"


class _Point(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)
    id: int | str
    score: float
    payload: Mapping[str, object] | None = None


class _QueryResult(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    points: Sequence[_Point]


class _QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    result: _QueryResult


class _RankingOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    ranker: str = "auto"
    score_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class _SearchOptions(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    filters: Mapping[str, object] | None = None
    max_num_results: int = Field(default=10, ge=1, le=50)
    ranking_options: _RankingOptions = Field(default_factory=_RankingOptions)
    rewrite_query: bool = False


class _ExtraOptions(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    filter: Mapping[str, object] | None = None
    params: Mapping[str, object] | None = None
    score_threshold: float | None = Field(default=None, allow_inf_nan=False)


def _payload_child(value: object, key: str) -> object:
    return _OBJECT.validate_python(value).get(key) if isinstance(value, dict) else None


def _payload_text(payload: Mapping[str, object], path: str) -> str:
    value: Final = payload[path] if path in payload else reduce(_payload_child, path.split("."), payload)
    if not isinstance(value, str):
        raise configuration_error(f"Qdrant payload field '{path}' must contain text")
    return value


def _search_result(point: _Point, text_field: str) -> VectorStoreSearchResult:
    payload: Final = point.payload or {}
    file_id: Final = payload.get("file_id")
    filename: Final = payload.get("filename")
    return VectorStoreSearchResult(
        score=point.score,
        content=[VectorStoreResultContent(type="text", text=_payload_text(payload, text_field))],
        file_id=str(file_id) if isinstance(file_id, (str, int)) else str(point.id),
        filename=filename if isinstance(filename, str) else None,
        attributes={key: value for key, value in payload.items() if key != text_field},
    )


class QdrantVectorStoreConfig(BaseQueryEmbeddingVectorStoreConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[VECTOR_STORE_OPENAI_PARAMS]:  # mutable-ok: the shared provider interface requires a list
        return ["filters", "max_num_results", "ranking_options"]

    def get_auth_credentials(self, litellm_params: Mapping[str, object]) -> BaseVectorStoreAuthCredentials:
        return BaseVectorStoreAuthCredentials(headers=dict(self._auth_headers(litellm_params.get("api_key"))))

    @staticmethod
    def _auth_headers(configured: object) -> Mapping[str, str]:
        api_key: Final = configured or get_secret_str("QDRANT_API_KEY")
        return {"api-key": _STRING.validate_python(api_key, strict=True)} if api_key else {}

    def validate_environment(
        self, headers: Mapping[str, object], litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, object]:  # mutable-ok: the shared HTTP handler mutates the returned headers
        api_key: Final = litellm_params.api_key if litellm_params is not None else None
        return {**headers, **self._auth_headers(api_key), "Content-Type": "application/json"}

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return VectorStoreIndexEndpoints(read=[], write=[])

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        resolved: Final = api_base or get_secret_str("QDRANT_API_BASE")
        if not resolved:
            raise configuration_error("Set api_base or QDRANT_API_BASE to your Qdrant HTTP endpoint")
        parsed: Final = urlsplit(resolved)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
            raise configuration_error("Qdrant api_base must be an HTTP or HTTPS URL without query or fragment")
        if parsed.username or parsed.password:
            raise configuration_error("Configure Qdrant authentication using api_key")
        return resolved.rstrip("/")

    @staticmethod
    def _request_options(
        options: VectorStoreSearchOptionalRequestParams, extra_body: Mapping[str, object] | None
    ) -> Mapping[str, object]:
        try:
            parsed: Final = _SearchOptions.model_validate(
                {key: value for key, value in options.items() if value is not None}
            )
            extra: Final = _ExtraOptions.model_validate(extra_body or {}).model_dump(exclude_none=True)
            filters: Final = translate_filter(parsed.filters) if parsed.filters is not None else None
        except ValidationError as error:
            raise configuration_error(str(error)) from error
        if parsed.rewrite_query:
            raise configuration_error("Qdrant does not support rewrite_query")
        if filters is not None and "filter" in extra:
            raise configuration_error("Use filters or extra_body.filter, not both")
        ranking: Final = parsed.ranking_options
        if ranking.ranker != "auto":
            raise configuration_error("Qdrant does not support custom rankers")
        if ranking.score_threshold is not None and "score_threshold" in extra:
            raise configuration_error("Use ranking_options.score_threshold or extra_body.score_threshold, not both")
        return {
            **extra,
            "limit": parsed.max_num_results,
            "with_payload": True,
            "with_vector": False,
            **({"filter": filters} if filters is not None else {}),
            **({"score_threshold": ranking.score_threshold} if ranking.score_threshold is not None else {}),
        }

    @staticmethod
    def _request_url(vector_store_id: str, api_base: str) -> str:
        if not vector_store_id or vector_store_id in (".", ".."):
            raise configuration_error("vector_store_id must be a Qdrant collection name")
        return f"{api_base}/collections/{quote(vector_store_id, safe='')}/points/query"

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the shared HTTP transformation interface requires a dict body
        url: Final = self._request_url(vector_store_id, api_base)
        options: Final = self._request_options(vector_store_search_optional_params, extra_body)
        params: Final = _SearchParams.model_validate(litellm_params)
        text: Final = self.query_text(query)
        if not text.strip():
            raise configuration_error("query must not be empty")
        vector: Final = self.embed_query(text, litellm_params, embedding_executor)
        litellm_logging_obj.model_call_details["qdrant_search_query"] = text  # rebind-ok: response state
        return url, {
            **options,
            "query": vector,
            **({"using": params.qdrant_vector_name} if params.qdrant_vector_name else {}),
        }

    async def atransform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | Sequence[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
        extra_body: Mapping[str, object] | None = None,
        embedding_executor: VectorStoreEmbeddingExecutor | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: the shared HTTP transformation interface requires a dict body
        url: Final = self._request_url(vector_store_id, api_base)
        options: Final = self._request_options(vector_store_search_optional_params, extra_body)
        params: Final = _SearchParams.model_validate(litellm_params)
        text: Final = self.query_text(query)
        if not text.strip():
            raise configuration_error("query must not be empty")
        vector: Final = await self.aembed_query(text, litellm_params, embedding_executor)
        litellm_logging_obj.model_call_details["qdrant_search_query"] = text  # rebind-ok: response state
        return url, {
            **options,
            "query": vector,
            **({"using": params.qdrant_vector_name} if params.qdrant_vector_name else {}),
        }

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        parsed: Final = _QueryResponse.model_validate_json(response.content)
        params: Final = _SearchParams.model_validate(litellm_logging_obj.model_call_details.get("litellm_params", {}))
        return VectorStoreSearchResponse(
            object="vector_store.search_results.page",
            search_query=_STRING.validate_python(litellm_logging_obj.model_call_details.get("qdrant_search_query", "")),
            data=[_search_result(point, params.text_field) for point in parsed.result.points],
        )

    def validate_create_vector_store(self) -> NoReturn:
        raise configuration_error(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_request(
        self, vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams, api_base: str
    ) -> NoReturn:
        raise configuration_error(_SEARCH_ONLY_MESSAGE)

    def transform_create_vector_store_response(self, response: httpx.Response) -> NoReturn:
        raise configuration_error(_SEARCH_ONLY_MESSAGE)
