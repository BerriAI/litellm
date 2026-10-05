from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol

import httpx
from typing_extensions import ReadOnly, TypedDict

from litellm import get_model_info
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
    VertexSearchDataStoreExtraBody,
    VertexSearchEngineExtraBody,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# Fields that select which data store / serving config to search. These are
# always determined by the request URL path (vector_store_id / vertex_engine_id),
# so allowing them per request could silently redirect the search to a different
# target. Rejected in both data-store and engine/app modes.
VERTEX_SEARCH_TARGET_SELECTING_FIELDS: Final = frozenset(
    {
        "branch",
        "servingConfig",
        "entity",
    }
)

# Allowlists of native Discovery Engine SearchRequest fields callers may forward
# via extra_body, derived from the TypedDicts so the type is the source of truth.
# Engine/app mode is a superset (adds dataStoreSpecs, numResultsPerDataStore),
# since an app fans out across multiple member data stores.
VERTEX_SEARCH_DATASTORE_EXTRA_BODY_FIELDS: Final = frozenset(VertexSearchDataStoreExtraBody.__annotations__)

VERTEX_SEARCH_ENGINE_EXTRA_BODY_FIELDS: Final = frozenset(VertexSearchEngineExtraBody.__annotations__)


class VertexSearchSnippet(TypedDict, total=False):
    snippet: ReadOnly[str]
    htmlSnippet: ReadOnly[str]


class VertexSearchExtractiveContent(TypedDict, total=False):
    """One ``extractive_answers`` or ``extractive_segments`` entry (opt-in via ``extractiveContentSpec``)."""

    content: ReadOnly[str]
    pageNumber: ReadOnly[str]


class VertexSearchDerivedStructData(TypedDict, total=False):
    """The ``derivedStructData`` blob Discovery Engine attaches to each document hit."""

    title: ReadOnly[str]
    link: ReadOnly[str]
    displayLink: ReadOnly[str]
    formattedUrl: ReadOnly[str]
    snippets: ReadOnly[list[VertexSearchSnippet]]
    extractive_answers: ReadOnly[list[VertexSearchExtractiveContent]]
    extractive_segments: ReadOnly[list[VertexSearchExtractiveContent]]


class VertexSearchDocument(TypedDict, total=False):
    id: ReadOnly[str]
    structData: ReadOnly[Mapping[str, object]]
    derivedStructData: ReadOnly[VertexSearchDerivedStructData]


class VertexSearchChunkDocumentMetadata(TypedDict, total=False):
    uri: ReadOnly[str]
    title: ReadOnly[str]
    structData: ReadOnly[Mapping[str, object]]


class VertexSearchChunkPageSpan(TypedDict, total=False):
    pageStart: ReadOnly[int]
    pageEnd: ReadOnly[int]


class VertexSearchChunk(TypedDict, total=False):
    """A hit when ``searchResultMode`` is ``CHUNKS``; such hits carry no ``document`` and no top-level ``id``."""

    id: ReadOnly[str]
    name: ReadOnly[str]
    content: ReadOnly[str]
    documentMetadata: ReadOnly[VertexSearchChunkDocumentMetadata]
    pageSpan: ReadOnly[VertexSearchChunkPageSpan]
    relevanceScore: ReadOnly[float]


class VertexSearchHit(TypedDict, total=False):
    id: ReadOnly[str]
    document: ReadOnly[VertexSearchDocument]
    chunk: ReadOnly[VertexSearchChunk]


class VertexSearchApiResponse(TypedDict, total=False):
    """Body of a Discovery Engine ``:search`` response."""

    results: ReadOnly[list[VertexSearchHit]]


class _SearchQueryView(TypedDict):
    """Holds the logged search query so the model call detail reads back as ``str``."""

    query: ReadOnly[str]


class _VertexSearchApiSource(Protocol):
    """An HTTP response whose JSON body is a Discovery Engine ``:search`` result."""

    def json(self) -> VertexSearchApiResponse: ...


def _vertex_search_payload(response: _VertexSearchApiSource) -> VertexSearchApiResponse:
    return response.json()


_UNKNOWN_DOCUMENT: Final = "Unknown Document"
_EMPTY_DOCUMENT: Final[VertexSearchDocument] = {}
_EMPTY_DERIVED_STRUCT_DATA: Final[VertexSearchDerivedStructData] = {}
_EMPTY_CHUNK_DOCUMENT_METADATA: Final[VertexSearchChunkDocumentMetadata] = {}


def _joined_content(entries: Sequence[VertexSearchExtractiveContent]) -> str:
    return "\n\n".join(content for entry in entries if (content := entry.get("content")))


def _snippet_text(snippets: Sequence[VertexSearchSnippet]) -> str:
    return " ".join(snippet.get("snippet", snippet.get("htmlSnippet", "")) for snippet in snippets)


def _document_text(derived: VertexSearchDerivedStructData) -> str:
    candidates: Final = (
        _joined_content(derived.get("extractive_segments", ())),
        _joined_content(derived.get("extractive_answers", ())),
        _snippet_text(derived.get("snippets", ())),
        derived.get("title", ""),
    )
    return next((text for text in candidates if text), "")


def _document_id_from_chunk_name(name: str) -> str:
    return name.partition("/documents/")[2].partition("/")[0]


def _non_empty_attributes(pairs: Iterable[tuple[str, object]]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in pairs if value})


def _chunk_result(chunk: VertexSearchChunk, positional_score: float) -> VectorStoreSearchResult:
    metadata: Final = chunk.get("documentMetadata", _EMPTY_CHUNK_DOCUMENT_METADATA)
    uri: Final = metadata.get("uri", "")
    title: Final = metadata.get("title", "")
    document_id: Final = _document_id_from_chunk_name(chunk.get("name", ""))
    return VectorStoreSearchResult(
        score=chunk.get("relevanceScore", positional_score),
        content=[VectorStoreResultContent(text=chunk.get("content", ""), type="text")],
        file_id=uri or document_id,
        filename=title or _UNKNOWN_DOCUMENT,
        attributes={
            "document_id": document_id,
            **_non_empty_attributes(
                (
                    ("chunk_id", chunk.get("id", "")),
                    ("link", uri),
                    ("title", title),
                    ("structData", metadata.get("structData")),
                    ("pageSpan", chunk.get("pageSpan")),
                )
            ),
        },
    )


def _document_result(hit: VertexSearchHit, score: float) -> VectorStoreSearchResult:
    document: Final = hit.get("document", _EMPTY_DOCUMENT)
    derived: Final = document.get("derivedStructData", _EMPTY_DERIVED_STRUCT_DATA)
    link: Final = derived.get("link", "")
    title: Final = derived.get("title", "")
    document_id: Final = hit.get("id", "")
    return VectorStoreSearchResult(
        score=score,
        content=[VectorStoreResultContent(text=_document_text(derived), type="text")],
        file_id=link or document_id,
        filename=title or _UNKNOWN_DOCUMENT,
        attributes={
            "document_id": document_id,
            **_non_empty_attributes(
                (
                    ("link", link),
                    ("title", title),
                    ("displayLink", derived.get("displayLink", "")),
                    ("formattedUrl", derived.get("formattedUrl", "")),
                    ("structData", document.get("structData")),
                )
            ),
        },
    )


def _search_result(hit: VertexSearchHit, position: int) -> VectorStoreSearchResult:
    score: Final = 1.0 / (position + 1)
    chunk: Final = hit.get("chunk")
    if chunk is not None:
        return _chunk_result(chunk, score)
    return _document_result(hit, score)


class VertexSearchAPIVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Search API Vector Store

    This implementation uses the Vertex AI Search API for vector store operations.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_supported_extra_body_fields(is_engine: bool = False) -> frozenset[str]:
        """
        Native SearchRequest fields callers may forward via ``extra_body``.

        The set depends on which serving config the request targets:
        - engine/app mode (``is_engine=True``): includes multi-store fields such
          as ``dataStoreSpecs`` and ``numResultsPerDataStore``.
        - data-store mode: the engine-only fields are excluded.
        """
        if is_engine:
            return VERTEX_SEARCH_ENGINE_EXTRA_BODY_FIELDS
        return VERTEX_SEARCH_DATASTORE_EXTRA_BODY_FIELDS

    @classmethod
    def _filter_extra_body(cls, extra_body: Mapping[str, object], is_engine: bool = False) -> dict[str, object]:
        """
        Validate ``extra_body`` against the supported-field allowlist for the
        active serving config (engine/app vs data store).

        Raises ``BadRequestError`` (HTTP 400) if the caller includes a
        target-selecting field (e.g. ``servingConfig``) or any field not
        supported for the active mode, so the request fails loudly instead of
        silently searching the wrong target. Engine-only fields
        (``dataStoreSpecs``, ``numResultsPerDataStore``) are rejected in
        data-store mode where they are meaningless.
        """
        supported: Final = cls.get_supported_extra_body_fields(is_engine=is_engine)
        filtered: Final = {key: value for key, value in extra_body.items() if value is not None}

        target_selecting: Final = set(filtered) & VERTEX_SEARCH_TARGET_SELECTING_FIELDS
        if target_selecting:
            raise BadRequestError(
                message=(
                    "Vertex AI Search extra_body may not set target-selecting fields "
                    f"{sorted(target_selecting)}: the data store is scoped by "
                    "vector_store_id / vertex_engine_id and cannot be overridden per request."
                ),
                model="vertex_ai/search_api",
                llm_provider="vertex_ai",
            )

        unsupported: Final = set(filtered) - supported
        if unsupported:
            mode: Final = "engine/app" if is_engine else "data store"
            raise BadRequestError(
                message=(
                    f"Unsupported Vertex AI Search extra_body fields {sorted(unsupported)} "
                    f"for {mode} mode. Supported fields: {sorted(supported)}."
                ),
                model="vertex_ai/search_api",
                llm_provider="vertex_ai",
            )

        return filtered

    def get_auth_credentials(self, litellm_params: dict) -> BaseVectorStoreAuthCredentials:
        # Get credentials and project info
        vertex_credentials: Final = self.get_vertex_ai_credentials(dict(litellm_params))
        vertex_project: Final = self.get_vertex_ai_project(dict(litellm_params))

        # Get access token using the base class method
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        return {
            "headers": {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        }

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [("POST", ":search")],
            "write": [],
        }

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        """
        Validate and set up authentication for Vertex AI RAG API
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        auth_headers: Final = self.get_auth_credentials(litellm_params.model_dump())
        headers.update(auth_headers.get("headers", {}))
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the Base endpoint for Vertex AI Search API.

        Branches on whether a `vertex_engine_id` is configured:
        - Engine ID present: route through the search app (engine) — required for website,
          healthcare, and connector-based data stores. Note the serving config name differs
          (`default_serving_config` vs `default_config` for direct data store search).
        - Engine ID absent: query the data store directly via `vector_store_id`.
        """
        if api_base:
            return api_base.rstrip("/")

        vertex_location: Final = self.get_vertex_ai_location(litellm_params)
        vertex_project: Final = self.get_vertex_ai_project(litellm_params)
        collection_id: Final = litellm_params.get("vertex_collection_id") or "default_collection"
        encoded_collection_id: Final = encode_url_path_segment(collection_id, field_name="vertex_collection_id")
        base: Final = (
            f"https://discoveryengine.googleapis.com/v1/"
            f"projects/{vertex_project}/locations/{vertex_location}/"
            f"collections/{encoded_collection_id}"
        )

        engine_id: Final = litellm_params.get("vertex_engine_id")
        if engine_id:
            encoded_engine_id: Final = encode_url_path_segment(engine_id, field_name="vertex_engine_id")
            return f"{base}/engines/{encoded_engine_id}/servingConfigs/default_serving_config"

        datastore_id: Final = litellm_params.get("vector_store_id")
        if not datastore_id:
            raise ValueError("vector_store_id is required when vertex_engine_id is not set")
        encoded_datastore_id: Final = encode_url_path_segment(datastore_id, field_name="vector_store_id")
        return f"{base}/dataStores/{encoded_datastore_id}/servingConfigs/default_config"

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | list[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        """
        Transform a search request for the Vertex AI Search (Discovery Engine) API.

        Per-request params pass through to the engine: max_num_results maps to
        pageSize, and extra_body fields on the supported allowlist
        (`get_supported_extra_body_fields`) are merged in with precedence, so
        callers can send native Discovery Engine tuning fields such as filter,
        boostSpec, or contentSearchSpec.

        The allowlist depends on the serving config: engine/app mode (when
        `vertex_engine_id` is set) additionally accepts multi-store fields like
        `dataStoreSpecs` and `numResultsPerDataStore`, while data-store mode
        rejects them. Target-selecting fields (e.g. servingConfig, branch) are
        rejected in both modes: the target is scoped by the URL path
        (vector_store_id / vertex_engine_id) and must not be overridable per
        request.
        """
        if isinstance(query, list):
            query = " ".join(query)

        url: Final = f"{api_base}:search"

        is_engine: Final = bool(litellm_params.get("vertex_engine_id"))

        request_body: Final[dict[str, object]] = {"query": query, "pageSize": 10}
        max_num_results: Final = vector_store_search_optional_params.get("max_num_results")
        if max_num_results is not None:
            request_body["pageSize"] = max_num_results
        if isinstance(extra_body, dict):
            request_body.update(self._filter_extra_body(extra_body, is_engine=is_engine))

        litellm_logging_obj.model_call_details["query"] = request_body.get("query", query)

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """
        Transform a Discovery Engine ``:search`` response into the standard vector store search response.

        Document hits (``results[].document``) take their text from ``derivedStructData`` in a fixed order:
        ``extractive_segments``, then ``extractive_answers``, then ``snippets``, then ``title``; ``structData``
        and the link metadata land in ``attributes``. Chunk hits (``results[].chunk``, returned when the
        caller sets ``contentSearchSpec.searchResultMode`` to ``CHUNKS`` via ``extra_body``) take their text
        from ``chunk.content`` and their file id and name from ``chunk.documentMetadata``.
        """
        try:
            response_json: Final = _vertex_search_payload(response)
            search_results: Final = [
                _search_result(hit, position) for position, hit in enumerate(response_json.get("results", ()))
            ]
            query_view: Final[_SearchQueryView] = {"query": litellm_logging_obj.model_call_details.get("query", "")}
            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=query_view["query"],
                data=search_results,
            )

        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> tuple[str, dict]:
        raise NotImplementedError

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        raise NotImplementedError

    def calculate_vector_store_cost(
        self,
        response: VectorStoreSearchResponse,
    ) -> tuple[float, float]:
        model_info: Final = get_model_info(
            model="vertex_ai/search_api",
        )

        input_cost_per_query: Final = model_info.get("input_cost_per_query") or 0.0
        return input_cost_per_query, 0.0
