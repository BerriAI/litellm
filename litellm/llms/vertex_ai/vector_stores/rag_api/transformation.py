from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol

import httpx
from typing_extensions import ReadOnly, TypedDict

from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
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
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.router import Router

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexRagPageSpan(TypedDict, total=False):
    """Page range a retrieved chunk came from, as ``:retrieveContexts`` returns it."""

    firstPage: ReadOnly[int]
    lastPage: ReadOnly[int]


class VertexRagContext(TypedDict, total=False):
    """One retrieved chunk in a Vertex AI RAG ``:retrieveContexts`` response."""

    text: ReadOnly[str]
    sourceUri: ReadOnly[str]
    sourceDisplayName: ReadOnly[str]
    score: ReadOnly[float]
    pageSpan: ReadOnly[VertexRagPageSpan]


class VertexRagContextGroup(TypedDict, total=False):
    contexts: ReadOnly[list[VertexRagContext]]


class VertexRagRetrieveContextsResponse(TypedDict, total=False):
    contexts: ReadOnly[VertexRagContextGroup]


class VertexRagCorpusResponse(TypedDict, total=False):
    """A RAG corpus resource, as ``POST /ragCorpora`` returns it."""

    name: ReadOnly[str]
    display_name: ReadOnly[str]
    createTime: ReadOnly[object]
    labels: ReadOnly[object]


class _SearchQueryView(TypedDict):
    """Holds the logged search query so the model call detail reads back as ``str``."""

    query: ReadOnly[str]


class _RetrieveContextsSource(Protocol):
    """An HTTP response whose JSON body is a Vertex AI RAG ``:retrieveContexts`` result."""

    def json(self) -> VertexRagRetrieveContextsResponse: ...


class _RagCorpusSource(Protocol):
    """An HTTP response whose JSON body is a Vertex AI RAG corpus resource."""

    def json(self) -> VertexRagCorpusResponse: ...


def _retrieve_contexts_payload(response: _RetrieveContextsSource) -> VertexRagRetrieveContextsResponse:
    return response.json()


def _rag_corpus_payload(response: _RagCorpusSource) -> VertexRagCorpusResponse:
    return response.json()


class VertexVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Vector Store RAG API

    This implementation uses the Vertex AI RAG Engine API for vector store operations.
    """

    def __init__(self):
        super().__init__()

    def get_auth_credentials(self, litellm_params: Mapping[str, object]) -> BaseVectorStoreAuthCredentials:
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
            "read": [("POST", ":retrieveContexts")],
            "write": [("POST", "/ragCorpora")],
        }

    def validate_environment(
        self, headers: dict[str, str], litellm_params: GenericLiteLLMParams | None
    ) -> dict[str, str]:
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
        litellm_params: dict[str, object],
    ) -> str:
        """
        Get the Base endpoint for Vertex AI RAG API
        """
        vertex_location: Final = self.get_vertex_ai_location(litellm_params)
        vertex_project: Final = self.get_vertex_ai_project(litellm_params)

        if api_base:
            return api_base.rstrip("/")

        # Vertex AI RAG API endpoint for retrieveContexts
        base_url: Final = get_vertex_base_url(vertex_location)
        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}"

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str | list[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: Mapping[str, object] | None = None,
        router: "Router | None" = None,
    ) -> tuple[str, dict[str, object]]:
        """
        Transform search request for Vertex AI RAG API
        """
        # Convert query to string if it's a list
        if isinstance(query, list):
            query = " ".join(query)

        # Vertex AI RAG API endpoint for retrieving contexts
        url: Final = f"{api_base}:retrieveContexts"

        # Use helper methods to get project and location, then construct full rag corpus path
        vertex_project: Final = self.get_vertex_ai_project(litellm_params)
        vertex_location: Final = self.get_vertex_ai_location(litellm_params)

        # Handle both full corpus path and just corpus ID
        if vector_store_id.startswith("projects/"):
            # Already a full path
            full_rag_corpus = vector_store_id
        else:
            # Just the corpus ID, construct full path
            full_rag_corpus = f"projects/{vertex_project}/locations/{vertex_location}/ragCorpora/{vector_store_id}"

        #########################################################
        # Update logging object with details of the request
        #########################################################
        litellm_logging_obj.model_call_details["query"] = query

        # Add optional parameters
        max_num_results: Final = vector_store_search_optional_params.get("max_num_results")
        filters: Final = vector_store_search_optional_params.get("filters")
        ranking_options: Final = vector_store_search_optional_params.get("ranking_options")
        rag_retrieval_config: Final[Mapping[str, object]] = {
            key: value
            for key, value in (
                ("top_k", max_num_results),
                ("filter", filters),
                ("ranking", ranking_options),
            )
            if value is not None
        }

        query_body: Final[Mapping[str, object]] = {
            key: value
            for key, value in (("text", query), ("rag_retrieval_config", rag_retrieval_config or None))
            if value is not None
        }
        request_body: Final[dict[str, object]] = {
            "vertex_rag_store": {"rag_resources": [{"rag_corpus": full_rag_corpus}]},
            "query": query_body,
        }

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """
        Transform Vertex AI RAG API response to standard vector store search response
        """
        try:
            response_json: Final = _retrieve_contexts_payload(response)
            # Extract contexts from Vertex AI response - handle nested structure
            context_group: Final[VertexRagContextGroup] = response_json.get("contexts", {})
            contexts: Final = context_group.get("contexts", [])

            # Transform contexts to standard format
            search_results: Final[list[VectorStoreSearchResult]] = []
            for context in contexts:
                content = [
                    VectorStoreResultContent(
                        text=context.get("text", ""),
                        type="text",
                    )
                ]

                # Extract file information
                source_uri = context.get("sourceUri", "")
                source_display_name = context.get("sourceDisplayName", "")

                # Generate file_id from source URI or use display name as fallback
                file_id = source_uri if source_uri else source_display_name
                filename = source_display_name if source_display_name else "Unknown Document"

                # Build attributes with available metadata
                attributes: dict[str, object] = {}
                if source_uri:
                    attributes["sourceUri"] = source_uri
                if source_display_name:
                    attributes["sourceDisplayName"] = source_display_name

                # Add page span information if available
                page_span = context.get("pageSpan", {})
                if page_span:
                    attributes["pageSpan"] = page_span

                result = VectorStoreSearchResult(
                    score=context.get("score", 0.0),
                    content=content,
                    file_id=file_id,
                    filename=filename,
                    attributes=attributes,
                )
                search_results.append(result)

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
    ) -> tuple[str, dict[str, object]]:
        """
        Transform create request for Vertex AI RAG Corpus
        """
        url: Final = f"{api_base}/ragCorpora"  # Base URL for creating RAG corpus

        # Add metadata if provided
        metadata: Final = vector_store_create_optional_params.get("metadata")

        request_body: Final[dict[str, object]] = {
            key: value
            for key, value in (
                ("display_name", vector_store_create_optional_params.get("name", "litellm-vector-store")),
                ("description", "Vector store created via LiteLLM"),
                ("labels", metadata),
            )
            if value is not None
        }

        return url, request_body

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        """
        Transform Vertex AI RAG Corpus creation response to standard vector store response
        """
        try:
            response_json: Final = _rag_corpus_payload(response)

            # Extract the corpus ID from the response name
            corpus_name: Final = response_json.get("name", "")
            corpus_id: Final = corpus_name.split("/")[-1] if "/" in corpus_name else corpus_name

            # Handle createTime conversion
            create_time = response_json.get("createTime", 0)
            if isinstance(create_time, str):
                # Convert ISO timestamp to Unix timestamp
                from datetime import datetime

                try:
                    dt: Final = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
                    create_time = int(dt.timestamp())
                except ValueError:
                    create_time = 0
            elif not isinstance(create_time, int):
                create_time = 0

            # Handle labels safely
            labels: Final = response_json.get("labels", {})
            metadata: Final = labels if isinstance(labels, dict) else {}

            return VectorStoreCreateResponse(
                id=corpus_id,
                object="vector_store",
                created_at=create_time,
                name=response_json.get("display_name", ""),
                bytes=0,  # Vertex AI doesn't provide byte count in the same way
                file_counts={
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 0,
                },
                status="completed",  # Vertex AI corpus creation is typically synchronous
                expires_after=None,
                expires_at=None,
                last_active_at=None,
                metadata=metadata,
            )

        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers,
            )
