from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

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
VERTEX_SEARCH_TARGET_SELECTING_FIELDS = frozenset(
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
VERTEX_SEARCH_DATASTORE_EXTRA_BODY_FIELDS = frozenset(VertexSearchDataStoreExtraBody.__annotations__)

VERTEX_SEARCH_ENGINE_EXTRA_BODY_FIELDS = frozenset(VertexSearchEngineExtraBody.__annotations__)


class VertexSearchAPIVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Search API Vector Store

    This implementation uses the Vertex AI Search API for vector store operations.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_supported_extra_body_fields(is_engine: bool = False) -> frozenset:
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
    def _filter_extra_body(cls, extra_body: Dict[str, Any], is_engine: bool = False) -> Dict[str, Any]:
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
        supported = cls.get_supported_extra_body_fields(is_engine=is_engine)
        filtered = {key: value for key, value in extra_body.items() if value is not None}

        target_selecting = set(filtered) & VERTEX_SEARCH_TARGET_SELECTING_FIELDS
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

        unsupported = set(filtered) - supported
        if unsupported:
            mode = "engine/app" if is_engine else "data store"
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
        vertex_credentials = self.get_vertex_ai_credentials(dict(litellm_params))
        vertex_project = self.get_vertex_ai_project(dict(litellm_params))

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

    def validate_environment(self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]) -> dict:
        """
        Validate and set up authentication for Vertex AI RAG API
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        auth_headers = self.get_auth_credentials(litellm_params.model_dump())
        headers.update(auth_headers.get("headers", {}))
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
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

        vertex_location = self.get_vertex_ai_location(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        collection_id = litellm_params.get("vertex_collection_id") or "default_collection"
        encoded_collection_id = encode_url_path_segment(collection_id, field_name="vertex_collection_id")
        base = (
            f"https://discoveryengine.googleapis.com/v1/"
            f"projects/{vertex_project}/locations/{vertex_location}/"
            f"collections/{encoded_collection_id}"
        )

        engine_id = litellm_params.get("vertex_engine_id")
        if engine_id:
            encoded_engine_id = encode_url_path_segment(engine_id, field_name="vertex_engine_id")
            return f"{base}/engines/{encoded_engine_id}/servingConfigs/default_serving_config"

        datastore_id = litellm_params.get("vector_store_id")
        if not datastore_id:
            raise ValueError("vector_store_id is required when vertex_engine_id is not set")
        encoded_datastore_id = encode_url_path_segment(datastore_id, field_name="vector_store_id")
        return f"{base}/dataStores/{encoded_datastore_id}/servingConfigs/default_config"

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
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

        url = f"{api_base}:search"

        is_engine = bool(litellm_params.get("vertex_engine_id"))

        request_body: Dict[str, Any] = {"query": query, "pageSize": 10}
        max_num_results = vector_store_search_optional_params.get("max_num_results")
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
        Transform Vertex AI Search API response to standard vector store search response

        Handles the format from Discovery Engine Search API which returns:
        {
            "results": [
                {
                    "id": "...",
                    "document": {
                        "derivedStructData": {
                            "title": "...",
                            "link": "...",
                            "snippets": [...]
                        }
                    }
                }
            ]
        }
        """
        try:
            response_json = response.json()

            # Extract results from Vertex AI Search API response
            results = response_json.get("results", [])

            # Transform results to standard format
            search_results: List[VectorStoreSearchResult] = []
            for result in results:
                document = result.get("document", {})
                derived_data = document.get("derivedStructData", {})

                # Extract text content from snippets
                snippets = derived_data.get("snippets", [])
                text_content = ""

                if snippets:
                    # Combine all snippets into one text
                    text_parts = [snippet.get("snippet", snippet.get("htmlSnippet", "")) for snippet in snippets]
                    text_content = " ".join(text_parts)

                # If no snippets, use title as fallback
                if not text_content:
                    text_content = derived_data.get("title", "")

                content = [
                    VectorStoreResultContent(
                        text=text_content,
                        type="text",
                    )
                ]

                # Extract file/document information
                document_link = derived_data.get("link", "")
                document_title = derived_data.get("title", "")
                document_id = result.get("id", "")

                # Use link as file_id if available, otherwise use document ID
                file_id = document_link if document_link else document_id
                filename = document_title if document_title else "Unknown Document"

                # Build attributes with available metadata
                attributes = {
                    "document_id": document_id,
                }

                if document_link:
                    attributes["link"] = document_link
                if document_title:
                    attributes["title"] = document_title

                # Add display link if available
                display_link = derived_data.get("displayLink", "")
                if display_link:
                    attributes["displayLink"] = display_link

                # Add formatted URL if available
                formatted_url = derived_data.get("formattedUrl", "")
                if formatted_url:
                    attributes["formattedUrl"] = formatted_url

                # Note: Search API doesn't provide explicit scores in the response
                # You can use the position/rank as an implicit score
                score = 1.0 / (float(search_results.__len__() + 1))  # Decreasing score based on position

                result_obj = VectorStoreSearchResult(
                    score=score,
                    content=content,
                    file_id=file_id,
                    filename=filename,
                    attributes=attributes,
                )
                search_results.append(result_obj)

            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=litellm_logging_obj.model_call_details.get("query", ""),
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
    ) -> Tuple[str, Dict]:
        raise NotImplementedError

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        raise NotImplementedError

    def calculate_vector_store_cost(
        self,
        response: VectorStoreSearchResponse,
    ) -> Tuple[float, float]:
        model_info = get_model_info(
            model="vertex_ai/search_api",
        )

        input_cost_per_query = model_info.get("input_cost_per_query") or 0.0
        return input_cost_per_query, 0.0
