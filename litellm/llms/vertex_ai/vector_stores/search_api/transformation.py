from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm import get_model_info
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
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexSearchAPIVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Search API Vector Store

    This implementation uses the Vertex AI Search API for vector store operations.
    """

    def __init__(self):
        super().__init__()

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
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

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
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
        Get the base endpoint for the Vertex AI Search API.

        Two parent resource types are supported:

        1. Data store (default, backwards-compatible):
           ``collections/{collection}/dataStores/{datastore}/servingConfigs/{serving_config}``
           Selected when no engine id is supplied. ``vector_store_id`` is the
           datastore id and is required in this mode.

        2. Engine / app (new):
           ``collections/{collection}/engines/{engine}/servingConfigs/{serving_config}``
           Selected when ``vertex_engine_id`` (alias ``vertex_app_id``) is set on
           ``litellm_params``. ``vector_store_id`` is optional in this mode — when
           the engine has multiple data stores, callers should pass
           ``vertex_data_store_specs`` (a list of ``DataStoreSpec`` dicts) so the
           search request body can scope results to specific data stores.

        ``vertex_serving_config_id`` (default ``"default_config"``) controls the
        trailing serving-config segment in either mode.
        """
        vertex_location = self.get_vertex_ai_location(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        collection_id = (
            litellm_params.get("vertex_collection_id") or "default_collection"
        )
        serving_config_id = (
            litellm_params.get("vertex_serving_config_id") or "default_config"
        )
        engine_id = litellm_params.get("vertex_engine_id") or litellm_params.get(
            "vertex_app_id"
        )

        if api_base:
            return api_base.rstrip("/")

        encoded_collection_id = encode_url_path_segment(
            collection_id, field_name="vertex_collection_id"
        )
        encoded_serving_config_id = encode_url_path_segment(
            serving_config_id, field_name="vertex_serving_config_id"
        )

        base = (
            f"https://discoveryengine.googleapis.com/v1/"
            f"projects/{vertex_project}/locations/{vertex_location}/"
            f"collections/{encoded_collection_id}"
        )

        if engine_id:
            # App / engine-level URL — vector_store_id is optional here. When the
            # engine has multiple data stores, the caller filters via
            # ``dataStoreSpecs`` in the request body (see
            # transform_search_vector_store_request).
            encoded_engine_id = encode_url_path_segment(
                engine_id, field_name="vertex_engine_id"
            )
            return (
                f"{base}/engines/{encoded_engine_id}"
                f"/servingConfigs/{encoded_serving_config_id}"
            )

        # Data store mode — keep existing requirement that the datastore id is
        # present so callers don't accidentally hit a global serving config.
        datastore_id = litellm_params.get("vector_store_id")
        if not datastore_id:
            raise ValueError("vector_store_id is required")
        encoded_datastore_id = encode_url_path_segment(
            datastore_id, field_name="vector_store_id"
        )
        return (
            f"{base}/dataStores/{encoded_datastore_id}"
            f"/servingConfigs/{encoded_serving_config_id}"
        )

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
        Transform a vector-store search request into a Vertex AI Search API call.

        In addition to the basic ``{query, pageSize}`` body, this supports two
        knobs needed for engine / app-level routing:

        - ``vertex_data_store_specs`` on ``litellm_params``: a list of
          ``DataStoreSpec`` dicts forwarded verbatim as ``dataStoreSpecs`` in the
          request body. Used to scope an engine-level search across multiple
          datastores. Each spec's ``dataStore`` field must be the full resource
          name, e.g. ``projects/<num>/locations/<loc>/collections/<col>/dataStores/<ds>``.
        - ``extra_body``: any caller-supplied keys are merged into the request
          body after the defaults are applied (callers can override
          ``pageSize`` or pass other SearchRequest fields like
          ``numResultsPerDataStore``). An ``extra_body["dataStoreSpecs"]`` entry
          takes precedence over ``vertex_data_store_specs`` so callers who
          plumb specs through ``extra_body`` keep working unchanged.
        """
        # Convert query to string if it's a list
        if isinstance(query, list):
            query = " ".join(query)

        # Vertex AI Search API endpoint for search
        url = f"{api_base}:search"

        request_body: Dict[str, Any] = {"query": query, "pageSize": 10}

        data_store_specs = litellm_params.get("vertex_data_store_specs")
        if data_store_specs:
            if not isinstance(data_store_specs, list):
                raise ValueError(
                    "vertex_data_store_specs must be a list of DataStoreSpec dicts"
                )
            for i, spec in enumerate(data_store_specs):
                if not isinstance(spec, dict):
                    raise ValueError(
                        f"vertex_data_store_specs[{i}] must be a DataStoreSpec dict, "
                        f"got {type(spec).__name__}"
                    )
            request_body["dataStoreSpecs"] = data_store_specs

        if extra_body:
            request_body.update(extra_body)

        #########################################################
        # Update logging object with details of the request
        #########################################################
        litellm_logging_obj.model_call_details["query"] = query

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
                    text_parts = [
                        snippet.get("snippet", snippet.get("htmlSnippet", ""))
                        for snippet in snippets
                    ]
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
                score = 1.0 / (
                    float(search_results.__len__() + 1)
                )  # Decreasing score based on position

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

    def transform_create_vector_store_response(
        self, response: httpx.Response
    ) -> VectorStoreCreateResponse:
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
