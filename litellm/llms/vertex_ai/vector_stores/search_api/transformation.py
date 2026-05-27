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


# Default serving config IDs for the Discovery Engine Search API.
# Datastore-level configs are created with the id "default_config",
# while Engine (app) -level configs are created with the id "default_search".
_DEFAULT_DATASTORE_SERVING_CONFIG = "default_config"
_DEFAULT_ENGINE_SERVING_CONFIG = "default_search"


class VertexSearchAPIVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Search API Vector Store

    Two upstream targets are supported:

    * **Datastore** (default) - pass ``vector_store_id`` to address a single
      Discovery Engine datastore directly:
      ``collections/{c}/dataStores/{ds}/servingConfigs/default_config``
    * **App / Engine** - pass ``vertex_app_id`` (alias ``vertex_engine_id``)
      to address a Search app that can span multiple datastores:
      ``collections/{c}/engines/{app_id}/servingConfigs/default_search``

      When using an engine, callers may additionally pass
      ``vertex_datastores`` (a list of datastore ids) to limit the search
      to a subset of the engine's underlying datastores; this is forwarded
      to Discovery Engine as ``dataStoreSpecs[].dataStore``.
    """

    def __init__(self):
        super().__init__()

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        vertex_credentials = self.get_vertex_ai_credentials(dict(litellm_params))
        vertex_project = self.get_vertex_ai_project(dict(litellm_params))

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
        litellm_params = litellm_params or GenericLiteLLMParams()
        auth_headers = self.get_auth_credentials(litellm_params.model_dump())
        headers.update(auth_headers.get("headers", {}))
        return headers

    @staticmethod
    def _get_vertex_app_id(litellm_params: dict) -> Optional[str]:
        """Return the engine/app id from litellm_params, accepting both
        ``vertex_app_id`` and the alias ``vertex_engine_id``.

        Returns ``None`` (not empty string) when neither is set, so callers
        can use ``is None`` to detect the datastore fallback path.
        """
        app_id = litellm_params.get("vertex_app_id") or litellm_params.get(
            "vertex_engine_id"
        )
        if app_id is None:
            return None
        app_id_str = str(app_id)
        if app_id_str == "":
            return None
        return app_id_str

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """Get the base endpoint for the Vertex AI Search API.

        Resolves to one of:

        * ``.../collections/{c}/engines/{app_id}/servingConfigs/{serving_config}``
          when ``vertex_app_id`` (or ``vertex_engine_id``) is set, or
        * ``.../collections/{c}/dataStores/{ds}/servingConfigs/{serving_config}``
          when only ``vector_store_id`` is set (existing behaviour).
        """
        vertex_location = self.get_vertex_ai_location(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        collection_id = (
            litellm_params.get("vertex_collection_id") or "default_collection"
        )

        if api_base:
            return api_base.rstrip("/")

        encoded_collection_id = encode_url_path_segment(
            collection_id, field_name="vertex_collection_id"
        )

        app_id = self._get_vertex_app_id(litellm_params)
        if app_id is not None:
            encoded_app_id = encode_url_path_segment(app_id, field_name="vertex_app_id")
            serving_config = (
                litellm_params.get("vertex_serving_config")
                or _DEFAULT_ENGINE_SERVING_CONFIG
            )
            encoded_serving_config = encode_url_path_segment(
                serving_config, field_name="vertex_serving_config"
            )
            return (
                f"https://discoveryengine.googleapis.com/v1/"
                f"projects/{vertex_project}/locations/{vertex_location}/"
                f"collections/{encoded_collection_id}/engines/{encoded_app_id}/"
                f"servingConfigs/{encoded_serving_config}"
            )

        datastore_id = litellm_params.get("vector_store_id")
        if not datastore_id:
            raise ValueError(
                "vector_store_id is required (or pass vertex_app_id / "
                "vertex_engine_id for app-level search)"
            )
        encoded_datastore_id = encode_url_path_segment(
            datastore_id, field_name="vector_store_id"
        )
        serving_config = (
            litellm_params.get("vertex_serving_config")
            or _DEFAULT_DATASTORE_SERVING_CONFIG
        )
        encoded_serving_config = encode_url_path_segment(
            serving_config, field_name="vertex_serving_config"
        )

        return (
            f"https://discoveryengine.googleapis.com/v1/"
            f"projects/{vertex_project}/locations/{vertex_location}/"
            f"collections/{encoded_collection_id}/dataStores/{encoded_datastore_id}/"
            f"servingConfigs/{encoded_serving_config}"
        )

    @staticmethod
    def _build_data_store_specs(
        litellm_params: dict,
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        collection_id: str,
    ) -> Optional[List[Dict[str, str]]]:
        """Build ``dataStoreSpecs`` from a ``vertex_datastores`` litellm_param.

        Each entry may already be a fully-qualified datastore resource path
        (``projects/.../dataStores/{id}``) or a bare id; bare ids are expanded
        to the full resource path the engine is already operating under so
        callers don't have to repeat the project/location.
        Returns ``None`` when no datastores are specified.
        """
        raw = litellm_params.get("vertex_datastores")
        if not raw:
            return None
        if not isinstance(raw, list):
            raise ValueError(
                "vertex_datastores must be a list of datastore ids or " "resource paths"
            )

        specs: List[Dict[str, str]] = []
        for entry in raw:
            if entry is None:
                continue
            entry_str = str(entry).strip()
            if not entry_str:
                continue
            if entry_str.startswith("projects/"):
                resource = entry_str
            else:
                if not vertex_project or not vertex_location:
                    raise ValueError(
                        "vertex_project and vertex_location are required to "
                        "expand a bare vertex_datastores entry into a full "
                        "Discovery Engine resource path"
                    )
                resource = (
                    f"projects/{vertex_project}/locations/{vertex_location}/"
                    f"collections/{collection_id}/dataStores/{entry_str}"
                )
            specs.append({"dataStore": resource})
        return specs or None

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
        """Transform search request for Vertex AI Search API.

        For engine (app-level) searches, optionally narrows the search to a
        subset of the engine's underlying datastores by forwarding
        ``vertex_datastores`` as ``dataStoreSpecs``.
        """
        if isinstance(query, list):
            query = " ".join(query)

        url = f"{api_base}:search"

        request_body: Dict[str, Any] = {"query": query, "pageSize": 10}

        if self._get_vertex_app_id(litellm_params) is not None:
            # Use safe_get_* (read-only) here so we don't double-pop the
            # vertex_project / vertex_location keys that get_complete_url
            # already pop'd off its own copy of litellm_params.
            data_store_specs = self._build_data_store_specs(
                litellm_params=litellm_params,
                vertex_project=self.safe_get_vertex_ai_project(litellm_params),
                vertex_location=self.safe_get_vertex_ai_location(litellm_params),
                collection_id=(
                    litellm_params.get("vertex_collection_id") or "default_collection"
                ),
            )
            if data_store_specs is not None:
                request_body["dataStoreSpecs"] = data_store_specs

        litellm_logging_obj.model_call_details["query"] = query

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        try:
            response_json = response.json()

            results = response_json.get("results", [])

            search_results: List[VectorStoreSearchResult] = []
            for result in results:
                document = result.get("document", {})
                derived_data = document.get("derivedStructData", {})

                snippets = derived_data.get("snippets", [])
                text_content = ""

                if snippets:
                    text_parts = [
                        snippet.get("snippet", snippet.get("htmlSnippet", ""))
                        for snippet in snippets
                    ]
                    text_content = " ".join(text_parts)

                if not text_content:
                    text_content = derived_data.get("title", "")

                content = [
                    VectorStoreResultContent(
                        text=text_content,
                        type="text",
                    )
                ]

                document_link = derived_data.get("link", "")
                document_title = derived_data.get("title", "")
                document_id = result.get("id", "")

                file_id = document_link if document_link else document_id
                filename = document_title if document_title else "Unknown Document"

                attributes = {
                    "document_id": document_id,
                }

                if document_link:
                    attributes["link"] = document_link
                if document_title:
                    attributes["title"] = document_title

                display_link = derived_data.get("displayLink", "")
                if display_link:
                    attributes["displayLink"] = display_link

                formatted_url = derived_data.get("formattedUrl", "")
                if formatted_url:
                    attributes["formattedUrl"] = formatted_url

                score = 1.0 / (float(search_results.__len__() + 1))

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
