from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.base_llm.vector_store.transformation import (
    BaseQueryEmbeddingVectorStoreConfig,
    VectorStoreEmbeddingExecutor,
)
from litellm.secret_managers.main import get_secret_str
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

MILVUS_OPTIONAL_PARAMS: Final = {
    "annsField",
    "limit",
    "filter",
    "offset",
    "groupingField",
    "outputFields",
    "searchParams",
    "consistencyLevel",
}


class MilvusVectorStoreConfig(BaseQueryEmbeddingVectorStoreConfig):
    """
    Configuration for Milvus Vector Store

    This implementation uses the Azure AI Search API for vector store operations.
    Supports vector search with embeddings generated via litellm.embeddings.
    """

    def __init__(self):
        super().__init__()

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        api_key: str | None = None
        if litellm_params is not None:
            api_key = litellm_params.api_key or get_secret_str("MILVUS_API_KEY")

        if not api_key:
            raise ValueError(
                "MILVUS_API_KEY is not set. Either set it in the litellm_params or set the MILVUS_API_KEY environment variable."
            )

        headers.update({"Authorization": f"Bearer {api_key}"})

        return headers

    def get_auth_credentials(self, litellm_params: dict) -> BaseVectorStoreAuthCredentials:
        api_key: Final = litellm_params.get("api_key")
        if not api_key:
            raise ValueError(
                "MILVUS_API_KEY is not set. Either set it in the litellm_params or set the MILVUS_API_KEY environment variable."
            )
        return {
            "headers": {
                "Authorization": f"Bearer {api_key}",
            },
        }

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [
                ("POST", "/v2/vectordb/entities/search"),
                ("POST", "/v2/vectordb/entities/get"),
                ("POST", "/v2/vectordb/entities/query"),
            ],
            "write": [
                ("POST", "/v2/vectordb/entities/upsert"),
                ("POST", "/v2/vectordb/entities/insert"),
            ],
        }

    def map_openai_params(self, non_default_params: dict, optional_params: dict, drop_params: bool) -> dict:
        for param, value in non_default_params.items():
            if param in MILVUS_OPTIONAL_PARAMS:
                optional_params[param] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the base endpoint for Milvus API

        Expected format: https://{milvus_api_base}.milvus.io
        """
        api_base = api_base or get_secret_str("MILVUS_API_BASE")

        if not api_base:
            raise ValueError(
                "Milvus API base URL is required. Set MILVUS_API_BASE environment variable or pass api_base in litellm_params."
            )

        if api_base:
            return api_base.rstrip("/")

        return api_base

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
    ) -> tuple[str, dict[str, object]]:
        query_text: Final = self.query_text(query)
        query_vector: Final = self.embed_query(query_text, litellm_params, embedding_executor)
        return self._search_request(
            vector_store_id,
            query_text,
            query_vector,
            vector_store_search_optional_params,
            api_base,
            litellm_logging_obj,
            litellm_params,
        )

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
    ) -> tuple[str, dict[str, object]]:
        query_text: Final = self.query_text(query)
        query_vector: Final = await self.aembed_query(query_text, litellm_params, embedding_executor)
        return self._search_request(
            vector_store_id,
            query_text,
            query_vector,
            vector_store_search_optional_params,
            api_base,
            litellm_logging_obj,
            litellm_params,
        )

    @staticmethod
    def _search_request(
        vector_store_id: str,
        query_text: str,
        query_vector: Sequence[float],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, object]]:
        scope: Final = {
            key: value
            for key, value in (
                ("dbName", litellm_params.get("milvus_db_name")),
                ("partitionNames", litellm_params.get("milvus_partition_names")),
            )
            if value
        }
        litellm_logging_obj.model_call_details["input"] = query_text
        litellm_logging_obj.model_call_details["embedding_model"] = litellm_params.get("litellm_embedding_model")
        return f"{api_base}/v2/vectordb/entities/search", {
            "collectionName": vector_store_id,
            "data": [query_vector],
            "annsField": "book_intro_vector",
            **vector_store_search_optional_params,
            **scope,
        }

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """
        Transform Azure AI Search API response to standard vector store search response

        Handles the format from Azure AI Search which returns:
        {
            "value": [
                {
                    "id": "...",
                    "content": "...",
                    "distance": 0.95,
                }
            ]
        }
        """
        try:
            response_json: Final = response.json()

            # Extract results from Azure AI Search API response
            results: Final = response_json.get("data", [])

            # Try to get text_field from optional_params first, then litellm_params
            optional_params: Final = litellm_logging_obj.model_call_details.get("optional_params", {})
            text_field = optional_params.get("milvus_text_field", "")

            # Fallback to litellm_params if not in optional_params

            if not text_field:
                text_field = litellm_logging_obj.model_call_details.get("litellm_params", {}).get(
                    "milvus_text_field", ""
                )

            # Transform results to standard format
            search_results: Final[list[VectorStoreSearchResult]] = []
            for result in results:
                # Extract text content
                text_content = result.get(text_field, "")

                content = [
                    VectorStoreResultContent(
                        text=text_content,
                        type="text",
                    )
                ]

                # Get the search score (distance from the query vector)
                score = result.get("distance", 0.0)

                # Build attributes with all available metadata
                # Exclude system fields and already-processed fields
                attributes = {}
                for key, value in result.items():
                    if key not in ["id", "content", "distance", text_field]:
                        attributes[key] = value

                result_obj = VectorStoreSearchResult(
                    score=score,
                    content=content,
                    file_id=None,
                    filename=None,
                    attributes=attributes,
                )
                search_results.append(result_obj)

            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=litellm_logging_obj.model_call_details.get("input", ""),
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
