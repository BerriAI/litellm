from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
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


class QdrantVectorStoreConfig(BaseVectorStoreConfig):
    """
    Configuration for Qdrant Vector Store.

    Supports vector search with embeddings generated via litellm.embeddings.
    """

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        api_key: Optional[str] = None
        if litellm_params is not None:
            api_key = litellm_params.api_key or get_secret_str("QDRANT_API_KEY")

        if api_key:
            headers.update({"api-key": api_key})

        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        api_key = litellm_params.get("api_key") or get_secret_str("QDRANT_API_KEY")
        headers: Dict[str, str] = {}
        if api_key:
            headers["api-key"] = api_key
        return {"headers": headers}

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [("POST", "/collections/{index_name}/points/search")],
            "write": [("PUT", "/collections/{index_name}")],
        }

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = (
            api_base
            or get_secret_str("QDRANT_API_BASE")
            or get_secret_str("QDRANT_URL")
        )

        if not api_base:
            raise ValueError(
                "Qdrant API base is required. Set QDRANT_API_BASE/QDRANT_URL or pass api_base in litellm_params."
            )

        return api_base.rstrip("/")

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> Tuple[str, Dict[str, Any]]:
        if isinstance(query, list):
            query = " ".join(query)

        embedding_model = litellm_params.get(
            "litellm_embedding_model"
        ) or litellm_params.get("embedding_model")
        if not embedding_model:
            raise ValueError(
                "litellm_embedding_model is required in litellm_params for Qdrant."
            )

        embedding_config = (
            litellm_params.get("litellm_embedding_config")
            or litellm_params.get("embedding_config")
            or {}
        )

        try:
            embedding_response = litellm.embedding(
                model=embedding_model,
                input=[query],
                **embedding_config,
            )
            query_vector = embedding_response.data[0]["embedding"]
        except Exception as e:
            raise Exception(f"Failed to generate embedding for query: {str(e)}")

        url = f"{api_base}/collections/{vector_store_id}/points/search"
        request_body: Dict[str, Any] = {
            "vector": query_vector,
            "limit": vector_store_search_optional_params.get("max_num_results", 10),
            "with_payload": True,
            "with_vectors": False,
        }

        if vector_store_search_optional_params.get("filters"):
            request_body["filter"] = vector_store_search_optional_params["filters"]

        litellm_logging_obj.model_call_details["input"] = query
        litellm_logging_obj.model_call_details["embedding_model"] = embedding_model

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        try:
            response_json = response.json()
            results = response_json.get("result", [])

            optional_params = litellm_logging_obj.model_call_details.get(
                "litellm_params", {}
            )
            text_field = optional_params.get("qdrant_text_field", "text")

            search_results: List[VectorStoreSearchResult] = []
            for result in results:
                payload = result.get("payload", {}) or {}
                text_content = payload.get(text_field) or payload.get("text")

                content: List[VectorStoreResultContent] = []
                if text_content:
                    content = [VectorStoreResultContent(text=text_content, type="text")]

                attributes = {
                    key: value for key, value in payload.items() if key != text_field
                }

                search_results.append(
                    VectorStoreSearchResult(
                        score=result.get("score", 0.0),
                        content=content,
                        file_id=None,
                        filename=None,
                        attributes=attributes,
                    )
                )

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
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Qdrant vector store creation is not supported.")

    def transform_create_vector_store_response(
        self, response: httpx.Response
    ) -> VectorStoreCreateResponse:
        raise NotImplementedError("Qdrant vector store creation is not supported.")
