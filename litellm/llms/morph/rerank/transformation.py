from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import OptionalRerankParams
from litellm.types.utils import RerankResponse


class MorphRerankConfig(BaseRerankConfig):
    """
    Reference: https://docs.morphllm.com/api-reference/endpoint/rerank
    
    Morph's Rerank API improves search quality by reordering candidate results based on 
    their relevance to a query.
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        if api_base:
            # Remove trailing slashes and ensure clean base URL
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/v1/rerank"):
                api_base = f"{api_base}/v1/rerank"
            return api_base
        return "https://api.morphllm.com/v1/rerank"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
            "embedding_ids",  # Morph specific - either documents or embedding_ids must be provided
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = None,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
        embedding_ids: Optional[List[str]] = None,
    ) -> OptionalRerankParams:
        """
        Map Morph rerank params
        
        Returns all supported params
        """
        # Start with basic parameters
        params: OptionalRerankParams = {
            "query": query,
            "documents": documents,
        }
        
        # Add optional parameters
        if top_n is not None:
            params["top_n"] = top_n
        if return_documents is not None:
            params["return_documents"] = return_documents
            
        # Add Morph-specific parameter if available
        if embedding_ids is not None:
            # We can't add this directly to OptionalRerankParams because it's not in the TypedDict
            # but we need to pass it to the API, so we'll add it as a custom parameter
            params = dict(params)  # type: ignore
            params["embedding_ids"] = embedding_ids  # type: ignore
        
        return params  # type: ignore

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("MORPH_API_KEY")

        if api_key is None:
            raise ValueError("Morph API key is required. Please set 'MORPH_API_KEY' environment variable.")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: OptionalRerankParams,
        headers: dict,
    ) -> dict:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Morph rerank")
        
        # Either documents or embedding_ids must be provided
        documents_provided = "documents" in optional_rerank_params
        # Cast to dict to access embedding_ids which is not in the OptionalRerankParams type
        params_dict = dict(optional_rerank_params)  # type: ignore
        embedding_ids_provided = "embedding_ids" in params_dict
        
        if not documents_provided and not embedding_ids_provided:
            raise ValueError("Either documents or embedding_ids is required for Morph rerank")
        
        # Create request with the model name stripped of any prefix
        request_data: Dict[str, Any] = {
            "model": model.replace("morph/", ""),
            "query": optional_rerank_params["query"],
        }
        
        # Add either documents or embedding_ids
        if documents_provided:
            request_data["documents"] = optional_rerank_params["documents"]
        if embedding_ids_provided:
            request_data["embedding_ids"] = params_dict.get("embedding_ids")
        
        # Add optional parameters
        if "top_n" in optional_rerank_params and optional_rerank_params["top_n"] is not None:
            request_data["top_n"] = optional_rerank_params["top_n"]
        if "return_documents" in optional_rerank_params and optional_rerank_params["return_documents"] is not None:
            request_data["return_documents"] = optional_rerank_params["return_documents"]
            
        return request_data

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        """
        Transform Morph rerank response

        The Morph API response format looks like:
        {
          "model": "morph-rerank-v2",
          "results": [
            {
              "index": 0,
              "document": "This Express.js middleware provides authentication using JWT tokens and protects routes.",
              "relevance_score": 0.92
            },
            ...
          ]
        }

        We need to transform it to the LiteLLM format which follows Cohere:
        {
          "id": "str",
          "results": [
            {
              "index": 0,
              "relevance_score": 0.92,
              "document": "str"
            },
            ...
          ]
        }
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise MorphError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        # Copy model field to id if needed
        if "model" in raw_response_json and "id" not in raw_response_json:
            raw_response_json["id"] = raw_response_json["model"]

        return RerankResponse(**raw_response_json)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return MorphError(message=error_message, status_code=status_code)


class MorphError(BaseLLMException):
    """
    Exception raised for Morph API errors.
    """
    pass
