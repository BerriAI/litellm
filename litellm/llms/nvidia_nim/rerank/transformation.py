from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from typing_extensions import Required, TypedDict

import litellm
from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
)


class NvidiaNimQueryObject(TypedDict):
    text: Required[str]


class NvidiaNimPassageObject(TypedDict):
    text: Required[str]


class NvidiaNimRerankRequest(TypedDict, total=False):
    model: Required[str]
    query: Required[NvidiaNimQueryObject]
    passages: Required[List[NvidiaNimPassageObject]]
    truncate: Literal["NONE", "END"]
    top_k: int


class NvidiaNimRankingResult(TypedDict):
    index: Required[int]
    logit: Required[float]


class NvidiaNimRerankResponse(TypedDict):
    rankings: Required[List[NvidiaNimRankingResult]]


class NvidiaNimRerankConfig(BaseRerankConfig):
    """
    Reference: https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_2-nv-rerankqa-1b-v2-infer
    
    Nvidia NIM rerank API uses a different format:
    - query is an object with 'text' field
    - documents are called 'passages' and have 'text' field
    """
    DEFAULT_NIM_RERANK_API_BASE = "https://ai.api.nvidia.com"

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self, 
        api_base: Optional[str], 
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        """
        Construct the Nvidia NIM rerank URL.
        
        Format: {api_base}/v1/retrieval/{model}/reranking
        
        If the user provides a full URL (e.g., {api_base}/v1/retrieval/{model}/reranking),
        it will be used as-is.
        """
        if not api_base:
            api_base = self.DEFAULT_NIM_RERANK_API_BASE
        
        api_base = api_base.rstrip("/")
        
        # Check if user already provided the full URL with /retrieval/ path
        if "/retrieval/" in api_base:
            return api_base
        
        # Ensure we don't have duplicate /v1
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]
        
        return f"{api_base}/v1/retrieval/{model}/reranking"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        """
        Nvidia NIM supports these rerank parameters.
        """
        return [
            "query",
            "documents",
            "top_n",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: Optional[dict],
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
    ) -> Dict:
        """
        Map Cohere/OpenAI rerank params to Nvidia NIM format.
        
        Parameter mapping:
        - top_n (Cohere) -> top_k (Nvidia)
        
        Nvidia NIM specific params (passed through as-is from non_default_params):
        - truncate: How to truncate input if too long (NONE, END)
        """
        optional_nvidia_nim_rerank_params: Dict[str, Any] = {
            "query": query,
            "documents": documents,
        }
        
        # Map Cohere's top_n to Nvidia's top_k
        if top_n is not None:
            optional_nvidia_nim_rerank_params["top_k"] = top_n
        
        # Pass through Nvidia-specific params from non_default_params
        if non_default_params:
            optional_nvidia_nim_rerank_params.update(non_default_params)
        return dict(optional_nvidia_nim_rerank_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> dict:
        """
        Validate that the Nvidia NIM API key is present.
        """
        if api_key is None:
            api_key = (
                get_secret_str("NVIDIA_NIM_API_KEY")
                or litellm.api_key
            )

        if api_key is None:
            raise ValueError(
                "Nvidia NIM API key is required. Please set 'NVIDIA_NIM_API_KEY' in your environment"
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        """
        Transform request to Nvidia NIM format.
        
        Nvidia NIM expects:
        - query as {text: "..."}
        - documents as passages: [{text: "..."}, ...]
        - Optional: truncate (NONE or END), top_k
        
        Note: optional_rerank_params may contain provider-specific params like 'top_k' and 'truncate'
        that aren't in the OptionalRerankParams TypedDict but are passed through at runtime.
        The mapping from Cohere's 'top_n' to Nvidia's 'top_k' already happened in map_cohere_rerank_params.
        """
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Nvidia NIM rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Nvidia NIM rerank")

        query = optional_rerank_params["query"]
        documents = optional_rerank_params["documents"]
        
        # Transform query to object format
        query_obj: NvidiaNimQueryObject = {"text": query}
        
        # Transform documents to passages format
        passages: List[NvidiaNimPassageObject] = []
        for doc in documents:
            if isinstance(doc, str):
                passages.append({"text": doc})
            elif isinstance(doc, dict):
                # If document is already a dict, check if it has 'text' field
                if "text" in doc:
                    passages.append({"text": doc["text"]})
                else:
                    # Otherwise, stringify the dict
                    import json
                    passages.append({"text": json.dumps(doc)})
            else:
                passages.append({"text": str(doc)})
        
        # Note: URL path uses underscores (llama-3_2) but JSON body uses periods (llama-3.2)
        # Convert underscores back to periods for the model field in request body
        model_for_body = model.replace("_", ".")
        
        # Build request using TypedDict
        request_data: NvidiaNimRerankRequest = {
            "model": model_for_body,
            "query": query_obj,
            "passages": passages,
        }
        
        # Add optional top_k parameter if provided (already mapped from top_n in map_cohere_rerank_params)
        if "top_k" in optional_rerank_params and optional_rerank_params.get("top_k") is not None:  # type: ignore
            request_data["top_k"] = optional_rerank_params.get("top_k")  # type: ignore
        
        # Add Nvidia-specific truncate parameter if provided
        # This is passed through from non_default_params, not in base OptionalRerankParams
        if "truncate" in optional_rerank_params and optional_rerank_params.get("truncate") is not None:  # type: ignore
            truncate_value = optional_rerank_params.get("truncate")  # type: ignore
            if truncate_value in ["NONE", "END"]:
                request_data["truncate"] = truncate_value  # type: ignore
        
        return dict(request_data)

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
        Transform Nvidia NIM rerank response to LiteLLM format.
        
        Nvidia NIM returns (NvidiaNimRerankResponse):
        {
            "rankings": [
                {
                    "index": 0,
                    "logit": 0.123
                }
            ]
        }
        
        LiteLLM expects (RerankResponse):
        {
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.123,
                    "document": {"text": "..."}  # optional
                }
            ]
        }
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            )

        # Parse as NvidiaNimRerankResponse
        nvidia_response: NvidiaNimRerankResponse = raw_response_json
        
        # Transform Nvidia NIM response to LiteLLM format
        results: List[RerankResponseResult] = []
        rankings = nvidia_response.get("rankings", [])
        
        # Get original documents from request if we need to include them
        original_passages: List[NvidiaNimPassageObject] = request_data.get("passages", [])
        
        for ranking in rankings:
            result_item: RerankResponseResult = {
                "index": ranking["index"],
                "relevance_score": ranking["logit"],
            }
            
            # Include document if it was in the original request
            index: int = ranking["index"]
            if index < len(original_passages):
                result_item["document"] = {"text": original_passages[index]["text"]}  # type: ignore
            
            results.append(result_item)
        
        # Construct metadata with billed_units
        # Nvidia NIM uses "usage" field with "total_tokens"
        usage = raw_response_json.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        
        billed_units: RerankBilledUnits = {
            "total_tokens": total_tokens if total_tokens > 0 else len(results)
        }
        
        meta: RerankResponseMeta = {
            "billed_units": billed_units
        }
        
        return RerankResponse(
            id=raw_response_json.get("id") or str(uuid.uuid4()),
            results=results,
            meta=meta,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

