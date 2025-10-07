"""
Transformation logic for Hosted VLLM rerank
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankRequest,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class HostedVLLMRerankError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class HostedVLLMRerankConfig(BaseRerankConfig):
    def __init__(self) -> None:
        pass

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        if api_base:
            # Remove trailing slashes and ensure clean base URL
            api_base = api_base.rstrip("/")
            # Preserve backward compatibility
            if api_base.endswith("/v1/rerank"):
                api_base = api_base.replace("/v1/rerank", "/rerank")
            elif not api_base.endswith("/rerank"):
                api_base = f"{api_base}/rerank"
            return api_base
        raise ValueError("api_base must be provided for Hosted VLLM rerank")

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "rank_fields",
            "return_documents",
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
        Map parameters for Hosted VLLM rerank
        """
        if max_chunks_per_doc is not None:
            raise ValueError("Hosted VLLM does not support max_chunks_per_doc")
            
        return dict(OptionalRerankParams(
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
        ))

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"

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
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Hosted VLLM rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Hosted VLLM rerank")
        
        rerank_request = RerankRequest(
            model=model,
            query=optional_rerank_params["query"],
            documents=optional_rerank_params["documents"],
            top_n=optional_rerank_params.get("top_n", None),
            rank_fields=optional_rerank_params.get("rank_fields", None),
            return_documents=optional_rerank_params.get("return_documents", None),
        )
        return rerank_request.model_dump(exclude_none=True)

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
        Process response from Hosted VLLM rerank API
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise ValueError(
                f"Error parsing response: {raw_response.text}, status_code={raw_response.status_code}"
            )

        return RerankResponse(**raw_response_json)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return HostedVLLMRerankError(message=error_message, status_code=status_code, headers=headers)

    def _transform_response(self, response: dict) -> RerankResponse:
        # Extract usage information
        usage_data = response.get("usage", {})
        _billed_units = RerankBilledUnits(total_tokens=usage_data.get("total_tokens", 0))
        _tokens = RerankTokens(input_tokens=usage_data.get("total_tokens", 0))
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        # Extract results
        _results: Optional[List[dict]] = response.get("results")

        if _results is None:
            raise ValueError(f"No results found in the response={response}")

        rerank_results: List[RerankResponseResult] = []

        for result in _results:
            # Validate required fields exist
            if not all(key in result for key in ["index", "relevance_score"]):
                raise ValueError(f"Missing required fields in the result={result}")

            # Get document data if it exists
            document_data = result.get("document", {})
            document = (
                RerankResponseDocument(text=str(document_data.get("text", "")))
                if document_data
                else None
            )

            # Create typed result
            rerank_result = RerankResponseResult(
                index=int(result["index"]),
                relevance_score=float(result["relevance_score"]),
            )

            # Only add document if it exists
            if document:
                rerank_result["document"] = document

            rerank_results.append(rerank_result)

        return RerankResponse(
            id=response.get("id") or str(uuid.uuid4()),
            results=rerank_results,
            meta=rerank_meta,
        ) 