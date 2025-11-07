from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import OptionalRerankParams, RerankRequest, RerankResponse

from ..common_utils import CohereError


class CohereRerankConfig(BaseRerankConfig):
    """
    Reference: https://docs.cohere.com/v2/reference/rerank
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self, 
        api_base: Optional[str], 
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        if api_base:
            # Remove trailing slashes and ensure clean base URL
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/v1/rerank"):
                api_base = f"{api_base}/v1/rerank"
            return api_base
        return "https://api.cohere.ai/v1/rerank"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "max_chunks_per_doc",
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
        Map Cohere rerank params

        No mapping required - returns all supported params
        """
        return dict(OptionalRerankParams(
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
        ))

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> dict:
        if api_key is None:
            api_key = (
                get_secret_str("COHERE_API_KEY")
                or get_secret_str("CO_API_KEY")
                or litellm.cohere_key
            )

        if api_key is None:
            raise ValueError(
                "Cohere API key is required. Please set 'COHERE_API_KEY' or 'CO_API_KEY' or 'litellm.cohere_key'"
            )

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
            raise ValueError("query is required for Cohere rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Cohere rerank")
        rerank_request = RerankRequest(
            model=model,
            query=optional_rerank_params["query"],
            documents=optional_rerank_params["documents"],
            top_n=optional_rerank_params.get("top_n", None),
            rank_fields=optional_rerank_params.get("rank_fields", None),
            return_documents=optional_rerank_params.get("return_documents", None),
            max_chunks_per_doc=optional_rerank_params.get("max_chunks_per_doc", None),
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
        Transform Cohere rerank response

        No transformation required, litellm follows cohere API response format
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise CohereError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        return RerankResponse(**raw_response_json)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CohereError(message=error_message, status_code=status_code)
