import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from typing_extensions import TypedDict

import litellm
from litellm._uuid import uuid
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)
from litellm.utils import token_counter

from ..common_utils import HuggingFaceError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class HuggingFaceRerankResponseItem(TypedDict):
    """Type definition for HuggingFace rerank API response items."""

    index: int
    score: float
    text: Optional[str]  # Optional, included when return_text=True


class HuggingFaceRerankResponse(TypedDict):
    """Type definition for HuggingFace rerank API complete response."""

    # The response is a list of HuggingFaceRerankResponseItem
    pass


# Type alias for the actual response structure
HuggingFaceRerankResponseList = List[HuggingFaceRerankResponseItem]


class HuggingFaceRerankConfig(BaseRerankConfig):
    def get_api_base(self, model: str, api_base: Optional[str]) -> str:
        if api_base is not None:
            return api_base
        elif os.getenv("HF_API_BASE") is not None:
            return os.getenv("HF_API_BASE", "")
        elif os.getenv("HUGGINGFACE_API_BASE") is not None:
            return os.getenv("HUGGINGFACE_API_BASE", "")
        else:
            return "https://api-inference.huggingface.co"

    def get_complete_url(
        self, 
        api_base: Optional[str], 
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        """
        Get the complete URL for the API call, including the /rerank suffix if necessary.
        """
        # Get base URL from api_base or default
        base_url = self.get_api_base(model=model, api_base=api_base)

        # Remove trailing slashes and ensure we have the /rerank endpoint
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/rerank"):
            base_url = f"{base_url}/rerank"

        return base_url

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
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
        optional_rerank_params = {}
        if non_default_params is not None:
            for k, v in non_default_params.items():
                if k == "documents" and v is not None:
                    optional_rerank_params["texts"] = v
                elif k == "return_documents" and v is not None and isinstance(v, bool):
                    optional_rerank_params["return_text"] = v
                elif k == "top_n" and v is not None:
                    optional_rerank_params["top_n"] = v
                elif k == "documents" and v is not None:
                    optional_rerank_params["texts"] = v
                elif k == "query" and v is not None:
                    optional_rerank_params["query"] = v

        return OptionalRerankParams(**optional_rerank_params)  # type: ignore

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # Get API credentials
        api_key, api_base = self.get_api_credentials(api_key=api_key, api_base=api_base)

        default_headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }

        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"

        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Union[OptionalRerankParams, dict],
        headers: dict,
    ) -> dict:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for HuggingFace rerank")
        if "texts" not in optional_rerank_params:
            raise ValueError(
                "Cohere 'documents' param is required for HuggingFace rerank"
            )
        # Ensure return_text is a boolean value
        # HuggingFace API expects return_text parameter, corresponding to our return_documents parameter
        request_body = {
            "raw_scores": False,
            "truncate": False,
            "truncation_direction": "Right",
        }

        request_body.update(optional_rerank_params)

        return request_body

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LoggingClass,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        try:
            raw_response_json: HuggingFaceRerankResponseList = raw_response.json()
        except Exception:
            raise HuggingFaceError(
                message=getattr(raw_response, "text", str(raw_response)),
                status_code=getattr(raw_response, "status_code", 500),
            )

        # Use standard litellm token counter for proper token estimation
        input_text = request_data.get("query", "")
        try:
            # Calculate tokens for the raw response JSON string
            response_text = str(raw_response_json)
            estimated_output_tokens = token_counter(model=model, text=response_text)

            # Calculate input tokens from query and documents
            query = request_data.get("query", "")
            documents = request_data.get("texts", [])

            # Convert documents to string if they're not already
            documents_text = ""
            for doc in documents:
                if isinstance(doc, str):
                    documents_text += doc + " "
                elif isinstance(doc, dict) and "text" in doc:
                    documents_text += doc["text"] + " "

            # Calculate input tokens using the same model
            input_text = query + " " + documents_text
            estimated_input_tokens = token_counter(model=model, text=input_text)
        except Exception:
            # Fallback to reasonable estimates if token counting fails
            estimated_output_tokens = (
                len(raw_response_json) * 10 if raw_response_json else 10
            )
            estimated_input_tokens = (
                len(input_text) * 4 if "input_text" in locals() else 0
            )

        _billed_units = RerankBilledUnits(search_units=1)
        _tokens = RerankTokens(
            input_tokens=estimated_input_tokens, output_tokens=estimated_output_tokens
        )
        rerank_meta = RerankResponseMeta(
            api_version={"version": "1.0"}, billed_units=_billed_units, tokens=_tokens
        )

        # Check if documents should be returned based on request parameters
        should_return_documents = request_data.get(
            "return_text", False
        ) or request_data.get("return_documents", False)
        original_documents = request_data.get("texts", [])

        results = []
        for item in raw_response_json:
            # Extract required fields with defaults to handle None values
            index = item.get("index")
            score = item.get("score")

            # Skip items that don't have required fields
            if index is None or score is None:
                continue

            # Create RerankResponseResult with required fields
            result = RerankResponseResult(index=index, relevance_score=score)

            # Add optional document field if needed
            if should_return_documents:
                text_content = item.get("text", "")

                # 1. First try to use text returned directly from API if available
                if text_content:
                    result["document"] = RerankResponseDocument(text=text_content)
                # 2. If no text in API response but original documents are available, use those
                elif original_documents and 0 <= item.get("index", -1) < len(
                    original_documents
                ):
                    doc = original_documents[item.get("index")]
                    if isinstance(doc, str):
                        result["document"] = RerankResponseDocument(text=doc)
                    elif isinstance(doc, dict) and "text" in doc:
                        result["document"] = RerankResponseDocument(text=doc["text"])

            results.append(result)

        return RerankResponse(
            id=str(uuid.uuid4()),
            results=results,
            meta=rerank_meta,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return HuggingFaceError(message=error_message, status_code=status_code)

    def get_api_credentials(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API key and base URL from multiple sources.
        Returns tuple of (api_key, api_base).

        Parameters:
            api_key: API key provided directly to this function, takes precedence over all other sources
            api_base: API base provided directly to this function, takes precedence over all other sources
        """
        # Get API key from multiple sources
        final_api_key = (
            api_key or litellm.huggingface_key or get_secret_str("HUGGINGFACE_API_KEY")
        )

        # Get API base from multiple sources
        final_api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("HF_API_BASE")
            or get_secret_str("HUGGINGFACE_API_BASE")
        )

        return final_api_key, final_api_base
