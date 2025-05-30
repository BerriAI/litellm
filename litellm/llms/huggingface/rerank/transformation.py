from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)

from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from ..common_utils import HuggingFaceError


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class HuggingFaceRerankConfig(BaseRerankConfig):
    """
    Reference: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    """
    def __init__(self) -> None:
        pass

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        # HuggingFace rerank endpoint, e.g. https://api.end.point/rerank
        if api_base:
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/rerank"):
                api_base = f"{api_base}/rerank"
            return api_base
        return "https://api-inference.huggingface.co/rerank"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "rank_fields",
            "return_documents",
            "max_chunks_per_doc",
            "max_tokens_per_doc",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: Optional[dict],  # unused: HuggingFace doesn't need additional params
        model: str,  # unused: HuggingFace transformation doesn't depend on model name
        drop_params: bool,  # unused: HuggingFace doesn't drop params
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,  # unused: provider is already known
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
    ) -> OptionalRerankParams:
        return OptionalRerankParams(
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            max_tokens_per_doc=max_tokens_per_doc,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("HUGGINGFACE_API_KEY")
        
        default_headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        
        # Only add Authorization header if API key exists
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]
        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: OptionalRerankParams,
        headers: dict,
    ) -> dict:
        # HuggingFace expects: query, texts (list), raw_scores, return_text, truncate, truncation_direction
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for HuggingFace rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for HuggingFace rerank")
        request_body = {
            "query": optional_rerank_params["query"],
            "texts": optional_rerank_params["documents"],
            "raw_scores": False,
            "return_text": True,
            "truncate": False,
            "truncation_direction": "Right",  # Use uppercase "Right"
        }
        # Add top_n, rank_fields and other parameters as needed
        if optional_rerank_params.get("top_n") is not None:
            request_body["top_n"] = optional_rerank_params["top_n"]
        return request_body

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text
        Using improved estimation method:
        - Chinese characters: approximately 1 token per character
        - English words: approximately 1.3 tokens per word on average
        - Punctuation: approximately 0.5 tokens per symbol
        """
        if not text:
            return 0
        
        # Check for Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        
        # Calculate number of English words (more accurate method)
        import re
        english_text = re.sub(r'[\u4e00-\u9fff]', '', text)  # Remove Chinese characters
        english_words = len(re.findall(r'\b\w+\b', english_text))
        
        # Calculate number of punctuation marks
        punctuation_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
        
        # Estimate token count
        estimated_tokens = (
            chinese_chars +  # Chinese: 1 character ≈ 1 token
            int(english_words * 1.3) +  # English: 1 word ≈ 1.3 tokens
            int(punctuation_chars * 0.5)  # Punctuation: 1 symbol ≈ 0.5 tokens
        )
        
        return max(1, estimated_tokens)  # Return at least 1
    
    def _calculate_total_input_tokens(self, request_data: dict) -> Optional[int]:
        """
        Calculate the estimated total token count for all text in the request
        """
        total_tokens = 0
        
        # Calculate tokens for query
        query = request_data.get("query", "")
        if query:
            total_tokens += self._estimate_tokens(query)
        
        # Calculate tokens for all documents
        documents = request_data.get("documents", [])
        for doc in documents:
            if isinstance(doc, str):
                total_tokens += self._estimate_tokens(doc)
            elif isinstance(doc, dict) and "text" in doc:
                total_tokens += self._estimate_tokens(doc["text"])
        
        return total_tokens if total_tokens > 0 else None

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
            raw_response_json = raw_response.json()
        except Exception:
            raise HuggingFaceError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        
        # Calculate estimated token count
        estimated_input_tokens = self._calculate_total_input_tokens(request_data)
        
        # Estimate output tokens for rerank task - more precise calculation method
        # Rerank task output mainly includes:
        # 1. Index for each result (usually 1 token)
        # 2. Relevance score for each result (usually 1 token)  
        # 3. Structured output overhead (JSON format etc.)
        base_output_tokens = len(raw_response_json) * 2 if raw_response_json else 0  # index + score
        structure_overhead = 2  # JSON structure overhead
        estimated_output_tokens = base_output_tokens + structure_overhead
        
        # Build billing and token information
        # HuggingFace API usually doesn't return detailed usage info, so we provide reasonable estimates
        total_estimated_tokens = (estimated_input_tokens or 0) + estimated_output_tokens
        
        _billed_units = RerankBilledUnits(
            search_units=1,  # Default to 1 search unit
            total_tokens=total_estimated_tokens  # Use estimated total token count
        )
        
        _tokens = RerankTokens(
            input_tokens=estimated_input_tokens,  # Use estimated input token count
            output_tokens=estimated_output_tokens  # Estimated processing cost for rerank task
        )
        
        rerank_meta = RerankResponseMeta(
            api_version=None,
            billed_units=_billed_units,
            tokens=_tokens
        )
        
        # Parse HuggingFace response and convert to litellm RerankResponse format
        results = []
        for item in raw_response_json:
            # Create correct RerankResponseResult object
            # Ensure document field is always of RerankResponseDocument type
            document = RerankResponseDocument(text=item.get("text", ""))
            result = RerankResponseResult(
                index=item.get("index"),
                relevance_score=item.get("score"),
                document=document
            )
            results.append(result)
            
        return RerankResponse(
            id=None,
            results=results,
            meta=rerank_meta,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return HuggingFaceError(message=error_message, status_code=status_code)