from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.types.rerank import OptionalRerankParams, RerankResponse
from litellm.types.utils import ModelInfo

from ..chat.transformation import BaseLLMException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseRerankConfig(ABC):
    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        pass

    @abstractmethod
    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: OptionalRerankParams,
        headers: dict,
    ) -> dict:
        return {}

    @abstractmethod
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
        return model_response

    @abstractmethod
    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    @abstractmethod
    def get_supported_cohere_rerank_params(self, model: str) -> list:
        pass

    @abstractmethod
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
    ) -> OptionalRerankParams:
        pass

    @abstractmethod
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        pass

    def calculate_total_queries(
        self, query_tokens: int, document_tokens: list[int]
    ) -> int:
        """
        Calculate the cost of a request based on token counts and document chunks.

        Args:
            query_tokens (int): Number of tokens in the query
            document_tokens (list[int]): List of token counts for each document
            cost_per_1000_queries (float): Cost per 1000 queries in dollars (default: $1.00)

        Returns:
            float: Total cost in dollars
        """
        TOKENS_PER_DOCUMENT = 512
        CHUNKS_PER_QUERY = 100

        # Validate query length
        if query_tokens >= TOKENS_PER_DOCUMENT:
            raise ValueError("Query tokens exceed maximum allowed tokens per document")

        # Calculate total chunks needed
        total_chunks = 0
        available_tokens = TOKENS_PER_DOCUMENT - query_tokens

        for doc_tokens in document_tokens:
            # Calculate chunks needed for this document
            chunks_needed = (doc_tokens + available_tokens - 1) // available_tokens
            total_chunks += max(1, chunks_needed)

        # Calculate total queries needed (rounded up to nearest multiple of CHUNKS_PER_QUERY)
        total_queries = (total_chunks + CHUNKS_PER_QUERY - 1) // CHUNKS_PER_QUERY

        return total_queries

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: Optional[str] = None,
        num_queries: int = 1,
        model_info: Optional[ModelInfo] = None,
    ) -> Tuple[float, float]:
        """
        Calculates the cost per query for a given rerank model.

        Input:
            - model: str, the model name without provider prefix
            - custom_llm_provider: str, the provider used for the model. If provided, used to check if the litellm model info is for that provider.
            - num_queries: int, the number of queries to calculate the cost for
            - model_info: ModelInfo, the model info for the given model

        Returns:
            Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
        """

        if (
            model_info is None
            or "input_cost_per_query" not in model_info
            or model_info["input_cost_per_query"] is None
        ):
            return 0.0, 0.0

        prompt_cost = model_info["input_cost_per_query"] * num_queries

        return prompt_cost, 0.0
