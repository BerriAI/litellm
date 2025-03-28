from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.types.rerank import OptionalRerankParams, RerankBilledUnits, RerankResponse
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
        non_default_params: dict,
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
    ) -> OptionalRerankParams:
        pass

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: Optional[str] = None,
        billed_units: Optional[RerankBilledUnits] = None,
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
            or billed_units is None
        ):
            return 0.0, 0.0

        search_units = billed_units.get("search_units")

        if search_units is None:
            return 0.0, 0.0

        prompt_cost = model_info["input_cost_per_query"] * search_units

        return prompt_cost, 0.0
