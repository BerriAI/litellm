"""
Translate between Cohere's `/rerank` format and Azure AI's `/rerank` format. 
"""

from typing import Optional

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.cohere.rerank.transformation import CohereRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import RerankResponse


class AzureAIRerankConfig(CohereRerankConfig):
    """
    Azure AI Rerank - Follows the same Spec as Cohere Rerank
    """

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        if api_base is None:
            raise ValueError(
                "Azure AI API Base is required. api_base=None. Set in call or via `AZURE_AI_API_BASE` env var."
            )
        if not api_base.endswith("/v1/rerank"):
            api_base = f"{api_base}/v1/rerank"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("AZURE_AI_API_KEY") or litellm.azure_key

        if api_key is None:
            raise ValueError(
                "Azure AI API key is required. Please set 'AZURE_AI_API_KEY' or 'litellm.azure_key'"
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
        rerank_response = super().transform_rerank_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        base_model = self._get_base_model(
            rerank_response._hidden_params.get("llm_provider-azureml-model-group")
        )
        rerank_response._hidden_params["model"] = base_model
        return rerank_response

    def _get_base_model(self, azure_model_group: Optional[str]) -> Optional[str]:
        if azure_model_group is None:
            return None
        if azure_model_group == "offer-cohere-rerank-mul-paygo":
            return "azure_ai/cohere-rerank-v3-multilingual"
        if azure_model_group == "offer-cohere-rerank-eng-paygo":
            return "azure_ai/cohere-rerank-v3-english"
        return azure_model_group
