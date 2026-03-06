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
from litellm.utils import _add_path_to_api_base


class AzureAIRerankConfig(CohereRerankConfig):
    """
    Azure AI Rerank - Follows the same Spec as Cohere Rerank
    """

    def get_complete_url(
        self, 
        api_base: Optional[str], 
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        if api_base is None:
            raise ValueError(
                "Azure AI API Base is required. api_base=None. Set in call or via `AZURE_AI_API_BASE` env var."
            )
        original_url = httpx.URL(api_base)
        if not original_url.is_absolute_url:
            raise ValueError(
                "Azure AI API Base must be an absolute URL including scheme (e.g. "
                "'https://<resource>.services.ai.azure.com'). "
                f"Got api_base={api_base!r}."
            )
        normalized_path = original_url.path.rstrip("/")

        # Allow callers to pass either full v1/v2 rerank endpoints:
        # - https://<resource>.services.ai.azure.com/v1/rerank
        # - https://<resource>.services.ai.azure.com/providers/cohere/v2/rerank
        if normalized_path.endswith("/v1/rerank") or normalized_path.endswith("/v2/rerank"):
            return str(original_url.copy_with(path=normalized_path or "/"))

        # If callers pass just the version path (e.g. ".../v2" or ".../providers/cohere/v2"), append "/rerank"
        if (
            normalized_path.endswith("/v1")
            or normalized_path.endswith("/v2")
            or normalized_path.endswith("/providers/cohere/v2")
        ):
            return _add_path_to_api_base(
                api_base=str(original_url.copy_with(path=normalized_path or "/")),
                ending_path="/rerank",
            )

        # Backwards compatible default: Azure AI rerank was originally exposed under /v1/rerank
        return _add_path_to_api_base(api_base=api_base, ending_path="/v1/rerank")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
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
