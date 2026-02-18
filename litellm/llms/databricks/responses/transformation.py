"""
Databricks Responses API configuration.

Inherits from OpenAIResponsesAPIConfig since Databricks' Responses API
is compatible with OpenAI's for GPT models.

Reference: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.llms.databricks.common_utils import DatabricksBase
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class DatabricksResponsesAPIConfig(DatabricksBase, OpenAIResponsesAPIConfig):
    """
    Configuration for Databricks Responses API.

    Inherits from OpenAIResponsesAPIConfig since Databricks' Responses API
    is largely compatible with OpenAI's for GPT models.

    Note: The Responses API on Databricks is only compatible with OpenAI GPT models.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.DATABRICKS

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or os.getenv("DATABRICKS_API_KEY")
        api_base = litellm_params.api_base or os.getenv("DATABRICKS_API_BASE")

        # Reuse Databricks auth logic (OAuth M2M, PAT, SDK fallback)
        api_base, headers = self.databricks_validate_environment(
            api_key=api_key,
            api_base=api_base,
            endpoint_type="chat_completions",  # reuse existing type, URL is built separately
            custom_endpoint=True,  # prevent appending /chat/completions to api_base
            headers=headers,
        )

        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or os.getenv("DATABRICKS_API_BASE")
        api_base = self._get_api_base(api_base)
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform request for Databricks Responses API.

        Strips the 'databricks/' prefix from model name if present,
        then delegates to OpenAI's transformation.
        """
        # Strip provider prefix if present (e.g., "databricks/databricks-gpt-5-nano" -> "databricks-gpt-5-nano")
        if model.startswith("databricks/"):
            model = model[len("databricks/") :]

        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
