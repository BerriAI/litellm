"""
Perplexity Responses API — OpenAI-compatible.

The only provider quirks:
- cost returned as dict → handled by ResponseAPIUsage.parse_cost validator
- preset models (preset/pro-search) → handled by transform_responses_api_request

Ref: https://docs.perplexity.ai/api-reference/responses-post
"""

from typing import Dict, Optional, Union

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class PerplexityResponsesConfig(OpenAIResponsesAPIConfig):

    def get_supported_openai_params(self, model: str) -> list:
        """Ref: https://docs.perplexity.ai/api-reference/responses-post"""
        return [
            "max_output_tokens",
            "stream",
            "temperature",
            "top_p",
            "tools",
            "reasoning",
            "instructions",
        ]

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.PERPLEXITY

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("PERPLEXITYAI_API_KEY")
            or get_secret_str("PERPLEXITY_API_KEY")
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(self, api_base: Optional[str], litellm_params: dict) -> str:
        api_base = api_base or get_secret_str("PERPLEXITY_API_BASE") or "https://api.perplexity.ai"
        return f"{api_base.rstrip('/')}/v1/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """Handle preset/ model prefix: send as {"preset": name} instead of {"model": name}."""
        if model.startswith("preset/"):
            input = self._validate_input_param(input)
            data: Dict = {
                "preset": model[len("preset/"):],
                "input": input,
            }
            data.update(response_api_optional_request_params)
            return data
        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
