"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as Vercel AI Gateway is openai-compatible.

Docs: https://vercel.com/docs/ai-gateway
"""

from typing import List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.secret_managers.main import get_secret_str
import litellm

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import VercelAIGatewayException


class VercelAIGatewayConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "vercel_ai_gateway"

    def get_supported_openai_params(self, model: str) -> list:
        base_params = super().get_supported_openai_params(model)
        if "extra_body" not in base_params:
            base_params.append("extra_body")
        return base_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:

        api_base = (
            api_base
            or get_secret_str("VERCEL_AI_GATEWAY_API_BASE")
            or "https://ai-gateway.vercel.sh/v1"
        )
        user_api_key = (
            api_key 
            or get_secret_str("VERCEL_AI_GATEWAY_API_KEY")
            or get_secret_str("VERCEL_OIDC_TOKEN")
        )
        return api_base, user_api_key

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Vercel AI Gateway-only parameters
        extra_body = {}
        provider_options = non_default_params.pop("providerOptions", None)
        
        if provider_options is not None:
            extra_body["providerOptions"] = provider_options
        
        mapped_openai_params["extra_body"] = extra_body  # openai client supports `extra_body` param
        return mapped_openai_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        return super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return VercelAIGatewayException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)
        
        if api_base is None:
            api_base = "https://ai-gateway.vercel.sh/v1"
            
        models_url = f"{api_base}/models"
        response = litellm.module_level_client.get(url=models_url)

        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")

        models = response.json()["data"]
        return [model["id"] for model in models]
