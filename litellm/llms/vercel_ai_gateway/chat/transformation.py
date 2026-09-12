"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as Vercel AI Gateway is openai-compatible.

Docs: https://vercel.com/docs/ai-gateway
"""

from typing import Final

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import VercelAIGatewayException


class VercelAIGatewayConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "vercel_ai_gateway"

    def get_supported_openai_params(self, model: str) -> list:
        base_params: Final = super().get_supported_openai_params(model)
        if "extra_body" not in base_params:
            base_params.append("extra_body")
        return base_params

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("VERCEL_AI_GATEWAY_API_BASE") or "https://ai-gateway.vercel.sh/v1"
        user_api_key = api_key or get_secret_str("VERCEL_AI_GATEWAY_API_KEY") or get_secret_str("VERCEL_OIDC_TOKEN")
        return api_base, user_api_key

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)

        # Vercel AI Gateway-only parameters
        extra_body: Final = {}
        provider_options: Final = non_default_params.pop("providerOptions", None)

        if provider_options is not None:
            extra_body["providerOptions"] = provider_options

        mapped_openai_params["extra_body"] = extra_body  # openai client supports `extra_body` param
        return mapped_openai_params

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        return super().transform_request(model, messages, optional_params, litellm_params, headers)

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return VercelAIGatewayException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        api_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)

        if api_base is None:
            api_base = "https://ai-gateway.vercel.sh/v1"

        models_url: Final = f"{api_base}/models"
        response: Final = litellm.module_level_client.get(url=models_url)

        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")

        models: Final = response.json()["data"]
        return [model["id"] for model in models]

    def get_models_with_info(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[dict] | None:
        """
        Fetch Vercel AI Gateway's public catalog with pricing and capabilities.
        """
        from litellm.litellm_core_utils.gateway_catalog_cache import optional_float

        resolved_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)
        if resolved_base is None:
            resolved_base = "https://ai-gateway.vercel.sh/v1"

        response: Final = litellm.module_level_client.get(url=f"{resolved_base}/models")
        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")

        entries: Final[list[dict]] = []
        for item in response.json().get("data", []):
            model_id: Final = item.get("id")
            item_type: Final = item.get("type")
            if not model_id or item_type not in ("language", "embedding"):
                continue

            pricing: Final = item.get("pricing") or {}
            modalities: Final = item.get("modalities") or {}
            input_modalities: Final = modalities.get("input") or []
            tags: Final = item.get("tags") or []
            context_window: Final = item.get("context_window")

            entries.append(
                {
                    "key": f"vercel_ai_gateway/{model_id}",
                    "litellm_provider": "vercel_ai_gateway",
                    "mode": "chat" if item_type == "language" else "embedding",
                    "max_tokens": context_window,
                    "max_input_tokens": context_window,
                    "max_output_tokens": item.get("max_tokens") if item_type == "language" else None,
                    "input_cost_per_token": optional_float(pricing.get("input")),
                    "output_cost_per_token": optional_float(pricing.get("output")),
                    "cache_read_input_token_cost": optional_float(pricing.get("input_cache_read")),
                    "cache_creation_input_token_cost": optional_float(pricing.get("input_cache_write")),
                    "supports_vision": "image" in input_modalities,
                    "supports_reasoning": "reasoning" in tags,
                }
            )
        return entries
