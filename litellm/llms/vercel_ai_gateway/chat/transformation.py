"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as Vercel AI Gateway is openai-compatible.

Docs: https://vercel.com/docs/ai-gateway
"""

from collections.abc import Mapping
from typing import Final

import httpx

import litellm
from litellm.litellm_core_utils.gateway_catalog_cache import (
    CATALOG_TIMEOUT_SECONDS,
    as_mapping,
    as_sequence,
    float_field,
    freeze_catalog,
    int_field,
    prefix_model_ids,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelInfoBase

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

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("VERCEL_AI_GATEWAY_API_KEY") or get_secret_str("VERCEL_OIDC_TOKEN")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("VERCEL_AI_GATEWAY_API_BASE") or "https://ai-gateway.vercel.sh/v1"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self.get_api_base(api_base), self.get_api_key(api_key)

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
        resolved_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)
        if resolved_base is None:
            resolved_base = "https://ai-gateway.vercel.sh/v1"
        response: Final = litellm.module_level_client.get(
            url=f"{resolved_base}/models",
            timeout=CATALOG_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")

        models: Final = response.json()["data"]
        return prefix_model_ids("vercel_ai_gateway", (model["id"] for model in models))

    def get_models_with_info(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> Mapping[str, ModelInfoBase] | None:
        """
        Fetch Vercel AI Gateway's public catalog with pricing and capabilities.
        """
        resolved_base, _ = self._get_openai_compatible_provider_info(api_base, api_key)
        if resolved_base is None:
            resolved_base = "https://ai-gateway.vercel.sh/v1"
        response: Final = litellm.module_level_client.get(
            url=f"{resolved_base}/models",
            timeout=CATALOG_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")

        return freeze_catalog(
            pair
            for item in as_sequence(as_mapping(response.json()).get("data"))
            if (pair := _vercel_catalog_entry(as_mapping(item))) is not None
        )


def _vercel_catalog_entry(item: Mapping[str, object]) -> tuple[str, ModelInfoBase] | None:
    """One catalog model as ``(bare model id, model info)``, or None if not chat/embedding."""
    model_id: Final = item.get("id")
    item_type: Final = item.get("type")
    if not isinstance(model_id, str) or not model_id or item_type not in ("language", "embedding"):
        return None

    pricing: Final = as_mapping(item.get("pricing"))
    input_modalities: Final = as_sequence(as_mapping(item.get("modalities")).get("input"))
    tags: Final = as_sequence(item.get("tags"))
    context_window: Final = int_field(item, "context_window")

    entry: Final[ModelInfoBase] = {
        "key": f"vercel_ai_gateway/{model_id}",
        "litellm_provider": "vercel_ai_gateway",
        "mode": "chat" if item_type == "language" else "embedding",
        "max_tokens": context_window,
        "max_input_tokens": context_window,
        "max_output_tokens": int_field(item, "max_tokens") if item_type == "language" else None,
        "input_cost_per_token": float_field(pricing, "input"),
        "output_cost_per_token": float_field(pricing, "output"),
        "cache_read_input_token_cost": float_field(pricing, "input_cache_read"),
        "cache_creation_input_token_cost": float_field(pricing, "input_cache_write"),
        "supports_vision": "image" in input_modalities,
        "supports_reasoning": "reasoning" in tags,
    }
    return model_id, entry
