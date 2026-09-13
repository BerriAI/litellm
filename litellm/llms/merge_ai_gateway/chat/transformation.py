"""
Support for OpenAI's `/v1/chat/completions` endpoint via Merge AI Gateway.

Calls done in OpenAI/openai.py as Merge AI Gateway is openai-compatible.

Docs: https://docs.merge.dev/
"""

from collections.abc import Iterator, Mapping
from typing import Final

import httpx

import litellm
from litellm.litellm_core_utils.gateway_catalog_cache import (
    CATALOG_TIMEOUT_SECONDS,
    EMPTY_MAPPING,
    as_mapping,
    as_sequence,
    bearer_auth_headers,
    bool_field,
    float_field,
    freeze_catalog,
    int_field,
    optional_float,
    page_query_params,
    per_token,
    prefix_model_ids,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ModelInfoBase

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import MergeAIGatewayException

DEFAULT_API_BASE: Final = "https://api-gateway.merge.dev/v1/openai"
CATALOG_ROOT: Final = "https://api-gateway.merge.dev/v1"
PAGE_LIMIT: Final = 500
MAX_PAGES: Final = 20
_FALLBACK_COST_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)


class MergeAIGatewayConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "merge_ai_gateway"

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("MERGE_AI_GATEWAY_API_KEY") or get_secret_str("MERGE_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("MERGE_AI_GATEWAY_API_BASE") or DEFAULT_API_BASE

    @staticmethod
    def get_catalog_root(api_base: str | None = None) -> str:
        """The native catalog lives beside the OpenAI-shaped surface, not under it.

        A deployment points at ``{root}/openai``; the catalog is ``{root}/models``.
        """
        resolved: Final = MergeAIGatewayConfig.get_api_base(api_base) or DEFAULT_API_BASE
        return resolved.split("/v1")[0] + "/v1"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self.get_api_base(api_base), self.get_api_key(api_key)

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: inherited get_models list contract
        return prefix_model_ids(
            "merge_ai_gateway",
            super().get_models(api_key=api_key, api_base=self.get_api_base(api_base)),
        )

    def get_models_with_info(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> Mapping[str, ModelInfoBase] | None:
        """
        Fetch Merge AI Gateway's native catalog with per-vendor pricing and
        capabilities, in ModelInfoBase field names.
        """
        items: Final = self._fetch_catalog_items(
            root=self.get_catalog_root(api_base),
            api_key=self.get_api_key(api_key),
        )
        return freeze_catalog(pair for item in items if (pair := _merge_catalog_entry(as_mapping(item))) is not None)

    def _fetch_catalog_items(
        self,
        *,
        root: str,
        api_key: str | None,
    ) -> tuple[Mapping[str, object], ...]:
        """Every catalog page's items, following ``next_cursor`` up to ``MAX_PAGES``."""
        return tuple(self._iter_catalog_items(root=root, api_key=api_key))

    @staticmethod
    def _iter_catalog_items(
        *,
        root: str,
        api_key: str | None,
    ) -> Iterator[Mapping[str, object]]:
        cursor: str | None = None  # rebind-ok: pagination cursor
        for _ in range(MAX_PAGES):
            response = litellm.module_level_client.get(
                url=f"{root}/models",
                params=page_query_params(cursor, PAGE_LIMIT),
                headers=bearer_auth_headers(api_key),
                timeout=CATALOG_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                raise Exception(f"Failed to get models: {response.text}")

            page = as_mapping(response.json())
            for item in as_sequence(page.get("data")):
                yield as_mapping(item)
            next_cursor = page.get("next_cursor")
            if not page.get("has_more") or not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor  # rebind-ok: advance pagination cursor

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict | httpx.Headers
    ) -> BaseLLMException:  # mutable-ok: BaseLLMException contract
        return MergeAIGatewayException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )


def _merge_catalog_entry(item: Mapping[str, object]) -> tuple[str, ModelInfoBase] | None:
    """One catalog model as ``(bare model id, model info)``, or None if it has no chat route."""
    model_id: Final = item.get("model")
    if not isinstance(model_id, str) or not model_id:
        return None

    selected: Final = _cheapest_available_vendor(as_mapping(item.get("vendors")))
    if selected is None:
        return None
    vendor_info: Final = selected[1]

    capabilities: Final = as_mapping(vendor_info.get("capabilities"))
    outputs: Final = as_sequence(capabilities.get("output"))
    if "text" not in outputs and "tool_use" not in outputs:
        return None  # embedding-only route

    inputs: Final = as_sequence(capabilities.get("input"))
    context_window: Final = int_field(vendor_info, "context_window")
    pricing: Final = as_mapping(vendor_info.get("pricing"))
    flex: Final = as_mapping(pricing.get("flex"))

    input_cost: Final = optional_float(pricing.get("input_per_million"))
    output_cost: Final = optional_float(pricing.get("output_per_million"))
    fallback: Final = _cost_map_fallback(model_id) if input_cost is None or output_cost is None else EMPTY_MAPPING

    entry: Final[ModelInfoBase] = {
        "key": f"merge_ai_gateway/{model_id}",
        "litellm_provider": "merge_ai_gateway",
        "mode": "chat",
        "max_tokens": context_window,
        "max_input_tokens": context_window,
        "max_output_tokens": int_field(vendor_info, "max_output_tokens"),
        "input_cost_per_token": _cost_or_fallback(per_token(input_cost), fallback, "input_cost_per_token"),
        "output_cost_per_token": _cost_or_fallback(per_token(output_cost), fallback, "output_cost_per_token"),
        "cache_read_input_token_cost": float_field(fallback, "cache_read_input_token_cost"),
        "cache_creation_input_token_cost": float_field(fallback, "cache_creation_input_token_cost"),
        "input_cost_per_token_flex": per_token(optional_float(flex.get("input_per_million"))),
        "output_cost_per_token_flex": per_token(optional_float(flex.get("output_per_million"))),
        "supports_vision": "image" in inputs,
        "supports_pdf_input": "document" in inputs,
        "supports_function_calling": bool_field(capabilities, "supports_tool_calling"),
        "supports_tool_choice": bool_field(capabilities, "supports_tool_choice"),
        "supports_response_schema": bool_field(capabilities, "supports_structured_outputs"),
        "supports_native_streaming": bool_field(capabilities, "streaming"),
    }
    return model_id, entry


def _cost_or_fallback(cost: float | None, fallback: Mapping[str, object], field: str) -> float | None:
    return cost if cost is not None else float_field(fallback, field)


def _cost_map_fallback(model_id: str) -> Mapping[str, object]:
    """Merge publishes no price for every route; borrow the model's public rate.

    OpenRouter's entry wins over the bare id because it is the same
    provider-agnostic catalog of the model's list price.
    """
    return as_mapping(litellm.model_cost.get(f"openrouter/{model_id}") or litellm.model_cost.get(model_id))


def _cheapest_available_vendor(vendors: Mapping[str, object]) -> tuple[str, Mapping[str, object]] | None:
    """The lowest-input-cost vendor that is marked available, ties broken by output cost then name."""
    available: Final = tuple(
        (name, as_mapping(info))
        for name, info in vendors.items()
        if as_mapping(info).get("availability_status") == "available"
    )
    if not available:
        return None
    return min(
        available,
        key=lambda vendor: (
            _per_million(vendor[1], "input_per_million"),
            _per_million(vendor[1], "output_per_million"),
            vendor[0],
        ),
    )


def _per_million(vendor_info: Mapping[str, object], field: str) -> float:
    resolved: Final = optional_float(as_mapping(vendor_info.get("pricing")).get(field))
    return float("inf") if resolved is None else resolved
