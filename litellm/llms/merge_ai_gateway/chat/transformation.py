"""
Support for OpenAI's `/v1/chat/completions` endpoint via Merge AI Gateway.

Calls done in OpenAI/openai.py as Merge AI Gateway is openai-compatible.

Docs: https://docs.merge.dev/
"""

from typing import Final

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import MergeAIGatewayException

DEFAULT_API_BASE: Final = "https://api-gateway.merge.dev/v1/openai"


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

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self.get_api_base(api_base), self.get_api_key(api_key)

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return super().get_models(api_key=api_key, api_base=self.get_api_base(api_base))

    def get_models_with_info(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[dict] | None:
        """
        Fetch Merge AI Gateway's native catalog with per-vendor pricing and
        capabilities. Paginates ``GET {root}/models`` where ``root`` is the
        api_base with the ``/openai`` surface segment removed.
        """
        from litellm.litellm_core_utils.gateway_catalog_cache import optional_float

        resolved_key: Final = self.get_api_key(api_key)
        resolved_base: Final = self.get_api_base(api_base) or DEFAULT_API_BASE
        root: Final = resolved_base.split("/v1")[0] + "/v1"

        items: Final[list[dict]] = []
        cursor: str | None = None
        for _ in range(20):  # page cap; catalog is ~300 models at limit=500
            params: Final = {"limit": 500, **({"cursor": cursor} if cursor else {})}
            response: Final = litellm.module_level_client.get(
                url=f"{root}/models",
                params=params,
                headers={"Authorization": f"Bearer {resolved_key}"},
            )
            if response.status_code != 200:
                raise Exception(f"Failed to get models: {response.text}")
            page: Final = response.json()
            items.extend(page.get("data") or [])
            if not page.get("has_more"):
                break
            cursor = page.get("next_cursor")
            if not cursor:
                break

        entries: Final[list[dict]] = []
        for item in items:
            model_id: Final = item.get("model")
            vendors: Final = item.get("vendors") or {}
            if not model_id:
                continue

            available: Final = sorted(
                (
                    (name, info)
                    for name, info in vendors.items()
                    if info.get("availability_status") == "available"
                ),
                key=lambda pair: (
                    optional_float((pair[1].get("pricing") or {}).get("input_per_million")) or float("inf"),
                    optional_float((pair[1].get("pricing") or {}).get("output_per_million")) or float("inf"),
                    pair[0],
                ),
            )
            if not available:
                continue

            vendor_info: Final = available[0][1]
            capabilities: Final = vendor_info.get("capabilities") or {}
            outputs: Final = capabilities.get("output") or []
            if "text" not in outputs and "tool_use" not in outputs:
                continue  # embedding-only route

            pricing: Final = vendor_info.get("pricing") or {}
            flex: Final = pricing.get("flex") or {}
            inputs: Final = capabilities.get("input") or []
            context_window: Final = vendor_info.get("context_window")

            input_cost: Final = optional_float(pricing.get("input_per_million"))
            output_cost: Final = optional_float(pricing.get("output_per_million"))
            flex_input: Final = optional_float(flex.get("input_per_million"))
            flex_output: Final = optional_float(flex.get("output_per_million"))

            entry: dict = {  # mutable-ok: cost fallback fills fields after construction
                "key": f"merge_ai_gateway/{model_id}",
                "litellm_provider": "merge_ai_gateway",
                "mode": "chat",
                "max_tokens": context_window,
                "max_input_tokens": context_window,
                "max_output_tokens": vendor_info.get("max_output_tokens"),
                "input_cost_per_token": input_cost / 1e6 if input_cost is not None else None,
                "output_cost_per_token": output_cost / 1e6 if output_cost is not None else None,
                "input_cost_per_token_flex": flex_input / 1e6 if flex_input is not None else None,
                "output_cost_per_token_flex": flex_output / 1e6 if flex_output is not None else None,
                "supports_vision": "image" in inputs,
                "supports_pdf_input": "document" in inputs,
                "supports_function_calling": capabilities.get("supports_tool_calling"),
                "supports_tool_choice": capabilities.get("supports_tool_choice"),
                "supports_response_schema": capabilities.get("supports_structured_outputs"),
                "supports_native_streaming": capabilities.get("streaming"),
            }

            if entry["input_cost_per_token"] is None or entry["output_cost_per_token"] is None:
                fallback: Final = litellm.model_cost.get(f"openrouter/{model_id}") or litellm.model_cost.get(model_id)
                if fallback:
                    for cost_field in (
                        "input_cost_per_token",
                        "output_cost_per_token",
                        "cache_read_input_token_cost",
                        "cache_creation_input_token_cost",
                    ):
                        if entry.get(cost_field) is None and fallback.get(cost_field) is not None:
                            entry[cost_field] = fallback[cost_field]

            entries.append(entry)
        return entries

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return MergeAIGatewayException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
