"""
DashScope (Alibaba Bailian) Anthropic-compatible Messages API adapter.

Bailian exposes an Anthropic-compatible Messages endpoint at
https://dashscope.aliyuncs.com/apps/anthropic/v1/messages, supporting
deepseek-v4-pro/flash, qwen3.7-max/plus, glm-5.2, kimi-k2.6, minimax-m2.5, etc.

Without this adapter, litellm falls back to
LiteLLMMessagesToCompletionTransformationHandler, whose
_add_cache_control_if_applicable() (adapters/transformation.py:302) strips
cache_control for any non-Claude/non-Bedrock model. That silently drops
Claude-Code cache markers and prevents Bailian from recognizing explicit
prompt caching on deepseek-v4-flash.

Docs: https://www.alibabacloud.com/help/zh/model-studio/anthropic-api-messages
"""

from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams


class DashScopeAnthropicMessagesConfig(AnthropicMessagesConfig):
    """Adapter for Alibaba Bailian's Anthropic-compatible Messages endpoint."""

    # Models Bailian explicitly documents on the Anthropic-compatible endpoint.
    # Extend this set when Bailian onboards more models.
    SUPPORTED_MODELS = frozenset({
        # DeepSeek
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash-0731",
        # Qwen
        #"qwen3.7-max", "qwen3.7-plus",
        #"qwen3.6-flash", "qwen3.6-plus",
        #"qwen-turbo", "qwen-plus", "qwen-max",
    })

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "dashscope"

    @staticmethod
    def is_supported_model(model: str) -> bool:
        stripped = model.split("/", 1)[-1] if "/" in model else model
        return stripped in DashScopeAnthropicMessagesConfig.SUPPORTED_MODELS

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return (
            api_key
            or get_secret_str("DASHSCOPE_API_KEY")
            or get_secret_str("BAILIAN_API_KEY")
            or litellm.api_key
        )

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        base = (
            api_base
            or get_secret_str("DASHSCOPE_ANTHROPIC_API_BASE")
            or get_secret_str("DASHSCOPE_API_BASE")
            or "https://dashscope.aliyuncs.com"
        ).rstrip("/")
        # Bailian docs warn: base_url must NOT end with /v1, otherwise the
        # final path becomes /v1/v1/messages and returns 404.
        if base.endswith("/v1"):
            base = base[:-3]
        return base

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        dynamic_api_key = self.get_api_key(api_key=api_key)

        # Bailian accepts either "x-api-key" or "Authorization: Bearer"
        if (
            "x-api-key" not in headers
            and "authorization" not in headers
            and dynamic_api_key is not None
        ):
            headers["Authorization"] = f"Bearer {dynamic_api_key}"

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = self.get_api_base(api_base=api_base)
        return f"{base}/apps/anthropic/v1/messages"

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        req = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        # Strip provider prefix; Bailian expects bare model name.
        if isinstance(req.get("model"), str) and "/" in req["model"]:
            req["model"] = req["model"].split("/", 1)[-1]
        return req
