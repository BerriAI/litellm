"""
Translates from OpenAI's `/v1/chat/completions` to Tencent TokenHub's
OpenAI-compatible endpoint.
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str
from litellm.utils import supports_reasoning

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class TencentChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        params: Final = super().get_supported_openai_params(model)
        if supports_reasoning(model, custom_llm_provider="tencent"):
            params.extend(["thinking", "reasoning_effort"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)

        thinking_value: Final = optional_params.pop("thinking", None)
        reasoning_effort: Final = optional_params.pop("reasoning_effort", None)

        thinking: dict | None = None
        if isinstance(thinking_value, dict):
            thinking = thinking_value
        elif reasoning_effort is not None:
            # TokenHub recommends explicitly disabling thinking instead of
            # relying on per-model defaults (deepseek-v4-* default to enabled).
            thinking = {"type": "disabled" if reasoning_effort == "none" else "enabled"}

        if thinking is not None:
            thinking = self._normalize_thinking_type_for_model(model=model, thinking=thinking)
            # Tencent TokenHub expects `thinking` in the request JSON body, but
            # the OpenAI SDK's chat.completions.create() rejects unknown
            # top-level kwargs. Route it through `extra_body` so it is merged
            # into the payload instead of passed as a keyword argument.
            extra_body: Final = optional_params.setdefault("extra_body", {})
            extra_body["thinking"] = thinking

        return optional_params

    @staticmethod
    def _normalize_thinking_type_for_model(model: str, thinking: dict) -> dict:
        """Coerce `thinking.type` values the model does not accept.

        MiniMax models on TokenHub only accept "adaptive" or "disabled" —
        sending "enabled" returns a 400. "adaptive" is the closest semantic
        (the model decides when to think), so "enabled" is coerced to it.
        Ref: https://www.tencentcloud.com/document/product/1300/82345
        """
        if thinking.get("type") == "enabled" and "minimax" in model.lower():
            return {**thinking, "type": "adaptive"}
        return thinking

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("TENCENT_API_BASE") or "https://tokenhub-intl.tencentcloudmaas.com/v1"
        dynamic_api_key: Final = api_key or get_secret_str("TENCENT_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            api_base = "https://tokenhub-intl.tencentcloudmaas.com/v1"

        api_base = api_base.rstrip("/")

        if api_base.endswith("/chat/completions"):
            return api_base

        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

        return f"{api_base}/chat/completions"
