"""
Translates from OpenAI's `/v1/chat/completions` to Tencent TokenHub's
OpenAI-compatible endpoint.
"""

from typing import Optional

from litellm.secret_managers.main import get_secret_str
from litellm.utils import supports_reasoning

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class TencentChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        params = super().get_supported_openai_params(model)
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

        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        if thinking_value is not None:
            if isinstance(thinking_value, dict):
                optional_params["thinking"] = thinking_value
        elif reasoning_effort is not None and reasoning_effort != "none":
            optional_params["thinking"] = {"type": "enabled"}

        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("TENCENT_API_BASE") or "https://tokenhub-intl.tencentcloudmaas.com/v1"
        dynamic_api_key = api_key or get_secret_str("TENCENT_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            api_base = "https://tokenhub-intl.tencentcloudmaas.com/v1"

        api_base = api_base.rstrip("/")

        if api_base.endswith("/chat/completions"):
            return api_base

        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

        return f"{api_base}/chat/completions"
