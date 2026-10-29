from collections.abc import Mapping
from typing import Final

from litellm.llms.zai.chat.transformation import ZAIChatConfig

from ..common_utils import get_bigmodel_api_key, get_bigmodel_url, is_unsupported_disabled_thinking

_SUPPORTED_PARAMS: Final = (
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
    "response_format",
    "thinking",
    "reasoning_effort",
)


class BigModelChatConfig(ZAIChatConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "bigmodel"

    def get_openai_compatible_provider_info(self, api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
        return get_bigmodel_url(api_base, "chat"), get_bigmodel_api_key(api_key)

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited interface requires a list
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: request pipeline adds parameters to the returned body
        mapped: Final = {
            **optional_params,
            **{
                ("max_tokens" if key == "max_completion_tokens" else key): value
                for key, value in non_default_params.items()
                if key in _SUPPORTED_PARAMS and key != "thinking"
            },
        }
        if "thinking" not in non_default_params:
            return mapped
        thinking: Final = non_default_params["thinking"]
        if is_unsupported_disabled_thinking(model, thinking):
            return {
                **mapped,
                "reasoning_effort": mapped.get("reasoning_effort", "low"),
                "extra_body": {"thinking": {"type": "enabled"}},
            }
        return {**mapped, "extra_body": {"thinking": thinking}}
