"""
Apodex chat completions — OpenAI-compatible, with two provider quirks:

- `stream` defaults to true upstream, so a non-streaming call has to say so
  explicitly or Apodex answers with SSE that a plain call cannot parse
- the Deep Research tiers ignore sampling parameters and reject OpenAI-style
  tools; only the core models take them

Ref: https://platform.apodex.ai/docs/chat-completions
"""

from collections.abc import Mapping
from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

from ..common_utils import (
    APODEX_API_BASE_URL,
    get_apodex_api_base,
    get_apodex_api_key,
    is_deep_research_model,
)

_DEEP_RESEARCH_PARAMS: Final = (
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "extra_headers",
    "max_retries",
)

_CORE_PARAMS: Final = (
    *_DEEP_RESEARCH_PARAMS,
    "temperature",
    "top_p",
    "stop",
    "seed",
    "n",
    "tools",
    "tool_choice",
    "function_call",
    "functions",
    "parallel_tool_calls",
)


class ApodexChatConfig(OpenAIGPTConfig):
    """
    Reference: https://platform.apodex.ai/docs
    API Key: APODEX_API_KEY
    Default API Base: https://api.apodex.ai/v1
    """

    API_BASE_URL = APODEX_API_BASE_URL

    @property
    def custom_llm_provider(self) -> str | None:
        return "apodex"

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return get_apodex_api_key(api_key)

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return get_apodex_api_base(api_base)

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return get_apodex_api_base(api_base), get_apodex_api_key(api_key)

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: matches the base-class signature
        supported: Final = _DEEP_RESEARCH_PARAMS if is_deep_research_model(model) else _CORE_PARAMS
        return list(supported)  # mutable-ok: matches the base-class signature

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: matches the base-class signature
        optional_params: dict,  # mutable-ok: matches the base-class signature
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: matches the base-class signature
        mapped: Final = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # Apodex documents max_tokens only.
        renamed: Final = (
            mapped
            if "max_completion_tokens" not in mapped
            else {  # mutable-ok: JSON request body
                **{  # mutable-ok: JSON request body
                    key: value for key, value in mapped.items() if key != "max_completion_tokens"
                },
                "max_tokens": mapped["max_completion_tokens"],
            }
        )

        if renamed.get("stream"):
            return renamed

        # The OpenAI SDK drops `stream` from the body when it is false, which would
        # leave Apodex on its streaming default. extra_body is merged into the
        # request body by the SDK, so it survives that drop.
        requested_extra_body: Final = renamed.get("extra_body")
        extra_body: Final = (
            requested_extra_body if isinstance(requested_extra_body, Mapping) else {}  # mutable-ok: JSON request body
        )
        return {  # mutable-ok: JSON request body
            **renamed,
            "extra_body": {"stream": False, **extra_body},  # mutable-ok: JSON request body
        }
