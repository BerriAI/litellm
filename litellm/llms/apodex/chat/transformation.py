"""
Apodex chat completions — OpenAI-compatible, with two provider quirks:

- the Deep Research tiers default `stream` to true, so a non-streaming call has
  to say so explicitly or Apodex answers with SSE that a plain call cannot
  parse. The core models follow OpenAI and default it to false
- the Deep Research tiers ignore sampling parameters and reject OpenAI-style
  tools; only the core models take them

Ref: https://platform.apodex.ai/docs/chat-completions
     https://platform.apodex.ai/docs/models
"""

from collections.abc import Mapping
from typing import Final

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues

from ..common_utils import (
    APODEX_API_BASE_URL,
    get_apodex_api_base,
    get_apodex_api_key,
    is_deep_research_model,
    is_responses_only_model,
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

_PIN_NON_STREAMING: Final = "_apodex_pin_non_streaming"


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
        if is_responses_only_model(model):
            raise litellm.BadRequestError(
                message=f"apodex model {model} is only available through /v1/responses",
                model=model,
                llm_provider="apodex",
            )

        mapped: Final = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

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

        return {  # mutable-ok: JSON request body
            **renamed,
            _PIN_NON_STREAMING: True,
        }

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: matches the base-class signature
        optional_params: dict,  # mutable-ok: matches the base-class signature
        litellm_params: dict,  # mutable-ok: matches the base-class signature
        headers: dict,  # mutable-ok: matches the base-class signature
    ) -> dict:  # mutable-ok: JSON request body
        pin_non_streaming: Final = bool(optional_params.get(_PIN_NON_STREAMING, False))
        forwarded_params: Final = {  # mutable-ok: base transformer requires a request dict
            key: value for key, value in optional_params.items() if key != _PIN_NON_STREAMING
        }
        transformed: Final = super().transform_request(
            model=model,
            messages=messages,
            optional_params=forwarded_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if not pin_non_streaming:
            return transformed

        requested_extra_body: Final = transformed.get("extra_body")
        extra_body: Final = (
            requested_extra_body if isinstance(requested_extra_body, Mapping) else {}  # mutable-ok: JSON request body
        )
        return {  # mutable-ok: JSON request body
            **transformed,
            "extra_body": {**extra_body, "stream": False},  # mutable-ok: JSON request body
        }
