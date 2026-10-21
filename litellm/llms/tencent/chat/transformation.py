"""
Translates from OpenAI's `/v1/chat/completions` to Tencent TokenHub's
OpenAI-compatible endpoint.
"""

from collections.abc import Mapping
from typing import Final, TypedDict

from typing_extensions import ReadOnly

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.utils import supports_reasoning

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class ThinkingPayload(TypedDict, total=False):
    """Tencent TokenHub `thinking` object.

    `type` ("enabled"/"disabled"/"adaptive") is required by TokenHub when the
    object is passed; `budget_tokens` is auto-filled server-side when omitted.
    Ref: https://www.tencentcloud.com/document/product/1300/82345
    """

    type: ReadOnly[str]
    budget_tokens: ReadOnly[int]


class ThinkingExtraBody(TypedDict, total=False):
    """`extra_body` payload carrying TokenHub's `thinking` object."""

    thinking: ReadOnly[Mapping[str, object]]


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
        mapped_params: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)

        thinking_value: Final = mapped_params.pop("thinking", None)
        reasoning_effort: Final = mapped_params.pop("reasoning_effort", None)

        thinking: Final = self._resolve_thinking_payload(
            model=model,
            thinking_value=thinking_value,  # pyright: ignore[reportUnknownArgumentType]  # value popped from the untyped provider params dict
            reasoning_effort=reasoning_effort,  # pyright: ignore[reportUnknownArgumentType]  # value popped from the untyped provider params dict
        )
        if thinking is not None:
            # TokenHub expects `thinking` in the request JSON body, but the
            # OpenAI SDK's chat.completions.create() rejects unknown top-level
            # kwargs, so it travels via `extra_body`, which the SDK merges into
            # the payload. A plain assignment is merge-safe: get_optional_params
            # spreads this dict into its own extra_body assembly downstream.
            extra_body: Final[ThinkingExtraBody] = {"thinking": thinking}
            mapped_params["extra_body"] = extra_body
        return mapped_params

    @classmethod
    def _resolve_thinking_payload(
        cls,
        model: str,
        thinking_value: object,
        reasoning_effort: object,
    ) -> Mapping[str, object] | None:
        if isinstance(thinking_value, dict):
            return cls._coerce_thinking_type_for_model(model=model, thinking=thinking_value)  # pyright: ignore[reportUnknownArgumentType]  # isinstance narrows to dict[Unknown, Unknown] out of the untyped provider params dict
        if isinstance(reasoning_effort, str):
            # TokenHub recommends explicitly disabling thinking rather than
            # relying on per-model defaults (deepseek-v4-* default to enabled).
            payload: Final[ThinkingPayload] = {"type": "disabled" if reasoning_effort == "none" else "enabled"}
            return cls._coerce_thinking_type_for_model(model=model, thinking=payload)
        return None

    @staticmethod
    def _coerce_thinking_type_for_model(model: str, thinking: Mapping[str, object]) -> Mapping[str, object]:
        """Coerce `thinking.type` to a value the model accepts.

        MiniMax models on TokenHub only accept "adaptive"/"disabled" and reject
        "enabled" with a 400; "adaptive" (the model decides when to think) is
        the closest semantic, so "enabled" is coerced for them. The capability
        is read from the model map's `supports_adaptive_thinking` flag, so
        aliases and newly onboarded adaptive-only models need no code change.
        Ref: https://www.tencentcloud.com/document/product/1300/82345
        """
        if thinking.get("type") != "enabled" or not TencentChatConfig._is_adaptive_thinking_model(model):
            return thinking

        budget: Final[object] = thinking.get("budget_tokens")
        if isinstance(budget, int):
            coerced_with_budget: Final[ThinkingPayload] = {"type": "adaptive", "budget_tokens": budget}
            return coerced_with_budget
        coerced: Final[ThinkingPayload] = {"type": "adaptive"}
        return coerced

    @staticmethod
    def _is_adaptive_thinking_model(model: str) -> bool:
        """Read `supports_adaptive_thinking` from the model map under tencent."""
        try:
            model_info: Final[Mapping[str, object]] = litellm.get_model_info(model=model, custom_llm_provider="tencent")
        except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for unmapped models
            return False
        return model_info.get("supports_adaptive_thinking") is True

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
