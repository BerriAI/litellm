"""
Translates from OpenAI's `/v1/chat/completions` to Baidu Qianfan's `/v2/chat/completions`
"""

from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

QIANFAN_API_BASE: Final = "https://qianfan.baidubce.com/v2"


def _reasoning_effort_string(value: object) -> str | None:
    """The /v1/messages and /v1/responses bridges wrap the level as {"effort", "summary"}. Qianfan
    takes only the bare string, so the level is unwrapped and the summary dropped."""
    effort: Final = value.get("effort") if isinstance(value, Mapping) else value
    return effort if isinstance(effort, str) else None


class QianfanChatConfig(OpenAIGPTConfig):
    """
    Baidu Qianfan (ModelBuilder) exposes an OpenAI-compatible chat endpoint. Two things differ
    from plain OpenAI and are handled here:

    - Thinking is switched with a Qianfan-specific `thinking: {"type": "enabled" | "disabled"}`
      body field, optionally bounded by `thinking_budget`. Some hosted models, such as the
      DeepSeek-V4 family, think by default, so a caller who wants plain completions has to send
      the disabled switch explicitly. LiteLLM's `thinking` and `reasoning_effort="none"` map onto it
    - `reasoning_effort` is accepted as `high` (default) or `max`; Qianfan itself folds `low` and
      `medium` into `high` and `xhigh` into `max`, so the value is passed through unchanged

    Qianfan-only request fields (`penalty_score`, `web_search`, `thinking_strategy`, `metadata`,
    `enable_thinking`) ride through the generic provider-specific-params path into the request body

    Reference: https://cloud.baidu.com/doc/qianfan/s/rmh4stp0j (Chat V2)
    """

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base class contract returns a list
        return [*super().get_supported_openai_params(model), "thinking", "reasoning_effort"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: base class contract
        optional_params: dict,  # mutable-ok: base class contract
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: base class contract returns a dict
        mapped: Final = dict(super().map_openai_params(non_default_params, optional_params, model, drop_params))

        thinking_value: Final = as_str_mapping(mapped.pop("thinking", None))
        reasoning_effort: Final = _reasoning_effort_string(mapped.pop("reasoning_effort", None))

        qianfan_params: Final[dict[str, object]] = {}  # mutable-ok: assembled then merged into the request body
        if thinking_value is not None and thinking_value.get("type") in ("enabled", "disabled"):
            qianfan_params["thinking"] = {"type": thinking_value["type"]}
            budget: Final = thinking_value.get("budget_tokens")
            if thinking_value["type"] == "enabled" and isinstance(budget, int):
                qianfan_params["thinking_budget"] = budget
        elif reasoning_effort == "none":
            qianfan_params["thinking"] = {"type": "disabled"}

        if reasoning_effort is not None and reasoning_effort != "none":
            qianfan_params["reasoning_effort"] = reasoning_effort

        return {**mapped, **qianfan_params}

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("QIANFAN_API_BASE") or QIANFAN_API_BASE
        dynamic_api_key: Final = api_key or get_secret_str("QIANFAN_API_KEY")
        return resolved_api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: base class contract
        litellm_params: dict,  # mutable-ok: base class contract
        stream: bool | None = None,
    ) -> str:
        return ensure_api_route(api_base or QIANFAN_API_BASE, "chat/completions")
