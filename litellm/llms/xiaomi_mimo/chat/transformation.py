"""
Translates from OpenAI's `/v1/chat/completions` to Xiaomi MiMo's `/v1/chat/completions`
"""

from typing import Optional

from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

# MiMo validates `reasoning_effort` as a strict literal — anything outside this set is
# rejected by the API ('none' -> 400 literal_error, 'minimal' -> 500).
XIAOMI_MIMO_REASONING_EFFORTS = ("low", "medium", "high")


class XiaomiMiMoChatConfig(OpenAIGPTConfig):
    """
    Xiaomi MiMo is an OpenAI-compatible provider. It authenticates with the standard
    OpenAI-style ``Authorization: Bearer <api_key>`` header (inherited from
    ``OpenAIGPTConfig``), which is what the production traffic to
    ``*.xiaomimimo.com/v1`` uses.

    Reasoning control (probed against the live API, 2026-07):
    - ``thinking={"type": "enabled"|"disabled"}`` is honored natively.
    - ``reasoning_effort`` accepts exactly ``low|medium|high``; ``none`` returns a
      400 literal_error and ``minimal`` a 500, so those values must be translated
      before they reach the API.
    """

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("XIAOMI_MIMO_API_BASE") or "https://api.xiaomimimo.com/v1"  # type: ignore
        dynamic_api_key = api_key or get_secret_str("XIAOMI_MIMO_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        params = super().get_supported_openai_params(model)
        for reasoning_param in ("reasoning_effort", "thinking"):
            if reasoning_param not in params:
                params.append(reasoning_param)
        return params

    @staticmethod
    def _merge_thinking_into_extra_body(optional_params: dict, thinking_value: dict) -> None:
        """`thinking` is not an OpenAI SDK kwarg, so it must ride ``extra_body`` to land
        top-level in the request JSON. Merge — never clobber caller-supplied extra_body,
        and never override a caller-supplied ``thinking`` key."""
        extra_body = optional_params.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        extra_body.setdefault("thinking", thinking_value)
        optional_params["extra_body"] = extra_body

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Xiaomi MiMo's API expects ``max_tokens``. Preserve the pre-migration behaviour of the
        old openai_like ``providers.json`` entry (``max_completion_tokens -> max_tokens``) by
        translating the OpenAI alias before delegating to the base OpenAI param mapping.

        Reasoning params are handled here because MiMo hard-fails on out-of-set values
        (see class docstring): supported values are forwarded natively, unsupported ones
        are translated to the nearest native equivalent instead of erroring the request.
        """
        non_default_params = dict(non_default_params)
        if "max_completion_tokens" in non_default_params:
            non_default_params["max_tokens"] = non_default_params.pop("max_completion_tokens")

        thinking = non_default_params.pop("thinking", None)
        reasoning_effort = non_default_params.pop("reasoning_effort", None)

        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        if isinstance(thinking, dict):
            # Natively honored; forwarded verbatim. When both `thinking` and
            # `reasoning_effort` are sent, `thinking` wins (their coexistence is
            # not documented by MiMo, so only one is forwarded).
            self._merge_thinking_into_extra_body(optional_params, thinking)
            return optional_params
        if thinking is not None:
            verbose_logger.warning(
                "xiaomi_mimo: ignoring `thinking` of type %s; expected a dict like "
                '{"type": "enabled"|"disabled"}',
                type(thinking).__name__,
            )
        if reasoning_effort is None:
            return optional_params
        if reasoning_effort == "none":
            # MiMo rejects 'none' (400); its native disable is thinking={"type": "disabled"}.
            self._merge_thinking_into_extra_body(optional_params, {"type": "disabled"})
        elif reasoning_effort == "minimal":
            # 'minimal' is rejected (500); clamp to the closest accepted literal.
            verbose_logger.debug(
                "xiaomi_mimo: mapping reasoning_effort='minimal' to 'low' (MiMo accepts low|medium|high)"
            )
            optional_params["reasoning_effort"] = "low"
        elif reasoning_effort in XIAOMI_MIMO_REASONING_EFFORTS:
            optional_params["reasoning_effort"] = reasoning_effort
        else:
            # Out-of-set values (e.g. 'xhigh' seen in the wild) hard-fail at MiMo; clamp.
            verbose_logger.warning(
                "xiaomi_mimo: reasoning_effort=%r is not supported by MiMo "
                "(accepts low|medium|high); clamping to 'high'",
                reasoning_effort,
            )
            optional_params["reasoning_effort"] = "high"
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default Xiaomi MiMo /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.xiaomimimo.com/v1"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
