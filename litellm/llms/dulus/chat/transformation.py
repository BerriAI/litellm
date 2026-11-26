from typing import Final
from urllib.parse import urlparse

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str

SUPPORTED_OPENAI_PARAMS: Final = (
    "stream",
    "stream_options",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "stop",
    "temperature",
    "top_p",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "user",
)


class DulusConfig(OpenAIGPTConfig):
    """
    Config for the Dulus OpenAI-compatible API (https://control.dulus.ai/v1).

    Dulus is a provider-independent agent platform that also serves a hosted,
    OpenAI-compatible chat completions endpoint. Usage:

        completion(model="dulus/dulus-b-27b", messages=messages)

    with `DULUS_API_KEY` set (keys look like `dulus_sk_live_...`). Billing is
    metered in Fuel on the tokens each call actually uses; there is no public
    per-model USD list price. The rows in `model_prices_and_context_window.json`
    carry proxy-accounting defaults (relative 1x/2x/4x Fuel tiers mapped to USD
    for budget tracking only — NOT Dulus billing). For budget-controlled proxy
    deployments, override input_cost_per_token/output_cost_per_token in config
    or enable `block_requests_for_models_without_pricing`.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "dulus"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: matches OpenAIGPTConfig's list return
        return [*SUPPORTED_OPENAI_PARAMS]

    def get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("DULUS_API_BASE") or "https://control.dulus.ai/v1"
        # DULUS_API_KEY from the environment is only sent over HTTPS. A plain-HTTP
        # base (a local gateway) has to be given its key explicitly.
        over_https: Final = urlparse(resolved_api_base).scheme.lower() == "https"
        dynamic_api_key: Final = api_key or (get_secret_str("DULUS_API_KEY") if over_https else None)
        if not dynamic_api_key and not over_https:
            raise AuthenticationError(
                message=f"Refusing to send DULUS_API_KEY to non-HTTPS api_base {resolved_api_base!r}. "
                "Pass api_key explicitly for a plain-HTTP gateway.",
                llm_provider="dulus",
                model="dulus/*",
            )
        if not dynamic_api_key:
            raise AuthenticationError(
                message="Missing DULUS_API_KEY - set DULUS_API_KEY or pass api_key. "
                "Dulus keys look like dulus_sk_live_... . Refusing to fall back to "
                "OPENAI_API_KEY for https://control.dulus.ai/v1.",
                llm_provider="dulus",
                model="dulus/*",
            )
        return resolved_api_base, dynamic_api_key

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        if not api_key:
            raise AuthenticationError(
                message="Missing DULUS_API_KEY - set DULUS_API_KEY or pass api_key. "
                "Dulus keys look like dulus_sk_live_... .",
                llm_provider="dulus",
                model=model,
            )
        return super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
