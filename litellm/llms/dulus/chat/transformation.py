from typing import Final

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
    per-model USD list price, so the model rows in
    `model_prices_and_context_window.json` intentionally carry no cost fields.
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
        dynamic_api_key: Final = api_key or get_secret_str("DULUS_API_KEY")
        return resolved_api_base, dynamic_api_key
