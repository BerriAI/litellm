from typing import Final

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str

# Narrower than the generic OpenAI surface on purpose: the gateway rejects legacy
# top-level fields (``functions``, ``function_call``, ``logit_bias``) with a 400.
SUPPORTED_OPENAI_PARAMS: Final = (
    "stream",
    "stream_options",
    "frequency_penalty",
    "presence_penalty",
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


class ClfAiGatewayConfig(OpenAIGPTConfig):
    """
    Reference: https://clfaigateway.dev/docs

    CLF AI Gateway is an OpenAI-compatible gateway (chat completions, streaming, tool
    calling, structured output, ``reasoning_effort``) serving open-weight models
    (GLM, Kimi, DeepSeek, Qwen) on Cloudflare Workers AI upstream. Model ids are the
    gateway's canonical names, e.g. ``clf_ai_gateway/glm-5.3``.

    The gateway validates request bodies strictly and rejects unknown or legacy
    top-level fields with a 400 (``functions``, ``function_call``, ``logit_bias``),
    so the supported-params list below is deliberately narrower than the generic
    OpenAI surface.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "clf_ai_gateway"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: matches OpenAIGPTConfig's list return
        supports_reasoning: Final = litellm.supports_reasoning(
            model=model,
            custom_llm_provider=self.custom_llm_provider,
        )
        extra: Final = ("reasoning_effort",) if supports_reasoning else ()
        return [*SUPPORTED_OPENAI_PARAMS, *extra]  # mutable-ok: one-shot build, list per base contract

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = (
            api_base or get_secret_str("CLF_AI_GATEWAY_API_BASE") or "https://api.clfaigateway.dev/v1"
        )
        dynamic_api_key: Final = api_key or get_secret_str("CLF_AI_GATEWAY_API_KEY")
        return resolved_api_base, dynamic_api_key
