from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.router_utils.reasoning_effort_capability import declared_reasoning_efforts_for_model
from litellm.secret_managers.main import get_secret_str

# Narrower than the generic OpenAI surface on purpose: the gateway rejects legacy
# top-level fields (``functions``, ``function_call``) and unknown ones with a 400.
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
    "store",
    "service_tier",
)


class ClfAiGatewayConfig(OpenAIGPTConfig):
    """
    Reference: https://clfaigateway.dev/docs

    CLF AI Gateway is an OpenAI-compatible gateway (chat completions, streaming, tool
    calling, structured output, ``reasoning_effort``) serving open-weight models
    (GLM, Kimi, DeepSeek, Qwen) on Cloudflare Workers AI upstream. Model ids are the
    gateway's canonical names, e.g. ``clf_ai_gateway/glm-5.3``.

    The gateway validates request bodies strictly and rejects unknown or legacy
    top-level fields with a 400 (``functions``, ``function_call``), so the
    supported-params list below is deliberately narrower than the generic OpenAI
    surface. It also rejects a ``reasoning_effort`` level the model does not take.
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
        return [*SUPPORTED_OPENAI_PARAMS, *extra]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: OpenAIConfig.map_openai_params signature
        optional_params: dict[str, object],  # mutable-ok: OpenAIConfig.map_openai_params signature
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: OpenAIConfig.map_openai_params signature
        effort: Final = non_default_params.get("reasoning_effort")
        declared: Final = declared_reasoning_efforts_for_model(model, "clf_ai_gateway")
        drops_effort: Final = (
            (drop_params or litellm.drop_params is True)
            and effort is not None
            and declared is not None
            and effort not in declared
        )
        if drops_effort:
            verbose_logger.warning(
                "Dropping reasoning_effort=%r for model=%r (drop_params=True) because clf_ai_gateway takes only %s for it.",
                effort,
                model,
                declared,
            )
        forwarded: Final = (
            {param: value for param, value in non_default_params.items() if param != "reasoning_effort"}
            if drops_effort
            else non_default_params
        )
        mapped: Final[dict[str, object]] = super().map_openai_params(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # OpenAIGPTConfig types its params and result as bare dicts  # mutable-ok: parent returns a dict
            non_default_params=forwarded,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        return mapped

    def get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = (
            api_base or get_secret_str("CLF_AI_GATEWAY_API_BASE") or "https://api.clfaigateway.dev/v1"
        )
        dynamic_api_key: Final = api_key or get_secret_str("CLF_AI_GATEWAY_API_KEY")
        return resolved_api_base, dynamic_api_key
