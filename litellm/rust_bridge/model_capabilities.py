from __future__ import annotations

import litellm


def _resolved_provider(model: str, custom_llm_provider: str | None) -> tuple[str, str]:
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # an unroutable model still shapes as a bare Anthropic id
        return model, custom_llm_provider or "anthropic"
    return resolved_model, provider


def anthropic_model_capabilities(model: str, custom_llm_provider: str | None) -> dict[str, object]:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    resolved_model, provider = _resolved_provider(model, custom_llm_provider)

    def supports(flag: str) -> bool:
        return AnthropicModelInfo._supports_model_capability(model, flag, provider)  # pyright: ignore[reportPrivateUsage]  # same probes the Python transform runs; forking them would drift

    def tier(level: str) -> bool:
        return AnthropicConfig._supports_effort_level(model, level, provider)  # pyright: ignore[reportPrivateUsage]  # same probe the Python transform runs

    return {
        "supports_reasoning": supports("supports_reasoning"),
        "supports_adaptive_thinking": supports("supports_adaptive_thinking"),
        "thinking_always_on": supports("thinking_always_on"),
        "supports_legacy_thinking": supports("supports_legacy_thinking"),
        "supports_output_config": supports("supports_output_config"),
        "supports_sampling_params": AnthropicModelInfo._supports_sampling_params(resolved_model),  # pyright: ignore[reportPrivateUsage]  # same gate the handler applies
        "supports_speed": AnthropicConfig._model_supports_speed_param(resolved_model, provider),  # pyright: ignore[reportPrivateUsage]  # same gate the handler applies
        "effort_tiers": {level: tier(level) for level in ("minimal", "low", "medium", "high", "xhigh", "max")},
    }
