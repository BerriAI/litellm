from __future__ import annotations

from litellm.rust_bridge.host.provider import resolved_provider


def anthropic_model_capabilities(model: str, custom_llm_provider: str | None) -> dict[str, object]:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    _, provider = resolved_provider(model, custom_llm_provider)

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
        "supports_mid_conversation_system": supports("supports_mid_conversation_system"),
        "effort_tiers": {level: tier(level) for level in ("minimal", "low", "medium", "high", "xhigh", "max")},
    }
