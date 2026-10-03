"""
Structured provider capabilities declaration for model routing and feature detection.

Replaces ad-hoc flags scattered across model_prices_and_context_window.json
with a single typed dataclass that each provider declares.

Design mirrors zeshim's `ProviderCapabilities` interface, adapted to LiteLLM's
Python dataclass conventions and existing model registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSupports:
    """Boolean feature flags for a provider/model.

    Each field defaults to False — providers opt IN to capabilities.
    This avoids the "silently false" problem where a missing key in a
    JSON dict is indistinguishable from a capability being absent.
    """

    vision: bool = False
    streaming: bool = False
    multi_image: bool = False
    function_calling: bool = False
    structured_output: bool = False
    audio_input: bool = False
    audio_output: bool = False
    document_input: bool = False
    prompt_caching: bool = False
    reasoning: bool = False


@dataclass(frozen=True)
class ProviderStrengths:
    """Scored provider strengths for intelligent routing decisions.

    Each dimension is a float in [0.0, 1.0] where higher is better.
    Used by Router strategies (latency-based, cost-based, quality-based)
    to make data-driven deployment selection.

    Not all providers need to declare all dimensions — missing scores
    default to 0.0 (neutral).
    """

    speed: float = 0.0  # Low latency relative to peer models
    cost_efficiency: float = 0.0  # Tokens-per-dollar relative to peer models
    quality: float = 0.0  # Output quality / benchmark performance
    reliability: float = 0.0  # Uptime / error-rate track record
    context_fidelity: float = 0.0  # Long-context adherence

    def __post_init__(self):
        for name in (
            "speed",
            "cost_efficiency",
            "quality",
            "reliability",
            "context_fidelity",
        ):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"strengths.{name} must be in [0.0, 1.0], got {val}")


@dataclass(frozen=True)
class ProviderCapabilities:
    """Complete capabilities declaration for a provider/model.

    Composed of:
      supports  — boolean feature flags (what the model CAN do)
      strengths — scored dimensions (how WELL it does them),
                  used by the Router for intelligent deployment selection

    Immutable by design — capabilities don't change at runtime.
    Models that gain capabilities (e.g., a vision update) should
    declare a new capabilities object.
    """

    supports: ProviderSupports = field(default_factory=ProviderSupports)
    strengths: ProviderStrengths = field(default_factory=ProviderStrengths)

    # ── Convenience accessors (backward-compatible with existing API) ──

    @property
    def supports_vision(self) -> bool:
        return self.supports.vision

    @property
    def supports_streaming(self) -> bool:
        return self.supports.streaming

    @property
    def supports_function_calling(self) -> bool:
        return self.supports.function_calling

    @property
    def supports_structured_output(self) -> bool:
        return self.supports.structured_output

    @property
    def supports_audio(self) -> bool:
        return self.supports.audio_input or self.supports.audio_output


# ── Pre-built capability profiles for common model families ──

# These serve as templates that individual providers can extend or override.
# The goal: reduce duplication — OpenAI-compatible providers don't each need
# to re-declare the same capability set.

CAPABILITIES_OPENAI_GPT4O = ProviderCapabilities(
    supports=ProviderSupports(
        vision=True,
        streaming=True,
        multi_image=True,
        function_calling=True,
        structured_output=True,
        reasoning=True,
    ),
    strengths=ProviderStrengths(
        speed=0.7,
        cost_efficiency=0.5,
        quality=0.9,
        reliability=0.95,
        context_fidelity=0.85,
    ),
)

CAPABILITIES_ANTHROPIC_CLAUDE = ProviderCapabilities(
    supports=ProviderSupports(
        vision=True,
        streaming=True,
        multi_image=True,
        function_calling=True,
        structured_output=True,
        prompt_caching=True,
        document_input=True,
        reasoning=True,
    ),
    strengths=ProviderStrengths(
        speed=0.6,
        cost_efficiency=0.4,
        quality=0.95,
        reliability=0.95,
        context_fidelity=0.90,
    ),
)

CAPABILITIES_GEMINI_FLASH = ProviderCapabilities(
    supports=ProviderSupports(
        vision=True,
        streaming=True,
        function_calling=True,
        audio_input=True,
        prompt_caching=True,
    ),
    strengths=ProviderStrengths(
        speed=0.9,
        cost_efficiency=0.85,
        quality=0.7,
        reliability=0.85,
        context_fidelity=0.95,
    ),
)

# ── Integration helper: build from existing model_cost_map entry ──


def capabilities_from_model_info(
    supports_vision: bool = False,
    supports_function_calling: bool = False,
    supports_streaming: bool = False,
    supports_audio_input: bool = False,
    supports_audio_output: bool = False,
    supports_prompt_caching: bool = False,
    supports_response_schema: bool = False,
    supports_reasoning: bool = False,
    **_: object,  # ignore unknown keys from model_cost_map
) -> ProviderCapabilities:
    """Bridge: construct ProviderCapabilities from existing model_cost_map flags.

    This allows gradual migration — existing JSON entries can populate
    capabilities via this helper without rewriting the registry.
    """
    return ProviderCapabilities(
        supports=ProviderSupports(
            vision=supports_vision,
            streaming=supports_streaming,
            function_calling=supports_function_calling,
            structured_output=supports_response_schema,
            audio_input=supports_audio_input,
            audio_output=supports_audio_output,
            prompt_caching=supports_prompt_caching,
            reasoning=supports_reasoning,
        ),
        # strengths comes from model_cost_map scoring or defaults to neutral
    )
