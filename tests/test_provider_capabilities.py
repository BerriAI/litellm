"""Tests for litellm.provider_capabilities — structured capability declarations."""

import pytest
from litellm.provider_capabilities import (
    CAPABILITIES_ANTHROPIC_CLAUDE,
    CAPABILITIES_GEMINI_FLASH,
    CAPABILITIES_OPENAI_GPT4O,
    ProviderCapabilities,
    ProviderStrengths,
    ProviderSupports,
    capabilities_from_model_info,
)


class TestProviderSupports:
    """Boolean feature flags — opt-in, all default to False."""

    def test_default_all_false(self):
        s = ProviderSupports()
        assert s.vision is False
        assert s.streaming is False
        assert s.function_calling is False
        assert s.structured_output is False

    def test_explicit_true(self):
        s = ProviderSupports(vision=True, streaming=True)
        assert s.vision is True
        assert s.streaming is True
        assert s.function_calling is False  # still default

    def test_frozen(self):
        s = ProviderSupports(vision=True)
        with pytest.raises(Exception):
            s.vision = False  # type: ignore[misc]


class TestProviderStrengths:
    """Scored dimensions — float in [0.0, 1.0], defaults to 0.0."""

    def test_default_all_zero(self):
        s = ProviderStrengths()
        assert s.speed == 0.0
        assert s.quality == 0.0

    def test_valid_scores(self):
        s = ProviderStrengths(speed=0.8, quality=0.9, reliability=0.95)
        assert s.speed == 0.8
        assert s.quality == 0.9

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            ProviderStrengths(speed=1.5)

    def test_score_negative_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            ProviderStrengths(quality=-0.1)

    def test_frozen(self):
        s = ProviderStrengths(speed=0.5)
        with pytest.raises(Exception):
            s.speed = 0.8  # type: ignore[misc]


class TestProviderCapabilities:
    """Composed capabilities declaration."""

    def test_default_empty(self):
        cap = ProviderCapabilities()
        assert cap.supports_vision is False
        assert cap.supports_streaming is False
        assert cap.supports_function_calling is False

    def test_backward_compat_accessors(self):
        cap = ProviderCapabilities(
            supports=ProviderSupports(
                vision=True, streaming=True, function_calling=True
            ),
        )
        assert cap.supports_vision is True
        assert cap.supports_streaming is True
        assert cap.supports_function_calling is True
        assert cap.supports_structured_output is False

    def test_supports_audio_combines_input_and_output(self):
        cap = ProviderCapabilities(
            supports=ProviderSupports(audio_input=True, audio_output=False),
        )
        assert cap.supports_audio is True

        cap2 = ProviderCapabilities(
            supports=ProviderSupports(audio_input=False, audio_output=False),
        )
        assert cap2.supports_audio is False

    def test_strengths_accessible(self):
        cap = ProviderCapabilities(
            strengths=ProviderStrengths(speed=0.9, quality=0.8),
        )
        assert cap.strengths.speed == 0.9
        assert cap.strengths.quality == 0.8

    def test_frozen(self):
        cap = ProviderCapabilities()
        with pytest.raises(Exception):
            cap.supports = ProviderSupports()  # type: ignore[misc]


class TestPrebuiltCapabilityProfiles:
    """Pre-built profiles for common model families."""

    def test_openai_gpt4o(self):
        assert CAPABILITIES_OPENAI_GPT4O.supports.vision is True
        assert CAPABILITIES_OPENAI_GPT4O.supports.streaming is True
        assert CAPABILITIES_OPENAI_GPT4O.supports.function_calling is True
        assert CAPABILITIES_OPENAI_GPT4O.supports.structured_output is True
        assert CAPABILITIES_OPENAI_GPT4O.strengths.quality > 0.8

    def test_anthropic_claude(self):
        assert CAPABILITIES_ANTHROPIC_CLAUDE.supports.prompt_caching is True
        assert CAPABILITIES_ANTHROPIC_CLAUDE.supports.document_input is True
        assert CAPABILITIES_ANTHROPIC_CLAUDE.strengths.context_fidelity > 0.8

    def test_gemini_flash(self):
        assert CAPABILITIES_GEMINI_FLASH.supports.audio_input is True
        assert CAPABILITIES_GEMINI_FLASH.strengths.speed > 0.8
        assert CAPABILITIES_GEMINI_FLASH.strengths.cost_efficiency > 0.8

    def test_all_prebuilts_are_frozen(self):
        for cap in (
            CAPABILITIES_OPENAI_GPT4O,
            CAPABILITIES_ANTHROPIC_CLAUDE,
            CAPABILITIES_GEMINI_FLASH,
        ):
            with pytest.raises(Exception):
                cap.supports.vision = False  # type: ignore[misc]


class TestCapabilitiesFromModelInfo:
    """Bridge: migrate from existing model_cost_map flags."""

    def test_all_false_by_default(self):
        cap = capabilities_from_model_info()
        assert cap.supports_vision is False
        assert cap.supports_function_calling is False

    def test_maps_individual_flags(self):
        cap = capabilities_from_model_info(
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
        )
        assert cap.supports_vision is True
        assert cap.supports_function_calling is True
        assert cap.supports_streaming is True

    def test_ignores_unknown_kwargs(self):
        """Should not crash on extra keys from model_cost_map."""
        cap = capabilities_from_model_info(
            max_tokens=128000,  # unknown key
            input_cost_per_token=0.0001,  # unknown key
            supports_vision=True,
        )
        assert cap.supports_vision is True

    def test_maps_response_schema_to_structured_output(self):
        """supports_response_schema should map to structured_output."""
        cap = capabilities_from_model_info(
            supports_response_schema=True,
        )
        assert cap.supports_structured_output is True

    def test_maps_reasoning_flag(self):
        """supports_reasoning should be preserved."""
        cap = capabilities_from_model_info(
            supports_reasoning=True,
        )
        assert cap.supports.reasoning is True
