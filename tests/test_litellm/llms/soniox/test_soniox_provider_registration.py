"""Tests verifying Soniox is correctly registered as a litellm provider."""

import pytest

import litellm


class TestProviderRegistration:
    def test_should_expose_soniox_in_llm_providers_enum(self):
        assert litellm.LlmProviders.SONIOX.value == "soniox"

    def test_should_list_soniox_in_provider_list(self):
        assert "soniox" in litellm.provider_list

    def test_should_list_soniox_in_models_by_provider(self):
        assert "soniox" in litellm.models_by_provider

    def test_should_lazy_import_soniox_audio_transcription_config(self):
        cls = litellm.SonioxAudioTranscriptionConfig
        assert cls.__name__ == "SonioxAudioTranscriptionConfig"
        # Calling again should return the same class (cached).
        assert litellm.SonioxAudioTranscriptionConfig is cls

    def test_should_resolve_soniox_via_get_llm_provider(self, monkeypatch):
        monkeypatch.setenv("SONIOX_API_KEY", "test-key")
        model, provider, api_key, api_base = litellm.get_llm_provider(
            model="soniox/stt-async-v4"
        )
        assert provider == "soniox"
        assert model == "stt-async-v4"
        assert api_key == "test-key"
        assert api_base == "https://api.soniox.com"

    def test_should_return_soniox_config_from_provider_config_manager(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_audio_transcription_config(
            model="stt-async-v4",
            provider=litellm.LlmProviders.SONIOX,
        )
        assert cfg is not None
        assert cfg.__class__.__name__ == "SonioxAudioTranscriptionConfig"
