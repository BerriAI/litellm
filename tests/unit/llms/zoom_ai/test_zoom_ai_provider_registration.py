from typing import Final

import pytest

import litellm
from litellm.constants import ZOOM_AI_DEFAULT_API_BASE
from litellm.llms.zoom_ai.audio_transcription.transformation import ZoomAIAudioTranscriptionConfig
from litellm.utils import ProviderConfigManager


def test_zoom_ai_models_resolve_to_the_zoom_ai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOOM_AI_API_KEY", "test-key")
    monkeypatch.delenv("ZOOM_AI_API_BASE", raising=False)

    model, provider, api_key, api_base = litellm.get_llm_provider(model="zoom_ai/zoom-scribe-en")

    assert (model, provider, api_key, api_base) == ("zoom-scribe-en", "zoom_ai", "test-key", ZOOM_AI_DEFAULT_API_BASE)


def test_zoom_ai_api_base_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZOOM_AI_API_BASE", "https://zoom.example.test/v2/aiservices")

    _, _, _, api_base = litellm.get_llm_provider(model="zoom_ai/zoom-scribe-en")

    assert api_base == "https://zoom.example.test/v2/aiservices"


def test_zoom_ai_transcription_config_is_registered() -> None:
    config: Final = ProviderConfigManager.get_provider_audio_transcription_config(
        model="zoom-scribe-en", provider=litellm.LlmProviders.ZOOM_AI
    )

    assert isinstance(config, ZoomAIAudioTranscriptionConfig)
    assert litellm.ZoomAIAudioTranscriptionConfig is ZoomAIAudioTranscriptionConfig


def test_every_priced_zoom_ai_model_is_a_transcription_model(local_model_cost_map: None) -> None:
    zoom_ai_rows: Final = {
        name: row for name, row in litellm.model_cost.items() if row.get("litellm_provider") == "zoom_ai"
    }

    assert set(litellm.models_by_provider["zoom_ai"]) == set(zoom_ai_rows)
    assert all(row["mode"] == "audio_transcription" for row in zoom_ai_rows.values())
    assert all(row["input_cost_per_second"] > 0 for row in zoom_ai_rows.values())
