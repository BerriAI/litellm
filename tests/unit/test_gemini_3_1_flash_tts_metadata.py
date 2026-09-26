import json
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.utils import _invalidate_model_cost_lowercase_map, is_gemini_tts_model

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

EXPECTED_MODELS = {
    "gemini-3.1-flash-tts-preview": "gemini",
    "gemini/gemini-3.1-flash-tts-preview": "gemini",
    "vertex_ai/gemini-3.1-flash-tts-preview": "vertex_ai-language-models",
}


def _load_model_cost_map(path: Path) -> dict:
    with open(path, encoding="utf-8") as model_cost_file:
        return json.load(model_cost_file)


@pytest.mark.parametrize("model,provider", EXPECTED_MODELS.items())
def test_gemini_3_1_flash_tts_model_metadata(model, provider):
    model_cost = _load_model_cost_map(Path(__file__).parents[2] / "model_prices_and_context_window.json")

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model cost map"

    assert info["litellm_provider"] == provider
    assert info["mode"] == "audio_speech"
    assert "/v1/audio/speech" in info["supported_endpoints"]


def test_gemini_3_1_flash_tts_backup_matches_main():
    repo_root = Path(__file__).parents[2]
    main_cost = _load_model_cost_map(repo_root / "model_prices_and_context_window.json")
    backup_cost = _load_model_cost_map(repo_root / "litellm" / "model_prices_and_context_window_backup.json")

    for model in EXPECTED_MODELS:
        assert backup_cost.get(model) == main_cost.get(model), (
            f"{model} differs between main and backup model cost maps"
        )


def test_gemini_tts_detection_uses_model_metadata(monkeypatch):
    model: str = "gemini/future-generative-speech"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "gemini",
            "mode": "audio_speech",
        },
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()

    assert is_gemini_tts_model(model)


def test_gemini_tts_detection_uses_bundled_metadata_when_runtime_provider_is_generic(monkeypatch):
    model: Final = "gemini-3.1-flash-tts-preview"
    monkeypatch.setitem(
        litellm.model_cost,
        f"vertex_ai/{model}",
        {"litellm_provider": "vertex_ai", "mode": "audio_speech"},
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()

    assert is_gemini_tts_model(model, custom_llm_provider="vertex_ai")
    assert not is_gemini_tts_model("lyria-3-clip-preview", custom_llm_provider="vertex_ai")


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash-preview-tts", "gemini-2.5-flash-tts", "gemini-2.5-pro-tts", "gemini-2.5-pro-preview-tts"],
)
@pytest.mark.parametrize("prefixed", [False, True])
def test_existing_vertex_tts_models_keep_gemini_dispatch(model: str, prefixed: bool, monkeypatch):
    model_cost = _load_model_cost_map(Path(__file__).parents[2] / "model_prices_and_context_window.json")
    monkeypatch.setattr(litellm, "model_cost", model_cost)
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    requested_model = f"vertex_ai/{model}" if prefixed else model
    assert is_gemini_tts_model(requested_model, custom_llm_provider="vertex_ai")


def test_existing_gemini_pro_preview_tts_keeps_speech_dispatch() -> None:
    assert is_gemini_tts_model("gemini/gemini-2.5-pro-preview-tts", custom_llm_provider="gemini")


def test_gemini_music_model_does_not_use_speech_dispatch() -> None:
    assert not is_gemini_tts_model("gemini/lyria-3-clip-preview", custom_llm_provider="gemini")
