import json
from pathlib import Path

import pytest


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
    assert info["input_cost_per_token"] == 1e-06
    assert info["output_cost_per_token"] == 2e-05
    assert info["output_cost_per_audio_token"] == 2e-05
    assert info["max_input_tokens"] == 8192
    assert info["max_output_tokens"] == 16384
    assert info["max_tokens"] == 16384
    assert info["supported_endpoints"] == ["/v1/audio/speech"]
    assert info["supported_modalities"] == ["text"]
    assert info["supported_output_modalities"] == ["audio"]
    assert info["supports_audio_input"] is False
    assert info["supports_audio_output"] is True
    assert info["supports_function_calling"] is False
    assert info["supports_prompt_caching"] is False
    assert info["health_check_voice"] == "Kore"


def test_gemini_3_1_flash_tts_backup_matches_main():
    repo_root = Path(__file__).parents[2]
    main_cost = _load_model_cost_map(repo_root / "model_prices_and_context_window.json")
    backup_cost = _load_model_cost_map(repo_root / "litellm" / "model_prices_and_context_window_backup.json")

    for model in EXPECTED_MODELS:
        assert backup_cost.get(model) == main_cost.get(model), (
            f"{model} differs between main and backup model cost maps"
        )
