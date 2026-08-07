import json
from pathlib import Path

import pytest

import litellm
from litellm.proxy.auth.model_checks import get_provider_models
from litellm.types.utils import TranscriptionResponse

TRANSCRIPTION_MODELS = ("mistral/voxtral-mini-2602", "mistral/voxtral-mini-latest")
CHAT_MODELS = ("mistral/voxtral-small-2507", "mistral/voxtral-small-latest")
ALL_VOXTRAL_MODELS = TRANSCRIPTION_MODELS + CHAT_MODELS


def _load_cost_maps() -> tuple[dict, dict]:
    repo_root = Path(__file__).parents[2]
    with open(repo_root / "model_prices_and_context_window.json") as f:
        main_cost = json.load(f)
    with open(repo_root / "litellm" / "model_prices_and_context_window_backup.json") as f:
        backup_cost = json.load(f)
    return main_cost, backup_cost


@pytest.fixture
def local_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    local_map = litellm.get_model_cost_map(url="")
    monkeypatch.setattr(litellm, "model_cost", local_map)
    litellm.add_known_models(local_map)
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", TRANSCRIPTION_MODELS)
def test_voxtral_transcription_models_are_priced_per_second(model: str) -> None:
    main_cost, _ = _load_cost_maps()
    info = main_cost[model]
    assert info["mode"] == "audio_transcription"
    assert info["supported_endpoints"] == ["/v1/audio/transcriptions"]
    assert info["input_cost_per_second"] == pytest.approx(0.003 / 60)
    assert "output_cost_per_second" not in info


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_voxtral_small_is_an_audio_capable_chat_model(model: str) -> None:
    main_cost, _ = _load_cost_maps()
    info = main_cost[model]
    assert info["mode"] == "chat"
    assert info["supports_audio_input"] is True
    assert info["input_cost_per_token"] == pytest.approx(1e-07)
    assert info["output_cost_per_token"] == pytest.approx(3e-07)


@pytest.mark.parametrize("model", ALL_VOXTRAL_MODELS)
def test_backup_cost_map_matches_main(model: str) -> None:
    main_cost, backup_cost = _load_cost_maps()
    assert backup_cost.get(model) == main_cost.get(model)


def test_voxtral_models_are_discoverable_for_the_mistral_wildcard(local_cost_map: None) -> None:
    """Regression for #34616: `mistral/*` on the proxy expands to the provider's known models,
    so voxtral was missing from /v1/models until it landed in the cost map."""
    provider_models = get_provider_models(provider="mistral") or []
    assert set(ALL_VOXTRAL_MODELS).issubset(set(provider_models))


def test_voxtral_transcription_cost_is_billed_per_audio_second(local_cost_map: None) -> None:
    response = TranscriptionResponse(text="demo text")
    response.duration = 90.0

    cost = litellm.completion_cost(
        completion_response=response,
        model="mistral/voxtral-mini-latest",
        custom_llm_provider="mistral",
        call_type="atranscription",
    )

    assert cost == pytest.approx(90.0 * 0.003 / 60)
