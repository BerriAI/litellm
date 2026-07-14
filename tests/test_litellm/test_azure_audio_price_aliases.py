"""Undated azure aliases for the audio models must exist and match their dated
variants — Azure deployments are commonly created against the undated model
name, and `base_model: azure/gpt-audio-mini` previously resolved to nothing
(text tokens billed at $0). Issue #33170."""

import pytest

import litellm


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost


COST_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
)


@pytest.mark.parametrize(
    "undated, dated",
    [
        ("azure/gpt-audio-mini", "azure/gpt-audio-mini-2025-10-06"),
        ("azure/gpt-realtime-mini", "azure/gpt-realtime-mini-2025-10-06"),
    ],
)
def test_undated_azure_audio_alias_matches_dated_entry(undated, dated):
    undated_info = litellm.get_model_info(undated)
    dated_info = litellm.get_model_info(dated)

    for field in COST_FIELDS:
        assert undated_info.get(field) == dated_info.get(field), field
        # the production symptom was text tokens billed at $0 — make sure the
        # alias carries real, non-zero prices
        assert (undated_info.get(field) or 0) > 0, f"{undated}.{field} must be non-zero"

    assert undated_info.get("litellm_provider") == "azure"
    assert undated_info.get("mode") == dated_info.get("mode")
