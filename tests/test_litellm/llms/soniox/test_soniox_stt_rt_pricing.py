"""Regression tests for the soniox/stt-rt-v5 pricing entry.

The realtime model is pricing metadata only (not invocable through LiteLLM),
so these tests pin what downstream spend tracking depends on: the entry's
shape in BOTH cost-map files (runtime consumers fetch the root file, the SDK
falls back to the bundled backup — a divergence breaks one silently),
provider-qualified model lookup, and duration-based cost calculation at the
published $0.12/hr realtime rate — 20% above the async rate.
"""

import json
import os

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.types.utils import TranscriptionResponse

REPO_ROOT = os.path.join(os.path.dirname(__file__), "../../../..")

MODEL_KEY = "soniox/stt-rt-v5"
REALTIME_RATE_PER_HOUR = 0.12
ASYNC_RATE_PER_HOUR = 0.10


def _load_root_cost_map() -> dict:
    json_path = os.path.join(REPO_ROOT, "model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


@pytest.mark.parametrize(
    "cost_map",
    [_load_root_cost_map(), GetModelCostMap.load_local_model_cost_map()],
    ids=["root", "bundled_backup"],
)
def test_realtime_entry_present_and_priced_in_both_cost_maps(cost_map):
    """The root file is what runtime consumers fetch; the bundled backup is
    the SDK fallback. An entry missing from either breaks that consumer
    silently, so both are pinned."""
    assert MODEL_KEY in cost_map, f"{MODEL_KEY} missing from cost map"
    entry = cost_map[MODEL_KEY]

    assert entry["litellm_provider"] == "soniox"
    assert entry["mode"] == "audio_transcription"
    assert entry["supports_audio_input"] is True
    # Soniox carries the rate on the output side (matching the async rows);
    # a rate moved to input_cost_per_second would double-charge any consumer
    # that sums both sides.
    assert entry["input_cost_per_second"] == 0.0
    assert entry["output_cost_per_second"] == pytest.approx(
        REALTIME_RATE_PER_HOUR / 3600, rel=1e-3
    )


def test_provider_qualified_lookup_resolves_realtime_model():
    model_info = litellm.get_model_info(MODEL_KEY)

    assert model_info["litellm_provider"] == "soniox"
    assert model_info["mode"] == "audio_transcription"
    assert model_info["input_cost_per_second"] == 0.0
    assert model_info["output_cost_per_second"] == pytest.approx(
        REALTIME_RATE_PER_HOUR / 3600, rel=1e-3
    )


def test_should_charge_realtime_transcription_by_audio_duration():
    ten_minutes = 600.0
    response = TranscriptionResponse(text="hello world")
    response._hidden_params = {"audio_transcription_duration": ten_minutes}

    cost = litellm.completion_cost(
        completion_response=response,
        model=MODEL_KEY,
        call_type="transcription",
    )

    assert cost > 0
    assert cost == pytest.approx(
        (REALTIME_RATE_PER_HOUR / 3600) * ten_minutes, rel=1e-3
    )


def test_realtime_rate_stays_above_async_rate():
    """Guards against the realtime entry regressing to the cheaper async rate."""
    realtime = litellm.get_model_info(MODEL_KEY)
    async_v5 = litellm.get_model_info("soniox/stt-async-v5")

    assert realtime["output_cost_per_second"] > async_v5["output_cost_per_second"]
    assert async_v5["output_cost_per_second"] == pytest.approx(
        ASYNC_RATE_PER_HOUR / 3600, rel=1e-3
    )
