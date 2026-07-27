"""Regression tests for the soniox/stt-rt-v5 pricing entry.

The realtime model is pricing metadata only (not invocable through LiteLLM),
so these tests pin the two things downstream spend tracking depends on:
provider-qualified model lookup, and duration-based cost calculation at the
published $0.12/hr realtime rate — 20% above the async rate.
"""

import pytest

from litellm.types.utils import TranscriptionResponse

REALTIME_RATE_PER_HOUR = 0.12
ASYNC_RATE_PER_HOUR = 0.10


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    import litellm

    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def test_provider_qualified_lookup_resolves_realtime_model():
    import litellm

    model_info = litellm.get_model_info("soniox/stt-rt-v5")

    assert model_info["litellm_provider"] == "soniox"
    assert model_info["mode"] == "audio_transcription"
    assert model_info["input_cost_per_second"] == 0.0
    assert model_info["output_cost_per_second"] == pytest.approx(
        REALTIME_RATE_PER_HOUR / 3600, rel=1e-3
    )


def test_should_charge_realtime_transcription_by_audio_duration():
    import litellm

    ten_minutes = 600.0
    response = TranscriptionResponse(text="hello world")
    response._hidden_params = {"audio_transcription_duration": ten_minutes}

    cost = litellm.completion_cost(
        completion_response=response,
        model="soniox/stt-rt-v5",
        call_type="transcription",
    )

    assert cost > 0
    assert cost == pytest.approx((REALTIME_RATE_PER_HOUR / 3600) * ten_minutes, rel=1e-3)


def test_realtime_rate_stays_above_async_rate():
    """Guards against the realtime entry regressing to the cheaper async rate."""
    import litellm

    realtime = litellm.get_model_info("soniox/stt-rt-v5")
    async_v5 = litellm.get_model_info("soniox/stt-async-v5")

    assert (
        realtime["output_cost_per_second"] > async_v5["output_cost_per_second"]
    )
    assert async_v5["output_cost_per_second"] == pytest.approx(
        ASYNC_RATE_PER_HOUR / 3600, rel=1e-3
    )
