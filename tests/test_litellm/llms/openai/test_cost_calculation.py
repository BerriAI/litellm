"""Tests for per-second transcription cost calculation."""

import pytest

import litellm
from litellm.llms.openai.cost_calculation import cost_per_second


def _register_stt(name: str, **pricing: float) -> None:
    litellm.register_model(
        {
            name: {
                "mode": "audio_transcription",
                "litellm_provider": "openai",
                **pricing,
            }
        },
        persist_across_reloads=False,
    )


def test_input_rate_bills_when_output_rate_is_zero():
    """A declared-but-zero output rate must not suppress the real input rate."""
    _register_stt(
        "test-stt-zero-output",
        input_cost_per_second=5e-05,
        output_cost_per_second=0.0,
    )

    prompt_cost, completion_cost = cost_per_second(
        model="test-stt-zero-output", custom_llm_provider="openai", duration=300.0
    )

    assert prompt_cost == pytest.approx(0.015)
    assert completion_cost == 0.0


def test_output_rate_takes_precedence_when_both_are_billable():
    """Entries duplicating one rate into both fields must not be billed twice."""
    _register_stt(
        "test-stt-both-rates",
        input_cost_per_second=1e-04,
        output_cost_per_second=1e-04,
    )

    prompt_cost, completion_cost = cost_per_second(
        model="test-stt-both-rates", custom_llm_provider="openai", duration=10.0
    )

    assert prompt_cost + completion_cost == pytest.approx(1e-03)


def test_output_rate_alone_still_bills():
    _register_stt("test-stt-output-only", output_cost_per_second=3e-05)

    prompt_cost, completion_cost = cost_per_second(
        model="test-stt-output-only", custom_llm_provider="openai", duration=60.0
    )

    assert prompt_cost == 0.0
    assert completion_cost == pytest.approx(1.8e-03)


@pytest.mark.parametrize(
    "model, provider",
    [
        ("deepgram/nova-3", "deepgram"),
        ("groq/whisper-large-v3", "groq"),
        ("elevenlabs/scribe_v1", "elevenlabs"),
        ("assemblyai/best", "assemblyai"),
        ("whisper-1", "openai"),
    ],
)
def test_shipped_per_second_models_bill_a_non_zero_cost(model, provider):
    prompt_cost, completion_cost = cost_per_second(model=model, custom_llm_provider=provider, duration=60.0)

    assert prompt_cost + completion_cost > 0.0


def test_whisper_bills_its_documented_rate_once():
    prompt_cost, completion_cost = cost_per_second(model="whisper-1", custom_llm_provider="openai", duration=30.0)

    assert prompt_cost + completion_cost == pytest.approx(0.003)
