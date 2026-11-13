"""Tests for per-second transcription cost calculation."""

from collections.abc import Callable
from typing import Final

import pytest

import litellm
from litellm.cost_calculator import default_video_cost_calculator
from litellm.llms.openai.cost_calculation import cost_per_second, video_clip_cost, video_generation_cost
from litellm.types.utils import ModelInfo
from litellm.types.videos.main import VideoObject


@pytest.mark.parametrize("calculator", [video_generation_cost, default_video_cost_calculator])
@pytest.mark.parametrize(
    ("duration", "resolution", "expected"),
    [
        (8.0, "720p", 0.6),
        (5.0, "540p", 0.1),
        (0.0, "720p", 0.0),
        (7.7, "720p", 0.6),
        (8.0, "1080p", 0.08),
        (3.0, "720p", 0.03),
    ],
)
def test_video_clip_price_precedes_per_second(
    calculator: Callable[..., float], duration: float, resolution: str, expected: float
) -> None:
    info: Final[ModelInfo] = {
        "output_cost_per_video_clip": {"8s": {"720p": 0.6}, "5s": {"720p": 0.3, "540p": 0.1}},
        "output_cost_per_second": 0.01,
    }
    assert calculator("test-clip", duration, model_info=info, video_resolution=resolution) == pytest.approx(expected)


@pytest.mark.parametrize("duration", [5.0, None])
@pytest.mark.parametrize("flat_price", [None, 0.0, 0.37])
def test_completion_cost_multiplies_applicable_video_price(
    duration: float | None, flat_price: float | None, local_model_cost_map: None
) -> None:
    litellm.register_model(
        {
            "fal_ai/test-video-clip": {
                "litellm_provider": "fal_ai",
                "mode": "video_generation",
                "output_cost_per_video_clip": {"5s": {"720p": 0.3}},
                "output_cost_per_video": flat_price,
            }
        },
        persist_across_reloads=False,
    )
    response: Final = VideoObject(
        id="clip",
        object="video",
        status="queued",
        model="test-video-clip",
        usage={
            "video_resolution": "720p",
            "video_count": 3,
            **({"duration_seconds": duration} if duration is not None else {}),
        },
    )
    assert litellm.completion_cost(
        completion_response=response, model="test-video-clip", custom_llm_provider="fal_ai", call_type="create_video"
    ) == pytest.approx(0.9 if duration is not None else 3 * (flat_price or 0.0))


@pytest.mark.parametrize(
    "duration", [None, 0, 0.0, "0", "invalid", "auto", False, True, [], -1, float("nan"), float("inf")]
)
def test_video_clip_cost_does_not_guess_unknown_duration(duration: object) -> None:
    info: Final = {"output_cost_per_video_clip": {"0s": {"720p": 0.01}, "5s": {"720p": 0.3}, "8s": {"720p": 0.6}}}
    assert video_clip_cost(info, duration, "720p") is None


@pytest.mark.parametrize("table", [None, {}, {"5s": {"720p": "invalid"}}, {"invalid": {"720p": 0.3}}])
def test_video_clip_cost_ignores_empty_or_invalid_catalog_table(table: object) -> None:
    assert video_clip_cost({"output_cost_per_video_clip": table}, 5, "720p") is None


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
