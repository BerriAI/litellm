import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final, cast

import pytest

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    BilledTokenRates,
    CostCalculatorUtils,
    PromptTokensDetailsResult,
    TokenRates,
    _calculate_input_cost,
    _get_token_base_cost,
    _is_off_peak,
    _is_within_off_peak_window,
    apply_off_peak_pricing,
    apply_provider_cache_read_default,
    calculate_cache_writing_cost,
    generic_cost_per_token,
    get_billed_token_rates,
    get_token_type_cost_breakdown,
)
from litellm.llms.gemini.image_generation.cost_calculator import (
    cost_calculator as gemini_image_generation_cost_calculator,
)
from litellm.llms.vertex_ai.image_generation.cost_calculator import (
    cost_calculator as vertex_image_generation_cost_calculator,
)
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CompletionTokensDetailsWrapper,
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
    ModelInfo,
    PromptTokensDetailsWrapper,
    Usage,
)


@pytest.fixture
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.mark.parametrize("prompt_tokens", [100, 200000, 200001])
@pytest.mark.parametrize("read_rate", [None, 0.0, 0.25e-6])
@pytest.mark.parametrize("service_tier", [None, "priority"])
def test_missing_cache_read_rate_resolves_to_input_rate(prompt_tokens, read_rate, service_tier):
    info = {
        "input_cost_per_token": 3e-6,
        "input_cost_per_token_priority": 4e-6,
        "input_cost_per_token_above_200k_tokens": 6e-6,
        "input_cost_per_token_above_200k_tokens_priority": 8e-6,
        "output_cost_per_token": 1e-6,
        "cache_read_input_token_cost": read_rate,
    }
    usage = Usage(prompt_tokens=prompt_tokens, prompt_tokens_details={"cached_tokens": 100})
    billed = _get_token_base_cost(info, usage, service_tier=service_tier)
    prompt_cost, _ = generic_cost_per_token(
        "policy-fixture", usage, "openai", service_tier=service_tier, model_info=info
    )
    assert billed[4] == pytest.approx(read_rate if read_rate is not None else billed[0])
    assert prompt_cost == pytest.approx((prompt_tokens - 100) * billed[0] + 100 * billed[4])


def test_generic_cost_per_token_bills_cache_reads_at_input_rate_when_no_cache_read_rate() -> None:
    model_info: ModelInfo = {
        "key": "bare-model",
        "max_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "input_cost_per_token": 2.4e-7,
        "output_cost_per_token": 9.7e-7,
        "litellm_provider": "bedrock",
        "mode": "chat",
        "supported_openai_params": None,
    }
    usage = Usage(
        prompt_tokens=12928,
        completion_tokens=380,
        total_tokens=13308,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=12288),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="bare-model",
        usage=usage,
        custom_llm_provider="bedrock",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx(12928 * 2.4e-7)
    assert completion_cost == pytest.approx(380 * 9.7e-7)


def test_apply_provider_cache_read_default_only_derives_a_rate_for_fireworks() -> None:
    openai_info: ModelInfo = {"input_cost_per_token": 2e-6}
    fireworks_info: ModelInfo = {"input_cost_per_token": 2e-6}

    assert apply_provider_cache_read_default(openai_info, "openai") is openai_info
    assert apply_provider_cache_read_default(openai_info, None) is openai_info

    processed_fireworks_info = apply_provider_cache_read_default(fireworks_info, "fireworks_ai")

    assert processed_fireworks_info is not fireworks_info
    assert processed_fireworks_info["cache_read_input_token_cost"] == pytest.approx(2e-6 * 0.5)


def test_generic_cost_per_token_prefers_audio_per_second_rate() -> None:
    model_info: ModelInfo = {
        "key": "gemini-embedding-2",
        "max_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "input_cost_per_token": 2e-7,
        "input_cost_per_audio_token": 6.5e-6,
        "input_cost_per_audio_per_second": 0.00016,
        "output_cost_per_token": 0.0,
        "litellm_provider": "vertex_ai",
        "mode": "embedding",
        "supported_openai_params": None,
    }
    usage = Usage(
        prompt_tokens=64,
        completion_tokens=0,
        total_tokens=64,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=64,
            audio_length_seconds=2,
        ),
    )

    prompt_cost, _ = generic_cost_per_token(
        model="gemini-embedding-2",
        usage=usage,
        custom_llm_provider="vertex_ai",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx(2 * 0.00016)


def test_generic_cost_per_token_prefers_image_per_image_rate() -> None:
    model_info: ModelInfo = {
        "key": "gemini-embedding-2",
        "max_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "input_cost_per_token": 2e-7,
        "input_cost_per_image_token": 4.5e-7,
        "input_cost_per_image": 0.00012,
        "output_cost_per_token": 0.0,
        "litellm_provider": "vertex_ai",
        "mode": "embedding",
        "supported_openai_params": None,
    }
    usage = Usage(
        prompt_tokens=258,
        completion_tokens=0,
        total_tokens=258,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            image_tokens=258,
            image_count=1,
        ),
    )

    prompt_cost, _ = generic_cost_per_token(
        model="gemini-embedding-2",
        usage=usage,
        custom_llm_provider="vertex_ai",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx(0.00012)


def test_generic_cost_per_token_prefers_video_per_second_rate() -> None:
    model_info: ModelInfo = {
        "key": "gemini-embedding-2",
        "max_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "input_cost_per_token": 2e-7,
        "input_cost_per_video_token": 1.2e-5,
        "input_cost_per_video_per_second": 0.00079,
        "output_cost_per_token": 0.0,
        "litellm_provider": "vertex_ai",
        "mode": "embedding",
        "supported_openai_params": None,
    }
    usage = Usage(
        prompt_tokens=516,
        completion_tokens=0,
        total_tokens=516,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            video_tokens=516,
            video_length_seconds=2,
        ),
    )

    prompt_cost, _ = generic_cost_per_token(
        model="gemini-embedding-2",
        usage=usage,
        custom_llm_provider="vertex_ai",
        model_info=model_info,
    )

    assert prompt_cost == pytest.approx(2 * 0.00079)


def test_missing_cache_read_uses_off_peak_input_rate():
    from datetime import datetime, timezone

    info = {
        "input_cost_per_token": 3e-6,
        "off_peak_pricing": {"hours_utc": "00:00-23:59", "input_cost_per_token": 5e-6},
    }
    when = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    billed = _get_token_base_cost(info, Usage(prompt_tokens=100), current_time=when)
    assert billed[0] == billed[4] == 5e-6


def test_reasoning_tokens_no_price_set(_local_model_cost_map):
    # Use o1 - o1-mini was deprecated/renamed; o1 has same reasoning-token semantics
    # (no separate output_cost_per_reasoning_token, so all completion tokens use output_cost_per_token)
    model = "o1"
    model_cost_map = litellm.model_cost[model]
    usage = Usage(
        completion_tokens=1578,
        prompt_tokens=17,
        total_tokens=1595,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=952,
            rejected_prediction_tokens=None,
            text_tokens=626,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=17, image_tokens=None
        ),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    expected_completion_cost = model_cost_map["output_cost_per_token"] * usage.completion_tokens
    assert round(completion_cost, 10) == round(
        expected_completion_cost,
        10,
    )


def test_reasoning_tokens_gemini(_local_model_cost_map):
    model = "gemini-2.5-flash"
    custom_llm_provider = "gemini"

    usage = Usage(
        completion_tokens=1578,
        prompt_tokens=17,
        total_tokens=1595,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=952,
            rejected_prediction_tokens=None,
            text_tokens=626,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=17, image_tokens=None
        ),
    )
    model_cost_map = litellm.model_cost[model]
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        (model_cost_map["output_cost_per_token"] * usage.completion_tokens_details.text_tokens)
        + (model_cost_map["output_cost_per_reasoning_token"] * usage.completion_tokens_details.reasoning_tokens),
        10,
    )




def test_image_tokens_with_custom_pricing():
    """Test that image_tokens in completion are properly costed with output_cost_per_image_token."""
    from unittest.mock import patch

    # Mock model info with image token pricing
    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "output_cost_per_image_token": 5e-6,  # Custom pricing for image tokens in output
    }

    usage = Usage(
        completion_tokens=1720,  # text_tokens (600) + image_tokens (1120)
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=600,
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=14, image_tokens=None
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    # Expected costs:
    # Prompt: 14 * 1e-6
    # Completion: (600 * 2e-6) + (1120 * 5e-6)
    expected_prompt_cost = 14 * 1e-6
    expected_completion_cost = (600 * 2e-6) + (1120 * 5e-6)

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_image_tokens_fallback_to_base_cost():
    """Test that image_tokens fall back to base cost when output_cost_per_image_token is not set."""
    from unittest.mock import patch

    # Mock model info without image token pricing
    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        # No output_cost_per_image_token defined
    }

    usage = Usage(
        completion_tokens=1720,
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=600,
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=14, image_tokens=None
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    # Expected costs:
    # Prompt: 14 * 1e-6
    # Completion: (600 * 2e-6) + (1120 * 2e-6)  # image_tokens use base cost
    expected_prompt_cost = 14 * 1e-6
    expected_completion_cost = (600 * 2e-6) + (1120 * 2e-6)

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_video_input_tokens_gemini_omni_flash_preview(_local_model_cost_map):
    """Video input tokens are billed at the standard input rate instead of being dropped."""
    model = "gemini-omni-flash-preview"

    usage = Usage(
        completion_tokens=10,
        prompt_tokens=10050,
        total_tokens=10060,
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=10),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=50, video_tokens=10000),
    )
    model_cost_map = litellm.model_cost[f"gemini/{model}"]

    prompt_cost, _ = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="gemini",
    )

    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token"] * usage.prompt_tokens,
        10,
    )


def test_video_tokens_fallback_to_base_cost():
    """Video output tokens fall back to the base output rate when output_cost_per_video_token is not set."""
    from unittest.mock import patch

    mock_model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
    }

    usage = Usage(
        completion_tokens=1720,
        prompt_tokens=14,
        total_tokens=1734,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            text_tokens=600,
            video_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=14),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="gemini"
        )

    assert round(prompt_cost, 12) == round(14 * 1e-6, 12)
    assert round(completion_cost, 12) == round((600 + 1120) * 2e-6, 12)


def test_generic_cost_per_token_above_200k_tokens(_local_model_cost_map):
    # gemini-2.5-pro-exp-03-25 was removed; gemini-2.5-pro has same above-200k pricing
    model = "gemini-2.5-pro"
    custom_llm_provider = "vertex_ai"

    model_cost_map = litellm.model_cost[model]
    prompt_tokens = 220 * 1e6
    completion_tokens = 150
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    assert round(prompt_cost, 10) == round(
        model_cost_map["input_cost_per_token_above_200k_tokens"] * usage.prompt_tokens,
        10,
    )
    assert round(completion_cost, 10) == round(
        model_cost_map["output_cost_per_token_above_200k_tokens"] * usage.completion_tokens,
        10,
    )


def test_get_token_base_cost_picks_highest_crossed_tier():
    """Regression test for #30345.

    With graduated tiers at 90k and 128k whose keys have different digit lengths, a request
    crossing both must be billed at the highest tier it crosses (128k), not the lower one that
    happens to sort first lexicographically.
    """
    model_info = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_above_90k_tokens": 5e-6,
        "input_cost_per_token_above_128k_tokens": 9e-6,
    }
    usage = Usage(prompt_tokens=150_000, completion_tokens=10, total_tokens=150_010)

    prompt_base_cost = _get_token_base_cost(model_info, usage)[0]

    assert prompt_base_cost == 9e-6


def test_is_within_off_peak_window_same_day():
    from datetime import datetime, timezone

    window = "09:00-17:00"
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 8, 59, tzinfo=timezone.utc)) is False
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 17, 0, tzinfo=timezone.utc)) is False


def test_is_within_off_peak_window_wraps_midnight():
    from datetime import datetime, timezone

    window = "16:30-00:30"
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 16, 30, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)) is False
    assert _is_within_off_peak_window(window, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)) is False


def test_is_within_off_peak_window_equal_start_and_end_covers_whole_day():
    """An equal start and end is the natural way to spell off-peak all day. It used to take the
    non-wrap branch, where start <= now < end can never hold, so it matched nothing and billed at
    standard rates around the clock without raising or logging anything."""
    from datetime import datetime, timezone

    for window in ("00:00-00:00", "10:00-10:00"):
        for hour in range(24):
            assert _is_within_off_peak_window(window, datetime(2026, 1, 1, hour, 0, tzinfo=timezone.utc)) is True, (
                f"{window} should cover {hour:02d}:00"
            )


def test_is_within_off_peak_window_multiple_windows():
    from datetime import datetime, timezone

    # Providers like DeepSeek V4 have more than one daily peak/off-peak window.
    windows = ["01:00-05:00", "13:00-16:00"]
    assert _is_within_off_peak_window(windows, datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window(windows, datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window(windows, datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)) is False
    # a malformed entry in the list is ignored, valid entries still match
    assert _is_within_off_peak_window(["bad", "13:00-16:00"], datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)) is True
    assert _is_within_off_peak_window([], datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)) is False


def test_is_within_off_peak_window_normalizes_timezone_aware_input():
    from datetime import datetime, timedelta, timezone

    # A caller may pass a non-UTC aware datetime; the window is UTC and must be
    # evaluated in UTC, not against the caller's wall-clock. 09:00 at UTC+8 is
    # 01:00 UTC, inside the 01:00-05:00 window.
    tz_plus_8 = timezone(timedelta(hours=8))
    assert _is_within_off_peak_window("01:00-05:00", datetime(2026, 1, 1, 9, 0, tzinfo=tz_plus_8)) is True
    assert _is_within_off_peak_window("01:00-05:00", datetime(2026, 1, 1, 12, 0, tzinfo=tz_plus_8)) is True
    # 06:00 at UTC+8 is 22:00 UTC the previous day, outside the window
    assert _is_within_off_peak_window("01:00-05:00", datetime(2026, 1, 1, 6, 0, tzinfo=tz_plus_8)) is False


def test_is_within_off_peak_window_malformed_returns_false():
    from datetime import datetime, timezone

    now = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
    assert _is_within_off_peak_window("not-a-window", now) is False
    assert _is_within_off_peak_window("16:30", now) is False
    assert _is_within_off_peak_window("25:00-26:00", now) is False


def test_is_off_peak_weekday_qualified_windows_deepseek_schedule():
    """DeepSeek since 2026-08-23: peak is 01:00-04:00 and 06:00-10:00 UTC on weekdays only, with
    weekends off-peak around the clock. The weekday axis is not a filter on one window set; on
    two days of seven the off-peak window becomes the whole day, so the schedule needs two
    day-qualified rules. The weekend instants inside would-be peak hours are the ones a
    time-only implementation bills wrong."""
    from datetime import datetime, timezone

    deepseek = {
        "windows": [
            {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
            {"hours_utc": "00:00-00:00", "weekdays": [6, 7]},
        ],
    }
    peak_instants = [
        datetime(2026, 8, 24, 1, 30, tzinfo=timezone.utc),
        datetime(2026, 8, 26, 7, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 28, 9, 59, tzinfo=timezone.utc),
    ]
    off_peak_instants = [
        datetime(2026, 8, 23, 1, 30, tzinfo=timezone.utc),
        datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 26, 5, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 28, 16, 30, tzinfo=timezone.utc),
        datetime(2026, 8, 24, 0, 30, tzinfo=timezone.utc),
    ]
    for when in peak_instants:
        assert _is_off_peak(deepseek, when) is False, f"{when.isoformat()} should bill peak"
    for when in off_peak_instants:
        assert _is_off_peak(deepseek, when) is True, f"{when.isoformat()} should bill off-peak"


def test_is_off_peak_holiday_weekday_is_off_peak_all_day():
    """A rule whose override_dates names a public holiday applies its off-peak window all
    day, including inside the peak hours a plain weekday would bill at standard rates."""
    from datetime import datetime, timezone

    deepseek = {
        "windows": [
            {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
            {"hours_utc": "00:00-00:00", "weekdays": [6, 7]},
            {"hours_utc": "00:00-00:00", "override_dates": ["2026-01-01", "2026-10-01"]},
            {
                "hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"],
                "override_dates": ["2026-01-04", "2026-02-14"],
            },
        ],
    }
    holiday = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
    normal_thursday = datetime(2026, 1, 8, 2, 0, tzinfo=timezone.utc)
    assert _is_off_peak(deepseek, holiday) is True, f"{holiday.isoformat()} is a public holiday"
    assert _is_off_peak(deepseek, normal_thursday) is False, (
        f"{normal_thursday.isoformat()} is a normal Thursday peak hour"
    )


def test_is_off_peak_make_up_weekend_day_uses_override_hours():
    """On a make-up workday only the rule listing the date decides: the weekend whole-day
    rule does not fire, and the make-up rule's weekday off-peak hours apply instead."""
    from datetime import datetime, timezone

    deepseek = {
        "windows": [
            {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
            {"hours_utc": "00:00-00:00", "weekdays": [6, 7]},
            {"hours_utc": "00:00-00:00", "override_dates": ["2026-01-01", "2026-10-01"]},
            {
                "hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"],
                "override_dates": ["2026-01-04", "2026-02-14"],
            },
        ],
    }
    make_up_sunday_peak = datetime(2026, 1, 4, 2, 0, tzinfo=timezone.utc)
    make_up_sunday_off_peak = datetime(2026, 1, 4, 5, 0, tzinfo=timezone.utc)
    plain_sunday = datetime(2026, 1, 11, 2, 0, tzinfo=timezone.utc)
    assert _is_off_peak(deepseek, make_up_sunday_peak) is False, (
        f"{make_up_sunday_peak.isoformat()} is a make-up workday peak hour"
    )
    assert _is_off_peak(deepseek, make_up_sunday_off_peak) is True, (
        f"{make_up_sunday_off_peak.isoformat()} is inside the make-up rule's off-peak hours"
    )
    assert _is_off_peak(deepseek, plain_sunday) is True, (
        f"{plain_sunday.isoformat()} is a normal Sunday, off-peak all day"
    )


def test_is_off_peak_override_skips_flat_hours_and_other_rules():
    """On a date an override rule lists, the flat hours_utc and every other rule are
    skipped: only the matching rules' windows decide."""
    from datetime import datetime, timezone

    block = {
        "hours_utc": "00:00-00:00",
        "windows": [
            {"hours_utc": "00:00-00:00", "weekdays": [1, 2, 3, 4, 5, 6, 7]},
            {"override_dates": ["2026-03-03"], "hours_utc": "12:00-13:00"},
        ],
    }
    outside_window = datetime(2026, 3, 3, 2, 0, tzinfo=timezone.utc)
    inside_window = datetime(2026, 3, 3, 12, 30, tzinfo=timezone.utc)
    next_day = datetime(2026, 3, 4, 2, 0, tzinfo=timezone.utc)
    assert _is_off_peak(block, outside_window) is False, (
        "the override rule's window is closed and nothing else applies on its date"
    )
    assert _is_off_peak(block, inside_window) is True, "the override rule's window is open"
    assert _is_off_peak(block, next_day) is True, (
        "the next day falls back to the flat hours_utc and weekday rules"
    )


def test_is_off_peak_override_rule_does_not_leak_onto_other_dates():
    """A rule carrying override_dates never participates on a date it does not list."""
    from datetime import datetime, timezone

    block = {"windows": [{"override_dates": ["2026-03-03"], "hours_utc": "00:00-00:00"}]}
    assert _is_off_peak(block, datetime(2026, 3, 3, 2, 0, tzinfo=timezone.utc)) is True
    assert _is_off_peak(block, datetime(2026, 3, 4, 2, 0, tzinfo=timezone.utc)) is False


def test_is_off_peak_override_dates_read_on_weekday_timezone_calendar():
    """Override dates are read on the weekday_timezone calendar, so the Shanghai date can
    diverge from the UTC date over 16:00-24:00 UTC."""
    from datetime import datetime, timezone

    shanghai = {
        "weekday_timezone": "Asia/Shanghai",
        "windows": [{"override_dates": ["2026-01-01"], "hours_utc": "00:00-00:00"}],
    }
    utc_block = {"windows": [{"override_dates": ["2026-01-01"], "hours_utc": "00:00-00:00"}]}
    in_shanghai = datetime(2025, 12, 31, 16, 30, tzinfo=timezone.utc)
    before_shanghai = datetime(2025, 12, 31, 15, 30, tzinfo=timezone.utc)
    assert _is_off_peak(shanghai, in_shanghai) is True, (
        f"{in_shanghai.isoformat()} is already 2026-01-01 in Shanghai"
    )
    assert _is_off_peak(shanghai, before_shanghai) is False, (
        f"{before_shanghai.isoformat()} is still 2025-12-31 in Shanghai"
    )
    assert _is_off_peak(utc_block, in_shanghai) is False, (
        f"{in_shanghai.isoformat()} is still 2025-12-31 on the UTC calendar"
    )


def test_is_off_peak_ignores_malformed_override_dates():
    """A bare string, non-string entries, or an unparseable date disable the rule carrying
    them, so malformed override_dates can never silently widen the off-peak hours."""
    from datetime import datetime, timezone

    peak_instant = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    peak_windows = [{"hours_utc": "00:30-01:00", "weekdays": [1, 2, 3, 4, 5]}]
    for bad in (
        "2026-01-01",
        [20260101, None],
        ["2026-1-1"],
        ["2026-W01-4"],
        ["2026-01-1 "],
    ):
        block = {"windows": peak_windows + [{"hours_utc": "00:00-00:00", "override_dates": bad}]}
        assert _is_off_peak(block, peak_instant) is False, (
            f"override_dates={bad!r} is malformed and must disable its rule"
        )


def test_is_off_peak_override_rule_without_valid_hours_is_disabled():
    """An override rule whose hours_utc does not parse never applies, even on a listed
    date: the flat hours_utc still decides instead of the broken rule billing peak all day."""
    from datetime import datetime, timezone

    block = {
        "hours_utc": "00:00-00:00",
        "windows": [
            {"override_dates": ["2026-03-03"], "hours_utc": 5},
            {"override_dates": ["2026-03-03"]},
        ],
    }
    assert _is_off_peak(block, datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc)) is True


def test_is_off_peak_disables_rules_with_partially_malformed_override_dates():
    """One bad entry disables the whole override_dates list: a rule whose dates do not all
    parse never applies on any date, so the flat hours decide alone."""
    from datetime import datetime, timezone

    peak_instant = datetime(2026, 3, 4, 12, 0, tzinfo=timezone.utc)
    for bad in (["2026-03-04", 20260304], ["2026-03-04", "2026-3-5"]):
        block = {
            "hours_utc": "00:30-01:00",
            "windows": [{"hours_utc": "00:00-00:00", "override_dates": bad}],
        }
        assert _is_off_peak(block, peak_instant) is False, (
            f"override_dates={bad!r} must disable its rule, leaving only the closed flat window"
        )


def test_is_off_peak_without_override_dates_is_unchanged():
    """The plain DeepSeek windows with no override_dates bill a holiday peak hour at
    standard rates."""
    from datetime import datetime, timezone

    deepseek = {
        "windows": [
            {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
            {"hours_utc": "00:00-00:00", "weekdays": [6, 7]},
        ],
    }
    assert _is_off_peak(deepseek, datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)) is False


def test_override_dates_use_weekday_timezone_calendar_on_shipped_deepseek_rows():
    """Every shipped DeepSeek row with off_peak_pricing carries the same block as
    deepseek/deepseek-flash: the 2026 PRC holidays and make-up workdays, read on the
    Asia/Shanghai calendar. The make-up Sunday bills weekday hours and a holiday is
    off-peak all day. Dates per the State Council 2026 notice,
    https://www.gov.cn/zhengce/zhengceku/202511/content_7047091.htm, read 2026-09-24."""
    from datetime import datetime, timezone

    import litellm

    block: Final = litellm.model_cost["deepseek/deepseek-flash"]["off_peak_pricing"]
    assert block.get("weekday_timezone") == "Asia/Shanghai"
    deepseek_rows: Final = {
        name: entry["off_peak_pricing"]
        for name, entry in litellm.model_cost.items()
        if name.startswith("deepseek")
        and not name.startswith("openrouter/")
        and isinstance(entry, dict)
        and "off_peak_pricing" in entry
    }
    assert deepseek_rows, "expected at least the deepseek/deepseek-flash row"
    drifted: Final = [
        name
        for name, row in deepseek_rows.items()
        if row.get("windows") != block["windows"] or row.get("weekday_timezone") != block["weekday_timezone"]
    ]
    assert not drifted, f"off_peak schedule drift on {drifted}"

    shanghai_make_up_sunday = datetime(2026, 1, 3, 17, 0, tzinfo=timezone.utc)
    make_up_sunday_peak = datetime(2026, 1, 4, 2, 0, tzinfo=timezone.utc)
    plain_sunday = datetime(2026, 1, 11, 2, 0, tzinfo=timezone.utc)
    holiday = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
    assert _is_off_peak(block, shanghai_make_up_sunday) is True, (
        f"{shanghai_make_up_sunday.isoformat()} is 2026-01-04 in Shanghai, a make-up workday"
    )
    assert _is_off_peak(block, make_up_sunday_peak) is False, (
        f"{make_up_sunday_peak.isoformat()} is a make-up workday peak hour"
    )
    assert _is_off_peak(block, plain_sunday) is True
    assert _is_off_peak(block, holiday) is True


def test_get_token_base_cost_override_dates_bill_holiday_rates():
    """End to end through _get_token_base_cost: a date listed on an override rule bills the
    off-peak input rate during what would be a weekday peak hour."""
    from datetime import datetime, timezone
    from typing import cast

    from litellm.types.utils import ModelInfo

    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "off_peak_pricing": {
                "windows": [
                    {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
                    {"hours_utc": "00:00-00:00", "weekdays": [6, 7]},
                    {"hours_utc": "00:00-00:00", "override_dates": ["2026-01-01"]},
                ],
                "input_cost_per_token": 5e-7,
                "output_cost_per_token": 1e-6,
            },
        },
    )
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    holiday_peak_hours = _get_token_base_cost(
        model_info, usage, current_time=datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)
    )
    assert holiday_peak_hours[:2] == (5e-7, 1e-6)

    normal_thursday_same_hours = _get_token_base_cost(
        model_info, usage, current_time=datetime(2026, 1, 8, 2, 0, tzinfo=timezone.utc)
    )
    assert normal_thursday_same_hours[:2] == (1e-6, 2e-6)


def test_is_off_peak_weekday_timezone_reads_vendor_calendar():
    """The UTC and Asia/Shanghai calendars only disagree about the date over 16:00-24:00 UTC, so
    a window in that stretch is the one place a vendor-local weekday differs from a UTC one:
    2026-08-28T16:30Z is Friday in UTC but already Saturday in Beijing."""
    from datetime import datetime, timezone

    shanghai_saturday = {
        "weekday_timezone": "Asia/Shanghai",
        "windows": [{"hours_utc": "16:00-17:00", "weekdays": [6]}],
    }
    assert _is_off_peak(shanghai_saturday, datetime(2026, 8, 28, 16, 30, tzinfo=timezone.utc)) is True
    assert _is_off_peak(shanghai_saturday, datetime(2026, 8, 29, 16, 30, tzinfo=timezone.utc)) is False


def test_is_off_peak_weekdays_default_utc_calendar_and_accept_names():
    from datetime import datetime, timezone

    named_weekend = {"windows": [{"hours_utc": "00:00-00:00", "weekdays": ["Sat", "sunday"]}]}
    assert _is_off_peak(named_weekend, datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)) is True
    assert _is_off_peak(named_weekend, datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)) is False

    utc_friday = {"windows": [{"hours_utc": "16:00-17:00", "weekdays": [5]}]}
    assert _is_off_peak(utc_friday, datetime(2026, 8, 28, 16, 30, tzinfo=timezone.utc)) is True
    assert _is_off_peak(utc_friday, datetime(2026, 8, 29, 16, 30, tzinfo=timezone.utc)) is False


def test_is_off_peak_naive_current_time_read_as_utc():
    from datetime import datetime

    block = {"windows": [{"hours_utc": "16:00-17:00", "weekdays": [5]}]}
    assert _is_off_peak(block, datetime(2026, 8, 28, 16, 30)) is True
    assert _is_off_peak(block, datetime(2026, 8, 29, 16, 30)) is False


def test_is_off_peak_invalid_weekday_timezone_falls_back_to_utc():
    from datetime import datetime, timezone

    block = {"weekday_timezone": "Not/AZone", "windows": [{"hours_utc": "16:00-17:00", "weekdays": [5]}]}
    assert _is_off_peak(block, datetime(2026, 8, 28, 16, 30, tzinfo=timezone.utc)) is True
    assert _is_off_peak(block, datetime(2026, 8, 29, 16, 30, tzinfo=timezone.utc)) is False


def test_is_off_peak_ignores_malformed_weekday_rules():
    from datetime import datetime, timezone

    when = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
    assert _is_off_peak({"windows": [{"hours_utc": "00:00-00:00", "weekdays": []}]}, when) is False
    assert _is_off_peak({"windows": [{"hours_utc": "00:00-00:00", "weekdays": [0, 8, "noday", True]}]}, when) is False
    assert _is_off_peak({"windows": [{"weekdays": [6]}]}, when) is False
    assert _is_off_peak({"windows": [{"hours_utc": 1630}]}, when) is False
    assert _is_off_peak({"windows": ["00:00-00:00"]}, when) is False
    assert _is_off_peak({"windows": "00:00-00:00"}, when) is False
    assert _is_off_peak({"hours_utc": 1630}, when) is False
    assert _is_off_peak({}, when) is False


def test_is_off_peak_flat_hours_and_windows_are_a_union():
    from datetime import datetime, timezone

    block = {
        "hours_utc": "04:00-06:00",
        "windows": [{"hours_utc": "00:00-00:00", "weekdays": [7]}],
    }
    assert _is_off_peak(block, datetime(2026, 8, 28, 5, 0, tzinfo=timezone.utc)) is True
    assert _is_off_peak(block, datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)) is True
    assert _is_off_peak(block, datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)) is False


def test_get_token_base_cost_weekend_only_off_peak_rate():
    from datetime import datetime, timezone
    from typing import cast

    from litellm.types.utils import ModelInfo

    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "off_peak_pricing": {
                "windows": [
                    {"hours_utc": ["00:00-01:00", "04:00-06:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]},
                    {"hours_utc": "00:00-00:00", "weekdays": [6, 7]},
                ],
                "input_cost_per_token": 5e-7,
                "output_cost_per_token": 1e-6,
            },
        },
    )
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    saturday_peak_hours = _get_token_base_cost(
        model_info, usage, current_time=datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc)
    )
    assert saturday_peak_hours[:2] == (5e-7, 1e-6)

    monday_same_hours = _get_token_base_cost(
        model_info, usage, current_time=datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc)
    )
    assert monday_same_hours[:2] == (1e-6, 2e-6)


def test_get_token_base_cost_applies_off_peak_pricing():
    from datetime import datetime, timezone
    from typing import cast

    from litellm.types.utils import ModelInfo

    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-7,
            "off_peak_pricing": {
                "hours_utc": "16:30-00:30",
                "input_cost_per_token": 5e-7,
                "output_cost_per_token": 1e-6,
                "cache_read_input_token_cost": 5e-8,
            },
        },
    )
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    off_peak = _get_token_base_cost(model_info, usage, current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc))
    assert off_peak[0] == 5e-7
    assert off_peak[1] == 1e-6
    assert off_peak[4] == 5e-8

    peak = _get_token_base_cost(model_info, usage, current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    assert peak[0] == 1e-6
    assert peak[1] == 2e-6
    assert peak[4] == 1e-7


def test_get_token_base_cost_non_mapping_off_peak_block_bills_standard_rates():
    """A truthy non-mapping off_peak_pricing value (a bare string or a list in
    YAML) must bill standard rates rather than raising, matching how every
    other malformed piece of the block behaves.
    """
    from datetime import datetime, timezone
    from typing import cast

    from litellm.types.utils import ModelInfo

    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    when = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)

    for malformed_block in ("16:00-19:00", ["16:00-19:00"], 5e-7, True):
        model_info = cast(
            ModelInfo,
            {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "off_peak_pricing": malformed_block,
            },
        )
        result = _get_token_base_cost(model_info, usage, current_time=when)
        assert result[0] == 1e-6
        assert result[1] == 2e-6


def test_get_token_base_cost_off_peak_falls_back_to_standard_when_unset():
    from datetime import datetime, timezone
    from typing import cast

    from litellm.types.utils import ModelInfo

    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "off_peak_pricing": {"hours_utc": "16:30-00:30", "input_cost_per_token": 5e-7},
        },
    )
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    result = _get_token_base_cost(model_info, usage, current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc))
    assert result[0] == 5e-7
    assert result[1] == 2e-6


def test_get_token_base_cost_off_peak_wins_over_threshold():
    from datetime import datetime, timezone
    from typing import cast

    from litellm.types.utils import ModelInfo

    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "input_cost_per_token_above_200k_tokens": 3e-6,
            "output_cost_per_token_above_200k_tokens": 4e-6,
            "off_peak_pricing": {
                "hours_utc": "16:30-00:30",
                "input_cost_per_token": 5e-7,
                "output_cost_per_token": 1e-6,
            },
        },
    )
    usage = Usage(prompt_tokens=250000, completion_tokens=250000, total_tokens=500000)

    off_peak = _get_token_base_cost(model_info, usage, current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc))
    assert off_peak[0] == 5e-7
    assert off_peak[1] == 1e-6

    peak = _get_token_base_cost(model_info, usage, current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    assert peak[0] == 3e-6
    assert peak[1] == 4e-6


def test_get_model_info_propagates_off_peak_fields():
    model_name = "test-off-peak-model"
    off_peak_pricing = {
        "hours_utc": "16:30-00:30",
        "input_cost_per_token": 5e-7,
        "output_cost_per_token": 1e-6,
        "cache_read_input_token_cost": 5e-8,
    }
    litellm.register_model(
        {
            model_name: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "off_peak_pricing": off_peak_pricing,
            }
        }
    )
    info = litellm.get_model_info(model=model_name)
    assert info["off_peak_pricing"] == off_peak_pricing


def test_get_token_base_cost_off_peak_wins_over_tiered_pricing():
    """Tiered pricing resolves base rates on its own path and returns early, so off-peak has to
    be applied there too or a model carrying both would silently bill the tier rate all day."""
    from datetime import datetime, timezone

    model_name = "litellm-test-off-peak-tiered"
    litellm.register_model(
        {
            model_name: {
                "litellm_provider": "openai",
                "mode": "chat",
                "tiered_pricing": [
                    {"range": [0, 128000], "input_cost_per_token": 3e-6, "output_cost_per_token": 6e-6},
                ],
                "off_peak_pricing": {
                    "hours_utc": "16:30-00:30",
                    "input_cost_per_token": 5e-7,
                    "output_cost_per_token": 1e-6,
                },
            }
        }
    )
    info = litellm.get_model_info(model=model_name)
    usage = Usage(prompt_tokens=1_000, completion_tokens=100, total_tokens=1_100)

    inside = _get_token_base_cost(info, usage, current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc))
    assert inside[:2] == (5e-7, 1e-6)

    outside = _get_token_base_cost(info, usage, current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    assert outside[:2] == (3e-6, 6e-6)


def _register_off_peak_reasoning_model(
    model_name: str, off_peak_pricing: dict, reasoning_rate: float | None = 4e-6, **service_tier_rates: float
) -> None:
    reasoning_entry = {} if reasoning_rate is None else {"output_cost_per_reasoning_token": reasoning_rate}
    litellm.register_model(
        {
            model_name: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "cache_read_input_token_cost": 1e-7,
                "cache_creation_input_token_cost": 1.25e-6,
                "off_peak_pricing": off_peak_pricing,
                **reasoning_entry,
                **service_tier_rates,
            }
        }
    )


def _off_peak_reasoning_usage() -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=80,
        total_tokens=180,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=30, text_tokens=50),
    )


def test_generic_cost_per_token_off_peak_reasoning_rate():
    from datetime import datetime, timezone

    model_name = "litellm-test-off-peak-reasoning"
    _register_off_peak_reasoning_model(
        model_name,
        {"hours_utc": "16:30-00:30", "output_cost_per_token": 1e-6, "output_cost_per_reasoning_token": 5e-7},
    )

    _, inside = generic_cost_per_token(
        model=model_name,
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc),
    )
    assert inside == pytest.approx(50 * 1e-6 + 30 * 5e-7)

    _, outside = generic_cost_per_token(
        model=model_name,
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert outside == pytest.approx(50 * 2e-6 + 30 * 4e-6)


def test_generic_cost_per_token_off_peak_block_without_reasoning_rate():
    from datetime import datetime, timezone

    inside_window = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)
    block = {"hours_utc": "16:30-00:30", "output_cost_per_token": 1e-6}

    _register_off_peak_reasoning_model("litellm-test-off-peak-model-reasoning-rate", block)
    _, with_model_rate = generic_cost_per_token(
        model="litellm-test-off-peak-model-reasoning-rate",
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        current_time=inside_window,
    )
    assert with_model_rate == pytest.approx(50 * 1e-6 + 30 * 4e-6)

    _register_off_peak_reasoning_model("litellm-test-off-peak-no-reasoning-rate", block, reasoning_rate=None)
    _, without_model_rate = generic_cost_per_token(
        model="litellm-test-off-peak-no-reasoning-rate",
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        current_time=inside_window,
    )
    assert without_model_rate == pytest.approx(80 * 1e-6)


def test_generic_cost_per_token_off_peak_reasoning_rate_wins_over_the_tier():
    from datetime import datetime, timezone

    model_name = "litellm-test-off-peak-tiered-reasoning"
    litellm.register_model(
        {
            model_name: {
                "litellm_provider": "openai",
                "mode": "chat",
                "tiered_pricing": [
                    {
                        "range": [0, 128000],
                        "input_cost_per_token": 3e-6,
                        "output_cost_per_token": 6e-6,
                        "output_cost_per_reasoning_token": 8e-6,
                    },
                ],
                "off_peak_pricing": {
                    "hours_utc": "16:30-00:30",
                    "output_cost_per_token": 1e-6,
                    "output_cost_per_reasoning_token": 5e-7,
                },
            }
        }
    )

    _, inside = generic_cost_per_token(
        model=model_name,
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc),
    )
    assert inside == pytest.approx(50 * 1e-6 + 30 * 5e-7)

    _, outside = generic_cost_per_token(
        model=model_name,
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert outside == pytest.approx(50 * 6e-6 + 30 * 8e-6)


def test_generic_cost_per_token_off_peak_reasoning_rate_wins_over_the_service_tier():
    from datetime import datetime, timezone

    model_name = "litellm-test-off-peak-reasoning-service-tier"
    _register_off_peak_reasoning_model(
        model_name,
        {"hours_utc": "16:30-00:30", "output_cost_per_token": 1e-6, "output_cost_per_reasoning_token": 5e-7},
        output_cost_per_token_priority=3e-6,
        output_cost_per_reasoning_token_priority=6e-6,
    )

    _, inside = generic_cost_per_token(
        model=model_name,
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        service_tier="priority",
        current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc),
    )
    assert inside == pytest.approx(50 * 1e-6 + 30 * 5e-7)

    _, outside = generic_cost_per_token(
        model=model_name,
        usage=_off_peak_reasoning_usage(),
        custom_llm_provider="openai",
        service_tier="priority",
        current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert outside == pytest.approx(50 * 3e-6 + 30 * 6e-6)


def test_apply_off_peak_pricing_treats_bool_as_unset_and_parses_strings():
    from datetime import datetime, timezone

    model_name = "litellm-test-off-peak-odd-values"
    _register_off_peak_reasoning_model(
        model_name,
        {
            "hours_utc": "16:30-00:30",
            "cache_creation_input_token_cost": True,
            "output_cost_per_reasoning_token": "5e-7",
        },
    )
    standard = TokenRates(
        input_rate=1e-6, output_rate=2e-6, cache_read_rate=1e-7, cache_creation_rate=1.25e-6, reasoning_rate=4e-6
    )

    rates = apply_off_peak_pricing(
        litellm.get_model_info(model_name, custom_llm_provider="openai"),
        datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc),
        standard,
    )
    assert rates.cache_creation_rate == 1.25e-6
    assert rates.reasoning_rate == 5e-7


def test_get_token_base_cost_off_peak_cache_creation_rate():
    from datetime import datetime, timezone
    from typing import cast

    from litellm.types.utils import ModelInfo

    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_creation_input_token_cost": 1.25e-6,
            "cache_creation_input_token_cost_above_1hr": 2e-6,
            "off_peak_pricing": {"hours_utc": "16:30-00:30", "cache_creation_input_token_cost": 5e-7},
        },
    )
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    inside_window = datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc)

    inside = _get_token_base_cost(model_info, usage, current_time=inside_window)
    assert inside[2] == 5e-7
    assert inside[3] == 2e-6

    outside = _get_token_base_cost(model_info, usage, current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    assert outside[2] == 1.25e-6

    without_key = cast(
        ModelInfo,
        {**model_info, "off_peak_pricing": {"hours_utc": "16:30-00:30", "input_cost_per_token": 5e-7}},
    )
    assert _get_token_base_cost(without_key, usage, current_time=inside_window)[2] == 1.25e-6


def test_get_token_type_cost_breakdown_reflects_off_peak_reasoning_and_cache_creation_rates():
    from datetime import datetime, timezone

    model_name = "litellm-test-off-peak-breakdown"
    _register_off_peak_reasoning_model(
        model_name,
        {
            "hours_utc": "16:30-00:30",
            "output_cost_per_reasoning_token": 5e-7,
            "cache_creation_input_token_cost": 5e-7,
        },
    )
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=80,
        total_tokens=1080,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=30, text_tokens=50),
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100, cache_creation_tokens=400, text_tokens=500),
    )

    inside = get_token_type_cost_breakdown(
        model=model_name,
        custom_llm_provider="openai",
        usage=usage,
        current_time=datetime(2026, 1, 1, 18, 0, tzinfo=timezone.utc),
    )
    assert inside.reasoning_cost == pytest.approx(30 * 5e-7)
    assert inside.cache_creation_cost == pytest.approx(400 * 5e-7)
    assert inside.cache_read_cost == pytest.approx(100 * 1e-7)

    outside = get_token_type_cost_breakdown(
        model=model_name,
        custom_llm_provider="openai",
        usage=usage,
        current_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert outside.reasoning_cost == pytest.approx(30 * 4e-6)
    assert outside.cache_creation_cost == pytest.approx(400 * 1.25e-6)


def test_generic_cost_per_token_gpt54_above_272k_tokens(_local_model_cost_map):
    """GPT-5.4/5.4-pro: prompts >272K input tokens priced at 2x input, 1.5x output."""
    model = "gpt-5.4"
    custom_llm_provider = "openai"

    model_cost_map = litellm.model_cost[model]
    prompt_tokens = 273000  # Above 272K threshold
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    expected_prompt = model_cost_map["input_cost_per_token_above_272k_tokens"] * prompt_tokens
    expected_completion = model_cost_map["output_cost_per_token_above_272k_tokens"] * completion_tokens
    assert round(prompt_cost, 10) == round(expected_prompt, 10)
    assert round(completion_cost, 10) == round(expected_completion, 10)


def test_generic_cost_per_token_minimax_m3_above_512k_tokens(_local_model_cost_map):
    """MiniMax-M3: prompts >512K input tokens priced at 2x input, output, and cache read."""
    model = "minimax/MiniMax-M3"
    custom_llm_provider = "minimax"

    model_cost_map = litellm.model_cost[model]
    prompt_tokens = 600000
    cached_tokens = 100000
    completion_tokens = 1000
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )
    expected_prompt = (
        model_cost_map["input_cost_per_token_above_512k_tokens"] * (prompt_tokens - cached_tokens)
        + model_cost_map["cache_read_input_token_cost_above_512k_tokens"] * cached_tokens
    )
    expected_completion = model_cost_map["output_cost_per_token_above_512k_tokens"] * completion_tokens
    assert round(prompt_cost, 10) == round(expected_prompt, 10)
    assert round(completion_cost, 10) == round(expected_completion, 10)


def test_generic_cost_per_token_honors_non_standard_above_threshold():
    """Regression for #30344: get_model_info must keep arbitrary
    input/output_cost_per_token_above_<N>_tokens thresholds, not only the hard-coded
    128k/200k/272k/512k set, so a custom tier boundary is applied past its limit."""
    model = "litellm-test-non-standard-tier"
    custom_llm_provider = "openai"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "input_cost_per_token_above_500k_tokens": 9e-6,
                "output_cost_per_token_above_500k_tokens": 18e-6,
            }
        }
    )

    try:
        prompt_tokens = 600000
        completion_tokens = 1000
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 10) == round(9e-6 * prompt_tokens, 10)
        assert round(completion_cost, 10) == round(18e-6 * completion_tokens, 10)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tiered_pricing_charges_cache_creation_at_tier_rate():
    """Regression for LIT-4375: a tier's cache_creation_input_token_cost must be billed
    on the generic (provider-agnostic) path, not silently dropped."""
    model = "litellm-test-tiered-cache-creation"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "tiered_pricing": [
                    {
                        "range": [0, 256000],
                        "input_cost_per_token": 3.25e-07,
                        "output_cost_per_token": 1.95e-06,
                        "cache_creation_input_token_cost": 4.063e-07,
                        "cache_read_input_token_cost": 3.25e-08,
                    },
                    {
                        "range": [256000, 1000000],
                        "input_cost_per_token": 6.5e-07,
                        "output_cost_per_token": 3.9e-06,
                        "cache_creation_input_token_cost": 8.125e-07,
                        "cache_read_input_token_cost": 6.5e-08,
                    },
                ],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=300000,  # 200k new + 60k cache creation + 40k cache read
            completion_tokens=1000,
            total_tokens=301000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=40000, cache_creation_tokens=60000),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )

        expected_prompt = (200000 * 6.5e-07) + (60000 * 8.125e-07) + (40000 * 6.5e-08)
        assert round(prompt_cost, 10) == round(expected_prompt, 10)
        assert round(completion_cost, 10) == round(1000 * 3.9e-06, 10)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tiered_pricing_is_all_or_nothing():
    """Tiered pricing bills the whole request at the tier picked from its input tokens,
    for any provider, and falls back to flat pricing when no tier matches."""
    model = "litellm-test-tiered-all-or-nothing"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "tiered_pricing": [
                    {
                        "range": [0, 32000],
                        "input_cost_per_token": 4.6e-07,
                        "output_cost_per_token": 2.3e-06,
                    },
                    {
                        "range": [32000, 128000],
                        "input_cost_per_token": 7e-07,
                        "output_cost_per_token": 3.5e-06,
                    },
                ],
            }
        }
    )

    try:
        usage = Usage(prompt_tokens=40000, completion_tokens=1000, total_tokens=41000)
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 10) == round(40000 * 7e-07, 10)
        assert round(completion_cost, 10) == round(1000 * 3.5e-06, 10)

        boundary_usage = Usage(prompt_tokens=32000, completion_tokens=10, total_tokens=32010)
        boundary_prompt_cost, _ = generic_cost_per_token(
            model=model,
            usage=boundary_usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(boundary_prompt_cost, 10) == round(32000 * 4.6e-07, 10)

        empty_prompt_usage = Usage(prompt_tokens=0, completion_tokens=100, total_tokens=100)
        empty_prompt_cost, empty_completion_cost = generic_cost_per_token(
            model=model,
            usage=empty_prompt_usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert empty_prompt_cost == 0.0
        assert round(empty_completion_cost, 10) == round(100 * 2e-06, 10)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_an_output_rate_bills_the_model_rate():
    """Regression: a tier table that spells out only input rates served every completion for
    free, since a tier's missing output rate has no tier-level fallback to stand in for it."""
    model = "litellm-test-tiered-input-only"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "output_cost_per_token": 2e-06,
                "output_cost_per_reasoning_token": 5e-06,
                "tiered_pricing": [{"range": [0, 128000], "input_cost_per_token": 1e-03}],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=13,
            completion_tokens=182,
            total_tokens=195,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=100),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 12) == round(13 * 1e-03, 12)
        assert round(completion_cost, 12) == round((82 * 2e-06) + (100 * 5e-06), 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_cache_rates_bills_cache_at_the_tier_input_rate():
    model = "litellm-test-tiered-no-cache-rates"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "cache_read_input_token_cost": 9e-09,
                "cache_creation_input_token_cost": 9e-06,
                "tiered_pricing": [
                    {
                        "range": [0, 32000],
                        "input_cost_per_token": 4.6e-07,
                        "output_cost_per_token": 2.3e-06,
                    },
                    {
                        "range": [32000, 128000],
                        "input_cost_per_token": 7e-07,
                        "output_cost_per_token": 3.5e-06,
                    },
                ],
            }
        }
    )

    try:
        uncached = Usage(prompt_tokens=40000, completion_tokens=100, total_tokens=40100)
        cached = Usage(
            prompt_tokens=40000,
            completion_tokens=100,
            total_tokens=40100,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=5000, cache_creation_tokens=15000),
        )
        uncached_prompt_cost, _ = generic_cost_per_token(
            model=model,
            usage=uncached,
            custom_llm_provider=custom_llm_provider,
        )
        cached_prompt_cost, cached_completion_cost = generic_cost_per_token(
            model=model,
            usage=cached,
            custom_llm_provider=custom_llm_provider,
        )

        tier_input_rate = 7e-07
        assert round(cached_prompt_cost, 12) == round(40000 * tier_input_rate, 12)
        assert round(cached_prompt_cost, 12) == round(uncached_prompt_cost, 12)
        assert round(cached_completion_cost, 12) == round(100 * 3.5e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_a_1hr_cache_rate_bills_the_tier_cache_creation_rate():
    model = "litellm-test-tiered-no-1hr-cache-rate"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "cache_creation_input_token_cost_above_1hr": 9e-05,
                "tiered_pricing": [
                    {
                        "range": [0, 128000],
                        "input_cost_per_token": 7e-07,
                        "output_cost_per_token": 3.5e-06,
                        "cache_creation_input_token_cost": 8.75e-07,
                    }
                ],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=10,
            total_tokens=1010,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cache_creation_tokens=800,
                cache_creation_token_details=CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=300, ephemeral_1h_input_tokens=500
                ),
            ),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )

        tier_cache_creation_rate = 8.75e-07
        expected_prompt = (200 * 7e-07) + (800 * tier_cache_creation_rate)
        assert round(prompt_cost, 12) == round(expected_prompt, 12)
        assert round(completion_cost, 12) == round(10 * 3.5e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_generic_cost_per_token_tier_without_an_input_rate_is_not_a_priced_tier():
    model = "litellm-test-tiered-no-input-rate"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
                "tiered_pricing": [{"range": [0, 128000], "output_cost_per_token": 3.5e-06}],
            }
        }
    )

    try:
        usage = Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 12) == round(1000 * 1e-06, 12)
        assert round(completion_cost, 12) == round(100 * 2e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


def test_router_deployment_with_input_only_tiers_bills_completions_at_the_backend_rate():
    """Regression: the router registers a deployment's custom pricing as a standalone
    model_cost entry holding only the supplied fields, so an input-only tier table left
    the output-rate fallback nothing to read and billed every completion at 0."""
    from litellm import Router

    model_id = "litellm-test-router-tiered-input-only"
    backend_model = "anthropic/claude-haiku-4-5"
    backend_output_rate = litellm.get_model_info(backend_model)["output_cost_per_token"]
    Router(
        model_list=[
            {
                "model_name": "tiered-input-only",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": "sk-test",
                    "tiered_pricing": [
                        {"range": [0, 3000], "input_cost_per_token": 3.25e-07},
                        {"range": [3000, 128000], "input_cost_per_token": 8.125e-07},
                    ],
                },
                "model_info": {"id": model_id},
            }
        ]
    )

    try:
        usage = Usage(prompt_tokens=21, completion_tokens=4, total_tokens=25)
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model_id,
            usage=usage,
            custom_llm_provider="anthropic",
        )
        assert round(prompt_cost, 12) == round(21 * 3.25e-07, 12)
        assert round(completion_cost, 12) == round(4 * backend_output_rate, 12)
        assert backend_output_rate > 0
    finally:
        litellm.model_cost.pop(model_id, None)


def test_generic_cost_per_token_tiered_pricing_bills_reasoning_at_tier_rate():
    """Regression: a tier's output_cost_per_reasoning_token must price reasoning tokens
    on the generic path and in the logged breakdown, not the tier's plain output rate."""
    model = "litellm-test-tiered-reasoning"
    custom_llm_provider = "openrouter"
    litellm.register_model(
        {
            model: {
                "litellm_provider": custom_llm_provider,
                "mode": "chat",
                "tiered_pricing": [
                    {
                        "range": [0, 256000],
                        "input_cost_per_token": 4e-07,
                        "output_cost_per_token": 1.2e-06,
                        "output_cost_per_reasoning_token": 4e-06,
                    },
                    {
                        "range": [256000, 1000000],
                        "input_cost_per_token": 1.2e-06,
                        "output_cost_per_token": 3.6e-06,
                        "output_cost_per_reasoning_token": 1.2e-05,
                    },
                ],
            }
        }
    )

    try:
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=400),
        )
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider=custom_llm_provider,
        )
        assert round(prompt_cost, 12) == round(1000 * 4e-07, 12)
        assert round(completion_cost, 12) == round((100 * 1.2e-06) + (400 * 4e-06), 12)

        breakdown = get_token_type_cost_breakdown(
            model=model,
            custom_llm_provider=custom_llm_provider,
            usage=usage,
        )
        assert round(breakdown.reasoning_cost, 12) == round(400 * 4e-06, 12)
    finally:
        litellm.model_cost.pop(model, None)


@pytest.mark.parametrize(
    "base_model,dated_model",
    [
        ("gpt-5.5", "gpt-5.5-2026-04-23"),
        ("gpt-5.5-pro", "gpt-5.5-pro-2026-04-23"),
    ],
)
def test_gpt55_dated_variants_match_base_reasoning_effort_capabilities(_local_model_cost_map, base_model, dated_model):
    """Dated snapshots must carry the same reasoning_effort capability flags as
    their non-dated counterparts.

    Regression guard: ``supports_{none,minimal,xhigh}_reasoning_effort`` gate
    downstream routing in ``OpenAIGPT5Config`` — a missing flag is treated as
    ``False`` for opt-in levels (e.g. ``xhigh``), which silently diverges
    behavior between ``gpt-5.5`` and ``gpt-5.5-2026-04-23``. Pinning to a
    dated variant must never lose capabilities relative to the base alias.
    """

    base = litellm.model_cost[base_model]
    dated = litellm.model_cost[dated_model]

    for flag in (
        "supports_none_reasoning_effort",
        "supports_minimal_reasoning_effort",
        "supports_xhigh_reasoning_effort",
    ):
        assert dated.get(flag) == base.get(flag), (
            f"{dated_model} has {flag}={dated.get(flag)!r}, "
            f"but {base_model} has {flag}={base.get(flag)!r}. "
            f"Dated snapshots must inherit the base model's reasoning_effort "
            f"capability profile."
        )


def test_string_cost_values():
    """Test that cost values defined as strings are properly converted to floats."""
    from unittest.mock import patch

    # Mock model info with string cost values (as might be read from config.yaml)
    mock_model_info = {
        "input_cost_per_token": "3e-7",  # String representation of scientific notation
        "output_cost_per_token": "6e-7",  # String representation of scientific notation
        "input_cost_per_audio_token": "0.000001",  # String representation of decimal
        "output_cost_per_audio_token": "0.000002",  # String representation of decimal
        "cache_read_input_token_cost": "1.5e-8",  # String representation of scientific notation
        "cache_creation_input_token_cost": "2.5e-8",  # String representation of scientific notation
    }

    # Test usage with various token types
    # Note: prompt_tokens must equal sum of details to avoid double-counting adjustment
    # text_tokens(700) + audio_tokens(100) + cached_tokens(200) + cache_creation_tokens(150) = 1150
    usage = Usage(
        prompt_tokens=1150,
        completion_tokens=500,
        total_tokens=1650,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=100,
            cached_tokens=200,
            text_tokens=700,
            image_tokens=None,
            cache_creation_tokens=150,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=50,
            reasoning_tokens=None,
            text_tokens=450,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
    )

    # Mock get_model_info to return our mock model info
    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Calculate expected costs manually
    # Prompt cost = text_tokens * input_cost + audio_tokens * audio_cost + cached_tokens * cache_read_cost + cache_creation_tokens * cache_creation_cost
    expected_prompt_cost = (
        700 * 3e-7  # text tokens
        + 100 * 1e-6  # audio tokens
        + 200 * 1.5e-8  # cached tokens
        + 150 * 2.5e-8  # cache creation tokens
    )

    # Completion cost = text_tokens * output_cost + audio_tokens * audio_output_cost
    expected_completion_cost = 450 * 6e-7 + 50 * 2e-6  # text tokens  # audio tokens

    # Assert costs are calculated correctly
    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_generic_cost_per_token_overlapping_cached_and_image_tokens():
    """Some providers report cached_tokens and image_tokens as overlapping subsets of
    prompt_tokens. Billing each in full charged the overlap twice, once at the cache rate
    and again at the input rate."""
    model = "litellm-test-overlapping-cached-image"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 1e-6,
                "cache_read_input_token_cost": 1e-7,
                "output_cost_per_token": 2e-6,
            }
        }
    )
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=None, cached_tokens=90, image_tokens=80),
    )

    prompt_cost, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")

    # 90 cached at 1e-7, the remaining 10 uncached tokens once at 1e-6
    assert prompt_cost == pytest.approx(90 * 1e-7 + 10 * 1e-6)
    assert completion_cost == pytest.approx(10 * 2e-6)


def test_generic_cost_per_token_warm_prefix_cache_spanning_text_and_image_tokens():
    """xAI reports text_tokens + image_tokens = prompt_tokens with cached_tokens overlapping
    both, so a warm prefix cache covering the whole image exceeds the text-only count.
    Observed live on grok-4.6 (issue #37281): the image tokens were billed a second time at
    the full input rate on top of the cache-read bucket, 0.003500 in vs the provider's own
    0.001274 bill."""
    model = "litellm-test-warm-prefix-cache-overlap"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 2e-6,
                "cache_read_input_token_cost": 5e-7,
                "output_cost_per_token": 6e-6,
            }
        }
    )
    usage = Usage(
        prompt_tokens=2461,
        completion_tokens=440,
        total_tokens=2901,
        prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=1319, cached_tokens=2432, image_tokens=1142),
    )

    prompt_cost, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")

    # 2432 cached at the cache-read rate, the 29 uncached tokens once at the input rate
    assert prompt_cost == pytest.approx(2432 * 5e-7 + 29 * 2e-6)
    assert completion_cost == pytest.approx(440 * 6e-6)


def test_calculate_cost_component_with_string_values():
    """Test the calculate_cost_component function directly with string cost values."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import calculate_cost_component

    # Test with valid string scientific notation
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 1000 * 3e-7

    # Test with valid string decimal notation
    model_info = {"output_cost_per_token": "0.000001"}
    cost = calculate_cost_component(model_info, "output_cost_per_token", 500)
    assert cost == 500 * 0.000001

    # Test with float value (should work as before)
    model_info = {"input_cost_per_token": 3e-7}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 1000 * 3e-7

    # Test with invalid string value (should return 0.0)
    model_info = {"input_cost_per_token": "invalid_number"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with None value (should return 0.0)
    model_info = {"input_cost_per_token": None}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with missing key (should return 0.0)
    model_info = {}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 1000)
    assert cost == 0.0

    # Test with zero usage (should return 0.0)
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", 0)
    assert cost == 0.0

    # Test with None usage (should return 0.0)
    model_info = {"input_cost_per_token": "3e-7"}
    cost = calculate_cost_component(model_info, "input_cost_per_token", None)
    assert cost == 0.0


def test_string_cost_values_edge_cases():
    """Test edge cases for string cost value handling."""
    from unittest.mock import patch

    # Test with mixed string and float cost values
    mock_model_info = {
        "input_cost_per_token": "1e-6",  # String
        "output_cost_per_token": 2e-6,  # Float
        "input_cost_per_audio_token": "invalid",  # Invalid string
        "output_cost_per_audio_token": None,  # None value
    }

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=100, cached_tokens=0, text_tokens=1000, image_tokens=None
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=50,
            reasoning_tokens=None,
            text_tokens=500,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Expected costs:
    # Prompt: 1000 * 1e-6 + 100 * 0 (invalid string becomes 0)
    # Completion: 500 * 2e-6 + 50 * 2e-6 (audio tokens fall back to base cost when output_cost_per_audio_token is None)
    expected_prompt_cost = 1000 * 1e-6
    expected_completion_cost = 500 * 2e-6 + 50 * 2e-6

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_string_cost_values_with_threshold():
    """Test that string cost values work correctly with threshold pricing."""
    from unittest.mock import patch

    # Mock model info with string cost values including threshold pricing
    mock_model_info = {
        "input_cost_per_token": "1e-6",  # String base cost
        "output_cost_per_token": "2e-6",  # String base cost
        "input_cost_per_token_above_200k_tokens": "5e-7",  # String threshold cost (lower)
        "output_cost_per_token_above_200k_tokens": "1e-6",  # String threshold cost (lower)
    }

    # Test usage above threshold
    usage = Usage(
        prompt_tokens=250000,  # Above 200k threshold
        completion_tokens=1000,
        total_tokens=251000,
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        prompt_cost, completion_cost = generic_cost_per_token(
            model="test-model", usage=usage, custom_llm_provider="test-provider"
        )

    # Expected costs using threshold pricing (string values converted to float)
    expected_prompt_cost = 250000 * 5e-7  # threshold cost
    expected_completion_cost = 1000 * 1e-6  # threshold cost

    assert round(prompt_cost, 12) == round(expected_prompt_cost, 12)
    assert round(completion_cost, 12) == round(expected_completion_cost, 12)


def test_calculate_cache_writing_cost():
    """Test the calculate_cache_writing_cost function with detailed cache creation token breakdown."""

    # Test case 1: With cache creation token details (matching the provided input)
    cache_creation_tokens = 14055
    cache_creation_token_details = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=56, ephemeral_1h_input_tokens=13999
    )
    cache_creation_cost_above_1hr = 6e-06
    cache_creation_cost = 3.75e-06

    result = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens,
        cache_creation_token_details=cache_creation_token_details,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        cache_creation_cost=cache_creation_cost,
    )

    # Expected calculation:
    # 5m tokens: 56 * 3.75e-06 = 0.00021
    # 1h tokens: 13999 * 6e-06 = 0.083994
    # Total: 0.00021 + 0.083994 = 0.084204
    expected_cost = (56 * 3.75e-06) + (13999 * 6e-06)

    assert round(result, 6) == round(expected_cost, 6)
    assert round(result, 6) == 0.084204

    # Test case 2: Without cache creation token details (fallback behavior)
    cache_creation_tokens_no_details = 1000
    cache_creation_token_details_none = None
    cache_creation_cost_fallback = 5e-06

    result_no_details = calculate_cache_writing_cost(
        cache_creation_tokens=cache_creation_tokens_no_details,
        cache_creation_token_details=cache_creation_token_details_none,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
        cache_creation_cost=cache_creation_cost_fallback,
    )

    # Expected calculation when no details: 1000 * 5e-06 = 0.005
    expected_cost_no_details = 1000 * 5e-06

    assert round(result_no_details, 6) == round(expected_cost_no_details, 6)
    assert result_no_details == 0.005

    # Test case 3: With cache creation token details but None values
    cache_creation_token_details_partial = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=None, ephemeral_1h_input_tokens=100
    )

    result_partial = calculate_cache_writing_cost(
        cache_creation_tokens=500,
        cache_creation_token_details=cache_creation_token_details_partial,
        cache_creation_cost_above_1hr=6e-06,
        cache_creation_cost=3e-06,
    )

    # Expected calculation: 0 (for None 5m tokens) + (100 * 6e-06) = 0.0006
    expected_cost_partial = (0.0) + (100 * 6e-06)

    assert round(result_partial, 6) == round(expected_cost_partial, 6)
    assert round(result_partial, 6) == 0.0006

    # Test case 4: Zero costs
    result_zero = calculate_cache_writing_cost(
        cache_creation_tokens=1000,
        cache_creation_token_details=CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=50, ephemeral_1h_input_tokens=950
        ),
        cache_creation_cost_above_1hr=0.0,
        cache_creation_cost=0.0,
    )

    assert result_zero == 0.0


def test_cache_writing_cost_with_zero_creation_tokens_and_ephemeral_details():
    """
    Regression test: when cache_creation_tokens is 0 but cache_creation_token_details
    has non-zero ephemeral tokens, the cost must still be calculated.
    This ensures the guard in _calculate_input_cost doesn't skip
    calculate_cache_writing_cost when only ephemeral token details are present.
    """
    cache_creation_cost = 3.75e-06
    cache_creation_cost_above_1hr = 6e-06

    prompt_tokens_details: PromptTokensDetailsResult = {
        "cache_hit_tokens": 0,
        "cache_hit_audio_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_creation_token_details": CacheCreationTokenDetails(
            ephemeral_5m_input_tokens=100,
            ephemeral_1h_input_tokens=200,
        ),
        "text_tokens": 0,
        "audio_tokens": 0,
        "image_tokens": 0,
        "video_tokens": 0,
        "character_count": 0,
        "image_count": 0,
        "video_length_seconds": 0.0,
        "audio_length_seconds": 0.0,
        "query_count": 0,
    }

    model_info: ModelInfo = {}

    result = _calculate_input_cost(
        prompt_tokens_details=prompt_tokens_details,
        model_info=model_info,
        prompt_base_cost=0.0,
        cache_read_cost=0.0,
        cache_creation_cost=cache_creation_cost,
        cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
    )

    # Expected: (100 * 3.75e-06) + (200 * 6e-06) = 0.000375 + 0.0012 = 0.001575
    expected = (100 * cache_creation_cost) + (200 * cache_creation_cost_above_1hr)
    assert result > 0, "Cost should not be zero when ephemeral token details are present"
    assert round(result, 6) == round(expected, 6)


def test_service_tier_ultrafast_pricing():
    """An ultrafast request bills the *_ultrafast rates for all token types.

    Regression for the ultrafast service tier being absent from ServiceTier:
    the cost-key lookup silently returned the standard keys, undercounting
    every ultrafast request.
    """
    cached_tokens = 200
    cache_write_tokens = 300
    text_tokens = 500
    usage = Usage(
        prompt_tokens=text_tokens + cached_tokens + cache_write_tokens,
        completion_tokens=400,
        total_tokens=text_tokens + cached_tokens + cache_write_tokens + 400,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens
        ),
    )
    model_info: ModelInfo = {
        "key": "gpt-5.6-sol",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 3e-05,
        "cache_creation_input_token_cost": 6.25e-06,
        "cache_read_input_token_cost": 5e-07,
        "input_cost_per_token_ultrafast": 5e-05,
        "output_cost_per_token_ultrafast": 3e-04,
        "cache_creation_input_token_cost_ultrafast": 6.25e-05,
        "cache_read_input_token_cost_ultrafast": 5e-06,
    }

    prompt_cost, completion_cost = generic_cost_per_token(
        model="gpt-5.6-sol",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="ultrafast",
        model_info=model_info,
    )

    expected_prompt_cost = text_tokens * 5e-05 + cached_tokens * 5e-06 + cache_write_tokens * 6.25e-05
    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(400 * 3e-04)


def test_service_tier_ultrafast_fallback_pricing(_local_model_cost_map):
    """Without *_ultrafast keys an ultrafast request bills the standard rate, not zero.

    Guards the suffix fallback in _get_cost_per_unit: "_fast" is a substring of
    "_ultrafast", so a shortest-first suffix match would strip the wrong suffix
    and price the request at 0.
    """

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    std_prompt_cost, std_completion_cost = generic_cost_per_token(
        model="gpt-5.6-sol",
        usage=usage,
        custom_llm_provider="openai",
        service_tier=None,
    )
    ultrafast_prompt_cost, ultrafast_completion_cost = generic_cost_per_token(
        model="gpt-5.6-sol",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="ultrafast",
    )

    assert std_prompt_cost + std_completion_cost > 0
    assert ultrafast_prompt_cost == pytest.approx(std_prompt_cost)
    assert ultrafast_completion_cost == pytest.approx(std_completion_cost)


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-lite-image",
    ],
)
def test_gemini_image_generation_cost_with_zero_text_tokens(_local_model_cost_map, model: str):
    """
    Test that image_tokens are correctly costed when text_tokens=0.

    Reproduces issue #17410: completion_cost calculates incorrectly for
    Gemini-3-pro-image model - image_tokens were treated as text tokens
    when text_tokens=0.

    https://github.com/BerriAI/litellm/issues/17410
    """

    custom_llm_provider = "vertex_ai"

    # Usage from the issue: text_tokens=0, image_tokens=1120, reasoning_tokens=225
    usage = Usage(
        completion_tokens=1345,
        prompt_tokens=10,
        total_tokens=1355,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=225,
            rejected_prediction_tokens=None,
            text_tokens=0,  # This is the key: text_tokens=0
            image_tokens=1120,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=None, text_tokens=10, image_tokens=None
        ),
    )

    model_cost_map = litellm.model_cost[model]
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
    )

    # Expected costs:
    # - text_tokens: 0 * output_cost_per_token = 0
    # - image_tokens: 1120 * output_cost_per_image_token
    # - reasoning_tokens: 225 * output_cost_per_token
    # Total completion should include both image + reasoning costs.

    output_cost_per_image_token = model_cost_map.get("output_cost_per_image_token", 0)
    output_cost_per_token = model_cost_map.get("output_cost_per_token", 0)

    expected_image_cost = 1120 * output_cost_per_image_token
    expected_reasoning_cost = 225 * output_cost_per_token  # reasoning uses base token cost
    expected_completion_cost = expected_image_cost + expected_reasoning_cost

    # The bug was: all completion tokens were treated as text tokens only.
    bugged_text_only_cost = 1345 * output_cost_per_token
    assert completion_cost > bugged_text_only_cost * 2, (
        f"Completion cost should be significantly larger than text-only bugged path. "
        f"Expected > {bugged_text_only_cost * 2:.6f}, got {completion_cost:.6f}"
    )
    assert round(completion_cost, 4) == round(expected_completion_cost, 4), (
        f"Expected completion cost ${expected_completion_cost:.6f}, got ${completion_cost:.6f}"
    )


def test_vertex_image_generation_cost_prefers_token_usage_metadata(_local_model_cost_map):
    """
    When usage metadata exists on image responses, Vertex image generation cost
    should be calculated from token pricing, not flat output_cost_per_image.
    """

    model = "gemini-3.1-flash-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")

    input_text_tokens = 50
    input_image_tokens = 1120
    output_image_tokens = 1120
    prompt_tokens = input_text_tokens + input_image_tokens

    image_response = ImageResponse(
        data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")],
        usage=ImageUsage(
            input_tokens=prompt_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                text_tokens=input_text_tokens,
                image_tokens=input_image_tokens,
            ),
            output_tokens=output_image_tokens,
            total_tokens=prompt_tokens + output_image_tokens,
        ),
    )

    cost = vertex_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_prompt_cost = prompt_tokens * model_info["input_cost_per_token"]
    expected_completion_cost = output_image_tokens * model_info["output_cost_per_image_token"]
    expected_total_cost = expected_prompt_cost + expected_completion_cost

    assert round(cost, 10) == round(expected_total_cost, 10)
    # Ensure this is not falling back to flat per-image pricing.
    assert cost != len(image_response.data) * model_info["output_cost_per_image"]


def test_vertex_image_generation_cost_falls_back_to_flat_image_pricing(_local_model_cost_map):
    """
    Without usage metadata, Vertex image generation cost should fall back to
    output_cost_per_image * number_of_images.
    """

    model = "gemini-3.1-flash-image-preview"
    model_info = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")

    image_response = ImageResponse(data=[ImageObject(b64_json="img1"), ImageObject(b64_json="img2")])

    cost = vertex_image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )

    expected_cost = len(image_response.data) * model_info["output_cost_per_image"]
    assert round(cost, 10) == round(expected_cost, 10)






def test_query_count_is_free_without_a_per_query_price(_local_model_cost_map):
    usage = Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        prompt_tokens_details=PromptTokensDetailsWrapper(query_count=1),
    )

    prompt_cost, _ = generic_cost_per_token(model="text-embedding-3-small", usage=usage, custom_llm_provider="openai")

    assert prompt_cost == 0.0


# ---------------------------------------------------------------------------
# Data-residency (OpenAI regional processing) tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-realtime-2.1", "gpt-realtime-2.1-mini"])
@pytest.mark.parametrize("data_residency", ["eu", "us"])
def test_data_residency_applies_uplift(data_residency, model, _local_model_cost_map):
    """Models released on/after 2026-03-05 (gpt-5.4/5.5 and gpt-realtime-2.1
    series) apply the 10% regional processing uplift multiplier when
    data_residency is set; gpt-5 and older models do not."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
    )
    regional = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="openai",
        data_residency=data_residency,
    )

    base_total = base[0] + base[1]
    regional_total = regional[0] + regional[1]

    assert base_total > 0
    assert regional_total == pytest.approx(base_total * 1.10, rel=1e-9)
    assert regional[0] == pytest.approx(base[0] * 1.10, rel=1e-9)
    assert regional[1] == pytest.approx(base[1] * 1.10, rel=1e-9)


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-4o", "gpt-4.1"])
def test_data_residency_no_uplift_for_pre_march_2026_models(model, _local_model_cost_map):
    """Models released before 2026-03-05 must not have the regional uplift."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")
    regional = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai", data_residency="eu")

    assert base == regional, f"{model} should not have a regional uplift, but cost changed with data_residency"


def test_data_residency_no_uplift_for_unmarked_model(_local_model_cost_map):
    """A model without a regional_processing_uplift_multiplier_* entry should
    fall back to base pricing, not error."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(
        model="gpt-3.5-turbo",
        usage=usage,
        custom_llm_provider="openai",
    )
    with_residency = generic_cost_per_token(
        model="gpt-3.5-turbo",
        usage=usage,
        custom_llm_provider="openai",
        data_residency="eu",
    )

    assert base == with_residency


def test_data_residency_none_no_uplift(_local_model_cost_map):
    """data_residency=None should be a no-op even for models with a multiplier."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
    )
    explicit_none = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
        data_residency=None,
    )

    assert base == explicit_none


def test_data_residency_composes_with_service_tier(_local_model_cost_map):
    """The uplift multiplies the priority-tier cost, not the standard one."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    priority_base = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
    )
    priority_eu = generic_cost_per_token(
        model="gpt-5.4",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        data_residency="eu",
    )

    priority_base_total = priority_base[0] + priority_base[1]
    priority_eu_total = priority_eu[0] + priority_eu[1]

    assert priority_base_total > 0
    assert priority_eu_total == pytest.approx(priority_base_total * 1.10, rel=1e-9)


@pytest.mark.parametrize("model", ["gemini-3.5-flash", "claude-haiku-4-5@20251001"])
@pytest.mark.parametrize("vertex_location", ["us-central1", "us-east5", "europe-west1", "asia-southeast1"])
def test_vertex_regional_location_applies_uplift(vertex_location, model, _local_model_cost_map):
    """Google bills every non-global Vertex endpoint at 1.1x the global rate for GA
    Gemini 3+ and regional-pricing Claude models, so a request served from a regional
    location must cost 1.1x what the same usage costs on the global endpoint."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="vertex_ai")
    regional = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="vertex_ai",
        vertex_location=vertex_location,
    )

    base_total = base[0] + base[1]
    regional_total = regional[0] + regional[1]

    assert base_total > 0
    assert regional_total == pytest.approx(base_total * 1.10, rel=1e-9)
    assert regional[0] == pytest.approx(base[0] * 1.10, rel=1e-9)
    assert regional[1] == pytest.approx(base[1] * 1.10, rel=1e-9)


@pytest.mark.parametrize("vertex_location", [None, "global", "GLOBAL"])
def test_vertex_global_or_absent_location_no_uplift(vertex_location, _local_model_cost_map):
    """The global endpoint prices at the base rate, whatever the casing, and an
    unresolved location must never uplift."""
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    base = generic_cost_per_token(model="claude-haiku-4-5@20251001", usage=usage, custom_llm_provider="vertex_ai")
    located = generic_cost_per_token(
        model="claude-haiku-4-5@20251001",
        usage=usage,
        custom_llm_provider="vertex_ai",
        vertex_location=vertex_location,
    )

    assert base == located




def test_vertex_uplift_invalid_multiplier_defaults_to_one():
    """A malformed multiplier in the cost map degrades to base pricing, never raises."""
    from litellm.litellm_core_utils.llm_cost_calc.utils import (
        get_vertex_regional_endpoint_uplift,
    )

    assert (
        get_vertex_regional_endpoint_uplift({"regional_endpoint_uplift_multiplier": "not-a-number"}, "us-east5") == 1.0
    )


def test_service_tier_suffixes_constant_in_sync_with_enum():
    from litellm.litellm_core_utils.llm_cost_calc.utils import _SERVICE_TIER_SUFFIXES
    from litellm.types.utils import ServiceTier

    assert set(_SERVICE_TIER_SUFFIXES) == {f"_{st.value}" for st in ServiceTier}
    # longest-first so a substring match resolves "_ultrafast" before "_fast"
    assert list(_SERVICE_TIER_SUFFIXES) == sorted(_SERVICE_TIER_SUFFIXES, key=len, reverse=True)


def test_get_cost_per_unit_falls_back_from_service_tier_key_to_base():
    from litellm.litellm_core_utils.llm_cost_calc.utils import _get_cost_per_unit

    model_info = {"input_cost_per_token": 2e-6}
    # service-tier key is absent -> falls back to the base key
    assert _get_cost_per_unit(model_info, "input_cost_per_token_priority") == 2e-6
    # service-tier key present -> used directly, no fallback
    model_info_direct = {
        "input_cost_per_token_priority": 5e-6,
        "input_cost_per_token": 2e-6,
    }
    assert _get_cost_per_unit(model_info_direct, "input_cost_per_token_priority") == 5e-6


def test_threshold_keys_exclude_service_tier_variants():
    from typing import cast

    from litellm.litellm_core_utils.llm_cost_calc.utils import _get_token_base_cost
    from litellm.types.utils import ModelInfo, Usage

    # The service-tier-suffixed above-threshold key must be excluded from
    # threshold detection. The _priority variant has a higher threshold (300k),
    # so if it were not excluded it would sort first and drive a 9e-6 rate for
    # this non-tier request. With the exclusion only the standard 200k key
    # applies, giving 3e-6.
    model_info = cast(
        ModelInfo,
        {
            "input_cost_per_token": 1e-6,
            "input_cost_per_token_above_200k_tokens": 3e-6,
            "input_cost_per_token_above_300k_tokens_priority": 9e-6,
            "output_cost_per_token": 2e-6,
        },
    )
    usage = Usage(prompt_tokens=350_000, completion_tokens=1_000, total_tokens=351_000)
    prompt_base, *_ = _get_token_base_cost(model_info=model_info, usage=usage)
    assert prompt_base == 3e-6


@pytest.mark.parametrize(
    "model,custom_llm_provider,reasoning_tokens,cached_tokens",
    [
        ("gemini-2.5-flash", "vertex_ai", 3114, 100),
        ("o3", "openai", 500, 200),
        ("azure/gpt-5", "azure", 300, 150),
        ("us.amazon.nova-2-lite-v1:0", "bedrock", 120, 80),
        ("perplexity/sonar-reasoning", "perplexity", 400, 0),
        ("cerebras/qwen-3-32b", "cerebras", 250, 0),
    ],
)
def test_token_type_cost_breakdown_is_provider_agnostic(
    _local_model_cost_map, model, custom_llm_provider, reasoning_tokens, cached_tokens
):
    """
    Reasoning and cache-read costs must be surfaced for every provider that reports
    those tokens, regardless of which cost calculator the provider routes through
    (Perplexity, Cerebras, Dashscope bypass generic_cost_per_token entirely).

    Cache tokens always land in prompt_tokens_details.cached_tokens, so reading from
    there - not the top-level cache_read_input_tokens attribute the old breakdown code
    relied on - is what makes Vertex/OpenAI/Azure cache costs show up at all.
    """

    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=2000,
        total_tokens=3000,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            reasoning_tokens=reasoning_tokens, text_tokens=2000 - reasoning_tokens
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens, text_tokens=1000 - cached_tokens),
    )

    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider=custom_llm_provider, usage=usage)

    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    reasoning_rate = model_info.get("output_cost_per_reasoning_token") or model_info["output_cost_per_token"]
    cache_read_rate = model_info.get("cache_read_input_token_cost") or 0.0

    assert breakdown.reasoning_cost == pytest.approx(reasoning_tokens * reasoning_rate)
    assert breakdown.cache_read_cost == pytest.approx(cached_tokens * cache_read_rate)


def test_token_type_cost_breakdown_includes_cache_creation_from_top_level_usage(_local_model_cost_map):
    """
    Bedrock/Anthropic report cache tokens as top-level usage fields; the Usage
    constructor maps them onto prompt_tokens_details, so the breakdown must still
    pick up both cache-read and cache-creation costs.
    """

    model = "anthropic.claude-3-5-haiku-20241022-v1:0"
    usage = Usage(
        prompt_tokens=500,
        completion_tokens=50,
        total_tokens=550,
        cache_creation_input_tokens=300,
        cache_read_input_tokens=120,
    )

    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider="bedrock", usage=usage)

    model_info = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    assert breakdown.cache_creation_cost == pytest.approx(300 * model_info["cache_creation_input_token_cost"])
    assert breakdown.cache_read_cost == pytest.approx(120 * model_info["cache_read_input_token_cost"])


def test_token_type_cost_breakdown_reads_cache_write_tokens(_local_model_cost_map):
    """
    Some OpenAI-compatible providers (e.g. kimi-k2) report cache-write tokens under
    `cache_write_tokens` rather than `cache_creation_tokens`. The breakdown must read
    it the same way the total-cost normalization does, so the two agree.
    """

    model = "anthropic.claude-3-5-haiku-20241022-v1:0"
    usage = Usage(
        prompt_tokens=500,
        completion_tokens=50,
        total_tokens=550,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=0, cache_write_tokens=300),
    )

    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider="bedrock", usage=usage)
    model_info = litellm.get_model_info(model=model, custom_llm_provider="bedrock")
    assert breakdown.cache_creation_cost == pytest.approx(300 * model_info["cache_creation_input_token_cost"])


def test_generic_cost_per_token_openai_cache_write_tokens_gpt_5_6(_local_model_cost_map):
    """
    Regression: OpenAI gpt-5.6 reports cache-write tokens under
    prompt_tokens_details.cache_write_tokens (not the Anthropic cache_creation_tokens
    name). Those tokens must be billed at the cache-write rate rather than the plain
    input rate. Customer report: cache creation tokens were never counted for the
    GPT-5.6 series, so cost was undercounted on cache-write requests.
    """

    model = "gpt-5.6"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=0, cache_write_tokens=800),
    )

    assert usage.prompt_tokens_details.cache_write_tokens == 800
    assert usage.prompt_tokens_details.cache_creation_tokens == 800

    prompt_cost, _ = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    expected_prompt = (1000 - 800) * info["input_cost_per_token"] + 800 * info["cache_creation_input_token_cost"]
    assert prompt_cost == pytest.approx(expected_prompt)
    assert info["cache_creation_input_token_cost"] > info["input_cost_per_token"]
    assert prompt_cost > 1000 * info["input_cost_per_token"]


def test_generic_cost_per_token_backs_out_cache_write_tokens_from_text_tokens(_local_model_cost_map):
    """
    Regression for #34801: when a provider reports text_tokens covering the whole
    prompt alongside cache-write tokens (and no cache reads), the cache-write tokens
    must be backed out of the text total instead of being billed twice.
    """

    model = "gpt-5.6"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=0, cache_write_tokens=800, text_tokens=1000),
    )

    prompt_cost, _ = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    expected_prompt = 200 * info["input_cost_per_token"] + 800 * info["cache_creation_input_token_cost"]
    assert prompt_cost == pytest.approx(expected_prompt)


def test_token_type_cost_breakdown_reconciles_with_generic_total(_local_model_cost_map):
    """
    Both-ways check: the reasoning subset must sum with the remaining (text) output
    cost to exactly the completion total, and the cache-read subset with the remaining
    input cost to exactly the prompt total, as computed by generic_cost_per_token.
    A mismatch here would mean the breakdown misrepresents what was actually billed.
    """

    model = "gemini-2.5-flash"
    custom_llm_provider = "vertex_ai"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=2000,
        total_tokens=3000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=1200, text_tokens=800),
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=300, text_tokens=700),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider=custom_llm_provider
    )
    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider=custom_llm_provider, usage=usage)

    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    text_output_cost = 800 * model_info["output_cost_per_token"]
    text_input_cost = 700 * model_info["input_cost_per_token"]

    assert text_output_cost + breakdown.reasoning_cost == pytest.approx(completion_cost)
    assert text_input_cost + breakdown.cache_read_cost == pytest.approx(prompt_cost)


def _custom_priced_usage() -> Usage:
    return Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=800, cache_creation_tokens=100),
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=200),
    )


def test_token_type_cost_breakdown_prices_custom_pricing_from_its_flat_rates():
    """
    A custom-priced deployment, usually absent from the cost map, used to get zero cache and
    reasoning lines while its total already billed cache tokens at the custom cache rates.
    The lines must come from the same flat rates: a configured cache rate, else the input
    rate for cache tokens and the output rate for reasoning tokens.
    """
    from litellm.types.utils import CostPerToken

    breakdown = get_token_type_cost_breakdown(
        model="openai/onprem-model",
        custom_llm_provider="openai",
        usage=_custom_priced_usage(),
        custom_cost_per_token=CostPerToken(
            input_cost_per_token=1e-6, output_cost_per_token=2e-6, cache_read_input_token_cost=1e-7
        ),
    )

    assert breakdown.cache_read_cost == pytest.approx(800 * 1e-7)
    assert breakdown.cache_creation_cost == pytest.approx(100 * 1e-6)
    assert breakdown.reasoning_cost == pytest.approx(200 * 2e-6)


def test_token_type_cost_breakdown_reconciles_with_custom_pricing_totals():
    from litellm.cost_calculator import cost_per_token
    from litellm.types.utils import CostPerToken

    usage = _custom_priced_usage()
    custom_cost_per_token = CostPerToken(
        input_cost_per_token=1e-6,
        output_cost_per_token=2e-6,
        cache_read_input_token_cost=1e-7,
        cache_creation_input_token_cost=1.25e-6,
    )

    prompt_cost, completion_cost = cost_per_token(
        model="openai/onprem-model",
        custom_llm_provider="openai",
        prompt_tokens=1000,
        completion_tokens=500,
        usage_object=usage,
        custom_cost_per_token=custom_cost_per_token,
    )
    breakdown = get_token_type_cost_breakdown(
        model="openai/onprem-model",
        custom_llm_provider="openai",
        usage=usage,
        custom_cost_per_token=custom_cost_per_token,
    )

    assert 100 * 1e-6 + breakdown.cache_read_cost + breakdown.cache_creation_cost == pytest.approx(prompt_cost)
    assert 300 * 2e-6 + breakdown.reasoning_cost == pytest.approx(completion_cost)


def test_billed_token_rates_follow_the_token_tier_the_breakdown_bills_at(monkeypatch):
    monkeypatch.setitem(
        litellm.model_cost,
        "tiered-cache-model",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "cache_creation_input_token_cost": 3.75e-6,
            "input_cost_per_token_above_200k_tokens": 6e-6,
            "output_cost_per_token_above_200k_tokens": 3e-5,
            "cache_read_input_token_cost_above_200k_tokens": 6e-7,
            "cache_creation_input_token_cost_above_200k_tokens": 7.5e-6,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    usage = Usage(
        prompt_tokens=250_000,
        completion_tokens=1_000,
        total_tokens=251_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=200_000, cache_creation_tokens=10_000),
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=200),
    )

    rates = get_billed_token_rates(model="tiered-cache-model", custom_llm_provider="openai", usage=usage)
    breakdown = get_token_type_cost_breakdown(model="tiered-cache-model", custom_llm_provider="openai", usage=usage)

    assert rates == BilledTokenRates(
        input_cost_per_token=6e-6,
        output_cost_per_token=3e-5,
        cache_read_input_token_cost=6e-7,
        cache_read_input_audio_token_cost=6e-7,
        cache_creation_input_token_cost=7.5e-6,
        cache_creation_input_token_cost_above_1hr=7.5e-6,
        output_cost_per_reasoning_token=3e-5,
    )
    assert breakdown.cache_read_cost == pytest.approx(200_000 * rates.cache_read_input_token_cost)
    assert breakdown.cache_creation_cost == pytest.approx(10_000 * rates.cache_creation_input_token_cost)
    assert breakdown.reasoning_cost == pytest.approx(200 * rates.output_cost_per_reasoning_token)


def test_a_pinned_billing_time_prices_the_totals_and_the_reported_rates_at_one_moment(monkeypatch):
    """Totals and reported rates resolve off-peak pricing on separate paths that each read the
    clock, so a window opening between the two reads used to leave them describing one request
    at two different prices. Pinned, both must answer for the pinned moment."""
    monkeypatch.setitem(
        litellm.model_cost,
        "off-peak-model",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "off_peak_pricing": {
                "hours_utc": "02:00-03:00",
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 5e-6,
            },
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    with pinned_billing_time(datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc)):
        off_peak_prompt_cost, off_peak_completion_cost = generic_cost_per_token(
            model="off-peak-model", usage=usage, custom_llm_provider="openai"
        )
        off_peak_rates = get_billed_token_rates(model="off-peak-model", custom_llm_provider="openai", usage=usage)
    with pinned_billing_time(datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)):
        peak_prompt_cost, peak_completion_cost = generic_cost_per_token(
            model="off-peak-model", usage=usage, custom_llm_provider="openai"
        )
        peak_rates = get_billed_token_rates(model="off-peak-model", custom_llm_provider="openai", usage=usage)

    assert off_peak_rates.input_cost_per_token == pytest.approx(1e-6)
    assert peak_rates.input_cost_per_token == pytest.approx(3e-6)
    assert off_peak_prompt_cost == pytest.approx(1000 * off_peak_rates.input_cost_per_token)
    assert off_peak_completion_cost == pytest.approx(500 * off_peak_rates.output_cost_per_token)
    assert peak_prompt_cost == pytest.approx(1000 * peak_rates.input_cost_per_token)
    assert peak_completion_cost == pytest.approx(500 * peak_rates.output_cost_per_token)


def test_the_token_type_breakdown_carries_the_rates_it_billed_at(monkeypatch):
    """Callers that report both the lines and the rates read the rates off the breakdown rather than
    resolving them a second time, so the breakdown has to hand back exactly what it billed at."""
    monkeypatch.setitem(
        litellm.model_cost,
        "xai/tiered-model",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "input_cost_per_token_above_200k_tokens": 6e-6,
            "output_cost_per_token_above_200k_tokens": 3e-5,
            "cache_read_input_token_cost_above_200k_tokens": 6e-7,
            "litellm_provider": "xai",
            "mode": "chat",
        },
    )
    usage = Usage(
        prompt_tokens=200_000,
        completion_tokens=1_000,
        total_tokens=201_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100_000),
    )

    breakdown = get_token_type_cost_breakdown(model="xai/tiered-model", custom_llm_provider="xai", usage=usage)

    assert breakdown.rates == get_billed_token_rates(model="xai/tiered-model", custom_llm_provider="xai", usage=usage)
    assert breakdown.rates.cache_read_input_token_cost == pytest.approx(6e-7)
    assert breakdown.cache_read_cost == pytest.approx(100_000 * breakdown.rates.cache_read_input_token_cost)


def test_the_token_type_breakdown_reports_no_rates_for_an_unpriced_model():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    breakdown = get_token_type_cost_breakdown(model="no-such-model-anywhere", custom_llm_provider="openai", usage=usage)

    assert breakdown.rates is None


def test_billed_token_rates_are_none_for_an_unpriced_model():
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    assert get_billed_token_rates(model="no-such-model-anywhere", custom_llm_provider="openai", usage=usage) is None


def test_token_type_cost_breakdown_zero_without_special_tokens(_local_model_cost_map):

    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    breakdown = get_token_type_cost_breakdown(model="gpt-4o", custom_llm_provider="openai", usage=usage)

    assert (breakdown.reasoning_cost, breakdown.cache_read_cost, breakdown.cache_creation_cost) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "raw_usage, expect_read, expect_write",
    [
        (
            {
                "input_tokens": 5000,
                "output_tokens": 10,
                "total_tokens": 5010,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 4012},
            },
            False,
            True,
        ),
        (
            {
                "input_tokens": 5000,
                "output_tokens": 10,
                "total_tokens": 5010,
                "input_tokens_details": {"cached_tokens": 4012, "cache_write_tokens": 0},
            },
            True,
            False,
        ),
    ],
)
def test_token_type_cost_breakdown_openai_responses_api_cache_write_read(
    _local_model_cost_map, raw_usage, expect_read, expect_write
):
    """Regression for #34309: OpenAI Responses API reports cache tokens under
    input_tokens_details.{cached_tokens, cache_write_tokens}, not the Anthropic-style
    top-level cache_creation_input_tokens. The itemized breakdown must still populate
    cache_read_cost / cache_creation_cost from the transformed usage."""
    from litellm.responses.utils import ResponseAPILoggingUtils

    model = "gpt-5.6"
    usage = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(raw_usage)

    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider="openai", usage=usage)

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    if expect_write:
        assert breakdown.cache_creation_cost == pytest.approx(4012 * info["cache_creation_input_token_cost"])
        assert breakdown.cache_creation_cost > 0
        assert breakdown.cache_read_cost == 0.0
    if expect_read:
        assert breakdown.cache_read_cost == pytest.approx(4012 * info["cache_read_input_token_cost"])
        assert breakdown.cache_read_cost > 0
        assert breakdown.cache_creation_cost == 0.0


def test_token_type_cost_breakdown_handles_unknown_model_gracefully():
    """A model with no pricing must yield zeros, never raise."""
    breakdown = get_token_type_cost_breakdown(
        model="this-model-does-not-exist-anywhere",
        custom_llm_provider="openai",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=5),
        ),
    )
    assert (breakdown.reasoning_cost, breakdown.cache_read_cost, breakdown.cache_creation_cost) == (0.0, 0.0, 0.0)


def test_token_type_cost_breakdown_applies_regional_uplift(_local_model_cost_map):
    """
    Regional OpenAI hosts (eu./us.) apply a flat uplift to every token cost. The
    per-type breakdown must apply the same uplift via data_residency so it stays
    reconciled with the uplifted input_cost/output_cost totals, instead of being
    logged at the base rate.
    """

    model = "gpt-5.4"
    custom_llm_provider = "openai"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=200, text_tokens=300),
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400, text_tokens=600),
    )

    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    uplift = model_info["regional_processing_uplift_multiplier_eu"]
    assert uplift > 1.0

    base = get_token_type_cost_breakdown(model=model, custom_llm_provider=custom_llm_provider, usage=usage)
    eu = get_token_type_cost_breakdown(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        data_residency="eu",
    )

    assert eu.reasoning_cost == pytest.approx(base.reasoning_cost * uplift)
    assert eu.cache_read_cost == pytest.approx(base.cache_read_cost * uplift)

    # The uplifted breakdown must still reconcile with the uplifted totals.
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        data_residency="eu",
    )
    text_output_cost = 300 * model_info["output_cost_per_token"] * uplift
    text_input_cost = 600 * model_info["input_cost_per_token"] * uplift
    assert text_output_cost + eu.reasoning_cost == pytest.approx(completion_cost)
    assert text_input_cost + eu.cache_read_cost == pytest.approx(prompt_cost)


def test_token_type_cost_breakdown_applies_vertex_regional_uplift(_local_model_cost_map):
    """
    Non-global Vertex endpoints apply a flat 1.1x uplift to every token cost. The
    per-type breakdown must apply the same uplift via vertex_location so it stays
    reconciled with the uplifted input_cost/output_cost totals, instead of being
    logged at the global rate.
    """

    model = "claude-haiku-4-5@20251001"
    custom_llm_provider = "vertex_ai"
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400, text_tokens=600),
    )

    model_info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    uplift = model_info["regional_endpoint_uplift_multiplier"]
    assert uplift > 1.0

    base = get_token_type_cost_breakdown(model=model, custom_llm_provider=custom_llm_provider, usage=usage)
    regional = get_token_type_cost_breakdown(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
        vertex_location="us-east5",
    )

    assert base.cache_read_cost > 0
    assert regional.cache_read_cost == pytest.approx(base.cache_read_cost * uplift)

    # The uplifted breakdown must still reconcile with the uplifted totals.
    prompt_cost, _completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider=custom_llm_provider,
        vertex_location="us-east5",
    )
    text_input_cost = 600 * model_info["input_cost_per_token"] * uplift
    assert text_input_cost + regional.cache_read_cost == pytest.approx(prompt_cost)


def test_token_type_cost_breakdown_applies_anthropic_geo_multiplier(_local_model_cost_map, monkeypatch):
    """
    Anthropic's regional (geo) uplift lives in provider_specific_entry and is
    applied to every token type in the totals, so the per-type breakdown must
    scale its cache and reasoning line items by it too. Otherwise the logged
    cache costs stay at the base rate and the cache uplift is misattributed to
    plain input for exactly the cache-heavy regional traffic the uplift targets.
    """
    from litellm.llms.anthropic.cost_calculation import (
        cost_per_token as anthropic_cost_per_token,
    )

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    model = "claude-test-geo-breakdown-model"
    litellm.register_model(
        model_cost={
            model: {
                "input_cost_per_token": 5e-6,
                "output_cost_per_token": 25e-6,
                "cache_creation_input_token_cost": 6.25e-6,
                "cache_read_input_token_cost": 0.5e-6,
                "litellm_provider": "anthropic",
                "max_tokens": 8192,
                "provider_specific_entry": {"us": 1.1},
            }
        }
    )

    def make_usage() -> Usage:
        return Usage(
            prompt_tokens=10_000,
            completion_tokens=500,
            total_tokens=10_500,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=2_000,
                cache_creation_tokens=6_000,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=200, text_tokens=300),
        )

    base_usage = make_usage()
    geo_usage = make_usage()
    geo_usage.inference_geo = "us"

    base = get_token_type_cost_breakdown(model=model, custom_llm_provider="anthropic", usage=base_usage)
    geo = get_token_type_cost_breakdown(model=model, custom_llm_provider="anthropic", usage=geo_usage)

    assert base.cache_read_cost == pytest.approx(2_000 * 0.5e-6)
    assert base.cache_creation_cost == pytest.approx(6_000 * 6.25e-6)
    assert geo.cache_read_cost == pytest.approx(base.cache_read_cost * 1.1)
    assert geo.cache_creation_cost == pytest.approx(base.cache_creation_cost * 1.1)
    assert geo.reasoning_cost == pytest.approx(base.reasoning_cost * 1.1)

    # The uplifted breakdown must still reconcile with the uplifted totals.
    prompt_cost, completion_cost = anthropic_cost_per_token(model=model, usage=geo_usage)
    text_input_cost = 2_000 * 5e-6 * 1.1
    text_output_cost = 300 * 25e-6 * 1.1
    assert text_input_cost + geo.cache_read_cost + geo.cache_creation_cost == pytest.approx(prompt_cost)
    assert text_output_cost + geo.reasoning_cost == pytest.approx(completion_cost)


@pytest.mark.parametrize("details_as_dict", [True, False])
def test_image_response_input_image_tokens_priced_at_image_rate(details_as_dict):
    """
    Image input tokens must be priced at input_cost_per_image_token even when
    input_tokens_details is a plain dict, as in OpenAI image edit responses.

    Regression test: dict-shaped input_tokens_details was read with getattr(),
    which returns None for dicts, so image input tokens silently fell back to
    the text input rate (e.g. $5/M instead of $8/M for gpt-image-2).
    """
    from unittest.mock import patch

    from litellm.litellm_core_utils.llm_cost_calc.utils import (
        calculate_image_response_cost_from_usage,
    )
    from litellm.types.utils import Usage

    mock_model_info = {
        "input_cost_per_token": 5e-6,
        "input_cost_per_image_token": 8e-6,
        "output_cost_per_image_token": 3e-5,
    }

    input_details = {"text_tokens": 19, "image_tokens": 512}
    image_response = ImageResponse(data=[ImageObject(b64_json="x")])
    # Mirror the usage shape of a real OpenAI images.edit response:
    # a Usage object carrying input_tokens/output_tokens with detail dicts.
    image_response.usage = Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=689,
        input_tokens=531,
        input_tokens_details=(input_details if details_as_dict else ImageUsageInputTokensDetails(**input_details)),
        output_tokens=158,
        output_tokens_details={"image_tokens": 158, "text_tokens": 0},
    )

    with patch(
        "litellm.litellm_core_utils.llm_cost_calc.utils.get_model_info",
        return_value=mock_model_info,
    ):
        cost = calculate_image_response_cost_from_usage(
            model="gpt-image-2",
            image_response=image_response,
            custom_llm_provider="openai",
        )

    expected = 19 * 5e-6 + 512 * 8e-6 + 158 * 3e-5
    assert cost is not None
    assert round(cost, 12) == round(expected, 12)


GEMINI_DAY0_LAUNCH_PRICING = [
    ("gemini-3.6-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("gemini/gemini-3.6-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("vertex_ai/gemini-3.6-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("gemini-3.5-flash-lite", 3e-07, 2.5e-06, 3e-08),
    ("gemini/gemini-3.5-flash-lite", 3e-07, 2.5e-06, 3e-08),
    ("vertex_ai/gemini-3.5-flash-lite", 3e-07, 2.5e-06, 3e-08),
]


GEMINI_36_FLASH_SERVICE_TIER_PRICING = [
    (None, 7.5e-07, 3.75e-06, 7.5e-08),
    ("flex", 3.75e-07, 1.875e-06, 3.75e-08),
    ("priority", 1.35e-06, 6.75e-06, 1.35e-07),
]


GEMINI_35_FLASH_LITE_TIER_RATES_BY_SURFACE = [
    ("gemini", None, 3e-07, 2.5e-06, 3e-08),
    ("gemini", "flex", 1.5e-07, 1.25e-06, 2e-08),
    ("gemini", "priority", 5.4e-07, 4.5e-06, 5e-08),
    ("vertex_ai", None, 3e-07, 2.5e-06, 3e-08),
    ("vertex_ai", "flex", 1.5e-07, 1.25e-06, 1.5e-08),
    ("vertex_ai", "priority", 5.4e-07, 4.5e-06, 5.4e-08),
]


def test_fast_service_tier_is_case_insensitive(_local_model_cost_map):
    from litellm.types.utils import Usage

    usage = Usage(prompt_tokens=1_000, completion_tokens=500)

    assert generic_cost_per_token(
        model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="FAST"
    ) == generic_cost_per_token(model="gpt-5.6-sol", usage=usage, custom_llm_provider="openai", service_tier="fast")


def test_priority_reasoning_tokens_bill_at_the_priority_output_rate(_local_model_cost_map):
    """Regression: gemini-3.5-flash publishes priority output pricing but no priority
    reasoning key, so reasoning tokens under priority/fast were billed at the standard
    output_cost_per_reasoning_token instead of following the tier's output rate."""
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000,
        completion_tokens=5_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=4_000),
    )

    model_info = litellm.get_model_info(model="gemini-3.5-flash", custom_llm_provider="gemini")
    standard_output_rate = model_info["output_cost_per_token"]
    standard_reasoning_rate = model_info["output_cost_per_reasoning_token"]
    priority_output_rate = model_info["output_cost_per_token_priority"]
    assert priority_output_rate is not None
    assert priority_output_rate != standard_reasoning_rate

    standard = generic_cost_per_token(
        model="gemini-3.5-flash", usage=usage, custom_llm_provider="gemini", service_tier=None
    )
    priority = generic_cost_per_token(
        model="gemini-3.5-flash", usage=usage, custom_llm_provider="gemini", service_tier="priority"
    )
    fast = generic_cost_per_token(
        model="gemini-3.5-flash", usage=usage, custom_llm_provider="gemini", service_tier="fast"
    )

    assert standard[1] == pytest.approx(1_000 * standard_output_rate + 4_000 * standard_reasoning_rate, rel=1e-9)
    assert priority[1] == pytest.approx(5_000 * priority_output_rate, rel=1e-9)
    assert fast == priority


def test_explicit_tier_reasoning_key_wins_over_the_tier_output_rate():
    from litellm.types.utils import Usage

    model_info = {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4e-06,
        "output_cost_per_reasoning_token": 6e-06,
        "input_cost_per_token_priority": 2e-06,
        "output_cost_per_token_priority": 8e-06,
        "output_cost_per_reasoning_token_priority": 1.2e-05,
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=1_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=600),
    )

    _, completion_cost = generic_cost_per_token(
        model="synthetic-model",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        model_info=model_info,
    )

    assert completion_cost == pytest.approx(400 * 8e-06 + 600 * 1.2e-05, rel=1e-9)


def test_null_tier_reasoning_key_falls_back_to_the_tier_output_rate():
    """get_model_info dumps every ModelInfo field, so an unpublished tier reasoning key
    arrives as an explicit None and must not shadow the tier output rate."""
    from litellm.types.utils import Usage

    model_info = {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4e-06,
        "output_cost_per_reasoning_token": 6e-06,
        "output_cost_per_reasoning_token_priority": None,
        "input_cost_per_token_priority": 2e-06,
        "output_cost_per_token_priority": 8e-06,
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=1_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=600),
    )

    _, completion_cost = generic_cost_per_token(
        model="synthetic-model",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        model_info=model_info,
    )

    assert completion_cost == pytest.approx(1_000 * 8e-06, rel=1e-9)


def test_tier_request_without_tier_pricing_keeps_the_standard_reasoning_rate():
    from litellm.types.utils import Usage

    model_info = {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 4e-06,
        "output_cost_per_reasoning_token": 6e-06,
    }
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=1_000,
        completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=600),
    )

    _, completion_cost = generic_cost_per_token(
        model="synthetic-model",
        usage=usage,
        custom_llm_provider="openai",
        service_tier="priority",
        model_info=model_info,
    )

    assert completion_cost == pytest.approx(400 * 4e-06 + 600 * 6e-06, rel=1e-9)


GEMINI_37_FLASH_LAUNCH_PRICING = [
    ("gemini-3.7-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("gemini/gemini-3.7-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("vertex_ai/gemini-3.7-flash", 7.5e-07, 3.75e-06, 7.5e-08),
]


GEMINI_38_FLASH_LAUNCH_PRICING = [
    ("gemini-3.8-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("gemini/gemini-3.8-flash", 7.5e-07, 3.75e-06, 7.5e-08),
    ("vertex_ai/gemini-3.8-flash", 7.5e-07, 3.75e-06, 7.5e-08),
]


GEMINI_38_FLASH_FIELDS_SHARED_WITH_37_FLASH = (
    "input_cost_per_token",
    "output_cost_per_token",
    "output_cost_per_reasoning_token",
    "cache_read_input_token_cost",
    "input_cost_per_token_batches",
    "output_cost_per_token_batches",
    "input_cost_per_token_flex",
    "output_cost_per_token_flex",
    "cache_read_input_token_cost_flex",
    "input_cost_per_token_priority",
    "output_cost_per_token_priority",
    "cache_read_input_token_cost_priority",
    "search_context_cost_per_query",
    "google_maps_grounding_cost_per_query",
    "prompt_cache_min_tokens",
    "max_input_tokens",
    "max_output_tokens",
    "supports_reasoning",
    "supports_function_calling",
    "supports_prompt_caching",
    "supports_vision",
    "supports_pdf_input",
    "supports_audio_input",
    "supports_video_input",
    "supports_response_schema",
    "supports_tool_choice",
    "supports_web_search",
    "supports_url_context",
)


@pytest.mark.parametrize(
    ("response_quality", "requested_quality", "expected_cost"),
    [
        (None, "low", 0.04),
        (None, None, 0.06),
        ("high", "low", 0.08),
    ],
)
def test_route_image_generation_cost_falls_back_to_requested_quality(
    monkeypatch, response_quality, requested_quality, expected_cost
):
    def tier(cost):
        return {"litellm_provider": "xai", "mode": "image_generation", "input_cost_per_image": cost}

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "xai/grok-imagine-image-2.0": tier(0.06),
            "low/1024-x-1024/grok-imagine-image-2.0": tier(0.04),
            "high/1024-x-1024/grok-imagine-image-2.0": tier(0.08),
        },
    )
    response = ImageResponse(data=[ImageObject(url="https://example.com/image.png")], quality=response_quality)
    optional_params = {} if requested_quality is None else {"quality": requested_quality}

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="xai/grok-imagine-image-2.0",
        completion_response=response,
        custom_llm_provider="xai",
        optional_params=optional_params,
        call_type="image_generation",
    )

    assert cost == expected_cost


@pytest.mark.parametrize(
    ("requested_size", "expected_cost"),
    [
        ("1536x1024", 0.05),
        ("1536-x-1024", 0.05),
        ("auto", 0.04),
        (None, 0.04),
    ],
)
def test_route_image_generation_cost_falls_back_to_requested_size(monkeypatch, requested_size, expected_cost):
    def tier(cost):
        return {"litellm_provider": "xai", "mode": "image_generation", "input_cost_per_image": cost}

    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "xai/grok-imagine-image-2.0": tier(0.06),
            "low/1024-x-1024/grok-imagine-image-2.0": tier(0.04),
            "low/1536-x-1024/grok-imagine-image-2.0": tier(0.05),
        },
    )
    response = ImageResponse(data=[ImageObject(url="https://example.com/image.png")])
    optional_params = {"quality": "low", **({} if requested_size is None else {"size": requested_size})}

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="xai/grok-imagine-image-2.0",
        completion_response=response,
        custom_llm_provider="xai",
        optional_params=optional_params,
        call_type="image_generation",
    )

    assert cost == expected_cost


def test_generic_cost_per_token_bills_reasoning_nested_in_text_tokens_once(_local_model_cost_map: None) -> None:
    """
    Realtime usage (OpenAI and Azure) reports output_tokens == text_tokens + audio_tokens with
    reasoning_tokens already counted inside text_tokens, so reasoning must not be billed on top.
    """

    model = "gpt-realtime-2.1-mini"
    usage = Usage(
        prompt_tokens=346,
        completion_tokens=29,
        total_tokens=375,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=152, image_tokens=194, audio_tokens=0, cached_tokens=128
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=29, audio_tokens=0, reasoning_tokens=19),
    )

    prompt_cost, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")
    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider="openai", usage=usage)

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    assert completion_cost == pytest.approx(29 * info["output_cost_per_token"])
    assert completion_cost - breakdown.reasoning_cost == pytest.approx(10 * info["output_cost_per_token"])
    assert prompt_cost == pytest.approx(
        24 * info["input_cost_per_token"]
        + 128 * info["cache_read_input_token_cost"]
        + 194 * info["input_cost_per_image_token"]
    )


def test_generic_cost_per_token_keeps_billing_reasoning_reported_beside_text_tokens(
    _local_model_cost_map: None,
) -> None:
    """Providers whose text_tokens exclude reasoning (text + reasoning == completion) stay billed in full."""

    model = "gpt-realtime-2.1-mini"
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=44,
        total_tokens=144,
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=25, audio_tokens=0, reasoning_tokens=19),
    )

    _, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    assert completion_cost == pytest.approx(44 * info["output_cost_per_token"])


def test_generic_cost_per_token_strips_only_the_reasoning_share_when_text_over_reports(
    _local_model_cost_map: None,
) -> None:
    """Text over-reported past the reasoning share keeps its extra tokens billed; only the nested reasoning is netted out."""

    model = "gpt-realtime-2.1-mini"
    usage = Usage(
        prompt_tokens=120,
        completion_tokens=100,
        total_tokens=220,
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=100, audio_tokens=70, reasoning_tokens=10),
    )

    _, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")
    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider="openai", usage=usage)

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    assert breakdown.reasoning_cost == pytest.approx(10 * info["output_cost_per_token"])
    assert completion_cost == pytest.approx(
        100 * info["output_cost_per_token"] + 70 * info["output_cost_per_audio_token"]
    )


def test_generic_cost_per_token_bills_nested_reasoning_once_beside_audio_output(_local_model_cost_map: None) -> None:
    """Audio-output realtime usage nests reasoning inside text_tokens next to audio_tokens; text is billed net of it."""

    model = "gpt-realtime-2.1-mini"
    usage = Usage(
        prompt_tokens=120,
        completion_tokens=100,
        total_tokens=220,
        completion_tokens_details=CompletionTokensDetailsWrapper(text_tokens=30, audio_tokens=70, reasoning_tokens=20),
    )

    _, completion_cost = generic_cost_per_token(model=model, usage=usage, custom_llm_provider="openai")
    breakdown = get_token_type_cost_breakdown(model=model, custom_llm_provider="openai", usage=usage)

    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    assert breakdown.reasoning_cost == pytest.approx(20 * info["output_cost_per_token"])
    assert completion_cost == pytest.approx(
        30 * info["output_cost_per_token"] + 70 * info["output_cost_per_audio_token"]
    )


def test_cached_audio_tokens_fall_back_to_cache_read_input_token_cost() -> None:
    model_info: ModelInfo = {
        "input_cost_per_token": 4e-6,
        "input_cost_per_audio_token": 32e-6,
        "cache_read_input_token_cost": 5e-7,
    }
    usage = Usage(
        prompt_tokens=283,
        completion_tokens=0,
        total_tokens=283,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=116,
            audio_tokens=167,
            cached_tokens=192,
            cached_tokens_details={"text_tokens": 64, "audio_tokens": 128},
        ),
    )

    prompt_cost, _ = generic_cost_per_token(
        model="some-realtime-model",
        usage=usage,
        custom_llm_provider="openai",
        model_info=model_info,
    )
    expected = 52 * 4e-6 + 64 * 5e-7 + 39 * 32e-6 + 128 * 5e-7
    assert prompt_cost == pytest.approx(expected)


def test_generic_cost_per_token_bills_cache_creation_at_the_input_rate_without_a_write_price():
    """Azure and OpenAI publish no cache-write price and bill cache writes as ordinary input.
    A deployment priced with only input, output, and cache-read rates must bill the creation
    tokens the provider reports at the input rate, never at 0. The numbers are a cold 7,336-token
    prompt on a deployment that reports all but 3 of them as cache creation."""
    model_info = {
        "input_cost_per_token": 2e-7,
        "output_cost_per_token": 1.25e-6,
        "cache_read_input_token_cost": 2e-8,
    }
    usage = Usage(
        prompt_tokens=7336,
        completion_tokens=23,
        total_tokens=7359,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=0, cache_creation_tokens=7333),
    )

    prompt_cost, completion_cost = generic_cost_per_token(
        model="custom-priced-deployment", usage=usage, custom_llm_provider="azure", model_info=model_info
    )

    assert prompt_cost == pytest.approx(7336 * 2e-7)
    assert completion_cost == pytest.approx(23 * 1.25e-6)


@pytest.mark.parametrize(
    ("cache_rates", "current_time", "expected_creation", "expected_creation_1h"),
    (
        pytest.param({}, None, 2e-7, 2e-7, id="no-write-price-uses-the-input-rate"),
        pytest.param(
            {"cache_creation_input_token_cost": 2.5e-7}, None, 2.5e-7, 2.5e-7, id="no-1h-price-uses-the-write-price"
        ),
        pytest.param({"cache_creation_input_token_cost": 0.0}, None, 0.0, 0.0, id="explicit-zero-stays-zero"),
        pytest.param(
            {"off_peak_pricing": {"hours_utc": "00:00-23:59", "input_cost_per_token": 1e-7}},
            datetime(2026, 9, 14, 12, tzinfo=timezone.utc),
            1e-7,
            1e-7,
            id="no-write-price-uses-the-off-peak-input-rate",
        ),
        pytest.param(
            {
                "off_peak_pricing": {
                    "hours_utc": "00:00-23:59",
                    "input_cost_per_token": 1e-7,
                    "cache_creation_input_token_cost": 3e-7,
                }
            },
            datetime(2026, 9, 14, 12, tzinfo=timezone.utc),
            3e-7,
            3e-7,
            id="no-1h-price-uses-the-off-peak-write-price",
        ),
    ),
)
def test_get_token_base_cost_resolves_missing_cache_write_rates_like_the_tiered_path(
    cache_rates: Mapping[str, float | Mapping[str, float | str]],
    current_time: datetime | None,
    expected_creation: float,
    expected_creation_1h: float,
) -> None:
    model_info = {"input_cost_per_token": 2e-7, "output_cost_per_token": 1.25e-6, **cache_rates}
    usage = Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11)

    _, _, creation, creation_1h, _ = _get_token_base_cost(model_info, usage, current_time=current_time)

    assert creation == pytest.approx(expected_creation)
    assert creation_1h == pytest.approx(expected_creation_1h)


def _image_response(num_images: int = 1, usage: ImageUsage | None = None) -> ImageResponse:
    return ImageResponse(
        data=[ImageObject(url="https://example.com/img.png") for _ in range(num_images)],
        usage=usage,
    )


_GPT_IMAGE_2_HIGH_1024: Final = {"quality": "high", "image_size": {"width": 1024, "height": 1024}}


@pytest.mark.parametrize(
    ("model", "optional_params", "model_info", "num_images", "expected_cost"),
    [
        ("fal-ai/unlisted-image-model", None, {"output_cost_per_image": 0.08}, 1, 0.08),
        ("fal-ai/unlisted-image-model", None, {"output_cost_per_image": 0.08}, 2, 0.16),
        ("fal-ai/unlisted-image-model", None, {"output_cost_per_image": "0.08"}, 1, 0.08),
        ("openai/gpt-image-2", _GPT_IMAGE_2_HIGH_1024, {"output_cost_per_image": 0.5}, 1, 0.5),
        ("openai/gpt-image-2", _GPT_IMAGE_2_HIGH_1024, {"mode": "image_generation"}, 1, 0.211),
        ("openai/gpt-image-2", _GPT_IMAGE_2_HIGH_1024, {"output_cost_per_image": "0.08 USD"}, 1, 0.211),
    ],
)
def test_route_image_generation_cost_honors_deployment_model_info(
    _local_model_cost_map: None,
    model: str,
    optional_params: dict[str, object] | None,
    model_info: ModelInfo,
    num_images: int,
    expected_cost: float,
) -> None:
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=_image_response(num_images),
        custom_llm_provider="fal_ai",
        optional_params=optional_params,
        call_type="image_generation",
        model_info=model_info,
    )

    assert cost == pytest.approx(expected_cost)


def test_route_image_generation_cost_openai_honors_deployment_input_cost_per_image(
    _local_model_cost_map: None,
) -> None:
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="dall-e-3",
        completion_response=_image_response(),
        custom_llm_provider="openai",
        quality="standard",
        size="1024-x-1024",
        call_type="image_generation",
        model_info={"input_cost_per_image": 0.07},
    )

    assert cost == pytest.approx(0.07)




@pytest.mark.parametrize(
    ("custom_llm_provider", "model"),
    [
        ("gemini", "gemini/unlisted-image-model"),
        ("vertex_ai", "vertex_ai/unlisted-image-model"),
        ("azure_ai", "unlisted-image-model"),
        ("openai", "gpt-image-unlisted"),
    ],
)
def test_route_image_generation_cost_bills_deployment_image_price_when_unlisted_model_reports_tokens(
    _local_model_cost_map: None,
    custom_llm_provider: str,
    model: str,
) -> None:
    usage = ImageUsage(
        input_tokens=10,
        input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=10),
        output_tokens=1290,
        total_tokens=1300,
    )

    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model=model,
        completion_response=_image_response(num_images=2, usage=usage),
        custom_llm_provider=custom_llm_provider,
        call_type="image_generation",
        model_info={"output_cost_per_image": 0.05},
    )

    assert cost == pytest.approx(0.10)


def _batch_rates_model_info(**rates: object) -> ModelInfo:
    return cast(ModelInfo, dict(rates))


def test_get_batch_cost_rates_parses_string_rates():
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_batch_cost_rates

    rates = get_batch_cost_rates(
        _batch_rates_model_info(
            input_cost_per_token_batches="1e-06",
            output_cost_per_token_batches="4e-06",
            cache_read_input_token_cost_batches="1e-07",
            input_cost_per_token_above_272k_tokens_batches="2e-06",
            output_cost_per_token_above_272k_tokens_batches="6e-06",
            cache_read_input_token_cost_above_272k_tokens_batches="2e-07",
        ),
        Usage(prompt_tokens=300_000, completion_tokens=1, total_tokens=300_001),
        "openai",
    )

    assert (rates.input, rates.output, rates.cache_read) == (2e-6, 6e-6, 2e-7)


def test_get_batch_cost_rates_falls_back_to_the_flat_rates_when_tier_rates_are_unparsable():
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_batch_cost_rates

    rates = get_batch_cost_rates(
        _batch_rates_model_info(
            input_cost_per_token_batches=1e-6,
            output_cost_per_token_batches=4e-6,
            cache_read_input_token_cost_batches=1e-7,
            input_cost_per_token_above_272k_tokens_batches="two",
            output_cost_per_token_above_272k_tokens_batches="six",
            cache_read_input_token_cost_above_272k_tokens_batches="none",
            cache_creation_input_token_cost_batches=1.25e-7,
            cache_creation_input_token_cost_above_272k_tokens_batches="nope",
        ),
        Usage(prompt_tokens=300_000, completion_tokens=1, total_tokens=300_001),
        "openai",
    )

    assert (rates.input, rates.output, rates.cache_read, rates.cache_creation) == (1e-6, 4e-6, 1e-7, 1.25e-7)


def test_get_batch_cost_rates_has_no_cached_rate_without_a_cached_batch_key():
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_batch_cost_rates

    rates = get_batch_cost_rates(
        _batch_rates_model_info(
            input_cost_per_token_batches=1e-6,
            output_cost_per_token_batches=4e-6,
            cache_read_input_token_cost=2e-7,
            input_cost_per_token_above_272k_tokens_batches=2e-6,
        ),
        Usage(prompt_tokens=300_000, completion_tokens=1, total_tokens=300_001),
        "openai",
    )

    assert (rates.input, rates.output, rates.cache_read) == (2e-6, 4e-6, None)


@pytest.mark.parametrize(("prompt_tokens", "expected"), [(1_000, 1.25e-7), (272_000, 1.25e-7), (300_000, 2.5e-7)])
def test_get_batch_cost_rates_reads_the_batch_cache_write_rate_for_the_crossed_tier(prompt_tokens, expected):
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_batch_cost_rates

    rates = get_batch_cost_rates(
        _batch_rates_model_info(
            input_cost_per_token_batches=1e-7,
            input_cost_per_token_above_272k_tokens_batches=2e-7,
            cache_creation_input_token_cost_batches=1.25e-7,
            cache_creation_input_token_cost_above_272k_tokens_batches=2.5e-7,
        ),
        Usage(prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1),
        "openai",
    )

    assert rates.cache_creation == expected


@pytest.mark.parametrize(
    ("tier_key", "attribute"),
    [
        ("output_cost_per_token_above_272k_tokens_batches", "output"),
        ("cache_read_input_token_cost_above_272k_tokens_batches", "cache_read"),
        ("cache_creation_input_token_cost_above_272k_tokens_batches", "cache_creation"),
    ],
)
def test_get_batch_cost_rates_crosses_a_tier_declared_without_an_input_tier_key(tier_key, attribute):
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_batch_cost_rates

    rates = get_batch_cost_rates(
        _batch_rates_model_info(
            input_cost_per_token_batches=1e-6,
            output_cost_per_token_batches=4e-6,
            cache_read_input_token_cost_batches=1e-7,
            cache_creation_input_token_cost_batches=1.25e-7,
            **{tier_key: 9e-6},
        ),
        Usage(prompt_tokens=300_000, completion_tokens=1, total_tokens=300_001),
        "openai",
    )

    assert getattr(rates, attribute) == 9e-6
    assert rates.input == 1e-6


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_input", "expected_output"),
    [(200_000, 1e-6, 4e-6), (250_000, 1e-6, 5e-6), (300_000, 2e-6, 5e-6)],
)
def test_get_batch_cost_rates_crosses_each_components_own_tier(prompt_tokens, expected_input, expected_output):
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_batch_cost_rates

    rates = get_batch_cost_rates(
        _batch_rates_model_info(
            input_cost_per_token_batches=1e-6,
            input_cost_per_token_above_272k_tokens_batches=2e-6,
            output_cost_per_token_batches=4e-6,
            output_cost_per_token_above_200k_tokens_batches=5e-6,
        ),
        Usage(prompt_tokens=prompt_tokens, completion_tokens=1, total_tokens=prompt_tokens + 1),
        "openai",
    )

    assert (rates.input, rates.output) == (expected_input, expected_output)


def test_get_batch_cost_rates_has_no_cache_write_rate_without_a_cache_write_batch_key():
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_batch_cost_rates

    rates = get_batch_cost_rates(
        _batch_rates_model_info(
            input_cost_per_token_batches=1e-7,
            input_cost_per_token_above_272k_tokens_batches=2e-7,
            cache_creation_input_token_cost=2.5e-7,
            cache_creation_input_token_cost_above_272k_tokens=5e-7,
        ),
        Usage(prompt_tokens=300_000, completion_tokens=1, total_tokens=300_001),
        "openai",
    )

    assert rates.cache_creation is None


@pytest.mark.parametrize("model_base", ["gpt-6-sol", "gpt-6-luna"])
def test_azure_gpt_6_foundry_price_sheet(_local_model_cost_map, model_base):
    """Azure Foundry hosts gpt-6-sol and gpt-6-luna at OpenAI's Global rates, with the
    US and EU data zones charging fixed uplifts on top of them."""
    price_fields = (
        "input_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "output_cost_per_token",
        "input_cost_per_token_above_272k_tokens",
        "cache_read_input_token_cost_above_272k_tokens",
        "cache_creation_input_token_cost_above_272k_tokens",
        "output_cost_per_token_above_272k_tokens",
    )
    openai_info = litellm.get_model_info(model=model_base, custom_llm_provider="openai")
    azure_info = litellm.get_model_info(model=f"azure/{model_base}", custom_llm_provider="azure")
    azure_us_info = litellm.get_model_info(model=f"azure/us/{model_base}", custom_llm_provider="azure")
    azure_eu_info = litellm.get_model_info(model=f"azure/eu/{model_base}", custom_llm_provider="azure")
    azure_ai_info = litellm.get_model_info(model=f"azure_ai/{model_base}", custom_llm_provider="azure_ai")

    for field in price_fields:
        base = openai_info[field]
        assert azure_info[field] == base
        assert azure_ai_info[field] == base
        assert azure_us_info[field] == pytest.approx(1.1 * base)
        assert azure_eu_info[field] == pytest.approx(1.2 * base)
