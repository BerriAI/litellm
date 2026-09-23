"""Generates differential fixtures by executing the real Python cost calculators.

The inputs are ours (synthetic rates and realistic usage shapes); only the
outputs are Python's. Run from litellm-rust with:

    PYTHONPATH=.. ../.venv/bin/python crates/cost/tests/generate_python_fixtures.py
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _is_off_peak,
    calculate_prompt_caching_savings,
    generic_cost_per_token,
    get_batch_cost_rates,
    get_billed_token_rates,
)
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.utils import register_model
from litellm.types.utils import Usage

UTC = timezone.utc


def details(
    cached: int | None = None,
    cache_write: int | None = None,
    audio_in: int | None = None,
    audio_out: int | None = None,
    image: int | None = None,
    text: int | None = None,
    reasoning: int | None = None,
) -> dict[str, int]:
    prompt: dict[str, int] = {}
    completion: dict[str, int] = {}
    if cached is not None:
        prompt["cached_tokens"] = cached
    if cache_write is not None:
        prompt["cache_write_tokens"] = cache_write
    if audio_in is not None:
        prompt["audio_tokens"] = audio_in
    if audio_out is not None:
        completion["audio_tokens"] = audio_out
    if image is not None:
        completion["image_tokens"] = image
    if text is not None:
        completion["text_tokens"] = text
    if reasoning is not None:
        completion["reasoning_tokens"] = reasoning
    return {"prompt": prompt, "completion": completion}


def usage(prompt: int, completion: int, shape: dict[str, int] | None = None) -> dict[str, Any]:
    shape = shape or {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": shape.get("prompt") or None,
        "completion_tokens_details": shape.get("completion") or None,
    }


def make_usage(payload: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=payload["prompt_tokens"],
        completion_tokens=payload["completion_tokens"],
        prompt_tokens_details=payload.get("prompt_tokens_details") or {},
        completion_tokens_details=payload.get("completion_tokens_details") or {},
    )


GENERIC_CASES: tuple[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]], ...] = (
    (
        "audio_mixed_modalities",
        usage(1000, 500, details(audio_in=200, audio_out=150, text=350, reasoning=0)),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "input_cost_per_audio_token": 1e-5,
            "output_cost_per_audio_token": 2e-5,
            "output_cost_per_reasoning_token": 8e-6,
        },
        {},
    ),
    (
        "anthropic_cache_ttl_split",
        usage(
            10_000,
            900,
            details(
                cached=8_000,
                text=800,
                reasoning=100,
            ),
        ),
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "output_cost_per_reasoning_token": 6e-6,
            "litellm_provider": "anthropic",
        },
        {},
    ),
    (
        "cache_write_plus_inclusive_threshold",
        usage(130_000, 2_000, details(cached=40_000, cache_write=5_000)),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 10e-6,
            "cache_creation_input_token_cost": 5e-6,
            "input_cost_per_token_above_128k_tokens": 4e-6,
            "output_cost_per_token_above_128k_tokens": 20e-6,
            "litellm_provider": "xai",
        },
        {},
    ),
    (
        "exclusive_threshold_at_boundary",
        usage(128_000, 500, details(cached=1_000)),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 10e-6,
            "input_cost_per_token_above_128k_tokens": 4e-6,
            "output_cost_per_token_above_128k_tokens": 20e-6,
            "litellm_provider": "openai",
        },
        {},
    ),
    (
        "service_tier_priority_rates",
        usage(5_000, 1_000, details(reasoning=400)),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "input_cost_per_token_priority": 3e-6,
            "output_cost_per_token_priority": 6e-6,
            "output_cost_per_reasoning_token_priority": 9e-6,
        },
        {"service_tier": "priority"},
    ),
    (
        "service_tier_flex_uses_flex_rates",
        usage(5_000, 1_000),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "input_cost_per_token_flex": 5e-7,
            "output_cost_per_token_flex": 1e-6,
        },
        {"service_tier": "flex"},
    ),
    (
        "service_tier_unpriced_falls_back_to_base",
        usage(5_000, 1_000),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
        },
        {"service_tier": "flex"},
    ),
    (
        "data_residency_uplift",
        usage(1_000, 200),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "regional_processing_uplift_multiplier_eu": 1.5,
        },
        {"data_residency": "eu"},
    ),
    (
        "image_output_tokens",
        usage(500, 700, details(image=600, text=100)),
        {
            "input_cost_per_token": 5e-7,
            "output_cost_per_token": 1e-6,
            "output_cost_per_image_token": 5e-6,
        },
        {},
    ),
    (
        "free_cache_read_and_write",
        usage(1_000, 200, details(cached=500, cache_write=300)),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "cache_read_input_token_cost": 0.0,
            "cache_creation_input_token_cost": 0.0,
        },
        {},
    ),
    (
        "reasoning_default_follows_output",
        usage(1_000, 600, details(reasoning=250, text=350)),
        {"input_cost_per_token": 1e-6, "output_cost_per_token": 4e-6},
        {},
    ),
    (
        "off_peak_window_active",
        usage(1_000, 200, details(reasoning=50)),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "off_peak_pricing": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "output_cost_per_reasoning_token": 1e-6,
                "hours_utc": "22:00-06:00",
            },
        },
        {"current_time": datetime(2026, 3, 12, 2, 30, tzinfo=UTC)},
    ),
    (
        "off_peak_window_inactive",
        usage(1_000, 200, details(reasoning=50)),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "off_peak_pricing": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "hours_utc": "22:00-06:00",
            },
        },
        {"current_time": datetime(2026, 3, 12, 14, 0, tzinfo=UTC)},
    ),
    (
        "off_peak_weekday_calendar",
        usage(1_000, 200),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "off_peak_pricing": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "hours_utc": "00:00-00:00",
                "weekdays": ["sat", "sun"],
                "weekday_timezone": "America/New_York",
            },
        },
        {"current_time": datetime(2026, 3, 14, 12, 0, tzinfo=UTC)},
    ),
    (
        "off_peak_weekday_blocked_on_weekday",
        usage(1_000, 200),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "off_peak_pricing": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "hours_utc": "00:00-00:00",
                "weekdays": ["sat", "sun"],
                "weekday_timezone": "America/New_York",
            },
        },
        {"current_time": datetime(2026, 3, 12, 12, 0, tzinfo=UTC)},
    ),
    (
        "tiered_base_costs_low_tier",
        usage(500, 100, details(reasoning=40)),
        {
            "tiered_pricing": [
                {"range": [0, 16_000], "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "output_cost_per_reasoning_token": 3e-6},
                {"range": [16_000, 128_000], "input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6, "output_cost_per_reasoning_token": 6e-6},
            ]
        },
        {"custom_llm_provider": "dashscope"},
    ),
    (
        "tiered_base_costs_high_tier",
        usage(50_000, 100, details(reasoning=40)),
        {
            "tiered_pricing": [
                {"range": [0, 16_000], "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "output_cost_per_reasoning_token": 3e-6},
                {"range": [16_000, 128_000], "input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6, "output_cost_per_reasoning_token": 6e-6},
            ]
        },
        {"custom_llm_provider": "dashscope"},
    ),
    (
        "tier_boundary_stays_in_lower_tier",
        usage(16_000, 100, details(reasoning=40)),
        {
            "tiered_pricing": [
                {"range": [0, 16_000], "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "output_cost_per_reasoning_token": 3e-6},
                {"range": [16_000, 128_000], "input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6, "output_cost_per_reasoning_token": 6e-6},
            ]
        },
        {"custom_llm_provider": "dashscope"},
    ),
    (
        "tier_cache_rates_fall_back_to_tier_input",
        usage(20_000, 100, details(cached=5_000, cache_write=1_000)),
        {
            "input_cost_per_token": 5e-6,
            "output_cost_per_token": 10e-6,
            "cache_read_input_token_cost": 1e-6,
            "tiered_pricing": [
                {"range": [0, 16_000], "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6},
                {"range": [16_000, 128_000], "input_cost_per_token": 2e-6, "output_cost_per_token": 4e-6},
            ],
        },
        {"custom_llm_provider": "dashscope"},
    ),
    (
        "cache_read_falls_back_to_input_rate",
        usage(1_000, 200, details(cached=600)),
        {"input_cost_per_token": 3e-6, "output_cost_per_token": 9e-6, "litellm_provider": "openai"},
        {},
    ),
    (
        "fireworks_cache_read_default_applies",
        usage(1_000, 200, details(cached=600)),
        {"input_cost_per_token": 2e-7, "output_cost_per_token": 6e-7, "litellm_provider": "fireworks_ai"},
        {"custom_llm_provider": "fireworks_ai"},
    ),
    (
        "fireworks_cache_read_default_off_peak_untouched",
        usage(1_000, 200, details(cached=600)),
        {
            "input_cost_per_token": 2e-7,
            "output_cost_per_token": 6e-7,
            "litellm_provider": "fireworks_ai",
            "off_peak_pricing": {"input_cost_per_token": 1e-7, "hours_utc": "00:00-00:00"},
        },
        {"custom_llm_provider": "fireworks_ai", "current_time": datetime(2026, 3, 12, 2, 30, tzinfo=UTC)},
    ),
    (
        "zero_rated_everything",
        usage(10_000, 5_000, details(cached=5_000, reasoning=1_000)),
        {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
        {},
    ),
    (
        "vertex_regional_endpoint_uplift",
        usage(1_000, 200),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "regional_endpoint_uplift_multiplier": 1.10,
        },
        {"vertex_location": "us-east1", "custom_llm_provider": "vertex_ai"},
    ),
    (
        "vertex_global_location_no_uplift",
        usage(1_000, 200),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "regional_endpoint_uplift_multiplier": 1.10,
        },
        {"vertex_location": "global", "custom_llm_provider": "vertex_ai"},
    ),
)


def generic_surface() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload, model_info, extra in GENERIC_CASES:
        call = {
            "usage": payload,
            "model_info": model_info,
            "custom_llm_provider": extra.get("custom_llm_provider", "openai"),
            "service_tier": extra.get("service_tier"),
            "data_residency": extra.get("data_residency"),
            "vertex_location": extra.get("vertex_location"),
            "current_time": extra["current_time"].isoformat() if "current_time" in extra else None,
        }
        result = generic_cost_per_token(
            model="fixture-model",
            usage=make_usage(payload),
            custom_llm_provider=call["custom_llm_provider"],
            service_tier=call["service_tier"],
            data_residency=call["data_residency"],
            model_info=model_info,
            vertex_location=call["vertex_location"],
            current_time=extra.get("current_time"),
        )
        rows.append({"id": name, **call, "expected_input_cost": result[0], "expected_output_cost": result[1]})
    return rows


BATCH_CASES: tuple[tuple[str, dict[str, Any], int, int, str], ...] = (
    (
        "single_batch_threshold_crossed",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "input_cost_per_token_batches": 1.5e-6,
            "output_cost_per_token_batches": 7.5e-6,
            "input_cost_per_token_above_100k_tokens_batches": 7.5e-7,
            "output_cost_per_token_above_100k_tokens_batches": 3.75e-6,
        },
        150_000,
        20_000,
        "anthropic",
    ),
    (
        "batch_threshold_not_crossed",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "input_cost_per_token_batches": 1.5e-6,
            "output_cost_per_token_batches": 7.5e-6,
            "input_cost_per_token_above_100k_tokens_batches": 7.5e-7,
            "output_cost_per_token_above_100k_tokens_batches": 3.75e-6,
        },
        99_999,
        20_000,
        "anthropic",
    ),
    (
        "inclusive_batch_provider_at_boundary",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "input_cost_per_token_batches": 1.5e-6,
            "output_cost_per_token_batches": 7.5e-6,
            "input_cost_per_token_above_100k_tokens_batches": 7.5e-7,
            "output_cost_per_token_above_100k_tokens_batches": 3.75e-6,
        },
        100_000,
        20_000,
        "vertex_ai",
    ),
    (
        "batch_cache_rates_use_batch_fractions",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "cache_creation_input_token_cost": 3.75e-6,
            "input_cost_per_token_batches": 1.5e-6,
            "output_cost_per_token_batches": 7.5e-6,
            "cache_read_input_token_cost_batches": 1.5e-7,
            "cache_creation_input_token_cost_batches": 1.875e-6,
        },
        50_000,
        10_000,
        "anthropic",
    ),
    (
        "batch_cache_rates_absent_stay_none",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "input_cost_per_token_batches": 1.5e-6,
            "output_cost_per_token_batches": 7.5e-6,
        },
        50_000,
        10_000,
        "anthropic",
    ),
    (
        "no_batch_rates_falls_back_to_base",
        {"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6},
        50_000,
        10_000,
        "anthropic",
    ),
)


def batch_surface() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, model_info, prompt, completion, provider in BATCH_CASES:
        payload = usage(prompt, completion, details(cached=2_000, cache_write=1_000))
        rates = get_batch_cost_rates(model_info, make_usage(payload), provider)
        rows.append(
            {
                "id": name,
                "usage": payload,
                "model_info": model_info,
                "custom_llm_provider": provider,
                "expected_input": rates.input,
                "expected_output": rates.output,
                "expected_cache_read": rates.cache_read,
                "expected_cache_creation": rates.cache_creation,
            }
        )
    return rows


OFF_PEAK_TIMES = (
    "2026-03-12T02:30:00+00:00",
    "2026-03-12T14:00:00+00:00",
    "2026-03-14T12:00:00+00:00",
    "2026-03-14T20:00:00+00:00",
    "2026-07-04T01:00:00+00:00",
)
OFF_PEAK_BLOCKS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("simple_window", {"hours_utc": "22:00-06:00"}),
    ("multi_window", {"hours_utc": ["00:30-02:00", "22:00-23:30"]}),
    ("weekend_only", {"hours_utc": "00:00-00:00", "weekdays": ["sat", "sun"]}),
    ("weekday_calendar_est", {"hours_utc": "00:00-00:00", "weekdays": ["mon", "wed"], "weekday_timezone": "America/New_York"}),
    ("window_rules_with_weekdays", {"windows": [{"hours_utc": "00:00-06:00", "weekdays": ["sat", "sun"]}]}),
    ("window_rule_wrong_weekday", {"windows": [{"hours_utc": "00:00-06:00", "weekdays": ["mon"]}]}),
    ("malformed_window_ignored", {"hours_utc": "22-06"}),
)


def off_peak_surface() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_name, block in OFF_PEAK_BLOCKS:
        for moment in OFF_PEAK_TIMES:
            at = datetime.fromisoformat(moment)
            rows.append(
                {
                    "id": f"{block_name}_{moment[11:16]}",
                    "off_peak": block,
                    "current_time": moment,
                    "expected_is_off_peak": _is_off_peak(block, at),
                }
            )
    return rows


RESPONSES_USAGE_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "cached_details_with_web_search",
        {
            "input_tokens": 1_200,
            "output_tokens": 300,
            "input_token_details": {"cached_tokens": 800, "web_search_requests": 2},
            "output_token_details": {"reasoning_tokens": 100},
        },
    ),
    (
        "realtime_plural_details",
        {
            "input_tokens": 900,
            "output_tokens": 250,
            "input_token_details": {"text_tokens": 700, "cached_tokens": 200, "audio_tokens": 0},
            "output_token_details": {"text_tokens": 150, "audio_tokens": 100},
        },
    ),
    (
        "missing_totals_derived",
        {"input_tokens": 410, "output_tokens": 90},
    ),
    (
        "already_chat_shaped_passthrough",
        {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    ),
    (
        "nested_cached_details",
        {
            "input_tokens": 1_000,
            "output_tokens": 200,
            "input_token_details": {"text_tokens": 400, "cached_tokens": 600},
        },
    ),
)


def responses_usage_surface() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, raw in RESPONSES_USAGE_CASES:
        transformed = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(raw)
        rows.append({"id": name, "raw": raw, "expected": transformed.model_dump(exclude_none=True)})
    return rows


CACHING_SAVINGS_CASES: tuple[tuple[str, dict[str, Any], dict[str, Any], str | None], ...] = (
    (
        "anthropic_read_discount_write_premium",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "cache_creation_input_token_cost": 3.75e-6,
            "litellm_provider": "anthropic",
        },
        usage(10_000, 500, details(cached=8_000, cache_write=1_500)),
        None,
    ),
    (
        "openai_cache_savings",
        {
            "input_cost_per_token": 1.25e-6,
            "output_cost_per_token": 1e-5,
            "cache_read_input_token_cost": 6.25e-8,
            "cache_creation_input_token_cost": 1.25e-6,
            "litellm_provider": "openai",
        },
        usage(50_000, 2_000, details(cached=45_000, cache_write=3_000)),
        None,
    ),
    (
        "unpriced_cache_no_savings",
        {"input_cost_per_token": 2e-6, "output_cost_per_token": 8e-6, "litellm_provider": "openai"},
        usage(10_000, 500, details(cached=8_000)),
        None,
    ),
    (
        "priority_tier_discount",
        {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "cache_creation_input_token_cost": 3.75e-6,
            "input_cost_per_token_priority": 6e-6,
            "litellm_provider": "anthropic",
        },
        usage(10_000, 500, details(cached=8_000, cache_write=1_500)),
        "priority",
    ),
)


def caching_savings_surface() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, model_info, payload, service_tier in CACHING_SAVINGS_CASES:
        savings = calculate_prompt_caching_savings(
            model_info=model_info,
            usage=make_usage(payload),
            custom_llm_provider=model_info.get("litellm_provider"),
            service_tier=service_tier,
        )
        rows.append(
            {
                "id": name,
                "model_info": model_info,
                "usage": payload,
                "custom_llm_provider": model_info.get("litellm_provider"),
                "service_tier": service_tier,
                "expected_savings": savings,
            }
        )
    return rows


ANTHROPIC_USAGE_CASES: tuple[tuple[str, dict[str, Any], str | None, bool], ...] = (
    (
        "plain_input_output",
        {"input_tokens": 1_000, "output_tokens": 300},
        None,
        False,
    ),
    (
        "cache_creation_and_read",
        {
            "input_tokens": 5_000,
            "output_tokens": 200,
            "cache_creation_input_tokens": 2_000,
            "cache_creation": {"ephemeral_5m_input_tokens": 1_500, "ephemeral_1h_input_tokens": 500},
            "cache_read_input_tokens": 3_000,
        },
        None,
        False,
    ),
    (
        "iterations_with_thinking_tokens",
        {
            "input_tokens": 600,
            "output_tokens": 300,
            "iterations": [
                {"input_tokens": 100, "output_tokens": 150, "output_tokens_details": {"thinking_tokens": 90}},
                {"input_tokens": 200, "output_tokens": 150, "output_tokens_details": {"thinking_tokens": 60}},
            ],
        },
        None,
        False,
    ),
    (
        "iterations_usage",
        {
            "input_tokens": 600,
            "output_tokens": 250,
            "server_tool_use": {"web_search_requests": 3},
        },
        None,
        False,
    ),
)


def anthropic_usage_surface() -> list[dict[str, Any]]:
    config = AnthropicConfig()
    rows: list[dict[str, Any]] = []
    for name, raw, reasoning, has_thinking in ANTHROPIC_USAGE_CASES:
        transformed = config.calculate_usage(raw, reasoning)
        rows.append(
            {
                "id": name,
                "raw": raw,
                "reasoning_content": reasoning,
                "response_has_thinking_block": has_thinking,
                "expected": transformed.model_dump(exclude_none=True),
            }
        )
    return rows


BILLED_RATES_CASES: tuple[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]], ...] = (
    (
        "geo_multiplier_applies",
        usage(1_000, 200),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "provider_specific_entry": {"us": 1.1},
        },
        {"usage_extra": {"inference_geo": "us"}, "custom_llm_provider": "anthropic"},
    ),
    (
        "geo_global_is_neutral",
        usage(1_000, 200),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "provider_specific_entry": {"us": 1.1},
        },
        {"usage_extra": {"inference_geo": "global"}, "custom_llm_provider": "anthropic"},
    ),
    (
        "all_multipliers_compose",
        usage(1_000, 200),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "cache_read_input_token_cost": 1e-7,
            "provider_specific_entry": {"us": 1.1},
            "regional_processing_uplift_multiplier_eu": 1.5,
            "regional_endpoint_uplift_multiplier": 1.10,
        },
        {
            "usage_extra": {"inference_geo": "us"},
            "data_residency": "eu",
            "vertex_location": "us-east1",
            "custom_llm_provider": "anthropic",
        },
    ),
    (
        "service_tier_priority_rates",
        usage(1_000, 200),
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "input_cost_per_token_priority": 3e-6,
            "output_cost_per_token_priority": 6e-6,
        },
        {"service_tier": "priority"},
    ),
    (
        "off_peak_rates_and_reasoning",
        usage(1_000, 200, details(reasoning=50)),
        {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 4e-6,
            "cache_read_input_token_cost": 2e-7,
            "cache_creation_input_token_cost": 5e-6,
            "off_peak_pricing": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "output_cost_per_reasoning_token": 1e-6,
                "hours_utc": "22:00-06:00",
            },
        },
        {"current_time": datetime(2026, 3, 12, 2, 30, tzinfo=UTC)},
    ),
)


def billed_rates_surface() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload, model_info, extra in BILLED_RATES_CASES:
        usage_obj = make_usage(payload)
        for key, value in extra.get("usage_extra", {}).items():
            setattr(usage_obj, key, value)
        model_name = f"fixture-billed/{name.replace('_', '-')}"
        register_model({model_name: model_info})
        rates = get_billed_token_rates(
            model=model_name,
            usage=usage_obj,
            custom_llm_provider=extra.get("custom_llm_provider", "openai"),
            service_tier=extra.get("service_tier"),
            data_residency=extra.get("data_residency"),
            vertex_location=extra.get("vertex_location"),
            current_time=extra.get("current_time"),
        )
        rows.append(
            {
                "id": name,
                "model": model_name,
                "usage": payload,
                "usage_extra": extra.get("usage_extra", {}),
                "model_info": model_info,
                "custom_llm_provider": extra.get("custom_llm_provider", "openai"),
                "service_tier": extra.get("service_tier"),
                "data_residency": extra.get("data_residency"),
                "vertex_location": extra.get("vertex_location"),
                "current_time": extra["current_time"].isoformat() if "current_time" in extra else None,
                "expected": {
                    "input_cost_per_token": rates.input_cost_per_token,
                    "output_cost_per_token": rates.output_cost_per_token,
                    "cache_read_input_token_cost": rates.cache_read_input_token_cost,
                    "cache_read_input_audio_token_cost": rates.cache_read_input_audio_token_cost,
                    "cache_creation_input_token_cost": rates.cache_creation_input_token_cost,
                    "cache_creation_input_token_cost_above_1hr": rates.cache_creation_input_token_cost_above_1hr,
                    "output_cost_per_reasoning_token": rates.output_cost_per_reasoning_token,
                },
            }
        )
    return rows


def main() -> None:
    revision = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    fixture = {
        "python_commit": revision,
        "generic": generic_surface(),
        "batch": batch_surface(),
        "billed_rates": billed_rates_surface(),
        "off_peak": off_peak_surface(),
        "responses_usage": responses_usage_surface(),
        "caching_savings": caching_savings_surface(),
        "anthropic_usage": anthropic_usage_surface(),
    }
    target = Path(__file__).parent / "python_fixtures.json"
    target.write_text(json.dumps(fixture, indent=1) + "\n")
    counts = {key: len(value) for key, value in fixture.items() if isinstance(value, list)}
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
