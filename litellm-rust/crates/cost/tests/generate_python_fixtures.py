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
from litellm.utils import get_model_info, register_model
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from openai.types.images_response import ImagesResponse

UTC = timezone.utc


def private_cache_attrs(usage: Usage) -> dict[str, int]:
    return {
        "_cache_read_input_tokens": usage._cache_read_input_tokens,
        "_cache_creation_input_tokens": usage._cache_creation_input_tokens,
    }


def chat_usage_payload(usage: Usage) -> dict[str, Any]:
    return {"format": "chat", **usage.model_dump(), **private_cache_attrs(usage)}


def tagged_response_payload(response: ModelResponse) -> dict[str, Any]:
    dump = response.model_dump()
    if dump.get("usage") is not None:
        dump["usage"] = {"format": "chat", **dump["usage"], **private_cache_attrs(response.usage)}
    return dump


WIRE_USAGE_CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "chat_full_projection",
        "chat",
        chat_usage_payload(
            Usage(
                prompt_tokens=1_000,
                completion_tokens=500,
                reasoning_tokens=120,
                prompt_tokens_details={
                    "cached_tokens": 800,
                    "cache_write_tokens": 300,
                    "web_search_requests": 2,
                    "cache_creation_token_details": {
                        "ephemeral_5m_input_tokens": 200,
                        "ephemeral_1h_input_tokens": 100,
                    },
                },
                completion_tokens_details={"reasoning_tokens": 120, "text_tokens": 380},
                server_tool_use={"web_search_requests": 3},
                cost=0.75,
                cache_read_input_tokens=800,
                **{
                    "inference_geo": "us",
                    "speed": "fast",
                    "citation_tokens": 1_500,
                    "server_side_tool_usage_details": {"web_search_calls": 4},
                },
            )
        ),
    ),
    (
        "chat_minimal",
        "chat",
        chat_usage_payload(Usage(prompt_tokens=10, completion_tokens=5)),
    ),
    (
        "chat_gateway_merged_input_token_keys",
        "chat",
        chat_usage_payload(Usage(prompt_tokens=100, completion_tokens=50, input_tokens=100, output_tokens=50)),
    ),
    (
        "responses_singular_details",
        "responses",
        {
            "input_tokens": 1_200,
            "output_tokens": 300,
            "input_token_details": {"cached_tokens": 800, "web_search_requests": 2},
            "output_token_details": {"reasoning_tokens": 100},
        },
    ),
    (
        "responses_plural_details",
        "responses",
        {
            "input_tokens": 900,
            "output_tokens": 250,
            "input_tokens_details": {"text_tokens": 700, "cached_tokens": 200},
            "output_tokens_details": {"text_tokens": 150, "audio_tokens": 100},
            "cost": {"total_cost": 0.25},
        },
    ),
    (
        "anthropic_full",
        "anthropic",
        {
            "input_tokens": 5_000,
            "output_tokens": 200,
            "cache_read_input_tokens": 3_000,
            "cache_creation_input_tokens": 2_000,
            "cache_creation": {"ephemeral_5m_input_tokens": 1_500, "ephemeral_1h_input_tokens": 500},
            "server_tool_use": {"web_search_requests": 2},
            "speed": "fast",
        },
    ),
    (
        "anthropic_iterations",
        "anthropic",
        {
            "input_tokens": 600,
            "output_tokens": 300,
            "iterations": [
                {
                    "input_tokens": 100,
                    "output_tokens": 150,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 10,
                    "cache_creation": {"ephemeral_5m_input_tokens": 10},
                    "output_tokens_details": {"thinking_tokens": 90},
                },
                {
                    "input_tokens": 200,
                    "output_tokens": 150,
                    "output_tokens_details": {"thinking_tokens": 60},
                },
            ],
        },
    ),
    (
        "interactions_modality_split",
        "interactions",
        {
            "total_input_tokens": 1_000,
            "total_output_tokens": 300,
            "total_tool_use_tokens": 50,
            "total_cached_tokens": 400,
            "total_reasoning_tokens": 120,
            "total_tokens": 1_470,
            "input_tokens_by_modality": [
                {"tokens": 600, "modality": "TEXT"},
                {"tokens": 50, "modality": "AUDIO"},
            ],
            "cached_tokens_by_modality": [{"tokens": 400, "modality": "TEXT"}],
            "output_tokens_by_modality": [{"tokens": 180, "modality": "TEXT"}],
            "grounding_tool_count": [{"type": "google_search", "count": 3}],
        },
    ),
    (
        "transcription_tokens",
        "transcription",
        {
            "type": "tokens",
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "input_token_details": {"text_tokens": 60, "audio_tokens": 40},
        },
    ),
    (
        "transcription_duration",
        "transcription",
        {"type": "duration", "seconds": 12.5},
    ),
)


def wire_usage_surface() -> list[dict[str, Any]]:
    return [{"id": name, "payload": {"format": declared, **payload}} for name, declared, payload in WIRE_USAGE_CASES]


def wire_response_surface() -> list[dict[str, Any]]:
    usage = Usage(
        prompt_tokens=2_000,
        completion_tokens=400,
        prompt_tokens_details={"cached_tokens": 1_500, "cache_write_tokens": 100},
        server_tool_use={"web_search_requests": 1},
        cost=0.5,
    )
    usage.inference_geo = "us"
    chat_response = ModelResponse(
        id="chatcmpl-wire",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="ok", role="assistant"),
            )
        ],
        created=1_774_000_000,
        model="synthetic-chat",
        usage=usage,
        _response_ms=1_234.5,
    )
    chat_response.ended = 1_774_000_012.5
    image_response = ImagesResponse(
        created=1_774_000_000,
        data=[
            {"url": "https://synthetic/1", "revised_prompt": None},
            {"url": "https://synthetic/2", "revised_prompt": None},
        ],
    )
    return [
        {
            "id": "chat_response_full",
            "response": tagged_response_payload(chat_response),
            "hidden_params": {
                "custom_llm_provider": "openai",
                "region_name": "us-west-2",
                "model": "synthetic-chat",
                "litellm_model_name": "synthetic-chat-alias",
                "additional_headers": {"llm_provider-x-litellm-response-cost": "0.0123"},
                "provider_specific_fields": {"traffic_type": "batch"},
            },
            "optional_params": {"service_tier": "priority", "query": ["synthetic-a", "synthetic-b"]},
        },
        {
            "id": "realtime_results",
            "response": {
                "results": [
                    {
                        "type": "response.completed",
                        "response": {
                            "usage": {
                                "format": "responses",
                                "input_tokens": 500,
                                "output_tokens": 120,
                                "input_token_details": {"cached_tokens": 100},
                            },
                            "service_tier": "priority",
                        },
                    }
                ]
            },
        },
        {
            "id": "image_response_usage",
            "response": {
                **image_response.model_dump(),
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                    "input_tokens_details": {"text_tokens": 40, "image_tokens": 60},
                    "output_tokens_details": {
                        "text_tokens": 10,
                        "image_tokens": 40,
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                    },
                },
            },
        },
    ]


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


def pinning_threshold_cases() -> list[str]:
    return [
        "input_cost_per_token_above_128k_tokens",
        "input_cost_per_token_above_200k_tokens",
        "input_cost_per_token_above_256k_tokens",
        "input_cost_per_token_above_272k_tokens",
        "input_cost_per_token_above_512k_tokens",
        "input_cost_per_token_above_1k_tokens",
        "input_cost_per_token_above_500_tokens",
        "cache_creation_input_token_cost_above_1hr_above_200k_tokens",
        "cache_creation_input_token_cost_above_1hr",
        "input_cost_per_video_per_second_above_15s_interval",
        "input_cost_per_token",
    ]


PINNING_COST_PER_UNIT_CASES: tuple[tuple[dict[str, Any], str, float | None], ...] = (
    ({"input_cost_per_token": 3e-6}, "input_cost_per_token", None),
    ({"input_cost_per_token": 3}, "input_cost_per_token", None),
    ({"input_cost_per_token": True}, "input_cost_per_token", None),
    ({"input_cost_per_token": False}, "input_cost_per_token", None),
    ({"input_cost_per_token": " 0.5 "}, "input_cost_per_token", None),
    ({"input_cost_per_token": "garbage"}, "input_cost_per_token", None),
    ({"input_cost_per_token": "garbage"}, "input_cost_per_token", 0.0),
    ({}, "input_cost_per_token", None),
    ({}, "input_cost_per_token", 0.0),
    ({"input_cost_per_token": None}, "input_cost_per_token", None),
    ({"input_cost_per_token": [1]}, "input_cost_per_token", None),
    ({"input_cost_per_token": 1.0}, "input_cost_per_token_priority", None),
    ({"input_cost_per_token": 1.0}, "input_cost_per_token_auto", None),
    ({"input_cost_per_token": 1.0}, "input_cost_per_token_flex", None),
    ({"input_cost_per_token_priority": 2.0}, "input_cost_per_token_priority", None),
    ({"input_cost_per_token_flex": 2.0}, "input_cost_per_token_priority", None),
    ({"input_cost_per_token": 1.0, "input_cost_per_token_priority": 2.0}, "input_cost_per_token_priority", None),
    ({"input_cost_per_token_ultrafast": 2.0}, "input_cost_per_token_fast", None),
)


def pinning_surface() -> dict[str, Any]:
    from litellm.litellm_core_utils.llm_cost_calc import utils as cost_utils

    thresholds: list[dict[str, Any]] = []
    for key in pinning_threshold_cases():
        try:
            thresholds.append({"key": key, "threshold": cost_utils._parse_above_token_threshold(key)})
        except Exception as error:
            thresholds.append({"key": key, "error": type(error).__name__})

    cost_per_unit: list[dict[str, Any]] = []
    for model_info, cost_key, default in PINNING_COST_PER_UNIT_CASES:
        result = cost_utils._get_cost_per_unit(model_info, cost_key, default)
        cost_per_unit.append(
            {
                "model_info": model_info,
                "cost_key": cost_key,
                "default": default,
                "result": result,
            }
        )
    from litellm.types.utils import CallTypes, LlmProviders

    return {
        "thresholds": thresholds,
        "service_tier_suffixes": list(cost_utils._SERVICE_TIER_SUFFIXES),
        "batch_tier_key": cost_utils._BATCH_TIER_KEY.pattern,
        "cost_per_unit": cost_per_unit,
        "llm_providers": [provider.value for provider in LlmProviders],
        "call_types": [call.value for call in CallTypes],
    }


def model_info_projection_surface() -> dict[str, Any]:
    entry = {
        "max_tokens": 4_096,
        "max_input_tokens": 4_096,
        "max_output_tokens": 1_024,
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_above_128k_tokens": 5e-7,
        "input_cost_per_token_above_128k_tokens_priority": 6e-7,
        "cache_read_input_token_cost": 1e-7,
        "litellm_provider": "synthetic",
        "mode": "chat",
        "supports_function_calling": True,
        "rpm": 100,
        "made_up_capability": "x",
        "input_cost_per_gadget": 0.5,
    }
    register_model({"synthetic-pinning-model": entry})
    info = get_model_info(model="synthetic-pinning-model", custom_llm_provider="synthetic")
    return {"registered": sorted(entry.keys()), "projected": sorted(info.keys())}


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
        "wire_usage": wire_usage_surface(),
        "wire_response": wire_response_surface(),
        "pinning": pinning_surface(),
        "model_info_projection": model_info_projection_surface(),
    }
    target = Path(__file__).parent / "python_fixtures.json"
    target.write_text(json.dumps(fixture, indent=1) + "\n")
    counts = {
        key: len(value)
        for key, value in fixture.items()
        if isinstance(value, (list, dict)) and key != "model_info_projection"
    }
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
