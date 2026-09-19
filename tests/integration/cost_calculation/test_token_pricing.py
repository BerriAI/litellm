"""Token pricing coverage for the integration scripted-provider cost shard."""

from __future__ import annotations

import uuid
from typing import Final, cast

import pytest
from pydantic import JsonValue

from integration._support.client import JSON_OBJECT, Gateway
from integration._support.scripted_provider import ScriptedUsage, Wire
from integration.cost_calculation.conftest import (
    approx_equal,
    assert_total_is_sum_of_components,
    poll_cost_row,
    register_scenario_deployment,
)
from integration.cost_calculation.cost_matrix import (
    AUDIO_INPUT_DATA_URL,
    FRONTIER_MODELS,
    IMAGE_INPUT_DATA_URL,
    SERVICE_TIER_REQUEST_WIRES,
    VIDEO_INPUT_DATA_URL,
    Case,
    FrontierModel,
    cases_for,
    matrix_data_errors,
    recount_cost,
)

if _data_errors := matrix_data_errors():
    raise ValueError("\n".join(_data_errors))

def _case_id(param: tuple[FrontierModel, Case]) -> str:
    model, case = param
    return f"{model.map_key.replace('/', '-')}-{case.name}"


_MATRIX: Final = tuple(
    pytest.param(
        (model, case),
        marks=pytest.mark.covers(
            "quota_management.spend_tracking.scripted_wire.logs_cost"
            if case.family == "transport"
            else "quota_management.spend_tracking.cost_matrix.logs_cost"
        ),
        id=_case_id((model, case)),
    )
    for model in FRONTIER_MODELS
    for case in cases_for(model)
)
_CACHE_WIRES: Final = frozenset({"anthropic_messages", "bedrock_converse"})
_WEB_SEARCH_OPTION_WIRES: Final = frozenset({"openai_chat", "azure_chat", "openai_responses"})


def _cache_control(usage: ScriptedUsage, wire: Wire) -> dict[str, JsonValue] | None:
    if wire not in _CACHE_WIRES:
        return None
    if not (usage.cache_read_tokens or usage.cache_write_5m_tokens or usage.cache_write_1h_tokens):
        return None
    return {"type": "ephemeral", **({"ttl": "1h"} if usage.cache_write_1h_tokens else {})}


def _chat_body(model: FrontierModel, case: Case, model_name: str, marker: str) -> dict[str, JsonValue]:
    usage: Final = case.usage_for(model.map_key)
    user_parts: Final = [
        {"type": "text", "text": f"{marker} summarize the attached material in one line and name the city weather"},
        *(
            [{"type": "image_url", "image_url": {"url": IMAGE_INPUT_DATA_URL, "detail": "high"}}]
            if case.image_input
            else []
        ),
        *(
            [{"type": "input_audio", "input_audio": {"data": AUDIO_INPUT_DATA_URL.split(",", 1)[1], "format": "wav"}}]
            if case.audio_input
            else []
        ),
        *(
            [{"type": "file", "file": {"file_data": VIDEO_INPUT_DATA_URL, "format": "mp4"}}]
            if case.video_input
            else []
        ),
    ]
    tools: Final[list[JsonValue]] = [
        *(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather and a short forecast for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "City name"},
                                "days": {"type": "integer", "description": "Forecast horizon in days"},
                                "units": {"type": "string", "enum": ["metric", "imperial"]},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ]
            if case.tool_call
            else []
        ),
        *(
            [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
            if case.web_search is not None and model.wire == "anthropic_messages"
            else []
        ),
        *(
            [{"googleSearch": {}}]
            if case.web_search is not None and model.wire in ("gemini_generate", "vertex_generate")
            else []
        ),
        *([{"googleMaps": {}}] if case.google_maps else []),
        *([{"type": "file_search", "vector_store_ids": ["vs_cost_calc_fixture"]}] if case.file_search else []),
    ]
    cache_control: Final = _cache_control(usage, model.wire)
    message: Final = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                **({"cache_control": cache_control} if cache_control else {}),
            }
        ],
    }
    return cast(dict[str, JsonValue], {
        "model": model_name,
        "messages": [message, {"role": "user", "content": user_parts}],
        "stream": case.stream,
        **({"stream_options": {"include_usage": True}} if case.stream else {}),
        **(
            {"service_tier": case.service_tier}
            if case.service_tier is not None and model.wire in SERVICE_TIER_REQUEST_WIRES
            else {}
        ),
        **({"reasoning_effort": "medium"} if case.reasoning else {}),
        **(
            {"modalities": ["text", "audio"] if case.audio_output else ["text"]}
            if case.audio_input or case.audio_output
            else {}
        ),
        **({"audio": {"voice": "alloy", "format": "pcm16"}} if case.audio_output else {}),
        **(
            {"web_search_options": {"search_context_size": case.web_search}}
            if case.web_search is not None and model.wire in _WEB_SEARCH_OPTION_WIRES
            else {}
        ),
        **({"tools": tools} if tools else {}),
        **({"tool_choice": "auto"} if case.tool_call and model.wire != "bedrock_converse" else {}),
        "allowed_openai_params": [
            name
            for name, sent in (
                ("tool_choice", case.tool_call and model.wire != "bedrock_converse"),
                ("modalities", case.audio_input or case.audio_output),
                ("audio", case.audio_output),
                ("web_search_options", case.web_search is not None),
                ("reasoning_effort", case.reasoning),
            )
            if sent
        ],
    })


def _assert_stream_has_no_error(response_text: str) -> None:
    for line in response_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        parsed = JSON_OBJECT.validate_json(payload)
        assert "error" not in parsed, f"stream carried an error event: {parsed}"


@pytest.mark.parametrize("model_case", _MATRIX)
def test_scripted_usage_bills_at_map_rates(
    gateway: Gateway,
    model_case: tuple[FrontierModel, Case],
) -> None:
    model, case = model_case
    marker: Final = uuid.uuid4().hex[:12]
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        model_name: Final = register_scenario_deployment(scenario, model, case, marker)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(model, case, model_name, marker),
            key=key,
        )
        assert response.is_success, (
            f"{model.map_key}/{case.name}: proxy returned {response.status_code}: {response.text[:400]}"
        )
        if case.stream:
            _assert_stream_has_no_error(response.text)
        row: Final = poll_cost_row(key)
        context: Final = f"{model.map_key}/{case.name}"
        if not case.exact_spend:
            assert row.prompt_tokens is not None and row.prompt_tokens > 0, (
                f"{context}: no-usage stream counted no input tokens: prompt_tokens={row.prompt_tokens}"
            )
            assert row.completion_tokens is not None and row.completion_tokens > 0, (
                f"{context}: no-usage stream counted no output tokens: completion_tokens={row.completion_tokens}"
            )
            if case.image_input:
                assert row.prompt_tokens < 4000, (
                    f"{context}: image data URL looks tokenized as text: prompt_tokens={row.prompt_tokens}"
                )
            recount: Final = recount_cost(model, case, row.prompt_tokens, row.completion_tokens)
            assert row.spend is not None and approx_equal(
                row.spend, recount
            ), f"{context}: no-usage stream spend {row.spend} != recount {recount} at map rates"
            assert_total_is_sum_of_components(row, context)
            return
        golden: Final = case.expected_for(model)
        if not case.stream:
            header: Final = cast(str | None, response.headers.get("x-litellm-response-cost"))
            assert header is not None and approx_equal(float(header), golden.spend), (
                f"{context}: x-litellm-response-cost {header} != golden {golden.spend}"
            )
        assert row.spend is not None and approx_equal(row.spend, golden.spend), (
            f"{context}: spend {row.spend} != golden {golden.spend} "
            f"(breakdown {row.breakdown.model_dump()})"
        )
        breakdown: Final = row.breakdown
        assert breakdown.input_cost is not None and approx_equal(breakdown.input_cost, golden.input_cost), (
            f"{context}: gross input_cost {breakdown.input_cost} != golden {golden.input_cost}; "
            "cached/written tokens billed at the input rate"
        )
        assert breakdown.output_cost is not None and approx_equal(breakdown.output_cost, golden.output_cost), (
            f"{context}: output_cost {breakdown.output_cost} != golden {golden.output_cost}"
        )
        assert row.prompt_tokens == golden.prompt_tokens, (
            f"{context}: prompt_tokens {row.prompt_tokens} != golden {golden.prompt_tokens}"
        )
        assert row.completion_tokens == golden.completion_tokens, (
            f"{context}: completion_tokens {row.completion_tokens} != golden {golden.completion_tokens}"
        )
        assert_total_is_sum_of_components(row, context)
