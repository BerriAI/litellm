"""Spend for `gemini/` calls under Google's service tiers, Maps grounding and audio tokens.

Service tiers. The tier a call is billed at is resolved in `completion_cost`
(`litellm/cost_calculator.py:1914`): the tier Google says it SERVED wins when it names a priced tier or
says base pricing, otherwise the tier the caller REQUESTED decides (`_resolve_billable_service_tier`,
`:1287`). Google reports the served tier two ways and the proxy reads both: the response header
`x-gemini-service-tier` (copied to the response's `service_tier`, with `standard` renamed `default`,
`vertex_and_google_ai_studio_gemini.py:3426`) and `usageMetadata.trafficType`
(`ON_DEMAND_PRIORITY` is priority, `FLEX`/`BATCH`/`ON_DEMAND_FLEX` are flex, `ON_DEMAND` is base,
`cost_calculator.py:1230`). A priced tier selects the `_priority` / `_flex` cost map keys, including the
200k long context keys (`input_cost_per_token_above_200k_tokens_priority`,
`llm_cost_calc/utils.py:787`). Rates used here, all from `model_prices_and_context_window.json`:
gemini-2.5-flash input 3e-07 / output 2.5e-06, priority 5.4e-07 / 4.5e-06, flex 1.5e-07 / 1.25e-06,
priority cache read 5.4e-08; gemini-2.5-pro above 200k priority 4.5e-06 / 2.7e-05.

Maps grounding. A `groundingMetadata` item with `maps` chunks or a `googleMapsWidgetContextToken` counts as
Maps grounding (`grounding_requests.py`): its distinct queries are Maps requests (at least 1), unless the
same item also has `web` chunks, in which case the queries count as web searches. Gemini 2.x bills one
`google_maps_grounding_cost_per_query` ($0.025) per grounded prompt, Gemini 3 ($0.014) per query
(`cost_per_google_maps_grounding_request`, `litellm/llms/gemini/cost_calculator.py:85`).

Each test registers its own model group and reads its single `LiteLLM_SpendLogs` row by `model_group`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_gemini2_support import TTS_MODEL, one_row, spend_of
from integration.providers.dec1_gemini_support import (
    GEMINI_3_FLASH,
    PRO,
    at,
    gemini,
    gemini_json,
    json_of,
    reply_json,
    usage_metadata,
)
from pydantic import JsonValue

_FLASH_BASE: Final = 1000 * 3e-07 + 100 * 2.5e-06
_FLASH_PRIORITY: Final = 1000 * 5.4e-07 + 100 * 4.5e-06
_FLASH_FLEX: Final = 1000 * 1.5e-07 + 100 * 1.25e-06
_SMALL_FLASH_TOKENS: Final = 11 * 3e-07 + 7 * 2.5e-06
_SMALL_GEMINI_3_TOKENS: Final = 11 * 5e-07 + 7 * 3e-06


def _answer(
    usage: dict[str, JsonValue] | None = None,
    headers: dict[str, str] | None = None,
    **candidate_fields: JsonValue,
) -> Callable[[Request], Reply]:
    """A responder that always answers with one Gemini JSON body (usage defaults to 1,000 in, 100 out)."""
    body: Final = gemini_json(usage=usage if usage is not None else usage_metadata(1000, 100), **candidate_fields)  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough

    def respond(_request: Request) -> Reply:
        return reply_json(body, headers=headers)

    return respond


def _maps_grounding(queries: list[JsonValue], *, web: bool = False) -> dict[str, JsonValue]:
    chunks: Final[list[JsonValue]] = [{"maps": {"uri": "https://maps.example/place", "title": "Cafe"}}]
    if web:
        chunks.append({"web": {"uri": "https://example.com/cafe", "title": "example.com"}})
    return {"webSearchQueries": queries, "groundingChunks": chunks}


# ---------------------------------------------------------------- service tiers


def test_a_requested_priority_tier_is_billed_at_priority_rates(gateway: Gateway) -> None:
    """`service_tier: "priority"` with no served tier signal from Google bills 1,000 prompt and 100 output
    tokens on gemini-2.5-flash at `1000 * 5.4e-07 + 100 * 4.5e-06 = 0.00099`, not the base 0.00055."""
    with gemini(gateway, _answer()) as rig:
        assert rig.chat(service_tier="priority").status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_PRIORITY)


def test_an_on_demand_priority_traffic_type_is_billed_at_priority_rates(gateway: Gateway) -> None:
    """A caller that asked for no tier, answered with `usageMetadata.trafficType: ON_DEMAND_PRIORITY`, is
    billed at priority rates (0.00099): the served tier wins (`_served_service_tier`)."""
    usage: Final = usage_metadata(1000, 100, trafficType="ON_DEMAND_PRIORITY")
    with gemini(gateway, _answer(usage)) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_PRIORITY)


def test_a_priority_service_tier_header_is_reported_and_billed(gateway: Gateway) -> None:
    """Google's response header `x-gemini-service-tier: priority` sets the caller's response field
    `service_tier: "priority"` and the row is billed at priority rates (0.00099)."""
    with gemini(gateway, _answer(headers={"x-gemini-service-tier": "priority"})) as rig:
        response: Final = rig.chat()
        assert json_of(response).get("service_tier") == "priority", response.text
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_PRIORITY)


def test_a_standard_header_on_a_priority_request_is_billed_at_base_rates(gateway: Gateway) -> None:
    """A caller asks for `service_tier: "priority"` but Google answers with `x-gemini-service-tier:
    standard` (it fell back to standard capacity): the response says `service_tier: "default"` and the row
    is billed at base rates `1000 * 3e-07 + 100 * 2.5e-06 = 0.00055`, because a served base tier overrides
    the request (`_BASE_PRICING_SERVICE_TIERS`)."""
    with gemini(gateway, _answer(headers={"x-gemini-service-tier": "standard"})) as rig:
        response: Final = rig.chat(service_tier="priority")
        assert json_of(response).get("service_tier") == "default", response.text
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_BASE)


def test_an_on_demand_traffic_type_on_a_priority_request_is_billed_at_base_rates(gateway: Gateway) -> None:
    """A priority request answered with `trafficType: ON_DEMAND` is billed at base rates (0.00055):
    `ON_DEMAND` maps to no tier and `_served_service_tier` then reports `default`, which wins over the
    request."""
    usage: Final = usage_metadata(1000, 100, trafficType="ON_DEMAND")
    with gemini(gateway, _answer(usage)) as rig:
        assert rig.chat(service_tier="priority").status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_BASE)


@pytest.mark.parametrize("traffic_type", ["FLEX", "BATCH", "ON_DEMAND_FLEX"])
def test_flex_traffic_types_are_billed_at_flex_rates(gateway: Gateway, traffic_type: str) -> None:
    """`trafficType` FLEX, BATCH and ON_DEMAND_FLEX all select the `_flex` keys:
    `1000 * 1.5e-07 + 100 * 1.25e-06 = 0.000275` on gemini-2.5-flash."""
    usage: Final = usage_metadata(1000, 100, trafficType=traffic_type)
    with gemini(gateway, _answer(usage)) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_FLEX)


def test_a_requested_flex_tier_is_billed_at_flex_rates(gateway: Gateway) -> None:
    """`service_tier: "flex"` (sent to Google as `serviceTier: flex`) with no served signal bills 0.000275."""
    with gemini(gateway, _answer()) as rig:
        assert rig.chat(service_tier="flex").status_code == 200
        assert rig.peer.one_body()["serviceTier"] == "flex"
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(_FLASH_FLEX)


def test_priority_cached_tokens_use_the_priority_cache_read_rate(gateway: Gateway) -> None:
    """A priority call with 1,000 prompt tokens of which 600 are cached and 100 output tokens costs
    `400 * 5.4e-07 + 600 * 5.4e-08 + 100 * 4.5e-06 = 0.0006984` (`cache_read_input_token_cost_priority`)."""
    usage: Final = usage_metadata(1000, 100, cachedContentTokenCount=600, trafficType="ON_DEMAND_PRIORITY")
    with gemini(gateway, _answer(usage)) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(400 * 5.4e-07 + 600 * 5.4e-08 + 100 * 4.5e-06)


def test_a_priority_prompt_above_200k_uses_the_priority_long_context_rates(gateway: Gateway) -> None:
    """gemini-2.5-pro at 200,001 prompt tokens served as `ON_DEMAND_PRIORITY` bills every token at the
    priority long context rates: `200001 * 4.5e-06 + 100 * 2.7e-05 = 0.9027045`, not the base long context
    rates (0.5015025) and not the short priority rates."""
    usage: Final = usage_metadata(200001, 100, trafficType="ON_DEMAND_PRIORITY")
    with gemini(gateway, _answer(usage, model=PRO), backend=PRO) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(200001 * 4.5e-06 + 100 * 2.7e-05)


def test_the_traffic_type_is_not_leaked_into_the_response_body(gateway: Gateway) -> None:
    """`trafficType` is kept in the response's hidden params for pricing; the caller's JSON has no
    `traffic_type` key at the top level and `usage` has no `trafficType`
    (`hidden_params["provider_specific_fields"]["traffic_type"]`). UNVERIFIED: whether a later proxy
    layer copies hidden provider fields into the body."""
    usage: Final = usage_metadata(1000, 100, trafficType="ON_DEMAND_PRIORITY")
    with gemini(gateway, _answer(usage)) as rig:
        body: Final = json_of(rig.chat())
        assert "traffic_type" not in body
        assert "trafficType" not in str(body.get("usage"))


# ---------------------------------------------------------------- Maps grounding


def test_maps_grounding_on_gemini_2_adds_one_fee_per_prompt(gateway: Gateway) -> None:
    """A gemini-2.5-flash answer grounded with Maps chunks and two distinct queries reports
    `usage.prompt_tokens_details.google_maps_grounding_requests: 2` and is billed one per prompt fee:
    `0.025 + 11 * 3e-07 + 7 * 2.5e-06 = 0.0250208`."""
    answer: Final = _answer(usage_metadata(), groundingMetadata=_maps_grounding(["cafe near me", "open now"]))
    with gemini(gateway, answer) as rig:
        response: Final = rig.chat()
        assert at(json_of(response), "usage", "prompt_tokens_details", "google_maps_grounding_requests") == 2
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(0.025 + _SMALL_FLASH_TOKENS)


def test_maps_grounding_on_gemini_3_adds_a_fee_per_query(gateway: Gateway) -> None:
    """On gemini-3-flash-preview (`web_search_billing_unit: per_query`) the same two Maps queries cost
    `2 * 0.014 + 11 * 5e-07 + 7 * 3e-06 = 0.0280265`."""
    answer: Final = _answer(
        usage_metadata(), model=GEMINI_3_FLASH, groundingMetadata=_maps_grounding(["cafe near me", "open now"])
    )
    with gemini(gateway, answer, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat().status_code == 200
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(2 * 0.014 + _SMALL_GEMINI_3_TOKENS)


def test_a_widget_token_alone_counts_as_one_maps_request(gateway: Gateway) -> None:
    """Grounding metadata with only `googleMapsWidgetContextToken` (no chunks, no queries) is floored at one
    Maps request on gemini-3-flash-preview: `0.014 + 11 * 5e-07 + 7 * 3e-06`."""
    answer: Final = _answer(
        usage_metadata(), model=GEMINI_3_FLASH, groundingMetadata={"googleMapsWidgetContextToken": "widget-dec1"}
    )
    with gemini(gateway, answer, backend=GEMINI_3_FLASH) as rig:
        response: Final = rig.chat()
        assert at(json_of(response), "usage", "prompt_tokens_details", "google_maps_grounding_requests") == 1
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(0.014 + _SMALL_GEMINI_3_TOKENS)


def test_queries_next_to_web_chunks_are_billed_as_search_plus_one_maps_request(gateway: Gateway) -> None:
    """One grounding item with both `maps` and `web` chunks and one query on gemini-3-flash-preview: the query
    counts as a web search (`web_search_requests: 1`, $0.014) and Maps is floored at one request ($0.014),
    so the row is `0.014 + 0.014 + 11 * 5e-07 + 7 * 3e-06`."""
    answer: Final = _answer(
        usage_metadata(), model=GEMINI_3_FLASH, groundingMetadata=_maps_grounding(["cafe reviews"], web=True)
    )
    with gemini(gateway, answer, backend=GEMINI_3_FLASH) as rig:
        response: Final = rig.chat()
        details: Final = at(json_of(response), "usage", "prompt_tokens_details")
        assert (at(details, "web_search_requests"), at(details, "google_maps_grounding_requests")) == (1, 1)
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(0.028 + _SMALL_GEMINI_3_TOKENS)


# ---------------------------------------------------------------- audio tokens


def test_audio_prompt_tokens_are_billed_at_the_audio_input_rate(gateway: Gateway) -> None:
    """A gemini-2.5-flash prompt of 1,000 tokens reported as 600 TEXT and 400 AUDIO in `promptTokensDetails`
    with 100 output tokens costs `600 * 3e-07 + 400 * 1e-06 + 100 * 2.5e-06 = 0.00083`
    (`input_cost_per_audio_token`), and the response reports `prompt_tokens_details.audio_tokens: 400`.
    UNVERIFIED: the split of text and audio prompt tokens inside `generic_cost_per_token` was read, not run."""
    usage: Final = usage_metadata(
        1000,
        100,
        promptTokensDetails=[{"modality": "TEXT", "tokenCount": 600}, {"modality": "AUDIO", "tokenCount": 400}],
    )
    with gemini(gateway, _answer(usage)) as rig:
        response: Final = rig.chat()
        assert at(json_of(response), "usage", "prompt_tokens_details", "audio_tokens") == 400
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(600 * 3e-07 + 400 * 1e-06 + 100 * 2.5e-06)


def test_audio_output_tokens_are_reported_as_completion_audio_tokens(gateway: Gateway) -> None:
    """A TTS answer whose `candidatesTokensDetails` is `[{"modality": "AUDIO", "tokenCount": 100}]` reports
    `completion_tokens_details.audio_tokens: 100` and `text_tokens: 0`, and costs
    `10 * 5e-07 + 100 * 1e-05 = 0.001005` on gemini-2.5-flash-preview-tts (`_calculate_usage`)."""
    usage: Final = usage_metadata(10, 100, candidatesTokensDetails=[{"modality": "AUDIO", "tokenCount": 100}])
    audio_part: Final[list[JsonValue]] = [{"inlineData": {"mimeType": "audio/pcm", "data": "AAAA"}}]
    with gemini(gateway, _answer(usage, parts=audio_part, model=TTS_MODEL), backend=TTS_MODEL) as rig:
        response: Final = rig.chat()
        details: Final = at(json_of(response), "usage", "completion_tokens_details")
        assert (at(details, "audio_tokens"), at(details, "text_tokens")) == (100, 0)
        assert spend_of(one_row(gateway, rig.model)) == pytest.approx(10 * 5e-07 + 100 * 1e-05)
