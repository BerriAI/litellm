"""Contract: what a Google call costs on Vertex AI (Claude and Gemini) and on Gemini AI Studio.

Vertex rows that carry `regional_endpoint_uplift_multiplier` (1.1 on the Claude partner rows and the newer
Gemini Flash rows) bill every token type 10 percent above list price unless the request was served from
the `global` location (`get_vertex_regional_endpoint_uplift`, `utils.py:1478`, applied at `utils.py:1772`).
The location is the deployment's `vertex_location`, and an unset location resolves to `us-central1`
(`VertexBase.get_vertex_region`, `litellm/llms/vertex_ai/vertex_llm_base.py:194`), which is regional.
AI Studio (`gemini/`) never applies the uplift: it has no location.

Gemini usage maps as follows: `promptTokensDetails` modalities split the prompt into text, audio and image
tokens, `cachedContentTokenCount` bills at the cache read rate, `thoughtsTokenCount` adds to completion and
bills at `output_cost_per_reasoning_token`, and `usageMetadata.trafficType` names the tier that was served
(`ON_DEMAND_PRIORITY` to priority, `FLEX`, `BATCH` and `ON_DEMAND_FLEX` to flex,
`cost_calculator.py:1230`), which wins over the requested tier.

Existing coverage this file does not repeat: AI Studio 200k tiers, cache reads and grounding fees on
gemini-2.5 in `providers/test_dec1_gemini_usage_cost.py`, and synthetic rows in
`cost_calculation/cases/vertex_ai.py` and `gemini.py`. Rows used (repo root
`model_prices_and_context_window.json`): `vertex_ai/claude-sonnet-4-5@20250929` (line 54076),
`vertex_ai/claude-haiku-4-5@20251001` (53070), `vertex_ai/gemini-3.8-flash` (31155), `gemini-2.5-pro`
(30687), `gemini/gemini-2.5-flash` (31949), `gemini/gemini-2.5-pro` (32495).
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.pricing.dec1_costacc_support import anthropic_message, anthropic_usage, assert_billed, json_reply
from integration.providers.dec1_gemini_support import gemini
from integration.providers.dec1_vertex_support import gemini_json, usage_metadata, vertex
from pydantic import JsonValue

VERTEX_SONNET: Final = "claude-sonnet-4-5@20250929"
VERTEX_HAIKU: Final = "claude-haiku-4-5@20251001"
VERTEX_FLASH: Final = "gemini-3.8-flash"
FLASH: Final = "gemini-2.5-flash"


def _claude(usage: dict[str, JsonValue], model: str = "claude-sonnet-4-5-20250929") -> Reply:
    """A Vertex rawPredict answer. `model` is what Anthropic echoes; the cost calculator may price the
    served model over the requested one, so it always names the model under test."""
    return Reply(body=json.dumps(anthropic_message(usage, model=model)).encode())


def _gemini(prompt: int, candidates: int, *, model: str, **extra: JsonValue) -> Reply:
    """A generateContent answer whose `modelVersion` names the model under test (the served model is priced)."""
    return json_reply({**gemini_json(usage=usage_metadata(prompt, candidates, **extra)), "modelVersion": model})


def _always(reply: Reply):  # noqa: ANN202
    def respond(_: Request) -> Reply:
        return reply

    return respond


# ---------------------------------------------------------------- Vertex Claude


def test_vertex_claude_on_a_regional_location_bills_the_uplifted_price(gateway: Gateway) -> None:
    """`vertex_ai/claude-sonnet-4-5@20250929` (line 54076: 3e-06 in, 1.5e-05 out, uplift 1.1) at
    `vertex_location: us-east5` reports 1000 input and 500 output: `(0.003 + 0.0075) * 1.1 = 0.01155`."""
    with vertex(
        gateway, _always(_claude(anthropic_usage(1000, 500))), backend=VERTEX_SONNET, vertex_location="us-east5"
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01155)


def test_vertex_claude_on_the_global_location_bills_list_price(gateway: Gateway) -> None:
    """The same call with `vertex_location: global` gets no uplift: `0.0105`."""
    with vertex(
        gateway, _always(_claude(anthropic_usage(1000, 500))), backend=VERTEX_SONNET, vertex_location="global"
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0105)


def test_vertex_claude_above_200k_on_global_bills_the_long_context_rates(gateway: Gateway) -> None:
    """250000 input and 500 output on global: `250000 * 6e-06 + 500 * 2.25e-05 = 1.51125` (line 54076
    `input_cost_per_token_above_200k_tokens` and `output_cost_per_token_above_200k_tokens`)."""
    with vertex(
        gateway, _always(_claude(anthropic_usage(250000, 500))), backend=VERTEX_SONNET, vertex_location="global"
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 1.51125)


def test_vertex_claude_above_200k_on_a_region_stacks_the_uplift_on_the_long_context_rates(gateway: Gateway) -> None:
    """The same long call at us-east5 costs `1.51125 * 1.1 = 1.662375`: the uplift multiplies the already
    tiered cost."""
    with vertex(
        gateway, _always(_claude(anthropic_usage(250000, 500))), backend=VERTEX_SONNET, vertex_location="us-east5"
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 1.662375)


def test_vertex_claude_one_hour_cache_writes_bill_at_the_one_hour_rate(gateway: Gateway) -> None:
    """On global, 1000 input, 2000 one hour writes and 500 output cost
    `0.003 + 2000 * 6e-06 + 0.0075 = 0.0225` (line 54076 `cache_creation_input_token_cost_above_1hr`)."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_5m=0, write_1h=2000)
    with vertex(gateway, _always(_claude(usage)), backend=VERTEX_SONNET, vertex_location="global") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0225)


def test_vertex_claude_cache_reads_on_a_region_are_uplifted(gateway: Gateway) -> None:
    """`vertex_ai/claude-haiku-4-5@20251001` (line 53070: 1e-06 in, 1e-07 cache read, 5e-06 out, uplift 1.1)
    at us-east5 with 1000 input, 4000 cache read and 500 output: `(0.001 + 0.0004 + 0.0025) * 1.1 = 0.00429`."""
    usage: Final = anthropic_usage(1000, 500, cache_read=4000)
    with vertex(
        gateway, _always(_claude(usage, "claude-haiku-4-5-20251001")), backend=VERTEX_HAIKU, vertex_location="us-east5"
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00429)


def test_vertex_claude_web_search_adds_the_per_search_fee(gateway: Gateway) -> None:
    """On global, sonnet 4.5 reports `server_tool_use.web_search_requests: 2`; the row's
    `search_context_size_medium` is 0.01, so the cost is `0.0105 + 2 * 0.01 = 0.0305`.
    UNVERIFIED: assumes the Anthropic web search fee path also runs for `vertex_ai` Claude and is not uplifted."""
    usage: Final = anthropic_usage(1000, 500, server_tool_use={"web_search_requests": 2})
    with vertex(gateway, _always(_claude(usage)), backend=VERTEX_SONNET, vertex_location="global") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0305)


def test_vertex_claude_custom_prices_are_not_uplifted(gateway: Gateway) -> None:
    """A us-east5 sonnet 4.5 deployment sets `input_cost_per_token: 1e-05`, `output_cost_per_token: 2e-05`.
    1000 input and 500 output cost `0.02`: the deployment's own entry does not inherit
    `regional_endpoint_uplift_multiplier` (it is not a pricing field `registry.py:427` copies).
    UNVERIFIED: a port that merges the row would bill `0.022`."""
    with vertex(
        gateway,
        _always(_claude(anthropic_usage(1000, 500))),
        backend=VERTEX_SONNET,
        vertex_location="us-east5",
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.02)


# ---------------------------------------------------------------- Vertex Gemini


def test_vertex_gemini_on_a_region_bills_the_uplifted_price(gateway: Gateway) -> None:
    """`vertex_ai/gemini-3.8-flash` (line 31155: 7.5e-07 in, 3.75e-06 out, uplift 1.1) at us-central1 with
    1000 prompt and 100 candidate tokens: `(7.5e-04 + 3.75e-04) * 1.1 = 0.0012375`."""
    with vertex(
        gateway, _always(_gemini(1000, 100, model=VERTEX_FLASH)), backend=VERTEX_FLASH, vertex_location="us-central1"
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0012375)


def test_vertex_gemini_on_global_bills_list_price(gateway: Gateway) -> None:
    """The same call at `vertex_location: global`: `0.001125`."""
    with vertex(
        gateway, _always(_gemini(1000, 100, model=VERTEX_FLASH)), backend=VERTEX_FLASH, vertex_location="global"
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.001125)


def test_vertex_gemini_with_no_location_is_priced_as_us_central1(gateway: Gateway) -> None:
    """A gemini-3.8-flash deployment with no `vertex_location` is served from `us-central1`
    (`vertex_llm_base.py:194`), a regional endpoint, so it is uplifted: `0.0012375`.
    UNVERIFIED: `resolve_vertex_location_for_cost` (`litellm_logging.py:601`) sees no location on the proxy
    path and falls back to the same default."""
    with vertex(
        gateway, _always(_gemini(1000, 100, model=VERTEX_FLASH)), backend=VERTEX_FLASH, vertex_location=None
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0012375)


def test_vertex_gemini_without_an_uplift_key_bills_list_price_on_a_region(gateway: Gateway) -> None:
    """`gemini-2.5-pro` (line 30687) has no `regional_endpoint_uplift_multiplier`, so at us-central1 1000 prompt
    and 100 candidates cost `1000 * 1.25e-06 + 100 * 1e-05 = 0.00225`."""
    with vertex(
        gateway,
        _always(_gemini(1000, 100, model="gemini-2.5-pro")),
        backend="gemini-2.5-pro",
        vertex_location="us-central1",
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00225)


def test_vertex_gemini_priority_traffic_bills_priority_rates_and_the_uplift(gateway: Gateway) -> None:
    """gemini-3.8-flash at us-central1 answers `usageMetadata.trafficType: ON_DEMAND_PRIORITY`. Priority rates
    1.35e-06 / 6.75e-06 (line 31155) then the uplift: `(0.00135 + 0.000675) * 1.1 = 0.0022275`."""
    reply: Final = _gemini(1000, 100, model=VERTEX_FLASH, trafficType="ON_DEMAND_PRIORITY")
    with vertex(gateway, _always(reply), backend=VERTEX_FLASH, vertex_location="us-central1") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0022275)


def test_vertex_gemini_on_demand_flex_traffic_bills_flex_rates(gateway: Gateway) -> None:
    """Vertex reports flex capacity as `trafficType: ON_DEMAND_FLEX` (not `FLEX`). On global that is
    `1000 * 3.75e-07 + 100 * 1.875e-06 = 0.0005625` (line 31155)."""
    reply: Final = _gemini(1000, 100, model=VERTEX_FLASH, trafficType="ON_DEMAND_FLEX")
    with vertex(gateway, _always(reply), backend=VERTEX_FLASH, vertex_location="global") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0005625)


def test_vertex_gemini_thinking_tokens_bill_at_the_reasoning_rate(gateway: Gateway) -> None:
    """On global, gemini-3.8-flash reports 1000 prompt, 100 candidates and `thoughtsTokenCount: 400` (total
    1500). Completion becomes 500 tokens at 3.75e-06 (`output_cost_per_reasoning_token` equals output here):
    `1000 * 7.5e-07 + 500 * 3.75e-06 = 0.002625`."""
    reply: Final = _gemini(1000, 100, model=VERTEX_FLASH, thoughtsTokenCount=400, totalTokenCount=1500)
    with vertex(gateway, _always(reply), backend=VERTEX_FLASH, vertex_location="global") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.002625)


def test_vertex_gemini_cached_content_bills_at_the_cache_read_rate(gateway: Gateway) -> None:
    """On global, 1000 prompt tokens of which `cachedContentTokenCount: 800`, and 100 candidates:
    `200 * 7.5e-07 + 800 * 7.5e-08 + 100 * 3.75e-06 = 0.000585` (line 31155)."""
    reply: Final = _gemini(1000, 100, model=VERTEX_FLASH, cachedContentTokenCount=800)
    with vertex(gateway, _always(reply), backend=VERTEX_FLASH, vertex_location="global") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.000585)


# ---------------------------------------------------------------- Gemini AI Studio


def test_studio_audio_prompt_tokens_bill_at_the_audio_rate(gateway: Gateway) -> None:
    """`gemini/gemini-2.5-flash` (line 31949: text 3e-07, audio 1e-06, out 2.5e-06) reports 1000 prompt tokens
    split `TEXT: 800`, `AUDIO: 200` in `promptTokensDetails`, and 100 candidates:
    `800 * 3e-07 + 200 * 1e-06 + 100 * 2.5e-06 = 0.00069`."""
    details: Final[list[JsonValue]] = [
        {"modality": "TEXT", "tokenCount": 800},
        {"modality": "AUDIO", "tokenCount": 200},
    ]
    with gemini(gateway, _always(_gemini(1000, 100, model=FLASH, promptTokensDetails=details)), backend=FLASH) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00069)


def test_studio_image_prompt_tokens_without_an_image_rate_bill_as_text(gateway: Gateway) -> None:
    """The same model with `TEXT: 800`, `IMAGE: 200`: the row has no `input_cost_per_image_token`, so image
    tokens bill at the text rate (`_modality_token_cost`, `utils.py:1258`): `1000 * 3e-07 + 100 * 2.5e-06 = 0.00055`."""
    details: Final[list[JsonValue]] = [
        {"modality": "TEXT", "tokenCount": 800},
        {"modality": "IMAGE", "tokenCount": 200},
    ]
    with gemini(gateway, _always(_gemini(1000, 100, model=FLASH, promptTokensDetails=details)), backend=FLASH) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00055)


def test_studio_priority_traffic_bills_priority_rates(gateway: Gateway) -> None:
    """`trafficType: ON_DEMAND_PRIORITY` on gemini/gemini-2.5-flash: `1000 * 5.4e-07 + 100 * 4.5e-06 = 0.00099`."""
    with gemini(
        gateway, _always(_gemini(1000, 100, model=FLASH, trafficType="ON_DEMAND_PRIORITY")), backend=FLASH
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.00099)


def test_studio_flex_traffic_bills_flex_rates(gateway: Gateway) -> None:
    """`trafficType: FLEX` on gemini/gemini-2.5-flash: `1000 * 1.5e-07 + 100 * 1.25e-06 = 0.000275`."""
    with gemini(gateway, _always(_gemini(1000, 100, model=FLASH, trafficType="FLEX")), backend=FLASH) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.000275)


def test_studio_on_demand_traffic_overrides_a_requested_priority(gateway: Gateway) -> None:
    """The caller asks for `service_tier: priority` but Google reports `trafficType: ON_DEMAND`, which maps to
    base pricing (`_served_service_tier` returns `default`), so the bill is the standard
    `1000 * 3e-07 + 100 * 2.5e-06 = 0.00055`."""
    with gemini(gateway, _always(_gemini(1000, 100, model=FLASH, trafficType="ON_DEMAND")), backend=FLASH) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.00055)


def test_studio_priority_above_200k_bills_the_priority_long_context_rates(gateway: Gateway) -> None:
    """`gemini/gemini-2.5-pro` (line 32495) with 250000 prompt, 100 candidates and priority traffic:
    `input_cost_per_token_above_200k_tokens_priority` 4.5e-06 and `output_cost_per_token_above_200k_tokens_priority`
    2.7e-05, so `250000 * 4.5e-06 + 100 * 2.7e-05 = 1.1277`."""
    reply: Final = _gemini(250000, 100, model="gemini-2.5-pro", trafficType="ON_DEMAND_PRIORITY")
    with gemini(gateway, _always(reply), backend="gemini-2.5-pro") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 1.1277)


def test_studio_audio_cache_reads_bill_at_the_audio_cache_rate(gateway: Gateway) -> None:
    """gemini/gemini-2.5-flash reports 1000 prompt tokens (`TEXT: 600`, `AUDIO: 400`), of which 500 cached
    (`cacheTokensDetails`: `TEXT: 300`, `AUDIO: 200`), and 100 candidates. Uncached text 300 * 3e-07, uncached
    audio 200 * 1e-06, cached text 300 * 3e-08, cached audio 200 * 1e-07 (`cache_read_input_audio_token_cost`),
    output 100 * 2.5e-06: `0.00009 + 0.0002 + 0.000009 + 0.00002 + 0.00025 = 0.000569`.
    UNVERIFIED: assumes `cacheTokensDetails` modalities reach `cached_audio_tokens`."""
    reply: Final = _gemini(
        1000,
        100,
        model=FLASH,
        cachedContentTokenCount=500,
        promptTokensDetails=[{"modality": "TEXT", "tokenCount": 600}, {"modality": "AUDIO", "tokenCount": 400}],
        cacheTokensDetails=[{"modality": "TEXT", "tokenCount": 300}, {"modality": "AUDIO", "tokenCount": 200}],
    )
    with gemini(gateway, _always(reply), backend=FLASH) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.000569)


def test_studio_never_applies_the_vertex_regional_uplift(gateway: Gateway) -> None:
    """`gemini/gemini-2.5-flash` with 1000 prompt and 100 candidates bills `0.00055` even though Vertex rows
    carry an uplift: `resolve_vertex_location_for_cost` returns None for any provider that is not `vertex_ai`."""
    with gemini(gateway, _always(_gemini(1000, 100, model=FLASH)), backend=FLASH) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00055)


@pytest.mark.parametrize(
    ("input_price", "output_price", "expected"),
    [
        pytest.param(1e-05, 2e-05, 0.012, id="both-flat-prices"),
        pytest.param(1e-05, None, 0.01, id="input-only-bills-no-output"),
    ],
)
def test_studio_deployment_prices_win_over_the_row(
    gateway: Gateway, input_price: float, output_price: float | None, expected: float
) -> None:
    """A gemini/gemini-2.5-flash deployment priced `input_cost_per_token: 1e-05` and, in one cell,
    `output_cost_per_token: 2e-05`; 1000 prompt and 100 candidates cost `0.01 + 0.002 = 0.012`, or with only
    an input price `0.01`. Pinned current behaviour for the input-only cell: the deployment's cost entry is its own
    custom prices plus only the inherited off-peak, cache, service tier and tiered output rates
    (`deployment_model_cost_payload`, litellm/router_rust/surface/registry.py:427-455), never the row's plain
    `output_cost_per_token`, so the 100 candidates bill 0 (CI run 44161684479)."""
    prices: Final[dict[str, JsonValue]] = {"input_cost_per_token": input_price}
    if output_price is not None:
        prices["output_cost_per_token"] = output_price
    with gemini(gateway, _always(_gemini(1000, 100, model=FLASH)), backend=FLASH, **prices) as rig:
        assert_billed(gateway, rig.model, rig.chat(), expected)
