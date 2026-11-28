"""Contract: what a direct Anthropic call costs, priced from the packaged cost map rows.

A deployment `anthropic/<model>` with no prices of its own is billed from the cost map row of `<model>`.
The Anthropic usage block is turned into a cost by `litellm/llms/anthropic/cost_calculation.py:22`
(`generic_cost_per_token`, then the `provider_specific_entry` multipliers): uncached input at
`input_cost_per_token`, `cache_read_input_tokens` at `cache_read_input_token_cost`, cache writes split by
`usage.cache_creation` into 5 minute writes at `cache_creation_input_token_cost` and 1 hour writes at
`cache_creation_input_token_cost_above_1hr` (`litellm/litellm_core_utils/llm_cost_calc/utils.py:1017`),
output at `output_cost_per_token`; `usage.inference_geo` other than `global` multiplies everything by the
row's `provider_specific_entry[<geo>]` and `usage.speed: fast` by `provider_specific_entry.fast`
(`utils.py:1505`); `server_tool_use.web_search_requests` adds `search_context_size_medium` per search.

Deployment prices win over the row, but a deployment that sets only flat token prices keeps the row's
cache prices (`litellm/router_rust/surface/registry.py:296` `_inherited_cache_pricing`).

The cost matrix in `cost_calculation/cases/anthropic.py` prices synthetic rows served by the scripted cost
map; this file pins the real packaged rows, which is what production bills from. Rows used (repo root
`model_prices_and_context_window.json`): `claude-haiku-4-5` (line 16555), `claude-sonnet-4-5`
(16582), `claude-opus-4-6` (16890), `claude-opus-5` (17198).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.pricing.dec1_costacc_support import (
    anthropic,
    anthropic_reply,
    anthropic_sse,
    anthropic_usage,
    assert_billed,
    assert_row_billed,
    marker,
)

HAIKU: Final = "claude-haiku-4-5"
SONNET: Final = "claude-sonnet-4-5"
OPUS_46: Final = "claude-opus-4-6"
OPUS_5: Final = "claude-opus-5"


def test_plain_input_and_output_bill_at_the_packaged_haiku_rates(gateway: Gateway) -> None:
    """Anthropic reports 1000 input and 500 output tokens on `claude-haiku-4-5` (line 16555: 1e-06 in,
    5e-06 out). The cost header and the spend row must both be `1000 * 1e-06 + 500 * 5e-06 = 0.0035`."""
    with anthropic(gateway, anthropic_reply(anthropic_usage(1000, 500), model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0035)


def test_cache_read_tokens_bill_at_the_cache_read_rate_on_top_of_uncached_input(gateway: Gateway) -> None:
    """Anthropic reports 1000 uncached input, 4000 `cache_read_input_tokens` and 500 output on haiku 4.5
    (line 16555: cache read 1e-07). Anthropic's `input_tokens` excludes cached tokens, so the bill is
    `1000 * 1e-06 + 4000 * 1e-07 + 500 * 5e-06 = 0.0039`; a rewrite that subtracts the cache from
    `input_tokens` (the OpenAI convention) would bill 0.0035 or less."""
    usage: Final = anthropic_usage(1000, 500, cache_read=4000)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0039)


def test_five_minute_cache_writes_bill_at_the_base_write_rate(gateway: Gateway) -> None:
    """2000 `cache_creation_input_tokens` reported as `ephemeral_5m_input_tokens: 2000` on haiku 4.5
    (line 16555: write 1.25e-06). Cost is `1000 * 1e-06 + 2000 * 1.25e-06 + 500 * 5e-06 = 0.006`."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_5m=2000, write_1h=0)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.006)


def test_one_hour_cache_writes_bill_at_the_above_1hr_write_rate(gateway: Gateway) -> None:
    """The same 2000 written tokens reported as `ephemeral_1h_input_tokens: 2000` bill at
    `cache_creation_input_token_cost_above_1hr` 2e-06 (line 16555), so the cost is
    `1000 * 1e-06 + 2000 * 2e-06 + 500 * 5e-06 = 0.0075`, 60 percent more write cost than the 5 minute case."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_5m=0, write_1h=2000)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0075)


def test_a_mixed_five_minute_and_one_hour_write_bills_each_part_at_its_own_rate(gateway: Gateway) -> None:
    """2000 written tokens split 1000 five minute and 1000 one hour on haiku 4.5 cost
    `1000 * 1e-06 + 1000 * 1.25e-06 + 1000 * 2e-06 + 500 * 5e-06 = 0.00675` (`utils.py:1038`)."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_5m=1000, write_1h=1000)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00675)


def test_a_cache_write_without_a_ttl_breakdown_bills_at_the_five_minute_rate(gateway: Gateway) -> None:
    """Anthropic reports `cache_creation_input_tokens: 2000` with no `cache_creation` object (older API
    versions). With no breakdown every write is a base write (`utils.py:1027`): cost `0.006`, the same
    as the explicit 5 minute case."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.006)


def test_write_tokens_the_breakdown_does_not_cover_bill_at_the_five_minute_rate(gateway: Gateway) -> None:
    """2000 written tokens with a breakdown naming only `ephemeral_1h_input_tokens: 500`. The 1500 tokens
    no lifetime covers bill at the base write rate (`utils.py:1037`):
    `1000 * 1e-06 + 1500 * 1.25e-06 + 500 * 2e-06 + 500 * 5e-06 = 0.006375`."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_1h=500)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.006375)


def test_us_inference_geo_multiplies_every_token_type_by_the_row_uplift(gateway: Gateway) -> None:
    """`claude-opus-4-6` (line 16890: 5e-06 in, 2.5e-05 out, `provider_specific_entry.us: 1.1`) answers with
    `usage.inference_geo: "us"`. The cost is `(1000 * 5e-06 + 500 * 2.5e-05) * 1.1 = 0.01925`
    (`utils.py:1505`): the US only inference premium must survive a cost calculator port."""
    usage: Final = anthropic_usage(1000, 500, inference_geo="us")
    with anthropic(gateway, anthropic_reply(usage, model=OPUS_46), backend=OPUS_46) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01925)


def test_global_inference_geo_bills_the_list_price(gateway: Gateway) -> None:
    """The same opus 4.6 call answered with `inference_geo: "global"` gets no uplift: `0.0175`. `global`
    and `not_available` are explicitly excluded even though they are not keys of the entry."""
    usage: Final = anthropic_usage(1000, 500, inference_geo="global")
    with anthropic(gateway, anthropic_reply(usage, model=OPUS_46), backend=OPUS_46) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0175)


def test_us_inference_geo_on_a_model_without_a_geo_entry_bills_the_list_price(gateway: Gateway) -> None:
    """`claude-haiku-4-5` (line 16555) has no `provider_specific_entry`, so `inference_geo: "us"` changes
    nothing: `1000 * 1e-06 + 500 * 5e-06 = 0.0035`."""
    usage: Final = anthropic_usage(1000, 500, inference_geo="us")
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0035)


def test_the_geo_uplift_also_applies_to_cache_reads(gateway: Gateway) -> None:
    """On opus 4.6 with `inference_geo: "us"`, 1000 input, 4000 cache read (5e-07) and 500 output cost
    `(0.005 + 4000 * 5e-07 + 0.0125) * 1.1 = 0.02145`: the uplift is applied to the whole prompt cost,
    not only to uncached input."""
    usage: Final = anthropic_usage(1000, 500, cache_read=4000, inference_geo="us")
    with anthropic(gateway, anthropic_reply(usage, model=OPUS_46), backend=OPUS_46) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.02145)


def test_fast_speed_doubles_the_cost_on_a_model_with_a_fast_multiplier(gateway: Gateway) -> None:
    """`claude-opus-5` (line 17198: 5e-06 in, 2.5e-05 out, `provider_specific_entry.fast: 2.0`) answers
    with `usage.speed: "fast"`. The cost is `(1000 * 5e-06 + 500 * 2.5e-05) * 2.0 = 0.035`
    (`get_provider_specific_speed_multiplier`, `utils.py:1522`)."""
    usage: Final = anthropic_usage(1000, 500, speed="fast")
    with anthropic(gateway, anthropic_reply(usage, model=OPUS_5), backend=OPUS_5) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.035)


def test_fast_speed_and_us_geo_multiply_together(gateway: Gateway) -> None:
    """Opus 5 answered with both `speed: "fast"` and `inference_geo: "us"` costs
    `0.0175 * 2.0 * 1.1 = 0.0385`: the two multipliers stack (`cost_calculation.py:59-65`)."""
    usage: Final = anthropic_usage(1000, 500, speed="fast", inference_geo="us")
    with anthropic(gateway, anthropic_reply(usage, model=OPUS_5), backend=OPUS_5) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0385)


def test_standard_speed_on_a_fast_capable_model_bills_the_list_price(gateway: Gateway) -> None:
    """Opus 5 answered with `usage.speed: "standard"` is not multiplied: `0.0175`."""
    usage: Final = anthropic_usage(1000, 500, speed="standard")
    with anthropic(gateway, anthropic_reply(usage, model=OPUS_5), backend=OPUS_5) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0175)


def test_priority_service_tier_on_a_row_without_priority_rates_bills_the_standard_rate(gateway: Gateway) -> None:
    """Anthropic answers `usage.service_tier: "priority"` on haiku 4.5, whose row (line 16555) has no
    `*_priority` keys. `get_cost_per_unit` falls back from `input_cost_per_token_priority` to the base key
    (`utils.py:941`), so the cost stays `0.0035`; a port must not bill zero for a missing tier key."""
    usage: Final = anthropic_usage(1000, 500, service_tier="priority")
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0035)


def test_web_search_requests_add_the_per_search_fee(gateway: Gateway) -> None:
    """`claude-sonnet-4-5` (line 16582: 3e-06 in, 1.5e-05 out, `search_context_size_medium: 0.01`) reports
    `server_tool_use.web_search_requests: 3`. The cost is `1000 * 3e-06 + 500 * 1.5e-05 + 3 * 0.01 = 0.0405`
    (`get_cost_for_anthropic_web_search`, `cost_calculation.py:94`)."""
    usage: Final = anthropic_usage(1000, 500, server_tool_use={"web_search_requests": 3})
    with anthropic(gateway, anthropic_reply(usage, model=SONNET), backend=SONNET) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0405)


def test_zero_web_search_requests_add_no_fee(gateway: Gateway) -> None:
    """The same sonnet 4.5 call reporting `web_search_requests: 0` costs only its tokens, `0.0105`."""
    usage: Final = anthropic_usage(1000, 500, server_tool_use={"web_search_requests": 0})
    with anthropic(gateway, anthropic_reply(usage, model=SONNET), backend=SONNET) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0105)


def test_a_prompt_above_200k_on_direct_sonnet_4_5_bills_the_flat_rate(gateway: Gateway) -> None:
    """The packaged direct Anthropic row `claude-sonnet-4-5` (line 16582) carries no
    `input_cost_per_token_above_200k_tokens` (unlike the Bedrock and Vertex rows), so 250000 input and 500
    output tokens bill flat: `250000 * 3e-06 + 500 * 1.5e-05 = 0.7575`. Pinned because a port that
    borrows long context rates from another provider's row would double the bill."""
    usage: Final = anthropic_usage(250000, 500)
    with anthropic(gateway, anthropic_reply(usage, model=SONNET), backend=SONNET) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.7575)


def test_thinking_tokens_are_billed_inside_output_tokens(gateway: Gateway) -> None:
    """Haiku 4.5 answers a thinking block and a text block with `output_tokens: 500` (Anthropic counts
    thinking inside output). With no `output_cost_per_reasoning_token` on the row, the bill is the plain
    `0.0035`: thinking is neither free nor charged twice."""
    content: Final = [
        {"type": "thinking", "thinking": "work it out", "signature": "c3ludGhldGlj"},
        {"type": "text", "text": "priced"},
    ]
    usage: Final = anthropic_usage(1000, 500)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU, content=content), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0035)


def test_a_streamed_chat_bills_the_message_start_and_message_delta_usage(gateway: Gateway) -> None:
    """A streamed haiku 4.5 call whose `message_start` reports 1000 input and 4000 cache read tokens and
    whose `message_delta` reports 500 output tokens writes one row with spend `0.0039`, the same as the
    non streamed cache read case."""
    usage: Final = anthropic_usage(1000, 500, cache_read=4000)
    with anthropic(gateway, anthropic_sse(usage, model=HAIKU), backend=HAIKU) as rig:
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": rig.model, "stream": True, "messages": [{"role": "user", "content": marker()}]},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200
            assert "".join(response.iter_text())
        assert_row_billed(gateway, rig.model, 0.0039)


def test_messages_route_bills_the_same_as_chat(gateway: Gateway) -> None:
    """A native `/v1/messages` call on haiku 4.5 reporting 1000 input, 2000 one hour writes and 500 output
    must cost `1000 * 1e-06 + 2000 * 2e-06 + 500 * 5e-06 = 0.0075`, the chat price: `/v1/messages` is one of
    the first routes moving to Rust."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_5m=0, write_1h=2000)
    with anthropic(gateway, anthropic_reply(usage, model=HAIKU), backend=HAIKU) as rig:
        assert_billed(gateway, rig.model, rig.messages(), 0.0075)


def test_messages_route_applies_the_us_geo_uplift(gateway: Gateway) -> None:
    """A `/v1/messages` call on opus 4.6 answered with `inference_geo: "us"` costs `0.01925`, the same as
    the chat route."""
    usage: Final = anthropic_usage(1000, 500, inference_geo="us")
    with anthropic(gateway, anthropic_reply(usage, model=OPUS_46), backend=OPUS_46) as rig:
        assert_billed(gateway, rig.model, rig.messages(), 0.01925)


def test_deployment_flat_prices_win_and_cache_reads_keep_the_packaged_cache_rate(gateway: Gateway) -> None:
    """A haiku 4.5 deployment sets only `input_cost_per_token: 1e-05` and `output_cost_per_token: 2e-05`.
    1000 input, 4000 cache read and 500 output cost `1000 * 1e-05 + 4000 * 1e-07 + 500 * 2e-05 = 0.0204`:
    the flat prices win but the cache read rate is inherited from the row (`registry.py:296`), not billed
    at the custom input rate (which would be 0.06)."""
    usage: Final = anthropic_usage(1000, 500, cache_read=4000)
    with anthropic(
        gateway,
        anthropic_reply(usage, model=HAIKU),
        backend=HAIKU,
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0204)


def test_a_deployment_cache_read_price_wins_over_the_packaged_one(gateway: Gateway) -> None:
    """The same deployment also sets `cache_read_input_token_cost: 1e-06`: the cost becomes
    `1000 * 1e-05 + 4000 * 1e-06 + 500 * 2e-05 = 0.024`."""
    usage: Final = anthropic_usage(1000, 500, cache_read=4000)
    with anthropic(
        gateway,
        anthropic_reply(usage, model=HAIKU),
        backend=HAIKU,
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
        cache_read_input_token_cost=1e-06,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.024)


def test_a_deployment_write_price_without_a_one_hour_price_keeps_the_packaged_one_hour_rate(
    gateway: Gateway,
) -> None:
    """A haiku 4.5 deployment sets input 1e-05, output 2e-05 and `cache_creation_input_token_cost: 3e-05`
    but no `cache_creation_input_token_cost_above_1hr`. A write of 1000 five minute and 1000 one hour
    tokens costs `0.01 + 1000 * 3e-05 + 1000 * 2e-06 + 0.01 = 0.052`: the one hour rate is inherited from
    the row (`_CACHE_PRICING_FIELDS`, `registry.py:164`), so the 1h write is cheaper than the custom 5m
    write. Pinned as today's behaviour; a port that derives 1h from the custom 5m rate changes the bill."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_5m=1000, write_1h=1000)
    with anthropic(
        gateway,
        anthropic_reply(usage, model=HAIKU),
        backend=HAIKU,
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
        cache_creation_input_token_cost=3e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.052)


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        pytest.param(HAIKU, 0.0035, id="haiku-4-5"),
        pytest.param(SONNET, 0.0105, id="sonnet-4-5"),
        pytest.param(OPUS_46, 0.0175, id="opus-4-6"),
        pytest.param(OPUS_5, 0.0175, id="opus-5"),
    ],
)
def test_each_packaged_claude_row_bills_its_own_list_price(gateway: Gateway, backend: str, expected: float) -> None:
    """1000 input and 500 output tokens with no geo, speed or tier. Each model bills its own row: haiku 4.5
    `0.0035` (line 16555), sonnet 4.5 `0.0105` (16582), opus 4.6 `0.0175` (16890), opus 5 `0.0175` (17198)."""
    with anthropic(gateway, anthropic_reply(anthropic_usage(1000, 500), model=backend), backend=backend) as rig:
        assert_billed(gateway, rig.model, rig.chat(), expected)
