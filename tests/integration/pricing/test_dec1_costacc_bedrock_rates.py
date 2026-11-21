"""Contract: what an AWS Bedrock call costs, priced from the packaged cost map rows.

A Bedrock deployment with no prices of its own is billed from the row of the exact model id it calls,
cross-region inference profile prefix included: `us.`, `eu.`, `jp.` and `au.` Claude profiles carry the
10 percent regional premium in their own rows, `global.` carries list price and `us-gov.` its own GovCloud
price. Converse usage (`inputTokens`, `cacheReadInputTokens`, `cacheWriteInputTokens`, `cacheDetails`)
becomes a LiteLLM usage block in `litellm/llms/bedrock/chat/converse_transformation.py:2665`
(`transform_usage`), where `prompt_tokens = inputTokens + cacheRead + cacheWrite`; the long context tier
is picked from that total (`utils.py:802`, strictly greater than 200000). A cache write is split into 5
minute and 1 hour writes only when `cacheDetails` accounts for every written token (`_parse_cache_details`,
line 2589). The tier Bedrock reports in `serviceTier.type` selects the `_priority` / `_flex` rates
(`cost_calculator.py:1287`, the served tier wins over the requested one).

Existing coverage this file does not repeat: synthetic rows in `cost_calculation/cases/bedrock_converse.py`,
plain per-profile list prices and the `ca.` haiku profile in `providers/test_dec1_bedrock_regions_credentials_wire.py`.
Rows used (repo root `model_prices_and_context_window.json`): `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
(line 50921), `eu.` (26126), `jp.` (41221), `au.` (3813), `global.` (35353), `us-gov.` (50965),
`anthropic.claude-sonnet-4-5-20250929-v1:0` (3520), `amazon.nova-pro-v1:0` (873), `us.amazon.nova-pro-v1:0` (50745).
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.pricing.dec1_costacc_support import (
    anthropic_message,
    anthropic_usage,
    assert_billed,
    json_reply,
)
from integration.providers.dec1_bedrock_support import bedrock, converse_json, path_of
from pydantic import JsonValue

US_SONNET: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
BASE_SONNET: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
NOVA_PRO: Final = "amazon.nova-pro-v1:0"
INVOKE_ECHO: Final = "claude-sonnet-4-5-20250929"
"""The `model` real InvokeModel echoes for a Claude profile; pricing must still follow the profile id."""


def _converse(usage: dict[str, JsonValue], **extra: JsonValue) -> Reply:
    return json_reply({**converse_json(usage=usage), **extra})


def _usage(input_tokens: int, output_tokens: int, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": input_tokens + output_tokens, **extra}


def _converse_or_invoke(converse_usage: dict[str, JsonValue], anthropic: dict[str, JsonValue], model: str):  # noqa: ANN202
    """Answer Converse paths with a Converse document and InvokeModel paths with an Anthropic message."""

    def respond(request: Request) -> Reply:
        if path_of(request).endswith(("/converse", "/converse-stream")):
            return _converse(converse_usage)
        return Reply(body=json.dumps(anthropic_message(anthropic, model=model)).encode())

    return respond


def test_us_profile_bills_the_regional_row_for_plain_tokens(gateway: Gateway) -> None:
    """`bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` (line 50921: 3.3e-06 in, 1.65e-05 out) on
    Converse reports 1000 input and 500 output: `1000 * 3.3e-06 + 500 * 1.65e-05 = 0.01155`."""
    with bedrock(gateway, lambda _: _converse(_usage(1000, 500)), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01155)


def test_a_prompt_above_200k_bills_every_token_at_the_long_context_rates(gateway: Gateway) -> None:
    """The us profile reports 250000 input and 500 output tokens. Above 200k every input token moves to
    `input_cost_per_token_above_200k_tokens` 6.6e-06 and output to 2.475e-05 (line 50921):
    `250000 * 6.6e-06 + 500 * 2.475e-05 = 1.662375`, not only the tokens past the threshold."""
    with bedrock(gateway, lambda _: _converse(_usage(250000, 500)), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 1.662375)


def test_exactly_200k_prompt_tokens_stay_in_the_base_tier(gateway: Gateway) -> None:
    """200000 input tokens do not cross the threshold (`prompt_tokens > threshold`, `utils.py:320`):
    `200000 * 3.3e-06 + 500 * 1.65e-05 = 0.66825`."""
    with bedrock(gateway, lambda _: _converse(_usage(200000, 500)), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.66825)


def test_cached_tokens_count_toward_the_200k_threshold(gateway: Gateway) -> None:
    """Converse reports `inputTokens: 1000`, `cacheReadInputTokens: 250000`, `outputTokens: 500`. The
    prompt is 251000 tokens, so the long context rates apply to all of it, cache reads included
    (`cache_read_input_token_cost_above_200k_tokens` 6.6e-07):
    `1000 * 6.6e-06 + 250000 * 6.6e-07 + 500 * 2.475e-05 = 0.183975`."""
    usage: Final = _usage(1000, 500, cacheReadInputTokens=250000)
    with bedrock(gateway, lambda _: _converse(usage), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.183975)


def test_cache_reads_below_200k_bill_at_the_regional_cache_rate(gateway: Gateway) -> None:
    """1000 input, 100000 cache read and 500 output stay below the threshold:
    `1000 * 3.3e-06 + 100000 * 3.3e-07 + 500 * 1.65e-05 = 0.04455`."""
    usage: Final = _usage(1000, 500, cacheReadInputTokens=100000)
    with bedrock(gateway, lambda _: _converse(usage), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.04455)


def test_a_five_minute_cache_write_bills_at_the_regional_write_rate(gateway: Gateway) -> None:
    """2000 `cacheWriteInputTokens` with `cacheDetails: [{ttl: 5m, inputTokens: 2000}]`:
    `1000 * 3.3e-06 + 2000 * 4.125e-06 + 500 * 1.65e-05 = 0.0198` (line 50921)."""
    usage: Final = _usage(1000, 500, cacheWriteInputTokens=2000, cacheDetails=[{"ttl": "5m", "inputTokens": 2000}])
    with bedrock(gateway, lambda _: _converse(usage), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0198)


def test_a_one_hour_cache_write_bills_at_the_regional_one_hour_rate(gateway: Gateway) -> None:
    """The same 2000 written tokens reported with `ttl: 1h` bill at `cache_creation_input_token_cost_above_1hr`
    6.6e-06: `0.0033 + 2000 * 6.6e-06 + 0.00825 = 0.02475`."""
    usage: Final = _usage(1000, 500, cacheWriteInputTokens=2000, cacheDetails=[{"ttl": "1h", "inputTokens": 2000}])
    with bedrock(gateway, lambda _: _converse(usage), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.02475)


def test_cache_details_that_do_not_cover_every_write_bill_all_writes_at_the_five_minute_rate(gateway: Gateway) -> None:
    """`cacheWriteInputTokens: 2000` with `cacheDetails` naming only 500 one hour tokens. A partial split is
    discarded (`_parse_cache_details` returns None, line 2601), so all 2000 bill at 4.125e-06: `0.0198`."""
    usage: Final = _usage(1000, 500, cacheWriteInputTokens=2000, cacheDetails=[{"ttl": "1h", "inputTokens": 500}])
    with bedrock(gateway, lambda _: _converse(usage), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0198)


def test_a_one_hour_write_above_200k_uses_the_long_context_one_hour_rate(gateway: Gateway) -> None:
    """250000 input, 2000 one hour writes and 500 output on the us profile. The one hour write rate above
    200k is `cache_creation_input_token_cost_above_1hr_above_200k_tokens` 1.32e-05 (line 50921):
    `250000 * 6.6e-06 + 2000 * 1.32e-05 + 500 * 2.475e-05 = 1.688775`."""
    usage: Final = _usage(250000, 500, cacheWriteInputTokens=2000, cacheDetails=[{"ttl": "1h", "inputTokens": 2000}])
    with bedrock(gateway, lambda _: _converse(usage), model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 1.688775)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        pytest.param("us", 1.662375, id="us-line-50921"),
        pytest.param("eu", 1.662375, id="eu-line-26126"),
        pytest.param("jp", 1.662375, id="jp-line-41221"),
        pytest.param("au", 1.662375, id="au-line-3813"),
        pytest.param("global", 1.51125, id="global-line-35353"),
        pytest.param("us-gov", 1.8135, id="us-gov-line-50965"),
    ],
)
def test_each_cross_region_profile_bills_its_own_long_context_row(gateway: Gateway, profile: str, expected: float) -> None:
    """250000 input and 500 output tokens on `<profile>.anthropic.claude-sonnet-4-5-20250929-v1:0`. Regional
    profiles bill `250000 * 6.6e-06 + 500 * 2.475e-05 = 1.662375`, `global.` bills list long context
    `250000 * 6e-06 + 500 * 2.25e-05 = 1.51125` and `us-gov.` `250000 * 7.2e-06 + 500 * 2.7e-05 = 1.8135`."""
    model: Final = f"bedrock/{profile}.anthropic.claude-sonnet-4-5-20250929-v1:0"
    with bedrock(gateway, lambda _: _converse(_usage(250000, 500)), model=model) as rig:
        assert_billed(gateway, rig.model, rig.chat(), expected)


def test_the_ca_profile_without_a_row_bills_the_base_model_long_context_rates(gateway: Gateway) -> None:
    """`ca.` is a recognized cross-region prefix since #52152 but `ca.anthropic.claude-sonnet-4-5-20250929-v1:0`
    has no row, so the base row `anthropic.claude-sonnet-4-5-20250929-v1:0` (line 3520) prices it: 250000
    input and 500 output cost `250000 * 6e-06 + 500 * 2.25e-05 = 1.51125`.
    UNVERIFIED: assumes the prefix strip falls back to the base row as it does for the haiku 5.5 profile."""
    model: Final = "bedrock/ca.anthropic.claude-sonnet-4-5-20250929-v1:0"
    with bedrock(gateway, lambda _: _converse(_usage(250000, 500)), model=model) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 1.51125)


def test_invoke_route_bills_the_profile_row_from_anthropic_usage(gateway: Gateway) -> None:
    """`bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0` answers an Anthropic message with 1000
    input, 2000 one hour writes and 500 output. The us profile rates apply on InvokeModel too:
    `1000 * 3.3e-06 + 2000 * 6.6e-06 + 500 * 1.65e-05 = 0.02475`.
    UNVERIFIED: the Invoke cost lookup resolves the `bedrock_converse` profile row by model id."""
    usage: Final = anthropic_usage(1000, 500, cache_write=2000, write_5m=0, write_1h=2000)
    respond: Final = _converse_or_invoke(_usage(1000, 500), usage, INVOKE_ECHO)
    with bedrock(gateway, respond, model=f"bedrock/invoke/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.02475)


def test_messages_route_keeps_the_cross_region_price(gateway: Gateway) -> None:
    """A `/v1/messages` call to the us sonnet 4.5 profile reporting 1000 input and 500 output costs the
    regional `0.01155`, not the base `0.0105` (#51479 preserved Bedrock cross-region pricing for
    Messages). The peer answers whichever Bedrock API the proxy picks."""
    respond: Final = _converse_or_invoke(_usage(1000, 500), anthropic_usage(1000, 500), INVOKE_ECHO)
    with bedrock(gateway, respond, model=f"bedrock/{US_SONNET}") as rig:
        assert_billed(gateway, rig.model, rig.messages(messages=[{"role": "user", "content": "price"}]), 0.01155)


def test_nova_served_priority_bills_the_priority_rates(gateway: Gateway) -> None:
    """`amazon.nova-pro-v1:0` (line 873: priority 1.4e-06 in, 5.6e-06 out) answers with
    `serviceTier: {"type": "priority"}`: `1000 * 1.4e-06 + 500 * 5.6e-06 = 0.0042`."""
    reply: Final = _converse(_usage(1000, 500), serviceTier={"type": "priority"})
    with bedrock(gateway, lambda _: reply, model=f"bedrock/{NOVA_PRO}") as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.0042)


def test_nova_served_flex_bills_the_flex_rates(gateway: Gateway) -> None:
    """Nova Pro answering `serviceTier: {"type": "flex"}` bills `1000 * 4e-07 + 500 * 1.6e-06 = 0.0012`
    (line 873, half the standard price)."""
    reply: Final = _converse(_usage(1000, 500), serviceTier={"type": "flex"})
    with bedrock(gateway, lambda _: reply, model=f"bedrock/{NOVA_PRO}") as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="flex"), 0.0012)


def test_nova_without_a_tier_bills_the_standard_rates(gateway: Gateway) -> None:
    """Nova Pro with no tier requested or served: `1000 * 8e-07 + 500 * 3.2e-06 = 0.0024`."""
    with bedrock(gateway, lambda _: _converse(_usage(1000, 500)), model=f"bedrock/{NOVA_PRO}") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0024)


def test_a_requested_tier_the_answer_does_not_echo_still_bills_the_requested_tier(gateway: Gateway) -> None:
    """The caller asks for `service_tier: "priority"` and Bedrock's answer has no `serviceTier`. With no
    served tier the requested one decides (`_resolve_billable_service_tier`, `cost_calculator.py:1294`):
    `0.0042`."""
    with bedrock(gateway, lambda _: _converse(_usage(1000, 500)), model=f"bedrock/{NOVA_PRO}") as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.0042)


def test_a_served_default_tier_overrides_a_requested_priority(gateway: Gateway) -> None:
    """The caller asks for priority but Bedrock answers `serviceTier: {"type": "default"}`. `default` means
    base pricing, so the bill is the standard `0.0024`: the provider's word wins over the request."""
    reply: Final = _converse(_usage(1000, 500), serviceTier={"type": "default"})
    with bedrock(gateway, lambda _: reply, model=f"bedrock/{NOVA_PRO}") as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.0024)


def test_priority_cache_reads_bill_at_the_priority_cache_rate(gateway: Gateway) -> None:
    """`us.amazon.nova-pro-v1:0` (line 50745) served at priority with 1000 input, 600 cache read and 500 output:
    `1000 * 1.4e-06 + 600 * 3.5e-07 + 500 * 5.6e-06 = 0.00441`."""
    reply: Final = _converse(_usage(1000, 500, cacheReadInputTokens=600), serviceTier={"type": "priority"})
    with bedrock(gateway, lambda _: reply, model="bedrock/us.amazon.nova-pro-v1:0") as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.00441)


def test_deployment_prices_override_the_profile_row(gateway: Gateway) -> None:
    """A us sonnet 4.5 profile deployment sets `input_cost_per_token: 1e-05` and `output_cost_per_token: 2e-05`;
    1000 input and 500 output cost `0.02`, not the row's 0.01155."""
    with bedrock(
        gateway,
        lambda _: _converse(_usage(1000, 500)),
        model=f"bedrock/{US_SONNET}",
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.02)


def test_deployment_flat_prices_apply_to_a_long_prompt_without_the_row_long_context_rate(gateway: Gateway) -> None:
    """The same custom priced deployment receives 250000 input and 500 output tokens. The deployment entry
    inherits only cache prices from the row (`registry.py:296`), not `input_cost_per_token_above_200k_tokens`,
    so the flat custom rate applies: `250000 * 1e-05 + 500 * 2e-05 = 2.51`.
    UNVERIFIED: read from the registration path; a port that merges the whole row would bill the row's
    long context rates instead."""
    with bedrock(
        gateway,
        lambda _: _converse(_usage(250000, 500)),
        model=f"bedrock/{US_SONNET}",
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 2.51)


def test_deployment_prices_keep_the_profile_cache_read_rate(gateway: Gateway) -> None:
    """Custom input 1e-05 and output 2e-05 on the us profile with 1000 input, 4000 cache read and 500 output:
    `1000 * 1e-05 + 4000 * 3.3e-07 + 500 * 2e-05 = 0.02132`; cache reads keep the row's 3.3e-07."""
    usage: Final = _usage(1000, 500, cacheReadInputTokens=4000)
    with bedrock(
        gateway,
        lambda _: _converse(usage),
        model=f"bedrock/{US_SONNET}",
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.02132)
