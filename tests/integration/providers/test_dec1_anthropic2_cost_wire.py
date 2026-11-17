"""Contract: newer Anthropic usage fields are priced the same on chat and on the Rust-native `/v1/messages` route.

Anthropic reports usage that changes the bill: `server_tool_use.web_search_requests` (a per search fee from
`search_context_cost_per_query.search_context_size_medium`, `get_cost_for_anthropic_web_search` in
`litellm/llms/anthropic/cost_calculation.py`), `server_tool_use.web_fetch_requests` (no fee), `service_tier`
(selects the `*_priority` rate keys through `get_service_tier_cost_key`), the 1h cache write breakdown
(`cache_creation_input_token_cost_above_1hr`) and long prompts above 200k tokens, which the 1M context window beta
makes possible (`*_above_200k_tokens` keys; the threshold compares `usage.prompt_tokens`, cache tokens included,
with a strict `>` for Anthropic, `llm_cost_calc/utils.py:319`).

The packaged direct Anthropic rows carry neither long context nor priority rates (see `index/test_dec1_costacc_.md`),
so every deployment here sets its own literal prices through `litellm_params`, which `CustomPricingLiteLLMParams`
accepts for every key used. Each test reads the one SpendLogs row of a model group created for it, so the expected
spend is a literal. The `/v1/messages` twins matter because that route's spend logging is being ported to Rust.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway
from integration.providers.dec1_anthropic2_support import messages_body, native, replying, spend_by_group
from integration.providers.dec1_anthropic_support import (
    CALL_COST,
    PRICES,
    chat,
    deployment,
    peer,
    served,
    usage_block,
)
from pydantic import JsonValue

_LONG_PRICES: Final[Mapping[str, JsonValue]] = {
    "input_cost_per_token": 1e-06,
    "output_cost_per_token": 5e-06,
    "cache_read_input_token_cost": 1e-07,
    "cache_creation_input_token_cost": 1.25e-06,
    "input_cost_per_token_above_200k_tokens": 2e-06,
    "output_cost_per_token_above_200k_tokens": 1e-05,
    "cache_read_input_token_cost_above_200k_tokens": 2e-07,
}
_PRIORITY_PRICES: Final[Mapping[str, JsonValue]] = {
    **PRICES,
    "input_cost_per_token_priority": 0.002,
    "output_cost_per_token_priority": 0.004,
}
_SEARCH_PRICES: Final[Mapping[str, JsonValue]] = {
    **PRICES,
    "search_context_cost_per_query": {
        "search_context_size_low": 0.05,
        "search_context_size_medium": 0.05,
        "search_context_size_high": 0.05,
    },
}
_ONE_HOUR_PRICES: Final[Mapping[str, JsonValue]] = {**PRICES, "cache_creation_input_token_cost_above_1hr": 0.006}


def _chat_spend(gateway: Gateway, usage: Mapping[str, JsonValue], prices: Mapping[str, JsonValue]) -> float:
    """One chat call on a fresh deployment priced by `prices`, answered with `usage`; returns its logged spend."""
    with peer(replying(usage=dict(usage))) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False, **dict(prices))
        served(chat(gateway, model))
        return spend_by_group(gateway, model)


def _native_spend(gateway: Gateway, usage: Mapping[str, JsonValue], prices: Mapping[str, JsonValue]) -> float:
    """The same through `POST /v1/messages`."""
    with peer(replying(usage=dict(usage))) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, priced=False, **dict(prices))
        response: Final = native(gateway, messages_body(model))
        assert response.status_code == 200, response.text
        return spend_by_group(gateway, model)


def test_web_fetch_requests_add_no_fee_on_chat(gateway: Gateway) -> None:
    """Anthropic reports 11 input, 7 output and `server_tool_use.web_fetch_requests: 3` on a deployment that also
    prices web search at 0.05 per query.

    Web fetch has no per call fee and the cost code reads only `web_search_requests`, so the spend must be the token
    cost alone, 11 * 0.001 + 7 * 0.002 = 0.025.
    """
    usage: Final = usage_block(server_tool_use={"web_fetch_requests": 3})
    assert _chat_spend(gateway, usage, _SEARCH_PRICES) == pytest.approx(CALL_COST)


def test_web_fetch_requests_add_no_fee_on_messages(gateway: Gateway) -> None:
    """The same usage on `/v1/messages` must also cost 0.025: a port that prices every `server_tool_use` counter
    would bill fetches as searches.
    """
    usage: Final = usage_block(server_tool_use={"web_fetch_requests": 3})
    assert _native_spend(gateway, usage, _SEARCH_PRICES) == pytest.approx(CALL_COST)


def test_web_search_requests_bill_the_deployment_per_search_price_on_chat(gateway: Gateway) -> None:
    """The deployment sets `search_context_cost_per_query.search_context_size_medium: 0.05`; Anthropic reports 11
    input, 7 output and `web_search_requests: 2`.

    The spend must be 0.025 + 2 * 0.05 = 0.125: the deployment's search price wins over the packaged 0.01 row
    price, and the medium size is the one charged (`cost_calculation.py:126`).
    """
    usage: Final = usage_block(server_tool_use={"web_search_requests": 2})
    assert _chat_spend(gateway, usage, _SEARCH_PRICES) == pytest.approx(0.125)


def test_web_search_requests_bill_the_deployment_per_search_price_on_messages(gateway: Gateway) -> None:
    """The same usage on `/v1/messages` must also cost 0.125. UNVERIFIED: that the Rust route hands the deployment
    `search_context_cost_per_query` to the same cost function rather than the packaged row's 0.01.
    """
    usage: Final = usage_block(server_tool_use={"web_search_requests": 2})
    assert _native_spend(gateway, usage, _SEARCH_PRICES) == pytest.approx(0.125)


def test_web_search_and_web_fetch_together_bill_only_the_searches(gateway: Gateway) -> None:
    """Anthropic reports `web_search_requests: 1` and `web_fetch_requests: 4` with 11 input and 7 output.

    The spend must be 0.025 + 1 * 0.05 = 0.075.
    """
    usage: Final = usage_block(server_tool_use={"web_search_requests": 1, "web_fetch_requests": 4})
    assert _chat_spend(gateway, usage, _SEARCH_PRICES) == pytest.approx(0.075)


def test_a_prompt_above_200k_tokens_bills_every_token_at_the_long_context_rates(gateway: Gateway) -> None:
    """A 1M context window call: Anthropic reports 200001 input and 100 output tokens on a deployment with
    `*_above_200k_tokens` prices.

    The prompt crosses the threshold, so input and output both use the long rates: 200001 * 2e-06 + 100 * 1e-05 =
    0.401002.
    """
    usage: Final = usage_block(input_tokens=200001, output_tokens=100)
    assert _chat_spend(gateway, usage, _LONG_PRICES) == pytest.approx(0.401002)


def test_exactly_200k_prompt_tokens_stay_at_the_base_rates(gateway: Gateway) -> None:
    """Anthropic reports exactly 200000 input and 100 output tokens.

    Anthropic's threshold is exclusive (`prompt_tokens > threshold`), so the call bills at base rates:
    200000 * 1e-06 + 100 * 5e-06 = 0.2005.
    """
    usage: Final = usage_block(input_tokens=200000, output_tokens=100)
    assert _chat_spend(gateway, usage, _LONG_PRICES) == pytest.approx(0.2005)


def test_cache_reads_count_toward_the_200k_threshold_and_bill_at_the_long_cache_rate(gateway: Gateway) -> None:
    """Anthropic reports 1000 uncached input, `cache_read_input_tokens: 250000` and 100 output.

    `prompt_tokens` is 251000, above the threshold, so the uncached input bills at 2e-06, the cache reads at the
    long cache read rate 2e-07 and the output at 1e-05: 0.002 + 0.05 + 0.001 = 0.053.
    """
    usage: Final = usage_block(input_tokens=1000, output_tokens=100, cache_read_input_tokens=250000)
    assert _chat_spend(gateway, usage, _LONG_PRICES) == pytest.approx(0.053)


def test_a_long_prompt_on_messages_bills_the_long_context_rates(gateway: Gateway) -> None:
    """Anthropic reports 250000 input and 100 output tokens to a `/v1/messages` call.

    The spend must be 250000 * 2e-06 + 100 * 1e-05 = 0.501, the same tiering as chat.
    """
    usage: Final = usage_block(input_tokens=250000, output_tokens=100)
    assert _native_spend(gateway, usage, _LONG_PRICES) == pytest.approx(0.501)


def test_a_long_prompt_without_long_context_prices_bills_the_flat_rate(gateway: Gateway) -> None:
    """A deployment priced only with the round 1 flat `PRICES` receives 250000 input and 100 output tokens.

    With no `*_above_200k_tokens` key the flat rates apply: 250000 * 0.001 + 100 * 0.002 = 250.2. Pinned because
    the packaged direct Anthropic rows have no long context keys either.
    """
    usage: Final = usage_block(input_tokens=250000, output_tokens=100)
    assert _chat_spend(gateway, usage, PRICES) == pytest.approx(250.2)


def test_priority_service_tier_bills_the_deployment_priority_rates_on_chat(gateway: Gateway) -> None:
    """Anthropic answers `usage.service_tier: "priority"` with 11 input and 7 output tokens on a deployment that
    sets `input_cost_per_token_priority: 0.002` and `output_cost_per_token_priority: 0.004`.

    The spend must be 11 * 0.002 + 7 * 0.004 = 0.05 (`get_service_tier_cost_key` appends `_priority`).
    """
    usage: Final = usage_block(service_tier="priority")
    assert _chat_spend(gateway, usage, _PRIORITY_PRICES) == pytest.approx(0.05)


def test_priority_service_tier_bills_the_deployment_priority_rates_on_messages(gateway: Gateway) -> None:
    """The same priority answer on `/v1/messages` must also cost 0.05. UNVERIFIED: that the native route passes
    `usage.service_tier` to the cost function; the chat route reads it in `calculate_usage`.
    """
    usage: Final = usage_block(service_tier="priority")
    assert _native_spend(gateway, usage, _PRIORITY_PRICES) == pytest.approx(0.05)


@pytest.mark.parametrize("tier", ["standard", "batch"])
def test_a_non_priority_service_tier_bills_the_base_rates(gateway: Gateway, tier: str) -> None:
    """Anthropic answers `usage.service_tier` `standard` or `batch` on the same priority priced deployment.

    Neither is in `_SERVICE_TIER_TO_COST_KEY_SUFFIX` (`llm_cost_calc/utils.py:96`, which maps only flex,
    balanced, priority, fast and ultrafast), so the spend must be the base 0.025.
    """
    usage: Final = usage_block(service_tier=tier)
    assert _chat_spend(gateway, usage, _PRIORITY_PRICES) == pytest.approx(CALL_COST)


def test_one_hour_cache_writes_bill_at_the_deployment_one_hour_rate_on_messages(gateway: Gateway) -> None:
    """Anthropic reports 11 input, 7 output and `cache_creation_input_tokens: 4` broken down as
    `ephemeral_1h_input_tokens: 4` on a deployment that sets `cache_creation_input_token_cost_above_1hr: 0.006`.

    The spend must be 11 * 0.001 + 4 * 0.006 + 7 * 0.002 = 0.049, the 1h write rate rather than the 0.003 five
    minute rate.
    """
    usage: Final = usage_block(
        cache_creation_input_tokens=4,
        cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4},
    )
    assert _native_spend(gateway, usage, _ONE_HOUR_PRICES) == pytest.approx(0.049)


def test_one_hour_cache_writes_bill_at_the_deployment_one_hour_rate_on_chat(gateway: Gateway) -> None:
    """The same usage on chat must also cost 0.049."""
    usage: Final = usage_block(
        cache_creation_input_tokens=4,
        cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4},
    )
    assert _chat_spend(gateway, usage, _ONE_HOUR_PRICES) == pytest.approx(0.049)


def test_the_chat_response_cost_header_includes_the_web_search_fee(gateway: Gateway) -> None:
    """A chat call answered with 11 input, 7 output and `web_search_requests: 2` on the search priced deployment.

    The caller's `x-litellm-response-cost` header must already be 0.125, the same number the spend row holds, so
    clients that budget per call see the search fee.
    """
    usage: Final = usage_block(server_tool_use={"web_search_requests": 2})
    with peer(replying(usage=usage)) as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider, priced=False, **dict(_SEARCH_PRICES)))
        served(response)
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.125)
