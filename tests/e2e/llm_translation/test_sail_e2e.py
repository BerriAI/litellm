"""Live e2e: Sail through the gateway, where LiteLLM turns ``service_tier`` into Sail's
``metadata.completion_window`` and bills the price columns of the window it sent.

The deployment carries its own base, balanced and flex rates, each distinct, so a bill at
the wrong tier cannot pass. They are registered on the deployment instead of read from the
proxy's cost map, because a stack that loads the published map has no ``sail/`` rows until
this provider ships. Requires SAIL_API_KEY on the proxy; no skip gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import openai
import pytest
from e2e_config import SLOW_PROVIDER_TIMEOUT_SECONDS, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, SpendLogRow
from openai import OpenAI
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients, response_header

pytestmark = pytest.mark.e2e

BACKEND: Final = "sail/zai-org/GLM-5.3"
PricedTier = Literal["base", "balanced", "flex"]
PRICED_TIERS: Final[tuple[PricedTier, ...]] = ("base", "balanced", "flex")
PROMPT: Final = "Reply with one word."
MAX_TOKENS: Final = 512


@dataclass(frozen=True, slots=True)
class _Rates:
    input: float
    output: float
    cache_read: float


RATES: Final[Mapping[PricedTier, _Rates]] = {
    "base": _Rates(input=3e-06, output=9e-06, cache_read=1e-06),
    "balanced": _Rates(input=2e-06, output=6e-06, cache_read=7e-07),
    "flex": _Rates(input=1e-06, output=3e-06, cache_read=4e-07),
}


@dataclass(frozen=True, slots=True)
class _Tokens:
    prompt: int
    cached: int
    completion: int


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-12, abs(expected) * 1e-2)


def _cost(rates: _Rates, tokens: _Tokens) -> float:
    return (
        (tokens.prompt - tokens.cached) * rates.input
        + tokens.cached * rates.cache_read
        + tokens.completion * rates.output
    )


def _register(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model: Final = f"e2e-sail-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=BACKEND,
            api_key="os.environ/SAIL_API_KEY",
            input_cost_per_token=RATES["base"].input,
            output_cost_per_token=RATES["base"].output,
            cache_read_input_token_cost=RATES["base"].cache_read,
            input_cost_per_token_balanced=RATES["balanced"].input,
            output_cost_per_token_balanced=RATES["balanced"].output,
            cache_read_input_token_cost_balanced=RATES["balanced"].cache_read,
            input_cost_per_token_flex=RATES["flex"].input,
            output_cost_per_token_flex=RATES["flex"].output,
            cache_read_input_token_cost_flex=RATES["flex"].cache_read,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _openai(sdk: SdkClients, key: str) -> OpenAI:
    return sdk.openai(key).with_options(timeout=SLOW_PROVIDER_TIMEOUT_SECONDS)


def _assert_billed_at(tier: PricedTier, tokens: _Tokens, header_cost: str | None) -> float:
    assert tokens.prompt > 0 and tokens.completion > 0, f"Sail reported no usage, so no cost is real: {tokens}"
    assert header_cost is not None, "x-litellm-response-cost header missing"
    costs: Final = {priced: _cost(rates, tokens) for priced, rates in RATES.items()}
    assert not any(_approx_equal(costs[other], costs[tier]) for other in PRICED_TIERS if other != tier), (
        f"{BACKEND} tier rates too close together to tell {tier} apart at {tokens}: {costs}"
    )
    assert _approx_equal(float(header_cost), costs[tier]), (
        f"header cost {header_cost} is not the {tier} price at {tokens}: expected {costs[tier]}, all tiers {costs}"
    )
    return float(header_cost)


def _assert_spend_row_matches(proxy: ProxyClient, key: str, header_cost: float) -> None:
    def priced(rows: list[SpendLogRow]) -> bool:
        return any((row.spend or 0) > 0 for row in rows)

    rows: Final = [row for row in proxy.poll_logs_for_key(key, predicate=priced) if (row.spend or 0) > 0]
    assert rows, f"no priced spend row landed for key {key}"
    assert rows[0].custom_llm_provider == "sail", f"spend row misattributed: {rows[0]}"
    assert rows[0].spend is not None and _approx_equal(rows[0].spend, header_cost), (
        f"logged spend {rows[0].spend} disagrees with the x-litellm-response-cost header {header_cost}"
    )


class TestSailChatCompletions:
    @pytest.mark.covers("llm.chat_completions.sail.service_tier.nonstream.cost_logged")
    @pytest.mark.parametrize(
        ("service_tier", "billed_tier"), [("flex", "flex"), ("balanced", "balanced"), ("auto", "base")]
    )
    def test_service_tier_bills_the_matching_completion_window(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
        service_tier: str,
        billed_tier: PricedTier,
    ) -> None:
        model, key = _register(proxy, resources)

        raw: Final = _openai(sdk, key).chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": f"{PROMPT} {unique_marker()}"}],
            max_completion_tokens=MAX_TOKENS,
            extra_body={**NO_PROXY_CACHE, "service_tier": service_tier},
        )
        usage: Final = raw.parse().usage
        assert usage is not None, "chat response carries no usage"
        details: Final = usage.prompt_tokens_details
        tokens: Final = _Tokens(
            prompt=usage.prompt_tokens,
            cached=(details.cached_tokens or 0) if details else 0,
            completion=usage.completion_tokens,
        )

        header_cost: Final = _assert_billed_at(
            billed_tier, tokens, response_header(raw.headers, "x-litellm-response-cost")
        )
        _assert_spend_row_matches(proxy, key, header_cost)

    @pytest.mark.covers("llm.chat_completions.sail.service_tier.nonstream.rejects_unknown_tier")
    def test_unknown_service_tier_is_rejected(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)

        with pytest.raises(openai.BadRequestError) as raised:
            _ = _openai(sdk, key).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                max_completion_tokens=MAX_TOKENS,
                extra_body={**NO_PROXY_CACHE, "service_tier": "bogus"},
            )
        assert "service_tier" in raised.value.message, f"400 does not name service_tier: {raised.value.message}"


class TestSailResponses:
    @pytest.mark.covers("llm.responses.sail.service_tier.nonstream.cost_logged")
    def test_flex_completion_window_bills_flex_rates(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)

        raw: Final = _openai(sdk, key).responses.with_raw_response.create(
            model=model,
            input=f"{PROMPT} {unique_marker()}",
            max_output_tokens=MAX_TOKENS,
            metadata={"completion_window": "flex"},
            extra_body=NO_PROXY_CACHE,
        )
        usage: Final = raw.parse().usage
        assert usage is not None, "responses answer carries no usage"
        tokens: Final = _Tokens(
            prompt=usage.input_tokens,
            cached=usage.input_tokens_details.cached_tokens,
            completion=usage.output_tokens,
        )

        header_cost: Final = _assert_billed_at("flex", tokens, response_header(raw.headers, "x-litellm-response-cost"))
        _assert_spend_row_matches(proxy, key, header_cost)


class TestSailMessages:
    @pytest.mark.covers("llm.messages.sail.basic.nonstream.works")
    def test_plain_call_returns_a_message(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)

        message: Final = sdk.anthropic(key).messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": PROMPT}],
            extra_body=NO_PROXY_CACHE,
        )
        assert message.role == "assistant" and message.content, f"/v1/messages returned no content: {message}"
        assert message.usage.output_tokens > 0, f"/v1/messages reported no output usage: {message.usage}"
