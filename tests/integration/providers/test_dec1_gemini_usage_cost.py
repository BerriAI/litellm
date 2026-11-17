"""Usage and spend for `gemini/` (Google AI Studio) calls, priced from the shipped cost map.

The proxy prices an AI Studio call with `litellm/llms/gemini/cost_calculator.py` on top of the generic
token pricing (`generic_cost_per_token`, `custom_llm_provider="gemini"`). Facts these tests pin, all
read from `model_prices_and_context_window.json` and `litellm_core_utils/llm_cost_calc/utils.py`:

`gemini/gemini-2.5-pro` has a 200k token price tier. When the prompt is MORE than 200,000 tokens
(strictly; only xAI uses an inclusive threshold) every input token is billed at
`input_cost_per_token_above_200k_tokens` 2.5e-06 instead of 1.25e-06, every output token at 1.5e-05
instead of 1e-05, and cached tokens at 2.5e-07 instead of 1.25e-07. The tier is picked from the whole
prompt, cached tokens included.

Cached tokens (`cachedContentTokenCount`) are part of `promptTokenCount` and billed at
`cache_read_input_token_cost`; the rest of the prompt at the input rate. Thinking tokens
(`thoughtsTokenCount`) are added to the completion when Google does not already include them, and are
billed at the output rate on a model without `output_cost_per_reasoning_token`.

Google Search grounding is billed on top of tokens: Gemini 2.x "per_prompt" (one $0.035 fee per grounded
prompt, however many queries), Gemini 3 "per_query" ($0.014 per distinct query) via
`web_search_billing_unit` (`cost_per_web_search_request`).

Each test registers its own model group and reads `LiteLLM_SpendLogs` by `model_group`, polling with
`eventually` because the row is written asynchronously.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration.providers.dec1_gemini_support import (
    GEMINI_3_FLASH,
    PRO,
    at,
    gemini,
    gemini_echo,
    gemini_reply,
    json_of,
    marker,
    search_grounding,
    spend_rows,
    usage_metadata,
)
from pydantic import JsonValue


def _one_row(gateway: Gateway, model_group: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: spend_rows(gateway, model_group), lambda found: len(found) >= 1, seconds=90)
    assert len(rows) == 1, rows
    return rows[0]


def _spend(row: dict[str, JsonValue]) -> float:
    return float(str(row["spend"]))


# ---------------------------------------------------------------- the spend row


def test_a_chat_writes_one_spend_row_naming_gemini(gateway: Gateway) -> None:
    """One chat on a `gemini/` deployment writes one SpendLogs row with `custom_llm_provider = "gemini"`
    (not `vertex_ai`), `prompt_tokens` 11, `completion_tokens` 7 and `total_tokens` 18 from `usageMetadata`."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        row: Final = _one_row(gateway, rig.model)
        assert (row["custom_llm_provider"], row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (
            "gemini",
            11,
            7,
            18,
        )


def test_deployment_custom_prices_win_over_the_cost_map(gateway: Gateway) -> None:
    """With `input_cost_per_token: 0.001` and `output_cost_per_token: 0.002` on the deployment the row's
    spend is `11 * 0.001 + 7 * 0.002 = 0.025`, not the cost map price of gemini-2.5-flash."""
    with gemini(gateway, gemini_echo, input_cost_per_token=0.001, output_cost_per_token=0.002) as rig:
        assert rig.chat().status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(0.025)


def test_a_streamed_chat_writes_one_row_with_the_final_usage(gateway: Gateway) -> None:
    """A streamed chat whose last frame carries 11 prompt and 7 candidate tokens writes exactly one row with
    those counts and the custom price `0.025`, even though the caller saw several chunks."""
    with gemini(gateway, gemini_echo, input_cost_per_token=0.001, output_cost_per_token=0.002) as rig:
        rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        row: Final = _one_row(gateway, rig.model)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert _spend(row) == pytest.approx(0.025)


def test_the_cost_map_price_of_gemini_2_5_flash(gateway: Gateway) -> None:
    """Without custom prices, 11 prompt and 7 output tokens on gemini-2.5-flash cost
    `11 * 3e-07 + 7 * 2.5e-06 = 2.08e-05` (cost map `gemini/gemini-2.5-flash`)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(11 * 3e-07 + 7 * 2.5e-06)


def test_the_response_cost_header_matches_the_row(gateway: Gateway) -> None:
    """The caller's `x-litellm-response-cost` header carries the same cost the row stores (custom prices,
    0.025), so client side accounting and the DB agree."""
    with gemini(gateway, gemini_echo, input_cost_per_token=0.001, output_cost_per_token=0.002) as rig:
        response: Final = rig.chat()
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.025)
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(0.025)


# ---------------------------------------------------------------- 200k tier


def test_a_prompt_below_200k_is_billed_at_the_base_rate(gateway: Gateway) -> None:
    """gemini-2.5-pro with 1,000 prompt and 100 output tokens costs `1000 * 1.25e-06 + 100 * 1e-05 = 0.00225`."""
    with gemini(gateway, lambda _r: gemini_reply(usage=usage_metadata(1000, 100), model=PRO), backend=PRO) as rig:
        assert rig.chat().status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(0.00225)


def test_a_prompt_of_exactly_200k_stays_in_the_base_tier(gateway: Gateway) -> None:
    """At exactly 200,000 prompt tokens the threshold is not crossed (`prompt_tokens > threshold`), so the
    cost is `200000 * 1.25e-06 + 100 * 1e-05 = 0.251`."""
    with gemini(gateway, lambda _r: gemini_reply(usage=usage_metadata(200000, 100), model=PRO), backend=PRO) as rig:
        assert rig.chat().status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(0.251)


def test_a_prompt_above_200k_bills_every_token_at_the_higher_tier(gateway: Gateway) -> None:
    """At 200,001 prompt tokens ALL input tokens move to 2.5e-06 and output to 1.5e-05:
    `200001 * 2.5e-06 + 100 * 1.5e-05 = 0.5015025`; not just the tokens past 200k."""
    with gemini(gateway, lambda _r: gemini_reply(usage=usage_metadata(200001, 100), model=PRO), backend=PRO) as rig:
        assert rig.chat().status_code == 200
        row: Final = _one_row(gateway, rig.model)
        assert row["prompt_tokens"] == 200001
        assert _spend(row) == pytest.approx(0.5015025)


def test_cached_tokens_above_200k_use_the_tiered_cache_rate(gateway: Gateway) -> None:
    """250,000 prompt tokens of which 200,000 are cached: `50000 * 2.5e-06 + 200000 * 2.5e-07 + 100 * 1.5e-05
    = 0.1765`; the tier is chosen from the full prompt and the cached part uses
    `cache_read_input_token_cost_above_200k_tokens`."""
    usage: Final = usage_metadata(250000, 100, cachedContentTokenCount=200000)
    with gemini(gateway, lambda _r: gemini_reply(usage=usage, model=PRO), backend=PRO) as rig:
        assert rig.chat().status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(0.1765)


def test_a_streamed_prompt_above_200k_is_billed_at_the_higher_tier(gateway: Gateway) -> None:
    """The tier also applies when the usage arrives on the last SSE frame: 200,001 prompt and 100 output
    tokens streamed cost `0.5015025`."""
    from integration.providers.dec1_gemini_support import gemini_sse

    with gemini(gateway, lambda _r: gemini_sse(usage=usage_metadata(200001, 100)), backend=PRO) as rig:
        rig.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}]})
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(0.5015025)


# ---------------------------------------------------------------- cache and thinking


def test_cached_tokens_are_billed_at_the_cache_read_rate(gateway: Gateway) -> None:
    """10,000 prompt tokens of which 8,000 are cached on gemini-2.5-pro: `2000 * 1.25e-06 + 8000 * 1.25e-07
    + 100 * 1e-05 = 0.0045`."""
    usage: Final = usage_metadata(10000, 100, cachedContentTokenCount=8000)
    with gemini(gateway, lambda _r: gemini_reply(usage=usage, model=PRO), backend=PRO) as rig:
        assert rig.chat().status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(0.0045)


def test_thinking_tokens_are_added_to_the_completion_and_billed(gateway: Gateway) -> None:
    """`thoughtsTokenCount: 50` with 100 candidates and a total of 1150 (so not already included) gives
    `completion_tokens: 150`, `completion_tokens_details.reasoning_tokens: 50`, and a cost of
    `1000 * 1.25e-06 + 150 * 1e-05 = 0.00275` on gemini-2.5-pro."""
    usage: Final = usage_metadata(1000, 100, thoughtsTokenCount=50, totalTokenCount=1150)
    with gemini(gateway, lambda _r: gemini_reply(usage=usage, model=PRO), backend=PRO) as rig:
        body: Final = json_of(rig.chat())
        assert at(body, "usage", "completion_tokens") == 150
        assert at(body, "usage", "completion_tokens_details", "reasoning_tokens") == 50
        row: Final = _one_row(gateway, rig.model)
        assert row["completion_tokens"] == 150
        assert _spend(row) == pytest.approx(0.00275)


# ---------------------------------------------------------------- grounding fees


def test_search_grounding_on_gemini_2_adds_one_fee_per_prompt(gateway: Gateway) -> None:
    """On gemini-2.5-flash (per_prompt billing) a grounded answer with two distinct queries adds ONE $0.035
    fee: `11 * 3e-07 + 7 * 2.5e-06 + 0.035 = 0.0350208`."""
    grounding: Final = search_grounding(("q one", "q two"))
    with gemini(gateway, lambda _r: gemini_reply(groundingMetadata=grounding)) as rig:
        assert rig.chat(tools=[{"googleSearch": {}}]).status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(11 * 3e-07 + 7 * 2.5e-06 + 0.035)


def test_search_grounding_on_gemini_3_adds_a_fee_per_query(gateway: Gateway) -> None:
    """On gemini-3-flash-preview (`web_search_billing_unit: per_query`, $0.014) two distinct queries add
    `2 * 0.014`: `11 * 5e-07 + 7 * 3e-06 + 0.028 = 0.0280265`."""
    grounding: Final = search_grounding(("q one", "q two"))
    with gemini(
        gateway, lambda _r: gemini_reply(groundingMetadata=grounding, model=GEMINI_3_FLASH), backend=GEMINI_3_FLASH
    ) as rig:
        assert rig.chat(tools=[{"googleSearch": {}}]).status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(11 * 5e-07 + 7 * 3e-06 + 0.028)


def test_url_context_metadata_alone_adds_no_search_fee(gateway: Gateway) -> None:
    """URL context emits grounding chunks but no `webSearchQueries`; with `groundingMetadata` holding only a
    web chunk the spend is the token cost alone, `11 * 3e-07 + 7 * 2.5e-06` (no $0.035 fee)."""
    grounding: Final[dict[str, JsonValue]] = {
        "groundingChunks": [{"web": {"uri": "https://example.com", "title": "x"}}]
    }
    with gemini(gateway, lambda _r: gemini_reply(groundingMetadata=grounding)) as rig:
        assert rig.chat(tools=[{"urlContext": {}}]).status_code == 200
        assert _spend(_one_row(gateway, rig.model)) == pytest.approx(11 * 3e-07 + 7 * 2.5e-06)


def test_a_failed_call_writes_a_failure_row_with_no_spend(gateway: Gateway) -> None:
    """A Google 400 writes a SpendLogs row for the model group with `spend = 0` and status `failure`.
    UNVERIFIED: column `status` exists on SpendLogs and failed calls are logged by default on the shared
    proxy."""
    from integration.providers.dec1_gemini_support import google_error

    with gemini(gateway, lambda _r: google_error(400, "INVALID_ARGUMENT", "bad request")) as rig:
        assert rig.chat().status_code == 400
        rows: Final = eventually(
            lambda: gateway.rows('SELECT spend, status FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (rig.model,)),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert float(str(rows[0]["spend"])) == 0.0
        assert rows[0]["status"] == "failure"
