"""Azure usage reporting and cost: cached tokens, reasoning tokens, SpendLogs rows and the cost header.

Azure OpenAI reports usage in OpenAI's shape. `prompt_tokens_details.cached_tokens` is part of
`prompt_tokens` and is priced at `cache_read_input_token_cost`; `completion_tokens_details.reasoning_tokens`
is part of `completion_tokens` and is priced at the output rate when no reasoning price is set
(`generic_cost_per_token`, called from `litellm/llms/azure/cost_calculation.py:10`). Deployment prices
given in `litellm_params` override the cost map. Every call writes one `LiteLLM_SpendLogs` row naming the
provider (`azure` or `azure_ai`), and the answer carries `x-litellm-response-cost`.

Spend logging is one of the pieces being ported to Rust this month, so these rows are the contract: same
token counts, same provider label, same cost to the cent, and a `failure` row for a failed call.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure_support import (
    CACHE_READ_COST,
    GPT5_DEPLOYMENT,
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    always,
    azure,
    azure_error,
    chat_reply,
    foundry,
    spend_row,
    spend_rows_for_group,
)

_CACHED_USAGE: Final = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "prompt_tokens_details": {"cached_tokens": 5},
}
_REASONING_USAGE: Final = {
    "prompt_tokens": 11,
    "completion_tokens": 7,
    "total_tokens": 18,
    "completion_tokens_details": {"reasoning_tokens": 4},
}


def _cached(_: Request) -> Reply:
    return chat_reply("pong", usage=_CACHED_USAGE)


def _plain(_: Request) -> Reply:
    return chat_reply("pong")


def test_cached_prompt_tokens_are_reported_to_the_caller(gateway: Gateway) -> None:
    """Azure reports 11 prompt tokens of which 5 were served from its prompt cache.

    The caller must get `usage.prompt_tokens == 11` and `usage.prompt_tokens_details.cached_tokens == 5`.
    """
    with azure(gateway, _cached) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        usage: Final = response.json()["usage"]
        assert usage["prompt_tokens"] == 11
        assert usage["prompt_tokens_details"]["cached_tokens"] == 5


def test_cached_prompt_tokens_are_priced_at_the_cache_read_rate(gateway: Gateway) -> None:
    """Same usage on a deployment priced 0.001 input, 0.0001 cache read and 0.002 output per token.

    The SpendLogs row must have `spend = 6 * 0.001 + 5 * 0.0001 + 7 * 0.002 = 0.0205`: the 5 cached tokens at the
    cache read rate and only the 6 uncached ones at the input rate.
    """
    with azure(gateway, _cached, **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert float(str(row["spend"])) == pytest.approx(6 * INPUT_COST + 5 * CACHE_READ_COST + 7 * OUTPUT_COST)


def test_reasoning_tokens_are_reported_inside_completion_tokens_for_azure_openai(gateway: Gateway) -> None:
    """A GPT-5 Azure deployment reports 7 completion tokens of which 4 were reasoning.

    Azure OpenAI already counts reasoning inside `completion_tokens`, so the caller must get
    `completion_tokens: 7` (not 11) and `completion_tokens_details.reasoning_tokens: 4`. The `azure_ai` fold
    rule does not apply to `azure/`.
    """
    with azure(gateway, always(chat_reply("pong", usage=_REASONING_USAGE)), model=f"azure/{GPT5_DEPLOYMENT}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        usage: Final = response.json()["usage"]
        assert usage["completion_tokens"] == 7
        assert usage["completion_tokens_details"]["reasoning_tokens"] == 4


def test_reasoning_tokens_are_priced_at_the_output_rate(gateway: Gateway) -> None:
    """Same reasoning usage on a GPT-5 deployment priced 0.001 input and 0.002 output with no reasoning price.

    The row must have `spend = 11 * 0.001 + 7 * 0.002 = 0.025`: reasoning is neither free nor billed twice.
    """
    with azure(
        gateway, always(chat_reply("pong", usage=_REASONING_USAGE)), model=f"azure/{GPT5_DEPLOYMENT}", **PRICES
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_an_azure_chat_call_writes_one_spend_row_naming_azure(gateway: Gateway) -> None:
    """A successful priced Azure chat call with 11 prompt and 7 completion tokens.

    One SpendLogs row must land with `call_type: acompletion`, `custom_llm_provider: azure`, `status: success`,
    `prompt_tokens: 11`, `completion_tokens: 7`, `total_tokens: 18`, `model_group` equal to the public name and
    `spend = 0.025`.
    """
    with azure(gateway, _plain, **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert row["call_type"] == "acompletion"
        assert row["custom_llm_provider"] == "azure"
        assert row["status"] == "success"
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 7, 18)
        assert row["model_group"] == rig.model
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_the_spend_row_records_the_azure_api_base(gateway: Gateway) -> None:
    """Same priced call against the owned peer.

    The row's `api_base` must start with the peer URL, so spend can be attributed to the Azure resource.
    UNVERIFIED: whether the stored value is the bare base or the full deployment URL; only the prefix is pinned.
    """
    with azure(gateway, _plain, **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert str(row["api_base"]).startswith(rig.wire.url), row["api_base"]


def test_the_answer_carries_the_response_cost_header(gateway: Gateway) -> None:
    """Same priced call.

    The caller must get `x-litellm-response-cost` whose float value is `0.025`.
    """
    with azure(gateway, _plain, **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_a_failed_azure_call_writes_one_failure_row(gateway: Gateway) -> None:
    """Azure answers 400 for every call; `num_retries: 0` and SDK `max_retries: 0`.

    Exactly one SpendLogs row for the group must land with `status: failure` and `spend` 0, so failed Azure
    calls stay visible without being billed.
    """
    with azure(gateway, always(azure_error(400, "invalid_request_error", "bad")), max_retries=0, **PRICES) as rig:
        assert rig.chat(num_retries=0).status_code == 400
        rows: Final = spend_rows_for_group(rig.model)
        assert [row["status"] for row in rows] == ["failure"]
        assert float(str(rows[0]["spend"])) == 0


def test_a_foundry_call_writes_one_spend_row_naming_azure_ai(gateway: Gateway) -> None:
    """A priced `azure_ai/mistral-large-3` call with 11 prompt and 7 completion tokens.

    One SpendLogs row must land with `custom_llm_provider: azure_ai` (not `azure`), 11 and 7 tokens and
    `spend = 0.025` from the deployment prices.
    """
    with foundry(gateway, always(chat_reply("pong")), model="azure_ai/mistral-large-3", **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert row["custom_llm_provider"] == "azure_ai"
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert float(str(row["spend"])) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_a_foundry_folded_reasoning_usage_is_billed_on_the_folded_completion(gateway: Gateway) -> None:
    """A priced `azure_ai/deepseek-v4-flash` call reporting 10 prompt, 5 completion, 25 total and 10 reasoning tokens.

    The fold makes 15 completion tokens, so the row must have `completion_tokens: 15` and
    `spend = 10 * 0.001 + 15 * 0.002 = 0.04`. UNVERIFIED: that the logged usage is the folded one.
    """
    usage: Final = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 25,
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    with foundry(gateway, always(chat_reply("pong", usage=usage)), model="azure_ai/deepseek-v4-flash", **PRICES) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert row["completion_tokens"] == 15
        assert float(str(row["spend"])) == pytest.approx(10 * INPUT_COST + 15 * OUTPUT_COST)


def test_without_deployment_prices_the_cost_map_price_of_the_azure_model_applies(gateway: Gateway) -> None:
    """An `azure_ai/deepseek-v4-flash` deployment with no prices; Foundry reports 11 prompt and 7 completion tokens.

    The cost map row (`model_prices_and_context_window.json`: 1.9e-07 input, 5.1e-07 output) must price the call:
    `spend = 11 * 1.9e-07 + 7 * 5.1e-07 = 5.66e-06`.
    """
    with foundry(gateway, always(chat_reply("pong")), model="azure_ai/deepseek-v4-flash") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert float(str(row["spend"])) == pytest.approx(11 * 1.9e-07 + 7 * 5.1e-07)
