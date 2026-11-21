"""Contract: what an OpenAI, Azure OpenAI or Azure AI Foundry call costs, priced from the packaged rows.

OpenAI style usage counts cached and audio tokens inside `prompt_tokens` and reasoning and audio tokens
inside `completion_tokens`; the generic calculator (`utils.py:1263` `_calculate_input_cost`) bills the
uncached text remainder at `input_cost_per_token`, `cached_tokens` at `cache_read_input_token_cost`,
audio at `input_cost_per_audio_token` / `output_cost_per_audio_token`, and reasoning at the output rate
unless the row has `output_cost_per_reasoning_token`. The GPT-5.4 rows have a long context tier above
272000 prompt tokens. The `service_tier` the provider echoes on the answer selects `_priority` / `_flex`
rates and wins over the requested tier (`cost_calculator.py:1287`).

Azure deployments are named by the customer, so cost comes from `model_info.base_model` when set
(`select_model_name_for_cost_calc`, `cost_calculator.py:1099`), else from the model the answer names; a
`base_model` of `azure/eu/...` or `azure/us/...` bills the data zone rows. Deployment prices beat both.

Batch output priced at the row's `*_batches` keys is covered at the end through `/v1/batches` with the
helpers of `providers/dec1_batchfiles_support.py`.

Rows used (repo root `model_prices_and_context_window.json`): `gpt-5.4` (line 39001), `gpt-4o-mini`
(36308), `gpt-audio` (36080), `azure/gpt-5.4` (8243), `azure/eu/gpt-5.4` (8353), `azure/us/gpt-5.4`
(8302), `azure/gpt-5.4-2026-03-05` (8404), `azure/gpt-audio` (6402), `azure/gpt-4o-mini` (6636),
`azure_ai/gpt-5.4` (4874), `azure_ai/claude-sonnet-4-5` (4243).
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.pricing.dec1_costacc_support import (
    anthropic_message,
    anthropic_usage,
    assert_billed,
    json_reply,
    openai,
    openai_chat,
    openai_reply,
    openai_usage,
)
from integration.providers.dec1_azure_support import azure, foundry
from integration.providers.dec1_batchfiles_support import (
    Script,
    batch_spend_rows,
    chat_row,
    json_of,
    jsonl,
    openai_batch,
    openai_file,
    openai_output_line,
    reply,
    text_reply,
    unique,
)
from pydantic import JsonValue

GPT54: Final = "gpt-5.4"
AZURE_DEPLOYMENT: Final = "azure/dec1-costacc-prod"
"""A customer chosen Azure deployment name that matches no cost map row."""


def _always(answer: Reply):  # noqa: ANN202
    def respond(_: Request) -> Reply:
        return answer

    return respond


# ---------------------------------------------------------------- OpenAI


def test_openai_plain_tokens_bill_at_the_gpt_5_4_rates(gateway: Gateway) -> None:
    """`openai/gpt-5.4` (line 39001: 2.5e-06 in, 1.5e-05 out) reports 1000 prompt and 500 completion tokens:
    `0.0025 + 0.0075 = 0.01`."""
    with openai(gateway, openai_reply(openai_usage(1000, 500), model=GPT54), backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01)


def test_openai_cached_prompt_tokens_bill_at_the_cache_read_rate(gateway: Gateway) -> None:
    """1000 prompt tokens of which `cached_tokens: 600`: `400 * 2.5e-06 + 600 * 2.5e-07 + 500 * 1.5e-05 = 0.00865`.
    Cached tokens are inside `prompt_tokens` on this wire, unlike Anthropic."""
    with openai(gateway, openai_reply(openai_usage(1000, 500, cached=600), model=GPT54), backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00865)


def test_openai_reasoning_tokens_bill_inside_completion_at_the_output_rate(gateway: Gateway) -> None:
    """500 completion tokens of which `reasoning_tokens: 300`. gpt-5.4 has no `output_cost_per_reasoning_token`,
    so reasoning bills at 1.5e-05 and is not added twice: `0.01`."""
    with openai(gateway, openai_reply(openai_usage(1000, 500, reasoning=300), model=GPT54), backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01)


def test_openai_served_priority_bills_the_priority_rates(gateway: Gateway) -> None:
    """The answer echoes `service_tier: "priority"`: `1000 * 5e-06 + 500 * 3e-05 = 0.02` (line 39001)."""
    answer: Final = openai_reply(openai_usage(1000, 500), model=GPT54, service_tier="priority")
    with openai(gateway, answer, backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.02)


def test_openai_served_default_overrides_a_requested_priority(gateway: Gateway) -> None:
    """The caller asks for priority, OpenAI answers `service_tier: "default"` (it had no priority capacity).
    The served tier wins: `0.01`."""
    answer: Final = openai_reply(openai_usage(1000, 500), model=GPT54, service_tier="default")
    with openai(gateway, answer, backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.01)


def test_openai_served_flex_bills_the_flex_rates(gateway: Gateway) -> None:
    """The answer echoes `service_tier: "flex"`: `1000 * 1.25e-06 + 500 * 7.5e-06 = 0.005`."""
    answer: Final = openai_reply(openai_usage(1000, 500), model=GPT54, service_tier="flex")
    with openai(gateway, answer, backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="flex"), 0.005)


def test_openai_flex_cache_reads_bill_at_the_flex_cache_rate(gateway: Gateway) -> None:
    """Flex with 600 of 1000 prompt tokens cached: `400 * 1.25e-06 + 600 * 1.3e-07 + 500 * 7.5e-06 = 0.004328`."""
    answer: Final = openai_reply(openai_usage(1000, 500, cached=600), model=GPT54, service_tier="flex")
    with openai(gateway, answer, backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="flex"), 0.004328)


def test_openai_prompt_above_272k_bills_the_long_context_rates(gateway: Gateway) -> None:
    """300000 prompt and 500 completion tokens: `300000 * 5e-06 + 500 * 2.25e-05 = 1.51125`."""
    with openai(gateway, openai_reply(openai_usage(300000, 500), model=GPT54), backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 1.51125)


def test_openai_prompt_of_exactly_272k_stays_in_the_base_tier(gateway: Gateway) -> None:
    """272000 prompt tokens do not cross the threshold: `272000 * 2.5e-06 + 500 * 1.5e-05 = 0.6875`."""
    with openai(gateway, openai_reply(openai_usage(272000, 500), model=GPT54), backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.6875)


def test_openai_priority_above_272k_without_a_priority_long_rate_bills_the_standard_long_rate(gateway: Gateway) -> None:
    """The OpenAI gpt-5.4 row (line 39001) has no `input_cost_per_token_above_272k_tokens_priority`, so a served
    priority call of 300000 prompt tokens falls back to the standard long context keys (`get_cost_per_unit`
    strips `_priority`, `utils.py:941`): `300000 * 5e-06 + 500 * 2.25e-05 = 1.51125`, less than the short
    priority rate. Pinned as today's behaviour; the Azure row has the priority long rates (see below)."""
    answer: Final = openai_reply(openai_usage(300000, 500), model=GPT54, service_tier="priority")
    with openai(gateway, answer, backend=GPT54) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 1.51125)


def test_openai_audio_prompt_tokens_bill_at_the_audio_rate(gateway: Gateway) -> None:
    """`openai/gpt-audio` (line 36080: text 2.5e-06, audio in 3.2e-05, out 1e-05) reports 1000 prompt tokens of
    which `audio_tokens: 400`, and 100 completion: `600 * 2.5e-06 + 400 * 3.2e-05 + 100 * 1e-05 = 0.0153`."""
    answer: Final = openai_reply(openai_usage(1000, 100, audio_in=400), model="gpt-audio")
    with openai(gateway, answer, backend="gpt-audio") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0153)


def test_openai_audio_completion_tokens_bill_at_the_audio_output_rate(gateway: Gateway) -> None:
    """gpt-audio reports 1000 text prompt tokens and 300 completion tokens of which `audio_tokens: 200`:
    `1000 * 2.5e-06 + 100 * 1e-05 + 200 * 6.4e-05 = 0.0163`."""
    answer: Final = openai_reply(openai_usage(1000, 300, audio_out=200), model="gpt-audio")
    with openai(gateway, answer, backend="gpt-audio") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0163)


def test_openai_web_search_options_add_the_search_context_fee(gateway: Gateway) -> None:
    """A `gpt-4o-mini` call (line 36308: 1.5e-07 in, 6e-07 out, `search_context_size_medium: 0.025`) sent with
    `web_search_options: {search_context_size: medium}`, whose answer carries a `url_citation` annotation:
    `1000 * 1.5e-07 + 500 * 6e-07 + 0.025 = 0.02545` (`tool_call_cost_tracking.py:681`). The fee is charged only
    when the answer shows a search was made: a chat answer counts as one through a `url_citation` annotation
    (`response_object_includes_web_search_call`, tool_call_cost_tracking.py:550-556), so a plain answer bills
    0.00045 whatever the request asked for (CI run 44161684479). UNVERIFIED: assumes the chat request's
    `web_search_options` reach `standard_built_in_tools_params`; without them the medium rate is the default
    (tool_call_cost_tracking.py:692-700), so the expected cost is the same."""
    body: Final = openai_chat(openai_usage(1000, 500), model="gpt-4o-mini")
    citation: Final[JsonValue] = {
        "type": "url_citation",
        "url_citation": {"start_index": 0, "end_index": 6, "url": "https://example.com/", "title": "example"},
    }
    choice: Final = object_value(list_value(body["choices"])[0])
    cited: Final[JsonValue] = {**choice, "message": {**object_value(choice["message"]), "annotations": [citation]}}
    answer: Final = json_reply({**body, "choices": [cited]})
    with openai(gateway, answer, backend="gpt-4o-mini") as rig:
        response: Final = rig.chat(web_search_options={"search_context_size": "medium"})
        assert_billed(gateway, rig.model, response, 0.02545)


def test_openai_deployment_prices_win_and_keep_the_row_cache_rate(gateway: Gateway) -> None:
    """A gpt-5.4 deployment sets `input_cost_per_token: 1e-05`, `output_cost_per_token: 2e-05`; 600 of 1000 prompt
    tokens are cached: `400 * 1e-05 + 600 * 2.5e-07 + 500 * 2e-05 = 0.01415` (cache rate inherited,
    `registry.py:296`)."""
    with openai(
        gateway,
        openai_reply(openai_usage(1000, 500, cached=600), model=GPT54),
        backend=GPT54,
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01415)


def test_openai_deployment_prices_still_bill_a_served_priority_call_at_the_row_priority_rates(gateway: Gateway) -> None:
    """The same custom priced deployment served at priority bills the row's priority rates
    `1000 * 5e-06 + 500 * 3e-05 = 0.02`, because a custom priced entry inherits every `_priority` / `_flex`
    key it does not set (`_inherited_service_tier_pricing`, `registry.py:315`). Pinned because it is
    surprising: a priority call can cost less than a standard one on a deployment with high custom prices."""
    answer: Final = openai_reply(openai_usage(1000, 500), model=GPT54, service_tier="priority")
    with openai(gateway, answer, backend=GPT54, input_cost_per_token=1e-05, output_cost_per_token=2e-05) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.02)


# ---------------------------------------------------------------- Azure OpenAI


def _azure_answer(usage: dict, *, model: str = "gpt-5.4-2026-03-05", service_tier: str | None = None) -> Reply:  # pyright: ignore[reportMissingTypeArgument]
    return openai_reply(usage, model=model, service_tier=service_tier)


def test_azure_base_model_prices_a_customer_named_deployment(gateway: Gateway) -> None:
    """`azure/dec1-costacc-prod` with `model_info.base_model: azure/gpt-5.4` (line 8243) reports 1000 prompt and
    500 completion tokens: `0.01`."""
    with azure(
        gateway,
        _always(_azure_answer(openai_usage(1000, 500))),
        model=AZURE_DEPLOYMENT,
        model_info={"base_model": "azure/gpt-5.4"},
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01)


def test_azure_base_model_wins_over_the_model_the_answer_names(gateway: Gateway) -> None:
    """The answer says `model: gpt-4o-mini` (a much cheaper row) but `base_model` is `azure/gpt-5.4`: the bill is
    `0.01`, not `1000 * 1.5e-07 + 500 * 6e-07 = 0.00045` (`cost_calculator.py:1099`)."""
    with azure(
        gateway,
        _always(_azure_answer(openai_usage(1000, 500), model="gpt-4o-mini")),
        model=AZURE_DEPLOYMENT,
        model_info={"base_model": "azure/gpt-5.4"},
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01)


def test_azure_without_base_model_prices_the_model_the_answer_names(gateway: Gateway) -> None:
    """No `base_model`; the answer names `gpt-5.4-2026-03-05`, which has row `azure/gpt-5.4-2026-03-05`
    (line 8404, same prices as gpt-5.4). The deployment name has no row, so the served model prices it: `0.01`."""
    with azure(gateway, _always(_azure_answer(openai_usage(1000, 500))), model=AZURE_DEPLOYMENT) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01)


@pytest.mark.parametrize(
    "zone",
    [pytest.param("eu", id="eu-line-8353"), pytest.param("us", id="us-line-8302")],
)
def test_azure_data_zone_base_model_bills_the_zone_premium(gateway: Gateway, zone: str) -> None:
    """`base_model: azure/<zone>/gpt-5.4` (2.75e-06 in, 1.65e-05 out) with 1000 prompt and 500 completion:
    `0.00275 + 0.00825 = 0.011`, 10 percent above the global deployment row."""
    with azure(
        gateway,
        _always(_azure_answer(openai_usage(1000, 500))),
        model=AZURE_DEPLOYMENT,
        model_info={"base_model": f"azure/{zone}/gpt-5.4"},
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.011)


def test_azure_cached_tokens_bill_at_the_cache_read_rate(gateway: Gateway) -> None:
    """base_model azure/gpt-5.4 with 600 of 1000 prompt tokens cached: `0.00865`."""
    with azure(
        gateway,
        _always(_azure_answer(openai_usage(1000, 500, cached=600))),
        model=AZURE_DEPLOYMENT,
        model_info={"base_model": "azure/gpt-5.4"},
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.00865)


def test_azure_served_priority_bills_the_priority_rates(gateway: Gateway) -> None:
    """base_model azure/gpt-5.4, answer `service_tier: priority`: `1000 * 5e-06 + 500 * 3e-05 = 0.02`."""
    answer: Final = _azure_answer(openai_usage(1000, 500), service_tier="priority")
    with azure(gateway, _always(answer), model=AZURE_DEPLOYMENT, model_info={"base_model": "azure/gpt-5.4"}) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 0.02)


def test_azure_priority_above_272k_bills_the_priority_long_context_rates(gateway: Gateway) -> None:
    """The Azure row (line 8243) has `input_cost_per_token_above_272k_tokens_priority` 1e-05 and the output
    counterpart 4.5e-05, so 300000 prompt and 500 completion at priority cost
    `300000 * 1e-05 + 500 * 4.5e-05 = 3.0225`, twice the OpenAI row's answer for the same usage."""
    answer: Final = _azure_answer(openai_usage(300000, 500), service_tier="priority")
    with azure(gateway, _always(answer), model=AZURE_DEPLOYMENT, model_info={"base_model": "azure/gpt-5.4"}) as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 3.0225)


def test_azure_audio_prompt_tokens_bill_at_the_azure_audio_rate(gateway: Gateway) -> None:
    """base_model `azure/gpt-audio` (line 6402: audio in 4e-05, higher than OpenAI's 3.2e-05) with 1000 prompt
    tokens of which 400 audio and 100 completion: `600 * 2.5e-06 + 400 * 4e-05 + 100 * 1e-05 = 0.0185`."""
    answer: Final = _azure_answer(openai_usage(1000, 100, audio_in=400), model="gpt-audio")
    with azure(gateway, _always(answer), model=AZURE_DEPLOYMENT, model_info={"base_model": "azure/gpt-audio"}) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0185)


def test_azure_deployment_prices_win_over_base_model(gateway: Gateway) -> None:
    """A deployment with `base_model: azure/gpt-5.4` and `input_cost_per_token: 1e-05`, `output_cost_per_token:
    2e-05` bills `1000 * 1e-05 + 500 * 2e-05 = 0.02`: custom pricing is checked first (`cost_calculator.py:1090`)."""
    with azure(
        gateway,
        _always(_azure_answer(openai_usage(1000, 500))),
        model=AZURE_DEPLOYMENT,
        model_info={"base_model": "azure/gpt-5.4"},
        input_cost_per_token=1e-05,
        output_cost_per_token=2e-05,
    ) as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.02)


# ---------------------------------------------------------------- Azure AI Foundry


def test_foundry_gpt_5_4_bills_the_azure_ai_row(gateway: Gateway) -> None:
    """`azure_ai/gpt-5.4` (line 4874) with 1000 prompt and 500 completion tokens: `0.01`."""
    with foundry(gateway, _always(openai_reply(openai_usage(1000, 500), model=GPT54)), model="azure_ai/gpt-5.4") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.01)


def test_foundry_served_priority_above_272k_bills_the_priority_long_rates(gateway: Gateway) -> None:
    """azure_ai/gpt-5.4 served at priority with 300000 prompt tokens: `300000 * 1e-05 + 500 * 4.5e-05 = 3.0225`."""
    answer: Final = openai_reply(openai_usage(300000, 500), model=GPT54, service_tier="priority")
    with foundry(gateway, _always(answer), model="azure_ai/gpt-5.4") as rig:
        assert_billed(gateway, rig.model, rig.chat(service_tier="priority"), 3.0225)


def test_foundry_claude_one_hour_cache_writes_bill_at_the_one_hour_rate(gateway: Gateway) -> None:
    """`azure_ai/claude-sonnet-4-5` (line 4243: 3e-06 in, 3e-07 read, 3.75e-06 5m write, 6e-06 1h write,
    1.5e-05 out) answers an Anthropic message with 1000 input, 4000 cache read, 2000 one hour writes and 500
    output: `0.003 + 0.0012 + 0.012 + 0.0075 = 0.0237`."""
    usage: Final = anthropic_usage(1000, 500, cache_read=4000, cache_write=2000, write_5m=0, write_1h=2000)
    answer: Final = json_reply(anthropic_message(usage, model="claude-sonnet-4-5"))
    with foundry(gateway, _always(answer), model="azure_ai/claude-sonnet-4-5") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0237)


def test_foundry_claude_web_search_adds_the_per_search_fee(gateway: Gateway) -> None:
    """azure_ai/claude-sonnet-4-5 (`search_context_size_medium: 0.01`) reports 2 web search requests on
    1000 input and 500 output: `0.0105 + 0.02 = 0.0305`."""
    usage: Final = anthropic_usage(1000, 500, server_tool_use={"web_search_requests": 2})
    answer: Final = json_reply(anthropic_message(usage, model="claude-sonnet-4-5"))
    with foundry(gateway, _always(answer), model="azure_ai/claude-sonnet-4-5") as rig:
        assert_billed(gateway, rig.model, rig.chat(), 0.0305)


# ---------------------------------------------------------------- batch discount


def _batch_spend(gateway: Gateway, model_param: str, served_model: str, **pricing: object) -> float:
    """Run one model routed OpenAI batch with two succeeded lines (100+20 and 50+10 tokens) and one 400 line
    against an owned peer, and return the spend of the creating key's single `aretrieve_batch` row."""
    stem: Final = unique("costacc-batch")
    batch_id: Final = f"batch_{stem}"
    output: Final = jsonl(
        openai_output_line("r1", served_model, 100, 20),
        openai_output_line("r2", served_model, 50, 10),
        openai_output_line("r3", served_model, 0, 0, status=400),
    )
    script: Final = Script()
    script.on("POST", "/v1/files", reply(openai_file(f"file-in-{stem}")))
    script.on("POST", "/v1/batches", reply(openai_batch(batch_id, "validating")))
    script.on(
        "GET",
        f"/v1/batches/{batch_id}",
        reply(
            openai_batch(
                batch_id,
                "completed",
                output_file_id=f"file-out-{stem}",
                completed_at=1763377200,
                request_counts={"total": 3, "completed": 2, "failed": 1},
            )
        ),
    )
    script.on("GET", f"/v1/files/file-out-{stem}/content", text_reply(output.decode()))
    with gateway.scenario() as scenario, wire_server(script) as wire:
        model: Final = scenario.model(
            model=model_param,
            api_key="sk-dec1-costacc",
            api_base=f"{wire.url}/v1",
            **pricing,  # pyright: ignore[reportArgumentType]
        )
        key: Final = scenario.key(models=[model])
        uploaded: Final = gateway.request_multipart(
            "/v1/files",
            {"purpose": "batch", "model": model},
            {"file": ("in.jsonl", jsonl(*(chat_row(f"r{i}", model) for i in (1, 2, 3))), "application/jsonl")},
            key=key,
        )
        assert uploaded.status_code == 200, uploaded.text
        created: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": json_of(uploaded)["id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model,
            },
            key=key,
        )
        assert created.status_code == 200, created.text
        retrieved: Final = gateway.request("GET", f"/v1/batches/{json_of(created)['id']}", key=key)
        assert retrieved.status_code == 200 and json_of(retrieved)["status"] == "completed", retrieved.text
        rows: Final = batch_spend_rows(gateway, key)
        assert len(rows) == 1, rows
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (150, 30)
        return float(str(rows[0]["spend"]))


def test_openai_batch_bills_the_row_batch_rates(gateway: Gateway) -> None:
    """A completed gpt-5.4 batch with 150 prompt and 30 completion tokens over two succeeded lines bills the
    row's `input_cost_per_token_batches` 1.25e-06 and `output_cost_per_token_batches` 7.5e-06 (line 39001):
    `150 * 1.25e-06 + 30 * 7.5e-06 = 0.0004125`, half the standard price."""
    assert _batch_spend(gateway, "openai/gpt-5.4", GPT54) == pytest.approx(0.0004125, rel=1e-6)


def test_openai_batch_on_gpt_4o_mini_bills_its_batch_rates(gateway: Gateway) -> None:
    """The same batch on gpt-4o-mini (line 36308: batches 7.5e-08 in, 3e-07 out):
    `150 * 7.5e-08 + 30 * 3e-07 = 0.00002025`."""
    assert _batch_spend(gateway, "openai/gpt-4o-mini", "gpt-4o-mini") == pytest.approx(0.00002025, rel=1e-6)


def test_openai_batch_with_flat_deployment_prices_bills_half_of_them(gateway: Gateway) -> None:
    """A gpt-5.4 deployment priced `input_cost_per_token: 1e-05`, `output_cost_per_token: 2e-05` and no batch
    prices. `batch_cost_calculator` (`cost_calculator.py:3306`) sees the deployment entry with flat rates
    and no `*_batches` keys and halves them: `150 * 5e-06 + 30 * 1e-05 = 0.00105`.
    UNVERIFIED: assumes the deployment entry does not inherit the row's `*_batches` keys."""
    spend: Final = _batch_spend(
        gateway, "openai/gpt-5.4", GPT54, input_cost_per_token=1e-05, output_cost_per_token=2e-05
    )
    assert spend == pytest.approx(0.00105, rel=1e-6)
