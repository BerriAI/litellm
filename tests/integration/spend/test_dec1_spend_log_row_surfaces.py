"""Contract: the `LiteLLM_SpendLogs` row for every inference surface, streaming or not.

The chat row is pinned column by column in `test_dec1_spend_log_row_chat.py`. This file pins what changes from
one surface to the next (call type, provider, token counts, spend) and what must not change (who is billed, the
public model name, the status, one row per request). The surfaces are chat completions, embeddings, Anthropic
`/v1/messages`, OpenAI `/v1/responses` and legacy text completions. `/v1/messages` and `/v1/responses` are the
first endpoints moving to Rust, and each has its own usage shape (`input_tokens` / `output_tokens`), so the
mapping into `prompt_tokens` / `completion_tokens` is where a rewrite is most likely to drift.

The scripted provider reports fixed usage per surface and every deployment costs 0.001 per input token and 0.002
per output token. Expected spends: chat 0.02, embeddings 0.007, messages 0.018, responses 0.017, text 0.01.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration.spend.dec1_spend_support import (
    CHAT,
    EMBEDDINGS,
    MESSAGES,
    RESPONSES,
    STREAMING_SURFACES,
    SURFACES,
    TEXT,
    Surface,
    hashed,
    metadata_of,
    number,
    provider_calls,
    rows_of_key,
    scripted_upstream,
    served,
    stream,
    stream_chunks,
)

_IDS: Final = tuple(surface.name for surface in SURFACES)
_STREAM_IDS: Final = tuple(surface.name for surface in STREAMING_SURFACES)


@pytest.mark.parametrize("surface", SURFACES, ids=_IDS)
def test_each_surface_writes_one_row_with_its_call_type_tokens_and_spend(gateway: Gateway, surface: Surface) -> None:
    """One non-streaming call per surface. The single row must carry that surface's `call_type` (`acompletion`,
    `aembedding`, `anthropic_messages`, `aresponses`, `atext_completion`), the provider's token counts mapped to
    prompt/completion/total, and spend equal to tokens times the deployment price.
    UNVERIFIED: `anthropic_messages` and `atext_completion` spellings are read from `CallTypes`, not from a run."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = surface.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(gateway.request("POST", surface.path, surface.body(model), key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert row["call_type"] == surface.call_type, row
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (
            surface.prompt_tokens,
            surface.completion_tokens,
            surface.prompt_tokens + surface.completion_tokens,
        ), row
        assert number(row["spend"]) == pytest.approx(surface.cost), row
        assert len(provider_calls(wire)) == 1


@pytest.mark.parametrize("surface", SURFACES, ids=_IDS)
def test_each_surface_attributes_the_row_to_the_same_key_user_team_and_model_group(
    gateway: Gateway, surface: Surface
) -> None:
    """A team key owned by a user calls each surface once. Whatever the surface, the row names the hashed key,
    the owner in `user`, the team in `team_id`, the public model name in `model_group`, the deployment's provider
    in `custom_llm_provider` and `status` `success`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = surface.model(scenario, wire)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        served(gateway.request("POST", surface.path, surface.body(model), key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["api_key"], row["user"], row["team_id"]) == (hashed(key), user, team), row
        assert (row["model_group"], row["custom_llm_provider"], row["status"]) == (
            model,
            surface.provider,
            "success",
        ), row


@pytest.mark.parametrize("surface", SURFACES, ids=_IDS)
def test_each_surface_reports_the_row_spend_in_the_response_cost_header(gateway: Gateway, surface: Surface) -> None:
    """On every non-streaming surface the `x-litellm-response-cost` response header must equal the `spend` the
    row ends up with. A surface that computes cost in the handler but logs a different number fails here."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = surface.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", surface.path, surface.body(model), key=key)
        served(response)
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(surface.cost), response.headers
        assert number(row["spend"]) == pytest.approx(float(response.headers["x-litellm-response-cost"])), row


@pytest.mark.parametrize("surface", STREAMING_SURFACES, ids=_STREAM_IDS)
def test_each_streaming_surface_writes_one_row_with_the_final_usage_and_spend(
    gateway: Gateway, surface: Surface
) -> None:
    """The same call with `stream: true`. Once the caller has read the stream to its end, exactly one row exists
    with the same call type, tokens and spend as the non-streaming call. Usage arrives in the last provider
    event, so a writer that logs when the first chunk is sent stores zero tokens and fails here."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = surface.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        lines: Final = stream(gateway, surface.path, surface.body(model, streaming=True), key)
        assert any(line.startswith("data:") for line in lines), lines
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert row["call_type"] == surface.call_type, row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (
            surface.prompt_tokens,
            surface.completion_tokens,
        ), row
        assert number(row["spend"]) == pytest.approx(surface.cost), row
        assert row["status"] == "success", row


def test_streamed_chat_row_request_id_is_the_id_on_every_chunk(gateway: Gateway) -> None:
    """A streamed chat completion sends several chunks that all share one `id`. The row's `request_id` must be
    that id, the same rule as for a non-streamed response."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = CHAT.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        chunks: Final = stream_chunks(stream(gateway, CHAT.path, CHAT.body(model, streaming=True), key))
        ids: Final = {string_value(chunk["id"]) for chunk in chunks}
        assert len(ids) == 1, chunks
        assert rows_of_key(gateway, key, 1)[0]["request_id"] in ids


def test_streamed_chat_is_billed_even_when_the_caller_did_not_ask_for_usage(gateway: Gateway) -> None:
    """The caller streams without `stream_options.include_usage`. The provider still ends its stream with a usage
    chunk, and the row must be billed from it: 10 prompt tokens, 5 completion tokens, spend 0.02."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = CHAT.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        body: Final = CHAT.body(model, streaming=True)
        assert "stream_options" not in body
        stream(gateway, CHAT.path, body, key)
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (10, 5), row
        assert number(row["spend"]) == pytest.approx(10 * 0.001 + 5 * 0.002), row


def test_streamed_chat_usage_chunk_sent_to_the_caller_matches_the_row(gateway: Gateway) -> None:
    """With `stream_options.include_usage` the caller receives a final chunk with a `usage` object. Its prompt
    and completion token counts must equal the row's token columns, so client-side and server-side metering
    agree."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = CHAT.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        body: Final = {**CHAT.body(model, streaming=True), "stream_options": {"include_usage": True}}
        chunks: Final = stream_chunks(stream(gateway, CHAT.path, body, key))
        usages: Final = [object_value(chunk["usage"]) for chunk in chunks if chunk.get("usage")]
        assert len(usages) == 1, chunks
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (usages[0]["prompt_tokens"], usages[0]["completion_tokens"]) == (
            row["prompt_tokens"],
            row["completion_tokens"],
        ), (usages, row)


def test_embeddings_row_has_no_completion_tokens_and_bills_input_only(gateway: Gateway) -> None:
    """An embeddings call reports 7 prompt tokens and no completion tokens. The row stores 7 / 0 / 7 and spend
    7 * 0.001 = 0.007, and the deployment's model `openai/text-embedding-3-small` in `model`, provider prefix
    included as on every request row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = EMBEDDINGS.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(gateway.request("POST", EMBEDDINGS.path, EMBEDDINGS.body(model), key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (7, 0, 7), row
        assert number(row["spend"]) == pytest.approx(7 * 0.001), row
        assert row["model"] == "openai/text-embedding-3-small", row


def test_messages_row_maps_anthropic_input_and_output_tokens(gateway: Gateway) -> None:
    """Anthropic reports `input_tokens: 12` and `output_tokens: 3`. The `/v1/messages` row stores them as
    prompt_tokens 12, completion_tokens 3, total 15, spend 12 * 0.001 + 3 * 0.002 = 0.018, and the row's
    `request_id` is the `msg_...` id the caller received."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = MESSAGES.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        body: Final = served(gateway.request("POST", MESSAGES.path, MESSAGES.body(model), key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (12, 3, 15), row
        assert number(row["spend"]) == pytest.approx(12 * 0.001 + 3 * 0.002), row
        assert row["request_id"] == body["id"], (row, body)


def test_messages_response_usage_returned_to_the_caller_matches_the_row(gateway: Gateway) -> None:
    """The `/v1/messages` response body carries Anthropic-shaped `usage`. Its `input_tokens` and `output_tokens`
    must equal the row's prompt and completion tokens."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = MESSAGES.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        usage: Final = object_value(
            served(gateway.request("POST", MESSAGES.path, MESSAGES.body(model), key=key))["usage"]
        )
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (usage["input_tokens"], usage["output_tokens"]) == (row["prompt_tokens"], row["completion_tokens"])


def test_responses_row_maps_input_and_output_tokens(gateway: Gateway) -> None:
    """The Responses API reports `input_tokens: 9` and `output_tokens: 4`. The row stores prompt_tokens 9,
    completion_tokens 4, total 13 and spend 9 * 0.001 + 4 * 0.002 = 0.017, with provider `openai`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = RESPONSES.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(gateway.request("POST", RESPONSES.path, RESPONSES.body(model), key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (9, 4, 13), row
        assert number(row["spend"]) == pytest.approx(9 * 0.001 + 4 * 0.002), row
        assert row["custom_llm_provider"] == "openai", row


def test_text_completion_row_is_billed_from_the_provider_usage(gateway: Gateway) -> None:
    """A legacy `/v1/completions` call reports 6 prompt and 2 completion tokens. The row stores 6 / 2 / 8 and
    spend 6 * 0.001 + 2 * 0.002 = 0.01. Assumes a `text-completion-openai/` deployment posts to `/completions`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = TEXT.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(gateway.request("POST", TEXT.path, TEXT.body(model), key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (6, 2, 8), row
        assert number(row["spend"]) == pytest.approx(6 * 0.001 + 2 * 0.002), row


@pytest.mark.parametrize("surface", SURFACES, ids=_IDS)
def test_each_surface_stores_the_usage_object_in_row_metadata(gateway: Gateway, surface: Surface) -> None:
    """Daily rollups read cache and reasoning token detail from `metadata.usage_object`, not from the token
    columns. On every surface that object must exist and agree with the row's prompt and completion tokens."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = surface.model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(gateway.request("POST", surface.path, surface.body(model), key=key))
        row: Final = rows_of_key(gateway, key, 1)[0]
        usage: Final = object_value(metadata_of(row)["usage_object"])
        assert usage["prompt_tokens"] == row["prompt_tokens"] == surface.prompt_tokens, usage
        assert (usage.get("completion_tokens") or 0) == surface.completion_tokens, usage
