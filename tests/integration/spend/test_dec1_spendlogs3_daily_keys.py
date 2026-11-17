"""Contract: how the ledger projection keys the daily spend tables, ahead of the Spend Logs V2 port.

Rust stops at the committed `LiteLLM_SpendLogs` row and its `LiteLLM_SpendLogDelivery`; the Python ledger worker
projects every daily table from that row (plan, "DB write and batching path"). So the row the Rust builder writes
must carry exactly the columns the projection keys on, and the projection's own rules must survive the rollup
move. Rounds 1 and 2 pin the user, end user, organization and tag daily rows of one call. This file pins:

1. `LiteLLM_DailyTeamSpend` keying: one row per (team, day, hashed key, model, provider, tool, endpoint), the
   failure counters, and `ptu_flat_cost` staying 0 for a token priced call (`_daily_contributions`,
   litellm/accounting/projection.py:170-231; unique index in schema.prisma).
2. The 2000 byte guard: a contribution whose key or value text is longer than 2000 bytes is dropped whole
   (`_indexable`, projection.py:248, `_MAX_INDEXED_TEXT_BYTES`), so an oversized tag gets no daily tag row and no
   tag spend while the call itself is still billed.
3. Duplicate request tags: the daily tag rows de-duplicate tags (`daily_spend_entity_ids`,
   litellm/proxy/db/daily_spend_bulk_upsert.py:90) but the `LiteLLM_TagTable` contributions do not
   (`_entity_contributions`, projection.py:128) and contributions to one destination are summed
   (`project_deliveries`, litellm/accounting/delivery_projection.py:43).
4. `LiteLLM_DailyGlobalSpend`, the key-free rollup the schema describes, which nothing writes today.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.wire import wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    EMBEDDING_COST,
    ask,
    chat_model,
    embedding_model,
    failing,
    hashed,
    new_tag,
    number,
    row_by_id,
    scripted_upstream,
    served,
    served_id,
    spend_reaches,
)
from tests.integration.spend.dec1_spendlogs2_support import wait_for_key_spend
from tests.integration.spend.dec1_spendlogs3_support import (
    counters,
    daily_table_rows,
    stored_json,
    utc_today,
    wait_rows,
)

pytestmark: Final = pytest.mark.shares_worker_state

MAX_INDEXED_BYTES: Final = 2000
"""`_MAX_INDEXED_TEXT_BYTES`, litellm/accounting/projection.py:30."""


def _embed(gateway: Gateway, model: str, key: str) -> None:
    response: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": "dec1 sl3 embed"}, key=key)
    assert response.status_code == 200, response.text


def test_a_team_key_on_two_endpoints_gets_two_daily_team_rows(gateway: Gateway) -> None:
    """A team key makes one chat call and one embeddings call. `LiteLLM_DailyTeamSpend` holds two rows for the
    team, one per `endpoint` (`/chat/completions` with spend 0.02, `/embeddings` with spend 0.007), both with the
    key's hash in `api_key` and today's UTC `date`: `endpoint` is part of the unique key."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        chat: Final = chat_model(scenario, wire)
        embed: Final = embedding_model(scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(models=[chat, embed], team_id=team)
        served(ask(gateway, chat, key))
        _embed(gateway, embed, key)
        rows: Final = daily_table_rows(gateway, "LiteLLM_DailyTeamSpend", "team_id", team, 2)
        assert [(row["endpoint"], row["api_key"], row["date"]) for row in rows] == [
            ("/chat/completions", hashed(key), utc_today()),
            ("/embeddings", hashed(key), utc_today()),
        ], rows
        assert [number(row["spend"]) for row in rows] == [pytest.approx(CHAT_COST), pytest.approx(EMBEDDING_COST)]


def test_two_keys_of_one_team_get_separate_daily_team_rows(gateway: Gateway) -> None:
    """Two keys of the same team each make one chat call on the same model. The team has two daily rows, one per
    hashed key, each with `api_requests` 1 and spend 0.02: `api_key` is part of the team table's unique key, so
    per key breakdowns of a team survive the projection."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        first: Final = scenario.key(models=[model], team_id=team)
        second: Final = scenario.key(models=[model], team_id=team)
        served(ask(gateway, model, first))
        served(ask(gateway, model, second))
        rows: Final = daily_table_rows(gateway, "LiteLLM_DailyTeamSpend", "team_id", team, 2)
        assert sorted(string_value(row["api_key"]) for row in rows) == sorted((hashed(first), hashed(second)))
        assert all(counters(row, "api_requests") == (1,) for row in rows), rows


def test_a_daily_team_row_keeps_model_provider_and_group(gateway: Gateway) -> None:
    """The team's daily row for a chat call has `model` `openai/gpt-4o-mini` (the request row's `model`, which is
    the deployment's `litellm_params.model`, projection.py:276), `custom_llm_provider`
    `openai`, `model_group` the public name, an empty `mcp_namespaced_tool_name` and token counters 10 and 5
    (`shared_key` and `model_group` value, projection.py:178-215; NULL key columns are normalized to '')."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(models=[model], team_id=team)
        served(ask(gateway, model, key))
        row: Final = daily_table_rows(gateway, "LiteLLM_DailyTeamSpend", "team_id", team, 1)[0]
        assert (row["model"], row["custom_llm_provider"], row["model_group"]) == (
            "openai/gpt-4o-mini",
            "openai",
            model,
        ), row
        assert row["mcp_namespaced_tool_name"] == "", row
        assert counters(row, "prompt_tokens", "completion_tokens") == (10, 5), row


def test_a_token_priced_call_leaves_ptu_flat_cost_at_zero(gateway: Gateway) -> None:
    """`ptu_flat_cost` is a team table only column that the projection never increments for a token priced
    deployment: it stays 0 while `spend` is 0.02 (the column is absent from `_daily_counters`,
    projection.py:141). UNVERIFIED: no PTU share applies to a deployment without `ptu_*` settings."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(models=[model], team_id=team)
        served(ask(gateway, model, key))
        row: Final = daily_table_rows(gateway, "LiteLLM_DailyTeamSpend", "team_id", team, 1)[0]
        assert (number(row["ptu_flat_cost"]), number(row["spend"])) == (0.0, pytest.approx(CHAT_COST)), row


def test_a_failed_team_call_counts_as_a_failed_request_in_the_team_daily_row(gateway: Gateway) -> None:
    """A team key's provider failure leaves a team daily row with `api_requests` 1, `failed_requests` 1,
    `successful_requests` 0, `timed_requests` 0 and spend 0: the team table uses the same counters as the user
    table (`_daily_counters`, projection.py:141-165)."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(models=[model], team_id=team)
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        row: Final = daily_table_rows(gateway, "LiteLLM_DailyTeamSpend", "team_id", team, 1)[0]
        assert counters(row, "api_requests", "failed_requests", "successful_requests", "timed_requests") == (
            1,
            1,
            0,
            0,
        ), row
        assert number(row["spend"]) == 0.0, row


def test_a_tag_longer_than_2000_bytes_gets_no_daily_tag_row_but_the_call_is_billed(gateway: Gateway) -> None:
    """A call tagged with a 2001 character tag and a short tag: the short tag gets its daily tag row, the long
    one gets none (its contribution fails `_indexable`, projection.py:248), and the key is still charged 0.02.
    The request row keeps the long tag in `request_tags`, since the guard only applies to the projection."""
    long_tag: Final = "t" * (MAX_INDEXED_BYTES + 1)
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        short_tag: Final = new_tag(gateway, scenario)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(gateway, model, key, metadata={"tags": [short_tag, long_tag]}))
        daily_table_rows(gateway, "LiteLLM_DailyTagSpend", "tag", short_tag, 1)
        assert wait_for_key_spend(gateway, key, CHAT_COST) == pytest.approx(CHAT_COST)
        assert (
            gateway.rows(
                'SELECT tag FROM "LiteLLM_DailyTagSpend" WHERE api_key = %s AND tag = %s',
                (
                    hashed(key),
                    long_tag,
                ),
            )
            == []
        )
        tags: Final = stored_json(row_by_id(gateway, request_id)["request_tags"])
        assert isinstance(tags, list) and long_tag in tags, tags


def test_a_tag_of_exactly_2000_bytes_still_gets_a_daily_tag_row(gateway: Gateway) -> None:
    """The guard is "longer than 2000 bytes": a 2000 character ASCII tag is projected, so it has one daily tag
    row for the key with `api_requests` 1 (`len(text.encode()) <= _MAX_INDEXED_TEXT_BYTES`, projection.py:250).
    The key also gets the proxy's own `User-Agent: ...` tag rows, so only the row for this tag is counted."""
    edge_tag: Final = "e" * MAX_INDEXED_BYTES
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": [edge_tag]}))
        rows: Final = wait_rows(
            gateway,
            'SELECT tag, api_requests FROM "LiteLLM_DailyTagSpend" WHERE api_key = %s AND tag = %s',
            (hashed(key), edge_tag),
            1,
        )
        assert rows[0]["api_requests"] == 1, rows


def test_a_multibyte_tag_is_measured_in_bytes_not_characters(gateway: Gateway) -> None:
    """A tag of 1001 two-byte characters (`é`, 2002 UTF-8 bytes, but only 1001 characters) is dropped from the
    projection like a 2001 byte tag: the guard counts encoded bytes (`text.encode("utf-8", "surrogatepass")`,
    projection.py:250). A Rust port that counts characters would start projecting it."""
    wide_tag: Final = "é" * 1001
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        anchor: Final = new_tag(gateway, scenario)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": [anchor, wide_tag]}))
        daily_table_rows(gateway, "LiteLLM_DailyTagSpend", "tag", anchor, 1)
        assert (
            gateway.rows(
                'SELECT tag FROM "LiteLLM_DailyTagSpend" WHERE api_key = %s AND tag = %s',
                (
                    hashed(key),
                    wide_tag,
                ),
            )
            == []
        )


def test_a_duplicated_request_tag_writes_one_daily_tag_row_with_one_request(gateway: Gateway) -> None:
    """A call sent with `metadata.tags` `[t, t]` leaves exactly one daily tag row for `t` and the key, with
    `api_requests` 1 and spend 0.02: daily tag identities are de-duplicated (`dict.fromkeys`,
    daily_spend_bulk_upsert.py:90)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        tag: Final = new_tag(gateway, scenario)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": [tag, tag]}))
        rows: Final = daily_table_rows(gateway, "LiteLLM_DailyTagSpend", "tag", tag, 1)
        assert (counters(rows[0], "api_requests"), number(rows[0]["spend"])) == ((1,), pytest.approx(CHAT_COST))


def test_a_duplicated_request_tag_charges_the_tag_once(gateway: Gateway) -> None:
    """The same `[t, t]` call must add 0.02 to `LiteLLM_TagTable.spend` of `t`, the same as its daily row.
    Today the entity contribution is built per listed tag without de-duplication (projection.py:128) and the
    contributions to one tag are summed (delivery_projection.py:43), so the tag is charged 0.04 if the row keeps
    both copies. UNVERIFIED: `merge_tags` (litellm/proxy/litellm_pre_call_utils.py:2538) keeps the request's own
    duplicates, so the row does keep both."""
    pytest.skip("BUG: a tag listed twice on one request is charged twice in LiteLLM_TagTable but once daily")
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        tag: Final = new_tag(gateway, scenario)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": [tag, tag]}))
        assert spend_reaches(gateway, "tag", tag, CHAT_COST) == pytest.approx(CHAT_COST)


def test_the_key_free_daily_global_rollup_is_written(gateway: Gateway) -> None:
    """The schema describes `LiteLLM_DailyGlobalSpend` as the "key-free daily rollup of LiteLLM_DailyUserSpend,
    read by the global usage view", keyed by (date, model, model_group, provider, tool, endpoint). A served call on
    a fresh model group should add a row for that group with `api_requests` 1 and spend 0.02. Today no Python,
    SQL or Rust code on the branch writes or reads the table (only migrations and an OTel table list mention it),
    so it stays empty."""
    pytest.skip("BUG: nothing writes LiteLLM_DailyGlobalSpend, the schema's key-free daily rollup")
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        rows: Final = daily_table_rows(gateway, "LiteLLM_DailyGlobalSpend", "model_group", model, 1)
        assert (counters(rows[0], "api_requests"), number(rows[0]["spend"])) == ((1,), pytest.approx(CHAT_COST))
