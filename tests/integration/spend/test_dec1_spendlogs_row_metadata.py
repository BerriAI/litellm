"""Contract: the `metadata` JSON and usage detail columns of a `LiteLLM_SpendLogs` row.

Spend Logs V2 moves the row builder (`get_logging_payload`, `litellm/proxy/spend_tracking/spend_tracking_utils.py`)
into the Rust `callbacks-spend-log` crate and plans to read the long tail of the row from the standard logging
payload by reference. The `metadata` column is that long tail: the key identity the proxy stamped
(`user_api_key_*`), the caller's user agent, the provider usage object, the prompt caching counters derived from
it (`additional_usage_values`), the cost breakdown and the model map entry. The usage UI, the daily tables
(cache token counters are projected from `metadata.usage_object`) and customer exports all read these keys by
name, so a builder that renames, drops or re-derives one of them is a visible regression even when `spend` is
right.

The scripted provider answers chat with 10 prompt and 5 completion tokens at 0.001 / 0.002 per token (0.02).
`caching_upstream` additionally reports 4 cached prompt tokens on chat, and 2 cache-write plus 3 cache-read
tokens on Anthropic `/v1/messages`.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration.spend.dec1_spend_support import (
    ask,
    chat_model,
    daily_row,
    messages_model,
    metadata_of,
    number,
    row_by_id,
    scripted_upstream,
    served,
    served_id,
)
from tests.integration.spend.dec1_spendlogs_support import (
    CACHE_READ_TOKENS,
    CACHE_WRITE_TOKENS,
    CACHED_PROMPT_TOKENS,
    caching_upstream,
    key_rows,
    owned_spend_proxy,
)

pytestmark: Final = pytest.mark.shares_worker_state


def test_metadata_names_the_key_alias_the_caller_used(gateway: Gateway) -> None:
    """A key is created with a random `key_alias` and makes one chat call. The row's `metadata.user_api_key_alias`
    must equal that alias exactly; `/spend/logs/v2?key_alias=` and the usage UI filter on this JSON path."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        alias: Final = f"dec1-sl-alias-{uuid.uuid4().hex[:8]}"
        key: Final = scenario.key(models=[model], key_alias=alias)
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert metadata_of(row)["user_api_key_alias"] == alias, row


def test_metadata_names_the_team_id_and_team_alias_of_a_team_key(gateway: Gateway) -> None:
    """A team with a `team_alias` owns a key that makes one call. The row's `team_id` column and
    `metadata.user_api_key_team_id` both equal the team id, and `metadata.user_api_key_team_alias` equals the
    alias, since `/global/spend/teams` and the team usage page label spend with that alias."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        alias: Final = f"dec1-sl-team-{uuid.uuid4().hex[:8]}"
        team: Final = scenario.team(team_alias=alias, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        metadata: Final = metadata_of(row)
        assert row["team_id"] == team, row
        assert metadata["user_api_key_team_id"] == team, metadata
        assert metadata["user_api_key_team_alias"] == alias, metadata


def test_metadata_names_the_organization_of_a_team_key(gateway: Gateway) -> None:
    """A key belongs to a team inside an organization. The row's `organization_id` column and
    `metadata.user_api_key_org_id` both equal the organization id; the ledger projection charges the
    organization from these, so a missing org id silently stops organization spend."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        org: Final = scenario.organization(models=[model])
        team: Final = scenario.team(organization_id=org, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert row["organization_id"] == org, row
        assert metadata_of(row)["user_api_key_org_id"] == org, row


def test_metadata_carries_the_user_id_and_email_of_the_keys_owner(gateway: Gateway) -> None:
    """A user created with `user_email` owns a key that makes one call. The row's `user` column and
    `metadata.user_api_key_user_id` equal the user id and `metadata.user_api_key_user_email` equals the email
    (`_stored_user_email`, spend_tracking_utils.py:234)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        email: Final = f"dec1-sl-{uuid.uuid4().hex[:8]}@example.com"
        user: Final = scenario.user(user_email=email, models=[model])
        key: Final = scenario.key(user_id=user, models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        metadata: Final = metadata_of(row)
        assert row["user"] == user, row
        assert metadata["user_api_key_user_id"] == user, metadata
        assert metadata["user_api_key_user_email"] == email, metadata


def test_metadata_user_email_is_null_for_a_user_without_an_email(gateway: Gateway) -> None:
    """A user without `user_email` owns a key that makes one call. `metadata.user_api_key_user_email` is JSON
    null, never an empty string: `_stored_user_email` maps blank values to None (spend_tracking_utils.py:234)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(models=[model])
        key: Final = scenario.key(user_id=user, models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert metadata_of(row).get("user_api_key_user_email") is None, row


def test_metadata_records_the_callers_user_agent(gateway: Gateway) -> None:
    """The caller sends `User-Agent: dec1-spendlogs-agent/1.0`. The row's `metadata.user_agent` must be that
    exact string (stamped at `litellm_pre_call_utils.py:3335`), which the usage UI groups client traffic by."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key, headers={"User-Agent": "dec1-spendlogs-agent/1.0"})
        row: Final = row_by_id(gateway, served_id(response))
        assert metadata_of(row)["user_agent"] == "dec1-spendlogs-agent/1.0", row


def test_metadata_status_and_status_column_agree_on_success(gateway: Gateway) -> None:
    """A served call writes `status` `success` in the column and `metadata.status` `success` as well, and
    `metadata.error_information` is null. The status column is derived from `metadata.status`
    (`_get_status_for_spend_log`, spend_tracking_utils.py:2367), so the two must never disagree."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        metadata: Final = metadata_of(row)
        assert row["status"] == "success", row
        assert metadata["status"] == "success", metadata
        assert metadata.get("error_information") is None, metadata


def test_additional_usage_values_lift_cached_prompt_tokens_into_cache_read_input_tokens(gateway: Gateway) -> None:
    """The provider reports `prompt_tokens_details.cached_tokens` 4 on a chat call. The row's
    `metadata.additional_usage_values` must hold `prompt_tokens_details` with `cached_tokens` 4 and a derived
    `cache_read_input_tokens` 4 (spend_tracking_utils.py:997-1012). Prompt caching reports read that derived
    key, not the OpenAI-shaped detail."""
    with caching_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        extra: Final = object_value(metadata_of(row)["additional_usage_values"])
        assert object_value(extra["prompt_tokens_details"])["cached_tokens"] == CACHED_PROMPT_TOKENS, extra
        assert extra["cache_read_input_tokens"] == CACHED_PROMPT_TOKENS, extra


def test_additional_usage_values_never_repeat_the_three_token_columns(gateway: Gateway) -> None:
    """`additional_usage_values` holds every usage field except `prompt_tokens`, `completion_tokens` and
    `total_tokens`, which have their own columns (`special_usage_fields`, spend_tracking_utils.py:990)."""
    with caching_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        extra: Final = object_value(metadata_of(row)["additional_usage_values"])
        assert not {"prompt_tokens", "completion_tokens", "total_tokens"} & set(extra), extra


def test_messages_row_keeps_anthropic_cache_write_and_read_tokens(gateway: Gateway) -> None:
    """An Anthropic `/v1/messages` call reports `cache_creation_input_tokens` 2 and `cache_read_input_tokens` 3.
    The row's `metadata.additional_usage_values` must carry both numbers under those exact names, and
    `metadata.usage_object` must carry them too, because the ledger projects daily cache counters from
    `usage_object` (`litellm/accounting/projection.py:155`)."""
    with caching_upstream() as wire, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": f"dec1 {uuid.uuid4().hex}"}]},
            key=key,
        )
        row: Final = key_rows(gateway, key, 1)[0]
        assert string_value(served(response)["id"]).startswith("msg_"), response.text
        extra: Final = object_value(metadata_of(row)["additional_usage_values"])
        usage: Final = object_value(metadata_of(row)["usage_object"])
        assert extra["cache_creation_input_tokens"] == CACHE_WRITE_TOKENS, extra
        assert extra["cache_read_input_tokens"] == CACHE_READ_TOKENS, extra
        assert usage["cache_creation_input_tokens"] == CACHE_WRITE_TOKENS, usage
        assert usage["cache_read_input_tokens"] == CACHE_READ_TOKENS, usage


def test_daily_user_row_counts_cached_prompt_tokens_from_the_request_row(gateway: Gateway) -> None:
    """After one chat call whose provider reports 4 cached prompt tokens, the key's `LiteLLM_DailyUserSpend`
    row has `cache_read_input_tokens` 4 and `cache_creation_input_tokens` 0. The projection reads them from the
    request row, so a row builder that drops the usage object zeroes the daily cache counters."""
    with caching_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served_id(ask(gateway, model, key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert daily["cache_read_input_tokens"] == CACHED_PROMPT_TOKENS, daily
        assert daily["cache_creation_input_tokens"] == 0, daily


def test_daily_user_row_counts_anthropic_cache_write_and_read_tokens(gateway: Gateway) -> None:
    """After one `/v1/messages` call reporting 2 cache-write and 3 cache-read tokens, the key's daily user row
    has `cache_creation_input_tokens` 2 and `cache_read_input_tokens` 3."""
    with caching_upstream() as wire, gateway.scenario() as scenario:
        model: Final = messages_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": f"dec1 {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert daily["cache_creation_input_tokens"] == CACHE_WRITE_TOKENS, daily
        assert daily["cache_read_input_tokens"] == CACHE_READ_TOKENS, daily


def test_cost_breakdown_splits_the_spend_into_input_and_output_cost(gateway: Gateway) -> None:
    """With 10 prompt tokens at 0.001 and 5 completion tokens at 0.002, `metadata.cost_breakdown` has
    `input_cost` 0.01, `output_cost` 0.01 and `total_cost` 0.02, matching the `spend` column. The UI's cost
    drill-down reads this object.
    UNVERIFIED: that a deployment-level custom price fills `cost_breakdown` the same way as the cost map;
    `CostBreakdown` is at litellm/types/utils.py:3856."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        breakdown: Final = object_value(metadata_of(row)["cost_breakdown"])
        assert number(breakdown["input_cost"]) == pytest.approx(0.01), breakdown
        assert number(breakdown["output_cost"]) == pytest.approx(0.01), breakdown
        assert number(breakdown["total_cost"]) == pytest.approx(number(row["spend"])), breakdown


def test_model_map_information_names_the_entry_the_call_was_priced_with(gateway: Gateway) -> None:
    """The row's `metadata.model_map_information` is an object with a non-empty `model_map_key` string, the
    pricing entry that was used. Support uses it to explain a price; it is part of the long tail the Rust port
    reads from the standard logging payload.
    UNVERIFIED: the exact key for a deployment with custom prices (likely the deployment model
    `openai/gpt-4o-mini` or `gpt-4o-mini`), so only non-emptiness is asserted."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        information: Final = object_value(metadata_of(row)["model_map_information"])
        assert string_value(information["model_map_key"]), information


def test_litellm_overhead_time_is_a_non_negative_number(gateway: Gateway) -> None:
    """The row's `metadata.litellm_overhead_time_ms` is a number of milliseconds of proxy overhead and is never
    negative. It is the same value the `x-litellm-overhead-duration-ms` header reports.
    UNVERIFIED: the header and the stored value are taken at the same moment; only sign and type are asserted."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        overhead: Final = metadata_of(row)["litellm_overhead_time_ms"]
        assert number(overhead) >= 0, row


def test_metadata_never_contains_the_raw_virtual_key_even_under_an_alias(gateway: Gateway) -> None:
    """A key with an alias makes one call. `metadata.user_api_key` is the SHA-256 hash, and neither the raw
    `sk-` secret nor its last four characters appended to `sk-...` appear in the stored metadata."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], key_alias=f"dec1-sl-{uuid.uuid4().hex[:6]}")
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        stored: Final = str(row["metadata"])
        assert key not in stored, "raw key leaked into metadata"
        assert len(string_value(metadata_of(row)["user_api_key"])) == 64, row


def test_cache_key_column_says_cache_off_when_the_response_cache_is_disabled(gateway: Gateway, tmp_path: Path) -> None:
    """An owned proxy runs with `litellm_settings.cache: false`. A served chat call writes `cache_key` exactly
    `Cache OFF` and `cache_hit` `False` (spend_tracking_utils.py:1022). Reports that count cacheable traffic rely
    on this sentinel."""
    with owned_spend_proxy(gateway, tmp_path, litellm_settings={"cache": False}) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert row["cache_key"] == "Cache OFF", row
        assert row["cache_hit"] == "False", row


def test_messages_column_stays_empty_for_chat_even_when_prompts_are_stored(gateway: Gateway, tmp_path: Path) -> None:
    """With `store_prompts_in_spend_logs: true`, a chat call stores its body in `proxy_server_request` but the
    legacy `messages` column stays the empty object `{}`: only realtime calls write `messages`
    (`_get_messages_for_spend_logs_payload`, spend_tracking_utils.py:1352). A port that fills `messages` doubles
    the row size."""
    with owned_spend_proxy(gateway, tmp_path, general_settings={"store_prompts_in_spend_logs": True}) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert row["messages"] == {}, row
        assert object_value(row["proxy_server_request"])["messages"], row
