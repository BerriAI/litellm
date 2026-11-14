"""Contract: what one non-streaming chat completion writes into `LiteLLM_SpendLogs`.

Every successful `/v1/chat/completions` call must leave exactly one row whose columns say who called (hashed key,
user, team, organization, end user), what was called (public model name, provider model, deployment id, provider,
api base), what it cost (tokens, spend), when (start, end, duration) and how it was labelled (tags, metadata,
session, cache, status). Billing exports, the usage UI and every `/spend/*` read route are built on these columns,
so a rewrite of the spend writer has to reproduce each of them. The scripted provider always reports 10 prompt and
5 completion tokens and the deployment costs 0.001 per input token and 0.002 per output token, so the expected
spend is 10 * 0.001 + 5 * 0.002 = 0.02.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, list_value, object_value, string_value
from tests.integration.spend.dec1_spend_support import (
    CHAT_COMPLETION_TOKENS,
    CHAT_COST,
    CHAT_PROMPT_TOKENS,
    ask,
    chat_model,
    hashed,
    metadata_of,
    number,
    provider_calls,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served_id,
)


def test_chat_row_request_id_is_the_response_id_returned_to_the_caller(gateway: Gateway) -> None:
    """A key makes one chat call. The row's `request_id` must equal the `id` in the JSON body the caller received,
    and that id must appear exactly once in the table, because callers join their own logs to ours on it."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response_id: Final = served_id(ask(gateway, model, key))
        rows: Final = rows_of_key(gateway, key, 1)
        assert rows[0]["request_id"] == response_id, rows


def test_chat_row_litellm_call_id_matches_the_call_id_response_header(gateway: Gateway) -> None:
    """The `x-litellm-call-id` response header and the row's `litellm_call_id` column (also copied into
    `metadata.litellm_call_id`) must be the same value, so a support ticket quoting the header finds the row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        row: Final = row_by_id(gateway, served_id(response))
        assert row["litellm_call_id"] == response.headers["x-litellm-call-id"], row
        assert metadata_of(row)["litellm_call_id"] == response.headers["x-litellm-call-id"], row


def test_chat_row_call_type_is_acompletion(gateway: Gateway) -> None:
    """A `/v1/chat/completions` call is stored with `call_type` exactly `acompletion`. Daily rollups map this
    string to the `/chat/completions` endpoint, so a different spelling silently breaks per-endpoint usage."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert row["call_type"] == "acompletion", row


def test_chat_row_api_key_is_the_sha256_hash_and_the_raw_key_is_stored_nowhere(gateway: Gateway) -> None:
    """The `api_key` column and `metadata.user_api_key` hold the SHA-256 hex digest of the virtual key. The raw
    `sk-` secret must not appear anywhere in the stored row, since spend logs are readable by non-admins."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert row["api_key"] == hashed(key), row
        assert metadata_of(row)["user_api_key"] == hashed(key), row
        assert key not in json.dumps(row), "the raw virtual key leaked into the spend log row"


def test_chat_row_model_group_is_the_public_name_and_model_is_the_provider_model(gateway: Gateway) -> None:
    """The deployment is published under a random public name and calls provider model `openai/gpt-4o-mini`. The
    row stores the public name in `model_group` and the deployment's model in `model`, never the other way round.
    `model` is the deployment's `litellm_params.model` with its provider prefix, as the daily tables and
    `/spend/logs` summaries also report it."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert row["model_group"] == model, row
        assert row["model"] == "openai/gpt-4o-mini", row


def test_chat_row_model_id_matches_the_model_id_response_header(gateway: Gateway) -> None:
    """`model_id` is the id of the deployment that served the call. It must be non-empty and equal the
    `x-litellm-model-id` response header, which is how a caller tells which deployment was billed."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        row: Final = row_by_id(gateway, served_id(response))
        assert row["model_id"], row
        assert row["model_id"] == response.headers["x-litellm-model-id"], row


def test_chat_row_records_provider_and_api_base_of_the_deployment(gateway: Gateway) -> None:
    """For an `openai/...` deployment pointed at the scripted provider, `custom_llm_provider` is `openai` and
    `api_base` is the deployment's configured base url, with no credentials in it. The chat path logs the base
    url of the OpenAI client, which carries a trailing slash, so the comparison ignores one."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert row["custom_llm_provider"] == "openai", row
        assert string_value(row["api_base"]).rstrip("/") == f"{wire.url}/v1", row
        assert "dec1-provider-key" not in json.dumps(row), "the provider api key leaked into the spend log row"


def test_chat_row_token_columns_copy_the_provider_usage(gateway: Gateway) -> None:
    """The provider reports 10 prompt and 5 completion tokens. The row must store prompt_tokens=10,
    completion_tokens=5, total_tokens=15, and `metadata.usage_object` must carry the same three numbers."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (
            CHAT_PROMPT_TOKENS,
            CHAT_COMPLETION_TOKENS,
            15,
        ), row
        usage: Final = object_value(metadata_of(row)["usage_object"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (10, 5, 15), usage


def test_chat_row_spend_is_tokens_times_the_deployment_price(gateway: Gateway) -> None:
    """With 0.001 per input token and 0.002 per output token on the deployment, 10 prompt and 5 completion
    tokens must be stored as spend 0.02. The price comes from the deployment, not from the public cost map."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert number(row["spend"]) == pytest.approx(10 * 0.001 + 5 * 0.002), row


def test_chat_response_cost_header_equals_the_spend_stored_in_the_row(gateway: Gateway) -> None:
    """The `x-litellm-response-cost` header the caller sees and the `spend` column must be the same number
    (0.02 here). Customers reconcile their own totals against this header."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        row: Final = row_by_id(gateway, served_id(response))
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(CHAT_COST), response.headers
        assert number(row["spend"]) == pytest.approx(float(response.headers["x-litellm-response-cost"])), row


def test_chat_row_times_are_ordered_and_duration_matches_them(gateway: Gateway) -> None:
    """`startTime` must not be after `endTime`, `completionStartTime` must be set, and `request_duration_ms`
    must be the non-negative whole-millisecond difference between the two stored timestamps."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert row["ends_after_start"] is True, row
        assert row["completion_start_time"] is not None, row
        stored_ms: Final = row["request_duration_ms"]
        assert isinstance(stored_ms, int) and stored_ms >= 0, row
        assert stored_ms == pytest.approx(number(row["measured_ms"]), abs=1), row


def test_chat_row_user_is_the_user_that_owns_the_key(gateway: Gateway) -> None:
    """A key created for an internal user stores that `user_id` in the `user` column and in
    `metadata.user_api_key_user_id`. This is the internal user, not the OpenAI `user` body field."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user()
        key: Final = scenario.key(models=[model], user_id=user)
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key, user="dec1-someone-else")))
        assert row["user"] == user, row
        assert metadata_of(row)["user_api_key_user_id"] == user, row


def test_chat_row_of_a_key_without_owner_or_team_leaves_attribution_blank(gateway: Gateway) -> None:
    """A key with no user, team or organization, called without a `user` field, stores empty `user`, `team_id`,
    `organization_id` and `end_user`. Assumes blank is stored as an empty string or NULL, never as another id."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        blanks: Final = {column: row[column] for column in ("user", "team_id", "organization_id", "end_user")}
        assert all(value in ("", None) for value in blanks.values()), blanks


def test_chat_row_records_team_and_organization_of_the_key(gateway: Gateway) -> None:
    """A key of a team inside an organization stores the team in `team_id` and the organization in
    `organization_id`, and the team alias in `metadata.user_api_key_team_alias`."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        alias: Final = f"dec1-team-{uuid.uuid4().hex}"
        organization: Final = scenario.organization(models=[model])
        team: Final = scenario.team(organization_id=organization, models=[model], team_alias=alias)
        key: Final = scenario.key(team_id=team, models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        assert (row["team_id"], row["organization_id"]) == (team, organization), row
        assert metadata_of(row)["user_api_key_team_alias"] == alias, row


def test_chat_row_end_user_is_the_user_field_of_the_request_body(gateway: Gateway) -> None:
    """The OpenAI `user` body field names the caller's customer. It is stored in the `end_user` column while the
    `user` column keeps the key owner, so the two kinds of user never mix."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        owner: Final = scenario.user()
        key: Final = scenario.key(models=[model], user_id=owner)
        customer: Final = f"dec1-customer-{uuid.uuid4().hex}"
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key, user=customer)))
        assert (row["end_user"], row["user"]) == (customer, owner), row


def test_chat_row_request_tags_hold_the_tags_sent_in_request_metadata(gateway: Gateway) -> None:
    """Tags sent as `metadata.tags` in the body are stored in `request_tags` as a JSON array of strings. Tag
    budgets, tag routing reports and `/spend/tags` all read this array."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        tags: Final = [f"dec1-a-{uuid.uuid4().hex}", f"dec1-b-{uuid.uuid4().hex}"]
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key, metadata={"tags": [*tags]})))
        stored: Final = list_value(row["request_tags"])
        assert set(tags) <= set(stored), stored
        assert all(isinstance(tag, str) for tag in stored), stored


def test_chat_row_request_tags_merge_key_tags_and_header_tags(gateway: Gateway) -> None:
    """The key carries one tag, the caller adds one through the comma-separated `x-litellm-tags` header. Both
    must be in `request_tags` of the row, so key-level tagging cannot be dropped by a caller sending its own."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key_tag: Final = f"dec1-key-tag-{uuid.uuid4().hex}"
        header_tag: Final = f"dec1-header-tag-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], tags=[key_tag])
        response: Final = ask(gateway, model, key, headers={"x-litellm-tags": f"{header_tag},dec1-second"})
        stored: Final = list_value(row_by_id(gateway, served_id(response))["request_tags"])
        assert {key_tag, header_tag, "dec1-second"} <= set(stored), stored


def test_chat_row_request_tags_include_the_caller_user_agent(gateway: Gateway) -> None:
    """A caller sending `User-Agent: dec1-agent/1.0` gets two automatic tags, `User-Agent: dec1-agent` and
    `User-Agent: dec1-agent/1.0`. Usage-by-client dashboards depend on both spellings."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key, headers={"User-Agent": "dec1-agent/1.0"})
        stored: Final = list_value(row_by_id(gateway, served_id(response))["request_tags"])
        assert {"User-Agent: dec1-agent", "User-Agent: dec1-agent/1.0"} <= set(stored), stored


def test_chat_row_metadata_names_the_key_alias_and_status(gateway: Gateway) -> None:
    """The row's `metadata` JSON must contain the key alias and the bookkeeping keys `usage_object`,
    `model_map_information`, `cost_breakdown` and `additional_usage_values`, and the row's `status` column must be
    `success`. The outcome lives in the `status` column (`litellm/proxy/spend_tracking/spend_tracking_utils.py:996`);
    `metadata.status` only copies a `failure` marker from the request metadata and is null on success."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        alias: Final = f"dec1-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[model], key_alias=alias)
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        metadata: Final = metadata_of(row)
        assert metadata["user_api_key_alias"] == alias, metadata
        assert row["status"] == "success", row
        assert metadata["status"] != "failure", metadata
        assert {"usage_object", "model_map_information", "cost_breakdown", "additional_usage_values"} <= set(metadata)


def test_chat_row_metadata_keeps_spend_logs_metadata_from_the_request_body(gateway: Gateway) -> None:
    """A caller sends `metadata.spend_logs_metadata` with its own key/value pairs. The same object must be stored
    under `metadata.spend_logs_metadata`, which is the supported way to attach custom billing dimensions."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        custom: Final = {"dec1_cost_center": "alpha", "dec1_ticket": 42}
        response: Final = ask(gateway, model, key, metadata={"spend_logs_metadata": {**custom}})
        assert metadata_of(row_by_id(gateway, served_id(response)))["spend_logs_metadata"] == custom


def test_chat_row_metadata_keeps_spend_logs_metadata_from_the_request_header(gateway: Gateway) -> None:
    """The `x-litellm-spend-logs-metadata` header carries a JSON object. It must be stored under
    `metadata.spend_logs_metadata` exactly like the body form, for clients that cannot change the body."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(
            gateway, model, key, headers={"x-litellm-spend-logs-metadata": json.dumps({"dec1_source": "header"})}
        )
        assert metadata_of(row_by_id(gateway, served_id(response)))["spend_logs_metadata"] == {"dec1_source": "header"}


def test_chat_row_spend_logs_metadata_of_the_key_is_merged_and_the_request_wins(gateway: Gateway) -> None:
    """The key's own `metadata.spend_logs_metadata` sets two pairs and the request overrides one of them. The
    stored object must have the request's value for the shared name and the key's value for the other."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(
            models=[model], metadata={"spend_logs_metadata": {"dec1_env": "from-key", "dec1_owner": "key-owner"}}
        )
        response: Final = ask(gateway, model, key, metadata={"spend_logs_metadata": {"dec1_env": "from-request"}})
        assert metadata_of(row_by_id(gateway, served_id(response)))["spend_logs_metadata"] == {
            "dec1_env": "from-request",
            "dec1_owner": "key-owner",
        }


def test_chat_row_cache_columns_mark_a_provider_served_call_as_not_a_hit(gateway: Gateway) -> None:
    """A first, uncached call reaches the provider once. Its row must not say `cache_hit` `True` and its
    `cache_key` is a sha256 hex digest. The `x-litellm-cache-key` header is only sent on a cache hit, so a
    provider-served answer carries none. A miss is stored as the string `None`
    (`litellm/proxy/spend_tracking/spend_tracking_utils.py:948`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        row: Final = row_by_id(gateway, served_id(response))
        assert len(provider_calls(wire)) == 1
        assert row["cache_hit"] in ("False", "None"), row
        assert "x-litellm-cache-key" not in response.headers, dict(response.headers)
        assert re.fullmatch(r"[0-9a-f]{64}", string_value(row["cache_key"])), row


@pytest.mark.parametrize("carrier", ["metadata.session_id", "x-litellm-session-id", "litellm_session_id"])
def test_chat_row_session_id_is_the_session_the_caller_named(gateway: Gateway, carrier: str) -> None:
    """The caller names a session in body `metadata.session_id`, in the `x-litellm-session-id` header, or in the
    body field `litellm_session_id`. In each case the row's `session_id` column is that exact string.
    UNVERIFIED for `litellm_session_id`: read from `_get_session_id_for_spend_log`, not run."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        session: Final = f"dec1-session-{uuid.uuid4().hex}"
        response: Final = ask(
            gateway,
            model,
            key,
            headers={"x-litellm-session-id": session} if carrier == "x-litellm-session-id" else None,
            **(
                {"metadata": {"session_id": session}}
                if carrier == "metadata.session_id"
                else {"litellm_session_id": session}
                if carrier == "litellm_session_id"
                else {}
            ),
        )
        assert row_by_id(gateway, served_id(response))["session_id"] == session


def test_chat_rows_without_a_session_get_distinct_generated_session_ids(gateway: Gateway) -> None:
    """Two calls that name no session each get a non-empty `session_id`, and the two ids differ. The default
    policy generates one per request, it must never group unrelated requests into one session."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served_id(ask(gateway, model, key))
        served_id(ask(gateway, model, key))
        sessions: Final = [row["session_id"] for row in rows_of_key(gateway, key, 2)]
        assert all(isinstance(session, str) and session for session in sessions), sessions
        assert sessions[0] != sessions[1], sessions


def test_chat_row_status_is_success_and_prompt_columns_are_empty_by_default(gateway: Gateway) -> None:
    """With `store_prompts_in_spend_logs` unset, a successful call stores `status` `success` and leaves the
    `messages`, `response` and `proxy_server_request` columns as empty JSON objects. The prompt text and the
    reply text must not be anywhere in the row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"dec1 private prompt {uuid.uuid4().hex}"
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key, text=prompt)))
        assert row["status"] == "success", row
        assert (row["messages"], row["response"], row["proxy_server_request"]) == ({}, {}, {}), row
        assert prompt not in json.dumps(row) and "scripted dec1 reply" not in json.dumps(row), row


def test_one_chat_call_writes_exactly_one_row(gateway: Gateway) -> None:
    """Three sequential calls with one key leave exactly three rows with three distinct request ids, one provider
    call each. A writer that retries or double-flushes would show up here as a fourth row."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ids: Final = [served_id(ask(gateway, model, key)) for _ in range(3)]
        rows: Final = rows_of_key(gateway, key, 3)
        assert sorted(string_value(row["request_id"]) for row in rows) == sorted(ids), rows
        assert len(provider_calls(wire)) == 3
