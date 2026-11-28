"""Contract: what the request and response columns hold when prompt storage is on, ahead of the Spend Logs V2 port.

The V2 plan lists "prompt and response storage behind `store_prompts_in_spend_logs`, ... 2048-char truncation
split 35/65" among the row policy the Rust builder must reproduce byte for byte. Round 1 pins the cut of a long
prompt and round 2 the redaction matrix. This file pins the other halves of the same sanitizer
(`_sanitize_request_body_for_spend_logs_payload`, litellm/proxy/spend_tracking/spend_tracking_utils.py:1574):

1. The `response` column: a long provider answer is cut with the same 35/65 marker and honours
   `MAX_STRING_LENGTH_PROMPT_IN_DB` (`_get_response_for_spend_logs_payload`, :2228).
2. Strings nested anywhere in the stored request body, not only the prompt, are cut the same way.
3. Credential masking by key name in the stored request body: a string under a key with a `key`, `token`,
   `secret`, `password`, `auth` or `credential` segment becomes `REDACTED_BY_LITELM`, a key with a `cost` segment
   never does, and `tokens` is not `token` (`_is_request_body_credential`, :1519, `SensitiveDataMasker.is_sensitive_key`,
   litellm/litellm_core_utils/sensitive_data_masker.py:80).
4. Lists and tool calls are stored as given.

Every test runs on an owned proxy with `general_settings.store_prompts_in_spend_logs: true`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, list_value, object_value
from tests.integration.spend.dec1_spend_support import (
    ask,
    chat_model,
    embedding_model,
    row_by_id,
    rows_of_key,
    served_id,
)
from tests.integration.spend.dec1_spendlogs3_support import (
    fresh_tool_name,
    long_answer_upstream,
    stored_json,
    tool_call_upstream,
)
from tests.integration.spend.dec1_spendlogs_support import owned_spend_proxy

pytestmark: Final = pytest.mark.shares_worker_state

STORE: Final = {"store_prompts_in_spend_logs": True}
REDACTED: Final = "REDACTED_BY_LITELM"
"""`REDACTED_BY_LITELM_STRING`, litellm/constants.py:727."""
TRUNCATION_NOTE: Final = (
    "Truncation is a DB storage safeguard. Full, untruncated data is logged to logging callbacks "
    "(OTEL, Datadog, etc.). To increase the truncation limit, set `MAX_STRING_LENGTH_PROMPT_IN_DB` in your env."
)
"""`LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE`, litellm/constants.py:1707."""


def _cut(head: str, skipped: int, tail: str) -> str:
    """The stored form of a cut string (`_truncated_for_db`, spend_tracking_utils.py:1544)."""
    return f"{head}... (litellm_truncated skipped {skipped} chars. {TRUNCATION_NOTE}) ...{tail}"


def _stored_answer(row: Mapping[str, JsonValue]) -> JsonValue:
    response: Final = object_value(stored_json(row["response"]))
    return object_value(object_value(list_value(response["choices"])[0])["message"])["content"]


def _stored_metadata(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(object_value(stored_json(row["proxy_server_request"]))["metadata"])


def test_a_long_answer_is_stored_cut_35_65_around_the_marker(gateway: Gateway, tmp_path: Path) -> None:
    """The provider answers with `x` * 1500 + `y` * 1500. With the default limit of 2048 the stored
    `response.choices[0].message.content` is `x` * 716 (int(2048 * 0.35)), the marker saying 953 characters were
    skipped, then `y` * 1331 (int(2048 * 0.65)). The caller still receives all 3000 characters."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, upstream=long_answer_upstream) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(owned, model, key)
        assert response.status_code == 200, response.text
        assert len(response.json()["choices"][0]["message"]["content"]) == 3000
        row: Final = row_by_id(owned, response.json()["id"])
        assert _stored_answer(row) == _cut("x" * 716, 953, "y" * 1331), row["response"]


def test_a_raised_limit_stores_the_long_answer_whole(gateway: Gateway, tmp_path: Path) -> None:
    """With `MAX_STRING_LENGTH_PROMPT_IN_DB=5000` the same 3000 character answer is stored unchanged: the limit
    is read from the environment at row build time (`_get_max_string_length_prompt_in_db`, :125) for the
    response column as well as the request."""
    with owned_spend_proxy(
        gateway,
        tmp_path,
        general_settings=STORE,
        environment={"MAX_STRING_LENGTH_PROMPT_IN_DB": "5000"},
        upstream=long_answer_upstream,
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert _stored_answer(row) == "x" * 1500 + "y" * 1500


def test_a_long_string_nested_in_request_metadata_is_cut_the_same_way(gateway: Gateway, tmp_path: Path) -> None:
    """With `MAX_STRING_LENGTH_PROMPT_IN_DB=100` a request whose `metadata.dec1_note` is `n` * 3000 is stored with
    that field as `n` * 35, the marker saying 2900 characters were skipped, then `n` * 65: the cut applies to every
    string at every depth of the stored body, not just the messages."""
    with owned_spend_proxy(
        gateway, tmp_path, general_settings=STORE, environment={"MAX_STRING_LENGTH_PROMPT_IN_DB": "100"}
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, metadata={"dec1_note": "n" * 3000})))
        assert _stored_metadata(row)["dec1_note"] == _cut("n" * 35, 2900, "n" * 65), row["proxy_server_request"]


@pytest.mark.parametrize("field", ["dec1_api_key", "dec1_session_token", "dec1_password", "dec1_client_secret"])
def test_a_credential_named_body_field_is_masked_in_the_stored_request(
    gateway: Gateway, tmp_path: Path, field: str
) -> None:
    """A request whose metadata carries a string under a credential shaped key (a `key`, `token`, `password` or
    `secret` segment) is stored with that value replaced by `REDACTED_BY_LITELM`, while the provider call is made
    as usual. UNVERIFIED: the proxy keeps caller metadata keys in `proxy_server_request.body.metadata`."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, metadata={field: "dec1-synthetic-value"})))
        assert _stored_metadata(row)[field] == REDACTED, row["proxy_server_request"]


@pytest.mark.parametrize("field", ["dec1_cost_token", "dec1_max_tokens_note", "dec1_keyboard"])
def test_fields_that_only_look_like_credentials_are_stored_as_sent(
    gateway: Gateway, tmp_path: Path, field: str
) -> None:
    """Keys are split on `_` and `-` and matched by whole segment: a `cost` segment overrides a `token` one
    (`dec1_cost_token`), `tokens` is not `token` (`dec1_max_tokens_note`) and `keyboard` is not `key`, so each of
    these values is stored unchanged. A port that matches substrings would mask pricing and limit fields."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, metadata={field: "dec1-plain-value"})))
        assert _stored_metadata(row)[field] == "dec1-plain-value", row["proxy_server_request"]


def test_a_non_string_value_under_a_credential_key_is_not_masked(gateway: Gateway, tmp_path: Path) -> None:
    """Masking only replaces strings (`isinstance(value, str)` in `_is_request_body_credential`): metadata
    `dec1_token_count` 42 under a `token` segment key is stored as the number 42."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, metadata={"dec1_token_count": 42})))
        assert _stored_metadata(row)["dec1_token_count"] == 42, row["proxy_server_request"]


def test_an_embedding_input_list_is_stored_as_the_list_sent(gateway: Gateway, tmp_path: Path) -> None:
    """An embeddings call with `input` `["dec1 first", "dec1 second"]` stores the same two element list under
    `proxy_server_request.input`, in order.
    UNVERIFIED: the stored body is the caller's body, not the provider's."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        answer: Final = owned.request(
            "POST", "/v1/embeddings", {"model": model, "input": ["dec1 first", "dec1 second"]}, key=key
        )
        assert answer.status_code == 200, answer.text
        row: Final = rows_of_key(owned, key, 1)[0]
        assert object_value(stored_json(row["proxy_server_request"]))["input"] == ["dec1 first", "dec1 second"]


def test_a_stored_tool_call_keeps_its_name_and_arguments(gateway: Gateway, tmp_path: Path) -> None:
    """The provider answers with one tool call of a fresh tool with arguments `{}`. The stored response keeps
    `choices[0].message.tool_calls[0].function` as `{"name": <tool>, "arguments": "{}"}`: tool payloads are
    stored as content and never masked (`_is_spend_log_content`, :1523)."""
    tool: Final = fresh_tool_name()
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, upstream=lambda: tool_call_upstream(tool)) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        response: Final = object_value(stored_json(row["response"]))
        message: Final = object_value(object_value(list_value(response["choices"])[0])["message"])
        function: Final = object_value(object_value(list_value(message["tool_calls"])[0])["function"])
        assert (function["name"], function["arguments"]) == (tool, "{}"), message


def test_without_storage_a_long_answer_leaves_the_response_column_empty(gateway: Gateway, tmp_path: Path) -> None:
    """Without `store_prompts_in_spend_logs` the long answer is not stored at all: `response` is `{}` and
    `proxy_server_request` is `{}` (both helpers return "{}" when storage is off, :2155 and :2283), so the
    truncation path never runs."""
    with owned_spend_proxy(gateway, tmp_path, upstream=long_answer_upstream) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert (stored_json(row["response"]), stored_json(row["proxy_server_request"])) == ({}, {}), row
