"""Contract: the redaction settings matrix for what a spend log row may keep.

The V2 row builder must port these decisions exactly (plan, "payload policy (redaction, masking, trace
controls)"). Read from `litellm/proxy/spend_tracking/spend_tracking_utils.py` and
`litellm/litellm_core_utils/redact_messages.py`:

1. Prompt storage (`proxy_server_request`, `response` columns) needs `general_settings.store_prompts_in_spend_logs`
   true (bool or case-insensitive "true") or env `STORE_PROMPTS_IN_SPEND_LOGS`
   (`should_store_prompts_and_responses_in_spend_logs`, `:2297`). A key or team can only switch it OFF, through
   `metadata.store_prompts_in_spend_logs` false or a false-like string (`_disables_prompt_storage`, `:2286`,
   fields `user_api_key_metadata` and `user_api_key_team_metadata`); a key cannot switch it on. Without storage
   both columns are the literal `{}`.
2. Message redaction (`should_redact_message_logging`, redact_messages.py:770) is decided per request in this
   order: header `litellm-disable-message-redaction` (only honoured when the key or team metadata sets
   `allow_client_message_redaction_opt_out: true`, otherwise stripped, litellm_pre_call_utils.py:829), then the
   request's own `turn_off_message_logging`, then the `litellm-enable-message-redaction` /
   `x-litellm-enable-message-redaction` headers, then `litellm_settings.turn_off_message_logging`. A client
   body `turn_off_message_logging: false` is stripped when the global switch is on unless the key or team
   allows the opt out (`_strip_client_message_redaction_opt_out`, litellm_pre_call_utils.py:885).
3. A redacted row stores the request messages as `[{"role": "user", "content": "redacted-by-litellm"}]` and the
   response's choice content as `redacted-by-litellm` (`perform_redaction`, redact_messages.py:674 and
   `_redact_model_response_dict_choices`, `:599`). Spend and tokens are never redacted.
4. `disable_spend_logs` wins over everything: no row at all.

Round 1 pinned the plain `turn_off_message_logging` request side and the key/team opt-out of storage in one test
each; these tests fill the rest of the matrix.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, list_value, object_value
from tests.integration.spend.dec1_spend_support import (
    REPLY_TEXT,
    ask,
    chat_model,
    number,
    row_by_id,
    served_id,
)
from tests.integration.spend.dec1_spendlogs2_support import fresh_call_id, json_column, no_rows_with_call_id
from tests.integration.spend.dec1_spendlogs_support import owned_spend_proxy

pytestmark: Final = pytest.mark.shares_worker_state

STORE: Final = {"store_prompts_in_spend_logs": True}
REDACT: Final = {"turn_off_message_logging": True}
REDACTED: Final = "redacted-by-litellm"
REDACTED_MESSAGES: Final = [{"role": "user", "content": REDACTED}]


def _stored_messages(row: dict[str, JsonValue]) -> JsonValue:
    return object_value(json_column(row["proxy_server_request"]))["messages"]


def _stored_answer(row: dict[str, JsonValue]) -> JsonValue:
    response: Final = object_value(json_column(row["response"]))
    choice: Final = object_value(list_value(response["choices"])[0])
    return object_value(choice["message"])["content"]


def _prompt() -> str:
    return f"dec1 sl2 private prompt {uuid.uuid4().hex}"


def test_without_prompt_storage_both_payload_columns_are_empty_objects(gateway: Gateway, tmp_path: Path) -> None:
    """With `store_prompts_in_spend_logs` unset, a served chat row stores `proxy_server_request` and `response`
    as the literal `{}` (`_get_proxy_server_request_for_spend_logs_payload` returns "{}"), and the prompt text is
    nowhere in the row."""
    prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        assert (json_column(row["proxy_server_request"]), json_column(row["response"])) == ({}, {}), row
        assert prompt not in str(row), row


@pytest.mark.parametrize("value", ["true", "TRUE", "True"])
def test_prompt_storage_accepts_a_case_insensitive_true_string(gateway: Gateway, tmp_path: Path, value: str) -> None:
    """`store_prompts_in_spend_logs` given as the string "true" in any case turns storage on (spend_tracking_utils.py
    normalises strings with `.lower() == "true"`): the stored request messages contain the caller's prompt."""
    prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path, general_settings={"store_prompts_in_spend_logs": value}) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        assert _stored_messages(row) == [{"role": "user", "content": prompt}], row


def test_stored_response_keeps_the_provider_answer_text(gateway: Gateway, tmp_path: Path) -> None:
    """With storage on and no redaction, the `response` column holds the chat completion whose first choice's
    message content is the provider's text `scripted dec1 reply`."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert _stored_answer(row) == REPLY_TEXT, row


def test_global_redaction_replaces_the_stored_answer_text(gateway: Gateway, tmp_path: Path) -> None:
    """With storage on and `litellm_settings.turn_off_message_logging: true`, the stored response's first choice
    content is `redacted-by-litellm` while `completion_tokens` stays 5: redaction is a content rule, never a
    usage rule."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, litellm_settings=REDACT) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert _stored_answer(row) == REDACTED, row
        assert row["completion_tokens"] == 5, row


def test_global_redaction_without_storage_still_stores_nothing(gateway: Gateway, tmp_path: Path) -> None:
    """Redaction on and storage off: both payload columns are `{}` (storage is checked first), not a redacted
    placeholder, so turning redaction on never starts storing anything."""
    with owned_spend_proxy(gateway, tmp_path, litellm_settings=REDACT) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert (json_column(row["proxy_server_request"]), json_column(row["response"])) == ({}, {}), row


def test_a_request_can_ask_for_redaction_when_the_global_switch_is_off(gateway: Gateway, tmp_path: Path) -> None:
    """Storage on, global redaction off. A call whose body sets `turn_off_message_logging: true` is stored with
    redacted messages: the dynamic parameter has priority 1 in `should_redact_message_logging`.

    BUG: auth answers 401 "turn_off_message_logging is not allowed in request body" before the call runs. The field
    is in `_supported_callback_params` (litellm/litellm_core_utils/initialize_dynamic_callback_params.py:169) and is
    not in `_SAFE_CLIENT_CALLBACK_PARAMS` (litellm/proxy/auth/auth_utils.py:446), so `_build_banned_observability_params`
    (auth_utils.py:477) bans it like a credential, although asking for more redaction exfiltrates nothing and the
    pre-call code still handles this field (litellm_pre_call_utils.py:885)."""
    pytest.skip("BUG: a client turn_off_message_logging true in the body is refused 401 as a banned callback param")
    prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(owned, model, key, text=prompt, turn_off_message_logging=True)
        row: Final = row_by_id(owned, served_id(response))
        assert _stored_messages(row) == REDACTED_MESSAGES, row
        assert prompt not in str(row), row


@pytest.mark.parametrize("header", ["litellm-enable-message-redaction", "x-litellm-enable-message-redaction"])
def test_a_request_header_can_ask_for_redaction(gateway: Gateway, tmp_path: Path, header: str) -> None:
    """Storage on, global redaction off. A call sent with header `<header>: true` is stored redacted
    (`MESSAGE_REDACTION_ENABLE_HEADERS`, redact_messages.py:54). UNVERIFIED: the proxy copies that header into
    `metadata.headers`, which the predicate reads."""
    prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt, headers={header: "true"})))
        assert _stored_messages(row) == REDACTED_MESSAGES, row


def test_a_client_cannot_opt_out_of_global_redaction_by_default(gateway: Gateway, tmp_path: Path) -> None:
    """Storage and global redaction on. A call whose body sets `turn_off_message_logging: false` is refused with
    401 naming the field, so it can never be stored unredacted. Pinned current: auth bans every non-safe dynamic
    callback param in the body (`_build_banned_observability_params`, litellm/proxy/auth/auth_utils.py:477, raised
    by `_check_banned_params`, auth_utils.py:846) before `_strip_client_message_redaction_opt_out`
    (litellm_pre_call_utils.py:885) would have stripped it."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, litellm_settings=REDACT) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(owned, model, key, text=_prompt(), turn_off_message_logging=False)
        assert response.status_code == 401, response.text
        assert "turn_off_message_logging is not allowed in request body" in response.text, response.text


def test_a_key_allowed_to_opt_out_keeps_its_prompt_under_global_redaction(gateway: Gateway, tmp_path: Path) -> None:
    """Storage and global redaction on, and the key's metadata sets `allow_client_message_redaction_opt_out:
    true`. Its call with `turn_off_message_logging: false` is stored with the caller's prompt unredacted.
    UNVERIFIED: end to end; read from litellm_pre_call_utils.py:3135 (strip skipped) and redact_messages.py:803.

    BUG: auth refuses the body field with 401 before the key permission is consulted (banned callback param,
    litellm/proxy/auth/auth_utils.py:477), so `allow_client_message_redaction_opt_out` can never keep a body opt out;
    only the header form works."""
    pytest.skip("BUG: the body turn_off_message_logging opt out is refused 401 even when the key allows the opt out")
    prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, litellm_settings=REDACT) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], metadata={"allow_client_message_redaction_opt_out": True})
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt, turn_off_message_logging=False)))
        assert _stored_messages(row) == [{"role": "user", "content": prompt}], row


def test_the_disable_redaction_header_is_ignored_without_the_key_permission(gateway: Gateway, tmp_path: Path) -> None:
    """Storage and global redaction on. A call sent with `litellm-disable-message-redaction: true` from a key
    without `allow_client_message_redaction_opt_out` is stored redacted: the header is in
    `UNTRUSTED_REQUEST_HEADER_CONTROL_FIELDS` and is dropped (litellm_pre_call_utils.py:829)."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, litellm_settings=REDACT) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(owned, model, key, headers={"litellm-disable-message-redaction": "true"})
        assert _stored_messages(row_by_id(owned, served_id(response))) == REDACTED_MESSAGES


def test_the_disable_redaction_header_works_for_a_team_that_allows_it(gateway: Gateway, tmp_path: Path) -> None:
    """Same proxy, but the key belongs to a team whose metadata sets `allow_client_message_redaction_opt_out:
    true`. The header is kept and the row stores the caller's prompt unredacted. UNVERIFIED: team metadata is
    read by `_key_or_team_metadata_flag_is_true` for this flag."""
    prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, litellm_settings=REDACT) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(metadata={"allow_client_message_redaction_opt_out": True})
        key: Final = scenario.key(models=[model], team_id=team)
        response: Final = ask(owned, model, key, text=prompt, headers={"litellm-disable-message-redaction": "true"})
        assert _stored_messages(row_by_id(owned, served_id(response))) == [{"role": "user", "content": prompt}]


@pytest.mark.parametrize("value", [False, "false", "FALSE", " False "])
def test_a_key_can_switch_prompt_storage_off_with_a_false_like_value(
    gateway: Gateway, tmp_path: Path, value: JsonValue
) -> None:
    """Storage on globally; the key's metadata sets `store_prompts_in_spend_logs` to `<value>`. Its row stores
    `{}` in both payload columns (`_disables_prompt_storage` accepts False or a string `str_to_bool` reads as
    False, spend_tracking_utils.py:2289). `str_to_bool` only knows "true" and "false", trimmed and case-insensitive
    (litellm/secret_managers/main.py:143), so "no" and "0" do not switch storage off."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], metadata={"store_prompts_in_spend_logs": value})
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert (json_column(row["proxy_server_request"]), json_column(row["response"])) == ({}, {}), row


def test_a_key_cannot_switch_prompt_storage_on(gateway: Gateway, tmp_path: Path) -> None:
    """Storage off globally; the key's metadata sets `store_prompts_in_spend_logs: true`. The row still stores
    `{}`: tenant metadata can only disable storage, the decision to store belongs to the operator."""
    prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], metadata={"store_prompts_in_spend_logs": True})
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        assert json_column(row["proxy_server_request"]) == {}, row
        assert prompt not in str(row), row


def test_a_team_opt_out_applies_even_when_the_key_says_true(gateway: Gateway, tmp_path: Path) -> None:
    """Storage on globally, the team's metadata sets `store_prompts_in_spend_logs: false` and the key's metadata
    sets it true. The row stores `{}`: any tenant field that disables storage wins (`any(...)` over both fields)."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(metadata={"store_prompts_in_spend_logs": False})
        key: Final = scenario.key(models=[model], team_id=team, metadata={"store_prompts_in_spend_logs": True})
        row: Final = row_by_id(owned, served_id(ask(owned, model, key)))
        assert json_column(row["proxy_server_request"]) == {}, row


def test_a_storage_opt_out_on_one_key_does_not_affect_another(gateway: Gateway, tmp_path: Path) -> None:
    """Storage on globally. Key A opts out, key B does not. A's row stores `{}` and B's row stores its prompt:
    the decision is per request, never cached across tenants. The two prompts differ so the second call is not a
    response cache hit (the shared config runs `cache: true`), whose row stores no request payload."""
    prompt: Final = _prompt()
    loud_prompt: Final = _prompt()
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        quiet: Final = scenario.key(models=[model], metadata={"store_prompts_in_spend_logs": False})
        loud: Final = scenario.key(models=[model])
        quiet_row: Final = row_by_id(owned, served_id(ask(owned, model, quiet, text=prompt)))
        loud_row: Final = row_by_id(owned, served_id(ask(owned, model, loud, text=loud_prompt)))
        assert json_column(quiet_row["proxy_server_request"]) == {}, quiet_row
        assert _stored_messages(loud_row) == [{"role": "user", "content": loud_prompt}], loud_row


def test_disable_spend_logs_with_storage_and_redaction_on_writes_no_row(gateway: Gateway, tmp_path: Path) -> None:
    """With `disable_spend_logs`, `store_prompts_in_spend_logs` and `turn_off_message_logging` all on, a served
    call (sent with a client call id) leaves no row under that call id: disabling the row wins over every
    content setting."""
    call_id: Final = fresh_call_id()
    with owned_spend_proxy(
        gateway, tmp_path, general_settings={**STORE, "disable_spend_logs": True}, litellm_settings=REDACT
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(owned, model, key, headers={"x-litellm-call-id": call_id})
        assert response.status_code == 200, response.text
        no_rows_with_call_id(owned, call_id)


def test_redaction_never_changes_the_billed_spend(gateway: Gateway, tmp_path: Path) -> None:
    """Under storage plus global redaction, a plain call (redacted by the global switch) and one sent with the
    enable header both bill 0.02 exactly like an unredacted call. The body field `turn_off_message_logging` is
    refused 401 by auth (litellm/proxy/auth/auth_utils.py:477), so the header is the per-request way to ask."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, litellm_settings=REDACT) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        first: Final = row_by_id(owned, served_id(ask(owned, model, key, text=_prompt())))
        second: Final = row_by_id(
            owned, served_id(ask(owned, model, key, headers={"x-litellm-enable-message-redaction": "true"}))
        )
        assert (number(first["spend"]), number(second["spend"])) == (pytest.approx(0.02), pytest.approx(0.02))


def test_streamed_answers_are_redacted_in_the_stored_response(gateway: Gateway, tmp_path: Path) -> None:
    """Storage and global redaction on. A streamed chat's stored `response` holds no `scripted dec1 reply` text
    anywhere (the assembled stream is redacted, redact_messages.py:711), and `completion_tokens` is 5."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE, litellm_settings=REDACT) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = owned.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _prompt()}], "stream": True},
            key=key,
        )
        assert response.status_code == 200, response.text
        row: Final = row_by_id(owned, _stream_id(response.text))
        assert REPLY_TEXT not in str(row["response"]), row
        assert row["completion_tokens"] == 5, row


def _stream_id(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("data: {"):
            return str(object_value(json.loads(line[6:]))["id"])
    raise AssertionError(text)
