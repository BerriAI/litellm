"""Contract: what the spend log row is allowed to keep of a request, and how long it may be.

The Spend Logs V2 plan lists this as row policy the Rust builder must port byte for byte: prompt storage behind
`store_prompts_in_spend_logs`, message redaction (`litellm_settings.turn_off_message_logging`), opt-in token
redaction (`general_settings.redact_tokens_in_spend_logs`, made opt-in by #51729), traceback suppression
(`LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS`) and the 2048 character truncation split 35/65 with a fixed marker
(`_truncated_for_db`, `litellm/proxy/spend_tracking/spend_tracking_utils.py:1544`). Each test boots an owned proxy
with one setting on, stores prompts, and reads back the row.

The truncation marker is, for a string of length L cut to the limit M:
`<first int(M*0.35) chars>... (litellm_truncated skipped <L - int(M*0.35) - int(M*0.65)> chars. Truncation is
a DB storage safeguard. Full, untruncated data is logged to logging callbacks (OTEL, Datadog, etc.). To increase
the truncation limit, set `MAX_STRING_LENGTH_PROMPT_IN_DB` in your env.) ...<last int(M*0.65) chars>`.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, list_value, object_value, string_value
from tests.integration.spend.dec1_spend_support import (
    ask,
    chat_model,
    metadata_of,
    row_by_id,
    served_id,
)
from tests.integration.spend.dec1_spendlogs_support import (
    failing_upstream,
    key_rows,
    owned_spend_proxy,
)

pytestmark: Final = pytest.mark.shares_worker_state

STORE: Final = {"store_prompts_in_spend_logs": True}
TRUNCATION_NOTE: Final = (
    "Truncation is a DB storage safeguard. Full, untruncated data is logged to logging callbacks "
    "(OTEL, Datadog, etc.). To increase the truncation limit, set `MAX_STRING_LENGTH_PROMPT_IN_DB` in your env."
)
FAKE_SECRET: Final = "sk-" + "Zq7" * 16


def _stored_prompt(row: dict[str, object]) -> str:
    messages: Final = list_value(object_value(row["proxy_server_request"])["messages"])
    return string_value(object_value(messages[0])["content"])


def test_long_prompt_is_cut_35_65_around_the_exact_truncation_marker(gateway: Gateway, tmp_path: Path) -> None:
    """With `MAX_STRING_LENGTH_PROMPT_IN_DB=100` a 1000 character prompt (`a` * 500 + `b` * 500) is stored as
    `a` * 35, then `... (litellm_truncated skipped 900 chars. <note>) ...`, then `b` * 65."""
    prompt: Final = "a" * 500 + "b" * 500
    with owned_spend_proxy(
        gateway, tmp_path, general_settings=STORE, environment={"MAX_STRING_LENGTH_PROMPT_IN_DB": "100"}
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        assert _stored_prompt(row) == (
            "a" * 35 + f"... (litellm_truncated skipped 900 chars. {TRUNCATION_NOTE}) ..." + "b" * 65
        ), row


def test_prompt_at_the_limit_is_stored_whole(gateway: Gateway, tmp_path: Path) -> None:
    """With `MAX_STRING_LENGTH_PROMPT_IN_DB=100` a prompt of exactly 100 characters is stored unchanged: the cut
    only applies to strings longer than the limit."""
    prompt: Final = "c" * 100
    with owned_spend_proxy(
        gateway, tmp_path, general_settings=STORE, environment={"MAX_STRING_LENGTH_PROMPT_IN_DB": "100"}
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=prompt)))
        assert _stored_prompt(row) == prompt, row


def test_default_truncation_limit_is_2048_characters(gateway: Gateway, tmp_path: Path) -> None:
    """With no override, a 3000 character prompt is cut to its first 716 characters (int(2048 * 0.35)) plus the
    marker plus its last 1331 characters (int(2048 * 0.65)), skipping 953 (`MAX_STRING_LENGTH_PROMPT_IN_DB`
    default, litellm/constants.py:633)."""
    prompt: Final = "".join(chr(ord("a") + index % 26) for index in range(3000))
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        stored: Final = _stored_prompt(row_by_id(owned, served_id(ask(owned, model, key, text=prompt))))
        assert stored.startswith(prompt[:716] + "... (litellm_truncated skipped 953 chars."), stored[:800]
        assert stored.endswith(") ..." + prompt[-1331:]), stored[-1400:]


def test_token_shaped_secrets_are_kept_unless_token_redaction_is_on(gateway: Gateway, tmp_path: Path) -> None:
    """Token redaction is opt-in since #51729. With prompts stored and `redact_tokens_in_spend_logs` unset, a
    prompt containing a fake `sk-` key keeps it verbatim in `proxy_server_request`."""
    with owned_spend_proxy(gateway, tmp_path, general_settings=STORE) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=f"my key is {FAKE_SECRET} ok")))
        assert FAKE_SECRET in _stored_prompt(row), row


def test_token_redaction_replaces_secrets_in_the_stored_prompt(gateway: Gateway, tmp_path: Path) -> None:
    """With `redact_tokens_in_spend_logs: true` the same prompt is stored with the key replaced by
    `REDACTED_BY_LITELM` (`_without_tokens`, spend_tracking_utils.py:1536) and the surrounding words kept."""
    with owned_spend_proxy(gateway, tmp_path, general_settings={**STORE, "redact_tokens_in_spend_logs": True}) as (
        owned,
        scenario,
        wire,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=f"my key is {FAKE_SECRET} ok")))
        stored: Final = _stored_prompt(row)
        assert FAKE_SECRET not in stored, stored
        assert stored == "my key is REDACTED_BY_LITELM ok", stored


def test_message_logging_off_replaces_the_stored_messages(gateway: Gateway, tmp_path: Path) -> None:
    """With prompts stored and `litellm_settings.turn_off_message_logging: true`, the stored request's
    `messages` is exactly `[{"role": "user", "content": "redacted-by-litellm"}]`
    (`perform_redaction`, litellm/litellm_core_utils/redact_messages.py:674); the caller's text is nowhere in the
    row, while `spend` is still 0.02."""
    secret_text: Final = f"dec1 private {uuid.uuid4().hex}"
    with owned_spend_proxy(
        gateway, tmp_path, general_settings=STORE, litellm_settings={"turn_off_message_logging": True}
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(owned, served_id(ask(owned, model, key, text=secret_text)))
        assert object_value(row["proxy_server_request"])["messages"] == [
            {"role": "user", "content": "redacted-by-litellm"}
        ], row
        assert secret_text not in str(row), "the caller's prompt survived message redaction"
        assert float(str(row["spend"])) == pytest.approx(0.02), row


def test_failure_row_error_information_has_the_documented_keys(gateway: Gateway) -> None:
    """A provider 500 writes `metadata.error_information` with `error_code` `500` (a string),
    `llm_provider` `openai`, an `error_class` naming the exception class, and an `error_message` containing the
    provider's message (`get_error_information`, litellm_logging.py:8135)."""
    with failing_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(gateway, model, key).status_code == 500
        error: Final = object_value(metadata_of(key_rows(gateway, key, 1)[0])["error_information"])
        assert error["error_code"] == "500", error
        assert error["llm_provider"] == "openai", error
        assert string_value(error["error_class"]), error
        assert "dec1 spendlogs upstream failure" in string_value(error["error_message"]), error


def test_failure_row_keeps_a_traceback_by_default(gateway: Gateway) -> None:
    """Without `LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS`, a provider 500 failure row has a `traceback` key in
    `metadata.error_information` (a string, possibly empty for expected client errors).
    UNVERIFIED: whether a 500 counts as an expected client error (`is_expected_client_error`); only the key's
    presence is asserted."""
    with failing_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ask(gateway, model, key)
        error: Final = object_value(metadata_of(key_rows(gateway, key, 1)[0])["error_information"])
        assert isinstance(error.get("traceback"), str), error


def test_traceback_suppression_drops_the_traceback_key(gateway: Gateway, tmp_path: Path) -> None:
    """With `LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS=true` (and the proxy logger above DEBUG) the failure row's
    `metadata.error_information` has no `traceback` key at all, not an empty string
    (proxy_track_cost_callback.py:529), while `error_code` is still `500`."""
    with (
        failing_upstream(500) as broken,
        owned_spend_proxy(
            gateway, tmp_path, environment={"LITELLM_SUPPRESS_SPEND_LOG_TRACEBACKS": "true", "LITELLM_LOG": "INFO"}
        ) as (owned, scenario, _),
    ):
        model: Final = chat_model(scenario, broken)
        key: Final = scenario.key(models=[model])
        assert ask(owned, model, key).status_code == 500
        error: Final = object_value(metadata_of(key_rows(owned, key, 1)[0])["error_information"])
        assert "traceback" not in error, error
        assert error["error_code"] == "500", error


def test_provider_api_key_is_never_stored_in_a_failure_row(gateway: Gateway) -> None:
    """A provider 500 failure row (prompts not stored) contains neither the deployment's provider key
    `dec1-provider-key` nor the caller's raw virtual key anywhere in its columns."""
    with failing_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        ask(gateway, model, key)
        stored: Final = str(key_rows(gateway, key, 1)[0])
        assert "dec1-provider-key" not in stored and key not in stored, stored
