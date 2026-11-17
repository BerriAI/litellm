"""Contract: the spend log rows of retried, fallen back, timed out and cancelled calls.

The Spend Logs V2 port subscribes to `attempt_ended` and `post_call_failure`. Its plan states the parity rule
this file pins: a failed attempt that later succeeds (by retry or by fallback) is charged only by the success
row, so `attempt_ended` with a failure outcome must be ignored, or every recovered call grows a second row.

The other half is the failure side (`litellm/proxy/hooks/proxy_track_cost_callback.py:318-420`): when every
attempt fails and earlier attempts ended in a `Timeout`, the provider accepted (and bills) those prompts, so the
proxy writes the final failure row under the request's call id plus one row per earlier billed attempt with
request id `<call id>-attempt-<n>`, metadata `billed_retry_attempt: true` and `parent_request_id`. The daily
tables keep those rows' spend and tokens but do not count them as requests (`litellm/accounting/projection.py:145`).

A stream the caller abandons is written with `metadata.error_information` `{error_code: "499", error_message:
"Client disconnected the request", error_class: "ClientDisconnected"}`
(`litellm/proxy/common_request_processing.py:389`).

Primary deployments here cost 0.001 / 0.002 per token (a served chat call is 0.02); fallback deployments are
priced at 0.002 / 0.004 so a fallback charge (0.04) is distinguishable.
"""

from __future__ import annotations

import json
import uuid
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, Scenario, eventually, held, object_value, string_value
from tests.integration._support.wire import Wire
from tests.integration.spend.dec1_spend_support import (
    LANDING_SECONDS,
    ask,
    chat_model,
    daily_row,
    hashed,
    metadata_of,
    number,
    row_by_id,
    scripted_upstream,
    served,
    served_id,
    spend_reaches,
)
from tests.integration.spend.dec1_spendlogs_support import (
    failing_upstream,
    flaky_upstream,
    hanging_upstream,
    key_rows,
    slow_stream_upstream,
)

pytestmark: Final = pytest.mark.timeout(300)

FALLBACK_COST: Final = 10 * 0.002 + 5 * 0.004


def _fallback_pair(scenario: Scenario, broken: Wire, healthy: Wire) -> tuple[str, str]:
    """A primary group whose provider always answers 500 and a fallback group on a healthy provider."""
    primary: Final = chat_model(scenario, broken)
    backup: Final = chat_model(scenario, healthy, input_cost_per_token=0.002, output_cost_per_token=0.004)
    return primary, backup


def _fallback_call(gateway: Gateway, primary: str, backup: str, key: str) -> httpx.Response:
    return ask(gateway, primary, key, fallbacks=[backup], num_retries=0)


def _no_more_rows(gateway: Gateway, key: str, count: int) -> list[dict[str, object]]:
    return held(
        lambda: gateway.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)),
        lambda found: len(found) == count,
        holding=5,
        seconds=LANDING_SECONDS,
    )


def test_a_call_saved_by_a_fallback_writes_exactly_one_row(gateway: Gateway) -> None:
    """The primary group's provider answers 500 and the request names a healthy fallback group. The caller gets
    200, and the key ends with exactly one `LiteLLM_SpendLogs` row, status `success`, for five seconds after it
    lands: the failed primary attempt gets no row of its own."""
    with failing_upstream(500) as broken, scripted_upstream() as healthy, gateway.scenario() as scenario:
        primary, backup = _fallback_pair(scenario, broken, healthy)
        key: Final = scenario.key(models=[primary, backup])
        served(_fallback_call(gateway, primary, backup, key))
        rows: Final = key_rows(gateway, key, 1)
        assert rows[0]["status"] == "success", rows
        _no_more_rows(gateway, key, 1)


def test_fallback_row_model_group_is_the_group_that_served(gateway: Gateway) -> None:
    """After a fallback, the row's `model_group` is the fallback group that answered, while
    `metadata.original_model_group` keeps the group the caller asked for and `metadata.attempted_fallbacks` is 1
    (stamped per hop in `litellm/router_utils/fallback_event_handlers.py:934-939`)."""
    with failing_upstream(500) as broken, scripted_upstream() as healthy, gateway.scenario() as scenario:
        primary, backup = _fallback_pair(scenario, broken, healthy)
        key: Final = scenario.key(models=[primary, backup])
        row: Final = row_by_id(gateway, served_id(_fallback_call(gateway, primary, backup, key)))
        metadata: Final = metadata_of(row)
        assert row["model_group"] == backup, row
        assert metadata["original_model_group"] == primary, metadata
        assert metadata["attempted_fallbacks"] == 1, metadata


def test_fallback_row_names_the_fallback_deployment_and_its_header(gateway: Gateway) -> None:
    """The fallback response carries `x-litellm-model-id` of the backup deployment and
    `x-litellm-attempted-fallbacks: 1`. The row's `model_id` equals that header and its `api_base` is the
    healthy provider, never the broken one."""
    with failing_upstream(500) as broken, scripted_upstream() as healthy, gateway.scenario() as scenario:
        primary, backup = _fallback_pair(scenario, broken, healthy)
        key: Final = scenario.key(models=[primary, backup])
        response: Final = _fallback_call(gateway, primary, backup, key)
        row: Final = row_by_id(gateway, served_id(response))
        assert response.headers["x-litellm-attempted-fallbacks"] == "1", dict(response.headers)
        assert row["model_id"] == response.headers["x-litellm-model-id"], row
        assert string_value(row["api_base"]).startswith(healthy.url), row


def test_fallback_row_is_priced_at_the_fallback_deployments_rates(gateway: Gateway) -> None:
    """The backup deployment costs 0.002 / 0.004 per token. The fallback row's `spend` is
    10 * 0.002 + 5 * 0.004 = 0.04, and the key's spend reaches exactly 0.04, not 0.04 plus a charge for the
    failed primary attempt."""
    with failing_upstream(500) as broken, scripted_upstream() as healthy, gateway.scenario() as scenario:
        primary, backup = _fallback_pair(scenario, broken, healthy)
        key: Final = scenario.key(models=[primary, backup])
        row: Final = row_by_id(gateway, served_id(_fallback_call(gateway, primary, backup, key)))
        assert number(row["spend"]) == pytest.approx(FALLBACK_COST), row
        assert spend_reaches(gateway, "key", hashed(key), FALLBACK_COST) == pytest.approx(FALLBACK_COST)


def test_fallback_counts_as_one_successful_request_in_the_daily_row(gateway: Gateway) -> None:
    """The daily user row after one fallback-saved call has `api_requests` 1, `successful_requests` 1 and
    `failed_requests` 0, keyed by the backup deployment's model."""
    with failing_upstream(500) as broken, scripted_upstream() as healthy, gateway.scenario() as scenario:
        primary, backup = _fallback_pair(scenario, broken, healthy)
        key: Final = scenario.key(models=[primary, backup])
        served(_fallback_call(gateway, primary, backup, key))
        daily: Final = daily_row(gateway, "LiteLLM_DailyUserSpend", key)
        assert (daily["api_requests"], daily["successful_requests"], daily["failed_requests"]) == (1, 1, 0), daily


def test_a_retry_that_succeeds_writes_one_success_row_billed_once(gateway: Gateway) -> None:
    """The provider fails the first call with 500 and answers the second. With `num_retries: 1` the caller gets
    200; the key has exactly one row, status `success`, spend 0.02, and the key's spend is 0.02. A 500 is not
    a billed attempt (only timeouts are, `litellm/proxy/utils.py:1544`), so no `-attempt-` row appears."""
    with flaky_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, num_retries=1))
        rows: Final = key_rows(gateway, key, 1)
        assert rows[0]["status"] == "success" and number(rows[0]["spend"]) == pytest.approx(0.02), rows
        assert "-attempt-" not in string_value(rows[0]["request_id"]), rows
        _no_more_rows(gateway, key, 1)
        assert spend_reaches(gateway, "key", hashed(key), 0.02) == pytest.approx(0.02)


def test_a_retried_call_reports_its_retry_in_the_response_header(gateway: Gateway) -> None:
    """The same flaky provider with `num_retries: 1`: the 200 response carries `x-litellm-attempted-retries: 1`,
    and the row's `metadata.attempted_retries` is 1.
    UNVERIFIED: the metadata stamp now comes from the Rust router core (`router-core/src/call.rs:49`); the
    header is set in `litellm/litellm_core_utils/add_retry_fallback_headers.py:320`."""
    with flaky_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key, num_retries=1)
        row: Final = row_by_id(gateway, served_id(response))
        assert response.headers["x-litellm-attempted-retries"] == "1", dict(response.headers)
        assert metadata_of(row)["attempted_retries"] == 1, row


def _timed_out(gateway: Gateway, scenario: Scenario, wire: Wire) -> tuple[str, str, list[dict[str, object]]]:
    """A deployment with `timeout: 1` and `num_retries: 2` on a provider that never answers. Returns the key,
    the `x-litellm-call-id` of the refused call and the key's three rows."""
    model: Final = chat_model(scenario, wire, timeout=1, num_retries=2)
    key: Final = scenario.key(models=[model])
    response: Final = ask(gateway, model, key)
    assert response.status_code in (408, 500), response.text
    call_id: Final = response.headers["x-litellm-call-id"]
    return key, call_id, key_rows(gateway, key, 3)


def test_timed_out_retries_write_one_row_per_billed_attempt(gateway: Gateway) -> None:
    """Every attempt of a `num_retries: 2` deployment times out. The key gets exactly three rows: the final
    failure row with `request_id` equal to the call id, and `<call id>-attempt-0` and `<call id>-attempt-1` for
    the two earlier timed-out attempts (`_attempt_spend_row_kwargs`, proxy_track_cost_callback.py:365)."""
    with hanging_upstream() as wire, gateway.scenario() as scenario:
        _, call_id, rows = _timed_out(gateway, scenario, wire)
        assert sorted(string_value(row["request_id"]) for row in rows) == sorted(
            (call_id, f"{call_id}-attempt-0", f"{call_id}-attempt-1")
        ), rows
        assert {row["status"] for row in rows} == {"failure"}, rows


def test_billed_attempt_rows_point_at_their_parent_request(gateway: Gateway) -> None:
    """Each `-attempt-N` row has `metadata.billed_retry_attempt` true, `metadata.parent_request_id` equal to the
    call id, and `litellm_call_id` equal to its own request id; the final row has no `billed_retry_attempt`."""
    with hanging_upstream() as wire, gateway.scenario() as scenario:
        _, call_id, rows = _timed_out(gateway, scenario, wire)
        attempts: Final = [row for row in rows if string_value(row["request_id"]) != call_id]
        final: Final = next(row for row in rows if row["request_id"] == call_id)
        for row in attempts:
            metadata = metadata_of(row)
            assert metadata["billed_retry_attempt"] is True, metadata
            assert metadata["parent_request_id"] == call_id, metadata
            assert row["litellm_call_id"] == row["request_id"], row
        assert metadata_of(final).get("billed_retry_attempt") is not True, final


def test_billed_attempt_rows_charge_the_estimated_prompt_and_no_output(gateway: Gateway) -> None:
    """A timed-out attempt is billed at its deployment's input rate for the estimated prompt: each `-attempt-N`
    row has `prompt_tokens` above 0, `completion_tokens` 0 and `spend` equal to `prompt_tokens * 0.001`
    (`_billed_earlier_attempt`, litellm/proxy/utils.py:1524)."""
    with hanging_upstream() as wire, gateway.scenario() as scenario:
        _, call_id, rows = _timed_out(gateway, scenario, wire)
        for row in (row for row in rows if row["request_id"] != call_id):
            prompt = number(row["prompt_tokens"])
            assert prompt > 0 and row["completion_tokens"] == 0, row
            assert number(row["spend"]) == pytest.approx(prompt * 0.001), row


def test_billed_attempts_add_spend_but_not_requests_to_the_daily_row(gateway: Gateway) -> None:
    """After the three-row timeout, the key's daily user row counts `api_requests` 1 and `failed_requests` 1
    (only the caller's request), while its `spend` equals the sum of all three rows' spend."""
    with hanging_upstream() as wire, gateway.scenario() as scenario:
        key, _, rows = _timed_out(gateway, scenario, wire)
        total: Final = sum(number(row["spend"]) for row in rows)
        daily: Final = eventually(
            lambda: daily_row(gateway, "LiteLLM_DailyUserSpend", key),
            lambda found: number(found["spend"]) >= total - 1e-12,
            seconds=LANDING_SECONDS,
        )
        assert (daily["api_requests"], daily["failed_requests"]) == (1, 1), daily
        assert number(daily["spend"]) == pytest.approx(total), daily


def test_billed_attempt_rows_charge_the_key_with_every_attempt(gateway: Gateway) -> None:
    """The key's `spend` reaches the sum of the three rows' `spend`: billed attempts are real provider charges and
    move budgets, as the budget test in `test_partial_stream_live_budget.py` relies on."""
    with hanging_upstream() as wire, gateway.scenario() as scenario:
        key, _, rows = _timed_out(gateway, scenario, wire)
        total: Final = sum(number(row["spend"]) for row in rows)
        assert total > 0, rows
        assert spend_reaches(gateway, "key", hashed(key), total) == pytest.approx(total)


def test_a_provider_error_without_retries_writes_no_attempt_rows(gateway: Gateway) -> None:
    """A deployment with `num_retries: 2` whose provider always answers 500 (not a timeout) fails. The key gets
    exactly one failure row, under the call id, and no `-attempt-` rows, because only timed-out earlier attempts
    are billed."""
    with failing_upstream(500) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire, num_retries=2)
        key: Final = scenario.key(models=[model])
        response: Final = ask(gateway, model, key)
        assert response.status_code == 500, response.text
        rows: Final = key_rows(gateway, key, 1)
        assert rows[0]["request_id"] == response.headers["x-litellm-call-id"], rows
        _no_more_rows(gateway, key, 1)


def _slow_stream_cancelled(gateway: Gateway, model: str, key: str) -> str:
    """Open a chat stream, read the first data line, then close the connection. Returns the chunk id."""
    body: Final = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": f"dec1 cancel {uuid.uuid4().hex}"}],
    }
    with gateway.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        for line in response.iter_lines():
            if line.startswith("data: "):
                return string_value(object_value(json.loads(line[6:]))["id"])
    raise AssertionError("the stream ended before its first data line")


def test_a_cancelled_stream_writes_one_row_marked_client_disconnected(gateway: Gateway) -> None:
    """The provider streams 40 chunks 0.25 s apart; the caller reads the first one and hangs up. The key gets
    exactly one row and its `metadata.error_information` has `error_code` `499`, `error_class`
    `ClientDisconnected` and `error_message` `Client disconnected the request`.
    UNVERIFIED: whether the row's `status` is `success` (written by the success path with the partial usage)
    or `failure`; it is not asserted here and should be pinned once observed."""
    with slow_stream_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        _slow_stream_cancelled(gateway, model, key)
        row: Final = key_rows(gateway, key, 1)[0]
        error: Final = object_value(metadata_of(row)["error_information"])
        assert error["error_code"] == "499", error
        assert error["error_class"] == "ClientDisconnected", error
        assert error["error_message"] == "Client disconnected the request", error
        _no_more_rows(gateway, key, 1)


def test_a_cancelled_stream_row_keeps_the_stream_id_as_request_id(gateway: Gateway) -> None:
    """The cancelled stream's row has `request_id` equal to the `id` of the chunk the caller did receive, so a
    caller can find the charge for a stream it abandoned."""
    with slow_stream_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        stream_id: Final = _slow_stream_cancelled(gateway, model, key)
        row: Final = key_rows(gateway, key, 1)[0]
        assert row["request_id"] == stream_id, row
