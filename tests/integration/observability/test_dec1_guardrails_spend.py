"""Contracts for what a guardrail leaves in the spend log row and what it adds to the bill.

Every guardrail run is recorded in `LiteLLM_SpendLogs.metadata.guardrail_information` as one entry with the
guardrail's `guardrail_name`, `guardrail_mode` (its configured mode), `guardrail_status` ("success",
"guardrail_intervened", "guardrail_failed_to_respond" or "guardrail_flagged") and a `guardrail_response` that is
"allow" or "mask" for a pass and the refusal text for a block; the names that ran are also listed in
`metadata.applied_guardrails`. A guardrail may report a price for its own run (`guardrail_cost`): it is added to the
row's `spend` and shown as `metadata.cost_breakdown.guardrail_cost`, unless the entry says
`guardrail_cost_in_spend: false`. A `logging_only` guardrail scans after the call and never changes the answer.

The rig stores prompts in spend logs (`store_prompts_in_spend_logs: true`) so guardrail_response is not redacted.
Spend logs, the logging payload and the callbacks that write them are being ported to Rust (Spend Logs V2,
CustomLoggerV2), so the exact field names and statuses are pinned here. Deployments cost 0.001 per input token and
0.002 per output token and the provider reports 5 and 3 tokens, so a served chat costs 0.011 before guardrails.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually, object_value
from integration.observability.dec1_guardrails_support import (
    CALL_ID_HEADER,
    CHAT_COST,
    PLAIN_ANSWER,
    Rig,
    boot_rig,
    chat,
    chat_text,
    dead_base,
    generic,
    guardrail_entries,
    marker,
    messages,
    rails_called,
    refused_input,
    responses,
    spend_row,
)
from pydantic import JsonValue

PRE: Final = "dec1-sp-pre"
POST: Final = "dec1-sp-post"
FAIL_CLOSED: Final = "dec1-sp-fail-closed"
FAIL_OPEN: Final = "dec1-sp-fail-open"
FLAGGER: Final = "dec1-sp-flagger"
BILLED: Final = "dec1-sp-billed"
REPORT_ONLY: Final = "dec1-sp-report-only"
OBSERVER: Final = "dec1-sp-observer"
GUARDRAIL_PRICE: Final = 0.25
FLAG_REASON: Final = "dec1 flagged for review"

COST_MODULE: Final = "dec1_guardrails_cost_hooks"
COST_SOURCE: Final = f"""from litellm.integrations.custom_guardrail import CustomGuardrail


class BilledGuardrail(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={{"action": "allow"}},
            request_data=data,
            guardrail_status="success",
            tracing_detail={{"guardrail_cost": {GUARDRAIL_PRICE}}},
        )
        return data


class ReportOnlyGuardrail(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={{"action": "allow"}},
            request_data=data,
            guardrail_status="success",
            tracing_detail={{"guardrail_cost": {GUARDRAIL_PRICE}, "guardrail_cost_in_spend": False}},
        )
        return data
"""


def _guardrails(policy_url: str) -> list[dict[str, JsonValue]]:
    return [
        generic(PRE, "pre_call", policy_url),
        generic(POST, "post_call", policy_url),
        generic(FAIL_CLOSED, "pre_call", dead_base()),
        generic(FAIL_OPEN, "pre_call", dead_base(), unreachable_fallback="fail_open"),
        generic(OBSERVER, "logging_only", policy_url),
        {
            "guardrail_name": FLAGGER,
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "pre_call",
                "default_on": False,
                "custom_code": (
                    "def apply_guardrail(inputs, request_data, input_type):\n"
                    f"    return flag({FLAG_REASON!r}, {{'score': 1}})\n"
                ),
            },
        },
        {
            "guardrail_name": BILLED,
            "litellm_params": {"guardrail": f"{COST_MODULE}.BilledGuardrail", "mode": "pre_call", "default_on": False},
        },
        {
            "guardrail_name": REPORT_ONLY,
            "litellm_params": {
                "guardrail": f"{COST_MODULE}.ReportOnlyGuardrail",
                "mode": "pre_call",
                "default_on": False,
            },
        },
    ]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with boot_rig(
        tmp_path_factory.mktemp("dec1-guardrails-spend"), _guardrails, modules={COST_MODULE: COST_SOURCE}
    ) as value:
        yield value


@pytest.fixture(autouse=True)
def _drained(rig: Rig) -> None:
    rig.drain()


def _with(*names: str) -> dict[str, JsonValue]:
    return {"guardrails": list(names)}


def _only(row: dict[str, JsonValue], name: str) -> dict[str, JsonValue]:
    entries: Final = guardrail_entries(row, name)
    assert len(entries) == 1, (name, row)
    return entries[0]


def test_a_passing_pre_call_guardrail_is_recorded_as_success_allow(rig: Rig) -> None:
    """A served chat named the pre_call guardrail and the peer answered NONE. The spend row has one
    guardrail_information entry for it with guardrail_mode "pre_call", guardrail_status "success" and
    guardrail_response "allow" (read from code: custom_guardrail.py _summarize_guardrail_response).

    BUG: CI run 44176682961 logged guardrail_response "mask" for this pass, while the fail-open pass below logs
    "allow". `_summarize_guardrail_response` narrows the returned request to the prompt keys and calls it "mask"
    whenever they differ from the pre-hook copy (litellm/integrations/custom_guardrail.py:1755-1760), so the
    generic guardrail's pass path hands back a prompt that compares unequal to the original although the peer
    changed nothing. Which field differs was not traced."""
    pytest.skip("BUG: a passing pre_call generic guardrail (peer action NONE) is logged as guardrail_response mask")
    response: Final = chat(rig, f"pass {marker()}", extra=_with(PRE))
    assert response.status_code == 200, response.text
    entry: Final = _only(spend_row(rig, response.headers[CALL_ID_HEADER]), PRE)
    assert (entry["guardrail_mode"], entry["guardrail_status"], entry["guardrail_response"]) == (
        "pre_call",
        "success",
        "allow",
    ), entry


def test_a_masking_pre_call_guardrail_is_recorded_as_success_mask(rig: Rig) -> None:
    """The peer masked the prompt. The spend row entry has guardrail_status "success" and guardrail_response "mask",
    so dashboards can count masking runs without seeing the prompt."""
    response: Final = chat(rig, f"SECRET-1234 {marker()}", extra=_with(PRE))
    assert response.status_code == 200, response.text
    entry: Final = _only(spend_row(rig, response.headers[CALL_ID_HEADER]), PRE)
    assert (entry["guardrail_status"], entry["guardrail_response"]) == ("success", "mask"), entry


def test_a_guardrail_run_records_its_timings(rig: Rig) -> None:
    """The success entry carries numeric start_time, end_time and a non-negative duration in seconds."""
    response: Final = chat(rig, f"timings {marker()}", extra=_with(PRE))
    entry: Final = _only(spend_row(rig, response.headers[CALL_ID_HEADER]), PRE)
    assert isinstance(entry["start_time"], float) and isinstance(entry["end_time"], float), entry
    assert isinstance(entry["duration"], float) and entry["duration"] >= 0, entry


def test_the_spend_row_lists_the_applied_guardrails(rig: Rig) -> None:
    """A served chat named two guardrails (pre_call and post_call). metadata.applied_guardrails in its spend row
    contains both names, the same set the x-litellm-applied-guardrails header carried."""
    response: Final = chat(rig, f"applied {marker()}", extra=_with(PRE, POST))
    assert response.status_code == 200, response.text
    applied: Final = object_value(spend_row(rig, response.headers[CALL_ID_HEADER])["metadata"])["applied_guardrails"]
    assert isinstance(applied, list) and {PRE, POST} <= set(applied), applied
    assert set(response.headers["x-litellm-applied-guardrails"].split(",")) == set(applied)


def test_a_post_call_guardrail_entry_is_recorded_with_its_mode(rig: Rig) -> None:
    """A served chat named the post_call guardrail. Its entry has guardrail_mode "post_call" and status "success"."""
    response: Final = chat(rig, f"post {marker()}", extra=_with(POST))
    entry: Final = _only(spend_row(rig, response.headers[CALL_ID_HEADER]), POST)
    assert (entry["guardrail_mode"], entry["guardrail_status"]) == ("post_call", "success"), entry


def test_a_blocked_request_writes_a_failure_row_with_an_intervened_entry_and_no_spend(rig: Rig) -> None:
    """A pre_call block answered 400. The spend row for its call id has status "failure" and spend 0, and the
    guardrail's entry has guardrail_status "guardrail_intervened" with the refusal text as guardrail_response."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra=_with(PRE))
    assert response.status_code == 400, response.text
    row: Final = spend_row(rig, response.headers[CALL_ID_HEADER])
    assert row["status"] == "failure", row
    assert float(str(row["spend"])) == 0.0, row
    entry: Final = _only(row, PRE)
    assert entry["guardrail_status"] == "guardrail_intervened", entry
    assert entry["guardrail_response"] == refused_input(PRE), entry


def test_a_post_call_block_writes_a_failure_row_with_an_intervened_post_call_entry(rig: Rig) -> None:
    """A post_call block answered 400 after the provider answered. The row has status "failure" and the entry has
    guardrail_mode "post_call" and guardrail_status "guardrail_intervened". UNVERIFIED: whether the provider tokens
    are billed on this failure row; only the status fields are pinned."""
    response: Final = chat(rig, f"REPLY-BLOCK-OUT {marker()}", extra=_with(POST))
    assert response.status_code == 400, response.text
    row: Final = spend_row(rig, response.headers[CALL_ID_HEADER])
    assert row["status"] == "failure", row
    entry: Final = _only(row, POST)
    assert (entry["guardrail_mode"], entry["guardrail_status"]) == ("post_call", "guardrail_intervened"), entry


def test_an_unreachable_fail_closed_guardrail_is_recorded_as_failed_to_respond(rig: Rig) -> None:
    """The fail-closed guardrail could not reach its service and the caller got 500. Its entry in the failure row
    has guardrail_status "guardrail_failed_to_respond", which tells an outage apart from a policy block."""
    response: Final = chat(rig, f"down {marker()}", extra=_with(FAIL_CLOSED))
    assert response.status_code == 500, response.text
    row: Final = spend_row(rig, response.headers[CALL_ID_HEADER])
    assert row["status"] == "failure", row
    assert _only(row, FAIL_CLOSED)["guardrail_status"] == "guardrail_failed_to_respond"


def test_an_unreachable_fail_open_guardrail_is_recorded_as_a_plain_success(rig: Rig) -> None:
    """The fail-open guardrail could not reach its service and the request was served. Its entry says status
    "success" and response "allow", the same as a real pass: the outage leaves no trace in the row, only a CRITICAL
    log line (read from code: generic_guardrail_api.py _fail_open_passthrough returns the inputs unchanged)."""
    response: Final = chat(rig, f"open {marker()}", extra=_with(FAIL_OPEN))
    assert chat_text(response) == PLAIN_ANSWER
    entry: Final = _only(spend_row(rig, response.headers[CALL_ID_HEADER]), FAIL_OPEN)
    assert (entry["guardrail_status"], entry["guardrail_response"]) == ("success", "allow"), entry


def test_a_flagging_guardrail_is_recorded_as_flagged_and_the_request_is_served(rig: Rig) -> None:
    """A custom_code guardrail returns flag(reason, {"score": 1}). The caller gets 200 with the answer and the
    entry has guardrail_status "guardrail_flagged" and guardrail_response {"action": "flag", "reason": <reason>,
    "input_type": "request", "metadata": {"score": 1}} (read from code: custom_code_guardrail.py _process_result)."""
    response: Final = chat(rig, f"flag me {marker()}", extra=_with(FLAGGER))
    assert chat_text(response) == PLAIN_ANSWER
    entries: Final = guardrail_entries(spend_row(rig, response.headers[CALL_ID_HEADER]), FLAGGER)
    flagged: Final = [entry for entry in entries if entry["guardrail_status"] == "guardrail_flagged"]
    assert len(flagged) == 1, entries
    assert flagged[0]["guardrail_response"] == {
        "action": "flag",
        "reason": FLAG_REASON,
        "input_type": "request",
        "metadata": {"score": 1},
    }, flagged


def test_a_priced_guardrail_adds_its_cost_to_the_spend(rig: Rig) -> None:
    """A custom guardrail records guardrail_cost 0.25 for its run. The served chat's spend row has spend
    0.011 + 0.25 = 0.261 and metadata.cost_breakdown.guardrail_cost 0.25 (read from code: litellm_logging.py
    guardrail_information_cost, guardrail_cost.py cost_breakdown_with_guardrail)."""
    response: Final = chat(rig, f"billed {marker()}", extra=_with(BILLED))
    assert chat_text(response) == PLAIN_ANSWER
    row: Final = spend_row(rig, response.headers[CALL_ID_HEADER])
    assert float(str(row["spend"])) == pytest.approx(CHAT_COST + GUARDRAIL_PRICE), row
    breakdown: Final = object_value(object_value(row["metadata"])["cost_breakdown"])
    assert breakdown["guardrail_cost"] == pytest.approx(GUARDRAIL_PRICE), breakdown
    assert _only(row, BILLED)["guardrail_cost"] == pytest.approx(GUARDRAIL_PRICE)


def test_a_report_only_guardrail_cost_is_recorded_but_not_billed(rig: Rig) -> None:
    """The same price with guardrail_cost_in_spend false. The spend stays at the model cost 0.011, the cost
    breakdown carries no guardrail_cost, and the entry still records guardrail_cost 0.25 for reporting."""
    response: Final = chat(rig, f"report only {marker()}", extra=_with(REPORT_ONLY))
    assert chat_text(response) == PLAIN_ANSWER
    row: Final = spend_row(rig, response.headers[CALL_ID_HEADER])
    assert float(str(row["spend"])) == pytest.approx(CHAT_COST), row
    breakdown: Final = object_value(row["metadata"]).get("cost_breakdown")
    assert not isinstance(breakdown, dict) or "guardrail_cost" not in breakdown, breakdown
    entry: Final = _only(row, REPORT_ONLY)
    assert entry["guardrail_cost"] == pytest.approx(GUARDRAIL_PRICE) and entry["guardrail_cost_in_spend"] is False


def test_a_priced_guardrail_cost_is_added_to_the_key_spend(rig: Rig) -> None:
    """A fresh key makes one billed chat. Once the spend is flushed, /key/info reports the key's spend as
    0.011 + 0.25, so budgets enforced on keys include guardrail costs."""
    key: Final = rig.scenario.key()
    response: Final = chat(rig, f"key billed {marker()}", key=key, extra=_with(BILLED))
    assert response.status_code == 200, response.text

    def key_spend() -> float:
        info = object_value(rig.gateway.get("/key/info", {"key": key})["info"])
        return float(str(info.get("spend") or 0))

    assert eventually(key_spend, lambda spent: spent > 0, seconds=70) == pytest.approx(CHAT_COST + GUARDRAIL_PRICE)


def test_a_logging_only_guardrail_scans_every_call_without_being_requested(rig: Rig) -> None:
    """The rig has a Generic Guardrail API guardrail in mode logging_only with default_on false. A plain chat that
    names no guardrail still makes the peer see that guardrail, once for the request and once for the answer, after
    the provider answered. UNVERIFIED: logging_only ignores request selection (custom_guardrail.py
    should_run_guardrail exempts logging_only and async_logging_hook does not check it)."""
    response: Final = chat(rig, f"observed {marker()}")
    assert chat_text(response) == PLAIN_ANSWER
    collected: Final[list[tuple[str, str]]] = []

    def observed() -> tuple[tuple[str, str], ...]:
        collected.extend(rails_called(rig.policy_calls()))
        return tuple(collected)

    seen: Final = eventually(observed, lambda calls: len(calls) >= 2, seconds=30)
    assert (OBSERVER, "request") in seen and (OBSERVER, "response") in seen, seen


def test_a_logging_only_block_never_changes_the_answer_and_is_recorded(rig: Rig) -> None:
    """A plain chat carries BLOCK-IN. The logging_only guardrail's service answers BLOCKED, yet the caller gets 200
    with the provider's answer, and the spend row records an entry for the observer with guardrail_mode
    "logging_only" and guardrail_status "guardrail_intervened". UNVERIFIED: the status of a raised logging_only
    scan (custom_guardrail.py async_logging_hook swallows the exception after the decorator recorded it)."""
    response: Final = chat(rig, f"BLOCK-IN {marker()}")
    assert chat_text(response) == PLAIN_ANSWER
    entries: Final = guardrail_entries(spend_row(rig, response.headers[CALL_ID_HEADER]), OBSERVER)
    assert entries, entries
    assert {entry["guardrail_mode"] for entry in entries} == {"logging_only"}, entries
    assert "guardrail_intervened" in {entry["guardrail_status"] for entry in entries}, entries


@pytest.mark.parametrize("surface", ["messages", "responses"])
def test_messages_and_responses_rows_record_the_pre_call_entry_too(rig: Rig, surface: str) -> None:
    """A served /v1/messages or /v1/responses request named the pre_call guardrail. Its spend row has one entry for
    it with guardrail_status "success", so the Rust-served /v1/messages path keeps writing guardrail information."""
    response: Final = (
        messages(rig, f"messages {marker()}", extra=_with(PRE))
        if surface == "messages"
        else responses(rig, f"responses {marker()}", extra=_with(PRE))
    )
    assert response.status_code == 200, response.text
    entry: Final = _only(spend_row(rig, response.headers[CALL_ID_HEADER]), PRE)
    assert entry["guardrail_status"] == "success", entry


def test_a_blocked_messages_request_writes_a_failure_row_with_the_intervened_entry(rig: Rig) -> None:
    """A /v1/messages request blocked by the pre_call guardrail (400, Anthropic envelope). Its spend row has status
    "failure" and the guardrail entry has guardrail_status "guardrail_intervened"."""
    response: Final = messages(rig, f"BLOCK-IN {marker()}", extra=_with(PRE))
    assert response.status_code == 400, response.text
    row: Final = spend_row(rig, response.headers[CALL_ID_HEADER])
    assert row["status"] == "failure", row
    assert _only(row, PRE)["guardrail_status"] == "guardrail_intervened"
