"""Abortable callbacks run one at a time in config order, and their Abort, Respond and Reroute outcomes have the
effects the CustomLoggerV2 contract (#51214, #51526) gives them on the proxy. The Rust ports of budgets, the
limiter and model capacity (#51535, #51537, #51540) are all abortable pre_call callbacks, so these are the rules
they will be scheduled by.

Contract under test, read from litellm/integrations/custom_logger_v2/planner.py and dispatcher.py:

- On pre_call, every handler declared `may_abort=True` starts its own serial step after everything listed before
  it (planner.py:44-47); a legacy pre-call hook counts as an abortable writer of everything (legacy.py:62-73).
  Steps run in list order and each step's patches are applied before the next starts (dispatcher.py:378-390).
- An Abort or Respond is terminal: later steps, inline loggers, awaited and off-path readers do not run
  (dispatcher.py:393-399, 254-256). At pre_call, during_call and post_call_success an Abort becomes
  `HTTPException(status, {"error": message}, headers)` (proxy/utils.py:2235-2246); `code` and `body` are not used.
- Respond is not wired: `kwargs_after` raises `V2Outcome` and nothing catches it (legacy.py:427-434, user-guide.md
  line 8). Reroute is stored on the scope and nobody reads it (dispatcher.py:258-260, user-guide.md line 9).
- On post_call_failure an Abort replaces the error the caller sees, the first one in list order winning
  (proxy/utils.py:2249-2256, 2146-2148, common_request_processing.py:5236-5243), and an exception raised by a
  handler there or on response_headers is swallowed (dispatcher.py:34, 81-95).

Every rig uses the probe instances of `dec1_cblv2b_probe`, listed so the legacy gate comes first, then `gate_y`,
then `gate_x`: the order is the config order, not the name order.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Scenario
from integration.observability.dec1_cblv2_support import (
    CHAT,
    CHAT_REJECTED,
    PROVIDER_REJECTION,
    REPLY_TEXT,
    Rig,
    cblv2_rig,
    chat,
    error_message,
    marker,
    reply_text,
    skip_until_v2_instances_load,
)
from integration.observability.dec1_cblv2b_probe import (
    ABORT_AT,
    ABORT_CODE,
    ABORT_CODE_AT,
    ABORT_HEADER,
    ABORT_STATUS,
    DURING_ABORT,
    DURING_ABORT_STATUS,
    FAILURE_ABORT_AT,
    FAILURE_RAISE,
    FAILURE_STATUSES,
    HEADERS_RAISE,
    ORDER,
    POST_ABORT,
    POST_ABORT_STATUS,
    REJECT_AT,
    REJECT_STATUS,
    REROUTE_AT,
    RESPOND_AT,
    RESPOND_TEXT,
    abort_message,
    failure_message,
)
from integration.observability.dec1_cblv2b_support import instances, loggers_called, new_key, records

CALLBACKS: Final = instances(
    "legacy_gate",
    "gate_y",
    "gate_x",
    "post_gate",
    "during_gate",
    "failure_gate_a",
    "failure_gate_b",
    "headers_raiser",
    "required_reader",
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    skip_until_v2_instances_load()
    directory: Final = tmp_path_factory.mktemp("cblv2b-outcomes")
    with cblv2_rig(directory, {"callbacks": CALLBACKS}) as booted:
        yield booted


@pytest.fixture(scope="module")
def scenario(rig: Rig) -> Iterator[Scenario]:
    with rig.proxy.scenario() as opened:
        yield opened


def test_abortable_pre_call_handlers_run_serially_in_config_order(rig: Rig) -> None:
    """The config lists `legacy_gate`, `gate_y`, `gate_x`. One plain chat must call the three pre-call handlers one
    after the other in exactly that order (by the probe's per-process sequence), because each may abort and so
    starts its own serial step after everything listed before it (planner.py:44-47)."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    records(rig, text, "gate_x", "pre_call", 1)
    order: Final = tuple(logger for logger, hook in rig.calls(text) if hook in {"pre_call", "async_pre_call_hook"})
    assert order == ("legacy_gate", "gate_y", "gate_x")


def test_each_abortable_handler_sees_the_patch_of_the_one_before(rig: Rig) -> None:
    """With the `cblv2b-order` directive every gate appends ` [<its name>]` to the last message. The provider must
    receive the text with the three tags in config order, and the record of `gate_x` must already show the tags of
    `legacy_gate` and `gate_y`: a step's patches are applied before the next step runs (dispatcher.py:378-390)."""
    text: Final = f"{ORDER} {marker()}"
    rig.provider_texts()
    assert chat(rig, text).status_code == 200
    assert rig.provider_texts() == (f"{text} [legacy_gate] [gate_y] [gate_x]",)
    seen_by_x: Final = str(records(rig, text, "gate_x", "pre_call", 1)[0]["text"])
    assert "[legacy_gate] [gate_y]" in seen_by_x
    assert "[gate_x]" not in seen_by_x


def test_abort_by_the_first_v2_gate_skips_the_second(rig: Rig) -> None:
    """`gate_y` returns Abort(452) on `cblv2b-abort-at-gate_y`. The caller must get 452 with error.message
    "aborted by the cblv2b gate gate_y" and the header `x-cblv2b-aborted-by: gate_y`, and `gate_x`, listed after
    it, must never be called for that request (terminal outcome, dispatcher.py:393-399)."""
    text: Final = f"{ABORT_AT}gate_y {marker()}"
    response: Final = chat(rig, text)
    assert response.status_code == ABORT_STATUS, response.text
    assert error_message(response) == abort_message("gate_y")
    assert response.headers.get(ABORT_HEADER) == "gate_y"
    records(rig, text, "gate_y", "pre_call", 1)
    assert records(rig, text, "gate_x", "pre_call", 0) == ()


def test_rejection_by_a_legacy_hook_listed_first_skips_both_v2_gates(rig: Rig) -> None:
    """`legacy_gate` raises HTTPException(400) on `cblv2b-reject-at-legacy_gate`. The caller must get 400 and
    neither `gate_y` nor `gate_x` may be called: a legacy pre-call hook is an abortable step like any other."""
    text: Final = f"{REJECT_AT}legacy_gate {marker()}"
    rig.provider_texts()
    response: Final = chat(rig, text)
    assert response.status_code == REJECT_STATUS, response.text
    records(rig, text, "legacy_gate", "async_pre_call_hook", 1)
    assert records(rig, text, "gate_y", "pre_call", 0) == ()
    assert records(rig, text, "gate_x", "pre_call", 0) == ()
    assert rig.provider_texts() == ()


def test_abort_by_the_last_gate_discards_the_earlier_patches_and_the_provider_call(rig: Rig) -> None:
    """With both `cblv2b-order` and `cblv2b-abort-at-gate_x`, `legacy_gate` and `gate_y` tag the request and then
    `gate_x` aborts. The caller must get 452 from gate_x and the provider must receive nothing."""
    text: Final = f"{ORDER} {ABORT_AT}gate_x {marker()}"
    rig.provider_texts()
    response: Final = chat(rig, text)
    assert response.status_code == ABORT_STATUS, response.text
    assert error_message(response) == abort_message("gate_x")
    assert rig.provider_texts() == ()


def test_abort_skips_the_readers_of_that_attempt(rig: Rig) -> None:
    """An aborted request never reaches a deployment, so `required_reader` (an `attempt_ended` reader that runs for
    every request) must get no attempt for it, while a plain chat gives it exactly one with outcome "success".
    UNVERIFIED: assumes no attempt_ended is dispatched for a request that ended before routing."""
    aborted: Final = f"{ABORT_AT}gate_y {marker()}"
    served: Final = marker()
    assert chat(rig, aborted).status_code == ABORT_STATUS
    assert chat(rig, served).status_code == 200
    assert records(rig, served, "required_reader", "attempt_ended", 1)[0]["outcome"] == "success"
    assert records(rig, aborted, "required_reader", "attempt_ended", 0) == ()


def test_gate_still_aborts_a_no_log_request(rig: Rig) -> None:
    """A `no-log: true` request carrying `cblv2b-abort-at-gate_y` must still be refused with 452. A handler that
    writes or may abort cannot declare honors_no_log or caller opt-out (manifest.py:40-41, 65-69), which is what
    keeps the Rust limiter, budget and capacity callbacks running for no-log traffic."""
    response: Final = chat(rig, f"{ABORT_AT}gate_y {marker()}", **{"no-log": True})
    assert response.status_code == ABORT_STATUS, response.text
    assert error_message(response) == abort_message("gate_y")


def test_abort_code_reaches_the_caller(rig: Rig) -> None:
    """`gate_y` returns Abort(452, message, code="cblv2b_policy", body={"reason": "cblv2b"}). The caller should get
    error.code "cblv2b_policy". Today v2_abort_error (proxy/utils.py:2235-2240) builds the HTTPException from
    status, message and headers only, so code and body are dropped."""
    pytest.skip("BUG: Abort.code and Abort.body are ignored by v2_abort_error on the proxy")
    response: Final = chat(rig, f"{ABORT_CODE_AT}gate_y {marker()}")
    assert response.status_code == ABORT_STATUS
    assert response.json()["error"]["code"] == ABORT_CODE


def test_respond_answers_the_caller_without_the_provider(rig: Rig) -> None:
    """`gate_y` returns Respond with one assistant message. The intended contract is a 200 whose reply text is that
    message. Today `kwargs_after` raises V2Outcome and nothing catches it (legacy.py:431-432, user-guide.md:8)."""
    pytest.skip("BUG: Respond at proxy pre_call raises an uncaught V2Outcome instead of answering")
    response: Final = chat(rig, f"{RESPOND_AT}gate_y {marker()}")
    assert response.status_code == 200, response.text
    assert reply_text(response) == RESPOND_TEXT


def test_respond_never_reaches_the_provider_or_the_next_gate(rig: Rig) -> None:
    """Whatever the caller gets for a Respond today, the request must not be sent upstream and `gate_x` must not
    run: Respond is terminal in the dispatcher (dispatcher.py:393-399). UNVERIFIED: the status of the uncaught
    V2Outcome (expected a 500 from the generic exception path) is not asserted."""
    text: Final = f"{RESPOND_AT}gate_y {marker()}"
    rig.provider_texts()
    assert chat(rig, text).status_code != 200
    assert rig.provider_texts() == ()
    assert records(rig, text, "gate_x", "pre_call", 0) == ()


def test_reroute_is_ignored_and_the_requested_model_answers(rig: Rig) -> None:
    """`gate_y` returns Reroute(model="cblv2-chat-rejected"), whose deployment always answers 400. Reroute is read
    by nobody on main (dispatcher.py:258-260, user-guide.md:9), so the call must still go to `cblv2-chat` and return
    200 with the scripted reply. Pinned so the change happens on purpose when Reroute is wired."""
    response: Final = chat(rig, f"{REROUTE_AT}gate_y {marker()}", model=CHAT)
    assert response.status_code == 200, response.text
    assert reply_text(response) == REPLY_TEXT


def test_reroute_sends_the_call_to_the_named_model(rig: Rig) -> None:
    """Intended contract: a Reroute to `cblv2-chat-rejected` sends the call there, so the caller gets that
    deployment's 400 with the scripted rejection message."""
    pytest.skip("BUG: Reroute is stored on the request scope and never applied (user-guide.md:9)")
    response: Final = chat(rig, f"{REROUTE_AT}gate_y {marker()}", model=CHAT)
    assert response.status_code == 400
    assert PROVIDER_REJECTION in error_message(response)


def test_post_call_success_abort_replaces_a_served_reply(rig: Rig) -> None:
    """`post_gate` aborts with 453 at post_call_success on `cblv2b-post-abort`. The provider must have received the
    request, and the caller must get 453 with error.message "aborted by the cblv2b gate post_gate" instead of the
    reply (proxy/utils.py:4781-4793 raise_v2_abort)."""
    text: Final = f"{POST_ABORT} {marker()}"
    rig.provider_texts()
    response: Final = chat(rig, text)
    assert response.status_code == POST_ABORT_STATUS, response.text
    assert error_message(response) == abort_message("post_gate")
    assert rig.provider_texts() == (text,)


def test_during_call_abort_fails_the_request(rig: Rig) -> None:
    """`during_gate` aborts with 454 at during_call on `cblv2b-during-abort` (a parallel step by default,
    logger.py:30). The caller must get 454 with that message (proxy/utils.py:4107-4113). UNVERIFIED: during_call
    runs next to the provider call, so whether the provider saw the request is not asserted."""
    response: Final = chat(rig, f"{DURING_ABORT} {marker()}")
    assert response.status_code == DURING_ABORT_STATUS, response.text
    assert error_message(response) == abort_message("during_gate")


def test_post_call_failure_abort_replaces_the_error_the_caller_sees(rig: Rig) -> None:
    """A chat to `cblv2-chat-rejected` (provider 400) carrying `cblv2b-failure-abort-at-failure_gate_a` must answer
    503 with error.message "replaced by the cblv2b failure gate failure_gate_a": the Abort is kept as the
    transformed error (proxy/utils.py:2249-2256) and the endpoint raises it instead of the original."""
    response: Final = chat(rig, f"{FAILURE_ABORT_AT}failure_gate_a {marker()}", model=CHAT_REJECTED)
    assert response.status_code == FAILURE_STATUSES["failure_gate_a"], response.text
    assert error_message(response) == failure_message("failure_gate_a")


def test_first_post_call_failure_abort_in_config_order_wins(rig: Rig) -> None:
    """Both failure gates abort (503 and 504) when the text names both. `failure_gate_a` is listed first, so the
    caller must get 503 and `failure_gate_b` must never be called for that request."""
    text: Final = f"{FAILURE_ABORT_AT}failure_gate_a {FAILURE_ABORT_AT}failure_gate_b {marker()}"
    response: Final = chat(rig, text, model=CHAT_REJECTED)
    assert response.status_code == FAILURE_STATUSES["failure_gate_a"], response.text
    records(rig, text, "failure_gate_a", "post_call_failure", 1)
    assert records(rig, text, "failure_gate_b", "post_call_failure", 0) == ()


def test_post_call_failure_handlers_see_the_provider_error_class(rig: Rig) -> None:
    """For a plain provider 400 both failure gates must be called once each, in config order, and see the error as
    `BadRequestError` with status 400 (litellm_core_utils/exception_mapping_utils.py:503-511)."""
    text: Final = marker()
    assert chat(rig, text, model=CHAT_REJECTED).status_code == 400
    first: Final = records(rig, text, "failure_gate_a", "post_call_failure", 1)[0]
    records(rig, text, "failure_gate_b", "post_call_failure", 1)
    assert (first["error"], first["status"]) == ("BadRequestError", 400)
    assert loggers_called(rig, text, "post_call_failure") == ("failure_gate_a", "failure_gate_b")


def test_crashing_post_call_failure_handler_keeps_the_original_error(rig: Rig) -> None:
    """With `cblv2b-failure-raise` both failure gates raise. post_call_failure swallows handler exceptions
    (SWALLOWING_HOOKS, dispatcher.py:34), so the caller must still get the provider's 400 with its message."""
    response: Final = chat(rig, f"{FAILURE_RAISE} {marker()}", model=CHAT_REJECTED)
    assert response.status_code == 400, response.text
    assert PROVIDER_REJECTION in error_message(response)


def test_post_call_failure_abort_also_replaces_a_budget_refusal(rig: Rig, scenario: Scenario) -> None:
    """A key with `max_budget: 0` is refused at auth with 422. That refusal goes through post_call_failure_hook
    with the parsed body (auth_exception_handler.py:285-299), so `cblv2b-failure-abort-at-failure_gate_a` must turn
    it into 503 with the failure gate's message. UNVERIFIED: assumes the messages reach the V2 view there."""
    key, _, _ = new_key(scenario, max_budget=0)
    response: Final = chat(rig, f"{FAILURE_ABORT_AT}failure_gate_a {marker()}", key=key)
    assert response.status_code == FAILURE_STATUSES["failure_gate_a"], response.text
    assert error_message(response) == failure_message("failure_gate_a")


def test_crashing_response_headers_handler_never_fails_the_request(rig: Rig) -> None:
    """`headers_raiser` raises at response_headers on `cblv2b-headers-raise`. response_headers swallows handler
    exceptions (dispatcher.py:34), so the chat must return 200 with the scripted reply; a plain chat carries the
    header `x-cblv2b-headers: seen` that the same handler merges when it does not raise."""
    raised: Final = chat(rig, f"{HEADERS_RAISE} {marker()}")
    assert raised.status_code == 200, raised.text
    assert reply_text(raised) == REPLY_TEXT
    plain: Final = chat(rig, marker())
    assert plain.headers.get("x-cblv2b-headers") == "seen"
