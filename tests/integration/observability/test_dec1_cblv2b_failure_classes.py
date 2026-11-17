"""Failure callbacks run for every class of error a proxy request can end in, and see the error as the class the
caller's status comes from. When budgets, the limiter and the callbacks dispatch move to Rust, every refusal must
keep reaching `async_post_call_failure_hook` (and V2 `post_call_failure`), and only errors from a model call may
reach `async_log_failure_event`.

Error classes covered, with where each is raised and how it reaches the hook:

- provider 400, 401, 429 and 500, mapped to BadRequestError, AuthenticationError, RateLimitError and
  InternalServerError (litellm_core_utils/exception_mapping_utils.py:503-562), then the endpoint's exception path
  (common_request_processing.py:5236-5243);
- a budget refusal at auth, BudgetExceededError with status 422 (exceptions.py:1149-1175, litellm/__init__.py:393,
  auth_exception_handler.py:285-299);
- a limiter refusal after the body (per-model window), ProxyRateLimitError 429
  (proxy/hooks/parallel_request_limiter_v3.py:3730-3743);
- a limiter refusal before the body on a cached key (#51428), the same class but with `request_data={}`
  (proxy/auth/user_api_key_auth.py `_reject_from_headers_before_reading_body`);
- an invalid virtual key, a 401 from auth.

The legacy `failure_ledger` records the hook, the key alias, the exception class name and its status; the V2
`failure_gate_a` (no directive, so it never changes the outcome) records the class it sees on `post_call_failure`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Scenario, eventually
from integration.observability.dec1_cblv2_support import (
    CHAT,
    CHAT_DENIED,
    CHAT_LIMITED,
    CHAT_OUTAGE,
    CHAT_REJECTED,
    Rig,
    cblv2_rig,
    chat,
    marker,
    stream_chat,
)
from integration.observability.dec1_cblv2b_support import instances, new_key, records
from pydantic import JsonValue

PROVIDER_ERRORS: Final = (
    pytest.param(CHAT_REJECTED, 400, "BadRequestError", id="400"),
    pytest.param(CHAT_DENIED, 401, "AuthenticationError", id="401"),
    pytest.param(CHAT_LIMITED, 429, "RateLimitError", id="429"),
    pytest.param(CHAT_OUTAGE, 500, "InternalServerError", id="500"),
)
HOOK: Final = "async_post_call_failure_hook"
LOG: Final = "async_log_failure_event"


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("cblv2b-failure-classes")
    with cblv2_rig(
        directory,
        {"callbacks": instances("failure_ledger", "failure_gate_a", "required_reader")},
        router_settings={"disable_cooldowns": True},
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def scenario(rig: Rig) -> Iterator[Scenario]:
    with rig.proxy.scenario() as opened:
        yield opened


def _by_alias(rig: Rig, alias: str, hook: str, count: int) -> tuple[dict[str, JsonValue], ...]:
    """Ledger records of `hook` whose key alias is `alias`, once exactly `count` arrived and stayed for 3 s."""
    return eventually(
        lambda: tuple(
            record
            for record in rig.probe.records()
            if record.get("logger") == "failure_ledger"
            and record.get("hook") == hook
            and record.get("key_alias") == alias
        ),
        lambda found: len(found) == count,
        seconds=30,
    )


@pytest.mark.parametrize(("model", "status", "error_class"), PROVIDER_ERRORS)
def test_provider_error_answers_the_caller_with_the_provider_status(
    rig: Rig, model: str, status: int, error_class: str
) -> None:
    """A chat to a deployment whose provider answers `status` must answer the caller with that same status, since
    num_retries is 0 and the mapped exception keeps the provider's status code."""
    assert chat(rig, marker(), model=model).status_code == status


@pytest.mark.parametrize(("model", "status", "error_class"), PROVIDER_ERRORS)
def test_provider_error_reaches_the_legacy_failure_hook_once_with_its_class(
    rig: Rig, model: str, status: int, error_class: str
) -> None:
    """The legacy `async_post_call_failure_hook` must be called exactly once for the request with the mapped class
    name (`BadRequestError`, `AuthenticationError`, `RateLimitError`, `InternalServerError`) and that status."""
    text: Final = marker()
    assert chat(rig, text, model=model).status_code == status
    found: Final = records(rig, text, "failure_ledger", HOOK, 1)[0]
    assert (found["error"], found["status"]) == (error_class, status)


@pytest.mark.parametrize(("model", "status", "error_class"), PROVIDER_ERRORS)
def test_provider_error_reaches_the_v2_failure_handler_with_the_same_class(
    rig: Rig, model: str, status: int, error_class: str
) -> None:
    """The V2 `post_call_failure` handler of `failure_gate_a` must see the same class and status as the legacy hook,
    exactly once: the adapter and the dispatcher hand the same exception to both (proxy/utils.py:4472-4482)."""
    text: Final = marker()
    assert chat(rig, text, model=model).status_code == status
    found: Final = records(rig, text, "failure_gate_a", "post_call_failure", 1)[0]
    assert (found["error"], found["status"]) == (error_class, status)


@pytest.mark.parametrize(("model", "status", "error_class"), PROVIDER_ERRORS)
def test_provider_error_reaches_the_failure_log_event_once(rig: Rig, model: str, status: int, error_class: str) -> None:
    """An error from the model call is also a failed attempt, so `async_log_failure_event` must run exactly once
    with the same class, and `required_reader` must see one attempt with outcome "failure". UNVERIFIED: the
    outcome string "failure" is assumed from the round 1 reader tests."""
    text: Final = marker()
    assert chat(rig, text, model=model).status_code == status
    assert records(rig, text, "failure_ledger", LOG, 1)[0]["error"] == error_class
    assert records(rig, text, "required_reader", "attempt_ended", 1)[0]["outcome"] == "failure"


def test_streamed_provider_error_reaches_the_failure_hook_once(rig: Rig) -> None:
    """A streamed chat to the 500 deployment fails before any chunk, so the caller gets 500 and the legacy failure
    hook runs exactly once with `InternalServerError`."""
    text: Final = marker()
    status, _, _ = stream_chat(rig, text, model=CHAT_OUTAGE)
    assert status == 500
    assert records(rig, text, "failure_ledger", HOOK, 1)[0]["error"] == "InternalServerError"


def test_budget_refusal_reaches_the_failure_hook_as_budget_exceeded(rig: Rig, scenario: Scenario) -> None:
    """A key with `max_budget: 0` is refused at auth with 422. The failure hook must run once for it with class
    `BudgetExceededError` and status 422 (exceptions.py:1161, litellm/__init__.py:393), with the request messages
    (auth passes the parsed body, auth_exception_handler.py:285-290) and the key's alias."""
    key, alias, _ = new_key(scenario, max_budget=0)
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 422
    found: Final = records(rig, text, "failure_ledger", HOOK, 1)[0]
    assert (found["error"], found["status"]) == ("BudgetExceededError", 422)
    assert found["key_alias"] == alias


def test_budget_refusal_reaches_the_v2_failure_handler(rig: Rig, scenario: Scenario) -> None:
    """The same budget refusal must reach the V2 `post_call_failure` handler once with `BudgetExceededError`."""
    key, _, _ = new_key(scenario, max_budget=0)
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 422
    assert records(rig, text, "failure_gate_a", "post_call_failure", 1)[0]["error"] == "BudgetExceededError"


def test_budget_refusal_is_not_a_failed_attempt(rig: Rig, scenario: Scenario) -> None:
    """A budget refusal never calls a model, so it must reach neither `async_log_failure_event` nor the
    `attempt_ended` reader, even after its failure hook has run. UNVERIFIED: assumes no logging object is
    created for an auth-time refusal."""
    key, _, _ = new_key(scenario, max_budget=0)
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 422
    records(rig, text, "failure_ledger", HOOK, 1)
    assert records(rig, text, "failure_ledger", LOG, 0) == ()
    assert records(rig, text, "required_reader", "attempt_ended", 0) == ()


def test_limiter_refusal_after_the_body_reaches_the_failure_hook(rig: Rig, scenario: Scenario) -> None:
    """A key with `model_rpm_limit: {"cblv2-chat": 1}` is served once and refused the second time after its body is
    read (per-model limits are not checked before the body). The refusal must reach the failure hook once with class
    `ProxyRateLimitError`, status 429 and the request messages."""
    key, alias, _ = new_key(scenario, model_rpm_limit={CHAT: 1})
    assert chat(rig, marker(), key=key).status_code == 200
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 429
    found: Final = records(rig, text, "failure_ledger", HOOK, 1)[0]
    assert (found["error"], found["status"], found["key_alias"]) == ("ProxyRateLimitError", 429, alias)


def test_limiter_refusal_before_the_body_reaches_the_failure_hook_without_the_request(
    rig: Rig, scenario: Scenario
) -> None:
    """A key with `rpm_limit: 1` is served once (which caches it) and its next chat is refused before the body is
    read (#51428). The failure hook must still run once, found by the key alias, with `ProxyRateLimitError` and 429,
    and with no messages, because the early path passes `request_data={}`."""
    key, alias, _ = new_key(scenario, rpm_limit=1)
    assert chat(rig, marker(), key=key).status_code == 200
    assert chat(rig, marker(), key=key).status_code == 429
    found: Final = _by_alias(rig, alias, HOOK, 1)[0]
    assert (found["error"], found["status"]) == ("ProxyRateLimitError", 429)
    assert found["text"] == '{"messages": null}'


def test_limiter_refusal_is_not_a_failed_attempt(rig: Rig, scenario: Scenario) -> None:
    """A limiter refusal after the body never calls a model, so it must not reach `async_log_failure_event`."""
    key, _, _ = new_key(scenario, model_rpm_limit={CHAT: 1})
    assert chat(rig, marker(), key=key).status_code == 200
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 429
    records(rig, text, "failure_ledger", HOOK, 1)
    assert records(rig, text, "failure_ledger", LOG, 0) == ()


def test_invalid_virtual_key_reaches_the_failure_hook_with_401(rig: Rig) -> None:
    """A chat with a well-formed but unknown `sk-` key must answer 401, and the failure hook must run once for it
    with status "401" and no key alias. ProxyException keeps its status as the string `code` (proxy/_types.py:4065),
    which is what a failure callback reading `.code` gets. UNVERIFIED: the class name is not asserted."""
    text: Final = marker()
    assert chat(rig, text, key="sk-cblv2b-unknown-key").status_code == 401
    found: Final = records(rig, text, "failure_ledger", HOOK, 1)[0]
    assert str(found["status"]) == "401"
    assert found["key_alias"] is None


def test_no_log_provider_error_still_runs_the_failure_hook_but_not_the_log_event(rig: Rig) -> None:
    """A `no-log: true` chat to the 400 deployment must still run the proxy failure hook (it is not logging) and
    must not run `async_log_failure_event` (Logging.should_run_callback, litellm_logging.py:3325-3332)."""
    text: Final = marker()
    assert chat(rig, text, model=CHAT_REJECTED, **{"no-log": True}).status_code == 400
    records(rig, text, "failure_ledger", HOOK, 1)
    assert records(rig, text, "failure_ledger", LOG, 0) == ()


def test_successful_chat_reaches_no_failure_callback(rig: Rig) -> None:
    """A plain successful chat must reach neither failure hook nor the V2 failure handler, once the reader has seen
    its success."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    records(rig, text, "required_reader", "attempt_ended", 1)
    assert records(rig, text, "failure_ledger", HOOK, 0) == ()
    assert records(rig, text, "failure_gate_a", "post_call_failure", 0) == ()
