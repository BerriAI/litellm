"""Prometheus /metrics after failed, fallen-back, rejected, no-log and disabled calls must not change when the
Prometheus logger becomes a native CustomLoggerV2 callback.

Contract. A provider error adds 1 to `litellm_llm_api_failed_requests_metric` (attempt level, from
async_log_failure_event) and 1 to `litellm_proxy_failed_requests_metric` plus a non-200
`litellm_proxy_total_requests_metric` (request level, from async_post_call_failure_hook). The deployment goes to
state 1 and counts a deployment failure labelled with the exception status and class, where the class is the
provider name capitalised plus the exception class (`Openai.BadRequestError`). A failure never counts on
`litellm_requests_metric` or spend. A router fallback adds one `litellm_deployment_successful_fallbacks`. A
request a proxy hook rejects counts as a proxy failure but never as an LLM API failure. A request with an
invalid virtual key leaves no series at all (`_should_skip_metrics_for_invalid_key`). Prometheus honours
`no-log` (its class name lacks `_PROXY_`) and, on a licensed proxy, `x-litellm-disable-callbacks: prometheus`.

The native port spec (step1/prometheus.md section 1.4) lists exactly these as differences of the customer-branch
module: per call instead of per attempt, exception_class derived from the status, no invalid-key skip, no
fallback metrics. Each one is pinned here.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Scenario, held
from integration._support.prometheus_series import scrape
from integration.observability.dec1_cblv2_probe import CHAIN_REJECTION, REJECT_AT
from integration.observability.dec1_cblv2_support import (
    CHAT,
    CHAT_FALLBACK_PRIMARY,
    CHAT_OUTAGE,
    CHAT_REJECTED,
    PROBE_MODULE,
    UPSTREAM_MODEL,
    Rig,
    cblv2_rig,
    chat,
    deployment_id,
    error_message,
    key_hash,
    label_names,
    marker,
    stream_chat,
    until_series,
    until_value,
    values,
)
from pydantic import JsonValue

LLM_FAILED: Final = "litellm_llm_api_failed_requests_metric_total"
PROXY_FAILED: Final = "litellm_proxy_failed_requests_metric_total"
PROXY_TOTAL: Final = "litellm_proxy_total_requests_metric_total"
REQUESTS: Final = "litellm_requests_metric_total"
SPEND: Final = "litellm_spend_metric_total"
STATE: Final = "litellm_deployment_state"
DEPLOYMENT_FAILURE: Final = "litellm_deployment_failure_responses_total"
DEPLOYMENT_TOTAL: Final = "litellm_deployment_total_requests_total"
FALLBACK_OK: Final = "litellm_deployment_successful_fallbacks_total"
FALLBACK_LOST: Final = "litellm_deployment_failed_fallbacks_total"
LICENSED: Final = bool(os.environ.get("LITELLM_LICENSE"))


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("cblv2-prometheus-failure")
    with cblv2_rig(
        directory,
        {"callbacks": ["prometheus", f"{PROBE_MODULE}.chain_first"]},
        router_settings={"disable_cooldowns": True},
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def scenario(rig: Rig) -> Iterator[Scenario]:
    with rig.proxy.scenario() as opened:
        yield opened


def _key(scenario: Scenario, **fields: JsonValue) -> tuple[str, str, str]:
    alias: Final = f"cblv2-promfail-{uuid.uuid4().hex[:10]}"
    key: Final = scenario.key(key_alias=alias, **fields)
    return key, alias, key_hash(key)


def _quiet(rig: Rig, name: str, hashed: str, *, holding: float = 3) -> None:
    """The metric has no series for the key and still has none `holding` seconds later."""
    held(lambda: values(scrape(rig.proxy), name, hashed_api_key=hashed), lambda found: found == [], holding=holding)


def test_provider_400_counts_one_llm_api_failure_for_the_key(rig: Rig, scenario: Scenario) -> None:
    """A chat the provider answers with 400 must add exactly 1 to `litellm_llm_api_failed_requests_metric_total` on
    the series with the key's hash and alias, model="gpt-4o-mini" and model_id="cblv2-chat-rejected-id"."""
    key, alias, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    until_value(
        rig,
        LLM_FAILED,
        1.0,
        hashed_api_key=hashed,
        api_key_alias=alias,
        model=UPSTREAM_MODEL,
        model_id=deployment_id(CHAT_REJECTED),
    )


def test_llm_api_failed_label_set_is_the_documented_one(rig: Rig, scenario: Scenario) -> None:
    """`litellm_llm_api_failed_requests_metric` must carry exactly end_user, hashed_api_key, api_key_alias, model,
    team, team_alias, user and model_id. It is not in `_org_label_metrics`, so it has no org labels."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    sample: Final = until_series(rig, LLM_FAILED, hashed_api_key=hashed)[0]
    assert label_names(sample) == {
        "end_user",
        "hashed_api_key",
        "api_key_alias",
        "model",
        "team",
        "team_alias",
        "user",
        "model_id",
    }


def test_provider_400_counts_one_proxy_failure_with_status_class_and_route(rig: Rig, scenario: Scenario) -> None:
    """The same 400 must add 1 to `litellm_proxy_failed_requests_metric_total` with exception_status="400",
    exception_class="Openai.BadRequestError", route="/v1/chat/completions" and requested_model
    "cblv2-chat-rejected". The class is the provider capitalised, a dot and the Python exception class
    (`_get_exception_class_name`, litellm/integrations/prometheus.py:4026); the native module derives it from the
    status code instead, which must not leak into this label."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    until_value(
        rig,
        PROXY_FAILED,
        1.0,
        hashed_api_key=hashed,
        exception_status="400",
        exception_class="Openai.BadRequestError",
        route="/v1/chat/completions",
        requested_model=CHAT_REJECTED,
    )


def test_proxy_failed_label_set_has_no_rate_limit_labels_by_default(rig: Rig, scenario: Scenario) -> None:
    """`litellm_proxy_failed_requests_metric` must carry exactly its declared labels plus org_id and org_alias, and
    no rate_limit_category or rate_limit_type, which are opt-in through `prometheus_emit_rate_limit_labels`."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    sample: Final = until_series(rig, PROXY_FAILED, hashed_api_key=hashed)[0]
    assert label_names(sample) == {
        "end_user",
        "hashed_api_key",
        "api_key_alias",
        "requested_model",
        "team",
        "team_alias",
        "user",
        "user_email",
        "exception_status",
        "exception_class",
        "route",
        "client_ip",
        "user_agent",
        "model_id",
        "api_provider",
        "org_id",
        "org_alias",
    }


def test_provider_400_counts_a_400_on_proxy_total_requests(rig: Rig, scenario: Scenario) -> None:
    """The 400 must add 1 to `litellm_proxy_total_requests_metric_total` with status_code="400" and no series with
    status_code="200" for the key."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="400")
    assert values(scrape(rig.proxy), PROXY_TOTAL, hashed_api_key=hashed, status_code="200") == []


def test_failure_never_counts_a_request_or_spend(rig: Rig, scenario: Scenario) -> None:
    """A provider 400 must create no `litellm_requests_metric_total` and no `litellm_spend_metric_total` series for
    the key: those are success-only counters."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    until_series(rig, LLM_FAILED, hashed_api_key=hashed)
    _quiet(rig, REQUESTS, hashed)
    _quiet(rig, SPEND, hashed, holding=0.5)


def test_provider_500_puts_the_deployment_in_partial_outage(rig: Rig, scenario: Scenario) -> None:
    """After a chat the provider answers with 500, `litellm_deployment_state` for model_id="cblv2-chat-outage-id"
    must be 1 (partial outage), not 2: complete outage is set only by the cooldown path, which is disabled here."""
    key, _, _ = _key(scenario)
    assert chat(rig, marker(), model=CHAT_OUTAGE, key=key).status_code == 500
    until_value(rig, STATE, 1.0, model_id=deployment_id(CHAT_OUTAGE))


def test_provider_500_counts_a_deployment_failure_with_status_and_class(rig: Rig, scenario: Scenario) -> None:
    """The 500 must add 1 to `litellm_deployment_failure_responses_total` for the key with
    exception_status="500", exception_class="Openai.InternalServerError", model_group and requested_model
    "cblv2-chat-outage", and 1 to `litellm_deployment_total_requests_total`."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_OUTAGE, key=key).status_code == 500
    until_value(
        rig,
        DEPLOYMENT_FAILURE,
        1.0,
        hashed_api_key=hashed,
        exception_status="500",
        exception_class="Openai.InternalServerError",
        model_group=CHAT_OUTAGE,
        requested_model=CHAT_OUTAGE,
    )
    until_value(rig, DEPLOYMENT_TOTAL, 1.0, hashed_api_key=hashed, model_id=deployment_id(CHAT_OUTAGE))


def test_streamed_request_to_a_failing_deployment_counts_one_failure(rig: Rig, scenario: Scenario) -> None:
    """A streamed chat the provider refuses with 400 before the first byte must count exactly once on
    `litellm_llm_api_failed_requests_metric_total` and once on `litellm_proxy_failed_requests_metric_total` with
    exception_status="400"."""
    key, _, hashed = _key(scenario)
    assert stream_chat(rig, marker(), model=CHAT_REJECTED, key=key)[0] == 400
    until_value(rig, LLM_FAILED, 1.0, hashed_api_key=hashed)
    until_value(rig, PROXY_FAILED, 1.0, hashed_api_key=hashed, exception_status="400")
    held(
        lambda: values(scrape(rig.proxy), LLM_FAILED, hashed_api_key=hashed),
        lambda found: found == [1.0],
        holding=3,
    )


def test_router_fallback_counts_one_successful_fallback(rig: Rig, scenario: Scenario) -> None:
    """The primary "cblv2-chat-fallback-primary" answers 500 and the router falls back to "cblv2-chat". The caller
    gets 200 and `litellm_deployment_successful_fallbacks_total` must be 1 for the key with
    requested_model="cblv2-chat-fallback-primary", fallback_model="cblv2-chat", exception_status="500" and
    exception_class="Openai.InternalServerError". The native module on the customer branch has no fallback
    metrics at all (router_event)."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_FALLBACK_PRIMARY, key=key).status_code == 200
    until_value(
        rig,
        FALLBACK_OK,
        1.0,
        hashed_api_key=hashed,
        requested_model=CHAT_FALLBACK_PRIMARY,
        fallback_model=CHAT,
        exception_status="500",
        exception_class="Openai.InternalServerError",
    )
    assert values(scrape(rig.proxy), FALLBACK_LOST, hashed_api_key=hashed) == []


def test_router_fallback_counts_the_failed_attempt_and_the_success(rig: Rig, scenario: Scenario) -> None:
    """The same fallback must add 1 to `litellm_llm_api_failed_requests_metric_total` for the primary attempt
    (model_id="cblv2-chat-fallback-primary-id") and 1 to `litellm_requests_metric_total` for the answered call.
    Prometheus counts per attempt today; a per-call port would drop the attempt failure."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_FALLBACK_PRIMARY, key=key).status_code == 200
    until_value(rig, LLM_FAILED, 1.0, hashed_api_key=hashed, model_id=deployment_id(CHAT_FALLBACK_PRIMARY))
    until_value(rig, REQUESTS, 1.0, hashed_api_key=hashed)


def test_router_fallback_success_is_a_200_on_proxy_total_requests(rig: Rig, scenario: Scenario) -> None:
    """A request rescued by a fallback answers 200, so `litellm_proxy_total_requests_metric_total` must count it
    once with status_code="200" and no proxy failure series for the key. UNVERIFIED: requested_model on this
    series is taken from `original_model_group` (litellm/integrations/prometheus.py:1972-1994), expected to be
    the primary group the caller asked for."""
    key, _, hashed = _key(scenario)
    assert chat(rig, marker(), model=CHAT_FALLBACK_PRIMARY, key=key).status_code == 200
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="200", requested_model=CHAT_FALLBACK_PRIMARY)
    assert values(scrape(rig.proxy), PROXY_FAILED, hashed_api_key=hashed) == []


def test_request_rejected_by_a_pre_call_hook_is_a_proxy_failure_only(rig: Rig, scenario: Scenario) -> None:
    """The legacy logger `chain_first` raises HTTPException(400) at pre-call when the text asks for it. The caller
    gets 400 with the hook's message, `litellm_proxy_failed_requests_metric_total` counts 1 with
    exception_status="400" and exception_class="HTTPException", and no LLM API failure is counted because no
    provider attempt was made."""
    key, _, hashed = _key(scenario)
    response: Final = chat(rig, f"{REJECT_AT}chain_first {marker()}", key=key)
    assert response.status_code == 400, response.text
    assert error_message(response) == f"{CHAIN_REJECTION} chain_first"
    until_value(rig, PROXY_FAILED, 1.0, hashed_api_key=hashed, exception_status="400", exception_class="HTTPException")
    _quiet(rig, LLM_FAILED, hashed)


def test_invalid_virtual_key_leaves_no_series(rig: Rig, scenario: Scenario) -> None:
    """A chat with a made-up `sk-` key is answered 401 and must create no series carrying that key's hash on
    `litellm_proxy_failed_requests_metric_total` or `litellm_proxy_total_requests_metric_total`
    (`_should_skip_metrics_for_invalid_key`, litellm/integrations/prometheus.py:3087). The native module has no
    such skip, so unauthenticated scanners would create unbounded series."""
    bogus: Final = f"sk-cblv2-{uuid.uuid4().hex}"
    assert chat(rig, marker(), key=bogus).status_code == 401
    _quiet(rig, PROXY_FAILED, key_hash(bogus))
    _quiet(rig, PROXY_TOTAL, key_hash(bogus), holding=0.5)


def test_no_log_request_leaves_no_success_series(rig: Rig, scenario: Scenario) -> None:
    """A successful chat with body `no-log: true` must create no `litellm_requests_metric_total`, no spend and no
    `litellm_proxy_total_requests_metric_total` series for the key: Prometheus is skipped by `should_run_callback`
    because its class name lacks `_PROXY_`, and the V2 Selection must keep `honors_no_log=True`."""
    key, _, hashed = _key(scenario)
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": CHAT, "messages": [{"role": "user", "content": marker()}], "no-log": True},
        key=key,
    )
    assert response.status_code == 200, response.text
    _quiet(rig, REQUESTS, hashed, holding=4)
    _quiet(rig, SPEND, hashed, holding=0.5)
    _quiet(rig, PROXY_TOTAL, hashed, holding=0.5)


def test_no_log_failure_still_counts_the_proxy_failure(rig: Rig, scenario: Scenario) -> None:
    """A `no-log: true` chat the provider answers with 400 must still add 1 to
    `litellm_proxy_failed_requests_metric_total`, because no-log gates the logging plane (async_log_failure_event)
    and not the proxy post_call_failure hook. It must add nothing to the attempt level
    `litellm_llm_api_failed_requests_metric_total`. UNVERIFIED: read from should_run_callback, which only the
    logging plane calls (step0 should-run.md L6); a port that applies no-log at every hook drops the proxy
    failure."""
    key, _, hashed = _key(scenario)
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": CHAT_REJECTED, "messages": [{"role": "user", "content": marker()}], "no-log": True},
        key=key,
    )
    assert response.status_code == 400, response.text
    until_value(rig, PROXY_FAILED, 1.0, hashed_api_key=hashed, exception_status="400")
    _quiet(rig, LLM_FAILED, hashed)


def test_disable_callbacks_header_turns_prometheus_off_for_one_request(rig: Rig, scenario: Scenario) -> None:
    """On a licensed proxy, `x-litellm-disable-callbacks: prometheus` on one chat must leave no
    `litellm_requests_metric_total` series for the key, while the next chat on the same key without the header
    counts 1. The port's Selection must keep `OptOutBy.caller`."""
    if not LICENSED:
        pytest.skip("dynamic callback disabling is honoured only with LITELLM_LICENSE set")
    key, _, hashed = _key(scenario)
    disabled: Final = chat(rig, marker(), key=key, headers={"x-litellm-disable-callbacks": "prometheus"})
    assert disabled.status_code == 200
    _quiet(rig, REQUESTS, hashed, holding=4)
    assert chat(rig, marker(), key=key).status_code == 200
    until_value(rig, REQUESTS, 1.0, hashed_api_key=hashed)


def test_key_disabled_callbacks_metadata_turns_prometheus_off_for_that_key(rig: Rig, scenario: Scenario) -> None:
    """On a licensed proxy, a key with metadata `litellm_disabled_callbacks: ["prometheus"]` must never create a
    `litellm_requests_metric_total` series, while a sibling key without it does."""
    if not LICENSED:
        pytest.skip("dynamic callback disabling is honoured only with LITELLM_LICENSE set")
    muted, _, muted_hash = _key(scenario, metadata={"litellm_disabled_callbacks": ["prometheus"]})
    loud, _, loud_hash = _key(scenario)
    assert chat(rig, marker(), key=muted).status_code == 200
    assert chat(rig, marker(), key=loud).status_code == 200
    until_value(rig, REQUESTS, 1.0, hashed_api_key=loud_hash)
    _quiet(rig, REQUESTS, muted_hash)


def test_failure_on_one_key_does_not_touch_another_keys_series(rig: Rig, scenario: Scenario) -> None:
    """A 400 on key A and a success on key B must leave A with one LLM API failure and no request series, and B
    with one request and no failure series: label values never bleed between concurrent calls."""
    first, _, first_hash = _key(scenario)
    second, _, second_hash = _key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=first).status_code == 400
    assert chat(rig, marker(), key=second).status_code == 200
    until_value(rig, LLM_FAILED, 1.0, hashed_api_key=first_hash)
    until_value(rig, REQUESTS, 1.0, hashed_api_key=second_hash)
    samples: Final = scrape(rig.proxy)
    assert values(samples, REQUESTS, hashed_api_key=first_hash) == []
    assert values(samples, LLM_FAILED, hashed_api_key=second_hash) == []
