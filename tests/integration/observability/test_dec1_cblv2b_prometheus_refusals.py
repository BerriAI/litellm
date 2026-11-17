"""Prometheus series after budget refusals, limiter refusals and vendor 429s, with the opt-in rate-limit and stream
labels switched on. The native Prometheus port (custom-logger-v2/step1/prometheus.md) must keep these label values,
and the Rust budget and limiter callbacks (#51535, #51537) must keep raising errors that produce them.

Contract under test, read from litellm/integrations/prometheus.py and litellm/types/integrations/prometheus.py:

- Every proxy-side failure adds 1 to `litellm_proxy_failed_requests_metric` and to
  `litellm_proxy_total_requests_metric` with the status (prometheus.py:3244-3301).
- `exception_class` is "BudgetExceededError" for a budget refusal (prometheus.py:4023-4024) and "HTTPException"
  for a proxy limiter 429 (`prometheus_exception_class_name`, proxy_rate_limit_error.py:346); a vendor 429 is the
  capitalised provider plus the class, "Openai.RateLimitError".
- With `prometheus_emit_rate_limit_labels: true` the failed metric gains `rate_limit_category` and
  `rate_limit_type` (types/integrations/prometheus.py:1042-1048): "litellm_rate_limit"/"budget" for a budget
  refusal (exceptions.py:1171-1172), "litellm_rate_limit"/"requests" for an rpm refusal, and
  "vendor_rate_limit" for a provider 429 (exceptions.py:547).
- With `prometheus_emit_stream_label: true` `litellm_proxy_total_requests_metric` gains `stream`
  (types/integrations/prometheus.py:1029-1035), whose value is `str(payload["stream"])` on success
  (prometheus.py:1854) and `str(request_data.get("stream"))` on failure (prometheus.py:3285).
- `requested_model` is bounded: an unknown model is "other" and the empty string stays empty
  (prometheus.py:580-613); the pre-body limiter refusal passes `request_data={}`, so its model is "".
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Scenario, held
from integration._support.prometheus_series import scrape
from integration.observability.dec1_cblv2_support import (
    CHAT,
    CHAT_LIMITED,
    CHAT_REJECTED,
    Rig,
    cblv2_rig,
    chat,
    label_names,
    marker,
    stream_chat,
    until_series,
    until_value,
    values,
)
from integration.observability.dec1_cblv2b_support import new_key

PROXY_FAILED: Final = "litellm_proxy_failed_requests_metric_total"
PROXY_TOTAL: Final = "litellm_proxy_total_requests_metric_total"
LLM_FAILED: Final = "litellm_llm_api_failed_requests_metric_total"
REQUESTS: Final = "litellm_requests_metric_total"
ROUTE: Final = "/v1/chat/completions"


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("cblv2b-prometheus-refusals")
    with cblv2_rig(
        directory,
        {
            "callbacks": ["prometheus"],
            "prometheus_emit_rate_limit_labels": True,
            "prometheus_emit_stream_label": True,
        },
        router_settings={"disable_cooldowns": True},
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def scenario(rig: Rig) -> Iterator[Scenario]:
    with rig.proxy.scenario() as opened:
        yield opened


def _quiet(rig: Rig, name: str, hashed: str) -> None:
    held(lambda: values(scrape(rig.proxy), name, hashed_api_key=hashed), lambda found: found == [], holding=3)


def test_budget_refusal_counts_one_proxy_failure_as_budget_exceeded(rig: Rig, scenario: Scenario) -> None:
    """A `max_budget: 0` key's chat is refused with 422. `litellm_proxy_failed_requests_metric_total` must reach
    exactly 1 on the series with the key's hash, exception_status "422", exception_class "BudgetExceededError" and
    route "/v1/chat/completions"."""
    key, _, hashed = new_key(scenario, max_budget=0)
    assert chat(rig, marker(), key=key).status_code == 422
    until_value(
        rig,
        PROXY_FAILED,
        1.0,
        hashed_api_key=hashed,
        exception_status="422",
        exception_class="BudgetExceededError",
        route=ROUTE,
    )


def test_budget_refusal_carries_the_budget_rate_limit_labels(rig: Rig, scenario: Scenario) -> None:
    """With rate-limit labels on, the budget refusal's failed series must carry rate_limit_category
    "litellm_rate_limit" and rate_limit_type "budget" (BudgetExceededError, exceptions.py:1171-1172)."""
    key, _, hashed = new_key(scenario, max_budget=0)
    assert chat(rig, marker(), key=key).status_code == 422
    until_series(
        rig, PROXY_FAILED, hashed_api_key=hashed, rate_limit_category="litellm_rate_limit", rate_limit_type="budget"
    )


def test_budget_refusal_counts_a_422_on_proxy_total_requests(rig: Rig, scenario: Scenario) -> None:
    """The budget refusal must add 1 to `litellm_proxy_total_requests_metric_total` with status_code "422" for the
    key, and nothing with status_code "200"."""
    key, _, hashed = new_key(scenario, max_budget=0)
    assert chat(rig, marker(), key=key).status_code == 422
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="422")
    assert values(scrape(rig.proxy), PROXY_TOTAL, hashed_api_key=hashed, status_code="200") == []


def test_budget_refusal_is_neither_a_request_nor_an_llm_api_failure(rig: Rig, scenario: Scenario) -> None:
    """A budget refusal never calls a model, so after its proxy failure is counted the key must have no
    `litellm_requests_metric_total` and no `litellm_llm_api_failed_requests_metric_total` series."""
    key, _, hashed = new_key(scenario, max_budget=0)
    assert chat(rig, marker(), key=key).status_code == 422
    until_series(rig, PROXY_FAILED, hashed_api_key=hashed)
    _quiet(rig, REQUESTS, hashed)
    _quiet(rig, LLM_FAILED, hashed)


def test_per_model_limiter_refusal_counts_http_exception_429(rig: Rig, scenario: Scenario) -> None:
    """A key with `model_rpm_limit: {"cblv2-chat": 1}` is refused on its second chat after the body is read. The
    failed metric must reach 1 with exception_status "429", exception_class "HTTPException" (pinned for old
    dashboards, proxy_rate_limit_error.py:346), requested_model "cblv2-chat" and the requests rate-limit labels."""
    key, _, hashed = new_key(scenario, model_rpm_limit={CHAT: 1})
    assert chat(rig, marker(), key=key).status_code == 200
    assert chat(rig, marker(), key=key).status_code == 429
    until_value(
        rig,
        PROXY_FAILED,
        1.0,
        hashed_api_key=hashed,
        exception_status="429",
        exception_class="HTTPException",
        requested_model=CHAT,
        rate_limit_category="litellm_rate_limit",
        rate_limit_type="requests",
    )


def test_pre_body_limiter_refusal_counts_with_an_empty_requested_model(rig: Rig, scenario: Scenario) -> None:
    """A key with `rpm_limit: 1` is refused before its body is read on the second chat (#51428). The early path
    calls the failure hook with `request_data={}`, so the failed series must have requested_model "" (the bounded
    label keeps the empty string, prometheus.py:596-597), exception_class "HTTPException" and status "429"."""
    key, _, hashed = new_key(scenario, rpm_limit=1)
    assert chat(rig, marker(), key=key).status_code == 200
    assert chat(rig, marker(), key=key).status_code == 429
    until_value(
        rig,
        PROXY_FAILED,
        1.0,
        hashed_api_key=hashed,
        exception_status="429",
        exception_class="HTTPException",
        requested_model="",
        route=ROUTE,
    )


def test_limiter_refusal_counts_a_429_on_proxy_total_requests(rig: Rig, scenario: Scenario) -> None:
    """The served chat and the refused one must leave the key with one status_code "200" and one status_code "429"
    sample on `litellm_proxy_total_requests_metric_total`."""
    key, _, hashed = new_key(scenario, rpm_limit=1)
    assert chat(rig, marker(), key=key).status_code == 200
    assert chat(rig, marker(), key=key).status_code == 429
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="429")
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="200")


def test_limiter_refusal_adds_no_request_beyond_the_served_one(rig: Rig, scenario: Scenario) -> None:
    """After one served and one refused chat, `litellm_requests_metric_total` for the key must be exactly 1."""
    key, _, hashed = new_key(scenario, rpm_limit=1)
    assert chat(rig, marker(), key=key).status_code == 200
    assert chat(rig, marker(), key=key).status_code == 429
    until_value(rig, PROXY_FAILED, 1.0, hashed_api_key=hashed)
    until_value(rig, REQUESTS, 1.0, hashed_api_key=hashed)


def test_vendor_429_carries_the_vendor_rate_limit_category(rig: Rig, scenario: Scenario) -> None:
    """A chat whose provider answers 429 must count on the failed metric with exception_class
    "Openai.RateLimitError" and rate_limit_category "vendor_rate_limit" (RateLimitError default, exceptions.py:547),
    which tells it apart from the proxy's own 429s."""
    key, _, hashed = new_key(scenario)
    assert chat(rig, marker(), model=CHAT_LIMITED, key=key).status_code == 429
    until_value(
        rig,
        PROXY_FAILED,
        1.0,
        hashed_api_key=hashed,
        exception_status="429",
        exception_class="Openai.RateLimitError",
        rate_limit_category="vendor_rate_limit",
    )


def test_failed_metric_label_set_gains_the_two_rate_limit_labels(rig: Rig, scenario: Scenario) -> None:
    """With `prometheus_emit_rate_limit_labels: true` the failed metric's label set must be the default set
    (types/integrations/prometheus.py:514-534) plus rate_limit_category and rate_limit_type, and nothing else."""
    key, _, hashed = new_key(scenario)
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
        "rate_limit_category",
        "rate_limit_type",
    }


def test_provider_400_has_no_rate_limit_label_values(rig: Rig, scenario: Scenario) -> None:
    """A provider 400 is not a rate limit, so its failed series must carry empty rate_limit_category and
    rate_limit_type (validate_rate_limit_category returns None, exceptions.py:85-95). UNVERIFIED: assumes a None
    label is exported as the empty string."""
    key, _, hashed = new_key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    until_series(rig, PROXY_FAILED, hashed_api_key=hashed, rate_limit_category="", rate_limit_type="")


def test_stream_label_is_true_on_a_streamed_success(rig: Rig, scenario: Scenario) -> None:
    """With the stream label on, a streamed chat must count on `litellm_proxy_total_requests_metric_total` with
    stream "True" and status_code "200"."""
    key, _, hashed = new_key(scenario)
    status, _, _ = stream_chat(rig, marker(), key=key)
    assert status == 200
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="200", stream="True")


def test_stream_label_is_none_on_a_plain_success(rig: Rig, scenario: Scenario) -> None:
    """A non-streamed success has `stream` None in its StandardLoggingPayload (logging_obj.passthrough_stream
    defaults to None and is set True only for streams, litellm_logging.py:1047, 8746-8751), so the label must be
    the string "None", not "False". Pinned as a surprise a port could "fix" silently."""
    key, _, hashed = new_key(scenario)
    assert chat(rig, marker(), key=key).status_code == 200
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="200", stream="None")


def test_stream_label_on_a_failure_is_read_from_the_request_body(rig: Rig, scenario: Scenario) -> None:
    """On failure the label is `str(request_data.get("stream"))`: a non-streamed 400 must count with stream "None"
    and a streamed 400 with stream "True"."""
    key, _, hashed = new_key(scenario)
    assert chat(rig, marker(), model=CHAT_REJECTED, key=key).status_code == 400
    status, _, _ = stream_chat(rig, marker(), model=CHAT_REJECTED, key=key)
    assert status == 400
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="400", stream="None")
    until_value(rig, PROXY_TOTAL, 1.0, hashed_api_key=hashed, status_code="400", stream="True")


def test_unknown_model_collapses_to_other_on_the_failed_metric(rig: Rig, scenario: Scenario) -> None:
    """A chat naming a model the router does not know is refused, and its failed series must carry
    requested_model "other", never the caller's string, so callers cannot grow label cardinality
    (prometheus.py:592-613). UNVERIFIED: the refusal status (400) is not asserted."""
    key, _, hashed = new_key(scenario)
    unknown: Final = f"cblv2b-no-such-model-{marker()}"
    assert chat(rig, marker(), model=unknown, key=key).status_code != 200
    until_series(rig, PROXY_FAILED, hashed_api_key=hashed, requested_model="other")
    assert values(scrape(rig.proxy), PROXY_FAILED, requested_model=unknown) == []
