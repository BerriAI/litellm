"""A provider 401 on a caller's own per-user GitHub Copilot session is the caller's credential failing, so the
shared Copilot deployment keeps no failure, request or outage series for it, while a provider 401 on the
deployment's own key in the same process is counted against that deployment."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.copilot_session_prometheus_worker import (
    CALLER_SESSION_TOKEN,
    COPILOT_DEPLOYMENT,
    DEPLOYMENT_SERIES,
    FAILURES,
    LLM_FAILED,
    OPERATOR_DEPLOYMENT,
    OPERATOR_KEY,
    STATE,
    TOTALS,
    Outcome,
    WorkerReport,
    run_copilot_session_worker,
)
from integration._support.wire import Reply, Request, wire_server


@pytest.fixture(scope="module")
def ran() -> Iterator[tuple[WorkerReport, tuple[Request, ...]]]:
    def reject(request: Request) -> Reply:
        assert request.method == "POST", request.method
        return Reply(
            status=401,
            body=json.dumps({"error": {"message": "bad credentials", "type": "authentication_error"}}).encode(),
        )

    with wire_server(reject) as upstream:
        report: Final = run_copilot_session_worker(upstream.url)
        yield report, upstream.drain()


def test_both_calls_reach_the_provider_with_their_own_credential_and_fail_with_its_401(
    ran: tuple[WorkerReport, tuple[Request, ...]],
) -> None:
    report, received = ran
    assert report.outcomes == (
        Outcome(deployment=COPILOT_DEPLOYMENT, status=401),
        Outcome(deployment=OPERATOR_DEPLOYMENT, status=401),
    )
    assert [r.headers.get("authorization") for r in received] == [
        f"Bearer {CALLER_SESSION_TOKEN}",
        f"Bearer {OPERATOR_KEY}",
    ]


def test_a_provider_401_on_a_callers_copilot_session_stays_out_of_the_shared_deployment_metrics(
    ran: tuple[WorkerReport, tuple[Request, ...]],
) -> None:
    report, _ = ran
    assert {s.metric for s in report.series if s.model_id == COPILOT_DEPLOYMENT} & DEPLOYMENT_SERIES == set()


def test_a_provider_401_on_the_deployments_own_key_counts_as_a_deployment_failure(
    ran: tuple[WorkerReport, tuple[Request, ...]],
) -> None:
    report, _ = ran
    assert report.values(FAILURES, OPERATOR_DEPLOYMENT) == (1.0,)
    assert report.values(TOTALS, OPERATOR_DEPLOYMENT) == (1.0,)
    assert report.values(LLM_FAILED, OPERATOR_DEPLOYMENT) == (1.0,)
    assert report.values(STATE, OPERATOR_DEPLOYMENT) == (1.0,)
