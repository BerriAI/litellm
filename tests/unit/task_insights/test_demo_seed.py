from datetime import datetime, timedelta, timezone

import pytest

from litellm.task_insights.demo_content import DEMO_PREFIX, DEMO_TASKS
from litellm.task_insights.demo_seed import (
    _deliveries,
    _spend_log,
    demo_decision,
    demo_placements,
    demo_requests,
    demo_taxonomy,
)

_NOW = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)


def test_the_sample_is_reproducible_and_namespaced():
    requests = demo_requests(_NOW)

    assert requests == demo_requests(_NOW)
    assert all(request.request_id.startswith(DEMO_PREFIX) for request in requests)
    assert all(_NOW - timedelta(days=15) <= request.start <= _NOW + timedelta(hours=1) for request in requests)


def test_the_sample_exercises_every_edge_the_ui_must_show():
    requests = demo_requests(_NOW)
    deliveries = [delivery for request in requests for delivery in _deliveries(request)]

    assert len(deliveries) > len(requests)
    assert any(request.spend is None for request in requests)
    assert any(request.latency_ms is None and request.status == "success" for request in requests)
    assert any(request.status == "failure" and request.spend for request in requests)
    assert any(not request.stored_prompt for request in requests)
    assert any(request.task is None and not request.analysis for request in requests)
    assert any(request.analysis for request in requests)
    assert {request.task.key for request in requests if request.task} == {task.key for task in DEMO_TASKS}


def test_a_request_without_stored_prompt_logs_no_text():
    request = next(request for request in demo_requests(_NOW) if not request.stored_prompt)

    assert _spend_log(request)["proxy_server_request"] == {}


def test_a_retried_request_bills_its_failed_attempt_on_another_model():
    request = next(request for request in demo_requests(_NOW) if request.retried_on is not None)
    attempt, final = _deliveries(request)

    assert attempt["accounting_payload"]["status"] == "failure"
    assert attempt["accounting_payload"]["model"] != final["accounting_payload"]["model"]
    assert attempt["accounting_payload"]["spend"] > 0


@pytest.mark.parametrize("off_topic", [False, True])
def test_sample_decisions_are_labeled_and_sum_to_one(off_topic):
    request = next(
        request for request in demo_requests(_NOW) if (request.task is None) is off_topic and not request.analysis
    )
    decision = demo_decision(request)

    assert decision.outcome == ("unknown" if off_topic else "assigned")
    assert decision.decision_model == "sample-precomputed"
    assert decision.provenance["source"] == "sample"
    assert sum(decision.probabilities.values()) == pytest.approx(1, abs=0.01)


def test_taxonomy_and_placements_cover_the_sample():
    requests = demo_requests(_NOW)
    taxonomy = demo_taxonomy(requests)
    placements = demo_placements(requests)

    assert len(taxonomy) == len(DEMO_TASKS)
    assert {point.task_id for point in placements} == {task.task_id for task in taxonomy}
    assert sum(task.discovery_size for task in taxonomy) == len(placements)
