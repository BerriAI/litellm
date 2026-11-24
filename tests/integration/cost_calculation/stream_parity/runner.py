from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
from types import MappingProxyType
from typing import Final

from integration._support.client import Gateway, Scenario
from integration.cost_calculation.assertions import assert_exact
from integration.cost_calculation.conftest import (
    CostRow,
    poll_rows,
    poll_rows_where,
    read_rows_now,
    register_scenario_deployment,
)
from integration.cost_calculation.cost_tracking_case import CostTrackingTestCase, ExactExpected
from integration.cost_calculation.stream_parity.bases import openai_responses
from integration.cost_calculation.stream_parity.case import StreamParityTestCase

FLUSH_CASE: Final = openai_responses.GPT_6_1_SOL_RESPONSES


def flush_spend_logs(scenario: Scenario, gateway: Gateway, key: str) -> str:
    """A /v1/responses row wakes the spend-log writer at once, and the queue is FIFO, so every row queued before it lands."""
    flush: Final = register_scenario_deployment(scenario, FLUSH_CASE, "flush", key)
    response: Final = gateway.request(
        "POST", FLUSH_CASE.endpoint, {**FLUSH_CASE.request, "model": flush.model_name}, key=key
    )
    assert response.status_code == 200, ("flush", response.text[:400])
    poll_rows_where(key, 1, lambda row: row.model_id == flush.identity)
    return flush.identity


def bill_legs(case_id: str, legs: tuple[CostTrackingTestCase, ...], gateway: Gateway) -> Mapping[str, CostRow]:
    """Send every leg through one key, wait for exactly one SpendLogs row per leg, and check each row against its leg's
    expected row."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        deployments: Final = {
            leg.name: register_scenario_deployment(scenario, leg, sha256(leg.name.encode()).hexdigest()[:12], key)
            for leg in legs
        }
        responses: Final = {
            leg.name: gateway.request(
                "POST", leg.endpoint, {**leg.request, "model": deployments[leg.name].model_name}, key=key
            )
            for leg in legs
        }
        for leg in legs:
            assert responses[leg.name].status_code == 200, (leg.name, responses[leg.name].text[:400])

        poll_rows(key, len(legs))
        flush_identity: Final = flush_spend_logs(scenario, gateway, key)
        rows: Final = read_rows_now(key)
        rows_per_deployment: Final = Counter(row.model_id for row in rows)
        expected_rows: Final = {deployments[leg.name].identity: 1 for leg in legs}
        assert rows_per_deployment == {**expected_rows, flush_identity: 1}, (
            f"{case_id}: expected exactly one SpendLogs row per leg plus the flush row, got {dict(rows_per_deployment)}",
            rows,
        )
        billed: Final = {
            leg.name: next(row for row in rows if row.model_id == deployments[leg.name].identity) for leg in legs
        }
        for leg in legs:
            expected: Final = leg.expected
            assert isinstance(expected, ExactExpected), leg.name
            assert_exact(leg.name, leg.response.content_type, expected, billed[leg.name], responses[leg.name])
        return MappingProxyType(billed)


def assert_stream_parity(case: StreamParityTestCase, gateway: Gateway) -> None:
    billed: Final = bill_legs(case.id, (case.plain, case.streamed), gateway)
    plain: Final = billed[case.plain.name]
    streamed: Final = billed[case.streamed.name]
    assert (streamed.spend, streamed.prompt_tokens, streamed.completion_tokens) == (
        plain.spend,
        plain.prompt_tokens,
        plain.completion_tokens,
    ), (plain, streamed)
