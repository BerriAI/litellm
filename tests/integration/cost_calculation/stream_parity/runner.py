import time
from collections import Counter
from hashlib import sha256
from typing import Final

from integration._support.client import Gateway
from integration.cost_calculation.assertions import assert_exact
from integration.cost_calculation.conftest import CostRow, poll_rows, read_rows_now, register_scenario_deployment
from integration.cost_calculation.cost_tracking_case import ExactExpected
from integration.cost_calculation.stream_parity.case import StreamParityTestCase

SPEND_LOG_SETTLE_SECONDS: Final = 2.0


def rows_after_settle(key: str, count: int) -> tuple[CostRow, ...]:
    poll_rows(key, count)
    time.sleep(SPEND_LOG_SETTLE_SECONDS)
    return read_rows_now(key)


def assert_stream_parity(case: StreamParityTestCase, gateway: Gateway) -> None:
    expected: Final = case.plain.expected
    assert isinstance(expected, ExactExpected), case.id
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        deployments: Final = {
            leg.name: register_scenario_deployment(scenario, leg, sha256(leg.name.encode()).hexdigest()[:12], key)
            for leg in (case.plain, case.streamed)
        }
        responses: Final = {
            leg.name: gateway.request(
                "POST", leg.endpoint, {**leg.request, "model": deployments[leg.name].model_name}, key=key
            )
            for leg in (case.plain, case.streamed)
        }
        for leg in (case.plain, case.streamed):
            assert responses[leg.name].status_code == 200, (leg.name, responses[leg.name].text[:400])

        rows: Final = rows_after_settle(key, 2)
        rows_per_deployment: Final = Counter(row.model_id for row in rows)
        assert rows_per_deployment == {deployments[leg.name].identity: 1 for leg in (case.plain, case.streamed)}, (
            f"{case.id}: expected exactly one SpendLogs row per leg, got {dict(rows_per_deployment)}",
            rows,
        )
        billed: Final = {
            leg.name: next(row for row in rows if row.model_id == deployments[leg.name].identity)
            for leg in (case.plain, case.streamed)
        }
        for leg in (case.plain, case.streamed):
            assert_exact(leg.name, leg.response.content_type, expected, billed[leg.name], responses[leg.name])
        plain: Final = billed[case.plain.name]
        streamed: Final = billed[case.streamed.name]
        assert (streamed.spend, streamed.prompt_tokens, streamed.completion_tokens) == (
            plain.spend,
            plain.prompt_tokens,
            plain.completion_tokens,
        ), (plain, streamed)
