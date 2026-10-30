"""A request through a bridged client API bills exactly what the provider's native API bills for the same response.

A bridge translates one API into another before the provider call and translates the provider's usage back after it
(LIT-7730, LIT-9098). Each case is one stored provider response sent twice through the same key, once through the
provider's own API and once through a client API LiteLLM bridges into it. Both LiteLLM_SpendLogs rows must bill the
case's row, and they must agree with each other except where a documented provider quirk says otherwise.
"""

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.bridge_cost_parity.case import BridgeCostParityTestCase
from integration.cost_calculation.bridge_cost_parity.runner import assert_bridge_cost_parity
from integration.cost_calculation.catalog import BRIDGE_COST_PARITY_CASES


def _marks(case: BridgeCostParityTestCase) -> tuple[pytest.MarkDecorator, ...]:
    """A known bridge bug is intermittent (it billed the native row in 1 of 4 local runs), so the xfail is not strict."""
    return (pytest.mark.xfail(strict=False, reason=f"BUG: {case.known_bug}"),) if case.known_bug is not None else ()


_CASES: Final = tuple(pytest.param(case, marks=_marks(case), id=case.id) for case in BRIDGE_COST_PARITY_CASES)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.timeout(180)
def test_bridged_request_bills_the_same_row_as_the_native_request(
    case: BridgeCostParityTestCase, gateway: Gateway
) -> None:
    assert_bridge_cost_parity(case, gateway)
