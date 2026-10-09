from typing import Final

import pytest
from integration._support.client import Gateway
from integration.cost_calculation.case import CostTrackingTestCase
from integration.cost_calculation.chat_completions.bases.anthropic import CLAUDE_SONNET_5_TEST_CASE
from integration.cost_calculation.runner import assert_cost_tracking

CASES: Final[tuple[CostTrackingTestCase, ...]] = (CLAUDE_SONNET_5_TEST_CASE,)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_chat_completions_basic_anthropic(case: CostTrackingTestCase, gateway: Gateway) -> None:
    assert_cost_tracking(case, gateway)
