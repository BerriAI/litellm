from typing import Final

import pytest

from litellm.litellm_core_utils.gateway_processing import GatewayProcessingTiming


def test_gateway_processing_subtracts_overlapping_intervals_once() -> None:
    timing: Final = GatewayProcessingTiming(1.0)
    timing.exclude(0.0, 3.0)
    timing.exclude(2.0, 4.0)
    timing.exclude(6.0, 8.0)
    timing.exclude(9.0, 14.0)
    timing.complete_attempt(timed=True)
    timing.complete_attempt(timed=True)

    assert timing.seconds(11.0) == pytest.approx(3.0)


@pytest.mark.parametrize("attempts", [(), (True, False), (False, True)])
def test_gateway_processing_requires_complete_provider_timing(attempts: tuple[bool, ...]) -> None:
    timing: Final = GatewayProcessingTiming(1.0)
    timing.exclude(2.0, 3.0)
    for timed in attempts:
        timing.complete_attempt(timed)

    assert timing.seconds(4.0) is None
