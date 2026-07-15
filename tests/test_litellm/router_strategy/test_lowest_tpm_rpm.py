"""
Boundary tests for the usage-based routing strategies (lowest_tpm_rpm v1 and v2).

Regression for an RPM off-by-one: the RPM filter used `rpm_dict[item] + 1 >= rpm`,
which excluded a deployment one request *before* its limit. A request that brings
usage to exactly the configured `rpm` is within budget (the TPM check in the same
function uses strict `>`, and the sibling lowest_cost / lowest_latency strategies
use `item_rpm + 1 > rpm`), so the boundary must be strict `>`.
"""

from datetime import datetime

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2


def _deployment(rpm: int, dep_id: str = "dep-1"):
    return {
        "model_name": "gpt",
        "litellm_params": {"model": "openai/gpt", "rpm": rpm, "tpm": 1_000_000},
        "model_info": {"id": dep_id},
    }


@pytest.mark.parametrize(
    "current_rpm, limit, expected_selectable",
    [
        (8, 10, True),   # well under limit
        (9, 10, True),   # boundary: next request reaches exactly the limit -> allowed
        (10, 10, False), # at the limit: next request would exceed it -> excluded
        (11, 10, False), # already over
    ],
)
def test_v2_return_potential_deployments_rpm_boundary(current_rpm, limit, expected_selectable):
    handler = LowestTPMLoggingHandler_v2(router_cache=DualCache(), routing_args={})
    dep = _deployment(rpm=limit)

    result = handler._return_potential_deployments(
        healthy_deployments=[dep],
        all_deployments={"dep-1": 0},  # tpm usage per deployment id
        input_tokens=0,
        rpm_dict={"dep-1": current_rpm},
    )

    assert bool(result) is expected_selectable


@pytest.mark.parametrize(
    "current_rpm, limit, expected_selectable",
    [
        (8, 10, True),
        (9, 10, True),   # boundary must be selectable
        (10, 10, False),
        (11, 10, False),
    ],
)
def test_v1_get_available_deployments_rpm_boundary(current_rpm, limit, expected_selectable):
    handler = LowestTPMLoggingHandler(router_cache=DualCache(), routing_args={})
    model_group = "gpt"
    deployments = [_deployment(rpm=limit)]

    # Seed the tpm/rpm usage caches for the current minute, matching how
    # get_available_deployments reads them, then call immediately.
    current_minute = datetime.now().strftime("%H-%M")
    handler.router_cache.set_cache(key=f"{model_group}:tpm:{current_minute}", value={"dep-1": 0})
    handler.router_cache.set_cache(key=f"{model_group}:rpm:{current_minute}", value={"dep-1": current_rpm})

    result = handler.get_available_deployments(
        model_group=model_group,
        healthy_deployments=deployments,
    )

    assert (result is not None) is expected_selectable
