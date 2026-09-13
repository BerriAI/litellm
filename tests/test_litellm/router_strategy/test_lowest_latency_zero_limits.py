#### What this tests ####
#    lowest-latency routing must treat an explicitly configured tpm=0 / rpm=0
#    as "this deployment is blocked", not as "unlimited". The limits were read
#    through an `or`-chain, and `or` treats 0 as falsy, so a configured 0 fell
#    through to float("inf") -- the exact opposite of the configured intent.
#    Issue #39744.

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler

MODEL_GROUP = "gpt-4o"
DEPLOYMENT_ID = "disabled-deployment"


def _deployment(*, level: str, limit_key: str, value: int) -> dict:
    """Build a deployment carrying `limit_key` at one of the three lookup levels."""
    deployment = {
        "model_name": MODEL_GROUP,
        "litellm_params": {"model": "azure/gpt-4o"},
        "model_info": {"id": DEPLOYMENT_ID},
    }
    if level == "top_level":
        deployment[limit_key] = value
    elif level == "litellm_params":
        deployment["litellm_params"][limit_key] = value
    elif level == "model_info":
        deployment["model_info"][limit_key] = value
    else:  # pragma: no cover - guards against a typo in the parametrize list
        raise ValueError(f"unknown level {level!r}")
    return deployment


def _select(deployment: dict):
    handler = LowestLatencyLoggingHandler(router_cache=DualCache(), routing_args={})
    return handler._get_available_deployments(
        model_group=MODEL_GROUP,
        healthy_deployments=[deployment],
        messages=[{"role": "user", "content": "hello there, this is a prompt"}],
        request_count_dict={},
    )


@pytest.mark.parametrize("level", ["top_level", "litellm_params", "model_info"])
@pytest.mark.parametrize("limit_key", ["tpm", "rpm"])
def test_zero_limit_blocks_deployment(level: str, limit_key: str):
    """A deployment configured with tpm=0 or rpm=0 must never be selected."""
    selected = _select(_deployment(level=level, limit_key=limit_key, value=0))

    assert selected is None, (
        f"deployment with {limit_key}=0 at {level} was selected; 0 was treated as unlimited instead of blocked"
    )


@pytest.mark.parametrize("level", ["top_level", "litellm_params", "model_info"])
@pytest.mark.parametrize("limit_key", ["tpm", "rpm"])
def test_nonzero_limit_still_allows_deployment(level: str, limit_key: str):
    """Control: a generous limit at the same level must still be selectable."""
    selected = _select(_deployment(level=level, limit_key=limit_key, value=1_000_000))

    assert selected is not None, f"deployment with {limit_key}=1000000 at {level} was wrongly excluded"
    assert selected["model_info"]["id"] == DEPLOYMENT_ID


def test_unset_limits_are_unlimited():
    """Control: with no tpm/rpm configured at all, the deployment stays selectable."""
    selected = _select(
        {
            "model_name": MODEL_GROUP,
            "litellm_params": {"model": "azure/gpt-4o"},
            "model_info": {"id": DEPLOYMENT_ID},
        }
    )

    assert selected is not None
    assert selected["model_info"]["id"] == DEPLOYMENT_ID
