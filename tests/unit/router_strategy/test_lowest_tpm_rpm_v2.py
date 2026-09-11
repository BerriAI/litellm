import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2
from litellm.types.router import RouterErrors, RouterNoDeploymentsAvailableError
from litellm.utils import get_utc_datetime


def test_every_deployment_over_its_tpm_limit_raises_a_429():
    cache = DualCache()
    handler = LowestTPMLoggingHandler_v2(router_cache=cache)
    deployment = {
        "model_name": "gpt-4o-mini",
        "litellm_params": {"model": "openai/gpt-4o-mini", "tpm": 10},
        "model_info": {"id": "d1"},
    }
    minute = get_utc_datetime().strftime("%H-%M")
    cache.set_cache(key=f"d1:openai/gpt-4o-mini:tpm:{minute}", value=100)

    with pytest.raises(RouterNoDeploymentsAvailableError) as raised:
        handler.get_available_deployments(
            model_group="gpt-4o-mini",
            healthy_deployments=[deployment],
            messages=[{"role": "user", "content": "hi"}],
        )
    assert raised.value.status_code == 429
    assert RouterErrors.no_deployments_available.value in str(raised.value)
