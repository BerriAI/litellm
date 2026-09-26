from datetime import datetime, timedelta
from typing import Final

from litellm import Router
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict

MODEL_GROUP: Final = "lowest-tpm-router"
HIGH_USAGE_DEPLOYMENT_ID: Final = "highest-usage"
LOW_USAGE_DEPLOYMENT_ID: Final = "lowest-usage"


def _deployment(deployment_id: str) -> DeploymentTypedDict:
    params: LiteLLMParamsTypedDict = {
        "model": "gpt-4o",
        "api_key": "key",
        "mock_response": f"from {deployment_id}",
    }
    return {
        "model_name": MODEL_GROUP,
        "litellm_params": params,
        "model_info": {"id": deployment_id},
    }


def test_usage_based_routing_v1_selects_the_lowest_recorded_tpm() -> None:
    router: Final = Router(
        model_list=[
            _deployment(HIGH_USAGE_DEPLOYMENT_ID),
            _deployment(LOW_USAGE_DEPLOYMENT_ID),
        ],
        routing_strategy="usage-based-routing",
        num_retries=0,
    )
    usage_by_deployment: Final = {
        HIGH_USAGE_DEPLOYMENT_ID: 100,
        LOW_USAGE_DEPLOYMENT_ID: 1,
    }
    now: Final = datetime.now()
    cache_keys: Final = tuple(
        f"{MODEL_GROUP}:tpm:{(now + timedelta(minutes=offset)).strftime('%H-%M')}"
        for offset in range(60)
    )

    for cache_key in cache_keys:
        router.cache.set_cache(
            key=cache_key, value=usage_by_deployment, ttl=float("inf")
        )

    deployment: Final = router.get_available_deployment(
        model=MODEL_GROUP,
        messages=[{"role": "user", "content": "test"}],
    )

    assert deployment["model_info"]["id"] == LOW_USAGE_DEPLOYMENT_ID
