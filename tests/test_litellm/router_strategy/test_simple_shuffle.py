from collections import Counter

import pytest

from litellm import Router
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict

DRAWS = 200


def _deployment(dep_id: str, metric: LiteLLMParamsTypedDict | None = None) -> DeploymentTypedDict:
    params: LiteLLMParamsTypedDict = {"model": "gpt-4o", "api_key": "key", "mock_response": f"from {dep_id}"}
    return {
        "model_name": "test-model",
        "litellm_params": {**params, **(metric or {})},
        "model_info": {"id": dep_id},
    }


async def _draw_model_ids(router: Router) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _ in range(DRAWS):
        response = await router.acompletion(model="test-model", messages=[{"role": "user", "content": "hi"}])
        counts[response._hidden_params["model_id"]] += 1
    return counts


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", [{"weight": 5}, {"rpm": 5}, {"tpm": 5}], ids=["weight", "rpm", "tpm"])
async def test_weighted_pick_when_only_a_later_deployment_carries_the_metric(metric: LiteLLMParamsTypedDict):
    router = Router(
        model_list=[_deployment("unweighted"), _deployment("weighted", metric)],
        routing_strategy="simple-shuffle",
        num_retries=0,
    )

    counts = await _draw_model_ids(router)

    assert counts["weighted"] == DRAWS
    assert counts["unweighted"] == 0


@pytest.mark.asyncio
async def test_uniform_pick_when_every_configured_weight_is_zero():
    router = Router(
        model_list=[_deployment("unweighted"), _deployment("standby", {"weight": 0})],
        routing_strategy="simple-shuffle",
        num_retries=0,
    )

    counts = await _draw_model_ids(router)

    assert counts["unweighted"] > 0
    assert counts["standby"] > 0
