from collections import Counter
from inspect import isawaitable

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


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", [
    "get_available_deployment", "async_get_available_deployment",
    "get_available_deployment_for_pass_through", "async_get_available_deployment_for_pass_through",
])
async def test_scoped_weights_are_request_local_and_respect_eligibility(selector: str) -> None:
    router = Router(model_list=[
        {
            **_deployment(deployment_id, {
                "weight": 100 if deployment_id == "global" else 0, "use_in_pass_through": True,
            }),
            "model_name": f"model_name_{team_id}_{deployment_id}",
            "model_info": {
                "id": deployment_id, "team_id": team_id, "team_public_model_name": "test-model", "blocked": blocked,
            },
        }
        for deployment_id, team_id, blocked in (
            ("global", "team-a", False), ("scoped", "team-a", False),
            ("blocked", "team-a", True), ("foreign", "other-team", False),
        )
    ], num_retries=0)

    for weights, expected in (
        ({"test-model": {"global": 0, "scoped": 100, "blocked": 100, "foreign": 100}}, "scoped"),
        ({"test-model": {"global": 100, "scoped": 0}}, "global"),
        ({"test-model": {"foreign": 100}}, "global"),
        ({"test-model": {"blocked": 100}}, "global"),
        (None, "global"),
    ):
        result = getattr(router, selector)(
            model="test-model",
            request_kwargs={"metadata": {"user_api_key_team_id": "team-a"}, "_router_weights": weights},
        )
        deployment = await result if isawaitable(result) else result
        assert deployment["model_info"]["id"] == expected


def test_scoped_weights_approximate_the_configured_split() -> None:
    router = Router(model_list=[_deployment("primary"), _deployment("secondary")], num_retries=0)
    counts = Counter(
        router.get_available_deployment(
            model="test-model",
            request_kwargs={"_router_weights": {"test-model": {"primary": 80, "secondary": 20}}},
        )["model_info"]["id"]
        for _ in range(1000)
    )
    assert 700 < counts["primary"] < 900
