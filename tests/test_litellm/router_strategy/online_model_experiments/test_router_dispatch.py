import pytest

from litellm import Router
from litellm.types.router import LiteLLM_Params


def _experiment_config() -> dict[str, object]:
    return {
        "experiment_id": "support-v1",
        "secret": "test-secret",
        "identity_metadata_key": "user_id",
        "variants": [
            {"name": "control", "model": "model-a", "weight_basis_points": 5000},
            {"name": "candidate", "model": "model-b", "weight_basis_points": 5000},
        ],
    }


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "support-router",
                "litellm_params": {
                    "model": "auto_router/online_model_experiment",
                    "online_model_experiment_config": _experiment_config(),
                },
            },
            {"model_name": "model-a", "litellm_params": {"model": "openai/gpt-4o-mini"}},
            {"model_name": "model-b", "litellm_params": {"model": "openai/gpt-4o"}},
        ]
    )


def test_online_model_experiment_deployment_is_registered():
    router = _router()

    assert "support-router" in router.online_model_experiment_routers
    assert router._is_online_model_experiment_deployment(
        LiteLLM_Params(model="auto_router/online_model_experiment")
    )


@pytest.mark.asyncio
async def test_online_model_experiment_dispatches_stably_for_an_identity():
    router = _router()
    request_kwargs = {"metadata": {"user_id": "user-123"}}

    first_kwargs = {"metadata": {"user_id": "user-123"}}
    second_kwargs = {"metadata": {"user_id": "user-123"}}
    first = await router.async_pre_routing_hook("support-router", first_kwargs)
    second = await router.async_pre_routing_hook("support-router", second_kwargs)

    assert first is not None
    assert second is not None
    assert first.model in {"model-a", "model-b"}
    assert second.model == first.model
    assert first.litellm_params == second.litellm_params
    assert first_kwargs["metadata"]["online_model_experiment_id"] == "support-v1"
    assert first_kwargs["metadata"]["online_model_experiment_variant"] in {"control", "candidate"}
    assert first_kwargs["metadata"]["online_model_experiment_variant"] == second_kwargs["metadata"][
        "online_model_experiment_variant"
    ]
    assert "user-123" not in first_kwargs["metadata"]["online_model_experiment_assignment_key"]


@pytest.mark.asyncio
async def test_online_model_experiment_requires_stable_identity():
    router = _router()

    with pytest.raises(ValueError, match="stable identity"):
        await router.async_pre_routing_hook("support-router", {"metadata": {}})
