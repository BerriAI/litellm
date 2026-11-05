import pytest
from litellm import Router


def _router_with_region_deployments(*, pass_through: bool = False) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "region-test",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "region_name": "eu",
                    "use_in_pass_through": pass_through,
                },
                "model_info": {"id": "eu-deployment"},
            },
            {
                "model_name": "region-test",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "region_name": "us",
                    "use_in_pass_through": pass_through,
                },
                "model_info": {"id": "us-deployment"},
            },
        ],
        enable_pre_call_checks=True,
    )


@pytest.mark.asyncio
async def test_allowed_model_region_filters_async_selection_without_messages_or_input() -> None:
    router = _router_with_region_deployments()

    deployments = await router.select(
        model="region-test",
        request_kwargs={"allowed_model_region": "eu"},
        kind="healthy",
        messages=None,
        input=None,
    )

    assert [deployment["model_info"]["id"] for deployment in deployments] == ["eu-deployment"]


def test_allowed_model_region_filters_sync_selection_without_messages_or_input() -> None:
    router = _router_with_region_deployments()

    deployment = router.get_available_deployment(
        model="region-test",
        request_kwargs={"allowed_model_region": "eu"},
        messages=None,
        input=None,
    )

    assert deployment["model_info"]["id"] == "eu-deployment"


def test_allowed_model_region_filters_passthrough_without_messages_or_input() -> None:
    router = _router_with_region_deployments(pass_through=True)

    deployment = router.get_available_deployment_for_pass_through(
        model="region-test",
        request_kwargs={"allowed_model_region": "eu"},
        messages=None,
        input=None,
    )

    assert deployment["model_info"]["id"] == "eu-deployment"
