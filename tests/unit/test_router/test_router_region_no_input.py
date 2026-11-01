import pytest
from litellm import Router


@pytest.mark.asyncio
async def test_allowed_model_region_filters_requests_without_messages_or_input() -> None:
    router = Router(
        model_list=[
            {
                "model_name": "region-test",
                "litellm_params": {"model": "openai/gpt-4o", "region_name": "eu"},
                "model_info": {"id": "eu-deployment"},
            },
            {
                "model_name": "region-test",
                "litellm_params": {"model": "openai/gpt-4o-mini", "region_name": "us"},
                "model_info": {"id": "us-deployment"},
            },
        ],
        enable_pre_call_checks=True,
    )

    deployments = await router.select(
        model="region-test",
        request_kwargs={"allowed_model_region": "eu"},
        kind="healthy",
        messages=None,
        input=None,
    )

    assert [deployment["model_info"]["id"] for deployment in deployments] == ["eu-deployment"]
