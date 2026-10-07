from __future__ import annotations

from typing import Final

import pytest

import litellm
from litellm import Router


@pytest.mark.asyncio
async def test_wildcard_openai_routing() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "test-key",
                },
            },
        ],
    )

    responses: Final = (
        await router.acompletion(
            model="gpt-4",
            messages=[{"content": "Tell me a joke.", "role": "user"}],
            mock_response="good morning",
        ),
        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"content": "Tell me a joke.", "role": "user"}],
            mock_response="good morning",
        ),
        await router.acompletion(
            model="gpt-4-turbo-preview",
            messages=[{"content": "Tell me a joke.", "role": "user"}],
            mock_response="good morning",
        ),
    )

    assert tuple(response["model"] for response in responses) == (
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4-turbo-preview",
    )


def test_get_available_deployment_for_pass_through() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "use_in_pass_through": True,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "test-key",
                    "use_in_pass_through": False,
                },
            },
        ],
    )

    selected_model: Final = router.get_available_deployment_for_pass_through("gpt-3.5-turbo")

    assert selected_model["litellm_params"]["model"] == "gpt-3.5-turbo"
    assert selected_model["litellm_params"]["use_in_pass_through"] is True


def test_get_available_deployment_for_pass_through_no_deployments() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "use_in_pass_through": False,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "test-key",
                    "use_in_pass_through": False,
                },
            },
        ],
    )

    with pytest.raises(litellm.BadRequestError, match="use_in_pass_through=True"):
        router.get_available_deployment_for_pass_through("gpt-3.5-turbo")


@pytest.mark.asyncio
async def test_async_get_available_deployment_for_pass_through() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "use_in_pass_through": True,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "test-key",
                    "use_in_pass_through": False,
                },
            },
        ],
    )

    selected_model: Final = await router.async_get_available_deployment_for_pass_through(
        model="gpt-3.5-turbo",
        request_kwargs={},
    )

    assert selected_model["litellm_params"]["model"] == "gpt-3.5-turbo"
    assert selected_model["litellm_params"]["use_in_pass_through"] is True


def test_filter_pass_through_deployments() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "use_in_pass_through": True,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "test-key",
                    "use_in_pass_through": False,
                },
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-35-turbo",
                    "api_key": "test-key",
                    "use_in_pass_through": True,
                },
            },
        ],
    )

    deployments: Final = router.get_model_list()
    pass_through_deployments: Final = router._filter_pass_through_deployments(deployments)

    assert tuple(deployment["litellm_params"]["model"] for deployment in pass_through_deployments) == (
        "gpt-3.5-turbo",
        "azure/gpt-35-turbo",
    )
