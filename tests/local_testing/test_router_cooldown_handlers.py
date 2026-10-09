#### What this tests ####
#    This tests calling router with fallback models

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import Router
from litellm.router_utils.cooldown_handlers import (
    async_get_cooldown_deployments,
)
from litellm.types.router import (
    AllowedFailsPolicy,
)


@pytest.mark.asyncio
async def test_cooldown_badrequest_error():
    """
    Test 1. It SHOULD NOT cooldown a deployment on a BadRequestError
    """

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_AI_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
                },
            }
        ],
        debug_level="DEBUG",
        set_verbose=True,
        cooldown_time=300,
        num_retries=0,
        allowed_fails=0,
    )

    # Act & Assert
    try:

        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "gm"}],
            bad_param=200,
        )
    except Exception:
        pass

    await asyncio.sleep(3)  # wait for deployment to get cooled-down

    response = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "gm"}],
        mock_response="hello",
    )

    assert response is not None

    print(response)

@pytest.mark.asyncio
async def test_single_deployment_cooldown_with_allowed_fails():
    """
    When `allowed_fails` is set, use the allowed_fails to determine cooldown for 1 deployment
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                },
            },
            {
                "model_name": "gpt-12",
                "litellm_params": {
                    "model": "openai/gpt-12",
                },
            },
        ],
        allowed_fails=1,
        num_retries=0,
    )

    with patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()
    ) as mock_client:
        for _ in range(2):
            try:
                await router.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hey, how's it going?"}],
                    timeout=0.0001,
                )
            except litellm.Timeout:
                pass

        # Poll until the mock is called (or timeout)
        for _ in range(40):
            if mock_client.call_count >= 1:
                break
            await asyncio.sleep(0.1)

        mock_client.assert_called_once()

@pytest.mark.asyncio
async def test_single_deployment_cooldown_with_allowed_fail_policy():
    """
    When `allowed_fails_policy` is set, use the allowed_fails_policy to determine cooldown for 1 deployment
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                },
            },
            {
                "model_name": "gpt-12",
                "litellm_params": {
                    "model": "openai/gpt-12",
                },
            },
        ],
        allowed_fails_policy=AllowedFailsPolicy(
            TimeoutErrorAllowedFails=1,
        ),
        num_retries=0,
    )

    with patch.object(
        router.cooldown_cache, "add_deployment_to_cooldown", new=MagicMock()
    ) as mock_client:
        for _ in range(2):
            try:
                await router.acompletion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hey, how's it going?"}],
                    timeout=0.0001,
                )
            except litellm.Timeout:
                pass

        # Poll until the mock is called (or timeout)
        for _ in range(40):
            if mock_client.call_count >= 1:
                break
            await asyncio.sleep(0.1)

        mock_client.assert_called_once()

@pytest.mark.asyncio
async def test_single_deployment_no_cooldowns_test_prod_mock_completion_calls():
    """
    Do not cooldown on single deployment.

    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                },
            },
            {
                "model_name": "gpt-12",
                "litellm_params": {
                    "model": "openai/gpt-12",
                },
            },
        ],
    )

    for _ in range(20):
        try:
            await router.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                mock_response="litellm.RateLimitError",
            )
        except litellm.RateLimitError:
            pass

    cooldown_list = await async_get_cooldown_deployments(
        litellm_router_instance=router, parent_otel_span=None
    )
    assert len(cooldown_list) == 0

    healthy_deployments, _ = await router._async_get_healthy_deployments(
        model="gpt-3.5-turbo", parent_otel_span=None
    )

    print("healthy_deployments: ", healthy_deployments)
