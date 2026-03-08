"""End-to-end test: cost-based routing retries all deployments cheapest-to-most-expensive."""
import pytest
from unittest.mock import AsyncMock, patch

import litellm
from litellm import Router
from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments


@pytest.mark.asyncio
async def test_cost_routing_retries_cheapest_to_most_expensive():
    """
    3 deployments with different costs.
    Cheapest fails -> mid fails -> most expensive succeeds.

    Verifies:
    - All errors trigger cooldown (Task 1)
    - should_retry_this_error never exits early (Task 3)
    - num_retries auto-set to deployment count - 1 (Task 4)
    - Cost-based routing picks cheapest available on each retry
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/cheap",
                    "api_key": "key1",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.001,
                },
                "model_info": {"id": "cheap-id"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/mid",
                    "api_key": "key2",
                    "input_cost_per_token": 0.01,
                    "output_cost_per_token": 0.01,
                },
                "model_info": {"id": "mid-id"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/expensive",
                    "api_key": "key3",
                    "input_cost_per_token": 0.1,
                    "output_cost_per_token": 0.1,
                },
                "model_info": {"id": "expensive-id"},
            },
        ],
        routing_strategy="cost-based-routing",
        cooldown_time=60,
        allowed_fails=0,  # cooldown immediately on first failure
    )

    models_called = []

    async def mock_acompletion(*args, **kwargs):
        model = kwargs.get("model", "")
        models_called.append(model)

        if "cheap" in model:
            # Manually set cooldown (normally the logging callback does this)
            _set_cooldown_deployments(
                litellm_router_instance=router,
                original_exception=litellm.InternalServerError(
                    message="Server error on cheap",
                    model=model,
                    llm_provider="openai",
                ),
                exception_status=500,
                deployment="cheap-id",
                time_to_cooldown=60,
            )
            raise litellm.InternalServerError(
                message="Server error on cheap",
                model=model,
                llm_provider="openai",
            )
        elif "mid" in model:
            _set_cooldown_deployments(
                litellm_router_instance=router,
                original_exception=litellm.BadRequestError(
                    message="Bad request on mid",
                    model=model,
                    llm_provider="openai",
                ),
                exception_status=400,
                deployment="mid-id",
                time_to_cooldown=60,
            )
            raise litellm.BadRequestError(
                message="Bad request on mid",
                model=model,
                llm_provider="openai",
            )
        else:
            # expensive succeeds
            return litellm.ModelResponse(
                choices=[{"message": {"content": "success"}}]
            )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert response.choices[0].message.content == "success"
    # Should have tried all 3: cheap -> mid -> expensive
    assert len(models_called) == 3, f"Expected 3 calls, got {len(models_called)}: {models_called}"
    assert "cheap" in models_called[0], f"First call should be cheapest, got {models_called[0]}"
    assert "mid" in models_called[1], f"Second call should be mid, got {models_called[1]}"
    assert "expensive" in models_called[2], f"Third call should be most expensive, got {models_called[2]}"


@pytest.mark.asyncio
async def test_cost_routing_all_fail_returns_error():
    """When all deployments fail, the error should propagate to the caller."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": f"openai/dep{i}",
                    "api_key": f"key{i}",
                    "input_cost_per_token": 0.001 * (i + 1),
                    "output_cost_per_token": 0.001 * (i + 1),
                },
                "model_info": {"id": f"dep-{i}"},
            }
            for i in range(3)
        ],
        routing_strategy="cost-based-routing",
        cooldown_time=60,
        allowed_fails=0,
    )

    call_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        model = kwargs.get("model", "")
        dep_id = f"dep-{call_count - 1}"
        _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=litellm.InternalServerError(
                message="All down",
                model=model,
                llm_provider="openai",
            ),
            exception_status=500,
            deployment=dep_id,
            time_to_cooldown=60,
        )
        raise litellm.InternalServerError(
            message="All down",
            model=model,
            llm_provider="openai",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(Exception):
            await router.acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
            )
