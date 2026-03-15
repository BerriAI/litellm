"""
Unit tests for dynamic deployment budget registration in RouterBudgetLimiting.

Tests the following changes:
1. RouterBudgetLimiting.register_deployment_budget() — dynamic registration
2. RouterBudgetLimiting.remove_deployment_budget() — dynamic removal
3. Router.add_deployment() — auto-registers deployment budget
4. Router.delete_deployment() — auto-removes deployment budget
5. Router.upsert_deployment() — removes old budget, registers new budget
6. Router._lazy_init_router_budget_limiter() — lazy init when no initial budget config
"""

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from litellm import Router
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.types.utils import BudgetConfig, GenericBudgetConfigType
from litellm.caching.caching import DualCache


# ──────────────────────────────────────────────────────────────────────────────
# RouterBudgetLimiting unit tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_deployment_budget():
    """
    Test register_deployment_budget adds a new entry to deployment_budget_config.
    """
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    assert budget_limiter.deployment_budget_config is None

    budget_limiter.register_deployment_budget(
        model_id="model-123",
        max_budget=50.0,
        budget_duration="1d",
    )

    assert budget_limiter.deployment_budget_config is not None
    assert "model-123" in budget_limiter.deployment_budget_config

    config = budget_limiter.deployment_budget_config["model-123"]
    assert config.max_budget == 50.0
    assert config.budget_duration == "1d"


@pytest.mark.asyncio
async def test_register_deployment_budget_overwrites_existing():
    """
    Test register_deployment_budget overwrites an existing entry for the same model_id.
    """
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    budget_limiter.register_deployment_budget(
        model_id="model-123",
        max_budget=50.0,
        budget_duration="1d",
    )

    budget_limiter.register_deployment_budget(
        model_id="model-123",
        max_budget=100.0,
        budget_duration="7d",
    )

    config = budget_limiter.deployment_budget_config["model-123"]
    assert config.max_budget == 100.0
    assert config.budget_duration == "7d"


@pytest.mark.asyncio
async def test_register_multiple_deployment_budgets():
    """
    Test registering budgets for multiple deployments.
    """
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    budget_limiter.register_deployment_budget("model-a", 50.0, "1d")
    budget_limiter.register_deployment_budget("model-b", 100.0, "7d")
    budget_limiter.register_deployment_budget("model-c", 200.0, "30d")

    assert len(budget_limiter.deployment_budget_config) == 3
    assert budget_limiter.deployment_budget_config["model-a"].max_budget == 50.0
    assert budget_limiter.deployment_budget_config["model-b"].max_budget == 100.0
    assert budget_limiter.deployment_budget_config["model-c"].max_budget == 200.0


@pytest.mark.asyncio
async def test_remove_deployment_budget():
    """
    Test remove_deployment_budget removes the entry from deployment_budget_config.
    """
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    budget_limiter.register_deployment_budget("model-123", 50.0, "1d")
    budget_limiter.register_deployment_budget("model-456", 100.0, "7d")

    budget_limiter.remove_deployment_budget("model-123")

    assert "model-123" not in budget_limiter.deployment_budget_config
    assert "model-456" in budget_limiter.deployment_budget_config


@pytest.mark.asyncio
async def test_remove_deployment_budget_nonexistent():
    """
    Test remove_deployment_budget does not raise when removing a non-existent model_id.
    """
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    # Should not raise when deployment_budget_config is None
    budget_limiter.remove_deployment_budget("nonexistent")

    # Should not raise when model_id is not in deployment_budget_config
    budget_limiter.register_deployment_budget("model-123", 50.0, "1d")
    budget_limiter.remove_deployment_budget("nonexistent")

    assert "model-123" in budget_limiter.deployment_budget_config


@pytest.mark.asyncio
async def test_get_budget_config_for_dynamically_registered_deployment():
    """
    Test that _get_budget_config_for_deployment works for dynamically registered deployments.
    """
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    assert budget_limiter._get_budget_config_for_deployment("model-123") is None

    budget_limiter.register_deployment_budget("model-123", 50.0, "1d")

    config = budget_limiter._get_budget_config_for_deployment("model-123")
    assert config is not None
    assert config.max_budget == 50.0
    assert config.budget_duration == "1d"


# ──────────────────────────────────────────────────────────────────────────────
# Router integration tests — add_deployment
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_deployment_registers_budget():
    """
    Test that Router.add_deployment() registers the deployment budget in RouterBudgetLimiting
    when the deployment has max_budget and budget_duration.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                    "max_budget": 10,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "seed-model"},
            }
        ],
    )

    # RouterBudgetLimiting should have been initialized with the seed model
    assert router.router_budget_logger is not None
    assert "seed-model" in router.router_budget_logger.deployment_budget_config

    # Now add a new deployment dynamically
    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key-2",
                max_budget=80,
                budget_duration="7d",
            ),
            model_info=ModelInfo(id="dynamic-model-1"),
        )
    )

    assert "dynamic-model-1" in router.router_budget_logger.deployment_budget_config
    config = router.router_budget_logger.deployment_budget_config["dynamic-model-1"]
    assert config.max_budget == 80.0
    assert config.budget_duration == "7d"


@pytest.mark.asyncio
async def test_add_deployment_without_budget_does_not_register():
    """
    Test that Router.add_deployment() does NOT register a deployment budget
    when the deployment does not have max_budget/budget_duration.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                    "max_budget": 10,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "seed-model"},
            }
        ],
    )

    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key-2",
            ),
            model_info=ModelInfo(id="no-budget-model"),
        )
    )

    assert "no-budget-model" not in (
        router.router_budget_logger.deployment_budget_config or {}
    )


# ──────────────────────────────────────────────────────────────────────────────
# Router integration tests — delete_deployment
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_deployment_removes_budget():
    """
    Test that Router.delete_deployment() removes the deployment budget from RouterBudgetLimiting.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                    "max_budget": 10,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "model-to-delete"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key-2",
                    "max_budget": 20,
                    "budget_duration": "7d",
                },
                "model_info": {"id": "model-to-keep"},
            },
        ],
    )

    assert "model-to-delete" in router.router_budget_logger.deployment_budget_config
    assert "model-to-keep" in router.router_budget_logger.deployment_budget_config

    router.delete_deployment(id="model-to-delete")

    assert "model-to-delete" not in router.router_budget_logger.deployment_budget_config
    assert "model-to-keep" in router.router_budget_logger.deployment_budget_config


# ──────────────────────────────────────────────────────────────────────────────
# Router integration tests — upsert_deployment
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_deployment_updates_budget():
    """
    Test that Router.upsert_deployment() updates the deployment budget
    when replacing an existing deployment with different litellm_params.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                    "max_budget": 10,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "upsert-model"},
            },
        ],
    )

    old_config = router.router_budget_logger.deployment_budget_config["upsert-model"]
    assert old_config.max_budget == 10.0
    assert old_config.budget_duration == "1d"

    # Upsert with new budget params
    router.upsert_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key-updated",
                max_budget=200,
                budget_duration="30d",
            ),
            model_info=ModelInfo(id="upsert-model"),
        )
    )

    new_config = router.router_budget_logger.deployment_budget_config["upsert-model"]
    assert new_config.max_budget == 200.0
    assert new_config.budget_duration == "30d"


@pytest.mark.asyncio
async def test_upsert_deployment_no_change_keeps_budget():
    """
    Test that Router.upsert_deployment() does not touch the budget
    when litellm_params are identical (no update needed).
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                    "max_budget": 10,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "unchanged-model"},
            },
        ],
    )

    # Upsert with identical params — should return None (no update)
    result = router.upsert_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key",
                max_budget=10,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="unchanged-model"),
        )
    )

    assert result is None
    config = router.router_budget_logger.deployment_budget_config["unchanged-model"]
    assert config.max_budget == 10.0
    assert config.budget_duration == "1d"


# ──────────────────────────────────────────────────────────────────────────────
# Router integration tests — lazy init
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lazy_init_no_yaml_budget_config():
    """
    Test that RouterBudgetLimiting is lazily initialized when:
    - Router is created with NO provider_budget_config
    - No YAML model has max_budget
    - A dynamically added deployment has max_budget + budget_duration

    This is the core bug fix scenario: DB models with budgets should work
    even without any YAML budget configuration.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "yaml-model-no-budget"},
            }
        ],
    )

    # RouterBudgetLimiting should NOT be initialized at this point
    assert router.router_budget_logger is None

    # Add a deployment with budget config — simulates DB model loading
    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key-2",
                max_budget=100,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="db-model-with-budget"),
        )
    )

    # RouterBudgetLimiting should now be lazily initialized
    assert router.router_budget_logger is not None
    assert "db-model-with-budget" in router.router_budget_logger.deployment_budget_config

    config = router.router_budget_logger.deployment_budget_config[
        "db-model-with-budget"
    ]
    assert config.max_budget == 100.0
    assert config.budget_duration == "1d"


@pytest.mark.asyncio
async def test_lazy_init_registers_in_callbacks():
    """
    Test that lazy-initialized RouterBudgetLimiting is properly registered
    in the Router's optional_callbacks and litellm callback manager,
    so that async_filter_deployments and async_log_success_event are called.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "yaml-model"},
            }
        ],
    )

    assert router.router_budget_logger is None
    assert router.optional_callbacks is None or not any(
        isinstance(cb, RouterBudgetLimiting)
        for cb in (router.optional_callbacks or [])
    )

    # Trigger lazy init
    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key-2",
                max_budget=50,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="trigger-lazy-init"),
        )
    )

    # Verify RouterBudgetLimiting is in optional_callbacks
    assert router.optional_callbacks is not None
    assert any(
        isinstance(cb, RouterBudgetLimiting) for cb in router.optional_callbacks
    )

    # Verify the instance in optional_callbacks is the same as router_budget_logger
    budget_logger_in_callbacks = next(
        cb for cb in router.optional_callbacks if isinstance(cb, RouterBudgetLimiting)
    )
    assert budget_logger_in_callbacks is router.router_budget_logger


@pytest.mark.asyncio
async def test_lazy_init_only_once():
    """
    Test that lazy init only creates one RouterBudgetLimiting instance
    even when multiple deployments with budgets are added.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "yaml-model"},
            }
        ],
    )

    assert router.router_budget_logger is None

    # Add first deployment with budget — triggers lazy init
    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="key-1",
                max_budget=50,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="model-a"),
        )
    )

    first_instance = router.router_budget_logger
    assert first_instance is not None

    # Add second deployment with budget — should reuse existing instance
    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="key-2",
                max_budget=100,
                budget_duration="7d",
            ),
            model_info=ModelInfo(id="model-b"),
        )
    )

    assert router.router_budget_logger is first_instance
    assert "model-a" in first_instance.deployment_budget_config
    assert "model-b" in first_instance.deployment_budget_config

    # Verify only one RouterBudgetLimiting in optional_callbacks
    budget_loggers = [
        cb
        for cb in (router.optional_callbacks or [])
        if isinstance(cb, RouterBudgetLimiting)
    ]
    assert len(budget_loggers) == 1


@pytest.mark.asyncio
async def test_lazy_init_without_any_yaml_models():
    """
    Test lazy init when Router starts with an empty model_list
    and all models come from DB.
    """
    router = Router(model_list=[], ignore_invalid_deployments=True)

    assert router.router_budget_logger is None

    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key",
                max_budget=100,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="pure-db-model"),
        )
    )

    assert router.router_budget_logger is not None
    assert "pure-db-model" in router.router_budget_logger.deployment_budget_config


@pytest.mark.asyncio
async def test_no_lazy_init_for_deployment_without_budget():
    """
    Test that adding a deployment WITHOUT max_budget/budget_duration
    does NOT trigger lazy init of RouterBudgetLimiting.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "yaml-model"},
            }
        ],
    )

    assert router.router_budget_logger is None

    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key-2",
            ),
            model_info=ModelInfo(id="no-budget-model"),
        )
    )

    # RouterBudgetLimiting should still be None
    assert router.router_budget_logger is None


# ──────────────────────────────────────────────────────────────────────────────
# Router integration tests — filter_deployments with dynamic budgets
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_filter_deployments_includes_dynamically_registered():
    """
    Test that async_filter_deployments correctly filters dynamically registered
    deployment budgets (both within budget and over budget).
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key",
                    "max_budget": 100,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "existing-model"},
            }
        ],
    )

    # Dynamically add a deployment with budget
    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4",
                api_key="fake-key-2",
                max_budget=50,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="dynamic-model"),
        )
    )

    budget_logger = router.router_budget_logger
    assert budget_logger is not None

    # Simulate the dynamic-model having spent over its budget
    spend_key = "deployment_spend:dynamic-model:1d"
    await budget_logger.dual_cache.async_set_cache(key=spend_key, value=60.0)

    # Build healthy_deployments list as the router would see them
    healthy_deployments = router.model_list

    # Run filter
    filtered = await budget_logger.async_filter_deployments(
        model="gpt-4",
        healthy_deployments=healthy_deployments,
        messages=[{"role": "user", "content": "test"}],
    )

    # dynamic-model should be filtered out (spent 60 > budget 50)
    # existing-model should remain (no spend recorded, within 100 budget)
    filtered_ids = [d.get("model_info", {}).get("id") for d in filtered]
    assert "existing-model" in filtered_ids
    assert "dynamic-model" not in filtered_ids
